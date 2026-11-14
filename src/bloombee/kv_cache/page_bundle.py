"""Lossless LMCache storage layout for BloomBee's ordinary paged KV service.

Wire blocks and their SHA-256 values retain their original per-model-layer
identities. LMCache sees 2*L storage slots for L actual transformer layers.
This codec does no discovery, network I/O, admission, or generation.
"""
import hashlib
import math

import torch

from bloombee.kv_cache.identity import CacheIdentity, canonical_digest, token_ids_digest
from bloombee.kv_cache.transport import DTYPES, MAX_HEADER, pack_snapshot, unpack_snapshot


FORMAT = "bloombee_paged_kv_and_aux_lmcache_v1"
_PER_BLOCK_FIELDS = {"layer_start", "layer_end", "prefix_tokens_digest", "prefix_length", "position_digest"}


class PageBundleLayout:
    def __init__(self, identities_by_block, tokens, *, block_size=16, max_bytes=256 * 1024**2):
        if type(block_size) is not int or block_size <= 0 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid storage limits")
        self._tokens = tuple(tokens)
        if not self._tokens or len(self._tokens) % block_size:
            raise ValueError("page bundle requires a nonempty prefix of complete blocks")
        self.block_size = block_size
        self.blocks = len(self._tokens) // block_size
        rows = tuple(tuple(i if isinstance(i, CacheIdentity) else CacheIdentity.from_dict(i)
                           for i in row) for row in identities_by_block)
        if len(rows) != self.blocks or not rows[0]:
            raise ValueError("incomplete page bundle identities")
        self.logical_layers = len(rows[0])
        self.storage_layers = self.logical_layers * 2
        base = {k: v for k, v in rows[0][0].to_dict().items() if k not in _PER_BLOCK_FIELDS}
        for block_index, row in enumerate(rows):
            end = (block_index + 1) * block_size
            prefix_digest = token_ids_digest(self._tokens[:end])
            position_digest = canonical_digest({"start": 0, "end": end})
            if len(row) != self.logical_layers:
                raise ValueError("missing model layer")
            for layer, identity in enumerate(row):
                if (identity.layer_start, identity.layer_end) != (layer, layer + 1):
                    raise ValueError("model layers must be complete and ordered")
                if (identity.prefix_length != end or identity.prefix_tokens_digest != prefix_digest
                        or identity.position_digest != position_digest):
                    raise ValueError("wrong tokens, prefix length, or absolute position")
                if any(getattr(identity, k) != v for k, v in base.items()):
                    raise ValueError("incompatible model, execution, or layout identities")
        self.identities_by_block = rows
        self.heads = base["num_attention_heads"]
        self.dim = base["head_dim"]
        self.width = self.heads * self.dim
        self.dtype = DTYPES.get(base["dtype"])
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32) or base["batch_size"] != 1 or base["layout"] != "bhsd_v1":
            raise ValueError("unsupported ordinary page dtype, batch, or layout")
        self.shape = (2, self.storage_layers, len(self._tokens), self.width)
        self.tensor_bytes = math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()
        if self.tensor_bytes > max_bytes:
            raise ValueError("page bundle exceeds the storage budget")
        self.wire_limit = self.tensor_bytes // (self.logical_layers * self.blocks) + MAX_HEADER + 12
        descriptor = {"format": FORMAT, "base": base, "logical_layers": self.logical_layers,
                      "block_size": block_size, "storage_layers": self.storage_layers}
        self.model_name = "bb_page_" + canonical_digest(descriptor)
        self.cache_salt = canonical_digest({
            "model_name": self.model_name, "blocks": [[i.digest() for i in row] for row in rows]})

    @property
    def tokens(self):
        return list(self._tokens)

    def _check_grid(self, grid):
        if len(grid) != self.blocks or any(len(row) != self.logical_layers for row in grid):
            raise ValueError("expected one item per ordered block and model layer")

    @staticmethod
    def _verify(blob, sha):
        if not isinstance(blob, bytes) or hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError("page wire SHA-256 differs from publisher")

    def _wire(self, identity):
        # Keep the deployed format definition in its original owner. The local
        # import also permits the server to import this codec during startup.
        from bloombee.server.dynamo_remote import wire_identity
        return wire_identity(identity, self.block_size)

    def encode(self, blobs_by_block, sha_by_block):
        """Convert the original per-block wire bytes to the official SDK tensor."""
        self._check_grid(blobs_by_block)
        self._check_grid(sha_by_block)
        native = torch.empty(self.shape, dtype=self.dtype, device="cpu")
        for block, identities in enumerate(self.identities_by_block):
            span = slice(block * self.block_size, (block + 1) * self.block_size)
            for layer, identity in enumerate(identities):
                blob = blobs_by_block[block][layer]
                self._verify(blob, sha_by_block[block][layer])
                tensors = unpack_snapshot(blob, self._wire(identity), max_bytes=self.wire_limit)
                if tensors.keys() != {"key", "value", "aux_input", "aux_output"}:
                    raise ValueError("ordinary page requires K/V and both auxiliary tensors")
                for plane, (kv_name, aux_name) in enumerate((("key", "aux_input"), ("value", "aux_output"))):
                    kv, aux = tensors[kv_name], tensors[aux_name]
                    if (kv.shape != (1, self.heads, self.block_size, self.dim)
                            or aux.shape != (1, 1, self.block_size, self.width)
                            or kv.dtype != self.dtype or aux.dtype != self.dtype):
                        raise ValueError("incompatible four-tensor layout")
                    native[plane, layer, span].copy_(kv[0].transpose(0, 1).reshape(self.block_size, self.width))
                    native[plane, self.logical_layers + layer, span].copy_(aux[0, 0])
        return native

    def decode(self, native_tensor, sha_by_block):
        """Return exact original wire bytes. Reject the entire bundle if any SHA differs."""
        self._check_grid(sha_by_block)
        if (not isinstance(native_tensor, torch.Tensor) or native_tensor.layout != torch.strided
                or tuple(native_tensor.shape) != self.shape or native_tensor.dtype != self.dtype):
            raise ValueError("LMCache storage geometry or dtype mismatch")
        native = native_tensor.detach().to("cpu")
        blobs = []
        for block, identities in enumerate(self.identities_by_block):
            span = slice(block * self.block_size, (block + 1) * self.block_size)
            layer_blobs = []
            for layer, identity in enumerate(identities):
                tensors = {}
                for plane, (kv_name, aux_name) in enumerate((("key", "aux_input"), ("value", "aux_output"))):
                    kv = native[plane, layer, span].reshape(self.block_size, self.heads, self.dim)
                    tensors[kv_name] = kv.transpose(0, 1).unsqueeze(0)
                    tensors[aux_name] = native[plane, self.logical_layers + layer, span].reshape(
                        1, 1, self.block_size, self.width)
                blob = pack_snapshot(self._wire(identity), tensors, max_bytes=self.wire_limit)
                self._verify(blob, sha_by_block[block][layer])
                layer_blobs.append(blob)
            blobs.append(layer_blobs)
        return blobs
