"""Versioned exact cache identities; never include prompts or tensors in DHT."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import re
from typing import Mapping, Sequence


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def token_ids_digest(tokens: Sequence[int]) -> str:
    tokens = list(tokens)
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('token IDs must be a nonempty sequence of nonnegative integers')
    return canonical_digest(tokens)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class CacheIdentity:
    model_id: str
    model_revision: str
    config_digest: str
    tokenizer_digest: str
    adapter_digest: str
    layer_start: int
    layer_end: int
    prefix_tokens_digest: str
    prefix_length: int
    position_digest: str
    attention_digest: str
    rope_digest: str
    layout: str
    dtype: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    execution_digest: str
    schema_version: int = 1
    position_start: int = 0
    batch_size: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported identity schema')
        if not isinstance(self.model_id, str) or not 0 < len(self.model_id) <= 512:
            raise ValueError('invalid model id')
        if not isinstance(self.model_revision, str) or not re.fullmatch('[0-9a-f]{40}', self.model_revision):
            raise ValueError('model revision must be a full lowercase commit hash')
        for f in fields(self):
            if f.name.endswith('_digest') and not _sha256(getattr(self, f.name)):
                raise ValueError('invalid SHA256 field: ' + f.name)
        for name in ('layer_start', 'layer_end', 'prefix_length', 'position_start', 'batch_size',
                     'num_attention_heads', 'num_key_value_heads', 'head_dim'):
            if type(getattr(self, name)) is not int:
                raise ValueError('expected integer: ' + name)
        if not 0 <= self.layer_start < self.layer_end or self.prefix_length <= 0 or self.position_start != 0:
            raise ValueError('invalid layer or prefix range')
        if self.batch_size != 1 or self.layout != 'bhsd_v1':
            raise ValueError('unsupported batch or layout')
        if self.dtype not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('unsupported dtype')
        if min(self.num_attention_heads, self.num_key_value_heads, self.head_dim) <= 0:
            raise ValueError('invalid head dimensions')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('query heads must be divisible by KV heads')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> 'CacheIdentity':
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('identity fields must match schema exactly')
        return cls(**value)

    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def dht_key(self) -> str:
        return 'kv_cache:v1:' + self.digest()
