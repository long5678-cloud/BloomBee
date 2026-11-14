"""Use PyTorch's fused SDPA on directly addressable KV spans.

Each call returns its log-normalizer. Combining the partial results therefore
normalizes over the entire logical prefix, including the newly projected tail.
"""
import torch


def _partial(q, k, v, mask):
    # q is [1,H,Q,D]; k/v view the physical backing as [1,H,K,D].
    k = k.permute(1, 0, 2).unsqueeze(0)
    v = v.permute(1, 0, 2).unsqueeze(0)
    if q.is_cuda:
        # CUTLASS SDPA needs aligned strides for additive bias. Pad metadata
        # only, keeping the key and value tensors as views.
        n = mask.shape[-1]
        pad = (-n) % 8
        bias = torch.nn.functional.pad(mask.contiguous(), (0, pad))[..., :n] if pad else mask.contiguous()
        out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, bias.expand(q.shape[0], q.shape[1], -1, -1), True, 0., False)
        lse = lse[..., :q.shape[-2]]
    else:
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            q, k, v, 0., False, attn_mask=mask)
    if mask.dtype.is_floating_point:
        # CPU flash attention can report a zero LSE for a fully masked span.
        # Its true probability mass is zero when merging with another span.
        visible = ~torch.isneginf(mask).all(dim=-1)
        lse = torch.where(visible, lse, -torch.inf)
    return out.float(), lse.float()


def physical_page_attention(q, key, k_new, v_new, mask, slots, contiguous):
    pool = key.session.pool
    if mask.dtype not in (q.dtype, torch.bool, torch.float32):
        mask = mask.float()
    q = q.unsqueeze(0)
    # CUDA efficient SDPA requires Q and additive bias to have identical dtype.
    # Keep wider user-supplied bias intact by reading it directly in the tiled kernel.
    if q.is_cuda and (not contiguous or mask.dtype != q.dtype):
        from bloombee.server.fragment_attention import fragmented_attention
        return fragmented_attention(q[0], key, k_new, v_new, mask, slots)

    flat_k = pool.k.view(-1, pool.num_heads, pool.head_dim)
    flat_v = pool.v.view_as(flat_k)
    heads = slice(key.head_start, key.head_end)
    def spans():
        start = 0
        while start < len(slots):
            slot = slots[start][0]
            stop = len(slots) if contiguous else start + 1
            while stop < len(slots) and slots[stop][0] == slot + stop - start:
                stop += 1
            logical = start * pool.block_size
            length = min(key.length, stop * pool.block_size) - logical
            base = slot * pool.block_size
            yield flat_k[base:base+length, heads], flat_v[base:base+length, heads], mask[..., logical:logical+length]
            start = stop
        yield k_new, v_new, mask[..., key.length:]

    def parts():
        for k, v, bias in spans():
            if bias.dtype != torch.bool:
                yield _partial(q, k, v, bias)
                continue
            # CPU SDPA accepts additive bias. Two disjoint populations exactly
            # represent masked_fill: visible logits keep QK; hidden logits are
            # the constant -65504. A zero query gives the hidden values equal weight.
            visible = torch.zeros_like(bias, dtype=q.dtype).masked_fill_(~bias, -torch.inf)
            hidden = torch.zeros_like(visible).masked_fill_(bias, -torch.inf)
            val, norm = _partial(q, k, v, visible)
            yield val, norm.double()
            val, norm = _partial(torch.zeros_like(q), k, v, hidden)
            yield val, norm.double() - 65504.

    combined = total = None
    for value, norm in parts():
        if total is None:
            combined, total = value, norm
        else:
            norm_next = torch.logaddexp(total, norm)
            history_weight = torch.where(torch.isneginf(total), 0., (total - norm_next).exp())
            span_weight = torch.where(torch.isneginf(norm), 0., (norm - norm_next).exp())
            combined = (torch.where(history_weight.unsqueeze(-1) == 0., 0., combined)
                        * history_weight.unsqueeze(-1)
                        + torch.where(span_weight.unsqueeze(-1) == 0., 0., value)
                        * span_weight.unsqueeze(-1))
            total = norm_next
    # softmax([-inf, ..., -inf]) in the original additive-mask path is NaN.
    return combined.masked_fill(torch.isneginf(total).unsqueeze(-1), torch.nan)[0]
