"""Device-side logical page traversal for non-contiguous physical pages."""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged(Q, K, V, NK, NV, MASK, TABLE, OUT,
           L, N, H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, QS: tl.constexpr,
           qh, qt, kd, kt, kh, nt, nh, mt, oh, ot, head0,
           IS_BOOL: tl.constexpr, SCALE: tl.constexpr, TILE: tl.constexpr, WIDTH: tl.constexpr):
    h = tl.program_id(0)
    pos = tl.program_id(1)
    d = tl.arange(0, WIDTH)
    q = tl.load(Q + h * qh + pos * qt + d, d < D, other=0).to(tl.float32)
    acc = tl.full((WIDTH,), 0, tl.float32)
    maximum = -float("inf")
    denom = 0.
    for start in range(0, tl.cdiv(L + N, TILE)):
        i = start * TILE + tl.arange(0, TILE)
        history = i < L
        tail = (i >= L) & (i < L + N)
        slots = tl.load(TABLE + i // BS, history, other=0)
        addr = slots * kd + (i % BS) * kt + (head0 + h) * kh
        newaddr = (i - L) * nt + h * nh
        keys = tl.load(K + addr[:, None] + d[None, :], history[:, None] & (d[None, :] < D), other=0)
        kval = tl.load(NK + newaddr[:, None] + d[None, :], tail[:, None] & (d[None, :] < D), other=0)
        key = tl.where(history[:, None], keys, kval).to(tl.float32)
        score = tl.sum(key * q[None, :], axis=1) * SCALE
        valid = i < L + N
        bias = tl.load(MASK + pos * mt + i, valid, other=0)
        if IS_BOOL:
            # Legacy BloomBee replaces the entire hidden logit with this finite
            # constant. All-False rows consequently average every logical value.
            score = tl.where(bias, score, -65504.)
        else:
            score += bias.to(tl.float32)
        score = tl.where(valid, score, -float("inf"))
        newmax = tl.maximum(maximum, tl.max(score, axis=0))
        safe = tl.where(newmax == -float("inf"), 0., newmax)
        alpha = tl.exp(maximum - safe)
        prob = tl.exp(score - safe)
        vals = tl.load(V + addr[:, None] + d[None, :], history[:, None] & (d[None, :] < D), other=0)
        vnew = tl.load(NV + newaddr[:, None] + d[None, :], tail[:, None] & (d[None, :] < D), other=0)
        val = tl.where(history[:, None], vals, vnew).to(tl.float32)
        acc = acc * alpha + tl.sum(prob[:, None] * val, axis=0)
        denom = denom * alpha + tl.sum(prob, axis=0)
        maximum = newmax
    tl.store(OUT + h * oh + pos * ot + d, acc / denom, d < D)


def fragmented_attention(q, key, k_new, v_new, mask, slots):
    pool = key.session.pool
    if (pool.k.stride() != pool.v.stride()
            or pool.k.stride(-1) != 1):
        raise ValueError("fragmented KV requires contiguous head features and matching page strides")
    if q.stride(-1) != 1:
        q = q.contiguous()
    k_new, v_new = k_new.contiguous(), v_new.contiguous()
    mask = mask.contiguous()
    indices = torch.tensor([s for s, _ in slots], dtype=torch.int32, device=q.device)
    heads, queries, dim = q.shape
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    _paged[(heads, queries)](
        q, pool.k, pool.v, k_new, v_new, mask, indices, out,
        key.length, k_new.shape[0], heads, dim, pool.block_size, queries,
        q.stride(0), q.stride(1), pool.k.stride(0), pool.k.stride(1), pool.k.stride(2),
        k_new.stride(0), k_new.stride(1), mask.stride(-2),
        out.stride(0), out.stride(1), key.head_start,
        mask.dtype == torch.bool, dim ** -0.5, 32, triton.next_power_of_2(dim))
    return out
