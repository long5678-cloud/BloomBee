"""Numerical and physical-address checks for real-page SDPA and device-side page traversal."""
import pytest
import torch
from bloombee.server.dynamo_pages import EnginePagePool, paged_attention_value


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("queries", [1, 3])
def test_real_pages_mask_tail_head_slice(device, fragmented, queries):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires actual GPU")
    device = torch.device("cuda:0" if device == "cuda" else device)
    torch.manual_seed(313)
    dtype = torch.float16
    block, heads, dim = 4, 4, 16
    pool = EnginePagePool(6, block, heads, dim, dtype=dtype, device=device)
    def sample(n): return torch.randn(n, heads, dim, dtype=dtype, device=device)
    k, v = sample(11), sample(11)
    with pool.begin() as seed:
        seed.append(k[:4], v[:4])
        seed.publish(["prefix"])
    with pool.begin() as other:
        if fragmented:
            other.append(sample(4), sample(4))
        with pool.begin(["prefix"]) as session:
            session.append(k[4:], v[4:])
            slot_ids = [s for s, _ in session.slots]
            assert slot_ids == ([0, 2, 3] if fragmented else [0, 1, 2])
            kd, vd = (op.head_slice(1,3) for op in session.operands())
            q = torch.randn(2, queries, dim, device=device, dtype=dtype)
            nk, nv = sample(queries)[:,1:3], sample(queries)[:,1:3]
            total = len(k) + queries
            mask = torch.zeros((1,1,queries,total), dtype=dtype,device=device)
            mask[..., 3] = -3.
            mask[..., -1] = -torch.inf
            ref_k = torch.cat((k[:,1:3],nk)).transpose(0,1)
            ref_v = torch.cat((v[:,1:3],nv)).transpose(0,1)
            logits = q.float() @ ref_k.float().transpose(1,2) / dim**0.5 + mask[0].float()
            ref = logits.softmax(-1) @ ref_v.float()
            actual = paged_attention_value(q,kd,vd,nk,nv,mask,batch=1,heads=2)
            torch.testing.assert_close(actual[0].float(),ref,rtol=.003,atol=.002)
            record = pool.events[-1]
            assert record["history_kv_gather_bytes"] == 0
            assert record["key_ptrs"] == [pool.k[s,0,1].data_ptr() for s in slot_ids]
            assert record["value_ptrs"] == [pool.v[s,0,1].data_ptr() for s in slot_ids]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fragmented", [False, True])
def test_bool_mask_replaces_logits_even_when_all_false_or_visible_logits_below_sentinel(device, fragmented):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires actual GPU")
    device = torch.device("cuda:0" if device == "cuda" else device)
    # The three rows exercise all-False, mixed visible/hidden and the finite
    # sentinel being HIGHER than a genuine visible logit. Values identify which won.
    heads, dim, block, length, queries = 1, 16, 4, 8, 3
    dtype = torch.float32
    pool = EnginePagePool(5, block, heads, dim, dtype=dtype, device=device)
    k = torch.zeros(length + queries, heads, dim, dtype=dtype, device=device)
    k[:block, :, 0] = 20.
    k[block:, :, 0] = -20.
    v = torch.zeros_like(k)
    v[:block, :, 0] = 1.
    v[block:, :, 1] = 1.
    q = torch.zeros(heads, queries, dim, dtype=dtype, device=device)
    q[0, :, 0] = torch.tensor([2., 2., 20000.], device=device)
    mask = torch.zeros(1, 1, queries, length + queries, dtype=torch.bool, device=device)
    mask[:, :, 1, [0, block]] = True
    mask[:, :, 2, block] = True
    with pool.begin() as seed:
        seed.append(k[:block], v[:block])
        seed.publish(["first"])
    with pool.begin() as occupied:
        if fragmented:
            occupied.append(k[:block], v[:block])
        with pool.begin(["first"]) as lease:
            lease.append(k[block:length], v[block:length])
            assert [s for s, _ in lease.slots] == ([0, 2] if fragmented else [0, 1])
            got = paged_attention_value(q, *lease.operands(), k[length:], v[length:], mask, batch=1, heads=heads)
            logits = (q @ k.permute(1, 2, 0)) * dim**-.5
            ref = logits.masked_fill(~mask[0], -65504.).softmax(-1) @ v.transpose(0, 1)
            torch.testing.assert_close(got[0], ref, rtol=.0001, atol=.00001)
