"""Service/allocator + production Llama attention components on CPU.

This is not RPC or an 8B certificate. The compute callback invokes the real
OptimizedLlamaAttention, fixed tiny weights, real manager select/update methods;
it does not instantiate a substitute model or emulate a transport.
"""
import asyncio
from dataclasses import replace
from contextlib import contextmanager

import pytest
import torch

from test_dynamo_service import guard_probe, actual_pipeline_wrapper
from bloombee.models.llama.block import OptimizedLlamaAttention
from bloombee.server.backend import TransformerBackend
from bloombee.server.memory_cache_manager import KVCacheManager
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor
from bloombee.flexgen_utils.utils import ValueHolder
from bloombee.utils.hivemind_compat import TensorDescriptor


@contextmanager
def actual_attention_components(monkeypatch):
    torch.manual_seed(51)
    backend,base,_=guard_probe()
    for name,value in {"BLOOMBEE_DYNAMO_SERVICE":"1","BLOOMBEE_DYNAMO_GPU_ENGINE":"pages",
        "BLOOMBEE_DYNAMO_CONTEXT":base.dynamo_cache["context"],"BLOOMBEE_DYNAMO_BLOCK_SIZE":"2",
        "BLOOMBEE_DYNAMO_CAPACITY_BLOCKS":"32","BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY":"50",
        "BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS":"8","BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES":"16384"}.items():
        monkeypatch.setenv(name,value)
    cpu=TorchDevice("cpu")
    env=ExecutionEnv(gpu=cpu,cpu=cpu,disk=None,mixed=None)
    comp=CompressionConfig(num_bits=4,group_size=64,group_dim=0,symmetric=False)
    policy=Policy(gpu_batch_size=1,num_gpu_batches=1,w_gpu_percent=100,w_cpu_percent=0,
        cache_gpu_percent=100,cache_cpu_percent=0,act_gpu_percent=100,act_cpu_percent=0,
        overlap=False,sep_layer=True,pin_weight=False,cpu_cache_compute=False,attn_sparsity=1.,
        compress_weight=False,comp_weight_config=comp,compress_cache=False,comp_cache_config=comp)
    manager=KVCacheManager(128,30,policy,env,backend.config)
    module=OptimizedLlamaAttention(config=backend.config,env=env,policy=policy,layer_id=0).eval()
    backend.module=actual_pipeline_wrapper(module)
    backend.cache_manager=manager
    backend.name,backend.dtype="page-components.0",torch.float16
    backend._ensure_model_on_device=lambda:None
    backend._estimate_max_chunk_length=lambda x,info:10000
    tt=lambda value:TorchTensor.create_from_torch(value,cpu)
    weights=[torch.randn(8,8,dtype=torch.float16)*.02 for _ in range(4)]
    norm=torch.ones(8,dtype=torch.float16)
    inv=1/(10000**(torch.arange(0,4,2).float()/4))
    def attention_step(inputs,hypo,info):
        n=inputs.shape[1];prefix=info.prefix_length;total=prefix+n
        allow=torch.arange(total)[None,:]<=torch.arange(prefix,total)[:,None]
        mask=torch.zeros(1,1,n,total).masked_fill(~allow[None,None],-65504.)
        hidden,read,wh,write,mh=[ValueHolder() for _ in range(5)]
        hidden.store(tt(inputs.clone()));mh.store(tt(mask))
        wh.store(tuple((tt(t),False) for t in weights+[norm,inv]))
        with manager.use_cache(*info.cache_handles):
            key,value,_=manager.select_cache(prefix,hypo_ids=hypo,requested_rows=1)
            if key is not None:read.store((key,value))
            positions=torch.arange(prefix,total)[None]
            output,_,_=module(hidden,read,wh,write,attention_mask=mh,position_ids=positions,
                               rotary_position_ids=positions)
            new_key,new_value=write.pop()
            manager.update_cache((new_key.data.permute(1,2,0),new_value.data.permute(1,0,2)),prefix)
        return output.data,torch.arange(n)[None]
    backend._inference_step_uncached=attention_step
    descriptor=TensorDescriptor((1,2,4,16),dtype=torch.float16,device=torch.device("cpu"))
    size=manager.get_allocation_size_tokens(descriptor)
    embedding=torch.randn(1024,8,dtype=torch.float16)
    pending={}
    def request(ids,priorities=None,opt_in=True,finish=True):
        handles=tuple(asyncio.run(manager.cache._schedule_alloc(size,descriptor,timeout=0)))
        cfg={"context":base.dynamo_cache["context"],"token_ids":ids,
             "priorities":[100]*((len(ids)+1)//2) if priorities is None else priorities}
        info=replace(base,cache_handles=handles,dynamo_cache=cfg if opt_in else None)
        pending[handles]=(info,len(ids))
        output,keep=TransformerBackend.inference_step(backend,embedding[ids][None],torch.tensor([0]),info)
        assert output.shape==(1,len(ids),8) and torch.equal(keep.cpu(),torch.arange(len(ids))[None])
        event=backend._dynamo_service_events[-1] if opt_in else None
        if finish:close(handles)
        return output,event,handles
    def close(handles):
        manager.cache.force_free(*handles)
        pending.pop(handles,None)
        with manager.cache.use_cache():pass
    def decode(handles,ids,hypo_ids=None,**changes):
        info,n=pending[handles]
        info=replace(info,prefix_length=n,dynamo_cache=None,keep_indices=None,**changes)
        result=TransformerBackend.inference_step(backend,embedding[ids][None],
            torch.tensor([0]) if hypo_ids is None else hypo_ids,info)
        pending[handles]=(info,n+len(ids))
        return result[0]
    try:
        yield backend,manager,request,decode,close
    finally:
        for handles in tuple(pending):close(handles)
        manager._reorder_executor.shutdown(wait=True)


def test_ordinary_entry_components_gpu_cpu_mixed_real_attention(monkeypatch):
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        q=list(range(1,9));p=q[:6]+[90,91]
        expected,_,_=request(p,opt_in=False)
        request(q)
        output,event,_=request(p,priorities=[0]*4)
        torch.testing.assert_close(output,expected,atol=2e-3,rtol=2e-3)
        assert event["gpu_hit_tokens"]==6 and event["cpu_hit_tokens"]==0
        assert event["cpu_read_payload_bytes"]==0 and event["history_kv_gather_bytes"]==0
        assert event["native_attention_in_place"] and event["new_payload_copy_bytes"]==0
        assert event["source_gpu_pages"]==event["active_gpu_pages"]
        for start in (100,200,300):request(list(range(start,start+8)))
        output,event,_=request(p,priorities=[0]*4)
        assert event["gpu_hit_tokens"]==0 and event["cpu_hit_tokens"]==6
        torch.testing.assert_close(output,expected,atol=2e-3,rtol=2e-3)
        for start in (400,500,600):request(list(range(start,start+8)))
        state=backend._dynamo_service_state
        first=state.cache._path(tuple(q))[0]
        # Unit-only removal creates the exact missing-ancestor state. Actual
        # service acceptance must produce it through normal pressure requests.
        state.cache.entries.pop(first)
        request(q[:2],priorities=[0])
        assert first not in state.cache.entries
        output,event,_=request(p,priorities=[0]*4)
        assert event["gpu_hit_tokens"]==2 and event["cpu_hit_tokens"]==4
        assert event["cpu_read_range"]==[2,6]
        torch.testing.assert_close(output,expected,atol=2e-3,rtol=2e-3)


def test_page_decode_lease_persists_and_second_active_session_falls_back(monkeypatch):
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        q=list(range(1,9))
        reference,_,_=request(q+[200,201],opt_in=False)
        _,_,handles=request(q,finish=False)
        state=backend._dynamo_service_state
        record=manager.cache._dynamo_page_records[handles[0]]
        session=record["session"]
        assert not session.closed and session.length==8
        _,event,_=request(list(range(100,108)))
        assert event["mode"]=="fallback" and "active page session" in event["reason"]
        assert not session.closed and session.length==8
        first=decode(handles,[200]);second=decode(handles,[201])
        torch.testing.assert_close(first,reference[:,8:9],atol=2e-3,rtol=2e-3)
        torch.testing.assert_close(second,reference[:,9:10],atol=2e-3,rtol=2e-3)
        assert session.length==10 and state.sessions
        close(handles)
        assert session.closed and not state.sessions and not manager.cache._dynamo_page_records


def test_unsupported_mid_session_cannot_silently_read_dense_slab(monkeypatch):
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        _,_,handles=request(list(range(1,9)),finish=False)
        with pytest.raises(ValueError,match="mid-session"):
            decode(handles,[200],kv_cache_position_ids=torch.tensor([0]))
        assert handles[0] in manager.cache._dynamo_failed_page_handles
        with pytest.raises(RuntimeError,match="failed page session"):
            decode(handles,[200])
        close(handles)
        assert not manager.cache._dynamo_failed_page_handles


@pytest.mark.parametrize("name,value",[("BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES","0"),
    ("BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS","0"),("BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS","1")])
def test_disabled_or_insufficient_pages_fall_back_before_binding(monkeypatch,name,value):
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        ids=list(range(1,9))
        reference,_,_=request(ids,opt_in=False)
        monkeypatch.setenv(name,value)
        output,event,_=request(ids)
        assert event["mode"]=="fallback" and event["computed_tokens"]==8
        assert not manager.cache._dynamo_page_records
        torch.testing.assert_close(output,reference,atol=0,rtol=0)


def test_zero_cpu_capacity_keeps_gpu_history_independent(monkeypatch):
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        monkeypatch.setenv("BLOOMBEE_DYNAMO_CAPACITY_BLOCKS","0")
        q=list(range(1,9));p=q[:6]+[90,91]
        reference,_,_=request(p,opt_in=False)
        request(q)
        output,event,_=request(p)
        assert event["gpu_hit_tokens"]==6 and event["cpu_hit_tokens"]==0 and event["cpu_blocks"]==0
        assert not backend._dynamo_service_state.cache._requests
        torch.testing.assert_close(output,reference,atol=2e-3,rtol=2e-3)


def test_real_empty_hypothesis_decode_and_separate_actual_continuation_log(monkeypatch,caplog):
    import json
    import logging
    from bloombee.utils.misc import DUMMY_INT64
    caplog.set_level(logging.INFO)
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        q=list(range(1,9))
        reference,_,_=request(q+[200],opt_in=False)
        _,event,handles=request(q,finish=False)
        output=decode(handles,[200],hypo_ids=DUMMY_INT64)
        torch.testing.assert_close(output,reference[:,-1:],atol=2e-3,rtol=2e-3)
        rows=[json.loads(r.getMessage().split("DYNAMO_PAGE_CONTINUATION ",1)[1])
              for r in caplog.records if "DYNAMO_PAGE_CONTINUATION " in r.getMessage()]
        assert len(rows)==1 and rows[0]["prefix_length"]==8 and rows[0]["committed_tokens"]==9
        assert rows[0]["requested_tokens"]==rows[0]["returned_tokens"]==1
        assert rows[0]["session_pages"][-1]["tokens"]==1
        attention=[e for e in rows[0]["page_events"] if e["event"]=="attention_direct_pages"]
        assert attention and len(attention[0]["value_ptrs"])==len(attention[0]["key_ptrs"])
        assert attention[0]["head_start"]==0 and attention[0]["heads"]==2
        assert not rows[0]["page_pool"]["content_proof_enabled"]
        assert rows[0]["page_pool"]["diagnostic_gpu_readback_bytes"]==0
        # Continuation events do not inflate the once-per-prefill event count.
        assert len(backend._dynamo_service_events)==1
        close(handles)


def test_opt_in_functional_page_content_proof_is_exact_and_cpu_not_gpu_label(monkeypatch):
    import hashlib
    with actual_attention_components(monkeypatch) as (backend,manager,request,decode,close):
        monkeypatch.setenv("BLOOMBEE_DYNAMO_PAGE_CONTENT_PROOF","1")
        _,event,_=request(list(range(1,9)))
        state=backend._dynamo_service_state
        snapshot=event["page_pool"]
        assert snapshot["content_proof_enabled"] and snapshot["content_hash_payload_bytes"]>0
        assert snapshot["diagnostic_gpu_readback_bytes"]==0  # actual CPU allocation
        assert len(event["resident_request_pages"])==4
        for page in snapshot["slots"]:
            if "key_content_sha256" not in page:continue
            actual=hashlib.sha256(state.pool.k[page["slot"]].view(torch.uint8).numpy().tobytes()).hexdigest()
            assert page["key_content_sha256"]==actual

@pytest.mark.parametrize("key,value,expected", [
    ("BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS", "1",
     ("required_pages=8", "capacity_pages=1", "session_tokens=16", "block_size=2")),
    ("BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES", "0",
     ("required_bytes=1024", "kv_bytes=512", "aux_bytes=512", "budget_bytes=0")),
])
def test_first_capacity_fallback_is_actionable(monkeypatch, capsys, key, value, expected):
    import json
    with actual_attention_components(monkeypatch) as (backend, manager, request, decode, close):
        ids = list(range(1, 9))
        reference, _, _ = request(ids, opt_in=False)
        monkeypatch.setenv(key, value)
        output, _, _ = request(ids)
        lines = capsys.readouterr().out.splitlines()
        rows = [json.loads(line.partition("DYNAMO_SERVICE ")[2])
                for line in lines if "DYNAMO_SERVICE " in line]
        assert len(rows) == 1
        row = rows[0]
        assert row["mode"] == "fallback" and row["skipped_tokens"] == 0
        for term in expected:
            assert term in row["reason"]
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
