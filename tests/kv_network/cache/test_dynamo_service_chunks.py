"""Full real CPU TransformerBackend automatic chunks + CPU policy integration.

Uses production TP loader/model/backend from the independently audited 13C
fixture. No stub forward, transport, chunk estimator or invented model class.
Actual ordinary cross-process 8B RPC acceptance remains 13G's responsibility.
"""
from dataclasses import replace

import pytest
import torch

from test_dynamo_pages_chunk import build_backend, full_reference


def setup(tmp_path,monkeypatch):
    for name,value in {"BLOOMBEE_DYNAMO_SERVICE":"1","BLOOMBEE_DYNAMO_GPU_ENGINE":"pages",
        "BLOOMBEE_DYNAMO_CONTEXT":"real-cpu-model/revision/fixed-shards",
        "BLOOMBEE_DYNAMO_BLOCK_SIZE":"2","BLOOMBEE_DYNAMO_CAPACITY_BLOCKS":"8",
        "BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY":"50","BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS":"8",
        "BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES":"16384"}.items():monkeypatch.setenv(name,value)
    backend,manager,module,unused_pool,info=build_backend(tmp_path)
    # The production estimator chooses 2-token chunks for this actual shape.
    backend.max_chunk_size_bytes=2*backend._max_shard_num_heads*backend.dtype_bytes*6
    torch.manual_seed(70)
    inputs=torch.randn(1,6,16,dtype=torch.float16)*.2
    cfg={"context":"real-cpu-model/revision/fixed-shards","token_ids":[1,2,3,4,5,6],
         "priorities":[80,0,80]}
    return backend,manager,module,replace(info,dynamo_cache=cfg),inputs


def cleanup(manager,info):
    manager.cache.force_free(*info.cache_handles)
    with manager.cache.use_cache():pass
    manager._reorder_executor.shutdown(wait=True)


def test_actual_backend_three_chunks_apply_persistent_cpu_cutoff_and_finish(tmp_path,monkeypatch):
    backend,manager,module,info,inputs=setup(tmp_path,monkeypatch)
    try:
        assert backend._estimate_max_chunk_length(inputs,info)==2
        output,keep=backend.inference_step(inputs,torch.tensor([0]),info)
        event=backend._dynamo_service_events[-1]
        assert event["gpu_engine"]=="physical_pages" and event["cpu_put_chunks"]==3
        rows=event["cpu_progress"]
        assert [(row["chunk_start"],row["ready_tokens"]) for row in rows]==[(0,2),(2,4),(4,6)]
        assert len({row["request_id"] for row in rows})==1
        assert [row["new_blocks"] for row in rows]==[1,0,0]
        assert [row["terminated_at_block"] for row in rows]==[None,1,1]
        assert [row["final"] for row in rows]==[False,False,True]
        assert event["new_blocks"]==1 and event["cpu_progress_finished"]
        state=backend._dynamo_service_state
        assert state.cache.used_blocks==1 and not state.cache._requests
        assert manager.cache._dynamo_page_records[info.cache_handles[0]]["session"].length==6
        torch.testing.assert_close(output,full_reference(module,inputs),atol=2e-3,rtol=2e-3)
        assert torch.equal(keep.cpu(),torch.arange(6)[None])
    finally:cleanup(manager,info)


def test_cpu_retention_error_does_not_discard_real_model_outputs_or_request_cleanup(tmp_path,monkeypatch):
    backend,manager,module,info,inputs=setup(tmp_path,monkeypatch)
    try:
        # Establish the real PageState without replacing any model computation.
        from bloombee.server.dynamo_page_service import PageState
        from bloombee.server.dynamo_service import _binding,native_slab
        backend._ensure_model_on_device()
        with native_slab(backend,info,6) as (key,value):
            state=PageState(_binding(backend),key,inputs)
        backend._dynamo_service_state=state
        actual_put=state.cache.put_completed_blocks
        calls=[]
        def injected_copy_failure(ids,read_block,**kwargs):
            calls.append(len(ids))
            if len(calls)>1:raise ValueError("injected CPU-copy failure after first real chunk")
            return actual_put(ids,read_block,**kwargs)
        monkeypatch.setattr(state.cache,"put_completed_blocks",injected_copy_failure)
        output,_=backend.inference_step(inputs,torch.tensor([0]),info)
        torch.testing.assert_close(output,full_reference(module,inputs),atol=2e-3,rtol=2e-3)
        event=backend._dynamo_service_events[-1]
        assert calls==[2,4,6] and event["cpu_put_chunks"]==3
        assert event["cpu_store_error"] and not state.cache._requests
        assert state.cache.used_blocks==1 and event["cpu_progress_finished"]
    finally:cleanup(manager,info)


def test_actual_full_backend_empty_decode_wire_placeholder(tmp_path,monkeypatch):
    from bloombee.utils.misc import DUMMY_INT64
    backend,manager,module,info,inputs=setup(tmp_path,monkeypatch)
    try:
        backend.inference_step(inputs,torch.tensor([0]),info)
        token=torch.randn(1,1,16,dtype=torch.float16)*.2
        output,_=backend.inference_step(token,DUMMY_INT64,replace(info,prefix_length=6,dynamo_cache=None))
        expected=full_reference(module,torch.cat((inputs,token),dim=1))[:,-1:]
        torch.testing.assert_close(output,expected,atol=2e-3,rtol=2e-3)
        assert manager.cache._dynamo_page_records[info.cache_handles[0]]["session"].length==7
    finally:cleanup(manager,info)
