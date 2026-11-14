"""Ordinary RPC lifecycle for engine-owned pages and CPU missing-block replicas.

One active page session per backend is supported; other sessions enter the
original dense path before their first prefill. No history K/V gather/import
into a dense GPU slab exists here. Aux inputs/outputs are separately budgeted.
"""
import hashlib
import json
import math
import os
from time import perf_counter

import torch
from bloombee.server.page_control import atomic_metadata, metadata_guard

from bloombee.server.dynamo_service import (
    _State, _binding, _emit, _fallback, _present, _enabled_flag,
    _suffix_metadata, native_slab, restore_complete_output, validate_mode,
    UnsupportedReuse,
    logger,
)


from bloombee.kv_cache import page_retention


class PageState(_State):
    def __init__(self, binding, key, inputs):
        from bloombee.server.dynamo_pages import EnginePagePool
        super().__init__(binding)
        capacity = int(os.environ.get("BLOOMBEE_DYNAMO_GPU_CAPACITY_BLOCKS", "17"))
        block = self.cache.block_size
        if capacity <= 0:
            raise UnsupportedReuse("physical GPU page capacity is disabled")
        self.input_dtype = inputs.dtype
        self.hidden = inputs.shape[-1]
        kv_bytes = capacity * block * key.shape[1] * key.shape[2] * key.element_size() * 2
        aux_bytes = capacity * block * self.hidden * inputs.element_size() * 2
        self.budget = int(os.environ.get("BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES", "0"))
        if kv_bytes + aux_bytes > self.budget:
            raise UnsupportedReuse(
                "physical pages plus auxiliary state exceed GPU history byte budget: "
                f"required_bytes={kv_bytes + aux_bytes} kv_bytes={kv_bytes} "
                f"aux_bytes={aux_bytes} budget_bytes={self.budget}"
            )
        self.pool = EnginePagePool(capacity, block, key.shape[1], key.shape[2],
                                   dtype=key.dtype, device=key.device)
        shape = (capacity, block, 1, self.hidden)
        self.aux_input = torch.empty(shape, device=key.device, dtype=inputs.dtype)
        self.aux_output = torch.empty_like(self.aux_input)
        self.aux_generation = [-1] * capacity
        self.payload_bytes = kv_bytes + aux_bytes
        self.sessions = {}
        self.remote_gpu_sources={}
        self.remote_cpu_sources={}


def _keys(state, ids):
    return tuple((state.binding, tuple(ids[:end]))
                 for end in range(state.cache.block_size, len(ids) + 1, state.cache.block_size))


def _page_records(memory):
    if not hasattr(memory, "_dynamo_page_records"):
        memory._dynamo_page_records = {}
        memory._dynamo_failed_page_handles = set()
    return memory._dynamo_page_records


def _register(backend, state, session, handle, info):
    memory = backend.cache_manager.cache
    record = dict(state=state, session=session, uid=str(backend.name),
                  binding=state.binding, initial_info=info)
    config = getattr(info, "dynamo_cache", None)
    if getattr(memory, "release_journal", None) is not None and isinstance(config, dict):
        memory.release_journal.register(config.get("attempt_id"), handle, str(backend.name),
                                        memory._release_allocations.get(handle))
    _page_records(memory)[handle] = record
    state.sessions[handle] = record
    def release():
        # A queued free can be drained by the ordinary dense path too. Include
        # it in the same real metadata write/version domain as page forwards.
        with metadata_guard(write=True):
            returned = None
            with session.pool.lock:
                if not session.closed:
                    session.release()
                    if getattr(memory, "release_journal", None) is not None:
                        from bloombee.server.page_release_ack import capture_release
                        returned = capture_release(record, handle, handle in memory._dynamo_failed_page_handles)
            state.sessions.pop(handle, None)
            _page_records(memory).pop(handle, None)
            memory._dynamo_failed_page_handles.discard(handle)
            _prune_sources(state)
            return returned
    memory._dynamo_session_release[handle] = release
    return record


def _fail(memory, handle):
    record = _page_records(memory).get(handle)
    if record is None:
        return  # A queued free already released the session and cleared its failure marker.
    try:
        if not record["session"].closed:
            record["session"].release()
    finally:
        # Preserve the registry until the actual free message: never silently
        # continue on the unused legacy dense slab following a page error.
        memory._dynamo_failed_page_handles.add(handle)


def _public_metadata(record):
    result = dict(record)
    if "key" in result:
        key = result.pop("key")
        result["key_sha256"] = hashlib.sha256(str(key).encode()).hexdigest()
    return result


def _public_pool(pool):
    record = pool.describe()
    record["slots"] = [_public_metadata(slot) for slot in record["slots"]]
    proof=os.environ.get("BLOOMBEE_DYNAMO_PAGE_CONTENT_PROOF","0")=="1"
    hashed=0
    if proof:
        for item in record["slots"]:
            slot=item["slot"]
            page=pool.pages[slot]
            if page.key is None:
                continue
            pool._wait(page)
            for label,backing in (("key",pool.k),("value",pool.v)):
                tensor=backing[slot].detach().to("cpu").contiguous()
                item[label+"_content_sha256"]=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                hashed+=tensor.numel()*tensor.element_size()
    record["content_proof_enabled"]=proof
    record["content_hash_payload_bytes"]=hashed
    record["diagnostic_gpu_readback_bytes"]=hashed if pool.device.type=="cuda" else 0
    return record


def _describe_prefix(state, session, tokens, *, include_partial=False):
    result = []
    blocks=(tokens+state.cache.block_size-1)//state.cache.block_size if include_partial else tokens//state.cache.block_size
    for index in range(blocks):
        view = session.block(index)
        slot = view["slot"]
        result.append(dict(slot=slot, generation=view["generation"],
            logical_start=index*state.cache.block_size,
            tokens=min(state.cache.block_size,tokens-index*state.cache.block_size),
            key_ptr=view["key"].data_ptr(), value_ptr=view["value"].data_ptr()))
    return result


def _validate_gpu_input(state, session, inputs, tokens):
    outputs = []
    block = state.cache.block_size
    for index in range(tokens // block):
        view = session.block(index)
        slot, generation = view["slot"], view["generation"]
        if state.aux_generation[slot] != generation:
            raise UnsupportedReuse("GPU page has no matching auxiliary generation")
        actual = inputs[:, index*block:(index+1)*block].transpose(0,1)
        if actual.dtype != state.input_dtype or not torch.equal(actual.to(state.pool.device),state.aux_input[slot]):
            raise UnsupportedReuse("actual incoming layer prefix differs from native GPU history")
        outputs.append(state.aux_output[slot].transpose(0,1))
    return torch.cat(outputs,dim=1) if outputs else inputs[:, :0].to(state.pool.device)


def _publish_aux(state, session, inputs, output):
    block = state.cache.block_size
    copied=0
    for index in range(inputs.shape[1] // block):
        view = session.block(index)
        slot, generation = view["slot"], view["generation"]
        # Shared pinned history is immutable. It already passed the actual
        # input check; do not rewrite its auxiliary bytes on every hit.
        if state.aux_generation[slot] != generation:
            state.aux_input[slot].copy_(inputs[:, index*block:(index+1)*block].transpose(0,1))
            state.aux_output[slot].copy_(output[:, index*block:(index+1)*block].transpose(0,1))
            state.aux_generation[slot] = generation
            copied+=state.aux_input[slot].numel()*state.aux_input.element_size()*2
    return copied


def _prune_sources(state):
    live={(i,page.generation) for i,page in enumerate(state.pool.pages) if page.refs or page.key is not None}
    state.remote_gpu_sources={k:v for k,v in state.remote_gpu_sources.items() if k in live}
    state.remote_cpu_sources={k:v for k,v in state.remote_cpu_sources.items() if k in state.cache.entries}


def _gpu_sources(state,session,tokens):
    rows=[]
    for i in range(tokens//state.cache.block_size):
        view=session.block(i)
        source=state.remote_gpu_sources.get((view["slot"],view["generation"]),dict(origin="local_compute"))
        rows.append(dict(source,logical_start=i*state.cache.block_size,
            slot=view["slot"],generation=view["generation"]))
    return rows


def _cpu_sources(state,ids):
    return [dict(state.remote_cpu_sources.get(key,dict(origin="local_cpu")),
        logical_start=i*state.cache.block_size,cpu_block_digest=state.cache.entries[key].digest)
        for i,key in enumerate(_keys(state,ids)) if key in state.cache.entries]


def _cpu_put(state, session, ids, config, inputs, output, *, request_id=None, final=True):
    if page_retention.enabled():
        # Chunk outputs are needed by the ordinary service; its terminal fence
        # owns the one historical page. Chunk callbacks have no admission right.
        return dict(admitted=False, new_blocks=0, copied_bytes=0, evicted=[],
                    terminated_at_block=None, evicted_metadata=[], resident_bytes=state.cache.resident_bytes())
    block = state.cache.block_size
    def read(index):
        view = session.block(index)
        if view["length"] != block:
            raise RuntimeError("cannot offload an incomplete physical page")
        start,end = index*block,(index+1)*block
        return {0:{"key":view["key"],"value":view["value"]},
                1:{"key":output[:,start:end].transpose(0,1),
                   "value":inputs[:,start:end].transpose(0,1)}}
    result=state.cache.put_completed_blocks(ids,read,priorities=config.get("priorities"),
                                             request_id=request_id,final=final)
    keys=_keys(state,ids)
    for key in result["evicted"]:state.remote_cpu_sources.pop(key,None)
    for index in result.get("new_block_indices",[]):
        view=session.block(index)
        source=dict(state.remote_gpu_sources.get((view["slot"],view["generation"]),dict(origin="local_compute")))
        digest=state.cache.entries[keys[index]].digest
        # Do not claim remote provenance if either actual KV or aux changed.
        if source.get("source_block_digest") not in (None,digest):
            source=dict(origin="local_transformed",previous_source=source.get("sha256"))
        source.update(last_transfer="gpu_to_cpu",cpu_block_digest=digest)
        state.remote_cpu_sources[keys[index]]=source
    return result


class ChunkOffload:
    """Per-ordinary-step CPU progress; consume actual completed page intervals."""
    def __init__(self,backend,state,session,info,config,inputs,prefix_output,skip):
        self.backend,self.state,self.session=backend,state,session
        self.config,self.inputs=config,inputs
        self.request_id=f"{os.getpid()}:{backend.name}:{info.cache_handles[0]}"
        self.ready=skip
        self.output=torch.empty(inputs.shape,dtype=inputs.dtype,device=state.pool.device)
        self.output[:,:skip].copy_(prefix_output)
        self.rows=[]
        self.results=[]
        self.put_seconds=0.

    def __call__(self,*,session,start,end,inputs,outputs):
        if (session is not self.session or start!=self.ready or end!=session.length
                or end>self.inputs.shape[1] or end<=start
                or tuple(outputs.shape)!=(1,end-start,self.inputs.shape[2])
                or tuple(inputs.shape)!=tuple(outputs.shape)):
            raise RuntimeError("ordinary chunk progress does not match the bound completed interval")
        self.output[:,start:end].copy_(outputs)
        self.ready=end
        final=end==self.inputs.shape[1]
        started=perf_counter()
        try:
            result=_cpu_put(self.state,session,self.config["token_ids"][:end],self.config,
                self.inputs[:,:end],self.output[:,:end],request_id=self.request_id,final=final)
            error=None
        except Exception as failure:
            result=dict(admitted=False,new_blocks=0,copied_bytes=0,terminated_at_block=None,
                        resident_bytes=self.state.cache.resident_bytes(),evicted_metadata=[])
            error=type(failure).__name__+": "+str(failure)
        self.put_seconds+=perf_counter()-started
        self.results.append(result)
        row=dict(event="dynamo_progress",uid=str(self.backend.name),request_id=self.request_id,
            chunk_index=len(self.rows),chunk_start=start,ready_tokens=end,final=final,
            new_blocks=result["new_blocks"],copied_bytes=result["copied_bytes"],
            terminated_at_block=result["terminated_at_block"],admitted=result["admitted"],
            reader_scope="completed_native_pages_only",error=error)
        self.rows.append(row)
        logger.info("DYNAMO_PROGRESS %s",json.dumps(row,sort_keys=True))

    def finish(self):
        self.state.cache.finish_request(self.request_id)

    def prime_remote(self,remote_start):
        """Admit the already validated imported prefix through the SAME request.

        This is retention, not another model chunk. Later real chunk callbacks
        continue its first-low-priority cutoff and finalization unchanged.
        """
        started=perf_counter()
        try:
            result=_cpu_put(self.state,self.session,self.config["token_ids"][:self.ready],self.config,
                self.inputs[:,:self.ready],self.output[:,:self.ready],request_id=self.request_id,final=False)
            error=None
        except Exception as failure:
            result=dict(admitted=False,new_blocks=0,copied_bytes=0,terminated_at_block=None,
                        resident_bytes=self.state.cache.resident_bytes(),evicted_metadata=[])
            error=type(failure).__name__+": "+str(failure)
        self.put_seconds+=perf_counter()-started
        self.results.append(result)
        indices=[i for i in result.get("new_block_indices",[]) if i>=remote_start//self.state.cache.block_size]
        return dict(remote_cpu_admitted_tokens=len(indices)*self.state.cache.block_size,
            remote_cpu_admitted_block_indices=indices,
            remote_cpu_cutoff=result["terminated_at_block"],remote_cpu_admission_error=error,
            remote_cpu_admission_total_copy_bytes=result["copied_bytes"])

    def aggregate(self):
        result=dict(self.results[-1])
        result["new_blocks"]=sum(row["new_blocks"] for row in self.results)
        result["copied_bytes"]=sum(row["copied_bytes"] for row in self.results)
        result["evicted_metadata"]=[item for row in self.results for item in row.get("evicted_metadata",[])]
        return result


@atomic_metadata(write=True)
def run_page_prefill(backend, inputs, hypo_ids, info, uncached):
    """Initial ordinary opt-in; all conditions checked before page binding."""
    requested = inputs.shape[1]
    session = state = None
    try:
        config = validate_mode(backend,inputs,hypo_ids,info)
        backend._ensure_model_on_device()
        binding = _binding(backend)
        with native_slab(backend,info,requested) as (key,value):
            key_device,key_dtype,key_shape = key.device,key.dtype,key.shape
            if inputs.dtype != key_dtype or backend.dtype != key_dtype:
                raise UnsupportedReuse("direct pages require matching input/model/native cache dtype")
            state = getattr(backend,"_dynamo_service_state",None)
            if not isinstance(state,PageState) or state.binding != binding or state.pid != os.getpid():
                state = PageState(binding,key,inputs)
                backend._dynamo_service_state = state
        if (state.input_dtype != inputs.dtype or state.hidden != inputs.shape[-1]
                or state.pool.device != key_device or state.pool.dtype != key_dtype):
            raise UnsupportedReuse("page input/native execution schema changed")
        required_pages = math.ceil(key_shape[0] / state.cache.block_size)
        if required_pages > state.pool.capacity_pages:
            raise UnsupportedReuse(
                "page pool cannot reserve this whole session's maximum length: "
                f"required_pages={required_pages} capacity_pages={state.pool.capacity_pages} "
                f"session_tokens={key_shape[0]} block_size={state.cache.block_size}"
            )
        if state.sessions:
            raise UnsupportedReuse("another active page session owns this backend; using original dense path")
        from bloombee.server.page_control import register
        register(backend, state)
    except Exception as error:
        return _fallback(backend,inputs,hypo_ids,info,uncached,error)

    handle, = info.cache_handles
    memory = backend.cache_manager.cache
    with state.lock:
        max_skip = min((requested-1)//state.cache.block_size*state.cache.block_size,
                       int(os.environ.get('BLOOMBEE_REUSABLE_PREFIX_BLOCKS', str(len(config['token_ids'])//state.cache.block_size)))*state.cache.block_size)
        keys = _keys(state,config["token_ids"])
        hints=config.get("priorities")
        gpu_hints=[(None if hints is None or i>=len(hints) else hints[i]) for i in range(len(keys))]
        started = perf_counter()
        remote=None
        remote_event=dict(remote_hit_tokens=0,remote_read_range=[0,0],remote_lookup_count=0,
            remote_http_bytes=0,remote_payload_bytes=0,remote_kv_h2d_bytes=0,remote_records=[],remote_errors=[])
        try:
            retained_plan = page_retention.begin_prefill(state, config) if page_retention.enabled() else None
            session = state.pool.begin(keys[:max_skip//state.cache.block_size],
                                       priorities=gpu_hints[:max_skip//state.cache.block_size])
            gpu_count = session.length
            prefix_output = _validate_gpu_input(state,session,inputs,gpu_count)
            source_pages = _describe_prefix(state,session,gpu_count)
            retained,cpu_count,cpu_get = state.cache.get_range(config["token_ids"],
                start_tokens=gpu_count,end_tokens=max_skip,target_device="cpu")
            count = gpu_count+cpu_count
            if cpu_count:
                previous = retained[1]["value"].transpose(0,1)
                if previous.dtype != inputs.dtype or not torch.equal(previous.to(inputs.device),inputs[:,gpu_count:count]):
                    raise UnsupportedReuse("CPU page prefix input differs from actual layer input")
                imported = session.append(retained[0]["key"],retained[0]["value"])
                for index in range(cpu_count//state.cache.block_size):
                    view = session.block(gpu_count//state.cache.block_size+index)
                    start,end = index*state.cache.block_size,(index+1)*state.cache.block_size
                    if (not torch.equal(view["key"].cpu(),retained[0]["key"][start:end])
                            or not torch.equal(view["value"].cpu(),retained[0]["value"][start:end])):
                        raise UnsupportedReuse("CPU import into native physical pages did not persist")
                    local_key=_keys(state,config["token_ids"])[gpu_count//state.cache.block_size+index]
                    source=dict(state.remote_cpu_sources.get(local_key,dict(origin="local_cpu")))
                    source["last_transfer"]="cpu_to_gpu"
                    state.remote_gpu_sources[(view["slot"],view["generation"])]=source
                prefix_output = torch.cat((prefix_output,
                    retained[1]["key"].transpose(0,1).to(state.pool.device)),dim=1)
            else:
                imported = dict(copied_bytes=0)
            # Real ordinary path: only the remaining continuous range may use
            # DHT/HTTP. Local GPU/CPU bytes are never fetched again.
            from bloombee.server.dynamo_remote import get_remote
            remote_event["remote_read_range"]=[count,count]
            remote_start=count
            try:
                from bloombee.server.page_control import remote_allowed
                remote=get_remote(backend) if remote_allowed() else None
                if remote is not None:
                    from bloombee.server.page_control import bundle_registry
                    remote=bundle_registry().ordinary_remote(remote, config)
                    remote_outputs,remote_event=remote.recover(backend,state,session,config,inputs,max_skip)
                    if remote_outputs:
                        prefix_output=torch.cat([prefix_output,*remote_outputs],dim=1)
                    count=session.length
            except Exception as error:
                # An I/O/configuration rejection has no model-output side
                # effects. If native page writes failed, let the outer safety
                # path release the lease rather than using a poisoned pool.
                if state.pool.poisoned or session.length!=count:raise
                remote_event["remote_errors"].append(type(error).__name__+": "+str(error))
            get_seconds = perf_counter()-started
        except Exception as error:
            if state.cache.active:state.cache.release()
            if session is not None and not session.closed:session.release()
            return _fallback(backend,inputs,hypo_ids,info,uncached,error)

        _register(backend,state,session,handle,info)
        progress=ChunkOffload(backend,state,session,info,config,inputs,prefix_output,count)
        try:
            remote_admission={}
            if remote_event["remote_hit_tokens"]:
                remote_admission=progress.prime_remote(remote_start)
            started = perf_counter()
            suffix = inputs[:,count:]
            changed = _suffix_metadata(info,count,suffix.shape[1],inputs.device)
            with backend.cache_manager.bind_dynamo_pages(info.cache_handles,session,
                                                         on_chunk_complete=progress):
                suffix_output,_ = uncached(suffix,hypo_ids,changed)
            if session.length != requested:
                raise RuntimeError("native backend did not commit exactly the newly computed suffix")
            if progress.rows:
                if progress.ready!=requested:
                    raise RuntimeError("backend did not report final ordinary chunk progress")
                output=progress.output.to(suffix_output.device)
            else:
                # Compatibility for a component caller without the production
                # backend chunk loop. This is not a chunk acceptance witness.
                output=restore_complete_output(prefix_output.to(suffix_output.device),suffix_output,requested)
            if output.shape != inputs.shape or output.dtype != inputs.dtype:
                raise RuntimeError("page cache changed ordinary RPC output shape")
            keep = torch.arange(requested,device=output.device).unsqueeze(0)
            backend._last_keep_indices = keep
            compute_seconds = perf_counter()-started
            aux_copied=_publish_aux(state,session,inputs,output)
            retained_limit = int(os.environ.get('BLOOMBEE_REUSABLE_PREFIX_BLOCKS', str(len(keys))))
            if retained_plan is None:
                session.publish(keys[:retained_limit],gpu_hints[:retained_limit])
            else:
                def read_first(index):
                    if index != 0:
                        raise RuntimeError("only first block is reusable")
                    view = session.block(0)
                    return {0: {"key":view["key"], "value":view["value"]},
                            1: {"key":output[:, :16].transpose(0, 1),
                                "value":inputs[:, :16].transpose(0, 1)}}
                retained_plan = page_retention.stage(state, session, config, read_first)
            started = perf_counter()
            try:
                stored = (progress.aggregate() if progress.rows else
                    _cpu_put(state,session,config["token_ids"],config,inputs,output))
                store_error = next((row["error"] for row in progress.rows if row["error"]),None)
            except Exception as error:
                stored = dict(admitted=False,new_blocks=0,copied_bytes=0,terminated_at_block=None,
                              evicted_metadata=[],resident_bytes=state.cache.resident_bytes())
                store_error = type(error).__name__+": "+str(error)
            put_seconds = progress.put_seconds+perf_counter()-started
            publication=dict(published_blocks=0,published_bytes=0,records=[])
            if remote is not None:
                try:publication=remote.publish(backend,state,session,config,inputs,output)
                except Exception as error:publication["errors"]=[type(error).__name__+": "+str(error)]
            events = state.pool.drain_events()
            _prune_sources(state)
            attention = [event for event in events if event["event"]=="attention_direct_pages"]
            telemetry_identity = None
            if count:
                from bloombee.server.dynamo_remote import telemetry_settings, block_identity
                identity_settings = telemetry_settings()
                if identity_settings is not None:
                    telemetry_identity = block_identity(identity_settings, backend, state,
                                                         config['token_ids'], count).to_dict()
            _emit(backend,mode="reuse" if count else "cold",gpu_engine="physical_pages",
                telemetry_attempt_id=config.get('attempt_id'), telemetry_identity=telemetry_identity,
                telemetry_identity_unavailable_reason='not_configured' if count and telemetry_identity is None else None,
                gpu_engine_class=type(state.pool).__module__+"."+type(state.pool).__qualname__,
                requested_tokens=requested,computed_tokens=requested-count,skipped_tokens=count,
                returned_tokens=requested,gpu_hit_tokens=gpu_count,cpu_hit_tokens=cpu_count,
                cpu_read_range=[gpu_count,gpu_count+cpu_count],cpu_read_payload_bytes=cpu_get["copy_bytes"],
                cpu_kv_h2d_bytes=imported["copied_bytes"],gpu_prefix_copy_bytes=0,
                cpu_import_verification_d2h_bytes=imported["copied_bytes"],
                history_kv_gather_bytes=sum(event["history_kv_gather_bytes"] for event in attention),
                new_gpu_kv_write_bytes=sum(event["copied_bytes"] for event in events
                    if event["event"]=="append" and event["source_device"].startswith("cuda")),
                new_gpu_aux_write_bytes=aux_copied,
                native_attention_in_place=bool(attention) if count else False,
                source_gpu_pages=source_pages,
                active_gpu_pages=_describe_prefix(state,session,count),
                resident_request_pages=_describe_prefix(state,session,requested,include_partial=True),
                page_pool=_public_pool(state.pool),page_events=[_public_metadata(e) for e in events],
                gpu_resident_payload_bytes=state.payload_bytes,gpu_budget_bytes=state.budget,
                gpu_auxiliary_payload_bytes=state.aux_input.numel()*state.aux_input.element_size()*2,
                cpu_blocks=state.cache.used_blocks,retained_payload_bytes=state.cache.resident_bytes(),
                new_blocks=stored["new_blocks"],new_payload_copy_bytes=stored["copied_bytes"],
                stored=stored["admitted"],terminated_at_block=stored["terminated_at_block"],
                cpu_evictions=[dict(prefix_tokens=len(e["key"][1]),
                    prefix_sha256=hashlib.sha256(json.dumps(e["key"][1]).encode()).hexdigest(),
                    priority=e["priority"],returned_tick=e["returned_tick"],acquired_tick=e["acquired_tick"])
                    for e in stored.get("evicted_metadata",[])],
                cpu_store_error=store_error,native_layers_imported=int(bool(cpu_count or remote_event["remote_hit_tokens"])),
                native_cpu_layers_imported=int(bool(cpu_count)),
                native_remote_layers_imported=int(bool(remote_event["remote_hit_tokens"])),
                native_import_exact=bool(cpu_count or remote_event["remote_hit_tokens"]),full_cache_hit=False,tail_forward_required=True,
                cache_methods=["get_range","put_completed_blocks"],auxiliary_activation_layer=True,
                cache_get_seconds=get_seconds,compute_restore_seconds=compute_seconds,cache_put_seconds=put_seconds,
                compute_restore_includes_progress_put=bool(progress.rows),
                cpu_progress=progress.rows,cpu_put_chunks=len(progress.rows),
                cpu_progress_request_id=progress.request_id,
                cpu_progress_finished=progress.ready==requested if progress.rows else False,
                remote_publication=publication,**remote_event,**remote_admission,
                gpu_block_sources=_gpu_sources(state,session,requested),
                cpu_block_sources=_cpu_sources(state,config["token_ids"]),
                retention_selection=retained_plan,
                active_session_reserved_tokens=memory.current_size_tokens,
                active_session_token_budget=memory.max_size_tokens)
            return output,keep
        except Exception:
            _fail(memory,handle)
            raise
        finally:
            if state.cache.active:state.cache.release()
            progress.finish()


@atomic_metadata(write=True)
def run_page_continuation(backend,inputs,hypo_ids,info,uncached,record):
    """Continue on the same leased native pages until actual session free."""
    memory = backend.cache_manager.cache
    handle, = info.cache_handles
    state,session = record["state"],record["session"]
    with state.lock:
        try:
            if handle in memory._dynamo_failed_page_handles or session.closed:
                raise RuntimeError("failed page session requires a new ordinary RPC session")
            if (record["uid"] != str(backend.name) or record["binding"] != _binding(backend)
                    or info.prefix_length != session.length or inputs.ndim!=3 or inputs.shape[0]!=1
                    or info.batch_offset or info.full_batch_size not in (0,1) or info.micro_batch_size not in (0,1)
                    or info.active_adapter or _enabled_flag(info.is_spec_dec) or _enabled_flag(info.need_pruning)
                    or _present(info.kv_cache_position_ids) or _enabled_flag(info.prefill_length)
                    or (_present(hypo_ids) and not torch.equal(hypo_ids.cpu(),torch.zeros(1,dtype=hypo_ids.dtype)))):
                raise UnsupportedReuse("unsupported mid-session page mutation; restart session on dense path")
            if _present(info.keep_indices):
                if not torch.equal(info.keep_indices.cpu(),torch.arange(inputs.shape[1]).unsqueeze(0)):
                    raise UnsupportedReuse("nonidentity page-session output selection")
            if _present(info.tree_attention_mask):
                mask=info.tree_attention_mask
                if tuple(mask.shape)!=(1,session.length+inputs.shape[1]) or not bool(torch.all(mask==1).item()):
                    raise UnsupportedReuse("padding/custom mask cannot switch an active page session")
            prefix=session.length
            with backend.cache_manager.bind_dynamo_pages(info.cache_handles,session):
                result=uncached(inputs,hypo_ids,info)
            if session.length!=prefix+inputs.shape[1] or result[0].shape!=inputs.shape:
                raise RuntimeError("page continuation changed the committed length/output contract")
            state.continuation_calls = getattr(state, 'continuation_calls', 0) + 1
            state.continuation_tokens = getattr(state, 'continuation_tokens', 0) + inputs.shape[1]
            state.last_continuation_length = session.length
            if os.environ.get('BLOOMBEE_PAGE_LIGHT_CONTINUATION', '0') == '1':
                # All actual model calls/checks above are identical. Long runs
                # record counters without per-token whole-pool readback/SHA.
                state.pool.drain_events()
                calls = state.continuation_calls
                if calls <= 3 or calls % 256 == 0:
                    logger.info('DYNAMO_PAGE_CONTINUATION %s', json.dumps(dict(
                        event='dynamo_page_continuation', uid=str(backend.name),
                        handle=handle, computed_tokens=inputs.shape[1],
                        returned_tokens=result[0].shape[1], committed_tokens=session.length,
                        cumulative_calls=calls, cumulative_computed_tokens=state.continuation_tokens,
                        diagnostic_gpu_readback_bytes=0, diagnostic_mode='light_counters_v1')))
                return result
            event=dict(event="dynamo_page_continuation",uid=str(backend.name),handle=handle,
                request_id=f"{os.getpid()}:{backend.name}:{handle}",
                backend_class=type(backend).__module__+"."+type(backend).__qualname__,
                cache_class=type(state.cache).__module__+"."+type(state.cache).__qualname__,
                gpu_engine_class=type(state.pool).__module__+"."+type(state.pool).__qualname__,
                prefix_length=prefix,requested_tokens=inputs.shape[1],computed_tokens=inputs.shape[1],
                returned_tokens=result[0].shape[1],committed_tokens=session.length,
                cpu_history_read_bytes=0,cpu_new_copy_bytes=0,gpu_prefix_copy_bytes=0,
                session_pages=_describe_prefix(state,session,session.length,include_partial=True),
                page_pool=_public_pool(state.pool),
                page_events=[_public_metadata(e) for e in state.pool.drain_events()])
            logger.info("DYNAMO_PAGE_CONTINUATION %s",json.dumps(event,sort_keys=True))
            return result
        except Exception:
            _fail(memory,handle)
            raise
