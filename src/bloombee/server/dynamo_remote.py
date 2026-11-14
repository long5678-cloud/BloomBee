"""Ordinary-page DHT discovery and bounded immutable RAM/HTTP data plane.

No CUDA tensor is accessed by the HTTP handler or directory worker. Publication
is invoked only with a live ordinary PageSession; consumption returns validated
single blocks for that same native session. There is no disk snapshot fallback.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.server
import ipaddress
import json
import math
import os
import re
import threading
import time
from urllib.parse import urlsplit
from collections import deque

import torch

from bloombee.kv_cache.identity import CacheIdentity, canonical_digest, token_ids_digest
from bloombee.kv_cache.directory import CacheRecord, KVCacheDirectory
from bloombee.kv_cache.transport import pack_snapshot, unpack_snapshot, fetch_snapshot
from hivemind.utils.logging import get_logger

logger=get_logger(__name__)

PREFIX="BLOOMBEE_DYNAMO_REMOTE_"
SCHEMA="dynamo_page_block_v1"


def telemetry_settings():
    """Optional identity logging must not activate an unconfigured remote plane.

    Explicit remote/identity configuration remains strict, including partial or
    invalid settings. This does not relax the explicit page-probe entrypoint.
    """
    mode = os.environ.get(PREFIX+'MODE', 'off')
    configured = any(key.startswith(PREFIX) and key != PREFIX+'MODE' for key in os.environ)
    if mode == 'off' and not configured:
        return None
    return Settings.from_env(identity_only=True)


def _positive(value,name):
    if isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<=0:
        raise ValueError("invalid remote "+name)
    return float(value)


@dataclass(frozen=True)
class Settings:
    mode: str
    context: str
    model_id: str
    revision: str
    execution_digest: str
    tokenizer_digest: str
    allowed_endpoints: tuple
    allowed_holders: tuple=()
    endpoint: str=""
    bind_host: str="0.0.0.0"
    ttl: float=3600.
    timeout: float=10.
    max_object: int=2*1024**2
    max_total: int=2*1024**3

    def __post_init__(self):
        if self.mode not in ("publish","consume") or not self.context or not self.model_id:
            raise ValueError("remote deployment identity/mode missing")
        if re.fullmatch("[a-f0-9]{40}",self.revision) is None:
            raise ValueError("remote model revision must be full commit")
        for value in (self.execution_digest,self.tokenizer_digest):
            if re.fullmatch("[a-f0-9]{64}",value) is None:raise ValueError("remote digest must be SHA256")
        if not self.allowed_endpoints:raise ValueError("remote endpoint allowlist is required")
        for endpoint in self.allowed_endpoints:
            parsed=urlsplit(endpoint)
            if (parsed.scheme!="http" or not parsed.hostname or not parsed.port
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in ("","/")):
                raise ValueError("remote endpoint must be an exact credential-free HTTP IP:port")
            ipaddress.ip_address(parsed.hostname)
        if self.mode=="publish" and self.endpoint not in self.allowed_endpoints:
            raise ValueError("publisher endpoint not approved")
        if self.mode=="consume" and not self.allowed_holders:
            raise ValueError("consumer requires approved publisher peer identities")
        _positive(self.ttl,"TTL");_positive(self.timeout,"timeout")
        if (type(self.max_object) is not int or not 0<self.max_object<=16*1024**2
                or type(self.max_total) is not int or not 0<self.max_total<=2*1024**3):
            raise ValueError("remote RAM/object bound invalid")
        ipaddress.ip_address(self.bind_host)

    @classmethod
    def from_env(cls, *, identity_only=False):
        mode=os.environ.get(PREFIX+"MODE","off")
        # Identity-only construction never creates a RemotePlane or starts I/O.
        # Local groups retain the same immutable model/execution namespace.
        if identity_only and mode=="off":
            mode="publish"
        return cls(mode=mode,context=os.environ.get(PREFIX+"CONTEXT",""),
            model_id=os.environ.get(PREFIX+"MODEL_ID",""),revision=os.environ.get(PREFIX+"MODEL_REVISION",""),
            execution_digest=os.environ.get(PREFIX+"EXECUTION_DIGEST",""),
            tokenizer_digest=os.environ.get(PREFIX+"TOKENIZER_DIGEST",""),
            allowed_endpoints=tuple(json.loads(os.environ.get(PREFIX+"ALLOWED_ENDPOINTS","[]"))),
            allowed_holders=tuple(json.loads(os.environ.get(PREFIX+"ALLOWED_HOLDERS","[]"))),
            endpoint=os.environ.get(PREFIX+"ENDPOINT",""),bind_host=os.environ.get(PREFIX+"BIND_HOST","0.0.0.0"),
            ttl=float(os.environ.get(PREFIX+"TTL_SECONDS","3600")),
            timeout=float(os.environ.get(PREFIX+"TIMEOUT_SECONDS","10")),
            max_object=int(os.environ.get(PREFIX+"MAX_OBJECT_BYTES",str(2*1024**2))),
            max_total=int(os.environ.get(PREFIX+"MAX_TOTAL_BYTES",str(2*1024**3))))


class RAMStore:
    def __init__(self,max_total,max_object):
        self.max_total,self.max_object=max_total,max_object
        self.objects={};self.nbytes=0;self.lock=threading.Lock()

    def put(self,blob):
        if not 0<len(blob)<=self.max_object:raise ValueError("remote object exceeds bound")
        digest=hashlib.sha256(blob).hexdigest()
        with self.lock:
            if digest in self.objects:
                if self.objects[digest]!=blob:raise ValueError("immutable object collision")
            else:
                if self.nbytes+len(blob)>self.max_total:raise ValueError("publisher RAM budget exhausted")
                self.objects[digest]=blob;self.nbytes+=len(blob)
        return digest


class _RAMHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        match=re.fullmatch(r"/objects/([a-f0-9]{64})",self.path)
        if match is None:self.send_error(400);return
        with self.server.store.lock:blob=self.server.store.objects.get(match[1])
        if blob is None:self.send_error(404);return
        self.send_response(200);self.send_header("Content-Type","application/octet-stream")
        self.send_header("Content-Length",str(len(blob)));self.end_headers()
        started=time.monotonic()
        error=None
        try:
            self.wfile.write(blob)
            self.wfile.flush()
        except OSError as exc:error=type(exc).__name__
        with self.server.events_lock:
            if error is None:self.server.sent_bytes+=len(blob)
            event=dict(event="dynamo_remote_http",peer_ip=self.client_address[0],
                       peer_port=self.client_address[1],sha256=match[1],
                       bytes=len(blob) if error is None else None,expected_bytes=len(blob),
                       status="sent" if error is None else "failed",error=error,
                       cumulative_sent_bytes=self.server.sent_bytes,
                       seconds=time.monotonic()-started,
                       holder=self.server.holder,store_kind="process_ram")
            self.server.events.append(event)
            logger.info("DYNAMO_REMOTE_HTTP %s",json.dumps(event,sort_keys=True))

    def log_message(self,*unused):pass


def _config(backend):
    value=json.loads(json.dumps(backend.config.to_dict(),default=str))
    for key in ("_name_or_path","name_or_path","name","transformers_version","_commit_hash"):
        value.pop(key,None)
    return value


def block_identity(settings,backend,state,ids,end):
    index=getattr(backend,"block_index",None)
    if index is None:index=int(str(backend.name).rsplit(".",1)[1])
    if type(index) is not int or not 0<=index<backend.config.num_hidden_layers:
        raise ValueError("actual remote layer index invalid")
    config=_config(backend);bs=state.cache.block_size
    if end<=0 or end%bs or end>len(ids):raise ValueError("remote block boundary invalid")
    return CacheIdentity(model_id=settings.model_id,model_revision=settings.revision,
        config_digest=canonical_digest(config),tokenizer_digest=settings.tokenizer_digest,
        adapter_digest=canonical_digest(None),layer_start=index,layer_end=index+1,
        prefix_tokens_digest=token_ids_digest(ids[:end]),prefix_length=end,
        position_digest=canonical_digest(dict(start=0,end=end)),attention_digest=canonical_digest("causal_unpadded_v1"),
        rope_digest=canonical_digest({k:config.get(k) for k in ("rope_theta","rope_scaling","max_position_embeddings")}),
        layout="bhsd_v1",dtype=str(state.pool.dtype).removeprefix("torch."),
        num_attention_heads=state.pool.num_heads,num_key_value_heads=backend.config.num_key_value_heads,
        head_dim=state.pool.head_dim,execution_digest=canonical_digest(dict(
            declared=settings.execution_digest,context=settings.context,schema=SCHEMA,block_size=bs,
            auxiliary_hidden=state.hidden,auxiliary_dtype=str(state.input_dtype))))


def wire_identity(identity,block_size):
    return dict(cache_identity=identity.to_dict(),schema=SCHEMA,
                block_start=identity.prefix_length-block_size,block_end=identity.prefix_length)


def normalized_layers(tensors):
    return {0:{name:tensors[name][0].permute(1,0,2) for name in ("key","value")},
            1:{"key":tensors["aux_output"][:,0].transpose(0,1),
               "value":tensors["aux_input"][:,0].transpose(0,1)}}


def source_digests(tensors,cache_digest):
    layers=normalized_layers(tensors)
    return dict(source_block_digest=cache_digest(layers),
        source_key_content_sha256=hashlib.sha256(layers[0]["key"].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        source_value_content_sha256=hashlib.sha256(layers[0]["value"].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())


class RemotePlane:
    def __init__(self,dht,settings):
        self.settings,self.dht=settings,dht
        self.directory=KVCacheDirectory(dht,allowed_endpoints=settings.allowed_endpoints,
            max_payload_bytes=settings.max_object,operation_timeout_s=settings.timeout)
        self.io=ThreadPoolExecutor(max_workers=1,thread_name_prefix="kv-network")
        self.store=RAMStore(settings.max_total,settings.max_object)
        self.http=self.thread=None
        self.bindings={}

    def _bound(self,backend,state):
        index=getattr(backend,"block_index",None)
        if index is None:index=int(str(backend.name).rsplit(".",1)[1])
        old=self.bindings.setdefault(index,state.binding)
        if old!=state.binding:raise ValueError("immutable remote deployment local parameter/owner binding changed")

    def _start_http(self):
        if self.http is not None:return
        parsed=urlsplit(self.settings.endpoint)
        self.http=http.server.HTTPServer((self.settings.bind_host,parsed.port),_RAMHandler)
        self.http.store=self.store
        peer=self.dht.peer_id
        self.http.holder=peer.to_base58() if hasattr(peer,"to_base58") else str(peer)
        self.http.events=deque(maxlen=4096)
        self.http.events_lock=threading.Lock()
        self.http.sent_bytes=0
        self.thread=threading.Thread(target=self.http.serve_forever,name="kv-http",daemon=True)
        self.thread.start()

    def _register(self,identity,blob):
        self._start_http()
        digest=self.store.put(blob)
        peer=self.dht.peer_id
        holder=peer.to_base58() if hasattr(peer,"to_base58") else str(peer)
        record=CacheRecord(identity.digest(),holder,self.settings.endpoint,digest,digest,len(blob),
                           self.directory.clock()+self.settings.ttl)
        if not self.directory.register(identity,record,ttl_seconds=self.settings.ttl):
            raise RuntimeError("remote DHT publish not acknowledged")
        return record

    def publish(self,backend,state,session,config,inputs,outputs):
        if self.settings.mode!="publish":return dict(published_blocks=0,published_bytes=0,records=[])
        started=time.monotonic()
        self._bound(backend,state)
        rows=[];errors=[];bs=state.cache.block_size;hints=config.get("priorities") or []
        for index in range(len(config["token_ids"])//bs):
            priority=0 if index>=len(hints) or hints[index] is None else hints[index]
            if priority<state.cache.host_offload_min_priority:break
            end=(index+1)*bs;start=end-bs
            try:
                view=session.block(index)
                identity=block_identity(self.settings,backend,state,config["token_ids"],end)
                tensors=dict(key=view["key"].permute(1,0,2).unsqueeze(0),
                    value=view["value"].permute(1,0,2).unsqueeze(0),
                    aux_input=inputs[:,start:end].unsqueeze(1),aux_output=outputs[:,start:end].unsqueeze(1))
                # Serialize on the actual runtime stream while the page lease is
                # alive. Only immutable CPU bytes go to the network worker.
                blob=pack_snapshot(wire_identity(identity,bs),tensors,max_bytes=self.settings.max_object)
                digests=source_digests(unpack_snapshot(blob,wire_identity(identity,bs),
                    max_bytes=self.settings.max_object),state.cache._digest)
                record=self.io.submit(self._register,identity,blob).result(timeout=self.settings.timeout+1)
                native_publication = None
                if os.environ.get('BLOOMBEE_NATIVE_PAGE_CONFIG'):
                    import inspect
                    from bloombee.kv_cache.network.native_page_bridge import get_source_collector
                    collector = get_source_collector()
                    if not config.get('attempt_id'):
                        raise ValueError('native page source publication needs exact ordinary attempt_id')
                    native_publication = collector.observe_page(
                        config['attempt_id'], identity, blob, record, config['token_ids'])
                    logger.info('NATIVE_PAGE_PUBLICATION %s', json.dumps(dict(
                        attempt_id=config['attempt_id'], layer=identity.layer_start,
                        bridge_file=inspect.getfile(type(collector)),
                        status=native_publication.get('status'),
                        object_id=(native_publication.get('manifest') or {}).get('object_id')),
                        sort_keys=True))
                rows.append(dict(block_start=start,block_end=end,identity_digest=identity.digest(),
                    native_bundle_publication=native_publication,
                    cache_identity=identity.to_dict(),
                    holder=record.holder_id,endpoint=record.endpoint,sha256=record.payload_sha256,
                    bytes=len(blob),logical_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
                    source_slot=view["slot"],source_generation=view["generation"],**digests))
            except Exception as error:
                errors.append(dict(block_start=start,error=type(error).__name__+": "+str(error)));break
        return dict(published_blocks=len(rows),published_bytes=sum(row["bytes"] for row in rows),
            records=rows,errors=errors,store_kind="process_ram",store_resident_bytes=self.store.nbytes,
            publication_seconds=time.monotonic()-started)

    def _fetch(self,identity,bs,hidden,input_dtype,expected_input,cache_digest):
        started=time.monotonic();deadline=started+self.settings.timeout
        records=self.directory.lookup(identity,timeout_s=self.settings.timeout)
        lookup_seconds=time.monotonic()-started
        rejections=[];received=0;attempts=0
        def read_bytes(nbytes):
            nonlocal received
            received+=nbytes
        for record in records[:4]:
            if record.holder_id not in self.settings.allowed_holders:
                rejections.append("unapproved_holder");continue
            try:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError("remote block phase deadline")
                attempts+=1
                blob=fetch_snapshot(record.endpoint,record.payload_sha256,timeout=remaining,
                    max_bytes=record.payload_nbytes,allowed_endpoints=self.settings.allowed_endpoints,
                    on_read=read_bytes)
                if len(blob)!=record.payload_nbytes or record.object_id!=record.payload_sha256:
                    raise ValueError("remote record/HTTP size or object mismatch")
                tensors=unpack_snapshot(blob,wire_identity(identity,bs),max_bytes=self.settings.max_object)
                if set(tensors)!={"key","value","aux_input","aux_output"}:
                    raise ValueError("remote block needs exact K/V and auxiliary pair")
                for name in ("key","value"):
                    if (tuple(tensors[name].shape)!=(1,identity.num_attention_heads,bs,identity.head_dim)
                            or str(tensors[name].dtype).removeprefix("torch.")!=identity.dtype):
                        raise ValueError("remote native block schema mismatch")
                for name in ("aux_input","aux_output"):
                    if tuple(tensors[name].shape)!=(1,1,bs,hidden) or tensors[name].dtype!=input_dtype:
                        raise ValueError("remote auxiliary schema mismatch")
                if not torch.equal(tensors["aux_input"][:,0],expected_input):
                    raise ValueError("remote actual layer input differs")
                if not all(bool(torch.isfinite(t).all().item()) for t in tensors.values()):
                    raise ValueError("nonfinite remote page payload")
                return tensors,dict(holder=record.holder_id,endpoint=record.endpoint,
                    cache_identity=identity.to_dict(),
                    sha256=record.payload_sha256,identity_digest=identity.digest(),bytes=len(blob),
                    logical_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
                    http_bytes=received,http_attempts=attempts,rejections=rejections,
                    lookup_seconds=lookup_seconds,fetch_validate_seconds=time.monotonic()-started-lookup_seconds,
                    **source_digests(tensors,cache_digest))
            except Exception as error:
                rejections.append(type(error).__name__+": "+str(error))
        return None,dict(http_bytes=received,http_attempts=attempts,rejections=rejections,
                         lookup_seconds=lookup_seconds,fetch_validate_seconds=time.monotonic()-started-lookup_seconds,
                         reason="directory_miss_or_no_valid_payload")

    def recover(self,backend,state,session,config,inputs,end_limit):
        started=time.monotonic()
        start=session.length;rows=[];outputs=[];lookups=received=attempts=0;errors=[]
        if self.settings.mode!="consume":
            return outputs,dict(remote_hit_tokens=0,remote_read_range=[start,start],remote_lookup_count=0,
                remote_http_bytes=0,remote_payload_bytes=0,remote_kv_h2d_bytes=0,remote_records=[],remote_errors=[])
        self._bound(backend,state);bs=state.cache.block_size
        for end in range(start+bs,end_limit+1,bs):
            identity=block_identity(self.settings,backend,state,config["token_ids"],end)
            expected=inputs[:,end-bs:end].detach().to("cpu").contiguous()
            lookups+=1
            try:
                tensors,row=self.io.submit(self._fetch,identity,bs,state.hidden,state.input_dtype,expected,
                    state.cache._digest).result(
                    timeout=self.settings.timeout+1)
            except Exception as error:
                errors.append(type(error).__name__+": "+str(error));break
            received+=row["http_bytes"];attempts+=row["http_attempts"]
            if tensors is None:
                errors.append(row);break
            key=tensors["key"][0].permute(1,0,2);value=tensors["value"][0].permute(1,0,2)
            session.append(key,value)
            actual=session.block(end//bs-1)
            if not torch.equal(actual["key"].cpu(),key) or not torch.equal(actual["value"].cpu(),value):
                raise ValueError("remote import into native physical page failed")
            outputs.append(tensors["aux_output"][:,0].to(state.pool.device))
            row.update(block_start=end-bs,block_end=end,target_slot=actual["slot"],
                       target_generation=actual["generation"])
            rows.append(row)
            state.remote_gpu_sources[(actual["slot"],actual["generation"])]=dict(row,
                origin="remote_http",last_transfer="remote_http_to_gpu")
        count=session.length-start
        return outputs,dict(remote_hit_tokens=count,remote_read_range=[start,start+count],
            remote_lookup_count=lookups,remote_http_attempts=attempts,remote_http_bytes=received,
            remote_payload_bytes=sum(row["logical_bytes"] for row in rows),
            remote_kv_h2d_bytes=count*state.pool.num_heads*state.pool.head_dim*state.pool.k.element_size()*2,
            remote_records=rows,remote_errors=errors,remote_seconds=time.monotonic()-started)

    def close(self):
        if self.http is not None:
            self.http.shutdown();self.http.server_close();self.thread.join(timeout=2)
        self.io.shutdown(wait=True)


def get_remote(backend):
    manager=backend.cache_manager
    plane=getattr(manager,"_dynamo_remote_plane",None)
    if plane is not None:return plane
    if os.environ.get(PREFIX+"MODE","off")=="off":return None
    dht=getattr(manager,"_dynamo_dht",None)
    if dht is None:raise ValueError("ordinary service DHT is not attached to cache manager")
    plane=RemotePlane(dht,Settings.from_env())
    manager._dynamo_remote_plane=plane
    return plane
