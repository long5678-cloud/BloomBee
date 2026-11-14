"""Independent normal rpc_inference method-two provider. No vLLM process.

Loaded only by the opt-in overlay; method-one source remains unmodified.
RemotePrefixSession retains trusted binding/epoch/complete-layer import checks.
"""
import atexit,importlib.util,json,os,threading,time,hashlib
from pathlib import Path
from collections import deque,Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import httpx,torch
from lmcache.sdk.kvcache import connect
from bloombee.kv_cache.network.snapshot_bridge import SnapshotLayout,LMCacheSnapshotBridge
from bloombee.kv_cache.network.native_probe import Geometry
from bloombee.kv_cache.network.history_candidates import HistoryCandidateFilter
from bloombee.kv_cache.network.native_warm import warm_registered,validate_ready
from bloombee.kv_cache.identity import CacheIdentity,canonical_digest,token_ids_digest
from bloombee.server.remote_prefix_session import RemotePrefixSession,export_prefix,_VARIABLE_FIELDS
from bloombee.utils.misc import DUMMY_INT64

# Load the exact existing method-one binding/probe helpers under a separate name.
import bloombee.server
_original=Path(bloombee.server.__path__[0])/'prefix_rpc.py'
_spec=importlib.util.spec_from_file_location('bloombee.server.prefix_rpc_method_one',_original)
_old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_old)
_SERVICES={}

class NativeRuntime(_old.PrefixRuntime):
    def _initialize(self,runner,config_path,runtime_infos):
        self.config=json.loads(Path(config_path).read_text());cfg=self.config
        self.template=CacheIdentity.from_dict(cfg['identity_template'])
        cfg['native_layers']=len(runner.backends)
        # A prefill RPC return is not full generation success. Only the explicit
        # terminal-history strategy entry may initiate a policy warm.
        cfg['candidate_prefetch_enabled']=False
        if cfg.get('known_prefix_tokens') is not None:raise ValueError('history-only candidate round forbids known/future prefix hints')
        if self.template.execution_digest!=canonical_digest(_old.execution_binding()):raise ValueError('execution binding mismatch')
        self.backends=tuple(sorted(runner.backends.values(),key=lambda b:b.block_index))
        if not self.backends or [b.block_index for b in self.backends]!=list(range(self.backends[0].config.num_hidden_layers)):
            raise ValueError('requires complete ordered model provider')
        self.event_lock=threading.Lock();self.prefix_length=int(cfg.get('prefix_length',16))
        if self.prefix_length<=0 or self.prefix_length%16:raise ValueError('requires complete16token chunks')
        for b in self.backends:self._ensure_probe(b)
        self.history=deque(maxlen=cfg.get('history_size',32));self._native_resources=[];self._native_lock=threading.RLock()
        self._warm_ready={};self._warm_pending=set();self._publication_groups=deque(maxlen=cfg.get('publication_groups',256))
        self.http=httpx.Client(trust_env=False,timeout=cfg.get('native_timeout_s',60))
        self.discovery=ThreadPoolExecutor(max_workers=1,thread_name_prefix='native-lmcache-warm');self.discovery_job=None
        self._geometry=None;self.sdk=None
        self._rpc_arrivals=0
        self.candidate_history=deque(maxlen=cfg.get('history_size',32))
        self._strategy_runner=None;self._strategy_receipt_cursor=0;self._strategy_receipts_seen=set()
        if cfg.get('initial_warm_receipt'):
            receipt=json.loads(Path(cfg['initial_warm_receipt']).read_text())
            self._warm_ready[receipt['cache_salt']]=receipt
        self.emit('native_runtime_ready',layers=len(self.backends),stage='unregistered_until_first_exact_layout',storage_contract=cfg.get('storage_contract'))
        atexit.register(self.close)

    def layout(self,tokens):
        return SnapshotLayout(self.identities(tokens,self.prefix_length),tokens[:self.prefix_length],max_bytes=self.config.get('capacity_bytes',256*1024**2))

    def ensure_sdk(self,layout):
        with self._native_lock:
            if self.sdk is not None:
                if layout.model_name!=self._model_name:raise ValueError('static native model namespace changed')
                return
            server=self.config['lmcache_target'];heads=layout.heads
            geometry=Geometry(server,layout.model_name,len(self.backends),heads,layout.dim,layout.dtype)
            self._native_resources.append(geometry)
            sdk=connect(server['url'],server['http_url'],layout.model_name,timeout=self.config.get('native_timeout_s',60))
            self._native_resources.append(sdk);self.sdk=sdk;self._model_name=layout.model_name
            self.emit('native_geometry_registered',model_name=layout.model_name,layers=len(self.backends),physical_heads=heads,head_dim=layout.dim,dtype=str(layout.dtype))

    def warm(self,tokens,reason):
        layout=self.layout(tokens);self.ensure_sdk(layout)
        receipt=warm_registered(self.http,self.config,layout,reason=reason)
        with self._native_lock:
            if len(self._warm_ready)>=self.config.get('publication_groups',256):self._warm_ready.clear()
            self._warm_ready[layout.cache_salt]=receipt
        self.emit('native_warm_complete',cache_salt=layout.cache_salt,receipt=receipt)

    def filter_history_candidates(self, history_snapshot=None):
        """Public explicit metadata-only entry; callable without any warm/rank."""
        frozen=tuple(self.candidate_history) if history_snapshot is None else tuple(history_snapshot)
        report=HistoryCandidateFilter(self.config,client=self.http).filter(frozen)
        self.emit('history_candidate_filter',report=report)
        return report

    def rank_prefetch_candidates(self,completed_history,boundary,decision_id,causal_cutoff_ns,*,warm=False):
        """Trusted terminal-ledger entry, never infer success from prefill/hit."""
        from bloombee.kv_cache.network.strategy_runner import StrategyRunner
        if self._strategy_runner is None:self._strategy_runner=StrategyRunner(self.config,self.emit)
        return self._strategy_runner.decide(completed_history,boundary,decision_id,causal_cutoff_ns,warm=warm)

    def warm_selected_strategy(self,context,selected_result,completed_history):
        """Trusted external dispatch of one already-ranked context/choice."""
        from bloombee.kv_cache.network.strategy_runner import StrategyRunner
        if self._strategy_runner is None:self._strategy_runner=StrategyRunner(self.config,self.emit)
        return self._strategy_runner.warm_selected(context,selected_result,completed_history)

    def _discover(self):
        self.emit('strategy_abstain',reason='prefill_is_not_full_successful_demand_use_terminal_history_entry')

    def _load_strategy_receipts(self):
        path=self.config.get('strategy_warm_receipts_path')
        if not path or not Path(path).exists():return
        with Path(path).open() as stream:
            stream.seek(self._strategy_receipt_cursor)
            while True:
                start=stream.tell();line=stream.readline(2*1024**2)
                if not line or not line.endswith('\n'):break
                self._strategy_receipt_cursor=stream.tell()
                try:
                    receipt=json.loads(line);sha=receipt['receipt_sha256']
                    if sha not in self._strategy_receipts_seen:
                        self._strategy_receipts_seen.add(sha)
                        self._warm_ready[receipt['cache_salt']]=receipt
                except (KeyError,ValueError) as error:self.emit('strategy_receipt_invalid',offset=start,error=str(error))

    def publish(self,infos,tokens):
        if not self.config.get('native_publish_enabled',True):
            self.emit('native_publication_disabled',role=self.config.get('native_role','consumer'));return
        if len(tokens)<=self.prefix_length:return
        layout=self.layout(tokens);self.ensure_sdk(layout)
        if layout.cache_salt in self._publication_groups:return
        start=time.monotonic_ns();blobs=[export_prefix(b,info,i,self.prefix_length) for b,info,i in zip(self.backends,infos,layout.identities)]
        digests=[hashlib.sha256(b).hexdigest() for b in blobs]
        with self._native_lock:ok=LMCacheSnapshotBridge(layout,self.sdk).store(blobs,digests)
        if not ok:
            # Official False can mean zero objects reserved because already present.
            # It can also mean failure: distinguish by exact native readback.
            with self._native_lock:existing=self.sdk.retrieve(layout.tokens,cache_salt=layout.cache_salt)
            expected=layout.to_native(blobs,digests)
            if existing is None or not torch.equal(existing,expected):raise RuntimeError('native SDK False and exact existing data not verified')
            self.emit('native_store_already_present_verified',cache_salt=layout.cache_salt,readback_bytes=existing.numel()*existing.element_size())
        self._publication_groups.append(layout.cache_salt)
        self.emit('native_store_complete',cache_salt=layout.cache_salt,layers=len(blobs),snapshot_bytes=sum(map(len,blobs)),native_tensor_bytes=2*len(blobs)*self.prefix_length*layout.heads*layout.dim*torch.empty((),dtype=layout.dtype).element_size(),started_ns=start,completed_ns=time.monotonic_ns())

    def run(self,runner,hidden,hypo,infos,prompts):
        envelope=infos[0].remote_prefix
        if any(i.remote_prefix!=envelope for i in infos) or tuple(runner.backends[i.uid] for i in infos)!=self.backends:raise ValueError('mixed/full-provider envelope mismatch')
        tokens,blocks=envelope['token_ids'],envelope['blocks'];request_id=envelope['request_id'];mode=envelope.get('mode','prefetch')
        if (not isinstance(tokens,list) or any(type(t)is not int or t<0 for t in tokens) or len(tokens)!=hidden.shape[1] or hidden.shape[0]!=1 or any(i.prefix_length or i.active_adapter for i in infos) or mode not in ('cold','demand','prefetch') or not isinstance(blocks,list) or any(type(b)not in(int,str) for b in blocks)):raise ValueError('unsupported initial native RPC envelope')
        clean=tuple(replace(i,remote_prefix=None) for i in infos);epoch=os.urandom(16).hex()
        for b in self.backends:b._remote_prefix_epoch=epoch
        self.emit('rpc_native_enter',request_id=request_id,mode=mode,token_count=len(tokens),rpc_session_id=envelope.get('rpc_session_id'))
        origin='cold';blobs=None
        arrival_ns=time.monotonic_ns();self._rpc_arrivals+=1
        try:
            if mode!='cold' and len(tokens)>self.prefix_length:
                layout=self.layout(tokens);self.ensure_sdk(layout)
                self._load_strategy_receipts()
                with self._native_lock:
                    warm_receipt=self._warm_ready.pop(layout.cache_salt,None)
                    readiness=validate_ready(self.http,self.config,layout,warm_receipt,arrival_ns)
                    ready=readiness['ready']
                    claim_started=time.monotonic_ns()
                    blobs=LMCacheSnapshotBridge(layout,self.sdk).retrieve()
                if blobs is not None:origin='prefetch' if mode=='prefetch' and ready else 'demand'
                self.emit('native_claim',request_id=request_id,actual_cache_salt=layout.cache_salt,readiness=readiness,strategy_link=None if warm_receipt is None else warm_receipt.get('strategy_link'),rpc_arrival_index=self._rpc_arrivals,warm_receipt_sha256=None if warm_receipt is None else warm_receipt.get('receipt_sha256'),warm_complete_before_claim=ready,hit=blobs is not None,origin=origin,sdk_started_ns=claim_started,sdk_completed_ns=time.monotonic_ns(),snapshot_bytes=0 if blobs is None else sum(map(len,blobs)),network_accounting='LMCache native trace/backend logs required; importer network_bytes counts only snapshot stage')
            if blobs is None:
                output=runner(hidden,hypo,clean,*prompts);self.emit('native_cold',request_id=request_id,computed_tokens=len(tokens))
            else:
                payloads={hashlib.sha256(b).hexdigest():b for b in blobs};records={}
                for i,b in zip(layout.identities,blobs):
                    digest=hashlib.sha256(b).hexdigest();records[CacheIdentity.from_dict(i).digest()]=dict(identity_digest=CacheIdentity.from_dict(i).digest(),holder_id='native-lmcache-sdk',endpoint='http://127.0.0.1:1',object_id=digest,payload_sha256=digest,payload_nbytes=len(b),expires_at=time.time()+3600,record_version=1)
                def lookup(i,**kw):return records.get(i.digest())
                def fetch(endpoint,digest,**kw):return payloads[digest]
                fetch.source_kind='lmcache_sdk_native'
                start_counts=[getattr(b,'_remote_prefix_event_counter',0) for b in self.backends]
                session=RemotePrefixSession(self.backends,clean,layout.identities,self.prefix_length,lookup,token_ids=tokens,allowed_endpoints=('http://127.0.0.1:1',),timeout_s=self.config.get('import_timeout_s',60),max_total_bytes=self.config.get('capacity_bytes',256*1024**2),fetcher=fetch)
                if not session.prepare():
                    self.emit('import_failure_requires_fresh_session',request_id=request_id,metrics=session.metrics)
                    raise RuntimeError('native import failed; retry whole client session cold')
                state=session.run(hidden,hypo)
                receipt=dict(schema=1,rpc_session_id=envelope.get('rpc_session_id'),request_id=request_id,native_session_id=session.receipt.session_id,input_tokens=len(tokens),skipped_tokens=self.prefix_length,token_ids_digest=token_ids_digest(tokens),layer_start=0,layer_end=len(self.backends),origin=origin)
                output=(state,DUMMY_INT64,dict(pool_inner_ms=0.,remote_prefix_receipt=receipt))
                self.emit('native_reuse',request_id=request_id,origin=origin,metrics=session.metrics,computed_tokens=int(state.shape[1]),skipped_tokens=self.prefix_length,native_backend_steps=[e for b,count in zip(self.backends,start_counts) for e in b._remote_prefix_step_events if e['sequence']>count])
            try:self.publish(clean,tokens)
            except Exception as e:self.emit('publication_failure',request_id=request_id,error_type=type(e).__name__,error=str(e))
            if len(tokens)>self.prefix_length and len(blocks)>=self.prefix_length//16:
                self.history.append((tuple(tokens),tuple(blocks)))
                history_event=dict(consumer_instance_id=self.config.get('candidate_consumer_id',self.config['lmcache_target']['instance_id']),processed_by_instance_id=self.config['lmcache_target']['instance_id'],request_id=request_id,outcome='processed_success',history_source='provider_successful_rpc',visibility_scope='consumer_own_completed_requests',processed_order=self._rpc_arrivals,token_ids=list(tokens),blocks=list(blocks),identities=self.identities(tokens,self.prefix_length))
                self.candidate_history.append(history_event)
                self.emit('processed_history_observed',history=history_event)
            else:self.emit('history_skipped_short_request',request_id=request_id,token_count=len(tokens))
            if mode=='prefetch' and self.config.get('candidate_prefetch_enabled',True) and (self.discovery_job is None or self.discovery_job.done()):
                if self.discovery_job is not None and self.discovery_job.exception() is not None:self.emit('discovery_failure',error=str(self.discovery_job.exception()))
                self.discovery_job=self.discovery.submit(self._discover)
            self.emit('rpc_native_exit',request_id=request_id,origin=origin);return output
        finally:
            for b in self.backends:b._remote_prefix_epoch=None

    def close(self):
        if self._closed:return
        self._closed=True;errors=[]
        if getattr(self,'discovery',None) is not None:self.discovery.shutdown(wait=True)
        if getattr(self,'_strategy_runner',None) is not None:self._strategy_runner.close()
        for resource in reversed(getattr(self,'_native_resources',[])):
            try:resource.close()
            except Exception as e:errors.append(str(e))
        if hasattr(self,'http'):self.http.close()
        self.cleanup_errors=errors

def run_prefix_rpc(runner,hidden,hypo,infos,prompts):
    path=os.environ.get('BLOOMBEE_LMCACHE_CONFIG')
    if not path:raise RuntimeError('method two config missing')
    key=(os.getpid(),id(runner))
    if key not in _SERVICES:_SERVICES[key]=NativeRuntime(runner,path,infos)
    return _SERVICES[key].run(runner,hidden,hypo,infos,prompts)
