"""Real LMCache v0.5.7 SDK/coordinator bridge for ordinary page bundles.

Codec is owned by Dynamo. No model KV is fabricated here, no GPU/SSH calls,
no alternate cache API. Native storage64 is NOT model layer64.
"""
import importlib.util,json,os,threading,time
from pathlib import Path
from bloombee.kv_cache.network.bundle_contract import digest,sha,value_sha,layout_metadata,make_manifest,validate_manifest


def codec_layout(identities,tokens,*,max_bytes=256*1024**2):
    from bloombee.kv_cache.page_bundle import PageBundleLayout
    return PageBundleLayout(identities,tokens,block_size=16,max_bytes=max_bytes)

def _identity_dict(value):return value.to_dict() if hasattr(value,'to_dict') else dict(value)

class StorageGeometry:
    """Official CPU registration with one16-token dummy block, 64 storage slots.
    VLLM is only the KV-layout enum; no vLLM model/process is used.
    """
    def __init__(self,endpoint,layout):
        import torch,zmq,uuid
        from lmcache.utils import EngineType
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo
        from lmcache.v1.multiprocess.transfer_context import create_transfer_context
        from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
        self.buffers={f'layer.{i}':torch.zeros((1,2,layout.heads,16,layout.dim),dtype=layout.dtype) for i in range(layout.storage_layers)}
        self.buffer_bytes=sum(t.numel()*t.element_size() for t in self.buffers.values())
        self.zmq=zmq.Context();self.zmq.setsockopt(zmq.LINGER,0);self.client=RequestClientFactory.create(endpoint['url'],context=self.zmq)
        self.transfer=create_transfer_context(self.buffers,mode='lmcache_driven',instance_id=uuid.uuid4().int%(2**30),req_client=self.client)
        self.transfer.register(self.buffers,layout.model_name,1,1,120,layout_hints=dict(kv_layout='HND',num_kv_heads=layout.heads,tokens_per_block=16,head_dim=layout.dim),engine_group_infos=[EngineGroupInfo(engine_group_id=0,layer_indices=tuple(range(layout.storage_layers)),tokens_per_block=16)],engine_type=EngineType.VLLM)
    def close(self):
        self.transfer.close()
        if hasattr(self.client,'close'):self.client.close()
        self.zmq.destroy(linger=0);self.buffers.clear()

class NativeEndpoint:
    def __init__(self,endpoint,timeout=120):self.endpoint=endpoint;self.timeout=timeout;self.contexts={};self.lock=threading.RLock()
    def sdk(self,layout):
        with self.lock:
            layout_metadata(layout)
            if layout.model_name not in self.contexts:
                from lmcache.sdk.kvcache import connect
                if self.contexts:raise ValueError('one frozen static page model namespace per endpoint owner')
                geometry=StorageGeometry(self.endpoint,layout)
                try:sdk=connect(self.endpoint['url'],self.endpoint['http_url'],layout.model_name,timeout=self.timeout)
                except Exception:geometry.close();raise
                self.contexts[layout.model_name]=(geometry,sdk)
            return self.contexts[layout.model_name][1]
    def registration_evidence(self,layout):
        import httpx
        with httpx.Client(trust_env=False,timeout=self.timeout) as client:
            r=client.get(self.endpoint['http_url']+'/status');r.raise_for_status();metadata=r.json().get('cache_context_meta',{})
        matches=[{k:entry[k] for k in ('model_name','world_size','kv_cache_layout') if k in entry} for entry in metadata.values() if entry.get('model_name')==layout.model_name]
        if not matches or any(entry.get('kv_cache_layout',{}).get('num_layers')!=layout.storage_layers for entry in matches):raise ValueError('native registered storage geometry is not64')
        sdk_buffers=getattr(self.contexts[layout.model_name][1],'_kv_caches',{})
        workspace=dict(geometry_dummy_bytes=self.contexts[layout.model_name][0].buffer_bytes,sdk_dummy_bytes=sum(t.numel()*t.element_size() for t in sdk_buffers.values()),not_history_capacity=True)
        return dict(registration_workspace=workspace,endpoint=self.endpoint,model_layers=layout.logical_layers,storage_layers=layout.storage_layers,registered_layouts=matches)
    def close(self):
        for geometry,sdk in self.contexts.values():sdk.close();geometry.close()
        self.contexts.clear()

class PageBundlePublisher:
    def __init__(self,config,emit=None):
        self.config=config;self.emit=emit or (lambda event,**data:None);self.native=NativeEndpoint(config['native_endpoint'],config.get('native_timeout_s',120));self.manifests={};self.lock=threading.RLock()
    def publish_bundle(self,layout,blobs_by_block,sha_by_block,source_attempt_id):
        with self.lock:
            layout_metadata(layout);key=(layout.model_name,layout.cache_salt)
            if key in self.manifests:
                original=self.manifests[key]
                observed=[[sha(blob) for blob in row] for row in blobs_by_block]
                if observed!=sha_by_block:raise ValueError('observed source hashes do not match current source bytes')
                self.emit('source_immutable_publication_reused',source_attempt_id=source_attempt_id,object_id=original['object_id'],observed_shas=observed,original_shas=original['sha_by_block'],wire_values_differ=observed!=original['sha_by_block'],native_store_called=False)
                reused=dict(original,publication_reused=True,observed_shas_differ=observed!=original['sha_by_block'],observed_sha_by_block=observed,observation_attempt_id=source_attempt_id)
                if self.config.get('published_manifest_path'):
                    with Path(self.config['published_manifest_path']).open('a') as stream:stream.write(json.dumps(reused,sort_keys=True)+'\n');stream.flush()
                return reused
            started=time.monotonic_ns();tensor=layout.encode(blobs_by_block,sha_by_block)
            manifest=make_manifest(layout,blobs_by_block,sha_by_block,tensor,source_attempt_id=source_attempt_id,run_id=self.config['run_id'],source_instance=self.config['native_endpoint']['instance_id'])
            sdk=self.native.sdk(layout);stored=bool(sdk.store(tensor,layout.tokens,cache_salt=layout.cache_salt))
            # True may also mean no newly reserved objects. Never assume it
            # overwrote a previous value under this identity-only salt.
            actual=sdk.retrieve(layout.tokens,cache_salt=layout.cache_salt)
            if actual is None or tuple(actual.shape)!=layout.shape or value_sha(actual)!=manifest['native_value_sha256']:raise ValueError('source native immutable value differs; no publication acknowledgment')
            manifest.update(source_store_return=stored,source_readback_exact=True,native_registration=self.native.registration_evidence(layout),source_publish_started_ns=started,source_publish_completed_ns=time.monotonic_ns())
            self.manifests[key]=manifest
            path=self.config.get('published_manifest_path')
            if path:
                with Path(path).open('a') as stream:stream.write(json.dumps(manifest,sort_keys=True)+'\n');stream.flush()
            self.emit('source_bundle_published',manifest=manifest,model_layers=32,native_storage_layers=64)
            return manifest
    def close(self):self.native.close()

class SourceCollector:
    """Product hook: original immutable wire bytes, not a second source GET."""
    def __init__(self,config,emit=None):
        self.config=config;self.publisher=PageBundlePublisher(config,emit);self.pending={};self.lock=threading.RLock();self.emit=emit or (lambda event,**data:None)
        self.max_pending_bytes=int(config.get('source_pending_max_bytes',64*1024**2));self.pending_bytes=0
    def observe_page(self,attempt_id,identity,blob,record,tokens):
        identity=_identity_dict(identity);tokens=list(tokens)
        if identity['prefix_length']!=16:return dict(status='ignored_nonselected_page_range',prefix_length=identity['prefix_length'])
        if identity['layer_end']!=identity['layer_start']+1 or not 0<=identity['layer_start']<32:raise ValueError('requires original32 logical model layer identities')
        if not isinstance(blob,bytes):raise TypeError('source hook requires original immutable RAM bytes')
        expected=record.get('payload_sha256') if isinstance(record,dict) else record.payload_sha256
        if sha(blob)!=expected:raise ValueError('source product publication record differs from bytes')
        base={k:v for k,v in identity.items() if k not in ('layer_start','layer_end')};group=(attempt_id,digest(base))
        with self.lock:
            row=self.pending.setdefault(group,dict(identities={},blobs={},shas={},tokens=tokens[:16]))
            if row['tokens']!=tokens[:16]:raise ValueError('attempt prefix token identity changed')
            layer=identity['layer_start']
            if layer in row['blobs']:
                if row['shas'][layer]!=expected:raise ValueError('same live attempt/layer changed immutable bytes')
                return dict(status='duplicate_layer_observation')
            if self.pending_bytes+len(blob)>self.max_pending_bytes:raise ValueError('source aggregation staging byte budget exhausted')
            row['identities'][layer]=identity;row['blobs'][layer]=blob;row['shas'][layer]=expected;self.pending_bytes+=len(blob)
            if len(row['blobs'])!=32:return dict(status='collecting',model_layers_received=len(row['blobs']))
            try:
                layout=codec_layout([[row['identities'][i] for i in range(32)]],row['tokens'])
                manifest=self.publisher.publish_bundle(layout,[[row['blobs'][i] for i in range(32)]],[[row['shas'][i] for i in range(32)]],attempt_id)
                return dict(status='published',manifest=manifest)
            finally:
                self.pending_bytes-=sum(map(len,row['blobs'].values()));self.pending.pop(group,None)
    def abandon(self,attempt_id):
        with self.lock:
            for key in [k for k in self.pending if k[0]==attempt_id]:self.pending_bytes-=sum(map(len,self.pending.pop(key)['blobs'].values()))
    def close(self):self.publisher.close();self.pending.clear();self.pending_bytes=0

_COLLECTORS={}
def get_source_collector(config_path=None):
    path=str(config_path or os.environ['BLOOMBEE_NATIVE_PAGE_CONFIG'])
    key=(os.getpid(),path)
    if key not in _COLLECTORS:_COLLECTORS[key]=SourceCollector(json.loads(Path(path).read_text()))
    return _COLLECTORS[key]


def transfer_evidence(before,after_warm,after_read,transfer_id):
    samples=(before,after_warm,after_read)
    known=all(isinstance(s,dict) and s.get('known') is True and s.get('exclusive') is True and type(s.get('gets'))is int and type(s.get('value_bytes'))is int and s.get('scope_id') for s in samples)
    known=known and len({s['scope_id'] for s in samples})==1
    receipt=dict(transfer_id=transfer_id,phase='native_warm_and_sdk_readback',snapshots=list(samples),evidence_scope='external exact-key physical counters; not tensor size or local HTTP bytes')
    if known:
        gets=after_read['gets']-before['gets'];nbytes=after_read['value_bytes']-before['value_bytes'];extra=after_read['gets']-after_warm['gets']
        if min(gets,nbytes,extra)<0:known=False
    if not known:receipt.update(classification='unknown');return dict(network_bytes=None,network_attempts=None,network_receipt=receipt)
    receipt.update(native_warm_gets=after_warm['gets']-before['gets'],readback_extra_gets=extra,classification='zero_additional_native_GETs_proven' if extra==0 else 'late_or_additional_native_GETs')
    return dict(network_bytes=nbytes,network_attempts=gets,network_receipt=receipt)

class NativePageBundleBridge:
    def __init__(self,config,bank,*,emit=None,transfer_probe=None,release_fn=None):
        self.config=config;self.bank=bank;self.emit=emit or (lambda event,**data:None);self.probe=transfer_probe;self.release_fn=release_fn
        self.native=NativeEndpoint(config['native_endpoint'],config.get('native_timeout_s',120));self.locks={};self.lock=threading.Lock();self.pending_retirements={};self.retirement_lock=threading.Lock()
    def key_lock(self,object_id):
        with self.lock:return self.locks.setdefault(object_id,threading.RLock())
    def warm_to_bank(self,layout,manifest,decision_link):
        import httpx
        from bloombee.kv_cache.network.native_warm import warm_registered
        validate_manifest(manifest,layout)
        if manifest.get('source_readback_exact') is not True or manifest.get('native_registration',{}).get('storage_layers')!=64:raise ValueError('source native store/readback64 proof missing')
        if manifest['run_id']!=self.config.get('source_run_id',self.config['run_id']) or manifest['source_instance']!=self.config['source_instance']:raise ValueError('source bundle belongs to another arm/source')
        if any(key in decision_link for key in ('attempt_id','next_request','next_token_ids','test_path')):raise ValueError('prediction cannot bind future attempt/input')
        object_id=manifest['object_id']
        with self.key_lock(digest(dict(model_name=layout.model_name,cache_salt=layout.cache_salt))):
            if self.bank.has_object(object_id):return dict(status='already_in_managed_bank',object_id=object_id,official_warm_posts=0)
            with self.retirement_lock:
                if digest(dict(model_name=layout.model_name,cache_salt=layout.cache_salt)) in self.pending_retirements:return dict(status='native_retirement_pending_not_ready',object_id=object_id,official_warm_posts=0)
            native_ready=None;encoded_key=None;before=None;after_warm=None;after_read=None
            try:
                sdk=self.native.sdk(layout);cfg=dict(self.config,lmcache_target=self.config['native_endpoint'])
                probe=lambda phase:self.probe(phase,manifest) if self.probe else None
                before=probe('before_native_warm')
                with httpx.Client(trust_env=False,timeout=cfg.get('native_timeout_s',120)) as client:
                    native_ready=warm_registered(client,cfg,layout,require_empty=False,reason='joint_page_bundle')
                    query=dict(model_name=layout.model_name,world_size=1,token_ids=layout.tokens,cache_salt=layout.cache_salt)
                    response=client.post(cfg['coordinator_url']+'/directory/lookup',json=query);response.raise_for_status();keys=response.json()
                if keys.get('chunks')!=1 or len(keys.get('results',[]))!=1:raise ValueError('one exact official native chunk required')
                encoded_key=keys['results'][0]['key']
                if encoded_key['model_name']!=layout.model_name or encoded_key.get('cache_salt')!=layout.cache_salt:raise ValueError('official warm key differs from codec namespace')
                after_warm=probe('after_native_warm');self.emit('native_warm_ready',object_id=object_id,receipt=native_ready,decision_link=decision_link)
                read_started_ns=time.monotonic_ns()
                tensor=sdk.retrieve(layout.tokens,cache_salt=layout.cache_salt)
                read_completed_ns=time.monotonic_ns()
                after_read=probe('after_sdk_readback')
                if tensor is None or value_sha(tensor)!=manifest['native_value_sha256']:raise ValueError('native readback value SHA differs from full source bundle')
                blobs=layout.decode(tensor,manifest['sha_by_block'])
                transfer=transfer_evidence(before,after_warm,after_read,digest(dict(object_id=object_id,decision_link=decision_link,native_request_id=native_ready['request_id'])))
                transfer['network_receipt'].update(native_warm_ready=dict(request_id=native_ready['request_id'],status=native_ready['status'],completed_ns=native_ready['completed_ns'],clock_domain=native_ready['clock_domain']),sdk_read_started_ns=read_started_ns,sdk_read_completed_ns=read_completed_ns,native_value_sha256=manifest['native_value_sha256'],decode_completed_ns=time.monotonic_ns(),native_registration=self.native.registration_evidence(layout))
                # The bank is the only extra bounded transfer owner, not a hidden
                # persistent native/GPU/CPU history cache.
                try:ticket=self.bank.offer(manifest,blobs,transfer,decision_link,native_warm_receipt=native_ready,expected_key=encoded_key)
                except Exception as error:
                    abandoned=dict(manifest=manifest,native_warm_receipt=native_ready,expected_key=encoded_key,ticket_id='unadmitted_'+digest(decision_link))
                    retirement=self.retire_staging(abandoned,reason='bank_admission_failed')
                    self.emit('bundle_bank_admission_failed',object_id=object_id,error_type=type(error).__name__,error=str(error),transfer=transfer,retirement=retirement,page_ready=False,bank=self.bank.accounting())
                    return dict(status='bank_admission_failed',object_id=object_id,error=str(error),transfer=transfer,release_result=retirement,page_ready=False,official_warm_posts=1)
                self.emit('bundle_decoded_bank_ready',object_id=object_id,ticket_id=ticket['ticket_id'],native_value_sha256=manifest['native_value_sha256'],transfer=transfer,bank=self.bank.accounting(),page_ready=False)
                return dict(status='decoded_bank_ready_not_page_ready',ticket_id=ticket['ticket_id'],object_id=object_id,transfer=transfer,official_warm_posts=1)
            except Exception as error:
                failed=dict(ticket_id='failed_'+digest(decision_link),manifest=manifest,native_warm_receipt=native_ready,expected_key=encoded_key)
                evidence=transfer_evidence(before,after_warm,after_read,digest(dict(object_id=object_id,decision_link=decision_link,status='failed')))
                failed['header']=dict(transfer=evidence,decision_link=decision_link)
                # Missing warm/key proof must remain explicit pending, never an
                # invented DELETE key or a claimed zero-byte failure.
                if native_ready is not None and encoded_key is not None:release=self.retire_staging(failed,reason='warm_read_or_decode_failed')
                else:
                    self.defer_retirement(failed,'warm_or_registration_failed_unknown_staging')
                    release=dict(status='pending_unverified_staging',reason='complete_warm_key_receipt_unavailable')
                self.emit('joint_warm_failed',object_id=object_id,error_type=type(error).__name__,error=str(error),transfer=evidence,release_result=release,page_ready=False)
                return dict(status='warm_failed_not_ready',object_id=object_id,error_type=type(error).__name__,error=str(error),transfer=evidence,release_result=release,page_ready=False,official_warm_posts=1 if native_ready else None)
    def defer_retirement(self,ticket,reason):
        # Called under bank.lock: metadata only, no network and no per-key lock.
        metadata={k:ticket.get(k) for k in ('ticket_id','manifest','native_warm_receipt','expected_key','header')}
        key_id=digest({k:ticket['manifest']['layout'][k] for k in ('model_name','cache_salt')})
        with self.retirement_lock:self.pending_retirements.setdefault(key_id,dict(ticket=metadata,reason=reason,last_result=ticket.get('release_result')))
    def retire_staging(self,ticket,*,reason):
        import httpx
        key_id=digest({k:ticket['manifest']['layout'][k] for k in ('model_name','cache_salt')})
        with self.key_lock(key_id):
            try:
                if self.release_fn is None:raise ValueError('release_helper_not_configured')
                from bloombee.kv_cache.page_bundle import PageBundleLayout
                info=ticket['manifest']['layout'];layout=PageBundleLayout(info['identities_by_block'],info['tokens'],block_size=info['block_size']);warm=ticket['native_warm_receipt']
                with httpx.Client(trust_env=False,timeout=self.config.get('native_timeout_s',120)) as client:
                    release=self.release_fn(client,layout=layout,target=self.config['native_endpoint'],coordinator_url=self.config['coordinator_url'],source_instance=ticket['manifest']['source_instance'],expected_key=ticket['expected_key'],warm_registration=warm['after']['registration'],target_incarnation=warm['after']['chunks'][0]['incarnation'])
            except Exception as error:release=dict(status='release_error',error_type=type(error).__name__,error=str(error))
            with self.retirement_lock:
                if release.get('status')=='released':self.pending_retirements.pop(key_id,None)
                else:
                    metadata={k:ticket.get(k) for k in ('ticket_id','manifest','native_warm_receipt','expected_key','header')}
                    self.pending_retirements[key_id]=dict(ticket=metadata,reason=reason,last_result=release)
            self.emit('native_staging_release',ticket_id=ticket['ticket_id'],reason=reason,release_result=release,pending_retirements=len(self.pending_retirements))
            return release
    def retirement_status(self):
        with self.retirement_lock:
            return dict(pending_count=len(self.pending_retirements),pending_queue_empty=not bool(self.pending_retirements),pending=[dict(key_id=key,ticket_id=item['ticket']['ticket_id'],object_id=item['ticket']['manifest']['object_id'],reason=item['reason'],last_result=item['last_result'],transfer=(item['ticket'].get('header') or {}).get('transfer')) for key,item in self.pending_retirements.items()])
    def acknowledge(self,ticket,attempt_id,accepted):
        # prepared/failed preparation is not an active page consumer; exact
        # force=False retirement is safe for either acceptance or rejection.
        return self.retire_staging(ticket,reason='page_prepared_accept' if accepted is True else 'page_preparation_rejected')
    def close(self):
        # One finite final retry, never force, never erase the failure receipt.
        with self.retirement_lock:pending=list(self.pending_retirements.values())
        for item in pending:self.retire_staging(item['ticket'],reason='finite_close_drain_'+item['reason'])
        self.native.close()
        result=self.retirement_status();self.emit('native_staging_close_summary',**result);return result
