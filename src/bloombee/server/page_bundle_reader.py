"""Bounded process-local ready-bundle consumption. No SDK or transport side effects.

The integration owner creates one lease per outer ordinary request, resolves the
same lease for every real layer, and closes it after that request completes.
Never install a per-request method on the shared RemotePlane.
"""
from dataclasses import dataclass
import base64
import hashlib
import math
import os
import threading
import time
import torch

from bloombee.kv_cache.identity import CacheIdentity
from bloombee.kv_cache.transport import unpack_snapshot
from bloombee.server.dynamo_remote import RemotePlane, wire_identity, source_digests


@dataclass(frozen=True)
class ReadySelection:
    object_id: str
    selection_id: str
    layout: object
    identities_by_block: object
    sha_by_block: object
    expires_at: float  # consumer process monotonic clock only
    is_valid: object  # SDK owner checks this exact selected object is still ready


@dataclass(frozen=True)
class RetrievedBundle:
    native: object
    object_id: str
    network_bytes: object = None
    network_attempts: object = None
    network_receipt: object = None  # external evidence, not inferred from tensor.numel


@dataclass(frozen=True)
class DecodedBundle:
    blobs_by_block: object
    object_id: str
    network_bytes: object = None
    network_attempts: object = None
    network_receipt: object = None


class BundleRetrievalError(RuntimeError):
    def __init__(self,message,*,network_bytes=None,network_attempts=None,network_receipt=None):
        super().__init__(message)
        self.accounting=RetrievedBundle(None,'',network_bytes,network_attempts,network_receipt)


class BundleLease:
    def __init__(self,selection,retrieve,*,max_bytes=2*1024**3,clock=time.monotonic):
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('finite bundle byte bound required')
        self.pid=os.getpid();self.selection=selection;self.retrieve=retrieve
        self.max_bytes=max_bytes;self.clock=clock;self.lock=threading.RLock()
        self.attempted=False;self.closed=False;self.error=None;self.blobs={};self.charged=False
        self.accounting=None;self.decoded_bytes=0;self.native_bytes=0
        self.sdk_retrieve_calls=None
        if not selection.object_id or not selection.selection_id or not callable(selection.is_valid):
            raise ValueError('exact selected ready object and validity check required')
        if (len(selection.object_id)>256 or len(selection.selection_id)>256
                or not math.isfinite(selection.expires_at)):
            raise ValueError('bounded selection identity and finite local expiry required')
        self.index={}
        if len(selection.identities_by_block)!=len(selection.sha_by_block):raise ValueError('block metadata mismatch')
        if len(selection.identities_by_block)>4096 or sum(len(r) for r in selection.identities_by_block)>16384:
            raise ValueError('bundle metadata entry bound exceeded')
        for identities,shas in zip(selection.identities_by_block,selection.sha_by_block):
            if len(identities)!=len(shas):raise ValueError('layer metadata mismatch')
            for identity,sha in zip(identities,shas):
                identity=identity if isinstance(identity,CacheIdentity) else CacheIdentity.from_dict(identity)
                if identity.digest() in self.index:raise ValueError('duplicate page identity')
                if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError('original snapshot SHA required')
                self.index[identity.digest()]=(identity,sha)
        if not self.index:raise ValueError('empty bundle')

    def __getstate__(self):raise TypeError('BundleLease is process-local; send metadata, not Python objects')

    def _check(self):
        if self.pid!=os.getpid():raise RuntimeError('bundle lease belongs to a different process')
        if self.closed:raise RuntimeError('bundle lease closed')
        if self.clock()>=self.selection.expires_at or not self.selection.is_valid():
            raise ValueError('selected ready object expired or invalidated')

    @staticmethod
    def _validate_accounting(value):
        for item in (value.network_bytes,value.network_attempts):
            if item is not None and (type(item) is not int or item<0):raise ValueError('invalid network accounting')
        if value.network_bytes is not None and not value.network_receipt:
            raise ValueError('known physical bytes require external receipt; tensor size is not evidence')

    def _open(self):
        if self.attempted:return
        self.attempted=True
        try:
            value=self.retrieve(self.selection)
            if not isinstance(value,(RetrievedBundle,DecodedBundle)):raise TypeError('provider must return RetrievedBundle or DecodedBundle')
            self.accounting=RetrievedBundle(None,value.object_id,value.network_bytes,value.network_attempts,value.network_receipt)
            self._validate_accounting(value)
            if value.object_id!=self.selection.object_id:raise ValueError('SDK returned a different selected object')
            if isinstance(value,DecodedBundle):
                self.sdk_retrieve_calls=0
                decoded=value.blobs_by_block
            else:
                self.sdk_retrieve_calls=1
                native=value.native
                if not isinstance(native,torch.Tensor) or native.device.type!='cpu':raise ValueError('native CPU tensor required')
                self.native_bytes=native.numel()*native.element_size()
                if self.native_bytes>self.max_bytes:raise ValueError('native bundle byte bound exceeded')
                decoded=self.selection.layout.decode(native,self.selection.sha_by_block)
            if len(decoded)!=len(self.selection.identities_by_block):raise ValueError('decoded block count mismatch')
            staged={};total=0
            for blobs,identities in zip(decoded,self.selection.identities_by_block):
                if len(blobs)!=len(identities):raise ValueError('decoded layer count mismatch')
                for blob,identity in zip(blobs,identities):
                    identity=identity if isinstance(identity,CacheIdentity) else CacheIdentity.from_dict(identity)
                    key=identity.digest();expected_sha=self.index[key][1]
                    if not isinstance(blob,bytes) or hashlib.sha256(blob).hexdigest()!=expected_sha:
                        raise ValueError('original snapshot SHA mismatch')
                    total+=len(blob)
                    if total>self.max_bytes:raise ValueError('decoded bundle byte bound exceeded')
                    staged[key]=blob
            self._check()  # readiness can change during SDK retrieve/decode
            self.blobs=staged;self.decoded_bytes=total
        except Exception as error:
            if isinstance(error,BundleRetrievalError):self.accounting=error.accounting
            if self.accounting is None:self.accounting=RetrievedBundle(None,self.selection.object_id)
            self.error=type(error).__name__+': '+str(error)
        finally:
            if self.accounting is not None:
                v=self.accounting
                self.accounting=RetrievedBundle(None,v.object_id,v.network_bytes,v.network_attempts,v.network_receipt)

    def fetch(self,identity,bs,hidden,input_dtype,expected_input,cache_digest):
        with self.lock:
            self._check()
            key=identity.digest()
            # A request/layer not represented by this selected object cannot cause retrieve.
            if key not in self.index:return None,{'reason':'bundle_identity_miss'}
            self._open()
            if self.error:return None,{'reason':'bundle_rejected','error':self.error}
            try:
                self._check();blob=self.blobs[key]
                tensors=unpack_snapshot(blob,wire_identity(identity,bs),max_bytes=self.max_bytes)
                if set(tensors)!={'key','value','aux_input','aux_output'}:raise ValueError('exact native/aux pair required')
                for name in ('key','value'):
                    if tuple(tensors[name].shape)!=(1,identity.num_attention_heads,bs,identity.head_dim) or str(tensors[name].dtype).removeprefix('torch.')!=identity.dtype:
                        raise ValueError('native layer geometry/dtype mismatch')
                for name in ('aux_input','aux_output'):
                    if tuple(tensors[name].shape)!=(1,1,bs,hidden) or tensors[name].dtype!=input_dtype:
                        raise ValueError('auxiliary geometry/dtype mismatch')
                if not torch.equal(tensors['aux_input'][:,0],expected_input):raise ValueError('actual layer input differs')
                if not all(bool(torch.isfinite(t).all().item()) for t in tensors.values()):raise ValueError('nonfinite payload')
                return tensors,dict(cache_identity=identity.to_dict(),identity_digest=key,
                    sha256=self.index[key][1],bytes=len(blob),logical_bytes=sum(t.numel()*t.element_size() for t in tensors.values()),
                    holder='sdk_selected_ready_bundle',endpoint='sdk_bundle',http_bytes=0,http_attempts=0,rejections=[],
                    bundle_object_id=self.selection.object_id,bundle_selection_id=self.selection.selection_id,
                    bundle_source=True,**source_digests(tensors,cache_digest))
            except Exception as error:return None,dict(reason='bundle_layer_rejected',error=type(error).__name__+': '+str(error))

    def take_accounting(self):
        """Charge the SDK retrieval exactly once, including errors before fallback."""
        with self.lock:
            if self.pid!=os.getpid():raise RuntimeError('different process')
            if self.charged or not self.attempted:
                return dict(bundle_open_calls=0,bundle_network_bytes=0,bundle_network_accounting_complete=True,
                    bundle_sdk_retrieve_calls=0,
                    bundle_selection_id=self.selection.selection_id,bundle_accounting_already_charged=self.charged)
            self.charged=True;value=self.accounting
            known=value is not None and type(value.network_bytes) is int and value.network_bytes>=0 and bool(value.network_receipt)
            return dict(bundle_open_calls=1,bundle_object_id=self.selection.object_id,
                bundle_sdk_retrieve_calls=self.sdk_retrieve_calls,
                bundle_network_bytes=value.network_bytes if known else None,
                bundle_network_attempts=value.network_attempts if value else None,
                bundle_network_accounting_complete=known,
                bundle_network_receipt=value.network_receipt if value else None,
                bundle_selection_id=self.selection.selection_id,bundle_accounting_already_charged=False,
                bundle_native_bytes=self.native_bytes,bundle_decoded_bytes=self.decoded_bytes,bundle_error=self.error)

    def close(self):
        with self.lock:
            if self.pid!=os.getpid():raise RuntimeError('different process')
            self.blobs.clear();self.index.clear();self.retrieve=None;self.closed=True

    def __enter__(self):self._check();return self
    def __exit__(self,*unused):self.close()


class BundleRemotePlane:
    """Per-request proxy; uses ordinary RemotePlane native recovery unchanged.

    The shared base remains untouched. One lease may back all 32 layer proxies.
    The caller owns lease/base lifecycle and resolves the lease in the runtime
    process using explicit request metadata; never serialize a lease to workers.
    """
    def __init__(self,base,lease):
        self.base=base;self.lease=lease;self.settings=base.settings;self.io=base.io
        self._bound=base._bound

    def _fetch(self,*args):
        try:tensors,row=self.lease.fetch(*args)
        except Exception as error:tensors,row=None,dict(reason='bundle_unavailable',error=type(error).__name__+': '+str(error))
        if tensors is not None:return tensors,row
        tensors,fallback=self.base._fetch(*args)
        fallback=dict(fallback,bundle_rejection=row)
        return tensors,fallback

    def recover(self,*args,**kwargs):
        try:
            outputs,event=RemotePlane.recover(self,*args,**kwargs)
        except Exception:
            # The request owner must drain take_accounting even if native append fails.
            raise
        charge=self.lease.take_accounting();event.update(charge)
        # Native append is still owned by RemotePlane. Correct its historical
        # HTTP-only provenance label for this new, independently proven path.
        state=args[1] if len(args)>1 else kwargs['state']
        used=[]
        for row in event['remote_records']:
            if row.get('bundle_source'):
                key=(row['target_slot'],row['target_generation'])
                state.remote_gpu_sources[key]['origin']='remote_sdk_bundle'
                state.remote_gpu_sources[key]['last_transfer']='sdk_bundle_to_gpu'
                used.append(dict(sha256=row['sha256'],cache_identity=row['cache_identity'],
                    block_start=row['block_start'],block_end=row['block_end'],
                    slot=key[0],generation=key[1]))
        event['bundle_model_used']=used
        event['bundle_hit_tokens']=sum(row['block_end']-row['block_start'] for row in used)
        known=charge['bundle_network_accounting_complete']
        event['remote_network_bytes']=event['remote_http_bytes']+charge['bundle_network_bytes'] if known else None
        event['remote_network_accounting_complete']=known
        event['remote_network_bytes_scope']='fallback HTTP plus once-referenced SDK receipt; deduplicate transfer_id and use receipt phase, not request-network delta'
        event['remote_http_bytes_scope']='ordinary HTTP fallback only; SDK physical bytes separate'
        return outputs,event

    def publish(self,*args,**kwargs):return self.base.publish(*args,**kwargs)


class RequestBundleRegistry:
    """Explicit bounded attempt registry, owned by the ordinary runtime process.

    install() is ready metadata only. The provider is invoked at most once by
    the first represented layer. At most max_active leases reserve max_bytes
    each; no token-prefix shadow store or completed-attempt history is kept.
    """
    def __init__(self,*,max_active=1,max_bytes=256*1024**2,clock=time.monotonic,clock_domain=None):
        if type(max_active) is not int or not 1<=max_active<=64:raise ValueError('invalid active lease bound')
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('invalid byte bound')
        self.pid=os.getpid();self.max_active=max_active;self.max_bytes=max_bytes
        self.clock=clock;self.lock=threading.RLock();self.leases={}
        self.active=set();self.clock_domain=clock_domain

    def _owned(self):
        if self.pid!=os.getpid():raise RuntimeError('registry cannot cross processes')

    def install(self,attempt_id,selection,retrieve,*,active=True):
        self._owned()
        if not isinstance(attempt_id,str) or not 0<len(attempt_id)<=256:raise ValueError('bounded attempt_id required')
        with self.lock:
            if attempt_id in self.leases:raise ValueError('attempt already has an installed object')
            if len(self.leases)>=self.max_active:raise ValueError('active request bundle capacity exhausted')
            lease=BundleLease(selection,retrieve,max_bytes=self.max_bytes,clock=self.clock)
            lease._check()
            self.leases[attempt_id]=lease
            if active:self.active.add(attempt_id)
        return dict(installed=True,attempt_id=attempt_id,object_id=selection.object_id,
                    pid=self.pid,active=len(self.leases))

    def lookup(self,attempt_id):
        self._owned()
        with self.lock:return self.leases.get(attempt_id)

    def install_decoded(self,attempt_id,selection,decoded,*,active=True):
        """Adopt the SDK owner's already-decoded ticket; no retrieve/decode.

        page_ready means original SHA grid validated in this runtime process.
        This does not claim current official SDK L1 residency or model use.
        Failures return accounting and retire the rejected in-memory lease.
        """
        if not isinstance(decoded,DecodedBundle):raise TypeError('DecodedBundle ticket required')
        result=self.install(attempt_id,selection,lambda unused:decoded,active=active)
        lease=self.lookup(attempt_id)
        with lease.lock:lease._open()
        if lease.error:
            result.update(installed=False,page_ready=False,error=lease.error,
                terminal=self.terminal(None,dict(attempt_id=attempt_id,status='failed')))
        else:result.update(page_ready=True,decoded_bytes=lease.decoded_bytes,sdk_retrieve_calls=0)
        return result

    def accept(self,*,attempt_id,layout,blobs_by_block,sha_by_block,object_id,selection_id,
               expires_at,is_valid,network_bytes=None,network_attempts=None,network_receipt=None,active=True):
        """SDK bridge handoff: already retrieved/decoded original wire bytes.

        All arguments are in THIS runtime process. For IPC the SDK owner sends
        bounded bytes/metadata and reconstructs these arguments locally; a
        local handoff is never proof of B-to-A network traffic. network_receipt
        must identify the physical SDK transfer, not the handoff byte count.
        """
        selection=ReadySelection(object_id,selection_id,layout,layout.identities_by_block,
                                 sha_by_block,expires_at,is_valid)
        try:
            return self.install_decoded(attempt_id,selection,DecodedBundle(blobs_by_block,
                object_id,network_bytes,network_attempts,network_receipt),active=active)
        except Exception as error:
            known=type(network_bytes) is int and network_bytes>=0 and bool(network_receipt)
            return dict(installed=False,page_ready=False,attempt_id=attempt_id,
                error=type(error).__name__+': '+str(error),
                transfer_accounting=dict(network_bytes=network_bytes if known else None,
                    network_accounting_complete=known,network_attempts=network_attempts,
                    network_receipt=network_receipt,sdk_retrieve_calls=0))

    def ordinary_remote(self,base,config):
        attempt=config.get('attempt_id')
        with self.lock:lease=self.lookup(attempt) if attempt in self.active else None
        return BundleRemotePlane(base,lease) if lease is not None else base

    def terminal(self,state,value):
        """Idempotent callback for page_control's per-state terminal loop.

        Called after generate exits, under runtime state locks. Returns any
        pending retrieval accounting even when native append/model failed.
        The caller writes this returned receipt; close is memory release only.
        """
        self._owned()
        if value.get('status') not in ('complete_success','failed','cancelled'):
            raise ValueError('explicit terminal status required')
        with self.lock:
            lease=self.leases.pop(value['attempt_id'],None)
            self.active.discard(value['attempt_id'])
        if lease is None:return dict(closed=False,attempt_id=value['attempt_id'],reason='no_active_lease')
        try:
            accounting=lease.take_accounting()
            return dict(closed=True,attempt_id=value['attempt_id'],accounting=accounting)
        finally:lease.close()

    def close(self):
        self._owned()
        with self.lock:
            pending=list(self.leases.values());self.leases.clear();self.active.clear()
        receipts=[]
        for lease in pending:
            try:receipts.append(lease.take_accounting())
            finally:lease.close()
        return receipts

    def accept_ready_payload(self,payload,*,layout_factory=None,max_handoff_bytes=64*1024**2):
        """Bounded local-control schema v1: prepare bytes, then activate receipt.

        The HTTP owner also bounds Content-Length before parsing JSON. Its
        measured local body bytes/time are separate from the SDK transfer.
        """
        self._owned();started=time.monotonic()
        if not isinstance(payload,dict) or payload.get('version')!=1:raise ValueError('ready payload version1 required')
        if payload.get('runtime_pid')!=self.pid or self.clock_domain is None or payload.get('clock_domain')!=self.clock_domain:
            raise ValueError('ready ticket target process/clock domain differs')
        attempt=payload.get('attempt_id');phase=payload.get('phase')
        if phase=='activate':
            with self.lock:
                lease=self.leases.get(attempt)
                if lease is None:raise ValueError('no prepared ticket')
                if attempt in self.active:raise ValueError('ticket already active')
                lease._check();s=lease.selection
                if payload.get('selection_id')!=s.selection_id or payload.get('object_id')!=s.object_id:
                    raise ValueError('activation is for another selected object')
                release=payload.get('release_result',{})
                key=release.get('key',{})
                if (release.get('status')!='released' or key.get('model_name')!=s.layout.model_name
                        or key.get('cache_salt')!=s.layout.cache_salt):
                    raise ValueError('exact target L1 retirement not verified')
                self.active.add(attempt)
                return dict(page_ready=True,activated=True,attempt_id=attempt,selection_id=s.selection_id,
                            object_id=s.object_id,release_result=release)
        if phase!='prepare':raise ValueError('prepare or activate required')
        identities=payload.get('identities_by_block',[]);shas=payload.get('sha_by_block',[])
        encoded=payload.get('blobs_b64_by_block',[])
        if not identities or len(identities)>4096 or sum(len(row) for row in identities)>16384:
            raise ValueError('bounded nonempty identity grid required')
        if len(encoded)!=len(identities) or len(shas)!=len(identities):raise ValueError('handoff grid mismatch')
        total=0
        for row,ids,sha in zip(encoded,identities,shas):
            if len(row)!=len(ids) or len(sha)!=len(ids):raise ValueError('handoff layer grid mismatch')
            for value in row:
                if not isinstance(value,str):raise ValueError('base64 wire string required')
                total+=len(value)
                if total>max_handoff_bytes or total>4*((self.max_bytes+2)//3)+4*16384:
                    raise ValueError('encoded local handoff bound exceeded')
        blobs=[];decoded_bytes=0
        for row in encoded:
            values=[]
            for value in row:
                blob=base64.b64decode(value,validate=True);decoded_bytes+=len(blob)
                if decoded_bytes>self.max_bytes:raise ValueError('decoded local handoff bound exceeded')
                values.append(blob)
            blobs.append(values)
        if layout_factory is None:
            from bloombee.kv_cache.page_bundle import PageBundleLayout
            layout_factory=PageBundleLayout
        layout=layout_factory(identities,payload['tokens'],block_size=payload['block_size'],max_bytes=self.max_bytes)
        transfer=payload.get('transfer') or {}
        result=self.accept(attempt_id=attempt,layout=layout,blobs_by_block=blobs,sha_by_block=shas,
            object_id=payload['object_id'],selection_id=payload['selection_id'],expires_at=payload['expires_at'],
            is_valid=lambda:True,network_bytes=transfer.get('network_bytes'),
            network_attempts=transfer.get('network_attempts'),network_receipt=transfer.get('network_receipt'),active=False)
        if result.get('page_ready'):
            with self.lock:self.active.discard(attempt)
            result.update(prepared=True,page_ready=False,activation_required=True)
        result.update(handoff_encoded_chars=total,handoff_decoded_bytes=decoded_bytes,
                      handoff_cpu_seconds=time.monotonic()-started,
                      handoff_scope='target-local IPC only, not B-to-A physical proof')
        return result
