"""Bounded A-local RAM ticket bank and one real loopback handoff.

Prediction never binds a future attempt. Only /claim from an actually arrived
request can bind it. /claim never invokes SDK/warm or waits for an in-flight warm.
"""
import http.server,ipaddress,json,threading,time,uuid
from urllib.parse import urlsplit
from bloombee.kv_cache.network.bundle_contract import digest,sha,validate_manifest,ticket_header,frame_parts,unpack_ticket_frame

class ReadyBank:
    def __init__(self,*,max_bytes=64*1024**2,max_prefixes=4,ticket_ttl_s=300,emit=None):
        if not 0<max_bytes<=64*1024**2 or not 0<max_prefixes<=4 or not 0<ticket_ttl_s<=3600:raise ValueError('explicit bounded transfer bank limits required')
        self.max_bytes=max_bytes;self.max_prefixes=max_prefixes;self.ttl=ticket_ttl_s;self.emit=emit or (lambda event,**data:None)
        self.entries={};self.index={};self.closed={};self.lock=threading.RLock();self.claim_count=0;self.retire_notify=None;self.epoch=0
    def _remove(self,ticket,status):
        if ticket.get('sending'):
            ticket['deferred_remove_status']=status;self.epoch+=1;return
        self.entries.pop(ticket['ticket_id'],None);self.epoch+=1
        for identity in ticket['manifest']['layout']['identities_by_block'][0]:
            key=digest(identity)
            if self.index.get(key)==ticket['ticket_id']:self.index.pop(key,None)
        # Callback only records metadata; no key lock/network while bank.lock is held.
        if ticket.get('release_result',{} ) is None or (ticket.get('release_result') or {}).get('status')!='released':
            if self.retire_notify:self.retire_notify(ticket,status)
        self.closed[ticket['ticket_id']]=dict(attempt_id=ticket.get('attempt_id'),status=status,release_result=ticket.get('release_result'),object_id=ticket['manifest']['object_id'],native_retirement_pending=(ticket.get('release_result') or {}).get('status')!='released')
        ticket['blobs']=None
    def _expire(self):
        for ticket in list(self.entries.values()):
            if ticket['state']=='ready' and time.monotonic()>=ticket['expires_at']:self._remove(ticket,'expired_unclaimed')
    def accounting(self):
        with self.lock:
            return dict(prefixes=len(self.entries),payload_bytes=sum(t['payload_bytes'] for t in self.entries.values()),metadata_json_bytes=sum(t['metadata_bytes'] for t in self.entries.values()),managed_total_bytes=sum(t['payload_bytes']+t['metadata_bytes'] for t in self.entries.values()),byte_limit=self.max_bytes,prefix_limit=self.max_prefixes,python_heap_and_transient_copies_separate=True)
    def inventory(self):
        """Pure read, no claim/SDK/expiry deletion; bounded public metadata only.
        Epoch binds actual mutations; compare objects too because time alone
        can expire readiness without an explicit mutation.
        """
        with self.lock:
            now=time.monotonic();objects=[]
            for ticket in self.entries.values():
                manifest=ticket['manifest'];warm=ticket.get('native_warm_receipt') or {};after=warm.get('after') or {}
                objects.append(dict(ticket_id=ticket['ticket_id'],object_id=manifest['object_id'],source_manifest=manifest,manifest_sha256=digest(manifest),validated_complete_32layer_sha=True,bank_ready_ns=ticket['header']['bank_ready_ns'],state=ticket['state'],claimable=ticket['state']=='ready' and now<ticket['expires_at'],expired=now>=ticket['expires_at'],sending=ticket.get('sending',False),deferred_remove_status=ticket.get('deferred_remove_status'),expires_at=ticket['expires_at'],attempt_id=ticket['attempt_id'],local_payload_bytes=ticket['payload_bytes'],native_target=warm.get('target'),native_registration=after.get('registration'),native_chunk_incarnations=after.get('chunks'),expected_key=ticket.get('expected_key')))
            return dict(schema_version=1,inventory_revision='joint_bank_inventory_a1',bank_epoch=self.epoch,actual_monotonic_ns=time.monotonic_ns(),objects=sorted(objects,key=lambda row:row['ticket_id']),scope='A-local bank metadata; not physical native or model usage proof')
    def has_object(self,object_id):
        with self.lock:
            self._expire();return any(t['manifest']['object_id']==object_id for t in self.entries.values())
    def offer(self,manifest,blobs,transfer,decision_link,*,native_warm_receipt=None,expected_key=None):
        validate_manifest(manifest);payload=0
        if len(blobs)!=1 or len(manifest['wire_bytes_by_block'])!=1 or len(manifest['wire_bytes_by_block'][0])!=32:raise ValueError('complete one-page32-layer wire grid required')
        for row,expected,sizes in zip(blobs,manifest['sha_by_block'],manifest['wire_bytes_by_block']):
            if len(row)!=len(expected):raise ValueError('incomplete decoded page layer set')
            for blob,h,size in zip(row,expected,sizes):
                if not isinstance(blob,bytes) or len(blob)!=size or sha(blob)!=h:raise ValueError('decoded original object mismatch')
                payload+=len(blob)
        with self.lock:
            self._expire();ticket_id=uuid.uuid4().hex;expires=time.monotonic()+self.ttl
            header=ticket_header(manifest,ticket_id,expires,transfer,decision_link);header['bank_ready_ns']=time.monotonic_ns()
            metadata_bytes=len(json.dumps(header,sort_keys=True,separators=(',',':')).encode())
            if len(self.entries)>=self.max_prefixes or self.accounting()['managed_total_bytes']+payload+metadata_bytes>self.max_bytes:raise ValueError('managed ready-bank budget exhausted')
            ticket=dict(ticket_id=ticket_id,manifest=manifest,blobs=tuple(tuple(r) for r in blobs),header=header,payload_bytes=payload,metadata_bytes=metadata_bytes,expires_at=expires,state='ready',attempt_id=None,native_warm_receipt=native_warm_receipt,expected_key=expected_key)
            identities=manifest['layout']['identities_by_block'][0]
            if any(digest(i) in self.index for i in identities):raise ValueError('another ticket already owns an identical page identity')
            header['bank_ready_ns']=time.monotonic_ns()
            self.entries[ticket_id]=ticket;self.epoch+=1
            for identity in identities:self.index[digest(identity)]=ticket_id
            return ticket
    def claim(self,*,attempt_id,page_identity_digest,object_id=None):
        with self.lock:
            self._expire();ticket=self.entries.get(self.index.get(page_identity_digest))
            if ticket is None or ticket['state']!='ready':return None
            if object_id is not None and object_id!=ticket['manifest']['object_id']:return None
            ticket['state']='claimed';ticket['attempt_id']=attempt_id;ticket['sending']=True;self.claim_count+=1;self.epoch+=1
            self.emit('local_bundle_claimed',ticket_id=ticket['ticket_id'],object_id=ticket['manifest']['object_id'],attempt_id=attempt_id,local_payload_bytes=ticket['payload_bytes'],cross_machine_sdk_calls=0)
            return ticket
    def handoff_finished(self,ticket_id,succeeded):
        with self.lock:
            ticket=self.entries.get(ticket_id)
            if ticket is None:return
            ticket['sending']=False;self.epoch+=1
            if not succeeded:self._remove(ticket,'local_handoff_failed')
            elif ticket.get('deferred_remove_status'):self._remove(ticket,ticket['deferred_remove_status'])
    def acknowledge(self,*,ticket_id,attempt_id,accepted,release_callback):
        with self.lock:
            if ticket_id in self.closed:
                old=self.closed[ticket_id]
                if old['attempt_id']!=attempt_id:raise ValueError('closed ticket belongs to another attempt')
                return dict(status=old['status'],release_result=old.get('release_result'),idempotent=True)
            ticket=self.entries.get(ticket_id)
            if ticket is not None and ticket['attempt_id']==attempt_id and ticket['state']=='acknowledging':return dict(status='release_pending_not_ready')
            if ticket is None or ticket['attempt_id']!=attempt_id or ticket['state']!='claimed':raise ValueError('ack has no matching claimed ticket')
            if ticket.get('release_result') is not None:return dict(status='page_ready' if ticket['release_result'].get('status')=='released' else 'not_ready',release_result=ticket['release_result'],idempotent=True)
            ticket['state']='acknowledging';self.epoch+=1
        # No SDK read here. Callback only the owner's exact L1 retirement helper.
        try:release=release_callback(ticket,attempt_id,accepted)
        except Exception as error:release=dict(status='release_error',error_type=type(error).__name__,error=str(error))
        with self.lock:
            ticket['release_result']=release
            status='page_ready' if accepted and release.get('status')=='released' else 'not_ready'
            self.emit('page_accept_acknowledged',ticket_id=ticket_id,attempt_id=attempt_id,status=status,release_result=release,bank=self.accounting())
            # Provider now owns the bytes or must abort its prepared reservation.
            # Drop payload references here; terminal metadata stays idempotent.
            self._remove(ticket,status)
            return dict(status=status,release_result=release)
    def close(self):
        with self.lock:
            for ticket in list(self.entries.values()):self._remove(ticket,'sidecar_close')
    def terminal(self,attempt_id,status):
        with self.lock:
            for ticket in list(self.entries.values()):
                if ticket['attempt_id']==attempt_id:self._remove(ticket,'terminal_'+status)
            return dict(status='terminal_recorded',attempt_id=attempt_id,bank=self.accounting())

class TicketHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,bank,release_callback,*,token=None):
        if not ipaddress.ip_address(address[0]).is_loopback:raise ValueError('ticket handoff must bind target-local loopback')
        super().__init__(address,TicketHandler);self.bank=bank;self.release_callback=release_callback;self.token=token

class TicketHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*unused):pass
    def json_response(self,status,value):
        raw=json.dumps(value,sort_keys=True).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        if self.server.token and self.headers.get('X-Bridge-Token')!=self.server.token:return self.json_response(403,dict(status='forbidden'))
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=1024**2:raise ValueError('bounded control JSON required')
            request=json.loads(self.rfile.read(n))
            if self.path=='/claim':
                ticket=self.server.bank.claim(attempt_id=request['attempt_id'],page_identity_digest=request['page_identity_digest'],object_id=request.get('object_id'))
                if ticket is None:return self.json_response(404,dict(status='not_ready_immediate_fallback'))
                succeeded=False
                try:
                    parts=frame_parts(ticket['header'],ticket['blobs']);total=sum(map(len,parts))
                    self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(total));self.end_headers()
                    for part in parts:self.wfile.write(part)
                    self.wfile.flush();succeeded=True
                    self.server.bank.emit('local_handoff_sent',ticket_id=ticket['ticket_id'],attempt_id=request['attempt_id'],http_body_bytes=total,native_sdk_calls=0)
                finally:self.server.bank.handoff_finished(ticket['ticket_id'],succeeded)
            elif self.path=='/ack':self.json_response(200,self.server.bank.acknowledge(ticket_id=request['ticket_id'],attempt_id=request['attempt_id'],accepted=request['accepted'],release_callback=self.server.release_callback))
            elif self.path=='/terminal':self.json_response(200,self.server.bank.terminal(request['attempt_id'],request.get('status','unknown')))
            elif self.path=='/status':self.json_response(200,dict(bank=self.server.bank.accounting(),inventory=self.server.bank.inventory(),native_staging=self.server.retirement_status() if hasattr(self.server,'retirement_status') else None))
            else:self.json_response(404,dict(status='unknown_control_route'))
        except Exception as error:self.json_response(409,dict(status='handoff_error',error_type=type(error).__name__,error=str(error)))

class LocalTicketClient:
    def __init__(self,endpoint,*,timeout=10,max_bytes=64*1024**2,token=None):
        from urllib.parse import urlsplit
        p=urlsplit(endpoint)
        if p.scheme!='http' or not ipaddress.ip_address(p.hostname).is_loopback or p.path not in ('','/'):raise ValueError('handoff endpoint must be A-local loopback')
        import httpx
        self.endpoint=endpoint.rstrip('/');self.client=httpx.Client(trust_env=False,timeout=timeout);self.max_bytes=max_bytes;self.headers={'X-Bridge-Token':token} if token else {}
    def claim(self,attempt_id,page_identity_digest,object_id=None):
        with self.client.stream('POST',self.endpoint+'/claim',headers=self.headers,json=dict(attempt_id=attempt_id,page_identity_digest=page_identity_digest,object_id=object_id)) as response:
            if response.status_code==404:return None
            response.raise_for_status();limit=self.max_bytes+1024**2+4;parts=[];total=0
            for part in response.iter_bytes():
                total+=len(part)
                if total>limit:raise ValueError('local handoff exceeded body bound')
                parts.append(part)
        header,blobs=unpack_ticket_frame(b''.join(parts),max_bytes=self.max_bytes)
        header['local_handoff_http_body_bytes']=total
        return header,blobs
    def claim_for_layout(self,attempt_id,layout):
        return self.claim(attempt_id,digest(layout.identities_by_block[0][0].to_dict()))
    def ack(self,ticket_id,attempt_id,accepted):
        r=self.client.post(self.endpoint+'/ack',headers=self.headers,json=dict(ticket_id=ticket_id,attempt_id=attempt_id,accepted=accepted));r.raise_for_status();return r.json()
    def terminal(self,attempt_id,status):
        r=self.client.post(self.endpoint+'/terminal',headers=self.headers,json=dict(attempt_id=attempt_id,status=status));r.raise_for_status();return r.json()
    def close(self):self.client.close()
