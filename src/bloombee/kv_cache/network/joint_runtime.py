"""Thin CPU sidecar factory. Controller/HTTP ingress stays with final execution."""
import hashlib,importlib.util,json,threading
from pathlib import Path
from bloombee.kv_cache.network.ready_bank import ReadyBank,TicketHTTPServer,LocalTicketClient
from bloombee.kv_cache.network.native_page_bridge import NativePageBundleBridge
from bloombee.kv_cache.network.joint_strategy import make_joint_runner,joint_context


def pinned_callable(spec,name):
    path=Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('release helper source SHA differs')
    module_spec=importlib.util.spec_from_file_location('joint_release_owner',path);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return getattr(module,name)


def page_probe(config):
    def probe(layout,manifest):
        import httpx
        try:
            with httpx.Client(trust_env=False,timeout=config.get('page_probe_timeout_s',5)) as client:
                r=client.post(config['page_control_url'].rstrip('/')+'/page_probe',json=dict(tokens=layout.tokens,identities_by_block=manifest['layout']['identities_by_block']));r.raise_for_status();body=r.json()
            if type(body.get('known'))is not bool or type(body.get('complete'))is not bool:return dict(known=False,complete=False,reason='malformed_actual_page_probe')
            return body
        except Exception as error:return dict(known=False,complete=False,reason='actual_page_probe_unavailable',error=str(error))
    return probe

class PayloadLogProbe:
    """A-owned native adapter log, not B private logs or tensor-size guesses.
    Caller supplies the actual logging code/PID proof and exact-key exclusivity.
    Missing file/incomplete line/unverified logger is unknown, never zero.
    """
    def __init__(self,config):self.config=config
    def __call__(self,phase,manifest):
        spec=self.config.get('native_payload_probe') or {};path=Path(spec.get('path',''))
        proof=spec.get('logger_proof')
        if not proof or not path.is_file():return dict(known=False,phase=phase,reason='physical_native_payload_logger_not_verified_or_file_absent')
        raw=path.read_bytes()
        if raw and not raw.endswith(b'\n'):return dict(known=False,phase=phase,reason='partial_payload_log')
        model=manifest['layout']['model_name'];salt=manifest['layout']['cache_salt'];events=[]
        for line in raw.splitlines():
            event=json.loads(line);parts=event.get('wire_key','').split('@')
            if len(parts)>=6 and parts[-5]==model and parts[-1]==salt:
                if event.get('event')!='native_valkey_GET_complete' or event.get('size_matches') is not True:return dict(known=False,phase=phase,reason='failed_or_ambiguous_native_read')
                events.append(event)
        from bloombee.kv_cache.network.bundle_contract import digest
        return dict(known=True,exclusive=spec.get('exclusive_native_reader') is True,scope_id=digest(dict(model_name=model,cache_salt=salt,run_id=self.config['run_id'])),gets=len(events),value_bytes=sum(e['actual_value_bytes'] for e in events),phase=phase,logger_proof=proof,event_refs=[dict(pid=e['pid'],monotonic_ns=e['monotonic_ns'],batch_task_id=e.get('batch_task_id')) for e in events])

class JointRuntime:
    def __init__(self,config,*,emit=None,transfer_probe=None,local_page_probe=None):
        self.config=config;self.emit=emit or (lambda event,**data:None)
        self.bank=ReadyBank(max_bytes=config.get('ready_bank_max_bytes',64*1024**2),max_prefixes=config.get('ready_bank_max_prefixes',4),ticket_ttl_s=config.get('ready_ticket_ttl_s',300),emit=self.emit)
        release=pinned_callable(config['release_helper'],'release_l1_chunk')
        self.bridge=NativePageBundleBridge(config,self.bank,emit=self.emit,transfer_probe=transfer_probe or PayloadLogProbe(config),release_fn=release)
        self.bank.retire_notify=self.bridge.defer_retirement
        self.runner=make_joint_runner(config,self.bridge,self.bank,self.emit,local_page_probe or page_probe(config))
        self.server=TicketHTTPServer(('127.0.0.1',int(config.get('ticket_port',0))),self.bank,self.bridge.acknowledge,token=config.get('ticket_control_token'))
        self.server.retirement_status=self.bridge.retirement_status
        self.http_thread=None
        self.ticket_endpoint='http://127.0.0.1:'+str(self.server.server_port)
    def start(self):
        if self.http_thread is None:
            self.http_thread=threading.Thread(target=self.server.serve_forever,name='page-bundle-loopback',daemon=True);self.http_thread.start()
        return dict(status='joint_sidecar_protocol_ready',ticket_endpoint=self.ticket_endpoint,bank=self.bank.accounting(),model_layers=32,native_storage_layers=64,native_sdk_or_GPU_verification=False)
    def score(self,history,boundary,decision_id,causal_cutoff_ns):
        from bloombee.kv_cache.network.strategy_runner import validate_demand_ledger,score_context
        validate_demand_ledger(history,boundary,causal_cutoff_ns)
        report=self.runner.qualify(history)
        context=joint_context(self.config,report,history,boundary,decision_id,causal_cutoff_ns)
        scores={strategy:score_context(context,strategy,self.config.get('strategy_transition_plugin') if strategy=='transition' else None) for strategy in ('lrfu','transition')}
        return dict(qualification=report,context=context,shadow_scores=scores,active_selection=scores[self.config['prefetch_strategy']])
    def close(self):
        if self.http_thread is not None:self.server.shutdown();self.http_thread.join();self.http_thread=None
        self.server.server_close();self.bank.close();self.runner.close()
        return dict(bank=self.bank.accounting(),native_staging=self.bridge.retirement_status())


def create_joint_runtime(config,**kwargs):return JointRuntime(config,**kwargs)
