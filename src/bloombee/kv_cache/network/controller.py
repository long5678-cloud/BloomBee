"""Thin ordinary-client adapter for the frozen joint worker file protocol.

No worker/model launch, queue reader, SDK calls or ranking implementation.
Only actual terminal H reaches score_request. Current ids only reach claim.
"""
import copy,hashlib,importlib.util,json,os,threading,time,uuid
from pathlib import Path
from bloombee.kv_cache.network.release_ack import wait_released
from bloombee.kv_cache.network.lifecycle import LifecycleError, failure
from bloombee.kv_cache.network.bundle_contract import digest,page_ready_payload,validate_manifest
from bloombee.kv_cache.network.joint_strategy import layout_for_tokens
from bloombee.kv_cache.network.ready_bank import LocalTicketClient

META_ALLOWED={'segment_id','run_id','active_strategy','source_id','registry_sha256','completed_trace_rows','clock_domain_id','freeze_sha256','raw_qualification_ref'}

def read(path):
    # A writer creates the file before its JSON body is complete. A complete
    # JSON parse is the visibility gate; existence alone is not a commit.
    deadline=time.monotonic()+10
    while True:
        try:return json.loads(Path(path).read_text())
        except (json.JSONDecodeError,FileNotFoundError):
            if time.monotonic()>=deadline:raise
            time.sleep(.01)
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def write_bytes(path, body):
    """Publish the exact frozen bytes without replacement; retain draft evidence."""
    path = Path(path)
    draft = path.with_name('.draft_' + uuid.uuid4().hex)
    with draft.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(draft, path)
    return file_sha(path)


def write(path, value):
    return write_bytes(path, json_bytes(value))


def pinned(spec):
    path=Path(spec['path'])
    if file_sha(path)!=spec['sha256']:raise ValueError('independent audit source SHA differs')
    module_spec=importlib.util.spec_from_file_location('joint_controller_independent_audit',path);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return getattr(module,spec.get('function','audit'))

class Controller:
    def __init__(self,job,config,output_root):
        import httpx
        self.job=job;self.config=config;self.spec=config['prefetch_controller'];self.root=Path(output_root)/'prefetch_controller_a1';self.root.mkdir(exist_ok=False)
        self.inbox=Path(self.spec['inbox']);self.worker_config_path=Path(self.spec['worker_config_path']);self.worker_config=read(self.worker_config_path);self.config_sha=file_sha(self.worker_config_path)
        self.boundary=copy.deepcopy(self.spec['history_boundary']);self.history=copy.deepcopy(self.spec['initial_history']);self.metadata=copy.deepcopy(self.spec['score_metadata'])
        self.request_count=int(self.spec['request_count'])
        if self.request_count<=0:raise ValueError('explicit formal request count required')
        self.manifests={};source=self.spec.get('source_manifests',[])
        if isinstance(source,str):
            path=Path(source)
            try:source=json.loads(path.read_text())
            except json.JSONDecodeError:source=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if isinstance(source,dict):source=source['manifests']
        for manifest in source:
            validate_manifest(manifest);key=(manifest['layout']['model_name'],manifest['layout']['cache_salt'])
            if manifest['run_id']!=self.worker_config.get('source_run_id',self.worker_config['run_id']) or manifest['source_instance']!=self.worker_config['source_instance']:raise ValueError('wrong disclosed source library')
            if key in self.manifests and self.manifests[key]['object_id']!=manifest['object_id']:raise ValueError('immutable published source key changed')
            self.manifests[key]=manifest
        audit_spec=self.spec.get('independent_audit', 'builtin')
        if audit_spec == 'builtin':
            from bloombee.kv_cache.network.independent_audit import audit
            self.audit=audit
        else:
            self.audit=pinned(audit_spec) if audit_spec is not None else None
        self.http=httpx.Client(trust_env=False,timeout=self.spec.get('control_timeout_s',120));self.page_endpoint=self.worker_config['page_control_url'].rstrip('/')
        self.ticket=None;self.ready=None;self.started=False;self.closed=False;self.decisions=[];self.pending_attempts={};self.terminal_attempts=set();self.disclosed_attempts=set();self.lock=threading.RLock()
        self._attach_manifests(self.history)
        from bloombee.kv_cache.network.strategy_runner import validate_demand_ledger
        validate_demand_ledger(self.history,self.boundary,time.monotonic_ns())

    def _attach_manifests(self,history):
        for event in history:
            layout=layout_for_tokens(self.worker_config,event['token_ids']);manifest=self.manifests.get((layout.model_name,layout.cache_salt))
            if manifest:event['source_bundle_manifest']=copy.deepcopy(manifest)

    def _wait(self,path):
        path=Path(path);start=time.monotonic()
        while not path.exists():
            errors=self.inbox/'worker_errors.jsonl'
            if errors.exists() and errors.stat().st_size:raise RuntimeError('actual worker error; inspect '+str(errors))
            if path.name!='worker_done.json' and (self.inbox/'worker_done.json').exists() and not path.exists():raise RuntimeError('worker exited before required protocol marker '+path.name)
            timeout=self.spec.get('ipc_timeout_s')
            if timeout is not None and time.monotonic()-start>timeout:raise TimeoutError('IPC marker unavailable; worker not stopped: '+str(path))
            if path.exists():break
            if self.ready is not None:
                try:os.kill(self.ready['worker_pid'],0)
                except ProcessLookupError:
                    if path.exists():break
                    raise RuntimeError('joint worker PID no longer exists')
            time.sleep(.01)
        return read(path)

    def _connect(self):
        if self.ticket is not None and not getattr(self,'_ticket_closed',False):return
        ready=self._wait(self.inbox/'joint_ready.json')
        if ready.get('config_sha256')!=self.config_sha:raise ValueError('joint_ready frozen config SHA differs')
        self.ready=ready;self.ticket=LocalTicketClient(ready['ticket_endpoint'],timeout=self.spec.get('control_timeout_s',120),token=self.worker_config.get('ticket_control_token'))
        self._ticket_closed=False

    def _post(self,route,value):
        if getattr(self,'_http_closed',False):
            import httpx
            self.http=httpx.Client(trust_env=False,timeout=self.spec.get('control_timeout_s',120));self._http_closed=False
        response=self.http.post(self.page_endpoint+route,json=value);response.raise_for_status();return response.json()

    def start_after_t0(self):
        """Call only after actual offered T0. First real claim invokes lazily."""
        with self.lock:
            if self.started:return
            if not getattr(self,'actual_t0_ns',None):raise ValueError('actual ingress proof required')
            self._connect();self.started=True
            decision = self._score(final=False, enabled=bool(self.spec.get('initial_warm_enabled', True)))
            self._dispatch(decision)
            return decision

    def claim_current(self,*,attempt_id,token_ids,actual_ingress_ns=None,clock_domain=None):
        with self.lock:
            if self.closed or getattr(self,'closing',False):raise RuntimeError('controller closed or closing')
            if attempt_id in self.pending_attempts or attempt_id in self.terminal_attempts:raise ValueError('attempt already served; do not reserve/claim twice')
            self._connect()
            if type(actual_ingress_ns) is not int or actual_ingress_ns>time.monotonic_ns():raise ValueError('actual ingress required')
            self.actual_t0_ns=actual_ingress_ns
            # Match whatever is ready at claim time. A pending warm can finish
            # independently and remain available to a later request.
            self.start_after_t0()
            reserved=self._post('/reserve_current',dict(attempt_id=attempt_id,tokens=list(token_ids[:16])))
            if reserved.get('reserved') is not True:raise ValueError('actual current request not reserved')
            self.pending_attempts[attempt_id]=dict(token_ids=list(token_ids),reserved=reserved)
            outcome=dict(status='not_ready_immediate_fallback',attempt_id=attempt_id,reservation=reserved,
                         actual_ingress_ns=actual_ingress_ns,actual_ingress_clock_domain=clock_domain)
            header=None;prepared=False;ack=None
            try:
                layout=layout_for_tokens(self.worker_config,token_ids);packet=self.ticket.claim_for_layout(attempt_id,layout)
                if packet is None:return outcome
                header,blobs=packet
                status=self.http.get(self.page_endpoint+'/status');status.raise_for_status();runtime=status.json()
                same_clock=isinstance(clock_domain,str) and bool(clock_domain) and clock_domain==runtime['clock_domain']
                outcome.update(actual_ingress_ns=actual_ingress_ns,actual_ingress_clock_domain=clock_domain,actual_runtime_clock_domain=runtime['clock_domain'],bank_ready_ns=header.get('bank_ready_ns'),ready_before_actual_ingress=(header['bank_ready_ns']<actual_ingress_ns) if same_clock and type(actual_ingress_ns)is int and type(header.get('bank_ready_ns'))is int else None)
                payload=page_ready_payload(header,blobs,attempt_id=attempt_id,runtime_pid=runtime['pid'],clock_domain=runtime['clock_domain'])
                prep=self._post('/ready',payload);prepared=prep.get('prepared') is True and prep.get('installed') is True and prep.get('page_ready') is False
                ack=self.ticket.ack(header['ticket_id'],attempt_id,prepared)
                outcome.update(ticket_id=header['ticket_id'],object_id=header['object_id'],prepare=prep,ack=ack,local_handoff_http_body_bytes=header['local_handoff_http_body_bytes'],transfer=header['transfer'])
                if not prepared or ack.get('status')!='page_ready' or (ack.get('release_result') or {}).get('status')!='released':raise ValueError('prepared payload or exact native retirement not confirmed')
                activation=self._post('/ready',dict(version=1,phase='activate',attempt_id=attempt_id,runtime_pid=runtime['pid'],clock_domain=runtime['clock_domain'],selection_id=header['selection_id'],object_id=header['object_id'],release_result=ack['release_result']))
                if activation.get('activated') is not True or activation.get('page_ready') is not True:raise ValueError('ordinary bundle activation not confirmed')
                outcome.update(status='page_ready_activated',activation=activation)
                return outcome
            except Exception as error:
                outcome.update(status='claim_failed_ordinary_fallback',error_type=type(error).__name__,error=str(error))
                if header is not None and ack is None:
                    try:outcome['rejection_ack']=self.ticket.ack(header['ticket_id'],attempt_id,False)
                    except Exception as cleanup:outcome['rejection_ack_error']=str(cleanup)
                # Abort only the prepared reader; retain the CURRENT demand reservation.
                if header is not None:
                    try:
                        outcome['prepared_abort']=self._post('/abort_ready',dict(attempt_id=attempt_id))
                    except Exception as cleanup:
                        outcome['cleanup_error']=str(cleanup)
                        write(self.root/('claim_'+digest(attempt_id)+'.json'),outcome)
                        raise RuntimeError('prepared bundle abort/reservation unconfirmed; do not dispatch ordinary model') from cleanup
                return outcome
            finally:
                path=self.root/('claim_'+digest(attempt_id)+'.json')
                if not path.exists():write(path,outcome)

    owns_page_terminal = True

    def _release_post(self, attempt, timeout):
        import httpx
        try:
            response = self.http.post(self.page_endpoint + '/release_ack',
                                      json={'attempt_id': attempt}, timeout=timeout)
        except httpx.TimeoutException as error:
            raise TimeoutError('release_ack transport timeout') from error
        response.raise_for_status()
        return response.json()

    def terminal(self,*,terminal):
        """Every status calls this; failures/cancellations never join H."""
        with self.lock:
            attempt=terminal['attempt_id']
            if not hasattr(self,'terminal_intents'):self.terminal_intents={}
            prior=self.terminal_intents.get(attempt)
            if prior is not None and prior['status']!=terminal['status']:raise ValueError('contradictory original terminal status')
            self.terminal_intents[attempt]=copy.deepcopy(terminal)
            if attempt in self.terminal_attempts:return dict(status='terminal_idempotent')
            self._connect()
            value=dict(attempt_id=attempt,status=terminal['status'])
            if not hasattr(self, 'terminal_steps'):
                self.terminal_steps = {}
            steps = self.terminal_steps.setdefault(attempt, {})
            errors = []
            if 'page' not in steps:
                try:
                    steps['page'] = self._post('/terminal', value)
                except Exception as error:
                    errors.append(failure('page.terminal', error))
            if 'bank' not in steps:
                try:
                    steps['bank'] = self.ticket.terminal(attempt, terminal['status'])
                except Exception as error:
                    errors.append(failure('ticket.terminal', error))
            if errors:
                raise LifecycleError(errors, terminal=terminal)
            result = dict(page=steps['page'], bank=steps['bank'],
                          reservation=self.pending_attempts.get(attempt, {}).get('reserved'))
            target=self.root/('terminal_'+digest(attempt)+'.json')
            if not target.exists():write(target,result)
            self.terminal_attempts.add(attempt)
            self.pending_attempts.pop(attempt, None)
            del self.terminal_steps[attempt]
            return result

    def disclose_complete(self,*,history,terminal):
        with self.lock:
            if self.closed or getattr(self, 'closing', False):
                raise RuntimeError('controller closed or closing')
            self.terminal(terminal=terminal)
            if terminal['status']!='complete_success':return dict(status='failed_terminal_not_disclosed')
            attempt=terminal['attempt_id']
            if attempt in self.disclosed_attempts:
                # Resume the frozen request on a late score receipt or retry its audited command.
                for decision in reversed(self.decisions):
                    if decision.get('disclosed_attempt_id') == attempt:
                        if 'execute_command' not in decision and 'no_execute_reason' not in decision:
                            self._collect_score(decision)
                        self._dispatch(decision)
                        return dict(status='disclosure_idempotent')
                raise RuntimeError('previous disclosure has no completed score receipt')
            known={h['request_id'] for h in self.history}
            fresh=[h for h in history if h['request_id'] not in known]
            if len(fresh)!=1 or fresh[0].get('attempt_id')!=attempt:raise ValueError('exactly one newly completed ordinary request required')
            raw=fresh[0]
            if raw.get('status')!='complete_success' or raw.get('actual_output_tokens')!=raw.get('requested_output_tokens'):raise ValueError('incomplete model output cannot advance H')
            ref=Path(raw['terminal_ref']);proof=read(ref)
            if proof.get('attempt_id')!=attempt or proof.get('request_id')!=terminal['request_id'] or proof.get('status')!='complete_success' or proof.get('actual_output_tokens')!=proof.get('requested_output_tokens'):raise ValueError('full terminal file receipt differs')
            if (raw['request_id'] != terminal['request_id'] or
                    type(proof.get('requested_output_tokens')) is not int or
                    proof['requested_output_tokens'] <= 0 or
                    proof['requested_output_tokens'] != raw.get('requested_output_tokens')):
                raise ValueError('model output receipt mismatch')
            frozen = read(self.root / ('terminal_' + digest(attempt) + '.json'))
            ack = wait_released(attempt, frozen['page'].get('release_expected'),
                                frozen['reservation'] or {}, self._release_post,
                                self.spec.get('release_ack_timeout_s', 3.0))
            ack_path = self.root / ('release_' + digest(attempt) + '.json')
            if not ack_path.exists():
                write(ack_path, ack)
            event=dict(request_id=raw['request_id'],attempt_id=attempt,token_ids=list(raw['token_ids']),outcome='processed_success',consumer_instance_id=self.boundary['consumer_id'],completion_event_seq=self.boundary['initial_completion_seq']+len(self.history)+1,disclosed_ns=time.monotonic_ns(),clock_domain_id=self.boundary['clock_domain_id'],terminal_receipt_ref=dict(path=str(ref),sha256=file_sha(ref)),history_source='ordinary_full_terminal',visibility_scope='consumer')
            self._attach_manifests([event])
            write(self.root/('history_'+str(len(self.disclosed_attempts)+1)+'.json'),event)
            self.history.append(event)
            self.disclosed_attempts.add(attempt)
            final=terminal.get('index',len(self.terminal_attempts)-1)+1>=self.request_count
            decision = self._score(final=final, enabled=True, disclosed_attempt_id=attempt)
            self._dispatch(decision)
            return decision

    def _score(self,*,final,enabled,disclosed_attempt_id=None):
        from bloombee.kv_cache.network.strategy_runner import validate_demand_ledger
        H=copy.deepcopy(self.history);cutoff=time.monotonic_ns();validate_demand_ledger(H,self.boundary,cutoff)
        n=len(self.decisions)+1;decision_id=self.worker_config['run_id']+'_d'+str(n);directory=self.inbox/('run_'+str(n))
        metadata=copy.deepcopy(self.metadata)
        metadata.update(active_strategy=self.worker_config['prefetch_strategy'],clock_domain_id=self.boundary['clock_domain_id'],raw_qualification_ref=str(directory/'qualification.json'))
        if self.spec.get('source_manifests') and isinstance(self.spec['source_manifests'],str):metadata['registry_sha256']=file_sha(self.spec['source_manifests'])
        if set(metadata)-META_ALLOWED or not (META_ALLOWED-{'completed_trace_rows'})<=metadata.keys():raise ValueError('original a2 metadata allowed set required')
        request=dict(history=H,boundary=copy.deepcopy(self.boundary),decision_id=decision_id,causal_cutoff_ns=cutoff,metadata=metadata)
        # Reserve the decision and its exact wire bytes before touching the inbox.
        wire = json_bytes(request)
        record = dict(decision_id=decision_id, directory=str(directory), final=final, enabled=enabled,
                      disclosed_attempt_id=disclosed_attempt_id, score_request_sha256=hashlib.sha256(wire).hexdigest(),
                      execute_requested=False)
        self.decisions.append(record)
        record['pending_score_wire'] = wire
        return self._collect_score(record)

    def _publish_score_request(self, record):
        wire = record['pending_score_wire']
        directory = Path(record['directory'])
        intent = self.root / ('score_intent_' + digest(record['decision_id']) + '.json')
        # The private receipt binds the frozen request to the attempt and run
        # even when the worker directory or the first public hard link fails.
        if not record.get('intent_saved'):
            envelope = dict(request=json.loads(wire), decision_id=record['decision_id'],
                            disclosed_attempt_id=record['disclosed_attempt_id'], directory=str(directory),
                            final=record['final'], enabled=record['enabled'],
                            score_request_sha256=record['score_request_sha256'])
            try:
                write(intent, envelope)
            except FileExistsError:
                if read(intent) != envelope:
                    raise ValueError('conflicting frozen score intent')
            record['intent_saved'] = True
        if not record.get('directory_created'):
            directory.mkdir(exist_ok=False)
            record['directory_created'] = True
        path = directory/'score_request.json'
        # Reconcile an ambiguous link result without overwriting an existing command.
        if not path.exists():
            try:
                write_bytes(path, wire)
            except FileExistsError:
                pass
        if file_sha(path) != record['score_request_sha256']:
            raise ValueError('published score request differs from frozen score intent')
        del record['pending_score_wire']

    def _collect_score(self, record):
        if 'pending_score_wire' in record:
            self._publish_score_request(record)
        directory = Path(record['directory'])
        decision_id = record['decision_id']
        request_sha = record['score_request_sha256']
        body = (directory/'score_request.json').read_bytes()
        if hashlib.sha256(body).hexdigest() != request_sha:
            raise ValueError('frozen score request changed')
        request = json.loads(body)
        if request['decision_id'] != decision_id:
            raise ValueError('frozen score decision identity mismatch')
        ready = directory/'score_ready.json'
        if ready.exists():
            if read(ready) != dict(score_request_sha256=request_sha):
                raise ValueError('score readiness marker differs from frozen request')
        else:
            write(ready, dict(score_request_sha256=request_sha))
        done = self._wait(directory/'score_done.json')
        if done.get('status') != 'scored':
            write(directory/'controller_not_executed.json', dict(reason='not_scored', score_done=done))
            record['no_execute_reason'] = 'not_scored'
            return record
        if done.get('config_sha256') != self.config_sha or done.get('score_request_sha256') != request_sha:
            raise ValueError('worker scored different frozen request')
        for name,h in done['files'].items():
            if name not in ('qualification.json','context.json','shadow.json','score_result.json') or file_sha(directory/name)!=h:raise ValueError('worker output file pin differs')
        result=read(directory/'score_result.json');context=read(directory/'context.json');qualification=read(directory/'qualification.json')
        if digest(context)!=result['context_sha256'] or digest(result['selection'])!=result['selection_sha256']:raise ValueError('semantic score/context gate differs')
        if self.audit is None:audit=dict(ec_match=False,unknown=['independent_audit_not_configured'])
        else:
            audit_config=copy.deepcopy(self.worker_config)
            if self.ready is not None:audit_config['ticket_endpoint']=self.ready['ticket_endpoint']
            audit=self.audit(history=request['history'],context=copy.deepcopy(context),qualification=copy.deepcopy(qualification),config=audit_config,decision_dir=directory)
        audit_path=directory/'controller_independent_audit.json';write(audit_path,audit)
        enabled, final = record['enabled'], record['final']
        eligible=enabled and not final and audit.get('ec_match') is True and audit.get('unknown')==[]
        if eligible and result['selection']['selected_key'] is not None:
            record['execute_command']=dict(decision_id=decision_id,context_sha256=result['context_sha256'],selection_sha256=result['selection_sha256'],ec_match=True,ec_audit_sha256=file_sha(audit_path),audit_ref=str(audit_path))
        else:
            reason='closed_no_future_service' if final else ('policy_disabled' if not enabled else 'independent_snapshot_unknown_or_mismatch')
            if eligible:reason='no_selected_prefix'
            write(directory/'controller_not_executed.json',dict(reason=reason,audit_ref=str(audit_path)))
            record['no_execute_reason'] = reason
        return record

    def _dispatch(self, decision):
        command = decision.get('execute_command')
        if command is None or decision['execute_requested']:
            return
        path = Path(decision['directory'])/'execute.json'
        try:
            write(path, command)
        except FileExistsError:
            if read(path) != command:
                raise ValueError('conflicting execute command for this decision')
        decision['execute_requested'] = True

    def close(self):
        with self.lock:
            if self.closed:return
            self.closing=True
            errors=[]
            for attempt in list(self.pending_attempts):
                original=getattr(self,'terminal_intents',{}).get(attempt,
                    dict(attempt_id=attempt,status='cancelled'))
                try:self.terminal(terminal=original)
                except Exception as error:errors.append(failure('pending.terminal',error))
            # Stop future work and drain owned worker resources independently
            # of an unconfirmed remote terminal. Never rewrite success as abort.
            try:
                finished=dict(reason='ordinary group closed; no more execute',
                    decision_ids=[d['decision_id'] for d in self.decisions],
                    no_execute_reasons={d['decision_id']:d.get('no_execute_reason','no_execute') for d in self.decisions if not d['execute_requested']})
                marker=self.inbox/'finished.json'
                if marker.exists():
                    if read(marker)!=finished:raise ValueError('contradictory close marker')
                else:write(marker,finished)
                done=self._wait(self.inbox/'worker_done.json')
                drained=self.root/'worker_drained.json'
                if drained.exists():
                    if read(drained)!=done:raise ValueError('contradictory worker drain receipt')
                else:write(drained,done)
            except Exception as error:errors.append(failure('worker.drain',error))
            for name,resource in [('ticket',self.ticket),('http',self.http)]:
                flag='_'+name+'_closed'
                if resource is not None and not getattr(self,flag,False):
                    try:resource.close();setattr(self,flag,True)
                    except Exception as error:errors.append(failure(name+'.close',error))
            self.closed=not errors
            if errors:raise LifecycleError(errors)
