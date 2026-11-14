"""Shared provider/external-worker strategy -> at most one official warm.

Only explicit terminal-success history is accepted. No future trace reader.
"""
import argparse,importlib.util,json,os,threading,time,copy
from pathlib import Path
from bloombee.kv_cache.network.strategy_contract import digest,candidate_key,identity_scope,validate_context,validate_result

def validate_demand_ledger(history,boundary,causal_cutoff_ns):
    """Fail before ANY metadata query when terminal causality is unproved."""
    required={'truncated','consumer_id','window_id','initial_completion_seq','clock_domain_id','completion_sequence_semantics'}
    if not required<=boundary.keys() or type(boundary['truncated'])is not bool:raise ValueError('explicit full history boundary required')
    if boundary['completion_sequence_semantics']!='unique_successful_full_request_disclosure':raise ValueError('wrong completion sequence semantics')
    sequence=boundary['initial_completion_seq'];last_time=-1;seen={}
    for event in history:
        if event.get('outcome')!='processed_success' or not event.get('terminal_receipt_ref'):raise ValueError('full-request success receipt required')
        if event.get('consumer_instance_id')!=boundary['consumer_id'] or event.get('clock_domain_id')!=boundary['clock_domain_id']:raise ValueError('consumer or clock domain mismatch')
        if event['request_id'] in seen:
            if digest(seen[event['request_id']])!=digest(event):raise ValueError('conflicting duplicate request')
            continue
        if type(event['completion_event_seq'])is not int or event['completion_event_seq']!=sequence+1:raise ValueError('noncontiguous completion disclosure sequence')
        if type(event['disclosed_ns'])is not int or not last_time<=event['disclosed_ns']<=causal_cutoff_ns:raise ValueError('future or unordered disclosure')
        sequence=event['completion_event_seq'];last_time=event['disclosed_ns'];seen[event['request_id']]=event
    return sequence


def build_context(config,report,history,boundary,decision_id,causal_cutoff_ns):
    from bloombee.kv_cache.network.native_warm import layout_from_config
    from bloombee.kv_cache.network.snapshot_bridge import SnapshotLayout
    history=tuple(history)
    validate_demand_ledger(history,boundary,causal_cutoff_ns)
    if any(len(event.get('token_ids',[]))<int(config.get('prefix_length',16)) for event in history):raise ValueError('unrepresentable_short_terminal_demand; never drop or bridge')
    if report.get('history_snapshot_sha256')!=digest(history):raise ValueError('qualification/history snapshot mismatch')
    if boundary['consumer_id']!=config.get('candidate_consumer_id',config['lmcache_target']['instance_id']):raise ValueError('consumer boundary mismatch')
    current_model=None;candidates=[]
    for item in report['candidates']:
        candidate=dict(item);candidate['candidate_key']=candidate_key(item);candidate['identity_scope']=identity_scope(item);candidates.append(candidate);current_model=candidate['model_name']
    events=[];seen={}
    for raw in history:
        if raw.get('outcome')!='processed_success' or raw.get('consumer_instance_id')!=boundary['consumer_id']:raise ValueError('history not disclosed successful demand of this consumer')
        if raw.get('identities'):
            # Preserve explicit other-model scopes so a transition plugin can
            # break chains, rather than connecting across discarded events.
            n=int(config.get('prefix_length',16));layout=SnapshotLayout(raw['identities'],raw['token_ids'][:n])
        else:layout=layout_from_config(config,raw['token_ids'])
        descriptor=dict(model_name=layout.model_name,cache_salt=layout.cache_salt,prefix_length=len(layout.tokens),prefix_token_ids=list(layout.tokens))
        event=dict(candidate_key=candidate_key(descriptor),identity_scope=identity_scope(descriptor),completion_event_seq=raw['completion_event_seq'],request_id=raw['request_id'],disclosed_ns=raw['disclosed_ns'],consumer_id=raw['consumer_instance_id'],clock_domain_id=raw['clock_domain_id'],outcome=raw['outcome'],terminal_receipt_ref=raw['terminal_receipt_ref'],history_source=raw['history_source'],visibility_scope=raw['visibility_scope'])
        if event['request_id'] in seen:
            if seen[event['request_id']]!=event:raise ValueError('conflicting replay of same terminal request')
            continue
        seen[event['request_id']]=event;events.append(event)
    # Static namespace matches SnapshotLayout, without inventing token inputs.
    variable={'layer_start','layer_end','prefix_tokens_digest','prefix_length','position_digest'}
    static={k:v for k,v in config['identity_template'].items() if k not in variable}
    expected_model='bb_native_'+digest(dict(static=static,layers=int(config.get('native_layers',32))))
    if current_model is not None and current_model!=expected_model:raise ValueError('C model namespace differs from provider')
    scope=identity_scope(dict(model_name=expected_model,prefix_length=int(config.get('prefix_length',16))))
    unknown=report.get('status')=='unknown' or bool(report.get('concurrent_changed_emitters')) or any(any(word in str(row.get('reason','')) for word in ('unknown','stale','incomplete_key_fanout')) for row in report.get('decisions',[]))
    context=dict(schema_version=1,decision_id=decision_id,decision_seq=events[-1]['completion_event_seq'] if events else boundary['initial_completion_seq'],identity_scope=scope,candidates=candidates,completed_history=events,causal_cutoff_ns=causal_cutoff_ns,history_boundary=dict(boundary,raw_records=len(history),unique_full_requests=len(events)),qualification_status='unknown' if unknown else 'qualified',qualification_sha256=digest(report),history_snapshot_sha256=digest(history))
    return validate_context(context)

def score_context(context,strategy,plugin=None):
    validate_context(context)
    if strategy=='lrfu':
        from bloombee.kv_cache.network.lrfu_policy import score_candidates
    elif strategy=='transition':
        if not plugin:raise ValueError('transition plugin path/SHA required')
        path=Path(plugin['path']);actual=__import__('hashlib').sha256(path.read_bytes()).hexdigest()
        if actual!=plugin['sha256']:raise ValueError('transition plugin SHA mismatch')
        spec=importlib.util.spec_from_file_location('pinned_transition_policy',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);score_candidates=module.score_candidates
    else:raise ValueError('only fixed lrfu/transition options supported')
    result=validate_result(context,score_candidates(context))
    if result.get('strategy')!=strategy:raise ValueError('plugin strategy mismatch')
    return result

class StrategyRunner:
    def __init__(self,config,emit=None):
        self.config=config;self.emit=emit or (lambda event,**data:None);self.lock=threading.Lock();self.warm_lock=threading.Lock();self.geometries={}
    def invalid_result(self,decision_id,reason,error=None,report=None):
        scores=[]
        for candidate in (report or {}).get('candidates',[]):
            scores.append(dict(candidate_key=candidate_key(candidate),score=0.,reason=reason,evidence={}))
        return dict(schema_version=1,strategy=self.config.get('prefetch_strategy'),decision_id=decision_id,scores=scores,selected_key=None,abstain_reason=reason,error=error,warm_attempts=0,qualification_status='unknown' if report is not None else 'not_queried')
    def qualify(self,history):
        import httpx
        from bloombee.kv_cache.network.history_candidates import HistoryCandidateFilter
        with httpx.Client(trust_env=False,timeout=self.config.get('native_timeout_s',60)) as client:return HistoryCandidateFilter(self.config,client=client).filter(history)
    def decide(self,history,boundary,decision_id,causal_cutoff_ns,*,warm=False,report=None):
        from bloombee.kv_cache.network.history_candidates import HistoryCandidateFilter
        with self.lock:
            started=time.monotonic_ns();history=tuple(history)
            try:
                validate_demand_ledger(history,boundary,causal_cutoff_ns)
                if any(len(event.get('token_ids',[]))<int(self.config.get('prefix_length',16)) for event in history):
                    self.emit('strategy_abstain',decision_id=decision_id,reason='unrepresentable_short_terminal_demand')
                    return self.invalid_result(decision_id,'unrepresentable_short_terminal_demand')
            except (KeyError,TypeError,ValueError) as error:
                self.emit('strategy_abstain',decision_id=decision_id,reason='invalid_terminal_history_before_qualification',error=str(error))
                return self.invalid_result(decision_id,'invalid_terminal_history_before_qualification',str(error))
            report=report if report is not None else self.qualify(history)
            try:
                context=build_context(self.config,report,history,boundary,decision_id,causal_cutoff_ns)
                result=score_context(context,self.config['prefetch_strategy'],self.config.get('strategy_transition_plugin'))
            except (KeyError,TypeError,ValueError) as error:
                self.emit('strategy_abstain',decision_id=decision_id,reason='invalid_context_or_plugin',error=str(error),qualification=report)
                return self.invalid_result(decision_id,'invalid_context_or_plugin',str(error),report)
            result.update(distinct_choice_possible=len(context['candidates'])>1,context_sha256=digest(context),qualification_sha256=digest(report),history_boundary=context['history_boundary'],warm_attempts=0)
            try:
                from bloombee.kv_cache.network.clock_domain import current
                execution_clock_domain=current(self.config)
            except (OSError,ValueError):execution_clock_domain=None
            self.emit('strategy_score',execution_clock_domain=execution_clock_domain,decision_id=decision_id,context=context,result=result,started_ns=started,completed_ns=time.monotonic_ns(),target=self.config['lmcache_target'],remote_rules=self.config['candidate_remote_placements'])
            if not warm or result['selected_key'] is None:return result
            return self.warm_selected(context,result,history)

    def warm_selected(self,context,selected_result,history):
        """Warm an already-scored choice. No ranking or new history observed.

        Caller may submit this method to its serial warm executor and immediately
        dispatch the next ordinary model request. Pure scoring is independent of
        this warm_lock. The supplied shadow/context are never mutated.
        """
        with self.warm_lock:
            validate_context(context);validate_result(context,selected_result)
            if selected_result['strategy']!=self.config['prefetch_strategy']:raise ValueError('arm strategy mismatch')
            if selected_result.get('context_sha256',digest(context))!=digest(context):raise ValueError('scored context digest mismatch')
            history=tuple(history);boundary=context['history_boundary'];decision_id=context['decision_id'];causal_cutoff_ns=context['causal_cutoff_ns']
            validate_demand_ledger(history,boundary,causal_cutoff_ns)
            if digest(history)!=context['history_snapshot_sha256']:raise ValueError('warm history differs from scored history')
            result=copy.deepcopy(selected_result)
            result.update(context_sha256=digest(context),ranked_selected_key=result['selected_key'],warm_attempts=0)
            if result['selected_key'] is None:return result
            # Recheck qualification only; never pick a replacement or tune rank.
            try:
                recheck=self.qualify(history)
                fresh_context=build_context(self.config,recheck,history,boundary,decision_id,causal_cutoff_ns)
            except (KeyError,TypeError,ValueError) as error:
                result.update(selected_key=None,abstain_reason='qualification_recheck_invalid',error=str(error));self.emit('strategy_abstain',decision_id=decision_id,result=result);return result
            eligible={c['candidate_key']:c for c in fresh_context['candidates']}
            if fresh_context['qualification_status']!='qualified' or result['selected_key'] not in eligible:
                result.update(selected_key=None,abstain_reason='selected_no_longer_qualified')
                self.emit('strategy_abstain',decision_id=decision_id,result=result,recheck=recheck);return result
            candidate=eligible[result['selected_key']]
            gate=Path(self.config['strategy_attempt_dir']);gate.mkdir(parents=True,exist_ok=True)
            attempt=gate/(digest(dict(run_id=self.config['strategy_run_id'],decision_id=decision_id))+'.json')
            try:
                with attempt.open('x') as f:json.dump(dict(decision_id=decision_id,strategy=self.config['prefetch_strategy'],candidate_key=result['selected_key'],context_sha256=digest(context)),f)
            except FileExistsError:
                result.update(selected_key=None,abstain_reason='decision_already_attempted');return result
            from bloombee.kv_cache.network.native_warm import layout_from_config,warm_registered,receipt_digest
            from bloombee.kv_cache.network.native_probe import Geometry
            import httpx
            try:
                layout=layout_from_config(self.config,candidate['prefix_token_ids'])
                if layout.model_name!=candidate['model_name'] or layout.cache_salt!=candidate['cache_salt']:raise ValueError('warm identity changed')
                if layout.model_name not in self.geometries:self.geometries[layout.model_name]=Geometry(self.config['lmcache_target'],layout.model_name,len(layout.identities),layout.heads,layout.dim,layout.dtype)
                self.emit('strategy_warm_submit',decision_id=decision_id,strategy=self.config['prefetch_strategy'],candidate_key=result['selected_key'],target=self.config['lmcache_target'],context_sha256=digest(context),qualification_recheck=recheck,monotonic_ns=time.monotonic_ns())
                result['warm_attempts']=1
                with httpx.Client(trust_env=False,timeout=self.config.get('native_timeout_s',60)) as client:receipt=warm_registered(client,self.config,layout,require_empty=False,reason='qualified_'+self.config['prefetch_strategy'])
                receipt['strategy_link']=dict(decision_id=decision_id,strategy=self.config['prefetch_strategy'],candidate_key=result['selected_key'],context_sha256=digest(context),qualification_sha256=digest(recheck))
                receipt['receipt_sha256']=receipt_digest(receipt)
                with Path(self.config['strategy_warm_receipts_path']).open('a') as f:f.write(json.dumps(receipt,sort_keys=True)+'\n')
                result['warm_receipt']=receipt
                self.emit('strategy_warm_complete',decision_id=decision_id,receipt=receipt)
            except Exception as error:
                result.update(warm_error_type=type(error).__name__,warm_error=str(error));self.emit('strategy_warm_failed',decision_id=decision_id,result=result)
            return result
    def close(self):
        for geometry in self.geometries.values():geometry.close()
        self.geometries.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--history-json',required=True);p.add_argument('--boundary-json',required=True);p.add_argument('--decision-id',required=True);p.add_argument('--causal-cutoff-ns',type=int,required=True);p.add_argument('--warm',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    def emit(event,**data):
        with Path(config['strategy_events_path']).open('a') as f:f.write(json.dumps(dict(event=event,**data),sort_keys=True)+'\n')
    runner=StrategyRunner(config,emit)
    try:
        result=runner.decide(json.loads(Path(a.history_json).read_text()),json.loads(Path(a.boundary_json).read_text()),a.decision_id,a.causal_cutoff_ns,warm=a.warm)
        with Path(a.out).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    finally:runner.close()
