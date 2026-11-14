"""Pure-stdlib common score/abstain protocol, no cache or network access."""
import hashlib,json,math

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def candidate_key(candidate):
    return digest({k:candidate[k] for k in ('model_name','cache_salt','prefix_length','prefix_token_ids')})

def identity_scope(candidate):
    return digest(dict(model_name=candidate['model_name'],prefix_length=candidate['prefix_length'],world_size=1,object_group_id=0))

def validate_context(context):
    if context.get('schema_version')!=1 or (not isinstance(context.get('decision_id'),str) or not context['decision_id']):raise ValueError('invalid context schema/decision')
    boundary=context.get('history_boundary',{})
    if not {'truncated','consumer_id','window_id','completion_sequence_semantics','clock_domain_id','initial_completion_seq'}<=boundary.keys():raise ValueError('history boundary/provenance missing')
    if boundary['completion_sequence_semantics']!='unique_successful_full_request_disclosure':raise ValueError('history is not full successful demand disclosure')
    n=context['decision_seq'];cutoff=context['causal_cutoff_ns']
    if type(n)is not int or n<0 or type(cutoff)is not int:raise ValueError('invalid event boundary')
    if type(boundary['truncated'])is not bool:raise ValueError('truncated must be explicit bool')
    ids=set();seqs=set();prior=boundary['initial_completion_seq'];previous_disclosed=-1
    for event in context['completed_history']:
        seq=event['completion_event_seq']
        if event['request_id'] in ids or seq in seqs:raise ValueError('duplicate full request or completion sequence')
        if type(seq)is not int or seq!=prior+1 or seq>n or type(event['disclosed_ns'])is not int or not previous_disclosed<=event['disclosed_ns']<=cutoff:raise ValueError('future/unordered completion disclosure')
        if event.get('consumer_id')!=boundary['consumer_id'] or event.get('clock_domain_id')!=boundary['clock_domain_id']:raise ValueError('consumer or disclosure clock domain mismatch')
        if event.get('outcome')!='processed_success' or not event.get('terminal_receipt_ref'):raise ValueError('missing full-request terminal success evidence')
        if not event.get('candidate_key') or not event.get('identity_scope'):raise ValueError('missing exact prefix identity')
        ids.add(event['request_id']);seqs.add(seq);prior=seq;previous_disclosed=event['disclosed_ns']
    if prior!=n:raise ValueError('decision_seq does not match latest disclosed success')
    keys=[]
    for candidate in context['candidates']:
        if candidate['candidate_key']!=candidate_key(candidate) or candidate['identity_scope']!=identity_scope(candidate):raise ValueError('candidate key/scope mismatch')
        if candidate['identity_scope']!=context['identity_scope']:raise ValueError('candidate outside current model/prefix scope')
        keys.append(candidate['candidate_key'])
    if len(set(keys))!=len(keys):raise ValueError('C must be deduplicated')
    return context

def abstain(context,strategy,reason,scores=None,parameters=None):
    return dict(schema_version=1,strategy=strategy,decision_id=context['decision_id'],scores=scores or [dict(candidate_key=c['candidate_key'],score=0.,reason=reason,evidence={}) for c in context['candidates']],selected_key=None,abstain_reason=reason,parameters=parameters or {})

def validate_result(context,result):
    keys={c['candidate_key'] for c in context['candidates']}
    if result.get('schema_version')!=1 or result.get('decision_id')!=context['decision_id']:raise ValueError('result decision mismatch')
    rows=result['scores'];reported=[r['candidate_key'] for r in rows]
    if len(reported)!=len(keys) or set(reported)!=keys:raise ValueError('score must cover C exactly once')
    for row in rows:
        if isinstance(row['score'],bool) or not isinstance(row['score'],(int,float)) or not math.isfinite(row['score']) or row['score']<0:raise ValueError('invalid score')
        if not isinstance(row['reason'],str) or not isinstance(row['evidence'],dict):raise ValueError('score evidence missing')
    selected=result['selected_key']
    if selected is not None and (selected not in keys or result.get('abstain_reason') is not None):raise ValueError('selected key outside C or inconsistent abstention')
    if selected is None and not result.get('abstain_reason'):raise ValueError('abstention reason required')
    if context['qualification_status']!='qualified' and selected is not None:raise ValueError('cannot select from unknown qualification')
    return result
