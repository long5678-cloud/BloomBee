"""Causal completed-prefix transition option, NOT a CacheScout agent replica.

Pure standard library. Qualification and real LMCache warm belong to the common
provider. The strategy never queries a directory, changes C, or sees a future row.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math

MIN_SUPPORT=2
MIN_CONFIDENCE=Fraction(1,2)
PARAMETERS=dict(min_edge_support=MIN_SUPPORT,min_confidence_numerator=1,
    min_confidence_denominator=2,tie_break='candidate_key_lexicographic_ascending',
    denominator='all outgoing transitions in supplied causal completed-history corpus, including destinations outside C',
    scope_reset='use only the current identity-scope contiguous completion suffix',
    history_order='completion_event_seq; never source arrival order or cache access',
    smoothing='none',fallback='abstain; no LRFU fallback',warm_budget='at most one selected complete prefix')


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def candidate_key(candidate):
    return canonical({key:candidate[key] for key in ('model_name','cache_salt','prefix_length','prefix_token_ids')})


def score_candidates(context):
    """Shared schema-v1 plugin entry; does not mutate its input or retain state."""
    candidates=context.get('candidates',[]) if isinstance(context,dict) else []
    decision_id=context.get('decision_id') if isinstance(context,dict) else None
    scores=[]
    for candidate in candidates:
        if isinstance(candidate,dict) and isinstance(candidate.get('candidate_key'),str):
            scores.append(dict(candidate_key=candidate['candidate_key'],score=0.0,
                               reason='not_scored',evidence={}))
    result=dict(schema_version=1,strategy='transition',decision_id=decision_id,
                scores=scores,selected_key=None,abstain_reason=None,parameters=dict(PARAMETERS))

    def abstain(reason,evidence=None):
        result['abstain_reason']=reason
        for score in result['scores']:
            score['reason']=reason
            if evidence is not None:score['evidence'].update(evidence)
        return result

    if not isinstance(context,dict) or context.get('schema_version')!=1:
        return abstain('invalid_context_schema')
    if context.get('qualification_status')!='qualified':return abstain('qualification_unknown')
    if not candidates:return abstain('empty_C')
    if not isinstance(decision_id,str) or not decision_id:return abstain('invalid_decision_id')
    seq=context.get('decision_seq');cutoff=context.get('causal_cutoff_ns');scope=context.get('identity_scope')
    if type(seq) is not int or seq<0 or type(cutoff) is not int or cutoff<0 or not isinstance(scope,str) or not scope:
        return abstain('invalid_causal_boundary')
    keys=[]
    try:
        for candidate in candidates:
            key=candidate['candidate_key'];tokens=candidate['prefix_token_ids'];length=candidate['prefix_length']
            if (not isinstance(key,str) or candidate['identity_scope']!=scope or candidate_key(candidate)!=key
                    or type(length) is not int or length<=0 or length%16
                    or not isinstance(tokens,list) or len(tokens)!=length
                    or any(type(t)is not int or t<0 for t in tokens)):
                return abstain('invalid_or_mixed_C')
            keys.append(key)
    except (KeyError,TypeError,ValueError):return abstain('invalid_or_mixed_C')
    if len(keys)!=len(set(keys)):return abstain('duplicate_C_key')
    history=context.get('completed_history')
    if not isinstance(history,list):return abstain('missing_completed_history')
    boundary=context.get('history_boundary') or {}
    required={'truncated','consumer_id','window_id','clock_domain_id','initial_completion_seq','completion_sequence_semantics'}
    if not required<=boundary.keys() or boundary['completion_sequence_semantics']!='unique_successful_full_request_disclosure':
        return abstain('missing_or_invalid_history_boundary')
    if boundary['truncated'] is not False:return abstain('truncated_history_denominator_unknown')
    prior=boundary['initial_completion_seq']
    if type(prior) is not int or prior<0:return abstain('invalid_initial_completion_seq')
    if not history:return abstain('no_completed_history')
    seen_seq=set();seen_requests=set()
    try:
        for event in history:
            event_seq=event['completion_event_seq'];disclosed=event['disclosed_ns']
            request=event['request_id'];key=event['candidate_key']
            if (type(event_seq) is not int or event_seq<0 or event_seq>seq
                    or event_seq!=prior+1
                    or type(disclosed) is not int or disclosed<0 or disclosed>cutoff
                    or not isinstance(request,str) or not request or not isinstance(key,str) or not key
                    or not event['history_source'] or not event['visibility_scope']
                    or not isinstance(event['identity_scope'],str) or not event['identity_scope']
                    or event.get('consumer_id')!=boundary['consumer_id'] or event.get('clock_domain_id')!=boundary['clock_domain_id']
                    or event.get('outcome')!='processed_success' or not event.get('terminal_receipt_ref')
                    or event_seq in seen_seq or request in seen_requests):
                return abstain('invalid_future_or_duplicate_completion_history')
            seen_seq.add(event_seq);seen_requests.add(request)
            prior=event_seq
    except (KeyError,TypeError):return abstain('invalid_completion_history')
    if max(seen_seq)!=seq:return abstain('decision_seq_does_not_match_last_visible_completion')
    ordered=sorted(history,key=lambda e:e['completion_event_seq'])
    # Never bridge S -> other model/execution -> S as an observed S -> S edge.
    suffix=[]
    for event in reversed(ordered):
        if event['identity_scope']!=scope:break
        suffix.append(event)
    suffix.reverse()
    if not suffix:return abstain('no_current_scope_predecessor')
    predecessor=suffix[-1]['candidate_key']
    outgoing=Counter(right['candidate_key'] for left,right in zip(suffix,suffix[1:])
                     if left['candidate_key']==predecessor)
    denominator=sum(outgoing.values())
    evidence=dict(predecessor_key=predecessor,all_outgoing_count=denominator,
        outside_C_outgoing_count=sum(count for key,count in outgoing.items() if key not in set(keys)),
        visible_completed_events=len(history),current_scope_suffix_events=len(suffix),
        first_suffix_completion_seq=suffix[0]['completion_event_seq'],last_completion_seq=seq,
        causal_cutoff_ns=cutoff,history_boundary=context.get('history_boundary','caller supplied causal corpus; no implicit strategy truncation'))
    if denominator==0:return abstain('no_observed_outgoing_transition',evidence)
    by_key={score['candidate_key']:score for score in scores}
    for key in keys:
        count=outgoing[key];probability=Fraction(count,denominator)
        by_key[key].update(score=float(probability),reason='empirical_completed_prefix_transition',
            evidence=dict(evidence,edge_support=count,confidence_numerator=count,confidence_denominator=denominator))
        assert math.isfinite(by_key[key]['score'])
    winner=min(keys,key=lambda key:(-outgoing[key],key))
    support=outgoing[winner];confidence=Fraction(support,denominator)
    if support<MIN_SUPPORT:
        result['abstain_reason']='selected_edge_below_min_support'
    elif confidence<MIN_CONFIDENCE:
        result['abstain_reason']='selected_edge_below_min_confidence'
    else:
        result['selected_key']=winner
    return result
