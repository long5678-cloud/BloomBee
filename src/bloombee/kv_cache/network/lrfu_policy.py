"""Fixed event-half-life4 scoring, reused placement_j2::_heat_scores formula.

Only successful full demand disclosures count. No cache-hit/access statistics,
wall-time window, parameter sweep or future request information.
"""
import math
from bloombee.kv_cache.network.strategy_contract import validate_context,validate_result,abstain

def score_candidates(context):
    validate_context(context);parameters=dict(half_life_success_events=4,wall_clock_window=None,tie_break='candidate_key_ascending')
    if context['qualification_status']!='qualified':return abstain(context,'lrfu','qualification_unknown',parameters=parameters)
    if not context['candidates']:return abstain(context,'lrfu','empty_C',parameters=parameters)
    n=context['decision_seq'];scores=[]
    for candidate in context['candidates']:
        events=[e for e in context['completed_history'] if e['identity_scope']==context['identity_scope'] and e['candidate_key']==candidate['candidate_key']]
        terms=[dict(request_id=e['request_id'],completion_event_seq=e['completion_event_seq'],age=n-e['completion_event_seq'],weight=2.0 ** (-(n-e['completion_event_seq'])/4.)) for e in events]
        scores.append(dict(candidate_key=candidate['candidate_key'],score=math.fsum(t['weight'] for t in terms),reason='fixed_event_half_life4_demand_sum',evidence=dict(successful_demand_terms=terms,observations=len(terms))))
    ordered=sorted(scores,key=lambda row:(-row['score'],row['candidate_key']))
    if ordered[0]['score']<=0:return abstain(context,'lrfu','no_matching_successful_demand',scores,parameters)
    return validate_result(context,dict(schema_version=1,strategy='lrfu',decision_id=context['decision_id'],scores=scores,selected_key=ordered[0]['candidate_key'],abstain_reason=None,parameters=parameters,distinct_choice_possible=len(scores)>1))
