"""Pure contract/regression assertions, never a synthetic trace efficacy result."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_transition_contract():
    from bloombee.kv_cache.network import strategy_contract as contract
    from bloombee.kv_cache.network import transition_policy as policy
    def candidate(label):
        item=dict(model_name='model_共享',cache_salt=label,prefix_length=16,prefix_token_ids=list(range(16)))
        item['candidate_key']=contract.candidate_key(item);item['identity_scope']=contract.identity_scope(item)
        return item
    candidates={label:candidate(label) for label in ('A','B','C','X')}
    scope=candidates['A']['identity_scope']
    def context(sequence,eligible=('B','C')):
        events=[dict(candidate_key=candidates[label]['candidate_key'],identity_scope=scope,
            completion_event_seq=i,request_id=f'request_{i}',disclosed_ns=i,
            history_source='unit_boundary_assertion_only',visibility_scope='consumer_a',
            consumer_id='consumer_a',clock_domain_id='domain_a',outcome='processed_success',
            terminal_receipt_ref=f'unit_terminal_{i}') for i,label in enumerate(sequence,1)]
        return dict(schema_version=1,decision_id='decision_boundary',decision_seq=len(events),
            identity_scope=scope,candidates=[candidates[label] for label in eligible],
            completed_history=events,causal_cutoff_ns=1000,qualification_status='qualified',
            history_boundary=dict(truncated=False,consumer_id='consumer_a',window_id='unit_only',
                clock_domain_id='domain_a',initial_completion_seq=0,
                completion_sequence_semantics='unique_successful_full_request_disclosure'))
    def checked(ctx):
        before=copy.deepcopy(ctx);contract.validate_context(ctx)
        output=policy.score_candidates(ctx);contract.validate_result(ctx,output)
        assert ctx==before
        return output
    result=checked(context(['A','B','A','B','A']))
    assert result['selected_key']==candidates['B']['candidate_key']
    assert all(row['evidence']['all_outgoing_count']==2 for row in result['scores'])
    low=checked(context(['A','B','A']))
    assert low['selected_key'] is None and low['abstain_reason']=='selected_edge_below_min_support'
    outside=checked(context(['A','B','A','B','A']+['X','A']*6))
    assert outside['selected_key'] is None and outside['abstain_reason']=='selected_edge_below_min_confidence'
    b=next(row for row in outside['scores'] if row['candidate_key']==candidates['B']['candidate_key'])
    assert b['score']==.25 and b['evidence']['confidence_denominator']==8 and b['evidence']['outside_C_outgoing_count']==6
    tie=checked(context(['A','B','A','B','A','C','A','C','A']))
    assert tie['selected_key']==min(candidates['B']['candidate_key'],candidates['C']['candidate_key'])
    reset=context(['A','B','A','B','A','X','A'])
    reset['completed_history'][-2]['identity_scope']='different_model'
    assert checked(reset)['abstain_reason']=='no_observed_outgoing_transition'
    truncated=context(['A','B','A','B','A']);truncated['history_boundary']['truncated']=True
    assert checked(truncated)['abstain_reason']=='truncated_history_denominator_unknown'
    unknown=context(['A']);unknown['qualification_status']='unknown'
    assert checked(unknown)['abstain_reason']=='qualification_unknown'
    empty=context(['A'],eligible=())
    assert checked(empty)['abstain_reason']=='empty_C'
    assert checked(context([]))['abstain_reason']=='no_completed_history'
    for mutate in ('future','duplicate','failed_terminal','out_of_order'):
        invalid=context(['A','B','A','B','A'])
        if mutate=='future':invalid['completed_history'][-1]['disclosed_ns']=1001
        elif mutate=='duplicate':invalid['completed_history'][-1]['request_id']='request_1'
        elif mutate=='failed_terminal':invalid['completed_history'][-1]['outcome']='failed'
        else:invalid['completed_history'][1],invalid['completed_history'][2]=invalid['completed_history'][2],invalid['completed_history'][1]
        assert policy.score_candidates(invalid)['selected_key'] is None
