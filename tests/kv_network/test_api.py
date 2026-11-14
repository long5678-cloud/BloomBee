"""Packaged API contracts; model doubles here are wiring tests only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from contextlib import nullcontext
import pytest
from bloombee.kv_cache.network.api import Options, Components, score
from bloombee.kv_cache.network import strategy_contract as contract


def context(sequence=('A','B','A','B','A'), eligible=('B','C')):
    def candidate(label):
        row=dict(model_name='model',cache_salt=label,prefix_length=16,prefix_token_ids=list(range(16)))
        row.update(candidate_key=contract.candidate_key(row),identity_scope=contract.identity_scope(row))
        return row
    candidates={k:candidate(k) for k in ('A','B','C','X')}
    scope=candidates['A']['identity_scope']
    events=[dict(candidate_key=candidates[k]['candidate_key'],identity_scope=scope,
            completion_event_seq=i,request_id='r'+str(i),disclosed_ns=i,
            consumer_id='consumer',clock_domain_id='clock',outcome='processed_success',
            terminal_receipt_ref='fixture',history_source='test',visibility_scope='consumer')
            for i,k in enumerate(sequence,1)]
    return dict(schema_version=1,decision_id='decision',decision_seq=len(events),identity_scope=scope,
        candidates=[candidates[k] for k in eligible],completed_history=events,causal_cutoff_ns=100,
        qualification_status='qualified',history_boundary=dict(truncated=False,consumer_id='consumer',
        window_id='window',clock_domain_id='clock',initial_completion_seq=0,
        completion_sequence_semantics='unique_successful_full_request_disclosure'))


def test_disabled_import_no_components():
    c=Components()
    assert c.local is c.telemetry is c.prefetch is None
    c.close()
    program='''import sys
from bloombee.kv_cache.network.api import Components, Options
Components()
Components(Options(local="lru"), local_kwargs=dict(capacity_bytes=0))
assert not any(k in sys.modules for k in ['torch','transformers','hivemind','lmcache'])
'''
    subprocess.run([sys.executable,'-c',program],check=True)

@pytest.mark.parametrize('policy',['lru','hotness','decay'])
def test_local_factory_does_not_start_optional_components(policy):
    c=Components(Options(local=policy),local_kwargs=dict(capacity_bytes=100))
    assert c.local.policy==policy and c.telemetry is c.prefetch is None

@pytest.mark.parametrize('bad',[dict(local='DINO'),dict(prefetch='best'),dict(telemetry=1)])
def test_invalid_options(bad):
    with pytest.raises((ValueError,TypeError)):Options(**bad)


def test_pure_scoring_both_options_and_fixed_lrfu_formula():
    c=context();before=copy.deepcopy(c)
    l=score(c,'lrfu');t=score(c,'transition')
    expected=sum(2**(-(5-i)/4) for i in (2,4))
    assert l['selected_key']==t['selected_key']
    assert next(r['score'] for r in l['scores'] if r['candidate_key']==l['selected_key'])==pytest.approx(expected)
    assert c==before
    for strategy in ('lrfu','transition'):
        empty=context(eligible=())
        assert score(empty,strategy)['selected_key'] is None
        unknown=context();unknown['qualification_status']='unknown'
        assert score(unknown,strategy)['selected_key'] is None
        bad=context();bad['completed_history'][0]['disclosed_ns']=101
        with pytest.raises(ValueError):score(bad,strategy)


def test_ordinary_cold_and_cache_session_wiring(tmp_path):
    import torch
    from bloombee.kv_cache.network.ordinary import OrdinaryClient
    class Model:
        def prepare_inputs_for_generation(self,*args,**kwargs):return {}
        def inference_session(self,**kwargs):self.options=kwargs;return nullcontext()
        def generate(self,ids,**kwargs):return torch.cat([ids,torch.tensor([[9]*kwargs['max_new_tokens']])],dim=1)
    model=Model();ids=torch.tensor([list(range(17))])
    with OrdinaryClient(model,output_root=tmp_path/'cold') as client:
        out,terminal=client.generate(ids,max_new_tokens=3,request_id='r')
        assert out.shape==(1,20) and terminal['status']=='complete_success'
        assert 'dynamo_cache' not in model.options
        with pytest.raises(ValueError):client.generate(ids,max_new_tokens=3,request_id='r')
    with OrdinaryClient(model,output_root=tmp_path/'local',context='identity') as client:
        client.generate(ids,max_new_tokens=3)
        assert model.options['dynamo_cache']['token_ids']==list(range(17))
        assert len(client.history)==1


def test_ordinary_failure_preserves_actual_outcome(tmp_path):
    import torch
    from bloombee.kv_cache.network.ordinary import OrdinaryClient
    class Model:
        def prepare_inputs_for_generation(self,*args,**kwargs):return {}
        def inference_session(self,**kwargs):return nullcontext()
        def generate(self,*args,**kwargs):raise RuntimeError('actual model failure')
    with OrdinaryClient(Model(),output_root=tmp_path/'failure') as client:
        with pytest.raises(RuntimeError,match='actual model failure'):
            client.generate(torch.tensor([list(range(17))]),max_new_tokens=3)
        assert not client.history
        receipt=json.loads(next((tmp_path/'failure').glob('*.json')).read_text())
        assert receipt['status']=='failed' and receipt['actual_output_tokens']==0


def test_completion_adapter_explicit_node_does_not_evaluate_experiment_gpu_job(tmp_path):
    from bloombee.kv_cache.network.completion import Adapter
    seq=SimpleNamespace(dht=object(),block_uids=['layer.'+str(i) for i in range(32)])
    model=SimpleNamespace(transformer=SimpleNamespace(h=SimpleNamespace(sequence_manager=seq)))
    adapter=Adapter(model=model,job={},config=dict(telemetry_namespace='test',consumer_node_id='consumer',
        distributed_telemetry=False),output_root=tmp_path)
    assert adapter.publisher is None and adapter.prefetch is None
    assert adapter.telemetry.publisher.ref.node_id=='consumer'
    adapter.close()
