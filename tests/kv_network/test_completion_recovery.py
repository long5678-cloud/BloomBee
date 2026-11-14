"""Cross-module recovery: exact-once telemetry and interrupted Prefetch disclosure."""
import json
import time
import pytest
import torch
from test_review_regressions import archived, Model
from test_background_dispatch import run
from bloombee.kv_cache.network.ordinary import OrdinaryClient
from bloombee.kv_cache.network.lifecycle import LifecycleError

def test_measured_reuse_not_duplicated_on_local_retry(tmp_path,monkeypatch):
    log=tmp_path/'provider.log';log.touch()
    rows, prompt, suffix=archived()
    class Reuse(Model):
        def generate(self,ids,**kw):
            with log.open('a') as file:
                for row in rows:
                    file.write('DYNAMO_SERVICE '+json.dumps(dict(row,telemetry_attempt_id=self.options['dynamo_cache']['attempt_id']))+'\n')
            return torch.tensor([prompt+suffix])
    cfg=dict(distributed_telemetry=True,telemetry_namespace='test',consumer_node_id='consumer',provider_log=str(log),control_endpoint='stub')
    with OrdinaryClient(Reuse(log),context='ctx',config=cfg,output_root=tmp_path/'client') as c:
        posts=[]
        def post(endpoint,payload):
            posts.append(payload.copy())
            if len(posts)==1:raise OSError('ack lost')
        monkeypatch.setattr(c,'_post_page_terminal',post)
        monkeypatch.setattr(c.adapter.publisher,'publish',lambda **kw:dict(observed_until=time.time()))
        with pytest.raises(LifecycleError):
            c.generate(torch.tensor([prompt]),request_id='r1',max_new_tokens=3)
        c.generate(torch.tensor([prompt]),request_id='r2',purpose='warmup',max_new_tokens=3)
        assert c._pending_local_terminal is None and posts[0]==posts[1] and len(posts)==3
        assert c.attempt_sequence==len(c.history)==2 and all(e['kv_reuse_success'] for e in c.history)
        assert [(e.request_id,e.kind) for e in c.adapter.publisher._events.values()]==[('r1','reuse_success')]

def test_disclosure_recovers_after_score_wait_interruption(run):
    original=run.c._wait
    def interrupted(path):
        if path.name=='score_done.json':raise TimeoutError('one transient score response timeout')
        return original(path)
    run.c._wait=interrupted
    with pytest.raises(TimeoutError):
        run.disclose()
    run.c._wait=original
    # Re-delivering the same completed request must resume the original decision.
    run.disclose()
    run.disclose()
    assert len(run.c.history)==len(run.c.decisions)==1
    assert len(list(run.c.inbox.glob('*/execute.json')))==1
