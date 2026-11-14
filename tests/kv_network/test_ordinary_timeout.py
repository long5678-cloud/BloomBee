"""CPU coverage of the ordinary request lifecycle and real session retry/timeout loops."""
import asyncio
import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bloombee.client.config import ClientConfig
from bloombee.client.inference_session import InferenceSession, _ServerInferenceSession
from bloombee.kv_cache.network.ordinary import OrdinaryClient
from test_review_regressions import Model, archived


@pytest.mark.parametrize('budget', [1, 2])
def test_timeout_exhausts_span_attempts_then_next_request_completes(tmp_path, monkeypatch, budget):
    log = tmp_path / 'provider.log'
    log.touch()
    calls = []
    closed = []
    failures = []
    attempt_ids = []

    class Manager:
        config = ClientConfig(max_retries=budget, request_timeout=0.01)
        block_uids = [row['uid'] for row in archived()[0]]
        dht = None

        def __len__(self):
            return len(self.block_uids)

        def make_sequence(self, start, end, **kwargs):
            return [SimpleNamespace(start=start, end=end, peer_id='test-peer')]

        def on_request_failure(self, peer):
            failures.append(peer)

        def on_request_success(self, peer):
            pass

        def get_retry_delay(self, attempt_no):
            return 0

    manager = Manager()

    async def stalled_response():
        await asyncio.Event().wait()
        yield None

    def connect(session, spans):
        # Replace network setup and peer payloads, keeping routing retry and wait_for.
        attempt = session._dynamo_cache['attempt_id']

        def step(inputs, *args, step_id):
            calls.append((attempt, step_id))
            if len(calls) > budget + 1 or calls.count((attempt, step_id)) > budget:
                pytest.fail('retry loop exceeded its configured attempt budget')
            if attempt == attempt_ids[0]:
                rpc = SimpleNamespace(config=manager.config, _inputs_queue=asyncio.Queue(),
                                      _outputs_stream=stalled_response())
                asyncio.run(_ServerInferenceSession._step(rpc, b'test input'))
                pytest.fail('stalled RPC returned without a timeout')
            return inputs, None

        class Peer:
            history = None
            span = spans[0]

            def __exit__(self, *exc):
                closed.append(attempt)

        peer = Peer()
        peer.step = step
        return [peer]

    monkeypatch.setattr(InferenceSession, '_enter_server_sessions', connect)

    class SessionsModel(Model):
        def inference_session(self, **options):
            self.options = options
            attempt_ids.append(options['dynamo_cache']['attempt_id'])
            self.session = InferenceSession(manager, **options)
            return self.session

        def generate(self, ids, **kwargs):
            self.session.step(ids.unsqueeze(-1).float())
            return super().generate(ids, **kwargs)

    model = SessionsModel(log)
    model.transformer.h.sequence_manager = manager
    ids = torch.tensor([archived()[1]])
    with OrdinaryClient(model, context='ctx', output_root=tmp_path/'out',
                        config=dict(distributed_telemetry=True, telemetry_namespace='test',
                                    consumer_node_id='consumer', provider_log=str(log))) as client:
        publisher = client.adapter.publisher
        monkeypatch.setattr(publisher, 'publish', lambda **kw: dict(observed_until=time.time()))
        with pytest.raises(TimeoutError) as error:
            client.generate(ids, max_new_tokens=3, request_id='times-out')
        failed = error.value.kv_terminal
        assert failed['status'] == failed['cache_terminal_status'] == 'failed'
        assert failed['completion_adapter']['observed']['kind'] == 'request_failed'
        assert client.history == [] and not client.ledger.read_text()
        assert client.attempt_sequence == 1 and attempt_ids == [failed['attempt_id']]
        assert len(calls) == budget and len(set(calls)) == 1
        assert failures == ['test-peer'] * budget and closed == attempt_ids * budget
        _, recovered = client.generate(ids, max_new_tokens=3, request_id='next')
        assert recovered['status'] == 'complete_success'
        assert recovered['completion_adapter']['observed']['kind'] == 'cold_recompute'
        assert recovered['attempt_id'] != failed['attempt_id']
        assert calls[-1][0] == recovered['attempt_id'] and len(calls) == budget + 1
        assert [entry['request_id'] for entry in client.history] == ['next']
        assert [json.loads(row)['request_id'] for row in client.ledger.read_text().splitlines()] == ['next']
        assert {(e.request_id, e.kind) for e in publisher._events.values()} == {
            ('times-out', 'request_failed'), ('next', 'cold_recompute')}
        assert len(publisher._events) == 2


def test_jsonl_cli_passes_bounded_client_settings(tmp_path, monkeypatch):
    import bloombee

    captured = {}

    class StopAtModelLoad(Exception):
        pass

    def from_pretrained(model, **kwargs):
        captured.update(kwargs)
        raise StopAtModelLoad

    monkeypatch.setitem(bloombee.__dict__, 'AutoDistributedModelForCausalLM',
                        SimpleNamespace(from_pretrained=from_pretrained))
    stream = tmp_path/'requests.jsonl'
    stream.write_text(json.dumps(dict(request_id='one', token_ids=[1]))+'\n')
    script = Path(__file__).resolve().parents[2]/'examples/ordinary_requests.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--model', 'fixture', '--peer', 'test-peer',
                                     '--dht-prefix', 'test', '--requests-jsonl', str(stream),
                                     '--output-root', str(tmp_path/'out')])
    with pytest.raises(StopAtModelLoad):
        runpy.run_path(str(script), run_name='__main__')
    assert {k: captured[k] for k in ('max_retries', 'request_timeout', 'connect_timeout', 'update_period')} == {
        'max_retries': 1, 'request_timeout': 600, 'connect_timeout': 20, 'update_period': 5}
