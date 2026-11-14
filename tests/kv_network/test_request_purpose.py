"""One CPU client, archived provider rows, and the real completion/record path."""
import json
import time
from types import SimpleNamespace

import pytest
import torch

from bloombee.kv_cache.network.ordinary import OrdinaryClient
from test_review_regressions import Model, archived


def test_request_purpose_preserves_history_and_counts_measured_reuse(tmp_path, monkeypatch):
    log = tmp_path / 'provider.log'
    log.touch()
    rows, prompt, suffix = archived()

    class ReuseModel(Model):
        # The model accepts only its own generation option.
        def generate(self, ids, *, max_new_tokens):
            assert max_new_tokens == len(suffix)
            cache = self.options['dynamo_cache']
            assert 'purpose' not in cache
            with log.open('a') as stream:
                for row in rows:
                    tagged = dict(row, telemetry_attempt_id=cache['attempt_id'])
                    stream.write('DYNAMO_SERVICE ' + json.dumps(tagged) + '\n')
            return torch.tensor([prompt + suffix])

    config = dict(distributed_telemetry=True, telemetry_namespace='test', consumer_node_id='consumer',
                  provider_log=str(log), purpose='prefetch')
    disclosed = []
    with OrdinaryClient(ReuseModel(log), output_root=tmp_path/'client', context='ctx', config=config) as client:
        publisher = client.adapter.publisher
        monkeypatch.setattr(publisher, 'publish', lambda **kw: {'observed_until': time.time()})
        client.adapter.prefetch = SimpleNamespace(
            claim_current=lambda **kw: {}, terminal=lambda **kw: None, close=lambda: None,
            disclose_complete=lambda history, terminal: disclosed.append(
                (terminal['purpose'], tuple(e['request_id'] for e in history))))
        ids = torch.tensor([prompt])
        for index, options in enumerate(({'purpose':'warmup'}, {'purpose':'measured'},
                                         {'purpose':'prefetch'}, {})):
            expected = options.get('purpose', 'prefetch')
            _, terminal = client.generate(ids, max_new_tokens=len(suffix), request_id=str(index), **options)
            observed = terminal['completion_adapter']['observed']
            assert terminal['purpose'] == observed['purpose'] == expected
            persisted = json.loads((client.root/(terminal['attempt_id']+'.json')).read_text())
            assert persisted['purpose'] == expected
            assert observed['row_count'] == 32 and observed['evidence_kind'] == 'reuse_success'
            assert observed['recorded'] == (expected == 'measured')
            assert client.history[-1]['kv_reuse_success']
            assert disclosed[-1] == (expected, tuple(str(i) for i in range(index+1)))
        assert [(e.request_id, e.kind) for e in publisher._events.values()] == [('1', 'reuse_success')]
    assert json.loads((client.root/'telemetry_publisher.json').read_text())['expected_reuse_success'] == 1


def test_invalid_request_purpose_precedes_model_and_history(tmp_path):
    log = tmp_path/'provider.log'
    log.touch()
    ids = torch.tensor([archived()[1]])
    with OrdinaryClient(Model(log), output_root=tmp_path/'client', context='ctx') as client:
        for bad in ('', 'measure', False, []):
            with pytest.raises(ValueError, match='unknown invocation purpose'):
                client.generate(ids, max_new_tokens=3, purpose=bad)
        assert client.attempt_sequence == 0 and client.history == [] and not log.read_text()
        _, terminal = client.generate(ids, max_new_tokens=3)
        assert terminal['purpose'] == 'measured'
