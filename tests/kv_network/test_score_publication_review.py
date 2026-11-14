"""A transient failure publishing the first score_request during real Controller.disclose_complete."""
import os
from pathlib import Path
import pytest
from test_background_dispatch import run


def test_retry_publication(run, monkeypatch):
    link = os.link
    faults = []
    def fail_once(src, dst, *args, **kwargs):
        if Path(dst).name == "score_request.json" and not faults:
            faults.append(True)
            raise OSError("injected temporary publication error")
        return link(src, dst, *args, **kwargs)
    monkeypatch.setattr(os, "link", fail_once)
    with pytest.raises(OSError, match="temporary publication"):
        run.disclose()
    state = run.c
    print("after failed publication", len(state.history), len(state.decisions),
          len(list(state.inbox.glob("*/score_request.json"))))
    monkeypatch.setattr(os, "link", link)
    # Healthy filesystem on the second attempt. Expected retry should complete.
    run.disclose()
    assert len(state.history) == len(state.decisions) == 1
    assert len(list(state.inbox.glob("*/execute.json"))) == 1


@pytest.mark.parametrize('where', ['intent', 'mkdir', 'link', 'linked_then_error', 'conflict'])
def test_freeze_before_publication_and_retry_exact_intent(run, monkeypatch, where):
    import copy
    import hashlib
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    from bloombee.kv_cache.network import controller as ctl

    c = run.c
    page_calls, score_calls, audit_inputs = [], [], []
    post, wait, audit = c._post, c._wait, c.audit
    def post_once(route, body):
        page_calls.append(route)
        return post(route, body)
    def score_once(path):
        score_calls.append(path)
        return wait(path)
    def audit_once(**kw):
        audit_inputs.append(copy.deepcopy(kw['history']))
        return audit(**kw)
    c._post, c._wait, c.audit = post_once, score_once, audit_once

    link, mkdir = os.link, Path.mkdir
    armed = [True]
    def interrupted_link(src, dst, *args, **kwargs):
        name = Path(dst).name
        if armed[0] and ((where == 'intent' and name.startswith('score_intent_')) or
                          (where in ('link', 'linked_then_error', 'conflict') and name == 'score_request.json')):
            armed[0] = False
            if where == 'linked_then_error':
                link(src, dst, *args, **kwargs)
            raise OSError('publication interrupted')
        return link(src, dst, *args, **kwargs)
    def interrupted_mkdir(path, *args, **kwargs):
        if armed[0] and where == 'mkdir' and path.name == 'run_1':
            armed[0] = False
            raise OSError('publication interrupted')
        return mkdir(path, *args, **kwargs)
    monkeypatch.setattr(os, 'link', interrupted_link)
    monkeypatch.setattr(Path, 'mkdir', interrupted_mkdir)
    with pytest.raises(OSError, match='publication interrupted'):
        run.disclose()
    record, = c.decisions
    wire = record['pending_score_wire']
    request = json.loads(wire)
    assert record['disclosed_attempt_id'] == 'first' and record['enabled'] and not record['final']
    assert record['score_request_sha256'] == hashlib.sha256(wire).hexdigest()
    assert request['causal_cutoff_ns'] <= time.monotonic_ns()
    assert [row['request_id'] for row in request['history']] == ['first']
    folder = c.inbox/'run_1'
    if where != 'intent':
        intent, = c.root.glob('score_intent_*.json')
        envelope = ctl.read(intent)
        assert ctl.json_bytes(envelope['request']) == wire
        assert envelope['disclosed_attempt_id'] == 'first'
        assert envelope['directory'] == str(folder) and envelope['decision_id'] == request['decision_id']
        assert envelope['enabled'] and not envelope['final']
    assert not list(c.inbox.glob('*/score_ready.json')) and not score_calls
    assert list(c.root.glob('history_*.json')) == [c.root/'history_1.json']
    later = dict(request_id='future', disclosed_ns=time.monotonic_ns())
    c.history.append(later.copy())
    run.history.append(later.copy())
    if where == 'conflict':
        (folder/'score_request.json').write_bytes(wire + b' ')
        with pytest.raises(ValueError, match='differs from frozen'):
            run.disclose()
        assert not score_calls and not list(c.inbox.glob('*/execute.json'))
        return
    with ThreadPoolExecutor(max_workers=2) as executor:
        retries = [executor.submit(run.disclose) for _ in range(2)]
        for retry in retries:
            retry.result(timeout=2)
    assert page_calls.count('/terminal') == 1 and len(score_calls) == 1
    assert audit_inputs == [request['history']]
    assert [row['request_id'] for row in c.history] == ['first', 'future']
    assert len(c.decisions) == len(list(c.root.glob('history_*.json'))) == 1
    assert list(c.inbox.iterdir()) == [folder]
    assert list(c.inbox.glob('*/execute.json')) == [folder/'execute.json']
    assert (folder/'score_request.json').read_bytes() == wire
    assert 'pending_score_wire' not in record
