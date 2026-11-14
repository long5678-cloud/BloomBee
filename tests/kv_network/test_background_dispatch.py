"""Local controller/worker timing. Model, metadata, ranking, audit and transfer use doubles."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from bloombee.kv_cache.network import controller as ctl
from bloombee.kv_cache.network.worker_protocol import Worker, COMMON_FILES


def receipt(attempt, generation=70):
    return dict(attempt_id=attempt, runtime_pid=17, clock_domain='cpu-test',
                status='released', owners=[dict(attempt_id=attempt, runtime_pid=17,
                    status='released', runtime_instance='instance', allocation_generation=generation,
                    handles=[generation], backend_uids=['layer.0'],
                    page_counts=[2], slot_generation_sha256=['a'*64])])


def reservation(attempt):
    return dict(attempt_id=attempt, runtime_pid=17, clock_domain='cpu-test', reserved=True)


@pytest.fixture
def run(tmp_path, monkeypatch):
    c = object.__new__(ctl.Controller)
    c.root, c.inbox = tmp_path, tmp_path/'inbox'
    c.inbox.mkdir()
    c.started, c.closed, c.ready = True, False, None
    c.lock = threading.RLock()
    c.request_count = 2
    c.spec, c.decisions, c.history = {}, [], []
    c.terminal_attempts, c.disclosed_attempts, c.pending_attempts = set(), set(), {}
    c.boundary = dict(consumer_id='target', clock_domain_id='test', initial_completion_seq=0,
                      truncated=False, window_id='test', completion_sequence_semantics='unique_successful_full_request_disclosure')
    c.worker_config = dict(run_id='test', prefetch_strategy='lrfu')
    c.config_sha = ctl.digest(c.worker_config)
    c.metadata = dict.fromkeys(ctl.META_ALLOWED - {'completed_trace_rows'}, 'test')
    c._connect = lambda: None
    c._attach_manifests = lambda history: None
    c.pending_attempts['first'] = dict(reserved=reservation('first'))
    c._post = lambda route, body: (dict(release_expected=receipt(body['attempt_id']))
                                   if route == '/terminal' else reservation(body['attempt_id']))
    c._release_post = lambda attempt, timeout: receipt(attempt)
    c.ticket = SimpleNamespace(terminal=lambda *args: {}, claim_for_layout=lambda *args: None)
    monkeypatch.setattr(ctl, 'layout_for_tokens', lambda cfg, tokens: None)
    c.audit = lambda **kw: dict(ec_match=True, unknown=[], context_sha256=ctl.digest(kw['context']))
    terminal = dict(attempt_id='first', request_id='first', index=0, status='complete_success',
                    requested_output_tokens=1, actual_output_tokens=1)
    proof = tmp_path/'source-terminal.json'
    ctl.write(proof, terminal)
    tokens = json.loads((Path(__file__).parent/'fixtures/published_identity.json').read_text())['ids'][0]
    history = [dict(terminal, token_ids=tokens, completed_monotonic=time.monotonic(), terminal_ref=str(proof))]
    state = SimpleNamespace(c=c, job={}, selection=ctl.digest(tokens[:16]), terminal=terminal,
                            history=history, disclose=lambda: c.disclose_complete(history=history, terminal=terminal))

    def score_response(path):
        assert path.name == 'score_done.json', f'unexpected synchronous wait: {path.name}'
        folder = path.parent
        req = ctl.read(folder/'score_request.json')
        ctx = dict(decision_id=req['decision_id'])
        sel = dict(selected_key=state.selection)
        result = dict(context_sha256=ctl.digest(ctx), selection_sha256=ctl.digest(sel), selection=sel)
        files = {name: ctl.write(folder/name, body) for name, body in
                 [('context.json', ctx), ('qualification.json', {}), ('score_result.json', result)]}
        state.job.update(decision_id=req['decision_id'], request=req, context=ctx, result=result,
                         execute_handled=False, future=None)
        return dict(status='scored', config_sha256=c.config_sha, files=files,
                    score_request_sha256=ctl.file_sha(folder/'score_request.json'))
    c._wait = score_response
    return state


def test_start_before_next_ingress_and_fallback_while_running(run):
    decision = run.disclose()
    folder = Path(decision['directory'])
    assert decision['execute_requested'] and (folder/'execute.json').exists()
    assert run.disclose()['status'] == 'disclosure_idempotent'
    assert len(run.c.decisions) == 1
    entered, release = threading.Event(), threading.Event()
    started = []
    def fake_transfer(*args):
        started.append(time.monotonic_ns())
        entered.set()
        assert release.wait(5)
        return dict(warm_attempts=0, scope='CPU transfer stub')
    worker = object.__new__(Worker)
    worker.runner = SimpleNamespace(warm_selected=fake_transfer)
    worker.emit = lambda *args, **kwargs: None
    with ThreadPoolExecutor(max_workers=1) as worker.pool:
        try:
            worker.execute(folder, run.job)
            assert entered.wait(2)
            ingress = time.monotonic_ns()
            worker.execute(folder, run.job)
            claim = run.c.claim_current(attempt_id='next', token_ids=run.history[0]['token_ids'],
                                        actual_ingress_ns=ingress, clock_domain='test')
            assert len(started) == 1 and started[0] < ingress
            assert claim['status'] == 'not_ready_immediate_fallback'
            assert claim['actual_ingress_ns'] == ingress
            assert not (folder/'warm_done.json').exists()
            ctl.write(folder/'cpu_timing.json', dict(worker_entered_stub_ns=started[0], ingress_ns=ingress,
                      fallback_returned_ns=time.monotonic_ns(), warm_finished_at_claim=False))
        finally:
            release.set()
        run.job['future'].result(timeout=2)
    assert (folder/'warm_done.json').exists()
    assert COMMON_FILES['controller.py'] == hashlib.sha256(Path(ctl.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize('case', ['failure', 'cancelled', 'disclosure', 'audit', 'final', 'empty', 'closed'])
def test_no_dispatch(run, case):
    if case in ('failure','cancelled'):
        run.terminal['status'] = case
    elif case == 'disclosure':
        run.history[0]['actual_output_tokens'] = 0
    elif case == 'audit':
        run.c.audit = lambda **kw: dict(ec_match=False, unknown=['incomplete'])
    elif case == 'final':
        run.c.request_count = 1
    elif case == 'empty':
        run.selection = None
    elif case == 'closed':
        run.c.closed = True
    if case in ('disclosure','closed'):
        with pytest.raises((ValueError, RuntimeError)):
            run.disclose()
    else:
        run.disclose()
    assert not list(run.c.inbox.glob('*/execute.json'))


def test_retry_publishes_original_decision_once(run, monkeypatch):
    original_write = ctl.write
    def fail_once(path, body):
        if path.name == 'execute.json':
            raise OSError('temporary publish failure')
        return original_write(path, body)
    monkeypatch.setattr(ctl, 'write', fail_once)
    with pytest.raises(OSError, match='temporary'):
        run.disclose()
    assert not list(run.c.inbox.glob('*/execute.json'))
    monkeypatch.setattr(ctl, 'write', original_write)
    run.disclose()
    run.disclose()
    assert len(run.c.decisions) == 1 and len(run.c.history) == 1
    assert len(list(run.c.inbox.glob('*/execute.json'))) == 1


@pytest.mark.parametrize('damage', [None, 'request', 'ready'])
def test_score_timeout_resumes_same_frozen_request(run, monkeypatch, damage):
    c = run.c
    original_wait = c._wait
    original_audit = c.audit
    audit_histories = []
    def audit(**kw):
        audit_histories.append([event['request_id'] for event in kw['history']])
        return original_audit(**kw)
    c.audit = audit
    finished = {}
    def late_receipt(path):
        # The worker has already scored and published its artifacts exactly once.
        finished['receipt'] = original_wait(path)
        raise TimeoutError('controller missed the completed score response')
    c._wait = late_receipt
    with pytest.raises(TimeoutError):
        run.disclose()

    folder = c.inbox/'run_1'
    request = folder/'score_request.json'
    ready = folder/'score_ready.json'
    snapshot = request.read_bytes()
    assert len(c.history) == len(c.decisions) == 1
    assert not (folder/'execute.json').exists()
    assert not audit_histories
    # Retry with extra caller-visible input. It must not enter the previous decision.
    run.history.append(dict(request_id='later'))
    if damage == 'request':
        request.write_bytes(snapshot + b' ')
    elif damage == 'ready':
        ready.write_text('{}')

    c._wait = lambda path: finished['receipt']
    if damage:
        with pytest.raises(ValueError, match='frozen request|frozen score request'):
            run.disclose()
        assert not (folder/'execute.json').exists() and not audit_histories
        return
    writes = []
    publish = ctl.write
    def trace_write(path, body):
        writes.append(path.name)
        return publish(path, body)
    monkeypatch.setattr(ctl, 'write', trace_write)
    run.disclose()
    run.disclose()
    assert writes == ['controller_independent_audit.json', 'execute.json']
    assert [event['request_id'] for event in c.history] == ['first']
    assert audit_histories == [['first']]
    assert request.read_bytes() == snapshot
    assert [p.name for p in c.inbox.iterdir()] == ['run_1']
    assert c.decisions[0]['disclosed_attempt_id'] == 'first'
    assert c.decisions[0]['decision_id'] == json.loads(snapshot)['decision_id']
    assert ctl.read(folder/'execute.json')['decision_id'] == c.decisions[0]['decision_id']
