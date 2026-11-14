"""CPU controller disclosure checks; protocol doubles leave the real audit unmodified."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from bloombee.kv_cache.network.release_ack import ReleasePending
from test_background_dispatch import receipt, run


@pytest.mark.parametrize('bad', ['attempt', 'runtime_pid', 'clock_domain', 'generation', 'instance',
                                 'handle', 'uid', 'empty', 'owner_pending', 'unknown', 'failed'])
def test_release_mismatch_cannot_disclose_or_score(run, bad):
    c = run.c
    ack = receipt('first')
    owner = ack['owners'][0]
    if bad == 'attempt': ack['attempt_id'] = 'other'
    elif bad == 'runtime_pid': ack['runtime_pid'] += 1
    elif bad == 'clock_domain': ack['clock_domain'] += 'other'
    elif bad == 'generation': ack = receipt('first', generation=80)
    elif bad == 'instance': owner['runtime_instance'] = 'other-runtime'
    elif bad == 'handle': owner['handles'] = [71]
    elif bad == 'uid': owner['backend_uids'] = ['layer.1']
    elif bad == 'empty': ack['owners'] = []
    elif bad == 'owner_pending': owner['status'] = 'pending'
    else: ack['status'] = bad
    c._release_post = lambda attempt, timeout: ack
    c.spec['release_ack_timeout_s'] = .06
    with pytest.raises(ReleasePending if bad == 'unknown' else ValueError):
        run.disclose()
    assert not c.history and not c.decisions and not list(c.inbox.iterdir())


def test_release_does_not_replace_full_model_terminal(run):
    run.history[0]['actual_output_tokens'] = 0
    run.c._release_post = lambda *args: pytest.fail('release polled without full model proof')
    with pytest.raises(ValueError, match='incomplete model output'):
        run.disclose()
    assert not run.c.history and not run.c.decisions


def test_pending_timeout_then_concurrent_resume_once(run):
    c = run.c
    calls = []
    orig_post = c._post
    def post(route, body):
        calls.append(route)
        return orig_post(route, body)
    c._post = post
    c.spec['release_ack_timeout_s'] = .08
    entered, release = threading.Event(), threading.Event()
    def get_ack(attempt, timeout):
        entered.set()
        return dict(receipt(attempt), status='released' if release.is_set() else 'pending')
    c._release_post = get_ack
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run.disclose)
        assert entered.wait(1) and not c.history and not c.decisions
        with pytest.raises(ReleasePending): first.result(timeout=2)
        assert not c.history and not list(c.inbox.iterdir())
        release.set()
        futures = [pool.submit(run.disclose) for _ in range(2)]
        for f in futures: f.result(timeout=2)
    assert calls.count('/terminal') == 1
    assert len(c.history) == len(c.decisions) == len(list(c.inbox.glob('*/execute.json'))) == 1


def test_ordered_handle_uid_bindings_and_extra_fences():
    from copy import deepcopy
    from bloombee.kv_cache.network.release_ack import identity
    original = receipt('a')
    owner = original['owners'][0]
    owner.update(handles=[70, 71], backend_uids=['l0', 'l1'])
    reordered = deepcopy(original)
    row = reordered['owners'][0]
    row.update(handles=[71, 70], backend_uids=['l1', 'l0'], page_counts=[2, 2])
    assert identity(original, 'a') == identity(reordered, 'a')
    row['backend_uids'].reverse()
    assert identity(original, 'a') != identity(reordered, 'a')


def test_missing_frozen_allocation_blocks_before_polling(run):
    run.c._post = lambda *args: {'release_expected': None}
    run.c._release_post = lambda *args: pytest.fail('unbound attempt reached release endpoint')
    with pytest.raises(ValueError):
        run.disclose()
    assert not run.c.history and not list(run.c.inbox.iterdir())


def test_ordinary_timeout_close_resumes_same_attempt(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from bloombee.kv_cache.network import controller as ctl
    from bloombee.kv_cache.network.lifecycle import LifecycleError
    from bloombee.kv_cache.network.ordinary import OrdinaryClient
    from test_background_dispatch import reservation
    from test_review_regressions import Model, archived

    log = tmp_path/'model.log'
    log.touch()
    model = Model(log)
    client = OrdinaryClient(model, output_root=tmp_path/'ordinary', context='ctx',
                            config=dict(cache_evidence=True, provider_log=str(log)))
    c = object.__new__(ctl.Controller)
    c.root = tmp_path
    c.lock = threading.RLock()
    c.terminal_attempts, c.disclosed_attempts = set(), set()
    c.pending_attempts, c.decisions, c.history = {}, [], []
    c.closed = False
    c.spec = {'release_ack_timeout_s': .025}
    c.request_count = 1
    c.boundary = dict(consumer_id='consumer', clock_domain_id='cpu', initial_completion_seq=0)
    c._connect = lambda: None
    c._attach_manifests = lambda h: None
    pages, banks, polls = [], [], []
    def terminal(route, payload):
        pages.append(payload['attempt_id'])
        return {'release_expected': receipt(pages[-1])}
    def claim(**args):
        a = args['attempt_id']
        c.pending_attempts[a] = {'reserved': reservation(a)}
        return {}
    c.claim_current = claim
    c._post = terminal
    c.ticket = SimpleNamespace(terminal=lambda a, status: banks.append(a))
    c._score = lambda **kw: c.decisions.append(kw) or kw
    c._dispatch = lambda d: None
    c.close = lambda: setattr(c, 'closed', True)
    release = threading.Event()
    def ack(a, timeout):
        polls.append(a)
        return dict(receipt(a), status='released' if release.is_set() else 'pending')
    c._release_post = ack
    client.adapter.prefetch = c
    client.config['control_endpoint'] = 'unused'
    monkeypatch.setattr(client, '_post_page_terminal', lambda *args: pytest.fail('duplicate terminal'))
    with pytest.raises(LifecycleError):
        client.generate(torch.tensor([archived()[1]]), max_new_tokens=3, request_id='q')
    with pytest.raises(LifecycleError):
        client.close()
    assert not c.history and not c.decisions and not c.closed
    release.set()
    client.close()
    assert client.closed and c.closed and len(c.history) == len(c.decisions) == 1
    assert len(pages) == len(banks) == 1 and set(polls) == set(pages) == set(banks)


@pytest.mark.parametrize('key', [
    'status', 'attempt_id', 'runtime_pid', 'page_counts', 'slot_generation_sha256',
])
def test_missing_owner_proof_blocks_disclosure(run, key):
    ack = receipt('first')
    del ack['owners'][0][key]
    run.c._release_post = lambda *args: ack
    with pytest.raises(ValueError):
        run.disclose()
    assert not run.c.history and not run.c.decisions and not list(run.c.inbox.iterdir())


@pytest.mark.parametrize(('key', 'value'), [
    ('status', None), ('status', 'pending'), ('attempt_id', None), ('attempt_id', 'another'),
    ('runtime_pid', '17'), ('runtime_pid', 17.0), ('runtime_pid', 18),
    ('page_counts', None), ('page_counts', 2), ('page_counts', []), ('page_counts', [1, 2]),
    ('page_counts', [True]), ('page_counts', [0]), ('page_counts', [-1]), ('page_counts', [1.0]),
    ('slot_generation_sha256', None), ('slot_generation_sha256', 'a'*64),
    ('slot_generation_sha256', []), ('slot_generation_sha256', ['a'*64]*2),
    ('slot_generation_sha256', [None]), ('slot_generation_sha256', ['a'*63]),
    ('slot_generation_sha256', ['a'*65]), ('slot_generation_sha256', ['z'*64]),
    ('slot_generation_sha256', ['a'*64+'\n']),
])
def test_malformed_owner_proof_blocks_disclosure(run, key, value):
    ack = receipt('first')
    ack['owners'][0][key] = value
    run.c._release_post = lambda *args: ack
    with pytest.raises(ValueError):
        run.disclose()
    assert not run.c.history and not run.c.decisions and not list(run.c.inbox.iterdir())


def test_every_owner_and_every_returned_handle_is_checked():
    from copy import deepcopy
    from bloombee.kv_cache.network.release_ack import wait_released
    from test_background_dispatch import reservation
    ack = receipt('a')
    second = dict(ack['owners'][0], runtime_instance='other', handles=[3, 4],
                  allocation_generation=3, backend_uids=['layer.1', 'layer.2'],
                  page_counts=[3, 1], slot_generation_sha256=['b'*64, 'c'*64])
    ack['owners'].append(second)
    expected = deepcopy(ack)
    for row in expected['owners']:
        for field in ('attempt_id', 'runtime_pid', 'status', 'page_counts', 'slot_generation_sha256'):
            del row[field]
    ack['owners'].reverse()
    assert wait_released('a', expected, reservation('a'), lambda *args: ack, 1) == ack
    second['slot_generation_sha256'][1] = ''
    with pytest.raises(ValueError, match='handle 4'):
        wait_released('a', expected, reservation('a'), lambda *args: ack, 1)
