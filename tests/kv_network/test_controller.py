"""d1 protocol regression checks with explicit local control doubles, no GPU."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import threading
import time
from bloombee.kv_cache.network.controller import Controller


def test_worker_done_visible_after_exit_is_success(tmp_path):
    controller=object.__new__(Controller)
    controller.inbox=tmp_path;controller.spec={};controller.ready={'worker_pid':123}
    target=tmp_path/'worker_done.json'
    def exited(pid,signal):
        target.write_text(json.dumps({'status':'complete'}))
        raise ProcessLookupError()
    with patch('os.kill',side_effect=exited):
        assert controller._wait(target)=={'status':'complete'}


def test_pending_warm_does_not_block_current_claim(tmp_path):
    controller=object.__new__(Controller)
    controller.lock=threading.RLock();controller.closed=False
    controller.pending_attempts={};controller.terminal_attempts=set();controller.root=tmp_path
    controller.http=SimpleNamespace();controller.started=False
    order=[];controller.decisions=[dict(execute_requested=True,directory=str(tmp_path))]
    controller._connect=lambda:None
    controller.start_after_t0=lambda:order.append('history_score')
    def wait(path):
        order.append(Path(path).name)
        return {'status':'accepted_async'}
    controller._wait=wait
    def post(route,value):
        order.append(route);return {'reserved':True}
    controller._post=post
    controller.worker_config={}
    controller.ticket=SimpleNamespace(claim_for_layout=lambda *args:None)
    with patch('bloombee.kv_cache.network.controller.layout_for_tokens',return_value=object()):
        result=controller.claim_current(attempt_id='actual',token_ids=list(range(17)),
            actual_ingress_ns=time.monotonic_ns(),clock_domain=None)
    assert order==['history_score','/reserve_current']
    assert result['status']=='not_ready_immediate_fallback'
    # Current request is used only for reservation/claim, never passed to score.
    assert 'actual' in controller.pending_attempts


def test_external_audit_sha_failure_and_default_builtin(tmp_path):
    from bloombee.kv_cache.network.controller import pinned
    from bloombee.kv_cache.network import independent_audit
    bad=tmp_path/'audit.py';bad.write_text('raise AssertionError("must not execute")')
    import pytest
    with pytest.raises(ValueError,match='SHA'):pinned(dict(path=str(bad),sha256='0'*64))
    # Constructor chooses the real builtin while still validating history.
    from bloombee.kv_cache.network.strategy_runner import validate_demand_ledger
    from test_independent_audit import gate
    assert gate.audit is independent_audit.audit


def test_controller_terminal_attempts_both_sides_and_retryable_close(tmp_path):
    import pytest
    from bloombee.kv_cache.network.lifecycle import LifecycleError
    c=object.__new__(Controller);c.lock=threading.RLock();c.root=tmp_path
    c.inbox=tmp_path;c.closed=False;c.terminal_attempts=set();c.pending_attempts={'a':{}}
    c._connect=lambda:None;c.decisions=[];calls=[]
    fail=[True]
    def post(*args):
        calls.append('page')
        if fail[0]:raise RuntimeError('page unavailable')
        return {'ok':True}
    c._post=post
    c.ticket=SimpleNamespace(terminal=lambda *a:(calls.append('ticket') or {'ok':True}),close=lambda:calls.append('ticket.close'))
    c.http=SimpleNamespace(close=lambda:calls.append('http.close'))
    with pytest.raises(LifecycleError):c.terminal(terminal={'attempt_id':'a','status':'cancelled'})
    assert calls==['page','ticket'] and 'a' in c.pending_attempts
    fail[0]=False
    wait_count=[0]
    def wait(path):
        wait_count[0]+=1
        if wait_count[0]==1:raise TimeoutError('transient done marker')
        return {'status':'done'}
    c._wait=wait
    with pytest.raises(LifecycleError,match='transient done marker'):c.close()
    assert (tmp_path/'finished.json').exists()
    c.close();c.close()
    assert c.closed and calls.count('ticket.close')==calls.count('http.close')==1


def test_worker_rejects_tampered_audit_receipt(tmp_path):
    from bloombee.kv_cache.network.worker_protocol import Worker
    receipt=tmp_path/'controller_independent_audit.json';receipt.write_text('{}')
    command=dict(decision_id='d',context_sha256='c',selection_sha256='s',ec_match=True,
                 audit_ref=str(receipt),ec_audit_sha256='0'*64)
    (tmp_path/'execute.json').write_text(json.dumps(command))
    worker=object.__new__(Worker);worker.emit=lambda *a,**kw:None
    job=dict(execute_handled=False,decision_id='d',result=dict(context_sha256='c',selection_sha256='s'))
    worker.execute(tmp_path,job)
    assert json.loads((tmp_path/'execute_done.json').read_text())['status']=='rejected'
    assert job['execute_handled'] and 'future' not in job


def test_audit_evidence_write_failure_stays_unknown(tmp_path):
    from bloombee.kv_cache.network.independent_audit import audit
    result=audit({}, {}, [], {},tmp_path/'not-created')
    assert result['ec_match'] is False
    assert any('audit_evidence_write_failed' in s for s in result['unknown'])


def test_controller_close_preserves_success_and_closes_clients_on_persistent_terminal_error(tmp_path):
    import pytest
    from bloombee.kv_cache.network.lifecycle import LifecycleError
    c=object.__new__(Controller);c.lock=threading.RLock();c.root=tmp_path;c.inbox=tmp_path
    c.closed=False;c.terminal_attempts=set();c.pending_attempts={'a':{}};c.decisions=[]
    c._connect=lambda:None;c._wait=lambda path:{'status':'done'};calls=[]
    def post(route,body):
        calls.append(('page',body['status']));raise RuntimeError('persistent page outage')
    c._post=post
    c.ticket=SimpleNamespace(terminal=lambda a,status:calls.append(('ticket',status)),close=lambda:calls.append('ticket.close'))
    c.http=SimpleNamespace(close=lambda:calls.append('http.close'))
    with pytest.raises(LifecycleError):c.terminal(terminal=dict(attempt_id='a',status='complete_success'))
    with pytest.raises(LifecycleError):c.close()
    assert 'a' in c.pending_attempts and not c.closed
    assert (tmp_path/'finished.json').exists() and (tmp_path/'worker_drained.json').exists()
    assert calls.count('ticket.close')==calls.count('http.close')==1
    assert all(row[1]=='complete_success' for row in calls if isinstance(row,tuple))
    with pytest.raises(LifecycleError):c.close()
    assert calls.count('ticket.close')==calls.count('http.close')==1
