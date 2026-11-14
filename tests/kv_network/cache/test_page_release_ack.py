"""Real MemoryCache FIFO and page callbacks; no model compute is used to retire a lease."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest
import torch

from bloombee.server import page_control
from bloombee.server.page_release_ack import ReleaseJournal, ReturnedLease, RuntimeReleaseTask
from bloombee.server.task_pool import PrioritizedTaskPool
from hivemind.moe.server.runtime import Runtime
from test_dynamo_page_service import actual_attention_components


@contextmanager
def components(monkeypatch):
    with actual_attention_components(monkeypatch) as (backend, manager, request, decode, close):
        memory = manager.cache
        journal = memory.release_journal = ReleaseJournal()
        journal.runtime_thread = threading.get_ident()
        # Fixture's model request uses the production prefill; add the same
        # optional attempt metadata provided by the ordinary public client.
        from bloombee.server import dynamo_page_service
        register = dynamo_page_service._register
        def register_attempt(backend, state, session, handle, info):
            from dataclasses import replace
            cfg = dict(info.dynamo_cache, attempt_id="attempt-A")
            return register(backend, state, session, handle, replace(info, dynamo_cache=cfg))
        monkeypatch.setattr(dynamo_page_service, "_register", register_attempt)
        _, _, handles = request(list(range(1, 9)), finish=False)
        yield backend, memory, journal, handles


@pytest.mark.parametrize("already_drained", [False, True])
def test_close_ack_requires_matching_real_free_and_can_drain_without_forward(monkeypatch, already_drained):
    with components(monkeypatch) as (backend, memory, journal, handles):
        task = RuntimeReleaseTask(memory)
        args = (journal.instance, "attempt-A", handles, [backend.name])
        assert not task(*args)[0].item() and journal.status("attempt-A")["status"] == "unknown"
        expected = journal.expected("attempt-A")
        assert expected["handles"] == list(handles) and expected["backend_uids"] == [backend.name]
        current = memory._dynamo_page_records[handles[0]]["session"]
        # A separate active reader shares a registered physical page.
        key = next(iter(current.pool.lookup))
        other = current.pool.begin([key])
        shared = other.slots[0][0]
        memory.force_free(*handles)
        assert not current.closed and journal.status("attempt-A")["status"] == "unknown"
        if already_drained:
            with memory.use_cache():
                pass
            assert current.closed and journal.status("attempt-A")["status"] == "unknown"
        pool = PrioritizedTaskPool(task, max_batch_size=1, name="release_test")
        assert pool.get_task_size(SimpleNamespace(args=args)) == 1
        with ThreadPoolExecutor(max_workers=1) as worker:
            with pytest.raises(ValueError, match="wrong release runtime"):
                worker.submit(task, *args).result(3)
        # Exercise the same upstream process_batch and result sizing used by Runtime.run.
        out, batch_size = Runtime.process_batch(None, pool, 0, *args)
        assert out[0].item() and batch_size == 1 and current.closed
        assert current.pool.pages[shared].refs == 1
        owner = journal.status("attempt-A")
        assert owner["handles"] == list(handles) and owner["backend_uids"] == [backend.name]
        assert owner["allocation_generation"] == handles[0] and owner["status"] == "released"
        assert task(*args)[0].item() and owner == journal.status("attempt-A")
        assert not task(journal.instance, "wrong", handles, [backend.name])[0].item()
        assert not task(journal.instance, "attempt-A", [handles[0]+999], [backend.name])[0].item()
        with pytest.raises(ValueError, match="wrong release runtime"):
            task("old-runtime", *args[1:])
        other.release()


def test_captured_event_outlives_slot_reuse_and_needs_completion(monkeypatch):
    class Fence:
        ready = False
        def query(self):
            return self.ready
    fence = Fence()
    with components(monkeypatch) as (backend, memory, journal, handles):
        session = memory._dynamo_page_records[handles[0]]["session"]
        monkeypatch.setattr(session.pool, "_fence", lambda page: setattr(page, "completion", fence))
        memory.force_free(*handles)
        assert RuntimeReleaseTask(memory)(journal.instance, "attempt-A", handles, [backend.name])[0].item()
        assert journal.status("attempt-A")["status"] == "pending"
        # A later writer owns the slot metadata; the ack keeps its original event.
        for page in session.pool.pages:
            page.completion = None
        assert journal.status("attempt-A")["status"] == "pending"
        fence.ready = True
        assert journal.status("attempt-A")["status"] == "released"
        journal.register("attempt-A", 999, backend.name, (999,))
        assert journal.status("attempt-A")["status"] == "unknown"
        assert not journal.seal(journal.instance, "attempt-A", handles, [backend.name])


def test_read_only_endpoint_and_control_owner_coverage(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_PAGE_CONTROL_PORT", "0")
    monkeypatch.setenv("BLOOMBEE_PAGE_CONTROL_BIND", "127.0.0.1")
    for name in ("_server", "_terminal_hook", "_ready_hook", "_request_terminal_hook", "_bundle_registry"):
        monkeypatch.setattr(page_control, name, None)
    monkeypatch.setattr(page_control, "_states", {})
    try:
        with components(monkeypatch) as (backend, memory, journal, handles):
            url = f"http://127.0.0.1:{page_control._server.server_port}/release_ack"
            def get():
                with urlopen(Request(url, data=b'{"attempt_id":"attempt-A"}'), timeout=2) as response:
                    return json.load(response)
            def post(route, body):
                with urlopen(Request(url.rsplit("/", 1)[0] + route, data=json.dumps(body).encode()), timeout=2) as response:
                    return json.load(response)
            reservation = post("/reserve_current", {"attempt_id": "attempt-A", "tokens": [1] * 16})
            assert reservation["clock_domain"] == page_control.local_clock_domain()
            # Terminal before FIFO free is valid. Captured allocation identity
            # comes from create/prefill and exists with a still-live lease.
            before = post("/terminal", {"attempt_id": "attempt-A", "status": "complete_success"})
            expected = before["release_expected"]
            assert expected["owners"] == [journal.expected("attempt-A")]
            assert expected["runtime_pid"] == reservation["runtime_pid"] and get()["status"] == "unknown"
            memory.force_free(*handles)
            epoch = page_control.status()["page_epoch"]
            assert get()["status"] == "unknown" and page_control.status()["page_epoch"] == epoch
            assert handles[0] in memory._dynamo_page_records
            assert RuntimeReleaseTask(memory)(journal.instance, "attempt-A", handles, [backend.name])[0].item()
            receipt = get()
            assert receipt["status"] == "released"
            assert all(receipt["owners"][0][key] == value for key, value in expected["owners"][0].items())
            # The opposite ordering must yield the same independently frozen identity.
            again = post("/terminal", {"attempt_id": "attempt-A", "status": "complete_success"})
            assert again["release_expected"] == expected
            fake = SimpleNamespace(cache_manager=backend.cache_manager, name="other-owner")
            page_control._states[fake.name] = (fake, backend._dynamo_service_state)
            assert get()["status"] == "unknown"
            del page_control._states[fake.name]
    finally:
        if page_control._server:
            page_control._server.shutdown()
            page_control._server.server_close()


def test_close_maintenance_uses_task_queue_and_rpc_allocation_exit(monkeypatch):
    from bloombee.server.handler import TransformerConnectionHandler
    from bloombee.server.page_release_ack import ReleaseJournal, RuntimeReleaseTask
    from bloombee.server.dynamo_page_service import _register
    from bloombee.server.dynamo_pages import EnginePagePool
    from bloombee.utils.hivemind_compat import TensorDescriptor

    with actual_attention_components(monkeypatch) as (base, manager, *unused):
        memory = manager.cache
        journal = memory.release_journal = ReleaseJournal()
        pool = PrioritizedTaskPool(RuntimeReleaseTask(memory), max_batch_size=1, name="actual_free_queue")
        # Handler and runtime share a PID in this CPU-only traversal. Avoid
        # pickling MPFuture back into its originating PID; the real runtime
        # runs in another process in the two-node acceptance.
        from queue import Queue
        pool.submitted_tasks = Queue()
        backend = SimpleNamespace(name="block.0", cache_manager=manager, release_pool=pool)
        backend.get_inference_cache_descriptors = lambda batch, length: (
            TensorDescriptor((batch, 2, 4, length), dtype=torch.float16, device=torch.device("cpu")),)
        pages = EnginePagePool(2, 2, 1, 4, dtype=torch.float32, device="cpu")
        state = SimpleNamespace(binding=("context",), sessions={}, pool=pages, remote_gpu_sources={},
                                remote_cpu_sources={}, cache=SimpleNamespace(entries={}))
        attempt = "rpc-contract"
        fifo_observed = threading.Event()
        original_free = memory.force_free
        def observe_free(*handles, **kwargs):
            original_free(*handles, **kwargs)
            fifo_observed.set()
        monkeypatch.setattr(memory, "force_free", observe_free)
        # This integration test drives the handler in this process; the actual
        # runtime task executes on a distinct serial worker and the actual memory-cache FIFO.
        monkeypatch.setattr(manager, "runtime_pid", -1)
        pool.start()
        def process_one():
            journal.runtime_thread = threading.get_ident()
            uid, args = pool.load_batch_to_runtime(timeout=8)
            assert fifo_observed.is_set()
            try:
                out, size = Runtime.process_batch(None, pool, uid, *args)
                assert size == 1
                pool.send_outputs_from_runtime(uid, out)
            except BaseException as exc:
                pool.send_exception_from_runtime(uid, exc)
                raise
        async def rpc():
            async with TransformerConnectionHandler._allocate_cache(
                    None, (backend,), batch_size=1, max_length=8, timeout=0, release_attempt=attempt) as groups:
                (handle,), = groups
                with memory.use_cache(handle):
                    pass
                lease = pages.begin()
                x = torch.ones(2, 1, 4)
                lease.append(x, x)
                _register(backend, state, lease, handle, SimpleNamespace(dynamo_cache={"attempt_id": attempt}))
                assert journal.expected(attempt)["handles"] == [handle]
                assert journal.status(attempt)["status"] == "unknown"
            assert journal.status(attempt)["status"] == "released"
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_one)
                asyncio.run(asyncio.wait_for(rpc(), 12))
                future.result(timeout=2)
        finally:
            pool.shutdown()
            pool.join(timeout=3)


def _handler_in_child(pool, instance, handles, names, channel):
    # Do not import/initialize CUDA in a forked handler.
    async def submit():
        return await asyncio.wait_for(pool.submit_task(instance, "child", handles, names), 6)
    try:
        result = asyncio.run(submit())
        channel.send(("ok", result[0].item()))
    except Exception as error:
        channel.send(("error", repr(error)))
    finally:
        channel.close()


def test_maintenance_pool_returns_to_other_process_with_real_mpfuture(monkeypatch):
    import multiprocessing
    # Same fork model used on the deployed Linux worker. Test only the small
    # release transport; the previous tests cover the real FIFO and callbacks.
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires a forked connection handler")
    journal = ReleaseJournal()
    journal.runtime_thread = threading.get_ident()
    with actual_attention_components(monkeypatch) as (base, manager, *unused):
        memory = manager.cache
        memory.release_journal = journal
        lease = ReturnedLease("child", "uid", 21, 1, "digest", None)
        journal.register("child", 21, "uid", (21,))
        journal.observe_free((21,), (lease,))
        pool = PrioritizedTaskPool(RuntimeReleaseTask(memory), max_batch_size=1, name="cross_process_release")
        pool.start()
        ctx = multiprocessing.get_context("fork")
        rx, tx = ctx.Pipe(duplex=False)
        child = ctx.Process(target=_handler_in_child, args=(pool, journal.instance, (21,), ("uid",), tx))
        child.start()
        tx.close()
        try:
            uid, batch = pool.load_batch_to_runtime(timeout=8)
            outputs, size = Runtime.process_batch(None, pool, uid, *batch)
            assert size == 1
            pool.send_outputs_from_runtime(uid, outputs)
            assert rx.poll(8) and rx.recv() == ("ok", True)
            child.join(timeout=4)
            assert child.exitcode == 0
        finally:
            rx.close()
            pool.shutdown()
            pool.join(timeout=3)
            if child.is_alive():
                child.terminate()
                child.join(timeout=3)


@pytest.mark.parametrize("during_query", ["rebind", "replace_fence", "unrelated"])
def test_status_revalidates_lease_after_unlocked_fence_query(during_query):
    entered, resume = threading.Event(), threading.Event()

    class PausedFence:
        def query(self):
            entered.set()
            assert resume.wait(timeout=3)
            return True

    journal = ReleaseJournal()
    attempt, uid, handles = "in-flight", "layer.0", (17,)
    journal.register(attempt, handles[0], uid, handles)
    row = ReturnedLease(attempt, uid, handles[0], 1, "a" * 64, PausedFence())
    journal.observe_free(handles, (row,))
    assert journal.seal(journal.instance, attempt, handles, (uid,))
    with ThreadPoolExecutor(max_workers=1) as executor:
        poll = executor.submit(journal.status, attempt)
        try:
            assert entered.wait(timeout=3)
            if during_query == "rebind":
                journal.register(attempt, 18, uid, (18,))
            elif during_query == "replace_fence":
                pending = threading.Event()
                replacement = ReturnedLease(attempt, uid, handles[0], 1, "a" * 64,
                                            SimpleNamespace(query=pending.is_set))
                journal.observe_free(handles, (replacement,))
            else:
                journal.register("other", 18, uid, (18,))
        finally:
            resume.set()
        expected = "released" if during_query == "unrelated" else "unknown"
        assert poll.result(timeout=3)["status"] == expected
    expected_next = "pending" if during_query == "replace_fence" else expected
    assert journal.status(attempt)["status"] == expected_next
