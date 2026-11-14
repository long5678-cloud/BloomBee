"""Cancellation and allocation ordering on the ordinary cache pipe."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from urllib.request import urlopen

import pytest
import torch

from bloombee.server import page_control
from bloombee.utils.hivemind_compat import TensorDescriptor
from test_dynamo_page_service import actual_attention_components


class ObservedReader:
    """Use the real pipe; bound an unexpected wait so a regression cannot hang CI."""
    def __init__(self, reader):
        self.reader = reader
        self.waiting = threading.Event()

    def poll(self, *args):
        return self.reader.poll(*args)

    def recv(self):
        if not self.reader.poll():
            self.waiting.set()
            if not self.reader.poll(3):
                raise AssertionError("runtime is still waiting for a cache message")
        return self.reader.recv()


def test_cancelled_decode_releases_pages_and_status(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_PAGE_CONTROL_PORT", "0")
    monkeypatch.setenv("BLOOMBEE_PAGE_CONTROL_BIND", "127.0.0.1")
    for name in ("_server", "_terminal_hook", "_ready_hook", "_request_terminal_hook", "_bundle_registry"):
        monkeypatch.setattr(page_control, name, None)
    monkeypatch.setattr(page_control, "_states", {})
    try:
        with actual_attention_components(monkeypatch) as (backend, manager, request, decode, close):
            _, _, handles = request(list(range(1, 9)), finish=False)
            (handle,) = handles
            memory = manager.cache
            state = backend._dynamo_service_state
            session = memory._dynamo_page_records[handle]["session"]
            port = page_control._server.server_port
            def status_row():
                with urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
                    assert response.status == 200
                    snapshot = json.load(response)
                assert snapshot["snapshot_atomic"]
                row, = snapshot["states"]
                return row

            before = status_row()
            active_before = sum(page.refs > 0 for page in state.pool.pages)
            refs_before = sum(page.refs for page in state.pool.pages)
            reader = ObservedReader(memory._pipe_recv)
            monkeypatch.setattr(memory, "_pipe_recv", reader)
            assert any(page.refs for page in state.pool.pages)
            memory.force_free(handle)
            with pytest.raises(RuntimeError, match=f"KV cache handle {handle} is no longer available"):
                decode(handles, [200])
            assert not reader.waiting.is_set()
            assert before["active_sessions"] == 1
            assert before["active_pages"] == active_before > 0
            assert before["page_refs"] == refs_before > 0
            # A terminal arriving twice must leave the same released state.
            close(handles)
            assert session.closed and not state.sessions and not any(p.refs for p in state.pool.pages)
            assert handle not in memory._dynamo_failed_page_handles
            assert handle not in memory._allocated_tensors and handle not in memory._dynamo_session_release
            assert handle not in memory._dynamo_page_records and memory.current_size_tokens == 0
            after = status_row()
            assert after["active_sessions"] == after["active_pages"] == after["page_refs"] == 0
            # Same model runtime can serve another ordinary page request.
            _, _, fresh = request(list(range(11, 19)), finish=False)
            assert fresh != handles and decode(fresh, [201]).shape == (1, 1, 8)
    finally:
        if page_control._server is not None:
            page_control._server.shutdown()
            page_control._server.server_close()


@pytest.mark.parametrize("cancel_while_waiting", [False, True])
def test_pending_allocation_and_live_handle(monkeypatch, cancel_while_waiting):
    with actual_attention_components(monkeypatch) as (_, manager, request, _, close):
        _, _, (live,) = request(list(range(1, 9)), finish=False)
        memory = manager.cache
        original = memory._allocated_tensors[live]
        pending = memory.handle_counter
        reader = ObservedReader(memory._pipe_recv)
        monkeypatch.setattr(memory, "_pipe_recv", reader)

        def consume():
            with memory.use_cache(live, pending) as pair:
                return pair

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(consume)
            assert reader.waiting.wait(2)
            if cancel_while_waiting:
                # A free arrives *after* entering the missing-handle loop.
                memory.force_free(live)
                with pytest.raises(RuntimeError, match=f"KV cache handle {live} is no longer available"):
                    future.result(timeout=4)
                assert pending not in memory._allocated_tensors
            else:
                descriptor = TensorDescriptor((1, 2, 4, 16), dtype=torch.float16, device=torch.device("cpu"))
                size = manager.get_allocation_size_tokens(descriptor)
                allocated = asyncio.run(memory._schedule_alloc(size, descriptor, timeout=0))
                assert allocated == (pending,)
                first, second = future.result(timeout=4)
                assert first is original and second is memory._allocated_tensors[pending]
                close(allocated)


def test_live_model_error_propagates_and_stays_failed_until_free(monkeypatch):
    with actual_attention_components(monkeypatch) as (backend, manager, request, decode, close):
        _, _, handles = request(list(range(1, 9)), finish=False)
        (handle,) = handles
        error = ValueError("forward failure")

        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr(backend, "_inference_step_uncached", fail)
        with pytest.raises(ValueError) as caught:
            decode(handles, [200])
        assert caught.value is error
        assert handle in manager.cache._dynamo_failed_page_handles
        close(handles)
        assert handle not in manager.cache._dynamo_failed_page_handles
        assert not any(p.refs for p in backend._dynamo_service_state.pool.pages)
