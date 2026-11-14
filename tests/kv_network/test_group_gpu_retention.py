import json
import os
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
import torch

from bloombee.kv_cache import page_retention, retention_completion
from bloombee.kv_cache.dynamo import DynamoCache
from bloombee.server import page_control
from bloombee.server.dynamo_pages import EnginePagePool


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    states = {}
    for layer in range(32):
        state = SimpleNamespace(
            binding="test", lock=RLock(), sessions={}, remote_gpu_sources={}, remote_cpu_sources={},
            cache=DynamoCache(1, 16, context="test", expected_layers=2),
            pool=EnginePagePool(3, 16, 1, 2, dtype=torch.float32, device="cpu"),
        )
        states[f"model.{layer}"] = (None, state)
    plan = tmp_path / "plan.json"
    warm = tmp_path / "warm.jsonl"
    formal = tmp_path / "formal.jsonl"
    warm.touch()
    formal.touch()
    fake_root = Path("/dev/shm/test_group")
    # Only map the two frozen ledger paths on platforms without /dev/shm.
    files = {fake_root / warm.name: warm, fake_root / formal.name: formal}
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: real_is_file(files.get(p, p)))
    real_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: real_read(files.get(p, p)))
    monkeypatch.setenv("BLOOMBEE_PAGE_GROUP_CONFIGS", str(plan))
    monkeypatch.setenv("BLOOMBEE_RETAINED_GPU_BLOCKS", "2")
    for key in ("BLOOMBEE_RETENTION_POLICY", "BLOOMBEE_HISTORY_TIER",
                "BLOOMBEE_RETENTION_TERMINALS", "BLOOMBEE_PAGE_FORMAL_LEDGER"):
        monkeypatch.setenv(key, "test-original")
        monkeypatch.delenv(key)
    monkeypatch.setattr(page_control, "_states", states)
    monkeypatch.setattr(page_control, "_demand_reservations", {})
    monkeypatch.setattr(page_control, "_remote_allowed", False)
    monkeypatch.setattr(page_control, "_page_epoch", 0)
    monkeypatch.setattr(page_control, "_startup_retained_gpu_blocks", page_control._unset)

    def switch(**settings):
        group = dict(policy="dynamo", tier="gpu", mode="remote",
                     warm_ledger=str(fake_root / warm.name), formal_ledger=str(fake_root / formal.name))
        plan.write_text(json.dumps({"test": dict(group, **settings)}))
        return page_control.configure_group({"group_id": "test"})

    return [item[1] for item in states.values()], warm, switch


def test_zero_retains_no_idle_prefix_after_two_complete_requests(runtime):
    states, ledger, switch = runtime
    switch(retained_gpu_blocks=0)
    tokens = list(range(16))
    key = ("test", tuple(tokens))
    for seq in (1, 2):
        attempt = f"trace-{seq}"
        # Exercise actual pool allocation and the retention terminal, with CPU tensors.
        for state in states:
            cfg = dict(attempt_id=attempt, token_ids=tokens)
            receipt = page_retention.begin_prefill(state, cfg)
            assert receipt["cpu"]["action"] == "disabled"
            assert receipt["gpu"]["action"] == "no_idle_capacity"
            session = state.pool.begin([key])
            assert session.length == 0
            value = torch.ones(16, 1, 2)
            session.append(value, value)
            assert session.length == 16
            page_retention.stage(state, session, cfg, None)
            session.release()
        event = dict(request_id=attempt, attempt_id=attempt, status="complete_success",
                     kv_reuse_success=False, actual_output_tokens=1, requested_output_tokens=1,
                     completion_event_seq=seq, completed_monotonic=float(seq), terminal_ref=attempt,
                     token_ids=tokens, prefix_key=retention_completion.prefix_key(tokens))
        with ledger.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        for state in states:
            end = page_retention.terminal(state, dict(attempt_id=attempt, token_ids=tokens, status="complete_success"))
            assert not end["gpu_resident"] and not end["cpu_resident"]
            assert state.pool.capacity_pages == 3
    assert retention_completion.snapshot()["sequence"] == 2


@pytest.mark.parametrize("invalid", [True, None, "1", -1, 4])
def test_reject_before_changing_any_layer_or_environment(runtime, invalid):
    states, _, switch = runtime
    states[0].remote_gpu_sources["kept"] = "marker"
    with pytest.raises(ValueError, match="retained_gpu_blocks"):
        switch(retained_gpu_blocks=invalid)
    assert states[0].remote_gpu_sources == {"kept": "marker"}
    assert "BLOOMBEE_RETENTION_POLICY" not in os.environ
    assert page_control._page_epoch == 0


def test_minimum_layer_capacity_validated_and_unset_restores_inherited_limit(runtime):
    states, _, switch = runtime
    states[-1].pool = EnginePagePool(1, 16, 1, 2, dtype=torch.float32, device="cpu")
    states[0].remote_gpu_sources["kept"] = "marker"
    with pytest.raises(ValueError, match="retained_gpu_blocks"):
        switch(retained_gpu_blocks=2)
    assert states[0].remote_gpu_sources == {"kept": "marker"}
    assert page_control._page_epoch == 0
    switch(retained_gpu_blocks=1)
    cfg = dict(attempt_id="one", token_ids=list(range(16)))
    assert page_retention.begin_prefill(states[0], cfg)["gpu"]["capacity"] == 1
    page_retention.terminal(states[0], dict(status="cancelled", attempt_id="one"))
    with pytest.raises(ValueError, match="budget"):
        switch()
    states[-1].pool = EnginePagePool(3, 16, 1, 2, dtype=torch.float32, device="cpu")
    switch()
    assert page_retention.begin_prefill(states[0], cfg | {"attempt_id": "two"})["gpu"]["capacity"] == 2


def test_native_both_tiers_honor_the_limit_when_returning_pages(runtime):
    states, _, switch = runtime
    switch(tier="both", retained_gpu_blocks=0)
    state = states[0]
    session = state.pool.begin()
    value = torch.ones(16, 1, 2)
    session.append(value, value)
    session.publish([("test", tuple(range(16)))])
    session.release()
    assert not state.pool.lookup


def test_active_references_block_group_switch_without_reset(runtime):
    states, _, switch = runtime
    switch(retained_gpu_blocks=1)
    session = states[-1].pool.begin()
    x = torch.ones(16, 1, 2)
    session.append(x, x)
    epoch = page_control._page_epoch
    try:
        with pytest.raises(ValueError, match="drain"):
            switch(retained_gpu_blocks=0)
        assert os.environ["BLOOMBEE_RETAINED_GPU_BLOCKS"] == "1"
        assert page_control._page_epoch == epoch
        assert session.length == 16
    finally:
        session.release()


def test_unset_group_restores_unconfigured_process_default(runtime, monkeypatch):
    states, _, switch = runtime
    monkeypatch.delenv("BLOOMBEE_RETAINED_GPU_BLOCKS")
    switch(retained_gpu_blocks=0)
    switch()
    assert "BLOOMBEE_RETAINED_GPU_BLOCKS" not in os.environ
    plan = page_retention.begin_prefill(states[0], dict(attempt_id="default", token_ids=list(range(16))))
    assert plan["gpu"]["capacity"] == states[0].pool.capacity_pages
