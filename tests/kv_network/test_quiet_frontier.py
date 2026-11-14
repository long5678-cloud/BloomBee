"""LMCache v0.5.7 /status field contract for both independent gates."""
import pytest
from bloombee.kv_cache.network.history_candidates import stream_state
from bloombee.kv_cache.network.independent_audit import frontier


@pytest.fixture
def status():
    return {
        "ingest": {"batches_applied": 32, "streams": {"source": {
            "last_seq": 30, "incarnation": 7, "gap_detected": False,
            "missing_batches": 0, "events_dropped": 0,
            "seconds_since_last_event": 4572.85, "event_delivery_delay_seconds": 0.004}}},
        "registry": {"seconds_since_heartbeat": {"source": 0.181}},
        "consumers": {"KeyDirectory": {"batches_delivered": 32, "consume_failures": 0,
                                        "fence_failures": 0}}}


def test_quiet_healthy(status):
    assert stream_state(status, "source", 7)[0] == "fresh"
    assert frontier(status, ["source"], {})["source"]["last_seq"] == 30


@pytest.mark.parametrize("section,key,value", [
    ("stream", "event_delivery_delay_seconds", 61),
    ("stream", "event_delivery_delay_seconds", None),
    ("stream", "event_delivery_delay_seconds", -0.02),
    ("stream", "event_delivery_delay_seconds", float("nan")),
    ("stream", "event_delivery_delay_seconds", "omit"),
    ("stream", "last_seq", False),
    ("stream", "incarnation", True),
    ("stream", "gap_detected", True),
    ("stream", "missing_batches", 1),
    ("stream", "events_dropped", 1),
    ("ingest", "batches_applied", "omit"),
    ("heartbeat", "source", 11),
    ("heartbeat", "source", "omit"),
    ("consumer", "batches_delivered", 31),
    ("consumer", "consume_failures", 1),
    ("consumer", "fence_failures", 1),
    ("consumer", "consume_failures", "omit"),
    ("consumers", "KeyDirectory", "omit")])
def test_bad_or_unavailable_status_abstains(status, section, key, value):
    sections = {"stream": status["ingest"]["streams"]["source"], "ingest": status["ingest"],
                "heartbeat": status["registry"]["seconds_since_heartbeat"],
                "consumer": status["consumers"]["KeyDirectory"], "consumers": status["consumers"]}
    if value == "omit":
        del sections[section][key]
    else:
        sections[section][key] = value
    assert stream_state(status, "source")[0] != "fresh"
    with pytest.raises((KeyError, ValueError)):
        frontier(status, ["source"], {})


def test_tight_delivery_limit(status):
    assert stream_state(status, "source", max_event_age_s=0.001)[0] == "stale"
    with pytest.raises(ValueError, match="delivery"):
        frontier(status, ["source"], {"candidate_max_delivery_delay_s": 0.001})
