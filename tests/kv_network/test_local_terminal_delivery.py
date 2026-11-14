"""Review-only failure injection; no model or remote node."""
import http.server
import json
import threading
from types import SimpleNamespace

import pytest
import torch
from test_review_regressions import Model, archived
from bloombee.kv_cache import page_retention as retention
from bloombee.kv_cache.dynamo import DynamoCache
from bloombee.kv_cache.network.lifecycle import LifecycleError
from bloombee.kv_cache.network.ordinary import OrdinaryClient
from bloombee.server.dynamo_pages import EnginePagePool


@pytest.mark.parametrize("mode,terminal_status", [
    ("cold", "complete_success"), ("observer_error", "cancelled"), ("model_error", "failed")
])
@pytest.mark.parametrize("failure", ["before_post", "lost_ack", "two_failures"])
def test_local_terminal_recovers_on_close(tmp_path, monkeypatch, mode, terminal_status, failure):
    log = tmp_path / "provider.log"
    log.touch()
    ledger = tmp_path / "client" / "history.jsonl"
    for key, value in dict(BLOOMBEE_RETENTION_POLICY="dynamo", BLOOMBEE_HISTORY_TIER="gpu",
                           BLOOMBEE_RETAINED_GPU_BLOCKS="1", BLOOMBEE_RETENTION_TERMINALS=str(ledger)).items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(binding="ctx", lock=threading.RLock(), remote_gpu_sources={}, remote_cpu_sources={},
                            cache=DynamoCache(1, 16, context="ctx", expected_layers=2),
                            pool=EnginePagePool(2, 16, 1, 2, dtype=torch.float32, device="cpu"))

    def stage(attempt, tokens):
        cfg = dict(attempt_id=attempt, token_ids=tokens)
        retention.begin_prefill(state, cfg)
        session = state.pool.begin()
        value = torch.ones(16, 1, 2)
        session.append(value, value)
        retention.stage(state, session, cfg, None)
        session.release()

    posts = []
    class Endpoint(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append(payload)
            fail = len(posts) <= (2 if failure == "two_failures" else 1)
            if fail and failure != "lost_ack":
                self.send_error(503)
                return
            retention.terminal(state, payload)
            if fail:
                self.send_error(503)
                return
            self.send_response(200)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def model_stage(attempt, tokens):
            stage(attempt, tokens)
            if mode == "model_error":
                raise RuntimeError("original model exception")
        client = OrdinaryClient(Model(log, (mode,), stage=model_stage), output_root=ledger.parent, context="ctx",
                                config=dict(cache_evidence=True, provider_log=str(log),
                                            control_endpoint=f"http://127.0.0.1:{server.server_port}"))
        with pytest.raises(RuntimeError, match="original model exception") if mode == "model_error" else pytest.raises(LifecycleError):
            client.generate(torch.tensor([archived()[1]]), max_new_tokens=3, request_id="request")
        if failure == "two_failures":
            with pytest.raises(LifecycleError, match="cache.terminal"):
                client.close()
            assert not client.closed
        client.close()
        client.close()
        assert client.closed and len(posts) == (3 if failure == "two_failures" else 2)
        assert all(p == posts[0] for p in posts)
        report = dict(mode=mode, statuses=[p["status"] for p in posts], client_closed=client.closed,
                      ledger_entries=len(ledger.read_text().splitlines()), pending=len(state.retention_pending),
                      refs=sum(p.refs for p in state.pool.pages))
        (tmp_path/"terminal_outcome.json").write_text(json.dumps(report, indent=2))
        assert posts[0]["status"] == terminal_status
        assert report["ledger_entries"] == int(mode == "cold")
        assert report["pending"] == 0 and report["refs"] == 0, report
        assert len(state.pool.lookup) == int(mode == "cold")
        assert not state.cache.entries
        assert len(state.retention_done) == client.attempt_sequence == 1
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
