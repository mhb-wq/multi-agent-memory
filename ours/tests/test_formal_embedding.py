import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ours_memory.groupmem import ReplayClock, Trace, build_memory, close_memory


def test_formal_embedding_uses_shared_loopback_service(tmp_path, monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            vectors = [{"object": "embedding", "index": i, "embedding": [0.1] * 384}
                       for i, _ in enumerate(payload["input"])]
            body = json.dumps({"object": "list", "model": payload["model"], "data": vectors,
                               "usage": {"prompt_tokens": 0, "total_tokens": 0}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    protocol = {"memory_model": "fixture-model", "reasoning_policy": "provider_default",
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 384,
                              "base_url": base},
                "method": {"joint_state_update_enabled": True, "immediate_state_maintenance_enabled": True,
                           "independent_state_read_enabled": True}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setenv("OURS_LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    memory = None
    try:
        memory = build_memory(tmp_path, {"space": "case", "users": ["alice"], "access_mode": "shared_pool"},
                              Trace(tmp_path / "trace"), ReplayClock("2026-01-01T00:00:00+00:00"))
        vectors = memory.backend.embed_state_texts("case", ["fixture text"])
        assert len(vectors) == 1 and len(vectors[0]) == 384
        assert calls[0][0] == "/v1/embeddings"
        assert calls[0][1]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert calls[0][1]["input"] == ["fixture text"]
        assert memory.settings.model == "fixture-model"
        assert any(record["stage"] == "memos_embed" for record in memory.store.call_records())
    finally:
        if memory is not None:
            close_memory(memory)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
