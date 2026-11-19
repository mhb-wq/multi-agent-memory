"""A loopback-only OpenAI-compatible bridge for the chosen local embedder."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


def start_server():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu", local_files_only=True)
    model.max_seq_length = 256
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Never log input text or authorization headers.
            return

        def do_POST(self):
            try:
                if self.path != "/v1/embeddings":
                    self.send_error(404)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 8 * 1024 * 1024:
                    self.send_error(413)
                    return
                request = json.loads(self.rfile.read(size))
                if request.get("model") not in {MODEL, "all-MiniLM-L6-v2"}:
                    raise ValueError("Unexpected embedding model")
                if request.get("dimensions") not in (None, DIMENSION):
                    raise ValueError("Embedding dimensions must be 384")
                texts = request.get("input")
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
                    raise ValueError("Embedding input must be text or a list of text")
                with lock:
                    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                result = {"object": "list", "model": MODEL,
                          "data": [{"object": "embedding", "index": i, "embedding": v.tolist()}
                                   for i, v in enumerate(vectors)],
                          "usage": {"prompt_tokens": 0, "total_tokens": 0}}
                payload = json.dumps(result, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.send_error(400, "Invalid embedding request")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="local-minilm", daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"
