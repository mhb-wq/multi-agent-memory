"""Loopback relay recording response model IDs and usage without prompt content."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests


def start_relay(base: str, key: str, model: str, log_path: Path):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            if self.path != "/v1/chat/completions" or self.headers.get("Authorization") != "Bearer " + key:
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = self.rfile.read(size)
                data = json.loads(body)
                if data.get("model") != model or data.get("stream") is True:
                    self.send_error(400, "Unexpected model or streaming request")
                    return
            except (ValueError, TypeError, json.JSONDecodeError):
                self.send_error(400, "Invalid request")
                return
            started = time.monotonic()
            try:
                upstream = requests.post(base + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json",
                                                  "User-Agent": "GateMem-Research-Client/1.0"},
                                         timeout=150, allow_redirects=False)
                response_body, status = upstream.content, upstream.status_code
                if status >= 400:
                    response_body = response_body.replace(key.encode("utf-8"), b"[REDACTED]")
                try:
                    result = upstream.json()
                except ValueError:
                    result = {}
                if not isinstance(result, dict):
                    result = {}
                choices = result.get("choices") or []
                finish = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
                delivered_status = status
                if status == 200:
                    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
                    if not isinstance(message, dict):
                        message = {}
                    empty = not str(message.get("content") or "").strip() and not message.get("tool_calls")
                    if finish == "length" or empty:
                        delivered_status = 502
                        response_body = json.dumps({"error": {"message": "Model response truncated or empty"}}).encode("utf-8")
                record = {"requested_model": model, "reported_model": result.get("model"),
                          "usage": result.get("usage"), "finish_reason": finish,
                          "http_status": status, "delivered_status": delivered_status,
                          "elapsed_s": round(time.monotonic() - started, 3)}
                with lock:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.send_response(delivered_status)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except requests.RequestException as exc:
                with lock:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"requested_model": model, "http_error": type(exc).__name__,
                                            "elapsed_s": round(time.monotonic() - started, 3)}) + "\n")
                self.send_error(502, "Relay transport error")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, name="relay-audit", daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"
