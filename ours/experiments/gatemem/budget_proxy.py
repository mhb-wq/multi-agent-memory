"""Local serial accounting proxy for a bounded exploratory benchmark run."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

LIMIT = 3_000_000
MARGIN = 120_000
OUT = Path(__file__).resolve().parents[2] / "runtime/gatemem-compare-20261117/ledger.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    raise RuntimeError("ledger already exists")
state = {"limit": LIMIT, "margin": MARGIN, "used": 0, "calls": [], "status": "running"}
lock = threading.Lock()


def save():
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(OUT)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with lock:
            if self.path != "/v1/chat/completions" or length > 100000:
                return self.reply(413, {"error": {"message": "request rejected by experiment limits"}})
            if state["status"] != "running" or state["used"] + MARGIN > LIMIT:
                state["status"] = "budget_stopped"
                save()
                return self.reply(429, {"error": {"message": "experiment budget stopped"}})
            try:
                payload = json.loads(body)
                if payload.get("model") != "gpt-6-sol" or payload.get("stream") is True or int(payload.get("max_tokens", 4096)) > 4096:
                    return self.reply(400, {"error": {"message": "unsupported experiment request"}})
                response = requests.post("https://api.ciyuan.fast/v1/chat/completions", data=body,
                    headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "Content-Type": "application/json"}, timeout=120)
                data = response.json()
                usage = data.get("usage") or {}
                tokens = usage.get("total_tokens")
                state["calls"].append({"status": response.status_code, "usage": usage,
                                      "served_model": data.get("model"), "worker": self.headers.get("X-Experiment-Worker", "unspecified")})
                if isinstance(tokens, int):
                    state["used"] += tokens
                elif response.ok:
                    state["status"] = "usage_missing_stopped"
                save()
                return self.reply(response.status_code, data)
            except Exception as exc:
                # Usage is unknowable on a timeout; conservatively stop subsequent calls.
                state["status"] = "request_uncertain_stopped"
                state["calls"].append({"error": type(exc).__name__})
                save()
                return self.reply(503, {"error": {"message": "experiment proxy request failed"}})

    def reply(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


save()
print("budget proxy ready", flush=True)
ThreadingHTTPServer(("127.0.0.1", 18173), Handler).serve_forever()
