"""Probe relay access without displaying credentials or saving responses."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
local = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        name, value = line.split("=", 1)
        local[name.strip()] = value.strip()
secret = local["CIYUAN_API_KEY"]
base = local["CIYUAN_BASE_URL"].rstrip("/")
headers = {"Authorization": "Bearer " + secret, "Content-Type": "application/json", "User-Agent": "GateMem-Research-Client/1.0"}


def probe(method: str, path: str, body=None):
    req = urllib.request.Request(base + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=35) as res:
            data = json.load(res)
            print(path, "http_status", res.status)
            if path == "/models":
                ids = [x.get("id") for x in data.get("data", []) if isinstance(x, dict) and isinstance(x.get("id"), str)]
                print("models", len(ids))
                for name in ids:
                    if any(x in name.lower() for x in ("deepseek", "gpt-5.6", "glm-5.3")):
                        print("model", name)
            else:
                print("response_has_choices", bool(data.get("choices")))
                print("reported_model", data.get("model", ""))
                choices = data.get("choices") or []
                message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
                print("content_nonempty", bool(str(message.get("content") or "").strip()))
                print("tool_calls_present", bool(message.get("tool_calls")))
    except urllib.error.HTTPError as exc:
        raw = exc.read(4096).decode("utf-8", errors="replace").replace(secret, "[REDACTED]")
        try:
            parsed = json.loads(raw)
            error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            message = error.get("message", "") if isinstance(error, dict) else str(error)
        except json.JSONDecodeError:
            message = "non-JSON error response"
        print(path, "http_status", exc.code, "reason", str(message)[:300])
    except Exception as exc:
        print(path, "error", type(exc).__name__)


probe("GET", "/models")
probe("POST", "/chat/completions", {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 32})
probe("POST", "/chat/completions", {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Call the echo tool with text set to OK."}], "tools": [{"type": "function", "function": {"name": "echo", "description": "Echo the text.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}], "tool_choice": "auto", "max_tokens": 128})
