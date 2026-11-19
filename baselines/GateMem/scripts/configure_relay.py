"""Save a supplied relay key locally and check its advertised model IDs.

Run from GateMem: python scripts/configure_relay.py
The key is read from stdin, never displayed, and only written to ignored .env.
"""
from __future__ import annotations

import json
import getpass
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".env"
URL = "https://api.ciyuan.fast/v1/models"


def main() -> int:
    if DEST.exists():
        print("A local .env already exists; leaving it untouched.")
        return 1
    key = (getpass.getpass("") if sys.stdin.isatty() else sys.stdin.readline()).strip()
    if not key or any(ch in key for ch in "\r\n\0"):
        print("No valid key was received on stdin.")
        return 1
    fd = os.open(str(DEST), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write("CIYUAN_API_KEY=" + key + "\n")
        f.write("CIYUAN_BASE_URL=https://api.ciyuan.fast/v1\n")
    request = urllib.request.Request(URL, headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
            print("model_catalog_status:", response.status)
    except urllib.error.HTTPError as exc:
        print("model_catalog_http_status:", exc.code)
        return 2
    except Exception as exc:
        print("model_catalog_error:", type(exc).__name__)
        return 2
    if not isinstance(payload, dict):
        print("model_catalog_unexpected_format")
        return 2
    items = payload.get("data", [])
    ids = [v.get("id", "") for v in items if isinstance(v, dict) and isinstance(v.get("id"), str)]
    print("model_count:", len(ids))
    for model in ids:
        if any(needle in model.lower() for needle in ("deepseek", "gpt-5.6", "glm-5.3")):
            print("model:", model[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
