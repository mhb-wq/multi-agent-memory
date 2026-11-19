from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-user Card-State-Sleep memory")
    parser.add_argument("--config", default=str(ROOT / "configs" / "multiuser.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        raw = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))

        def variables(value):
            if isinstance(value, dict):
                return set().union(*(variables(v) for v in value.values())) if value else set()
            if isinstance(value, list):
                return set().union(*(variables(v) for v in value)) if value else set()
            return {value[2:-1]} if isinstance(value, str) and value.startswith("${") and value.endswith("}") else set()

        try:
            version = importlib.metadata.version("MemoryOS")
        except importlib.metadata.PackageNotFoundError:
            version = None
        print(json.dumps({"root": str(ROOT), "memos_distribution": version,
                          "required_environment_present": {k: bool(os.environ.get(k)) for k in sorted(variables(raw))},
                          "identity_tokens_present": {r["token_env"]: bool(os.environ.get(r["token_env"]))
                                                      for r in raw.get("identities", [])},
                          "verification_scope": "configuration_presence_only_no_remote_call"},
                         ensure_ascii=False, indent=2))
        return

    import uvicorn

    from .api_v2 import create_app
    from .runtime import build

    service, _, identities = build(args.config)
    try:
        app = create_app(service, identities)
        service.recover()
        uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
    finally:
        service.scheduler.close()
        service.backend.close()
        service.store.close()


if __name__ == "__main__":
    main()
