from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from .models import Caller, Event, ReadRequest

ROOT = Path(__file__).resolve().parents[2]


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="MemOS + governed Card-State-Sleep research service")
    parser.add_argument("--config", default=str(ROOT / "configs" / "multiuser.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    demo = sub.add_parser("demo")
    demo.add_argument("--output", required=True, help="New directory; never overwrite an existing database")
    demo_v2 = sub.add_parser("demo-v2")
    demo_v2.add_argument("--output", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    for name in ("ingest", "read", "forget", "trace", "retry", "job", "finish", "plan", "check-plan"):
        command = sub.add_parser(name)
        command.add_argument("--user", required=True)
        command.add_argument("--team", required=True)
        command.add_argument("--agent", required=True)
        if name in {"ingest", "read", "forget"}:
            command.add_argument("--input", required=True, help="UTF-8 JSON file")
        else:
            command.add_argument("--id", required=True)
        if name == "plan":
            command.add_argument("--task", required=True)
    acl = sub.add_parser("acl", help="Trusted local administrator only; stop the service first")
    acl.add_argument("--policy", required=True)
    acl.add_argument("--agent", required=True)
    acl.add_argument("--read", choices=["allow", "deny"])
    acl.add_argument("--write", choices=["allow", "deny"])
    disclosure = sub.add_parser("approve-disclosure", help="Approve exact Event JSON; trusted administrator")
    disclosure.add_argument("--input", required=True)
    revoke = sub.add_parser("revoke-disclosure")
    revoke.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        raw = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))

        def variables(item):
            if isinstance(item, dict):
                return set().union(*(variables(v) for v in item.values())) if item else set()
            if isinstance(item, list):
                return set().union(*(variables(v) for v in item)) if item else set()
            return (
                {item[2:-1]}
                if isinstance(item, str) and item.startswith("${") and item.endswith("}")
                else set()
            )

        try:
            version = importlib.metadata.version("MemoryOS")
        except importlib.metadata.PackageNotFoundError:
            version = None
        output(
            {
                "root": str(ROOT),
                "memos_distribution": version,
                "required_environment_present": {
                    name: bool(os.environ.get(name)) for name in sorted(variables(raw))
                },
                "identity_tokens_present": {
                    row["token_env"]: bool(os.environ.get(row["token_env"]))
                    for row in raw.get("identities", [])
                },
                "verification_scope": "configuration_presence_only_no_remote_call",
            }
        )
        return
    if args.command in {"demo", "demo-v2"}:
        if args.command == "demo-v2":
            from .demo_v2 import run_demo
        else:
            from .demo import run_demo

        # Local provider must not be routed through an ambient HTTP proxy.
        os.environ["NO_PROXY"] = ",".join(
            filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"])
        )
        report = run_demo(args.output)
        output(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "report": str(Path(args.output).resolve() / "report.json"),
                "mode": report["mode"],
            }
        )
        return
    if args.command in {"acl", "approve-disclosure", "revoke-disclosure"}:
        from .governance import AccessGraph
        from .store import Store

        raw = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if raw.get("mode") == "multiuser-v2":
            raise ValueError("多用户控制操作必须使用v2已认证接口，不能使用旧版管理员命令")
        store = Store(str(Path(args.config).resolve().parent.parent / raw["method"]["database"]))
        graph = AccessGraph(store)
        try:
            if args.command == "acl":
                from .backend import partition

                spec = raw["policies"][args.policy]
                policy = partition(
                    dict(user_id=spec["user_id"], team_id=spec["team_id"], allowed_reader_ids=spec["readers"])
                )
                output(
                    {
                        "policy": policy,
                        "revision": graph.change(
                            policy,
                            args.agent,
                            read=None if args.read is None else args.read == "allow",
                            write=None if args.write is None else args.write == "allow",
                        ),
                    }
                )
            elif args.command == "approve-disclosure":
                event = Event.from_dict(json.loads(Path(args.input).read_text(encoding="utf-8-sig")))
                graph.approve_disclosure(event)
                output({"approved": event.disclosure_id})
            else:
                graph.revoke_disclosure(args.id)
                output({"revoked": args.id})
        finally:
            store.close()
        return
    from .runtime import build

    service, policies, identities = build(args.config)
    try:
        if args.command == "serve":
            import uvicorn

            from .multiuser import MultiUserMemory

            if isinstance(service, MultiUserMemory):
                from .api_v2 import create_app

                app = create_app(service, identities)
                service.recover()
            else:
                from .api import create_app

                service.scheduler.recover()
                app = create_app(service, policies, identities)
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                workers=1,
                access_log=False,
            )
            return
        from .multiuser import MultiUserMemory

        if isinstance(service, MultiUserMemory):
            raise ValueError("多用户模式使用v2 HTTP接口；旧版CLI操作请显式指定configs/deepseek.json")
        caller = Caller(args.user, args.team, args.agent)
        body = (
            json.loads(Path(args.input).read_text(encoding="utf-8-sig")) if hasattr(args, "input") else None
        )
        if args.command == "ingest":
            output({"card_ids": service.ingest_event(Event.from_dict(body), caller)})
        elif args.command == "read":
            output(service.read_memory(ReadRequest(**body), caller).to_dict())
        elif args.command == "finish":
            output(service.scheduler.submit(args.id, caller))
        elif args.command == "forget":
            output(service.forget(body["source_ids"], caller))
        elif args.command == "trace":
            output(service.get_trace(args.id, caller))
        elif args.command == "retry":
            output(service.retry_job(args.id, caller))
        elif args.command == "job":
            output(service.job_status(args.id, caller))
        elif args.command == "plan":
            output(service.register_plan(args.id, args.task, caller))
        elif args.command == "check-plan":
            output(service.check_plan_dependencies(args.id, caller))
    finally:
        service.scheduler.close()
        service.backend.close()
        service.store.close()


if __name__ == "__main__":
    main()
