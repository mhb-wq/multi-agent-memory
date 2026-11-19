"""Synthetic engineering demo: REAL MemOS/Qdrant, FIXTURE LLM and vectors.

This deliberately uses a local HTTP provider to exercise upstream SDKs and schemas.
It is not a semantic quality evaluation or a remote-model result.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Settings
from .governance import AccessGraph, GovernedMemoryService
from .memos_backend import MemOSBackend
from .models import AccessDenied, Caller, Event, ReadRequest
from .store import Store

T0 = "2026-10-20T09:00:00+08:00"
T1 = "2026-10-21T09:00:00+08:00"
T2 = "2026-10-22T09:00:00+08:00"
T3 = "2026-10-23T09:00:00+08:00"
SPECS = {
    "本次杭州出差机票加酒店预算8000元。": ("travel_budget", 8000, "CNY", "INIT"),
    "本次杭州出差机票加酒店预算正式改为5000元。": ("travel_budget", 5000, "CNY", "UPDATE"),
    "健康私有记录：本周避免长距离步行。": ("walking_constraint", "避免长距离步行", None, "INIT"),
    "本次预算可能还要改，具体上限尚未确定。": ("travel_budget", None, "CNY", "CONFLICT"),
    "最终确认本次杭州出差机票加酒店预算4500元。": ("travel_budget", 4500, "CNY", "UPDATE"),
}


class FixtureHTTPProvider:
    def __enter__(self):
        self.calls = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.calls.append(self.path)
                if self.path.endswith("/embeddings"):
                    texts = body["input"]
                    if isinstance(texts, str):
                        texts = [texts]
                    rows = []
                    for index, text in enumerate(texts):
                        vector = [(v - 127.5) / 128 for v in hashlib.sha256(str(text).encode()).digest()[:16]]
                        rows.append(dict(object="embedding", index=index, embedding=vector))
                    result = dict(
                        object="list",
                        data=rows,
                        model="fixture-hash-16",
                        usage=dict(prompt_tokens=0, total_tokens=0),
                    )
                else:
                    prompt = body["messages"][-1]["content"]
                    found = [text for text in SPECS if text in prompt]
                    content = {
                        "memory list": [
                            {"key": "synthetic event", "value": text, "tags": []} for text in found
                        ]
                    }
                    result = dict(
                        id="fixture",
                        object="chat.completion",
                        created=0,
                        model="fixture-extractor",
                        choices=[
                            dict(
                                index=0,
                                finish_reason="stop",
                                message=dict(
                                    role="assistant", content=json.dumps(content, ensure_ascii=False)
                                ),
                            )
                        ],
                        usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    )
                data = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class DemoDecisions:
    """Exact synthetic-case oracle; never selected by build()/serve/live configuration."""

    def __init__(self):
        self.calls = []

    def complete(self, stage, payload):
        self.calls.append(stage)
        if stage == "cards":
            event = payload["event"]
            key, value, unit, _ = SPECS[event["text"]]
            return {
                "cards": [
                    dict(
                        memory_id=m["memory_id"],
                        subject=event["user_id"],
                        evidence_kind="explicit_user_statement",
                        state_key=key,
                        value=value,
                        unit=unit,
                        scope=event["scope"],
                        conditions=[],
                        event_time=event["recorded_at"],
                        valid_from=event["recorded_at"],
                        valid_to=None,
                        source_quote=event["text"],
                        correction_of=None,
                    )
                    for m in payload["memories"]
                ]
            }
        if stage == "state":
            return {
                "decisions": [
                    dict(
                        state_id=key,
                        operations=[
                            dict(
                                kind=SPECS[card["source_quote"]][3],
                                card_id=card["id"],
                                reason="synthetic fixture decision",
                            )
                            for card in group["cards"]
                        ],
                    )
                    for key, group in payload["groups"].items()
                ]
            }
        if stage == "route":
            return dict(
                state_ids=[s["id"] for s in payload["states"]],
                cards=[dict(card_id=c["id"], use="current") for c in payload["cards"]],
                missing=[],
            )
        if stage == "sleep":
            return dict(state_id=payload["state"]["id"], operations=[])
        raise ValueError(stage)


def native_config(directory, base_url):
    return {
        "directory": str(directory),
        "text_memory": {
            "backend": "general_text",
            "config": {
                "extractor_llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": "fixture-extractor",
                        "api_key": "synthetic-local-key",
                        "api_base": base_url,
                        "temperature": 0,
                        "max_tokens": 1024,
                    },
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {
                        "model_name_or_path": "fixture-hash-16",
                        "provider": "openai",
                        "api_key": "synthetic-local-key",
                        "base_url": base_url,
                        "embedding_dims": 16,
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": "overridden-per-policy",
                        "vector_dimension": 16,
                        "distance_metric": "cosine",
                    },
                },
            },
        },
    }


def run_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    db = directory / "state.sqlite3"
    if db.exists():
        raise ValueError("请选择新的输出目录，演示不会覆盖已有数据库")
    settings = Settings(
        database=str(db),
        team_agents={"travel": ["planner", "booking", "health"]},
        state_keys={
            "travel_budget": {"type": "number", "unit": "CNY", "min": 0},
            "walking_constraint": {"type": "text", "unit": None},
        },
    )
    store = Store(str(db))
    graph = AccessGraph(store)
    shared = graph.provision("xiaolin", "travel", ["planner", "booking"], ["planner", "booking"], ["planner"])
    graph.provision("xiaolin", "travel", ["health"], ["health"], ["health"])
    model = DemoDecisions()
    planner = Caller("xiaolin", "travel", "planner")
    booking = Caller("xiaolin", "travel", "booking")
    checks = {}
    with FixtureHTTPProvider() as provider:
        cfg = native_config(directory / "qdrant", provider.base_url)
        backend = MemOSBackend(cfg, store)
        service = GovernedMemoryService(settings, backend, model, store, graph)
        from .scheduler import MemOSSleepScheduler

        scheduler = MemOSSleepScheduler(service)
        texts = list(SPECS)

        def ingest(name, text, writer, time, readers):
            event = Event(
                name, "xiaolin", "travel", writer, "hangzhou", text, time, tuple(readers), scope="hangzhou"
            )
            ids = service.ingest_event(event, Caller("xiaolin", "travel", writer))
            assert service.job_status(name, Caller("xiaolin", "travel", writer))["status"] == "done"
            return ids

        def read(who, task="plan"):
            return service.read_memory(ReadRequest(task, "本次出差预算和限制", T3, "hangzhou"), who)

        try:
            old = ingest("budget-8000", texts[0], "planner", T0, ["planner", "booking"])
            assert read(booking).states[0]["value"] == 8000
            service.register_plan("booking-plan", "plan", booking)
            ingest("budget-5000", texts[1], "booking", T1, ["planner", "booking"])
            result = read(planner)
            checks["cross_agent_update"] = result.states[0]["value"] == 5000
            checks["stale_card_excluded"] = not any(c["id"] in old for c in result.cards)
            checks["stale_plan_blocked"] = not service.check_plan_dependencies("booking-plan", booking)[
                "ready"
            ]
            private = ingest("health-private", texts[2], "health", T0, ["health"])
            result = read(planner)
            checks["private_content_not_returned"] = "健康私有" not in json.dumps(
                result.to_dict(), ensure_ascii=False
            )
            try:
                service.get_card(private[0], planner)
                checks["private_id_denied"] = False
            except AccessDenied:
                checks["private_id_denied"] = True
            ingest("uncertain", texts[3], "planner", T2, ["planner", "booking"])
            checks["conflict_preserved"] = bool(read(booking).uncertain)
            sleep = {
                agent: scheduler.submit("hangzhou-review", Caller("xiaolin", "travel", agent))
                for agent in settings.team_agents["travel"]
            }
            checks["native_scheduler"] = type(scheduler.dispatcher).__module__.startswith("memos.")
            checks["sleep_review_called"] = "sleep" in model.calls
            checks["sleep_keeps_uncertainty"] = bool(read(booking).uncertain)
            ingest("final-4500", texts[4], "planner", T3, ["planner", "booking"])
            checks["explicit_resolution"] = read(booking).states[0]["value"] == 4500
            native_class = type(next(iter(backend.cubes.values())).text_mem)
            checks["real_native_class"] = native_class.__module__ == "memos.memories.textual.general"
            backend.close()
            backend = MemOSBackend(cfg, store)
            service.backend = backend
            checks["qdrant_reopen"] = read(booking).states[0]["value"] == 4500
            graph.change(shared, "booking", read=False)
            checks["revocation"] = not read(booking).states
            deletion = service.forget(["budget-8000", "budget-5000", "uncertain", "final-4500"], planner)
            checks["deletion_complete"] = deletion["status"] == "forgotten"
            checks["native_evidence_erased"] = not backend._cube(shared, "xiaolin").text_mem.get_all()
            checks["sidecar_evidence_erased"] = not any(
                e["event_id"].startswith("budget") for e in store.all("event")
            )
            report = dict(
                mode="engineering_fixture_not_model_quality",
                checks=checks,
                passed=all(checks.values()),
                sleep=sleep,
                native_calls=provider.calls,
                method_calls=model.calls,
                native_class=f"{native_class.__module__}.{native_class.__name__}",
                call_records=store.call_records(),
            )
            (directory / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not report["passed"]:
                raise AssertionError(checks)
            return report
        finally:
            scheduler.close()
            backend.close()
            store.close()
