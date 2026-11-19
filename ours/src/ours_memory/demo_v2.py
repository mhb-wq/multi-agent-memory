"""Multi-user integration with real MemOS/Qdrant and explicitly synthetic models."""

import json
from pathlib import Path

from .config import Settings
from .control import Actor
from .demo import SPECS, T0, T1, DemoDecisions, FixtureHTTPProvider, native_config
from .memos_backend import MemOSBackend
from .multiuser import MultiUserMemory
from .scheduler import MemOSSleepScheduler
from .state import state_view
from .store import Store


class Decisions(DemoDecisions):
    def complete(self, stage, payload):
        if stage == "select":
            self.calls.append(stage)
            return {"card_ids": [c["id"] for c in payload["cards"]]}
        return super().complete(stage, payload)


def run_demo(output):
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    raw = json.loads((Path(__file__).resolve().parents[2] / "configs/multiuser.json").read_text("utf8"))
    # This historical demonstration exercises the legacy governed workflow.
    raw["control"]["access_mode"] = "governed"
    settings = Settings(**raw["method"])
    settings.state_keys = {
        "travel_budget": {"type": "number", "unit": "CNY", "min": 0},
        "walking_constraint": {"type": "text", "unit": None},
    }
    store, model, checks = Store(str(directory / "state.sqlite3")), Decisions(), {}
    current = [T0]
    owner = Actor("marcus", "clinical", "care", "case")
    doctor = Actor("patel", "clinical", "care", "case")
    nina = Actor("nina", "logistics", "care", "case")
    martha = Actor("martha", "clinical", "care", "case")
    texts = list(SPECS)
    with FixtureHTTPProvider() as provider:
        cfg = native_config(directory / "qdrant", provider.base_url)
        backend = MemOSBackend(cfg, store)
        memory = MultiUserMemory(settings, backend, model, store, raw["control"], lambda: current[0])
        scheduler = MemOSSleepScheduler(memory.core)
        memory.scheduler = scheduler

        def write(who, name, index, category="logistics", subject="marcus"):
            result = memory.ingest(
                who,
                event_id=name,
                subject_id=subject,
                category=category,
                text=texts[index],
                scope="trip",
                item_id="trip-1",
            )
            assert result["status"] == "done"
            return result

        def read(who=owner, subject="marcus"):
            return memory.read(who, subject_ids=[subject], query="出差预算和限制", scope="trip")

        try:
            old = write(doctor, "initial", 0)
            checks["author_not_subject"] = read()["cards"][0]["author_id"] == "patel"
            checks["no_state_on_write"] = not store.states()
            memory.sleep(owner)
            checks["sleep_creates_state"] = state_view(store.states()[0], current[0])["value"] == 8000
            issued = read(nina)
            checks["no_state_in_agent_response"] = set(issued) == {"cards", "bundle_id"}
            current[0] = T1
            new = write(doctor, "update", 1)
            checks["pending_evidence_survives_stale_state"] = new["card_ids"][0] in [
                c["id"] for c in read()["cards"]
            ]
            memory.sleep(owner)
            checks["old_current_card_excluded"] = old["card_ids"][0] not in [c["id"] for c in read()["cards"]]
            checks["old_bundle_invalidated"] = not memory.validate_bundle(nina, issued["bundle_id"])["ready"]
            write(owner, "private", 2, "clinical")
            checks["private_category_excluded"] = "健康私有" not in json.dumps(read(nina), ensure_ascii=False)
            write(martha, "initial", 0, subject="martha")
            checks["other_subject_denied"] = not read(owner, "martha")["cards"]
            checks["other_subject_owner_can_read"] = bool(read(martha, "martha")["cards"])
            latest = read(nina)
            memory.control.revoke(owner, "nina-logistics")
            checks["revocation_invalidates_bundle"] = not memory.validate_bundle(nina, latest["bundle_id"])[
                "ready"
            ]
            backend.close()
            backend = MemOSBackend(cfg, store)
            memory.backend = memory.core.backend = backend
            checks["qdrant_reopen"] = bool(read()["cards"])
            deletion = memory.forget(owner, [old["event_id"], new["event_id"]])
            checks["deletion_complete"] = deletion["status"] == "forgotten"
            checks["deleted_cards_absent"] = not any(
                store.get("card", k) for k in old["card_ids"] + new["card_ids"]
            )
            # Verify native collections for the actual erased partition, not just facade filtering.
            from .backend import partition

            policy = partition(
                dict(
                    user_id="marcus",
                    team_id="care",
                    allowed_reader_ids=["clinical", "logistics"],
                    category="logistics",
                )
            )
            checks["native_deleted_partition_empty"] = not backend._cube(policy, "marcus").text_mem.get_all()
            native_class = type(next(iter(backend.cubes.values())).text_mem)
            checks["real_native_memos"] = native_class.__module__ == "memos.memories.textual.general"
            checks["real_native_scheduler"] = type(scheduler.dispatcher).__module__.startswith("memos.")
            report = {
                "mode": "engineering_fixture_not_model_quality",
                "passed": all(checks.values()),
                "checks": checks,
                "native_calls": provider.calls,
                "method_calls": model.calls,
            }
            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf8")
            if not report["passed"]:
                raise AssertionError(checks)
            return report
        finally:
            scheduler.close()
            backend.close()
            store.close()
