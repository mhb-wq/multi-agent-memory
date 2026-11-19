from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from ours_memory import Caller, Event, MemoryService, ReadRequest, Settings
from ours_memory.backend import FixtureBackend, partition, resolve_mem0_config
from ours_memory.config import boolean
from ours_memory.llm import ScriptedModel
from ours_memory.models import AccessDenied, MemoryChange, StaleRevision, ValidationError
from ours_memory.state import state_view
from ours_memory.store import Store

TIME = "2026-10-20T10:00:00+08:00"
NOV = "2026-11-01T00:00:00+08:00"


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(database=":memory:", team_agents={"team": ["dialogue", "planner", "secret"]})
        self.store = Store(":memory:")
        self.backend = FixtureBackend()
        self.model = ScriptedModel()
        self.service = MemoryService(self.settings, self.backend, self.model, self.store)
        self.caller = Caller("u1", "team", "dialogue")

    def tearDown(self):
        self.store.close()

    def test_model_provider_config_changes_without_changing_memory_logic(self):
        settings = Settings(
            model="comparison-model", base_url="http://127.0.0.1:9999/v1", api_key_env="TEST_MODEL_KEY"
        )
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "fixture-key"}):
            for provider, base_field in (("deepseek", "deepseek_base_url"), ("openai", "openai_base_url")):
                config = {
                    "llm": {"provider": provider, "config": {}},
                    "embedder": {"provider": "huggingface", "config": {"model": "fixture"}},
                }
                resolved = resolve_mem0_config(config, settings)
                self.assertEqual(resolved["llm"]["config"][base_field], settings.base_url)
                self.assertEqual(resolved["llm"]["config"]["model"], settings.model)
                self.assertEqual(config["llm"]["config"], {})

    def event(
        self,
        name="e1",
        value=12,
        *,
        readers=("dialogue", "planner"),
        recorded=TIME,
        text=None,
        roots=(),
        scope="exam",
        speaker="user",
    ):
        return Event(
            name,
            "u1",
            "team",
            "dialogue",
            "task1",
            text or f"每周可学习{value}小时",
            recorded,
            readers,
            speaker,
            scope,
            roots,
        )

    def enqueue(
        self,
        event,
        value=12,
        *,
        start=TIME,
        end=None,
        operation="INIT",
        kind="explicit_user_statement",
        key="available_study_hours_per_week",
        subject="u1",
        event_time=None,
    ):
        memory_id = "m_" + event.event_id
        self.backend.changes[event.event_id] = [MemoryChange(memory_id, event.text)]
        fields = {
            "memory_id": memory_id,
            "subject": subject,
            "evidence_kind": kind,
            "state_key": key,
            "value": value,
            "unit": "hours/week" if key else None,
            "scope": event.scope,
            "conditions": [],
            "event_time": event_time or event.recorded_at,
            "valid_from": start,
            "valid_to": end,
            "source_quote": event.text,
            "correction_of": None,
        }
        self.model.responses["cards"].append({"cards": [fields]})
        if key and self.settings.state_enabled:

            def update(payload):
                return {
                    "decisions": [
                        {
                            "state_id": state_key,
                            "expected_revision": group["state"]["revision"],
                            "operations": [{"kind": operation, "card_id": group["cards"][0]["id"]}],
                        }
                        for state_key, group in payload["groups"].items()
                    ]
                }

            self.model.responses["state"].append(update)
        return self.service.ingest_event(event, self.caller)[0]

    def current(self, at=TIME):
        return state_view(self.store.states()[0], at)

    def route(self, card_ids, *, at=NOV, purpose="current", caller=None, task="task2"):
        self.model.responses["route"].append(
            lambda p: {
                "state_ids": [state["id"] for state in p["states"]],
                "cards": [{"card_id": key, "use": "current"} for key in card_ids],
                "missing": [],
            }
        )
        return self.service.read_memory(
            ReadRequest(task, "安排英语学习", at, "exam", purpose=purpose),
            caller or Caller("u1", "team", "planner"),
        )

    def test_two_write_calls_and_one_joint_read_call(self):
        card = self.enqueue(self.event())
        result = self.route([card], at=TIME)
        self.assertEqual([stage for stage, _ in self.model.calls], ["cards", "state", "route"])
        self.assertEqual(result.states[0]["value"], 12)

    def test_sleep_retries_invalid_confirm_once_without_inventing_evidence(self):
        card = self.enqueue(self.event(), start=None)
        self.model.responses["sleep"].append(
            lambda p: {"state_id": p["state"]["id"], "operations": [{"kind": "CONFIRM", "card_id": card}]}
        )
        self.model.responses["sleep"].append(
            lambda p: {"state_id": p["state"]["id"], "operations": []}
        )
        result = self.service.review_changed_state(self.caller, "task1")
        self.assertEqual(result["status"], "completed")
        sleep_calls = [payload for stage, payload in self.model.calls if stage == "sleep"]
        self.assertEqual(len(sleep_calls), 2)
        self.assertIn("validation_feedback", sleep_calls[1])

    def test_future_update_and_history_preserved(self):
        first = self.enqueue(self.event())
        self.enqueue(
            self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, start=NOV, operation="UPDATE"
        )
        self.assertEqual(self.current(TIME)["value"], 12)
        self.assertEqual(self.current(NOV)["value"], 4)
        routed = self.route([first])
        self.assertFalse(routed.cards)
        self.assertTrue(routed.issues)

    def test_bounded_exception_restores_only_supported_default(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("e2", 2, recorded="2026-10-22T00:00:00+08:00"),
            2,
            start="2026-10-23T00:00:00+08:00",
            end="2026-10-30T00:00:00+08:00",
            operation="UPDATE",
        )
        self.assertEqual(self.current("2026-10-25T00:00:00+08:00")["value"], 2)
        self.assertEqual(self.current("2026-11-02T00:00:00+08:00")["value"], 12)

    def test_current_update_does_not_silently_cancel_future_arrangement(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("future", 4, recorded="2026-10-22T00:00:00+08:00"), 4, start=NOV, operation="UPDATE"
        )
        self.enqueue(
            self.event("now", 6, recorded="2026-10-23T00:00:00+08:00"),
            6,
            start="2026-10-23T00:00:00+08:00",
            operation="UPDATE",
        )
        self.assertEqual(self.current("2026-10-24T00:00:00+08:00")["value"], 6)
        self.assertEqual(self.current(NOV)["value"], 4)

    def test_old_event_imported_late_cannot_override(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("old", 30, recorded="2026-10-23T00:00:00+08:00"),
            30,
            start="2026-10-01T00:00:00+08:00",
            event_time="2026-10-01T00:00:00+08:00",
            operation="UPDATE",
        )
        self.assertEqual(self.current()["value"], 12)

    def test_unknown_after_expiry_does_not_invent_value(self):
        self.enqueue(self.event(), end=NOV)
        self.assertEqual(self.current(NOV)["status"], "unknown")

    def test_unresolved_effective_date_remains_pending(self):
        self.enqueue(self.event(), start=None)
        self.assertEqual(self.current()["status"], "pending")

    def test_unknown_initial_value_is_not_confirmed(self):
        self.enqueue(self.event(text="目前无法确定可用学习时间"), value=None)
        self.assertEqual(self.current()["status"], "unknown")

    def test_only_invalidation_without_replacement_is_unknown(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("e2", recorded="2026-10-22T00:00:00+08:00", text="之前的时间安排已不再适用"),
            None,
            start=NOV,
            operation="INVALIDATE",
        )
        self.assertEqual(self.current(NOV)["status"], "unknown")

    def test_external_course_fact_does_not_change_user_state(self):
        self.enqueue(self.event())
        card = self.enqueue(
            self.event("course", text="课程建议每周学习8小时", speaker="tool"),
            8,
            key=None,
            subject="course_A",
            kind="tool_observation",
        )
        self.assertEqual(len([s for s, _ in self.model.calls if s == "state"]), 1)
        self.assertEqual(self.current()["value"], 12)
        self.assertEqual(len(self.route([card], at=TIME).cards), 1)

    def test_same_event_retry_does_not_extract_or_update_twice(self):
        event = self.event()
        card = self.enqueue(event)
        before = len(self.model.calls)
        self.assertEqual(self.service.ingest_event(event, self.caller), [card])
        self.assertEqual(self.backend.add_count, 1)
        self.assertEqual(len(self.model.calls), before)

    def test_retelling_does_not_multiply_support(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("retell", recorded="2026-10-22T00:00:00+08:00", roots=("e1",), speaker="agent"),
            kind="retelling",
            operation="CONFIRM",
        )
        self.assertEqual(len(self.current()["assertions"][0]["source_root_ids"]), 1)
        self.assertEqual(len(self.current()["assertions"]), 1)
        self.service.finish_task("task1", self.caller)
        self.assertFalse(any(stage == "sleep" for stage, _ in self.model.calls))

    def test_conflict_cannot_be_used_as_confirmed(self):
        first = self.enqueue(self.event())
        self.enqueue(self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="CONFLICT")
        self.assertEqual(self.current()["status"], "conflict")
        result = self.route([first], at=TIME)
        self.assertFalse(result.cards)
        self.assertTrue(result.uncertain)

    def test_sleep_master_off_blocks_automatic_and_manual_calls(self):
        self.enqueue(self.event())
        self.enqueue(self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="CONFLICT")
        self.settings.sleep_enabled = False
        old = self.store.states()
        calls = len(self.model.calls)
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "disabled")
        self.assertEqual(self.service.review_changed_state(self.caller, "task1")["status"], "disabled")
        self.assertEqual(self.store.states(), old)
        self.assertEqual(len(self.model.calls), calls)
        self.assertTrue(self.store.dirty_states())

    def test_sleep_can_resolve_conflict_from_existing_validated_card(self):
        self.enqueue(self.event())
        second = self.enqueue(
            self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="CONFLICT"
        )
        self.model.responses["sleep"].append(
            lambda p: {
                "state_id": p["state"]["id"],
                "expected_revision": p["state"]["revision"],
                "operations": [{"kind": "UPDATE", "card_id": second}],
            }
        )
        answer = self.service.finish_task("task1", self.caller)
        self.assertEqual(answer["changed"], 1)
        self.assertEqual(self.current()["value"], 4)
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "already_completed")
        self.assertFalse(self.store.dirty_states())

    def test_sleep_keep_does_not_create_version_or_repeat_old_conflict(self):
        self.enqueue(self.event())
        self.enqueue(self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="CONFLICT")
        revision = self.store.states()[0]["revision"]
        self.model.responses["sleep"].append(
            lambda p: {
                "state_id": p["state"]["id"],
                "expected_revision": p["state"]["revision"],
                "operations": [],
            }
        )
        self.service.finish_task("task1", self.caller)
        self.assertEqual(self.store.states()[0]["revision"], revision)
        self.service.finish_task("task2", self.caller)
        self.assertEqual(len([s for s, _ in self.model.calls if s == "sleep"]), 1)

    def test_new_ignored_evidence_still_reaches_sleep(self):
        self.enqueue(self.event())
        self.store.clean_dirty(self.store.states()[0]["id"], self.store.dirty_states()[0]["generation"])
        self.enqueue(self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="IGNORE")
        self.assertTrue(self.store.dirty_states())

    def test_acl_filters_before_backend_and_routing(self):
        self.enqueue(self.event("private", readers=("dialogue", "secret")))
        caller = Caller("u1", "team", "planner")
        result = self.service.read_memory(ReadRequest("t", "learning", TIME, "exam"), caller)
        self.assertFalse(result.cards or result.states or result.uncertain)
        self.assertFalse(any(stage == "route" for stage, _ in self.model.calls))
        self.assertNotIn(
            partition(self.event("private", readers=("dialogue", "secret")).to_dict()),
            [p for row in self.backend.search_policies for p in row],
        )

    def test_retelling_cannot_expand_private_permissions(self):
        self.enqueue(self.event("private", readers=("dialogue", "secret")))
        expanded = self.event("retell", roots=("private",), speaker="agent")
        with self.assertRaises(AccessDenied):
            self.service.ingest_event(expanded, self.caller)

    def test_caller_cannot_impersonate_writer_or_cross_user(self):
        with self.assertRaises(AccessDenied):
            self.service.ingest_event(self.event(), Caller("u2", "team", "dialogue"))
        with self.assertRaises(AccessDenied):
            self.service.ingest_event(self.event(), Caller("u1", "team", "planner"))

    def test_wrong_quote_is_rejected_before_state_update_and_retry_recovers(self):
        event = self.event()
        self.backend.changes[event.event_id] = [MemoryChange("m", "12小时")]
        self.model.responses["cards"].append({"cards": [{"memory_id": "m", "source_quote": "不存在"}]})
        with self.assertRaises(ValidationError):
            self.service.ingest_event(event, self.caller)
        self.assertFalse(self.store.states())
        result = self.service.read_memory(ReadRequest("t", "学习", TIME, "exam"), self.caller)
        self.assertTrue(result.issues)
        self.assertFalse(result.cards or result.states)
        self.model.responses["cards"].append(
            {
                "cards": [
                    {
                        "memory_id": "m",
                        "source_quote": event.text,
                        "subject": "course",
                        "evidence_kind": "explicit_user_statement",
                        "scope": "exam",
                        "state_key": None,
                    }
                ]
            }
        )
        self.service.ingest_event(event, self.caller)
        self.assertEqual(self.backend.add_count, 1)

    def test_plan_checks_material_state_change(self):
        first = self.enqueue(self.event())
        self.route([first], at=NOV)
        planner = Caller("u1", "team", "planner")
        self.service.register_plan("p", "task2", planner)
        self.assertTrue(self.service.check_plan_dependencies("p", planner)["ready"])
        self.enqueue(
            self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, start=NOV, operation="UPDATE"
        )
        self.assertFalse(self.service.check_plan_dependencies("p", planner)["ready"])

    def test_forget_erases_sources_cards_history_and_blocks_replay(self):
        event = self.event(text="SECRET_USER_HOURS_12")
        self.enqueue(event)
        self.service.forget([event.event_id], self.caller)
        for kind in ("event", "card", "job"):
            self.assertNotIn("SECRET_USER_HOURS_12", json.dumps(self.store.all(kind)))
        self.assertFalse(self.store.history(self.store.states()[0]["id"]))
        self.assertFalse(self.backend.items)
        with self.assertRaises(AccessDenied):
            self.service.ingest_event(event, self.caller)

    def test_revision_guard_rejects_stale_proposal(self):
        self.enqueue(self.event())
        state = self.store.states()[0]
        with self.assertRaises(StaleRevision):
            self.store.commit_state(state, 0, task_id="t")

    def test_plan_preserves_basis_read_before_later_registration(self):
        first = self.enqueue(self.event())
        self.route([first], at=NOV)
        self.enqueue(
            self.event("e2", 4, recorded="2026-10-22T00:00:00+08:00"), 4, start=NOV, operation="UPDATE"
        )
        planner = Caller("u1", "team", "planner")
        self.service.register_plan("late-registration", "task2", planner)
        self.assertFalse(self.service.check_plan_dependencies("late-registration", planner)["ready"])

    def test_authorized_other_agent_can_check_plan(self):
        first = self.enqueue(self.event())
        self.route([first], at=TIME)
        self.service.register_plan("handoff", "task2", Caller("u1", "team", "planner"))
        self.assertTrue(self.service.check_plan_dependencies("handoff", self.caller)["ready"])
        with self.assertRaises(AccessDenied):
            self.service.check_plan_dependencies("handoff", Caller("u1", "team", "secret"))

    def test_unknown_route_card_is_rejected(self):
        self.enqueue(self.event())
        self.model.responses["route"].append(
            {"state_ids": [], "cards": [{"card_id": "invented", "use": "current"}]}
        )
        with self.assertRaises(ValidationError):
            self.service.read_memory(ReadRequest("t", "学习", TIME, "exam"), self.caller)

    def test_no_native_memory_means_no_extra_extraction_or_state_calls(self):
        self.service.ingest_event(self.event(), self.caller)
        self.assertFalse(self.store.all("card"))
        self.assertFalse(self.model.calls)

    def test_failed_native_erasure_blocks_new_writes_until_retry(self):
        self.enqueue(self.event())
        original = self.backend.delete
        self.backend.delete = lambda key: (_ for _ in ()).throw(RuntimeError("temporary index failure"))
        result = self.service.forget(["e1"], self.caller)
        self.assertEqual(result["pending_index_deletes"], 1)
        with self.assertRaises(RuntimeError):
            self.service.ingest_event(self.event("new"), self.caller)
        self.backend.delete = original
        self.assertEqual(self.service.retry_index_deletes(self.caller)["pending_index_deletes"], 0)
        self.assertFalse(self.backend.items)

    def test_state_and_history_survive_sqlite_reopen(self):
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "state.sqlite3")
        local = Store(path)
        try:
            self.enqueue(self.event())
            state = self.store.states()[0]
            local.commit_state(state, 0, task_id="task1")
            local.close()
            local = Store(path)
            self.assertEqual(state_view(local.states()[0], TIME)["value"], 12)
            self.assertEqual(len(local.history(state["id"])), 1)
        finally:
            local.close()
            # Remove only explicit known files; avoid an unchecked recursive deletion.
            for name in ("state.sqlite3", "state.sqlite3-wal", "state.sqlite3-shm"):
                file = os.path.join(folder, name)
                if os.path.exists(file):
                    os.unlink(file)
            os.rmdir(folder)

    def test_dirty_generation_keeps_newer_work(self):
        self.enqueue(self.event())
        item = self.store.dirty_states()[0]
        self.store.mark_dirty(item["state_id"], "later-task")
        self.store.clean_dirty(item["state_id"], item["generation"])
        self.assertTrue(self.store.dirty_states())

    def test_master_flag_strict_boolean_and_env_override(self):
        self.assertFalse(boolean("false"))
        with self.assertRaises(ValidationError):
            boolean("disabled-maybe")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as f:
            json.dump({"sleep_enabled": True}, f)
            path = f.name
        try:
            with patch.dict(os.environ, {"TEAM_MEMORY_SLEEP_ENABLED": "false"}):
                self.assertFalse(Settings.load(path).sleep_enabled)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
