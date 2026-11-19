"""Permanent reproductions for R1-R10 in the second review."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import test_memory as original
import test_regressions as first_review

from ours_memory import Caller, ReadRequest
from ours_memory.service import MemoryService
from ours_memory.backend import FixtureBackend, Mem0Backend
from ours_memory.migration import migrate_legacy_forget
from ours_memory.models import AccessDenied, MemoryChange, ValidationError, stable_id
from ours_memory.store import Store

TIME, NOV = original.TIME, original.NOV


class SecondReview(unittest.TestCase):
    setUp = original.MemoryTests.setUp
    tearDown = original.MemoryTests.tearDown
    event = original.MemoryTests.event
    enqueue = original.MemoryTests.enqueue
    route = original.MemoryTests.route
    current = original.MemoryTests.current
    fields = first_review.ReviewRegressions.fields
    raw = first_review.ReviewRegressions.raw

    def private_retelling(self):
        self.enqueue(self.event("source"))
        writer = self.caller
        self.caller = Caller("u1", "team", "planner")
        event = replace(
            self.event("copy", roots=("source",), readers=("planner",), speaker="agent"),
            writer_agent_id="planner",
        )
        self.raw(event, self.fields(event, "private-native", evidence_kind="retelling"))
        self.caller = writer

    def failed_event(self, name="bad"):
        event = self.event(name)
        self.backend.changes[name] = [MemoryChange("native-" + name, "unvalidated")]
        self.model.responses["cards"].append({"cards": []})
        with self.assertRaises(ValidationError):
            self.service.ingest_event(event, self.caller)

    def test_r1_source_owner_erases_private_copy_without_gaining_read_access(self):
        self.private_retelling()
        with self.assertRaises(AccessDenied):
            self.service.job_status("copy", self.caller)
        result = self.service.forget(["source"], self.caller)
        self.assertEqual(result["pending_index_deletes"], 0)
        self.assertFalse(self.backend.items)
        self.assertFalse(self.store.all("blocked_policy"))

    def test_r1_private_copy_failure_is_visible_to_cleanup_initiator_and_retryable(self):
        self.private_retelling()
        delete = self.backend.delete

        def unavailable(key):
            if key == "private-native":
                raise RuntimeError("offline")
            delete(key)

        self.backend.delete = unavailable
        result = self.service.forget(["source"], self.caller)
        self.assertGreater(result["pending_index_deletes"], 0)
        self.assertEqual(result["status"], "pending_cleanup")
        self.backend.delete = delete
        self.assertEqual(self.service.retry_index_deletes(self.caller)["pending_index_deletes"], 0)
        self.assertFalse(self.backend.items)

    def test_r1_cleanup_authority_survives_reopen_and_does_not_grant_other_agents_access(self):
        root = Path(__file__).resolve().parents[1] / "runtime" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=root)

        def cleanup():
            self.store.close()
            self.assertEqual(Path(directory.name).resolve().parent, root.resolve())
            directory.cleanup()

        self.addCleanup(cleanup)
        self.store.close()
        database = str(Path(directory.name) / "cleanup.sqlite3")
        self.store = Store(database)
        self.service = MemoryService(self.settings, self.backend, self.model, self.store)
        self.private_retelling()
        delete = self.backend.delete
        self.backend.delete = lambda key: (_ for _ in ()).throw(RuntimeError("offline"))
        self.service.forget(["source"], self.caller)
        self.store.close()
        self.store = Store(database)
        self.service = MemoryService(self.settings, self.backend, self.model, self.store)
        self.backend.delete = delete
        self.assertEqual(
            self.service.retry_index_deletes(Caller("u1", "team", "secret"))["pending_index_deletes"], 0
        )
        self.assertTrue(self.backend.items)
        self.assertEqual(
            self.service.retry_index_deletes(Caller("other-user", "team", "dialogue"))[
                "pending_index_deletes"
            ],
            0,
        )
        self.assertTrue(self.backend.items)
        self.assertEqual(self.service.retry_index_deletes(self.caller)["pending_index_deletes"], 0)
        self.assertFalse(self.backend.items)

    def test_r1_erasing_private_derivative_does_not_delete_unrelated_private_fact(self):
        self.private_retelling()
        owner = self.caller
        self.caller = Caller("u1", "team", "planner")
        unrelated = replace(self.event("unrelated", readers=("planner",)), writer_agent_id="planner")
        self.raw(unrelated, self.fields(unrelated, "unrelated-native"))
        self.caller = owner
        self.service.forget(["source"], owner)
        self.assertEqual(set(self.backend.items), {"unrelated-native"})
        result = self.service.read_memory(ReadRequest("t", "anything", TIME, "exam"), owner)
        self.assertFalse(result.cards or result.states)

    def test_r2_confirmed_with_pending_does_not_authorize_plan(self):
        card = self.enqueue(self.event())
        self.enqueue(
            self.event("guess", 4, speaker="agent"),
            4,
            kind="agent_inference",
            start="2026-10-25T00:00:00+08:00",
            operation="UPDATE",
        )
        result = self.route([card], at=NOV)
        self.assertFalse(result.states or result.cards)
        self.assertTrue(result.uncertain)
        planner = Caller("u1", "team", "planner")
        self.service.register_plan("p", "task2", planner)
        self.assertFalse(self.service.check_plan_dependencies("p", planner)["ready"])

    def test_r2_processing_issue_requires_reread_even_after_failed_event_discarded(self):
        card = self.enqueue(self.event())
        self.failed_event()
        result = self.route([card], at=TIME)
        self.assertTrue(result.uncertain)
        planner = Caller("u1", "team", "planner")
        self.service.register_plan("p", "task2", planner)
        self.service.discard_job("bad", self.caller)
        self.assertFalse(self.service.check_plan_dependencies("p", planner)["ready"])
        self.route([card], at=TIME)
        self.service.register_plan("p", "task2", planner)
        self.assertTrue(self.service.check_plan_dependencies("p", planner)["ready"])

    def test_r2_empty_read_with_unsettled_event_still_blocks_plan(self):
        self.failed_event()
        planner = Caller("u1", "team", "planner")
        result = self.service.read_memory(ReadRequest("task2", "study", TIME, "exam"), planner)
        self.assertTrue(result.issues)
        self.service.register_plan("p", "task2", planner)
        self.service.discard_job("bad", self.caller)
        self.assertFalse(self.service.check_plan_dependencies("p", planner)["ready"])

    def test_r3_confirm_null_is_rejected_without_changing_confirmed_value(self):
        self.enqueue(self.event())
        self.enqueue(self.event("confirm", text="仍然适用"), value=None, operation="CONFIRM")
        self.assertEqual(self.current()["value"], 12)
        self.assertEqual(self.service.job_status("confirm", self.caller)["status"], "partial")

    def test_r3_update_null_requires_explicit_invalidation(self):
        self.enqueue(self.event())
        self.enqueue(self.event("update", text="模型遗漏了数值"), value=None, operation="UPDATE")
        self.assertEqual(self.current()["value"], 12)

    def test_r3_incompatible_confirmation_cannot_create_a_second_confirmed_value(self):
        self.enqueue(self.event())
        self.enqueue(self.event("confirm", 4), 4, operation="CONFIRM")
        self.assertEqual(self.current()["value"], 12)
        self.assertEqual(self.service.job_status("confirm", self.caller)["status"], "partial")

    def test_r4_null_revision_uses_program_snapshot_for_online_update(self):
        self.enqueue(self.event())
        event = self.event("second", 4)
        self.backend.changes["second"] = [MemoryChange("m", event.text)]
        self.model.responses["cards"].append(
            {
                "cards": [
                    self.fields(event, state_key="available_study_hours_per_week", value=4, unit="hours/week")
                ]
            }
        )
        self.model.responses["state"].append(
            lambda p: {
                "decisions": [
                    {
                        "state_id": key,
                        "expected_revision": None,
                        "operations": [{"kind": "UPDATE", "card_id": g["cards"][0]["id"]}],
                    }
                    for key, g in p["groups"].items()
                ]
            }
        )
        self.service.ingest_event(event, self.caller)
        self.assertEqual(self.current()["value"], 4)
        self.assertEqual(self.service.job_status("second", self.caller)["status"], "done")

    def test_r4_null_revision_uses_program_snapshot_for_sleep(self):
        self.enqueue(self.event(), start=None)
        self.model.responses["sleep"].append(
            lambda p: {"state_id": p["state"]["id"], "expected_revision": None, "operations": []}
        )
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "completed")

    def test_r4_explicit_wrong_sleep_revision_is_still_rejected(self):
        self.enqueue(self.event(), start=None)
        self.model.responses["sleep"].append(
            lambda p: {"state_id": p["state"]["id"], "expected_revision": 0, "operations": []}
        )
        result = self.service.finish_task("task1", self.caller)
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(result["issues"][0]["code"], "StaleRevision")

    def test_r4_null_model_version_does_not_disable_commit_time_version_guard(self):
        self.enqueue(self.event())
        event = self.event("second", 4)
        self.backend.changes["second"] = [MemoryChange("m", event.text)]
        self.model.responses["cards"].append(
            {
                "cards": [
                    self.fields(event, state_key="available_study_hours_per_week", value=4, unit="hours/week")
                ]
            }
        )

        def racing_reply(payload):
            key, group = next(iter(payload["groups"].items()))
            previous = group["state"]
            self.store.commit_state(
                {**previous, "assertions": []}, previous["revision"], task_id="other-writer"
            )
            return {
                "decisions": [
                    {
                        "state_id": key,
                        "expected_revision": None,
                        "operations": [{"kind": "UPDATE", "card_id": group["cards"][0]["id"]}],
                    }
                ]
            }

        self.model.responses["state"].append(racing_reply)
        self.service.ingest_event(event, self.caller)
        self.assertFalse(self.store.states()[0]["assertions"])
        self.assertEqual(self.service.job_status("second", self.caller)["issues"][0]["code"], "StaleRevision")

    def test_r5_observation_calendar_date_preserves_original_offset(self):
        for observation in ("2026-10-01T00:30:00+08:00", "2026-10-01T23:30:00-05:00"):
            with self.subTest(observation=observation):
                native = MagicMock()
                native.custom_instructions = None
                native.get_all.return_value = {"results": []}
                native.add.return_value = {"results": []}
                backend = Mem0Backend(native)
                event = replace(self.event(), observed_at=observation)
                backend.add(event, "policy")
                self.assertIn("Observation Date is 2026-10-01 (", native.add.call_args.kwargs["prompt"])

    def test_r6_old_event_without_observed_at_replays_as_same_event(self):
        event = self.event()
        cards = self.enqueue(event)
        data = self.store.get("event", event.event_id)
        data.pop("observed_at")
        self.store.put("event", event.event_id, data)
        count = self.backend.add_count
        self.assertEqual(self.service.ingest_event(event, self.caller), [cards])
        self.assertEqual(self.backend.add_count, count)
        for changed in (
            replace(event, text="changed"),
            replace(event, allowed_reader_ids=("dialogue", "secret")),
        ):
            with self.subTest(changed=changed.allowed_reader_ids):
                with self.assertRaises(ValidationError):
                    self.service.ingest_event(changed, self.caller)

    def test_r6_failed_legacy_event_can_retry_without_extracting_native_memory_twice(self):
        self.failed_event()
        record = self.store.get("event", "bad")
        record.pop("observed_at")
        self.store.put("event", "bad", record)
        event = self.event("bad")
        self.model.responses["cards"].append({"cards": [self.fields(event, "native-bad")]})
        count = self.backend.add_count
        self.service.retry_job("bad", self.caller)
        self.assertEqual(self.service.job_status("bad", self.caller)["status"], "done")
        self.assertEqual(self.backend.add_count, count)

    def test_r6_legacy_tombstone_without_acl_reports_migration_instead_of_guessing(self):
        self.store.put("forgotten", stable_id("event", "u1", "team", "old"), {"at": TIME})
        with self.assertRaisesRegex(ValidationError, "旧.*权限"):
            self.service.forget(["old"], self.caller)

    def test_r6_offline_migration_restores_original_acl_without_resurrecting_event(self):
        key = stable_id("event", "u1", "team", "old")
        self.store.put("forgotten", key, {"at": TIME})
        metadata = [
            {
                "event_id": "old",
                "user_id": "u1",
                "team_id": "team",
                "allowed_reader_ids": ["dialogue", "planner"],
            }
        ]
        self.assertEqual(
            migrate_legacy_forget(self.store, metadata, self.settings.team_agents)["migrated"], 1
        )
        self.assertEqual(
            migrate_legacy_forget(self.store, metadata, self.settings.team_agents)["already_current"], 1
        )
        self.assertEqual(self.service.forget(["old"], self.caller)["status"], "already_forgotten")
        self.assertEqual(self.store.get("forgotten", key)["at"], TIME)
        with self.assertRaises(AccessDenied):
            self.service.ingest_event(self.event("old"), self.caller)
        with self.assertRaises(AccessDenied):
            self.service.forget(["old"], Caller("u1", "team", "secret"))

    def test_r6_migration_is_atomic_and_rejects_changed_or_unknown_permissions(self):
        key = stable_id("event", "u1", "team", "old")
        self.store.put("forgotten", key, {"at": TIME})
        valid = {"event_id": "old", "user_id": "u1", "team_id": "team", "allowed_reader_ids": ["dialogue"]}
        invalid = {**valid, "event_id": "missing"}
        with self.assertRaises(ValidationError):
            migrate_legacy_forget(self.store, [valid, invalid], self.settings.team_agents)
        self.assertEqual(self.store.get("forgotten", key), {"at": TIME})
        with self.assertRaises(ValidationError):
            migrate_legacy_forget(
                self.store, [{**valid, "allowed_reader_ids": ["unconfigured"]}], self.settings.team_agents
            )
        migrate_legacy_forget(self.store, [valid], self.settings.team_agents)
        with self.assertRaises(ValidationError):
            migrate_legacy_forget(
                self.store, [{**valid, "allowed_reader_ids": ["planner"]}], self.settings.team_agents
            )

    def test_r7_failed_job_without_dirty_state_is_not_completed(self):
        self.enqueue(self.event())
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "completed")
        self.failed_event()
        self.assertFalse(self.store.dirty_states())
        result = self.service.finish_task("task2", self.caller)
        self.assertEqual(result["status"], "needs_attention")
        self.assertTrue(any(issue.get("event_id") == "bad" for issue in result["issues"]))
        self.assertFalse(
            self.store.get("completion", stable_id("completion", "u1", "team", "dialogue", "task2"))
        )

    def test_r7_previous_completion_does_not_hide_later_failed_job(self):
        self.enqueue(self.event())
        self.service.finish_task("task1", self.caller)
        self.failed_event()
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "needs_attention")

    def test_r8_backend_missing_discovery_fails_before_any_write(self):
        class LegacyBackend:
            add = FixtureBackend.add
            search = FixtureBackend.search
            delete = FixtureBackend.delete

        with self.assertRaisesRegex(ValidationError, "find_event_memories"):
            MemoryService(self.settings, LegacyBackend(), self.model, self.store)
        self.assertFalse(self.store.all("event") or self.store.all("blocked_policy"))

    def test_r9_native_delete_retires_current_support_and_clears_history(self):
        card = self.enqueue(self.event())
        state_id = self.store.states()[0]["id"]
        self.assertTrue(self.store.history(state_id))
        memory_id = self.store.get("card", card)["memory_id"]
        self.backend.changes["delete"] = [MemoryChange(memory_id, "", "DELETE")]
        self.service.ingest_event(self.event("delete"), self.caller)
        self.assertFalse(self.store.history(state_id))
        self.assertFalse(self.store.states()[0]["assertions"])

    def test_r10_unicode_equivalent_text_values_do_not_conflict(self):
        for name, value in (("a", "Caf\u00e9"), ("b", "Cafe\u0301")):
            event = self.event(name, text=value)
            self.raw(event, self.fields(event, name, state_key="goal", value=value))
        self.assertEqual(self.current()["status"], "confirmed")
        self.assertEqual(self.current()["value"], "Caf\u00e9")
        view = self.current()
        self.assertEqual(
            self.service._fingerprint(view), self.service._fingerprint({**view, "value": "Cafe\u0301"})
        )
