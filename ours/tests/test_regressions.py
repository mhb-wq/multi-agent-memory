"""Reproductions of the 2026-10-23 review, including failure/recovery paths."""

from __future__ import annotations

import sqlite3
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_memory as support

from ours_memory import Caller, ReadRequest
from ours_memory.models import (
    AccessDenied,
    MemoryChange,
    OutputTruncated,
    ValidationError,
    canonical,
    stable_id,
)
from ours_memory.state import apply_operations

TIME, NOV = support.TIME, support.NOV


class ReviewRegressions(unittest.TestCase):
    setUp = support.MemoryTests.setUp
    tearDown = support.MemoryTests.tearDown
    event = support.MemoryTests.event
    enqueue = support.MemoryTests.enqueue
    current = support.MemoryTests.current
    route = support.MemoryTests.route

    def fields(self, event, memory_id="m", **overrides):
        return {
            "memory_id": memory_id,
            "subject": "u1",
            "evidence_kind": "explicit_user_statement",
            "state_key": None,
            "value": None,
            "unit": None,
            "scope": event.scope,
            "conditions": [],
            "event_time": event.recorded_at,
            "valid_from": TIME,
            "valid_to": None,
            "source_quote": event.text,
            **overrides,
        }

    def raw(self, event, fields, operation="INIT"):
        self.backend.changes[event.event_id] = [MemoryChange(fields["memory_id"], event.text)]
        self.model.responses["cards"].append({"cards": [fields]})
        if fields.get("state_key"):
            self.model.responses["state"].append(
                lambda p: {
                    "decisions": [
                        {
                            "state_id": key,
                            "expected_revision": group["state"]["revision"],
                            "operations": [{"kind": operation, "card_id": group["cards"][0]["id"]}],
                        }
                        for key, group in p["groups"].items()
                    ]
                }
            )
        return self.service.ingest_event(event, self.caller)

    def keep_sleep(self):
        self.model.responses["sleep"].append(
            lambda p: {
                "state_id": p["state"]["id"],
                "expected_revision": p["state"]["revision"],
                "operations": [],
            }
        )

    def test_p0_forget_failed_card_with_lost_job_changes_recovers_native_ids(self):
        event = self.event()
        self.backend.changes[event.event_id] = [MemoryChange("native-orphan", event.text)]
        self.model.responses["cards"].append({"cards": []})
        with self.assertRaises(ValidationError):
            self.service.ingest_event(event, self.caller)
        key = stable_id("event", "u1", "team", "e1")
        job = self.store.get("job", key)
        job.pop("changes")  # Simulate native success before sidecar recorded its result.
        self.store.put("job", key, job)
        result = self.service.forget(["e1"], self.caller)
        self.assertFalse(self.backend.items)
        self.assertEqual(result["pending_index_deletes"], 0)

    def test_p0_native_discovery_failure_persists_queue_after_sidecar_erasure(self):
        self.enqueue(self.event())
        original = self.backend.find_event_memories
        self.backend.find_event_memories = lambda *args: (_ for _ in ()).throw(RuntimeError("offline"))
        answer = self.service.forget(["e1"], self.caller)
        self.assertGreater(answer["pending_index_deletes"], 0)
        self.assertTrue(self.store.all("blocked_policy")[0]["event_ids"])
        self.backend.find_event_memories = original
        self.assertEqual(self.service.retry_index_deletes(self.caller)["pending_index_deletes"], 0)

    def test_p0_forget_erases_history_when_old_assertion_no_longer_current(self):
        first = self.enqueue(self.event())
        self.enqueue(self.event("new", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="UPDATE")
        state = self.store.states()[0]
        self.assertFalse(any(first in a["evidence_card_ids"] for a in state["assertions"]))
        self.assertIn(first, canonical(self.store.history(state["id"])))
        self.service.forget(["e1"], self.caller)
        self.assertNotIn(first, canonical(self.store.history(state["id"])))
        self.assertEqual(self.current()["value"], 4)

    def test_p0_inference_confirmation_cannot_survive_as_confirmed_after_forget(self):
        self.enqueue(self.event())
        inferred = self.enqueue(
            self.event("guess", speaker="agent"), kind="agent_inference", operation="CONFIRM"
        )
        confirmed = [a for a in self.current()["assertions"] if a["status"] == "confirmed"]
        self.assertFalse(any(inferred in a["evidence_card_ids"] for a in confirmed))
        self.service.forget(["e1"], self.caller)
        self.assertEqual(self.current()["status"], "pending")

    def test_p1_failed_event_does_not_block_unrelated_scope(self):
        self.enqueue(self.event("ok"))
        bad = self.event("bad", scope="other")
        self.backend.changes["bad"] = [MemoryChange("bad-m", "bad")]
        self.model.responses["cards"].append({"cards": []})
        with self.assertRaises(ValidationError):
            self.service.ingest_event(bad, self.caller)
        self.assertEqual(self.route([self.store.all("card")[0]["id"]], at=TIME).states[0]["value"], 12)
        self.assertEqual(self.service.job_status("bad", self.caller)["status"], "failed")
        self.service.discard_job("bad", self.caller)
        self.assertFalse(self.store.get("event", "bad"))

    def test_p1_failed_related_update_marks_old_value_uncertain_instead_of_silent_use(self):
        card = self.enqueue(self.event())
        bad = self.event("bad", 4)
        self.backend.changes["bad"] = [MemoryChange("bad-m", bad.text)]
        self.model.responses["cards"].append({"cards": []})
        with self.assertRaises(ValidationError):
            self.service.ingest_event(bad, self.caller)
        answer = self.route([card], at=TIME)
        self.assertFalse(answer.states or answer.cards)
        self.assertTrue(answer.uncertain and answer.issues)

    def test_p1_plan_can_be_refreshed_after_reread(self):
        card = self.enqueue(self.event())
        self.route([card], at=NOV)
        planner = Caller("u1", "team", "planner")
        self.service.register_plan("p", "task2", planner)
        new = self.enqueue(
            self.event("new", 4, recorded="2026-10-22T00:00:00+08:00"), 4, start=NOV, operation="UPDATE"
        )
        self.assertFalse(self.service.check_plan_dependencies("p", planner)["ready"])
        self.route([new], at=NOV)
        self.service.register_plan("p", "task2", planner)
        self.assertTrue(self.service.check_plan_dependencies("p", planner)["ready"])

    def test_p1_stale_model_revision_is_diagnosable_and_explicitly_retryable(self):
        self.enqueue(self.event())
        event = self.event("new", 4)
        fields = self.fields(
            event, "new-m", state_key="available_study_hours_per_week", value=4, unit="hours/week"
        )
        self.backend.changes[event.event_id] = [MemoryChange("new-m", event.text)]
        self.model.responses["cards"].append({"cards": [fields]})
        self.model.responses["state"].append(
            lambda p: {
                "decisions": [
                    {"state_id": key, "expected_revision": 0, "operations": []} for key in p["groups"]
                ]
            }
        )
        self.service.ingest_event(event, self.caller)
        issue = self.service.job_status("new", self.caller)["issues"][0]
        self.assertEqual(issue["code"], "StaleRevision")
        self.assertIn("1", issue["detail"])
        self.model.responses["state"].append(
            lambda p: {
                "decisions": [
                    {
                        "state_id": key,
                        "expected_revision": g["state"]["revision"],
                        "operations": [{"kind": "UPDATE", "card_id": g["cards"][0]["id"]}],
                    }
                    for key, g in p["groups"].items()
                ]
            }
        )
        count = self.backend.add_count
        self.service.retry_job("new", self.caller)
        self.assertEqual(self.service.job_status("new", self.caller)["status"], "done")
        self.assertEqual(self.current()["value"], 4)
        self.assertEqual(count, self.backend.add_count)

    def test_p1_revision_is_program_owned_even_when_model_omits_it(self):
        self.enqueue(self.event())
        event = self.event("second", 4)
        fields = self.fields(
            event, "new-m", state_key="available_study_hours_per_week", value=4, unit="hours/week"
        )
        self.backend.changes["second"] = [MemoryChange("new-m", event.text)]
        self.model.responses["cards"].append({"cards": [fields]})
        self.model.responses["state"].append(
            lambda p: {
                "decisions": [
                    {"state_id": key, "operations": [{"kind": "UPDATE", "card_id": group["cards"][0]["id"]}]}
                    for key, group in p["groups"].items()
                ]
            }
        )
        self.service.ingest_event(event, self.caller)
        self.assertEqual(self.current()["value"], 4)
        self.assertEqual(self.service.job_status("second", self.caller)["status"], "done")

    def test_p1_bad_card_does_not_discard_good_card_in_same_batch(self):
        event = self.event()
        self.backend.changes["e1"] = [MemoryChange("good", "good fact"), MemoryChange("bad", "bad fact")]
        self.model.responses["cards"].append(
            {"cards": [self.fields(event, "good"), self.fields(event, "bad", source_quote="invented")]}
        )
        cards = self.service.ingest_event(event, self.caller)
        self.assertEqual(len(cards), 1)
        job = self.service.job_status("e1", self.caller)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["issues"][0]["memory_id"], "bad")
        self.assertEqual(self.store.get("card", cards[0])["memory_id"], "good")

    def test_p1_input_limit_really_splits_independent_card_batches(self):
        event = self.event()
        self.settings.max_model_input_chars = 3000
        self.backend.changes["e1"] = [MemoryChange("a", "x" * 1500), MemoryChange("b", "y" * 1500)]
        for _ in range(2):
            self.model.responses["cards"].append(
                lambda p: {"cards": [self.fields(event, m["memory_id"]) for m in p["memories"]]}
            )
        self.assertEqual(len(self.service.ingest_event(event, self.caller)), 2)
        self.assertEqual(len([1 for stage, _ in self.model.calls if stage == "cards"]), 2)

    def test_p1_truncated_output_splits_batch_instead_of_retrying_same_size(self):
        event = self.event()
        self.backend.changes["e1"] = [MemoryChange("a", "a"), MemoryChange("b", "b")]
        self.model.responses["cards"].append(lambda p: (_ for _ in ()).throw(OutputTruncated("too long")))
        for _ in range(2):
            self.model.responses["cards"].append(
                lambda p: {"cards": [self.fields(event, m["memory_id"]) for m in p["memories"]]}
            )
        self.assertEqual(len(self.service.ingest_event(event, self.caller)), 2)
        self.assertEqual([len(p["memories"]) for stage, p in self.model.calls if stage == "cards"], [2, 1, 1])

    def test_p1_quotes_normalize_presentation_but_return_original_span(self):
        event = self.event(text="用户说：“Café  学习”。")
        card = self.raw(event, self.fields(event, source_quote='"Cafe\u0301 学习"'))[0]
        self.assertEqual(self.store.get("card", card)["source_quote"], "“Café  学习”")
        self.assertIsNone(self.service._source_quote("Café 每周12小时", event.text))

    def test_p1_state_group_error_does_not_discard_other_group(self):
        event = self.event(text="每周12小时，目标是通过考试")
        self.backend.changes["e1"] = [MemoryChange("hours", "12 hours"), MemoryChange("goal", "pass exam")]
        self.model.responses["cards"].append(
            {
                "cards": [
                    self.fields(
                        event,
                        "hours",
                        state_key="available_study_hours_per_week",
                        value=12,
                        unit="hours/week",
                    ),
                    self.fields(event, "goal", state_key="goal", value="pass"),
                ]
            }
        )

        def decisions(p):
            return {
                "decisions": [
                    {
                        "state_id": key,
                        "operations": [
                            {
                                "kind": "INVALID" if group["cards"][0]["state_key"] == "goal" else "INIT",
                                "card_id": group["cards"][0]["id"],
                            }
                        ],
                    }
                    for key, group in p["groups"].items()
                ]
            }

        self.model.responses["state"].append(decisions)
        self.service.ingest_event(event, self.caller)
        self.assertEqual(len(self.store.states()), 1)
        self.assertEqual(self.store.states()[0]["state_key"], "available_study_hours_per_week")
        job = self.service.job_status("e1", self.caller)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(len(job["states_done"]), 1)
        self.assertEqual(len(job["issues"]), 1)

    def test_p1_state_output_truncation_splits_independent_groups(self):
        event = self.event()
        self.backend.changes["e1"] = [MemoryChange("hours", "12 hours"), MemoryChange("goal", "pass exam")]
        self.model.responses["cards"].append(
            {
                "cards": [
                    self.fields(
                        event,
                        "hours",
                        state_key="available_study_hours_per_week",
                        value=12,
                        unit="hours/week",
                    ),
                    self.fields(event, "goal", state_key="goal", value="pass"),
                ]
            }
        )
        self.model.responses["state"].append(lambda p: (_ for _ in ()).throw(OutputTruncated("too long")))
        for _ in range(2):
            self.model.responses["state"].append(
                lambda p: {
                    "decisions": [
                        {
                            "state_id": key,
                            "operations": [{"kind": "INIT", "card_id": group["cards"][0]["id"]}],
                        }
                        for key, group in p["groups"].items()
                    ]
                }
            )
        self.service.ingest_event(event, self.caller)
        self.assertEqual(len(self.store.states()), 2)
        self.assertEqual([len(p["groups"]) for stage, p in self.model.calls if stage == "state"], [2, 1, 1])

    def test_p1_reusing_one_card_does_not_remove_independent_support(self):
        first = self.enqueue(self.event())
        second = self.enqueue(self.event("second"), operation="CONFIRM")
        cards = {c["id"]: c for c in self.store.all("card")}
        result = apply_operations(self.store.states()[0], [{"kind": "UPDATE", "card_id": second}], cards)
        self.assertTrue(any(first in a["evidence_card_ids"] for a in result["assertions"]))
        self.assertTrue(any(second in a["evidence_card_ids"] for a in result["assertions"]))

    def test_p1_numeric_equivalence_and_plan_fingerprint_are_consistent(self):
        self.enqueue(self.event())
        self.enqueue(self.event("float"), 12.0)
        self.assertEqual(self.current()["status"], "confirmed")
        view = self.current()
        self.assertEqual(self.service._fingerprint(view), self.service._fingerprint({**view, "value": 12.0}))

    def test_p1_sleep_reviews_conflict_without_start(self):
        self.enqueue(self.event(), start=None)
        self.enqueue(self.event("conflict", 4), 4, start=None, operation="CONFLICT")
        self.keep_sleep()
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "completed")
        self.assertEqual(len([1 for stage, _ in self.model.calls if stage == "sleep"]), 1)

    def test_p1_sleep_sees_pending_hidden_by_confirmed_and_expiry(self):
        self.enqueue(self.event(), end=NOV)
        self.enqueue(self.event("guess", 4, speaker="agent"), 4, kind="agent_inference")
        self.assertTrue(self.current()["has_pending"])
        self.keep_sleep()
        self.service.finish_task("task1", self.caller)
        self.assertTrue(any(stage == "sleep" for stage, _ in self.model.calls))
        self.assertEqual(self.current(NOV)["status"], "pending")

    def test_p1_later_pending_does_not_veto_late_arriving_trusted_correction(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("guess", 4, speaker="agent", recorded="2026-10-22T00:00:00+08:00"),
            4,
            kind="agent_inference",
            operation="UPDATE",
        )
        self.enqueue(
            self.event("correct", 6, recorded="2026-10-23T00:00:00+08:00"),
            6,
            operation="UPDATE",
            event_time="2026-10-21T00:00:00+08:00",
        )
        self.assertEqual(self.current()["value"], 6)

    def test_newer_explicit_conflict_is_not_erased_by_older_arriving_update(self):
        self.enqueue(self.event())
        self.enqueue(self.event("conflict", 4, recorded="2026-10-22T00:00:00+08:00"), 4, operation="CONFLICT")
        self.enqueue(
            self.event("late", 6, recorded="2026-10-23T00:00:00+08:00"),
            6,
            operation="UPDATE",
            event_time="2026-10-21T00:00:00+08:00",
        )
        self.assertEqual(self.current()["status"], "conflict")

    def test_plan_with_unresolved_state_is_not_ready_even_if_revision_unchanged(self):
        card = self.enqueue(self.event(), start=None)
        self.route([card], at=TIME)
        planner = Caller("u1", "team", "planner")
        self.service.register_plan("pending-plan", "task2", planner)
        self.assertFalse(self.service.check_plan_dependencies("pending-plan", planner)["ready"])

    def test_p1_disabled_sleep_can_be_enabled_for_same_completion(self):
        self.enqueue(self.event(), start=None)
        self.settings.sleep_enabled = False
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "disabled")
        self.settings.sleep_enabled = True
        self.keep_sleep()
        self.assertEqual(self.service.finish_task("task1", self.caller)["status"], "completed")

    def test_completion_by_one_agent_does_not_suppress_another_private_review(self):
        self.enqueue(self.event("private", readers=("dialogue", "secret")), start=None)
        original = self.caller
        self.caller = Caller("u1", "team", "planner")
        event = replace(self.event("planner-private", readers=("planner",)), writer_agent_id="planner")
        self.enqueue(event, start=None)
        planner = self.caller
        self.caller = original
        self.keep_sleep()
        self.assertEqual(self.service.finish_task("task1", original)["reviewed"], 1)
        self.keep_sleep()
        self.assertEqual(self.service.finish_task("task1", planner)["reviewed"], 1)
        self.assertFalse(self.store.dirty_states())

    def test_p1_invalid_sleep_is_quarantined_without_starving_other_states(self):
        first = self.enqueue(self.event(), start=None)
        self.enqueue(self.event("other", scope="other"), start=None)

        def proposal(p):
            return {
                "state_id": p["state"]["id"],
                "expected_revision": p["state"]["revision"],
                "operations": [{"kind": "UPDATE", "card_id": first}] if p["state"]["scope"] == "exam" else [],
            }

        self.model.responses["sleep"].extend([proposal, proposal])
        result = self.service.finish_task("task1", self.caller)
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(result["reviewed"], 1)
        calls = len(self.model.calls)
        self.service.finish_task("task1", self.caller)
        self.assertEqual(len(self.model.calls), calls)
        self.keep_sleep()
        self.assertEqual(self.service.retry_sleep(self.caller, "task1")["status"], "completed")

    def test_p1_unitless_registry_rejects_unregistered_currency(self):
        event = self.event()
        fields = self.fields(event, state_key="budget", value=100)
        self.raw(event, fields)
        wrong = self.event("wrong")
        self.backend.changes["wrong"] = [MemoryChange("wrong-m", wrong.text)]
        self.model.responses["cards"].append(
            {"cards": [self.fields(wrong, "wrong-m", state_key="budget", value=100, unit="CNY")]}
        )
        with self.assertRaises(ValidationError):
            self.service.ingest_event(wrong, self.caller)
        self.assertEqual(len(self.store.states()), 1)

    def test_p2_scope_and_conditions_are_filtered_before_route_prompt(self):
        good = self.raw(self.event(), self.fields(self.event()))[0]
        other = self.event("other", scope="private_project")
        self.raw(other, self.fields(other, "other-m"))
        condition = self.event("conditional")
        self.raw(condition, self.fields(condition, "conditional-m", conditions=["weekend"]))
        self.route([good], at=TIME)
        payload = [p for stage, p in self.model.calls if stage == "route"][-1]
        self.assertEqual([c["id"] for c in payload["cards"]], [good])

    def test_p2_nonstate_unknown_start_does_not_apply_before_observation(self):
        event = self.event()
        card = self.raw(event, self.fields(event, valid_from=None))[0]
        result = self.route([card], at="2020-01-01T00:00:00+08:00")
        self.assertFalse(result.cards)
        self.assertTrue(result.issues)

    def test_p2_missing_event_time_cannot_overwrite_trusted_current_value(self):
        self.enqueue(self.event())
        event = self.event("old", 4, recorded="2026-10-23T00:00:00+08:00")
        self.raw(
            event,
            self.fields(
                event,
                "old-m",
                state_key="available_study_hours_per_week",
                value=4,
                unit="hours/week",
                event_time=None,
            ),
            operation="UPDATE",
        )
        self.assertEqual(self.current()["value"], 12)
        self.assertTrue(self.current()["has_pending"])
        with self.assertRaises(ValidationError):
            self.service.ingest_event(self.event("future", recorded="2099-01-01T00:00:00+08:00"), self.caller)

    def test_p2_failed_local_update_also_marks_global_default_uncertain_for_that_scope(self):
        card = self.enqueue(self.event("global", scope="global"))
        event = self.event("bad", 4)
        self.backend.changes["bad"] = [MemoryChange("bad-m", "4")]
        self.model.responses["cards"].append({"cards": []})
        with self.assertRaises(ValidationError):
            self.service.ingest_event(event, self.caller)
        result = self.route([card], at=TIME)
        self.assertFalse(result.states or result.cards)
        self.assertTrue(result.uncertain)

    def test_p2_nonstate_plan_keeps_acl_and_detects_forgotten_card(self):
        event = self.event(readers=("dialogue", "secret"))
        card = self.raw(event, self.fields(event))[0]
        self.route([card], at=TIME, caller=self.caller, task="p-task")
        self.service.register_plan("p", "p-task", self.caller)
        with self.assertRaises(AccessDenied):
            self.service.check_plan_dependencies("p", Caller("u1", "team", "planner"))
        with self.assertRaises(AccessDenied):
            self.service.register_plan("p", "p-task", Caller("u1", "team", "secret"))
        self.service.forget(["e1"], self.caller)
        self.assertFalse(self.service.check_plan_dependencies("p", self.caller)["ready"])

    def test_p2_forget_is_atomic_and_retryable_after_sidecar_failure(self):
        self.enqueue(self.event())
        original = self.store.remove

        def fail(kind, key):
            if kind == "event":
                raise RuntimeError("simulated crash")
            return original(kind, key)

        with patch.object(self.store, "remove", side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.service.forget(["e1"], self.caller)
        self.assertTrue(self.store.get("event", "e1"))
        self.assertFalse(self.store.get("forgotten", stable_id("event", "u1", "team", "e1")))
        self.assertTrue(self.backend.items)
        self.service.forget(["e1"], self.caller)
        self.assertEqual(self.service.forget(["e1"], self.caller)["status"], "already_forgotten")

    def test_p2_forget_counts_only_authorized_pending_policies(self):
        self.enqueue(self.event())
        self.store.put(
            "blocked_policy",
            "foreign",
            {
                "user_id": "other",
                "team_id": "team",
                "allowed_reader_ids": ["dialogue"],
                "memory_ids": ["one", "two"],
                "cache_pending": True,
            },
        )
        self.assertEqual(self.service.forget(["e1"], self.caller)["pending_index_deletes"], 0)

    def test_p2_checkpoint_failure_does_not_undo_success_or_break_retry(self):
        self.enqueue(self.event())
        connection = self.store.connection

        class BusyCheckpoint:
            def __getattr__(self, name):
                return getattr(connection, name)

            def execute(self, sql, *args):
                if sql.startswith("PRAGMA wal_checkpoint"):
                    raise sqlite3.OperationalError("busy")
                return connection.execute(sql, *args)

        self.store.connection = BusyCheckpoint()
        self.assertTrue(self.service.forget(["e1"], self.caller)["checkpoint_pending"])
        self.assertEqual(self.service.forget(["e1"], self.caller)["status"], "already_forgotten")
        self.store.connection = connection

    def test_p2_missing_evidence_is_returned_to_caller(self):
        card = self.enqueue(self.event())
        self.model.responses["route"].append(
            {"state_ids": [], "cards": [{"card_id": card, "use": "current"}], "missing": ["需要确认考试日期"]}
        )
        result = self.service.read_memory(ReadRequest("t", "plan", TIME, "exam"), self.caller)
        self.assertEqual(result.missing, ["需要确认考试日期"])
