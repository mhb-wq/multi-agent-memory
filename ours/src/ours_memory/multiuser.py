"""Shared-pool multi-user memory; agents receive selected evidence only."""

import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from copy import deepcopy
from dataclasses import asdict

from .config import validate_current_method
from .control import Actor, ControlPlane
from .governance import AccessGraph, GovernedMemoryService
from .joint_validation import JointIds, JointValidationError, validate_joint
from .models import (
    AccessDenied,
    Caller,
    Event,
    ModelOutputError,
    OutputTruncated,
    ValidationError,
    active,
    canonical,
    now,
    stable_id,
    timestamp,
)
from .participants import resolve_participants
from .semantic_views import compact_card
from .service import MemoryService
from .state import state_id, state_view
from .state_vector import rank_state_entries


class CardFormatError(ValidationError):
    """A model response cannot be interpreted as the focused card schema."""


class SleepOnlyCore(GovernedMemoryService):
    """Write cards now; consolidate state only inside trusted maintenance entry."""

    def _storage_partition(self, data):
        return stable_id("shared_pool", data["team_id"])

    def _event(self, event, caller):
        return MemoryService._event(self, event, caller)

    def _visible(self, item, caller):
        return bool(item and item.get("team_id") == caller.team_id
                    and not item.get("deleted") and not item.get("blocked"))

    def _merge_state_evidence(self, previous, proposed, caller):
        """Retain valid prior provenance while adding this update's proposed sources."""
        merged = []
        for card_id in [*previous, *proposed]:
            card = self.store.get("card", card_id)
            if (card_id not in merged and card and self._visible(card, caller)
                    and card.get("team_id") == caller.team_id):
                merged.append(card_id)
        return merged

    def _joint_proposal(self, stage, payload, *, batch):
        """Retry one repairable invalid proposal, before committing any state."""
        ids = JointIds(payload)
        wire = ids.encode(payload)
        attempt_id = "joint_attempt_" + uuid.uuid4().hex
        attempts = []
        feedback = None
        card_ids = ([payload["new_card"]["id"]] if not batch else
                    [card["id"] for card in payload["new_cards"]])
        def record():
            self.store.put("joint_attempt", attempt_id,
                           {"id": attempt_id, "stage": stage, "card_ids": card_ids,
                            "attempts": attempts, "recorded_at": self.clock()})
        for attempt in range(2):
            request = deepcopy(wire)
            if feedback:
                request["validation_feedback"] = feedback
                request["repair_instruction"] = (
                    "Correct the indicated validation problem using only the supplied evidence and IDs. "
                    "Do not guess identities or facts. Use uncertain if the evidence is insufficient."
                )
            started = time.perf_counter()
            try:
                proposal = ids.decode(self._model(stage, request))
                validate_joint(proposal, payload, batch=batch)
            except (JointValidationError, ModelOutputError) as exc:
                issue = self._issue(stage, exc)
                attempts.append({"attempt": attempt, "is_repair": bool(attempt), "status": "invalid",
                                 "elapsed_seconds": time.perf_counter() - started, "issue": issue})
                record()
                if (attempt or isinstance(exc, OutputTruncated)
                        or isinstance(exc, JointValidationError) and not exc.repairable):
                    raise
                feedback = {"code": getattr(exc, "validation_code", "invalid_json"), "detail": str(exc)}
                continue
            except Exception as exc:
                attempts.append({"attempt": attempt, "is_repair": bool(attempt), "status": "failed",
                                 "elapsed_seconds": time.perf_counter() - started,
                                 "issue": self._issue(stage, exc)})
                record()
                raise
            attempts.append({"attempt": attempt, "is_repair": bool(attempt), "status": "valid",
                             "elapsed_seconds": time.perf_counter() - started})
            record()
            return proposal
        raise AssertionError("unreachable")

    def _update_cards(self, card_ids, task_id, job=None):
        for card_id in card_ids:
            card = self.store.get("card", card_id)
            if card:
                self.store.put("pending_state", card_id, {"card_id": card_id, "task_id": task_id})
        return []

    def _validate_card(self, fields, event, changes):
        dropped_key = None
        if not event.get("state_eligible", True):
            fields = {**fields, "state_key": None, "value": None, "unit": None}
        elif (fields.get("state_key") in {"goal", "preference", "constraint", "identity", "task_plan", "commitment", "deadline"}
              and not fields.get("state_aspect")):
            # Broad categories alone cannot identify the same real-world state.
            # Retain the card as evidence rather than merge unrelated facts.
            dropped_key = fields["state_key"]
            fields = {**fields, "state_key": None, "value": None, "unit": None}
        card = super()._validate_card(fields, event, changes)
        if dropped_key:
            card.update(state_key_dropped=dropped_key, state_drop_reason="missing_specific_aspect")
        metadata = self.store.get("envelope_v2", event["event_id"])
        if (
            metadata
            and not metadata["can_confirm"]
            and card["evidence_kind"] in {"explicit_user_statement", "tool_observation", "execution_result"}
        ):
            card["evidence_kind"] = "retelling"
        return card

    def update_state(self, *args, **kwargs):
        raise AccessDenied("State只能由Sleep维护")

    def finish_task(self, task_id, caller, *, card_ids=None):
        with self.lock:
            self._authorize(caller)
            if card_ids is not None and len(card_ids) > 1:
                return self._finish_joint_batch(task_id, caller, list(card_ids))
            return self._finish_joint(task_id, caller, card_ids=card_ids)

    def _finish_joint_batch(self, task_id, caller, card_ids):
        """Retry a truncated fresh batch in smaller, ordered pieces.

        Never apply this fallback to an earlier failed message: later events may
        already have changed the state and require an ordered replay instead.
        """
        before = {key: self.store.get("pending_state", key) for key in card_ids}
        result = self._finish_joint_batch_once(task_id, caller, card_ids)
        issues = result.get("issues", [])
        ordered_recovery = os.environ.get("OURS_ORDERED_BATCH_RECOVERY") == "1"
        recoverable = {"OutputTruncated"} | ({"ValidationError", "JointValidationError"} if ordered_recovery else set())
        if (result.get("status") != "needs_attention" or len(card_ids) < 2
                or not any(i.get("code") in recoverable
                           and i.get("validation_code") not in {"state_uncertain", "relation_uncertain"}
                           for i in issues)
                or any(not row or (row.get("status") == "needs_attention" and not ordered_recovery)
                       for row in before.values())):
            return result
        cards = [self.store.get("card", key) for key in card_ids]
        def sequence(card):
            envelope = self.store.get("envelope_v2", (card.get("source_event_ids") or [""])[0]) or {}
            return envelope.get("sequence", 0)
        if any(not c for c in cards):
            return result
        cards.sort(key=lambda c: (sequence(c), c["id"]))
        if any(e.get("space") == caller.team_id and e.get("sequence", 0) > sequence(cards[-1])
               for e in self.store.all("envelope_v2")):
            return result
        recovery_id = "joint_recovery_" + uuid.uuid4().hex
        outcomes = []
        marker = self.store.connection.execute("SELECT COALESCE(MAX(seq),0) FROM calls").fetchone()[0]
        existing_attempts = {row[0] for row in self.store.connection.execute(
            "SELECT id FROM objects WHERE kind='joint_attempt'")}

        class RollbackFallback(Exception):
            pass

        saved_calls, saved_attempts, failure_issues = [], [], []
        try:
            with self.store.transaction():
                for key, row in before.items():
                    if ordered_recovery:
                        row = {k: v for k, v in row.items() if k not in {"status", "last_issues"}}
                    self.store.put("pending_state", key, row)
                for card in cards:
                    one = self._finish_joint(task_id, caller, card_ids={card["id"]})
                    outcomes.append({"card_id": card["id"], "status": one["status"],
                                     "issues": one.get("issues", [])})
                    if one.get("status") != "completed":
                        failure_issues = one.get("issues", [])
                        saved_calls = [tuple(row) for row in self.store.connection.execute(
                            "SELECT stage,data FROM calls WHERE seq>? ORDER BY seq", (marker,))]
                        saved_attempts = [(row[0], row[1]) for row in self.store.connection.execute(
                            "SELECT id,data FROM objects WHERE kind='joint_attempt'") if row[0] not in existing_attempts]
                        raise RollbackFallback()
                self.store.put("joint_batch_recovery", recovery_id,
                               {"status": "committed", "card_ids": [c["id"] for c in cards],
                                "outcomes": outcomes, "recorded_at": self.clock()})
        except RollbackFallback:
            rollback_issue = {"stage": "focused_joint_batch", "code": "BatchFallbackRolledBack",
                              "detail": "拆分处理未全部成功；本批状态修改已回滚"}
            all_issues = issues + failure_issues + [rollback_issue]
            with self.store.transaction():
                for stage, data in saved_calls:
                    self.store.connection.execute("INSERT INTO calls(stage,data) VALUES (?,?)", (stage, data))
                for key, data in saved_attempts:
                    self.store.connection.execute("INSERT OR REPLACE INTO objects VALUES ('joint_attempt',?,?)",
                                                  (key, data))
                for card in cards:
                    row = self.store.get("pending_state", card["id"])
                    if row:
                        self.store.put("pending_state", card["id"],
                                       {**row, "status": "needs_attention", "last_issues": all_issues,
                                        "recovery_id": recovery_id})
                self.store.put("joint_batch_recovery", recovery_id,
                               {"status": "rolled_back", "card_ids": [c["id"] for c in cards],
                                "outcomes": outcomes, "issues": all_issues, "recorded_at": self.clock()})
            return {"status": "needs_attention", "processed": 0, "groups": 1,
                    "issues": all_issues, "recovery_id": recovery_id}
        return {"status": "completed", "processed": len(cards), "groups": 1,
                "issues": [], "fallback": "ordered_single_card_after_truncation", "recovery_id": recovery_id}

    def _finish_joint_batch_once(self, task_id, caller, card_ids):
        """Atomically consolidate the pending cards from one message with one model call."""
        known = set(self.control.spaces[caller.team_id]["members"]) & set(self.control.state_subjects)
        topics = {"preference", "constraint", "plan", "profile", "goal"}
        cards = [self.store.get("card", key) for key in card_ids]
        try:
            if (any(not c or not self._visible(c, caller) or c.get("team_id") != caller.team_id for c in cards)
                    or len({c.get("source_message_id") for c in cards}) != 1
                    or not cards[0].get("source_message_id")):
                raise ValidationError("批量维护要求同一消息的可见卡片")
            def sequence(card):
                envelope = self.store.get("envelope_v2", (card.get("source_event_ids") or [""])[0]) or {}
                return envelope.get("sequence", 0)
            cards.sort(key=lambda c: (sequence(c), c["id"]))
            batch_ids = {c["id"] for c in cards}
            first_sequence = sequence(cards[0])
            if any((self.store.get("pending_state", c["id"]) or {}).get("status") == "needs_attention" for c in cards):
                last_sequence = max(sequence(c) for c in cards)
                if any(e.get("space") == caller.team_id and e.get("sequence", 0) > last_sequence
                       for e in self.store.all("envelope_v2")):
                    raise ValidationError("旧消息有更新后的后续事件，需要按顺序重放")
            policy = stable_id("shared_pool", caller.team_id)
            old_cards, old_per_card, subjects, people = {}, {}, set(), {}
            states, state_per_card, all_states = {}, {}, {}
            eligible, linked = set(), []
            for subject in sorted(known):
                key = stable_id("user_state", caller.team_id, subject)
                state = self.store.get("user_state", key)
                if not state:
                    continue
                all_states[subject] = state
                for topic, rows in state.get("topics", {}).items():
                    for index, row in enumerate(rows):
                        support = [self.store.get("card", i) for i in row.get("evidence", [])]
                        if not support or any(not c or not self._visible(c, caller) or c.get("team_id") != caller.team_id
                                              or sequence(c) >= first_sequence for c in support):
                            continue
                        entry_id = row.get("id") or stable_id("state_item", key, topic, str(index))
                        eligible.add((subject, topic, entry_id))
                        linked.append({**row, "id": entry_id, "subject": subject, "topic": topic})
            for card in cards:
                query = " ".join(x for x in (card.get("claim_text", ""), card.get("matter", "")) if x)
                hits = self.backend.search_shared(query, caller, policy, self.settings.joint_candidate_limit + len(cards))
                ids = []
                for hit in hits:
                    mapping = self.store.get("memory", hit.memory_id) or {}
                    old = self.store.get("card", mapping.get("current_card_id", ""))
                    if (old and old["id"] not in batch_ids and self._visible(old, caller)
                            and old.get("team_id") == caller.team_id and sequence(old) < first_sequence
                            and old["id"] not in ids):
                        ids.append(old["id"])
                        old_cards[old["id"]] = old
                    if len(ids) >= self.settings.joint_candidate_limit:
                        break
                old_per_card[card["id"]] = ids
                ranked = rank_state_entries(self.store, self.backend, caller.team_id, known, query,
                                            self.settings.joint_candidate_limit, allowed_ids=eligible,
                                            threshold=self.settings.state_similarity_threshold)
                entries = [e for e in linked if set(ids).intersection(e.get("evidence", []))]
                entries += [{**e["row"], "id": e["id"], "subject": e["subject"], "topic": e["topic"]} for _, e in ranked]
                state_ids = []
                for entry in entries:
                    if entry["id"] not in state_ids:
                        states[entry["id"]] = entry
                        state_ids.append(entry["id"])
                        subjects.add(entry["subject"])
                    if len(state_ids) >= self.settings.joint_candidate_limit:
                        break
                state_per_card[card["id"]] = state_ids
            for card in [*cards, *old_cards.values()]:
                if card.get("subject") in known:
                    subjects.add(card["subject"])
                people[card["id"]] = resolve_participants(card, known, self.control.person_aliases)
                subjects.update(p["resolved_id"] for p in people[card["id"]] if p["resolved_id"])
            evidence_ids = batch_ids | set(old_cards)
            for entry in states.values():
                evidence_ids.update(i for i in entry.get("evidence", []) if self._visible(self.store.get("card", i) or {}, caller))
            payload = {"new_cards": [{**compact_card(c), "old_card_ids": old_per_card[c["id"]],
                                       "state_ids": state_per_card[c["id"]]} for c in cards],
                       "old_cards": [compact_card(c) for c in old_cards.values()], "states": list(states.values()),
                       "subjects": sorted(subjects), "participant_resolution": people}
            proposal = self._joint_proposal("focused_joint_batch", payload, batch=True)
            if not isinstance(proposal, dict) or set(proposal) != {"cards", "operations"} or not isinstance(proposal["cards"], list) or not isinstance(proposal["operations"], list):
                raise ValidationError("批量联合判断输出格式无效")
            decisions, earlier = {}, set()
            for result, card in zip(proposal["cards"], cards, strict=True):
                if not isinstance(result, dict) or set(result) != {"card_id", "relations"} or result["card_id"] != card["id"] or not isinstance(result["relations"], list):
                    raise ValidationError("批量卡片判断必须按顺序覆盖所有卡片")
                for rel in result["relations"]:
                    if (not isinstance(rel, dict) or set(rel) != {"old_card_id", "relation", "same_matter", "changed_part"}
                            or rel["old_card_id"] not in set(old_cards) | earlier
                            or rel["relation"] not in {"update", "supplement", "related", "unrelated", "uncertain"}
                            or rel["same_matter"] not in {"same", "related", "different", "uncertain"}
                            or (rel["relation"] == "update" and rel["same_matter"] != "same")
                            or not isinstance(rel["changed_part"], str)):
                        raise ValidationError("批量卡片关系无效")
                decisions[card["id"]] = result["relations"]
                earlier.add(card["id"])
            updates, touched = {}, set()
            required = {"subject", "action", "state_id", "topic", "content", "status", "changed_part", "evidence", "source_card_ids"}
            for op in proposal["operations"]:
                if not isinstance(op, dict) or set(op) != required or op["subject"] not in subjects:
                    raise ValidationError("批量状态操作字段或主体无效")
                if op["action"] == "uncertain":
                    raise ValidationError("批量状态关系不确定，需要重新判断")
                if (op["action"] not in {"add", "update", "none"} or op["topic"] not in topics
                        or op["status"] not in {"confirmed", "uncertain"} or not isinstance(op["source_card_ids"], list)
                        or not op["source_card_ids"] or any(i not in batch_ids for i in op["source_card_ids"])):
                    raise ValidationError("批量状态操作类型或来源无效")
                if op["action"] == "none":
                    continue
                if (not isinstance(op["content"], str) or not op["content"].strip() or not isinstance(op["changed_part"], str)
                        or not isinstance(op["evidence"], list) or not set(op["source_card_ids"]) <= set(op["evidence"])
                        or any(i not in evidence_ids for i in op["evidence"])):
                    raise ValidationError("批量状态内容或证据无效")
                subject = op["subject"]
                state = updates.setdefault(subject, deepcopy(all_states.get(subject) or {"subject": subject, "topics": {t: [] for t in topics}}))
                if op["action"] == "update":
                    old = states.get(op["state_id"])
                    if not old or old["subject"] != subject or old["topic"] != op["topic"] or op["state_id"] in touched:
                        raise ValidationError("批量更新目标不在候选或重复")
                    touched.add(op["state_id"])
                    rows = state["topics"][op["topic"]]
                    idx = next((i for i, row in enumerate(rows) if (row.get("id") or stable_id("state_item", stable_id("user_state", caller.team_id, subject), op["topic"], str(i))) == op["state_id"]), None)
                    if idx is None:
                        raise ValidationError("批量更新目标不存在")
                    evidence = self._merge_state_evidence(rows[idx].get("evidence", []), op["evidence"], caller)
                    rows[idx] = {"id": op["state_id"], "content": op["content"], "status": op["status"], "evidence": evidence}
                else:
                    if op["state_id"] is not None:
                        raise ValidationError("批量新增不能指定旧状态编号")
                    state["topics"].setdefault(op["topic"], []).append({"id": "state_item_" + secrets.token_hex(16), "content": op["content"], "status": op["status"], "evidence": op["evidence"]})
            with self.store.transaction():
                for subject, state in updates.items():
                    key = stable_id("user_state", caller.team_id, subject)
                    self.store.put("joint_state_history", "joint_history_" + secrets.token_hex(16),
                                   {"subject": subject, "card_ids": [c["id"] for c in cards], "before": all_states.get(subject),
                                    "after": state, "proposal": proposal, "recorded_at": self.clock()})
                    self.store.put("user_state", key, state)
                for card in cards:
                    self.store.put("joint_card_decision", stable_id("joint_card_decision", caller.team_id, card["id"]),
                                   {"card_id": card["id"], "relations": decisions[card["id"]], "participant_resolution": people, "recorded_at": self.clock()})
                    self.store.remove("pending_state", card["id"])
            if updates:
                try:
                    rank_state_entries(self.store, self.backend, caller.team_id, known, None, 1)
                    self.store.remove("state_vector_issue", caller.team_id)
                except Exception as exc:
                    issue = self._issue("state_vector_sync", exc)
                    self.store.put("state_vector_issue", caller.team_id, issue)
                    return {"status": "needs_attention", "processed": len(cards), "groups": 1, "issues": [issue]}
            return {"status": "completed", "processed": len(cards), "groups": 1, "issues": []}
        except Exception as exc:
            issue = self._issue("focused_joint_batch", exc)
            for key in card_ids:
                pending = self.store.get("pending_state", key)
                if pending:
                    self.store.put("pending_state", key, {**pending, "status": "needs_attention", "last_issues": [issue]})
            return {"status": "needs_attention", "processed": 0, "groups": 1, "issues": [issue]}

    def _finish_joint(self, task_id, caller, *, card_ids=None):
        """Experimental per-card joint relationship/state maintenance for shared pools."""
        topics = ("preference", "constraint", "plan", "profile", "goal")
        known = set(self.control.spaces[caller.team_id]["members"]) & set(self.control.state_subjects)
        pending = [p for p in self.store.all("pending_state")
                   if (task_id == "__all_pending__" or p.get("task_id") == task_id)
                   and (card_ids is None or p["card_id"] in card_ids)]

        def sequence(card):
            envelope = self.store.get("envelope_v2", (card.get("source_event_ids") or [""])[0]) or {}
            return envelope.get("sequence", 0)

        pending.sort(key=lambda p: (sequence(self.store.get("card", p["card_id"]) or {}), p["card_id"]))
        processed, issues = 0, []
        for item in pending:
            card = self.store.get("card", item["card_id"])
            if not card or not self._visible(card, caller) or card.get("team_id") != caller.team_id:
                continue
            try:
                current_sequence = sequence(card)
                if item.get("status") == "needs_attention" and any(
                    e.get("space") == caller.team_id and e.get("sequence", 0) > current_sequence
                    for e in self.store.all("envelope_v2")
                ):
                    raise ValidationError("旧消息有更新后的后续事件，需要按顺序重放")
                policy = stable_id("shared_pool", caller.team_id)
                query_text = " ".join(part for part in (card.get("claim_text", ""), card.get("matter", "")) if part)
                hits = self.backend.search_shared(query_text, caller, policy,
                                                  self.settings.joint_candidate_limit + 1)
                old_cards = []
                for hit in hits:
                    mapping = self.store.get("memory", hit.memory_id) or {}
                    old = self.store.get("card", mapping.get("current_card_id", ""))
                    if (old and old["id"] != card["id"] and self._visible(old, caller)
                            and old.get("team_id") == caller.team_id
                            and sequence(old) < current_sequence and old["id"] not in {c["id"] for c in old_cards}):
                        old_cards.append(old)
                old_cards = old_cards[:self.settings.joint_candidate_limit]
                old_ids = {c["id"] for c in old_cards}
                candidate_subjects = {card.get("subject")} | {c.get("subject") for c in old_cards}
                candidate_subjects &= known
                participant_views = {}
                for source_card in [card, *old_cards]:
                    people = resolve_participants(source_card, known, self.control.person_aliases)
                    participant_views[source_card["id"]] = people
                    candidate_subjects.update(p["resolved_id"] for p in people if p["resolved_id"])
                eligible = set()
                linked_entries = []
                all_states = {}
                for subject in sorted(known):
                    key = stable_id("user_state", caller.team_id, subject)
                    state = self.store.get("user_state", key)
                    if not state:
                        continue
                    all_states[subject] = state
                    for topic, rows in state.get("topics", {}).items():
                        for index, row in enumerate(rows):
                            support = [self.store.get("card", i) for i in row.get("evidence", [])]
                            if not support or any(not c or not self._visible(c, caller)
                                                  or c.get("team_id") != caller.team_id
                                                  or sequence(c) >= current_sequence for c in support):
                                continue
                            entry_id = row.get("id") or stable_id("state_item", key, topic, str(index))
                            eligible.add((subject, topic, entry_id))
                            linked = bool(old_ids.intersection(row.get("evidence", [])))
                            if linked:
                                linked_entries.append({**row, "id": entry_id, "subject": subject, "topic": topic})
                ranked = rank_state_entries(self.store, self.backend, caller.team_id, known,
                                            query_text, self.settings.joint_candidate_limit,
                                            allowed_ids=eligible, threshold=self.settings.state_similarity_threshold)
                candidate_entries = []
                for entry in linked_entries + [{**e["row"], "id": e["id"], "subject": e["subject"], "topic": e["topic"]}
                                               for _, e in ranked]:
                    if not any(old["id"] == entry["id"] for old in candidate_entries):
                        candidate_entries.append(entry)
                        candidate_subjects.add(entry["subject"])
                    if len(candidate_entries) >= self.settings.joint_candidate_limit:
                        break
                evidence_ids = {card["id"]} | old_ids
                for entry in candidate_entries:
                    evidence_ids.update(i for i in entry.get("evidence", []) if self._visible(self.store.get("card", i) or {}, caller))
                payload = {"new_card": compact_card(card), "old_cards": [compact_card(c) for c in old_cards],
                           "states": candidate_entries, "subjects": sorted(candidate_subjects),
                           "participant_resolution": participant_views}
                proposal = self._joint_proposal("focused_joint", payload, batch=False)
                if not isinstance(proposal, dict) or set(proposal) != {"relations", "operations"}:
                    raise ValidationError("联合判断输出字段无效")
                if not isinstance(proposal["relations"], list) or not isinstance(proposal["operations"], list):
                    raise ValidationError("联合判断须返回关系列表和操作列表")
                for relation in proposal["relations"]:
                    if (not isinstance(relation, dict) or not {"old_card_id", "relation", "changed_part"} <= set(relation)
                            or set(relation) - {"old_card_id", "relation", "changed_part", "same_matter"}
                            or relation["old_card_id"] not in old_ids
                            or relation["relation"] not in {"update", "supplement", "related", "unrelated", "uncertain"}
                            or relation.get("same_matter", "uncertain") not in {"same", "related", "different", "uncertain"}
                            or (relation.get("same_matter", "uncertain") in {"related", "different", "uncertain"}
                                and relation["relation"] == "update")
                            or not isinstance(relation["changed_part"], str)):
                        raise ValidationError("卡片关系无效")
                if any(op.get("action") == "uncertain" for op in proposal["operations"] if isinstance(op, dict)):
                    raise ValidationError("关系不确定，需要重新判断")
                entries = {entry["id"]: entry for entry in candidate_entries}
                updates, touched = {}, set()
                for op in proposal["operations"]:
                    required = {"subject", "action", "state_id", "topic", "content", "status", "changed_part", "evidence"}
                    if not isinstance(op, dict) or set(op) != required or op["subject"] not in candidate_subjects:
                        raise ValidationError("状态操作字段或主体无效")
                    if (op["action"] not in {"add", "update", "none"} or op["topic"] not in topics
                            or op["status"] not in {"confirmed", "uncertain"}):
                        raise ValidationError("状态操作类型或主题无效")
                    if op["action"] == "none":
                        continue
                    if (not isinstance(op["content"], str) or not op["content"].strip()
                            or not isinstance(op["changed_part"], str) or not isinstance(op["evidence"], list)
                            or card["id"] not in op["evidence"]
                            or any(i not in evidence_ids for i in op["evidence"])):
                        raise ValidationError("状态内容或证据无效")
                    subject = op["subject"]
                    state = updates.setdefault(subject, deepcopy(all_states.get(subject) or
                                               {"subject": subject, "topics": {t: [] for t in topics}}))
                    if op["action"] == "update":
                        old = entries.get(op["state_id"])
                        if not old or old["subject"] != subject or old["topic"] != op["topic"] or op["state_id"] in touched:
                            raise ValidationError("目标状态不在候选中或主体不符")
                        touched.add(op["state_id"])
                        rows = state["topics"][op["topic"]]
                        idx = next((i for i, row in enumerate(rows) if (row.get("id") or stable_id(
                            "state_item", stable_id("user_state", caller.team_id, subject), op["topic"], str(i))) == op["state_id"]), None)
                        if idx is None:
                            raise ValidationError("目标状态不存在")
                        evidence = self._merge_state_evidence(rows[idx].get("evidence", []), op["evidence"], caller)
                        rows[idx] = {"id": op["state_id"], "content": op["content"], "status": op["status"], "evidence": evidence}
                    else:
                        if op["state_id"] is not None:
                            raise ValidationError("新增状态不能指定旧状态编号")
                        state["topics"].setdefault(op["topic"], []).append({"id": "state_item_" + secrets.token_hex(16),
                            "content": op["content"], "status": op["status"], "evidence": op["evidence"]})
                with self.store.transaction():
                    for subject, state in updates.items():
                        key = stable_id("user_state", caller.team_id, subject)
                        self.store.put("joint_state_history", "joint_history_" + secrets.token_hex(16),
                                       {"subject": subject, "card_id": card["id"], "before": all_states.get(subject),
                                        "after": state, "proposal": proposal, "recorded_at": self.clock()})
                        self.store.put("user_state", key, state)
                    self.store.put("joint_card_decision", stable_id("joint_card_decision", caller.team_id, card["id"]),
                                   {"card_id": card["id"], "relations": proposal["relations"],
                                    "participant_resolution": participant_views, "recorded_at": self.clock()})
                    self.store.remove("pending_state", card["id"])
                processed += 1
                if updates:
                    try:
                        rank_state_entries(self.store, self.backend, caller.team_id, known, None, 1)
                        self.store.remove("state_vector_issue", caller.team_id)
                    except Exception as exc:
                        issue = self._issue("state_vector_sync", exc, card_id=card["id"])
                        self.store.put("state_vector_issue", caller.team_id, issue)
                        issues.append(issue)
            except Exception as exc:
                issue = self._issue("focused_joint", exc, card_id=card["id"])
                self.store.put("pending_state", card["id"], {**item, "status": "needs_attention", "last_issues": [issue]})
                issues.append(issue)
        return {"status": "needs_attention" if issues else "completed", "processed": processed,
                "groups": len(pending), "issues": issues}

class MultiUserMemory:
    def __init__(self, settings, backend, model, store, config, clock=now):
        validate_current_method(settings)
        if config.get("access_mode") != "shared_pool":
            raise ValidationError("仅支持共享记忆池")
        self.store, self.backend, self.model, self.settings = store, backend, model, settings
        self.control = ControlPlane(store, config, clock)
        self.clock = clock
        self.graph = AccessGraph(store)
        self.core = SleepOnlyCore(settings, backend, model, store, self.graph)
        self.core.control = self.control
        self.core.focused_schema = True
        self._validate_shared_storage()
        self.core.clock = clock
        self.lock = self.core.lock
        row = store.get("system_v2", "fingerprint_key")
        if not row:
            row = {"value": secrets.token_hex(32)}
            store.put("system_v2", "fingerprint_key", row)
        self._key = bytes.fromhex(row["value"])
        self.scheduler = None
        # Trusted in-process diagnostic observer, never an HTTP response or agent capability.
        self.observer = None




    def _digest(self, *values):
        return hmac.new(self._key, canonical(values).encode(), hashlib.sha256).hexdigest()

    def _internal(self, actor, subject):
        return Caller(subject, actor.space_id, actor.agent_id)

    def _validate_shared_storage(self):
        # Old experiment snapshots must not silently mix cohort indexes with a
        # public index. Rebuild in a new run; keep historical results untouched.
        if getattr(self, "_shared_storage_validated", False):
            return
        for event in self.store.all("event"):
            if event.get("policy") != stable_id("shared_pool", event["team_id"]):
                raise ValidationError("旧记忆按用户/类别分库存储；共享池新版需要在新运行目录重建记忆")
        if self.core.focused_schema and any("state_topic" not in c for c in self.store.all("card")):
            raise ValidationError("旧卡片不含新版状态主题；请在新运行目录重建记忆")
        if not callable(getattr(self.backend, "search_shared", None)):
            raise ValidationError("当前后端不支持公共记忆池统一检索")
        self._shared_storage_validated = True

    def _readers(self, space):
        return sorted(self.control.spaces[space]["agents"])

    def _available(self, row):
        return row and not row.get("deleted") and not row.get("blocked")

    def _visible(self, row, actor, seen=None):
        if not self._available(row) or row["space"] != actor.space_id:
            return False
        if not self.control.permits(actor, row["subject"], row["category"], "read", resource=row.get("item_id")):
            return False
        seen = set() if seen is None else seen
        if row["id"] in seen:
            return False
        return all(
            self._visible(self.store.get("envelope_v2", ref), actor, seen | {row["id"]})
            for ref in row.get("sources", [])
        )

    def ingest(
        self,
        actor,
        *,
        event_id,
        subject_id,
        category,
        text,
        item_id=None,
        scope="global",
        observed_at=None,
        sources=(),
        speaker="user",
        source_message_id=None,
        metadata=None,
        state_eligible=True,
        card_fields=None,
    ):
        """Trusted runner submits one subject/category fragment. Never an ACL instruction.

        ``card_fields`` is only set by the direct-card path (one model call per message);
        it carries the already validated card that belongs to this event's quote.
        """
        with self.lock:
            if self.control.access_mode == "shared_pool":
                self._validate_shared_storage()
            if self.core.focused_schema and card_fields is None:
                raise ValidationError("共享记忆池请使用消息抽取接口创建新版卡片")
            self.control.require(actor, subject_id, category, "write", resource=item_id)
            metadata = metadata or {}
            if (
                not isinstance(metadata, dict)
                or set(metadata)
                - {"domain", "channel", "role", "reply_to", "topic", "phase_name", "msg_node", "timestamp", "source_time_unknown", "context_ids"}
                or any(v is not None and not isinstance(v, str) for v in metadata.values())
            ):
                raise ValidationError("来源元数据须为支持的文本字段")
            if self.core.focused_schema and metadata.get("source_time_unknown") == "true":
                observed_at = None
            if type(state_eligible) is not bool:
                raise ValidationError("state_eligible必须是布尔值")
            if speaker not in {"user", "agent", "tool"} or not isinstance(text, str) or not text.strip():
                raise ValidationError("无效事件")
            if not isinstance(event_id, str) or not event_id or not isinstance(scope, str) or not scope:
                raise ValidationError("事件ID和范围不能为空")
            if item_id is not None and (not isinstance(item_id, str) or not item_id):
                raise ValidationError("事项ID须为非空文本")
            for source in sources:
                if not self._visible(self.store.get("envelope_v2", source), actor):
                    raise AccessDenied("不能引用未知或无权读取的来源")
            key = stable_id("event_v2", actor.space_id, actor.principal_id, event_id)
            fingerprint = self._digest(subject_id, category, text.strip())
            if self.store.get("erased_content_v2", stable_id("erased", actor.space_id, fingerprint)):
                raise AccessDenied("已删除内容不能通过更换事件ID重放")
            previous = self.store.get("envelope_v2", key)
            if previous and (previous.get("deleted") or previous.get("blocked")):
                raise AccessDenied("已删除事件不可重放")
            at = previous["recorded_at"] if previous else self.clock()
            if observed_at and timestamp(observed_at) > timestamp(at):
                raise ValidationError("观察时间晚于记录时间")
            spec = dict(
                subject=subject_id,
                category=category,
                space=actor.space_id,
                actor=asdict(actor),
                item_id=item_id,
                scope=scope,
                observed_at=observed_at,
                sources=sorted(set(sources)),
                speaker=speaker,
                source_message_id=source_message_id,
                metadata=metadata,
                state_eligible=state_eligible,
                source_fragment_id=self._digest(source_message_id, text) if source_message_id else None,
            )
            request_digest = self._digest(spec, text)
            if previous and previous["request_digest"] != request_digest:
                raise ValidationError("事件ID已绑定不同请求")
            row = previous or dict(
                id=key,
                **spec,
                recorded_at=at,
                request_digest=request_digest,
                fingerprint=fingerprint,
                sequence=len(self.store.all("envelope_v2")),
                can_confirm=self.control.permits(actor, subject_id, category, "confirm"),
                card_ids=[],
                deleted=False,
                blocked=False,
            )
            self.store.put("envelope_v2", key, row)
            readers = self._readers(actor.space_id)
            if self.control.access_mode != "shared_pool":
                self.graph.provision(subject_id, actor.space_id, readers, readers, readers, category=category)
            event = Event(
                key,
                subject_id,
                actor.space_id,
                actor.agent_id,
                actor.task_id,
                text,
                at,
                tuple(readers),
                speaker=speaker,
                scope=scope,
                observed_at=observed_at,
                author_id=actor.principal_id,
                category=category,
                item_id=item_id,
                source_message_id=source_message_id,
                metadata=metadata,
                state_eligible=state_eligible,
            )
            ids = self.core.ingest_event(event, self._internal(actor, subject_id), card_fields=card_fields)
            row["card_ids"] = ids
            self.store.put("envelope_v2", key, row)
            job = self.core.job_status(key, self._internal(actor, subject_id))
            return {
                "event_id": key,
                "card_ids": ids,
                "status": job["status"],
                "state_update": "pending_sleep",
            }

    def ingest_message(
        self,
        actor,
        *,
        message_id,
        text,
        scope="global",
        item_id=None,
        observed_at=None,
        metadata=None,
        fallback_category=None,
        prior_context=None,
    ):
        """LLM identifies subjects/categories using exact text spans, never grants permissions."""
        with self.lock:
            self.control.authenticate(actor)
            if (self.core.focused_schema and isinstance(metadata, dict)
                    and metadata.get("source_time_unknown") == "true"):
                observed_at = None
            key = stable_id("message_v2", actor.space_id, actor.principal_id, message_id)
            if prior_context is None:
                prior_context = []
            if (not isinstance(prior_context, list) or len(prior_context) > 10
                or any(not isinstance(x, dict) or set(x) - {"id", "speaker", "role", "text", "timestamp"}
                       or not isinstance(x.get("text"), str) for x in prior_context)
                or sum(len(x["text"]) for x in prior_context) > 6000):
                raise ValidationError("前文最多10条且不超过6000字符")
            if any(not isinstance(x.get("id"), str) or not x["id"] for x in prior_context) or len(
                {x["id"] for x in prior_context}
            ) != len(prior_context) or any(x["id"] == message_id for x in prior_context):
                raise ValidationError("前文消息ID须非空、唯一且不能是当前消息")
            if prior_context:
                metadata = {**(metadata or {}), "context_ids": ",".join(str(x.get("id") or "") for x in prior_context)}
            digest = self._digest(text, scope, item_id, observed_at, metadata, fallback_category,
                                  asdict(actor), prior_context)
            prior = self.store.get("message_v2", key)
            if prior and (prior.get("deleted") or prior["digest"] != digest):
                raise ValidationError("消息已删除或ID绑定不同内容")
            if not isinstance(message_id, str) or not message_id or not isinstance(text, str) or not text.strip():
                raise ValidationError("消息编号和内容不能为空")
            if metadata is not None and (not isinstance(metadata, dict) or any(
                value is not None and not isinstance(value, str) for value in metadata.values()
            )):
                raise ValidationError("来源元数据须为文本")
            source = self.store.get("source_message_v2", key)
            if source and (source.get("deleted") or source.get("digest") != digest):
                raise ValidationError("原始消息已删除或ID绑定不同内容")
            if not source:
                source = {"id": key, "external_id": message_id, "space": actor.space_id,
                          "author_id": actor.principal_id, "text": text, "observed_at": observed_at,
                          "recorded_at": self.clock(), "channel": (metadata or {}).get("channel"),
                          "reply_to": (metadata or {}).get("reply_to"), "metadata": deepcopy(metadata or {}),
                          "digest": digest, "status": "received", "deleted": False,
                          "arrival_sequence": 1 + max((r.get("arrival_sequence", 0) for r in
                              self.store.all("source_message_v2") if r.get("space") == actor.space_id), default=0)}
                self.store.put("source_message_v2", key, source)
            deferred = self.store.get("deferred_message_v2", key)
            outstanding_state = source.get("processing_status") == "state_maintenance_pending" and any(
                (self.store.get("card", pending.get("card_id", "")) or {}).get("source_message_id") == key
                for pending in self.store.all("pending_state"))
            if (deferred or outstanding_state) and any(r.get("space") == actor.space_id and
                r.get("arrival_sequence", 0) > source.get("arrival_sequence", 0)
                for r in self.store.all("source_message_v2")):
                # Reprocessing an older message after newer state changes can
                # overwrite the newer facts; retain it for ordered replay.
                return {"message_id": key, "events": [], "status": "accepted_pending",
                        "processing_status": "ordered_replay_required", "recovery": "ordered_replay_required"}
            if not prior:
                format_errors = []
                cards = None
                for attempt in range(2):
                    try:
                        cards = self._cards_focused(
                            actor, text, metadata, observed_at, fallback_category, prior_context, scope,
                            repair_feedback=format_errors[-1]["detail"] if format_errors else None,
                        )
                        break
                    except (CardFormatError, ModelOutputError) as exc:
                        format_errors.append(self.core._issue("focused_cards", exc))
                    except AccessDenied:
                        raise
                    except Exception as exc:  # isolate provider or per-message failure
                        format_errors.append(self.core._issue("focused_cards", exc))
                        break
                skipped = cards is None
                if self.core.focused_schema and skipped:
                    self.store.put("deferred_message_v2", key, {"id": key, "space": actor.space_id,
                        "actor": asdict(actor), "message_id": message_id, "text": text, "scope": scope,
                        "item_id": item_id, "observed_at": observed_at, "metadata": metadata,
                        "fallback_category": fallback_category, "prior_context": deepcopy(prior_context),
                        "stage": "extraction", "issues": format_errors, "status": "retryable"})
                    source["status"] = "extraction_pending"
                    self.store.put("source_message_v2", key, source)
                    return {"message_id": key, "events": [], "status": "accepted_pending",
                            "processing_status": "extraction_pending", "extraction_status": "deferred",
                            "extraction_attempts": len(format_errors), "issues": format_errors}
                prior = {"id": key, "digest": digest, "cards": cards or [], "events": [], "deleted": False}
                prior.update(extraction_status="skipped_format" if skipped else "extracted",
                             extraction_attempts=len(format_errors) + (0 if skipped else 1),
                             extraction_format_errors=format_errors)
                self.store.put("message_v2", key, prior)
                self.store.remove("deferred_message_v2", key)
            results = []
            batch_immediate = len(prior["cards"]) > 1
            for index, card in enumerate(prior["cards"]):
                expected_id = stable_id(
                    "event_v2", actor.space_id, actor.principal_id, f"{message_id}:{index}"
                )
                prior["events"] = sorted(set(prior["events"] + [expected_id]))
                self.store.put("message_v2", key, prior)
                try:
                    result = self.ingest(
                    actor,
                    event_id=f"{message_id}:{index}",
                    subject_id=card["subject_id"] or actor.principal_id,
                    category=card["category"],
                    text=text if self.core.focused_schema else card["source_quote"],
                    scope=scope,
                    item_id=item_id,
                    observed_at=observed_at,
                    source_message_id=key,
                    metadata=metadata,
                    state_eligible=(card["subject_id"] is not None and
                                    card["subject_id"] in self.control.state_subjects),
                    card_fields=card["fields"],
                    )
                except AccessDenied:
                    raise
                except Exception as exc:  # retain this card and attempt its siblings
                    result = {"event_id": expected_id, "card_ids": [], "status": "needs_attention",
                              "issues": [self.core._issue("card_storage", exc)]}
                if result["status"] == "done" and not batch_immediate:
                    ids = {card_id for card_id in result["card_ids"]
                           if self.store.get("pending_state", card_id)}
                    if ids:
                        maintenance = self.core.finish_task(actor.task_id, self._internal(actor, actor.principal_id),
                                                            card_ids=ids)
                        result["state_update"] = maintenance["status"]
                        result["state_maintenance"] = maintenance
                    elif self.store.get("state_vector_issue", actor.space_id):
                        try:
                            known = (set(self.control.spaces[actor.space_id]["members"])
                                     & set(self.control.state_subjects))
                            rank_state_entries(self.store, self.backend, actor.space_id, known, None, 1)
                            result["state_update"] = "already_processed"
                        except Exception as exc:
                            issue = self.core._issue("state_vector_sync", exc)
                            self.store.put("state_vector_issue", actor.space_id, issue)
                            result["state_update"] = "needs_attention"
                            result["state_maintenance"] = {"status": "needs_attention", "issues": [issue]}
                    else:
                        result["state_update"] = "already_processed"
                results.append(result)
                prior["events"] = sorted(set(prior["events"] + [result["event_id"]]))
                self.store.put("message_v2", key, prior)
                if result.get("state_update") == "needs_attention":
                    # Do not overtake a failed earlier card from the same message.
                    # Retrying the message reuses extracted cards and resumes here.
                    break
            if batch_immediate and len(results) == len(prior["cards"]) and all(r["status"] == "done" for r in results):
                ids = {card_id for result in results for card_id in result["card_ids"]
                       if self.store.get("pending_state", card_id)}
                if ids:
                    maintenance = self.core.finish_task(actor.task_id, self._internal(actor, actor.principal_id), card_ids=ids)
                    for result in results:
                        result["state_update"] = maintenance["status"]
                        result["state_maintenance"] = maintenance
                elif self.store.get("state_vector_issue", actor.space_id):
                    try:
                        known = set(self.control.spaces[actor.space_id]["members"]) & set(self.control.state_subjects)
                        rank_state_entries(self.store, self.backend, actor.space_id, known, None, 1)
                        self.store.remove("state_vector_issue", actor.space_id)
                    except Exception as exc:
                        issue = self.core._issue("state_vector_sync", exc)
                        self.store.put("state_vector_issue", actor.space_id, issue)
                        for result in results:
                            result["state_update"] = "needs_attention"
                            result["state_maintenance"] = {"status": "needs_attention", "issues": [issue]}
            response = {
                "message_id": key,
                "events": results,
                "status": "done" if all(r["status"] == "done" for r in results) else "needs_attention",
            }
            if self.core.focused_schema and any(r.get("status") != "done" for r in results):
                response["status"] = "accepted_pending"
                response["processing_status"] = "card_storage_pending"
            if any(r.get("state_update") == "needs_attention" for r in results):
                response["status"] = "accepted_pending" if self.core.focused_schema else "needs_attention"
                response["processing_status"] = "state_maintenance_pending"
                response["unprocessed_card_count"] = len(prior["cards"]) - len(results)
                response["pending_state_card_count"] = sum(
                    1 for r in results for card_id in r.get("card_ids", [])
                    if self.store.get("pending_state", card_id)
                )
            response.update(extraction_status=prior.get("extraction_status", "extracted"),
                            extraction_attempts=prior.get("extraction_attempts", 1))
            if response["status"] == "accepted_pending":
                self.store.put("deferred_message_v2", key, {"id": key, "space": actor.space_id,
                    "actor": asdict(actor), "message_id": message_id, "text": text, "scope": scope,
                    "item_id": item_id, "observed_at": observed_at, "metadata": metadata,
                    "fallback_category": fallback_category, "prior_context": deepcopy(prior_context),
                    "stage": response["processing_status"], "status": "retryable"})
            else:
                self.store.remove("deferred_message_v2", key)
            source = self.store.get("source_message_v2", key)
            if source:
                source["status"] = prior.get("extraction_status", "extracted")
                source["processing_status"] = response.get("processing_status", "complete")
                self.store.put("source_message_v2", key, source)
            return response
            for index, fragment in enumerate(prior["fragments"]):
                expected_id = stable_id(
                    "event_v2", actor.space_id, actor.principal_id, f"{message_id}:{index}"
                )
                prior["events"] = sorted(set(prior["events"] + [expected_id]))
                self.store.put("message_v2", key, prior)
                result = self.ingest(
                    actor,
                    event_id=f"{message_id}:{index}",
                    subject_id=fragment["subject_id"] or actor.principal_id,
                    category=fragment["category"],
                    text=fragment["quote"],
                    scope=scope,
                    item_id=item_id,
                    observed_at=observed_at,
                    source_message_id=key,
                    metadata=metadata,
                    state_eligible=fragment["subject_id"] is not None,
                )
                results.append(result)
                prior["events"] = sorted(set(prior["events"] + [result["event_id"]]))
                self.store.put("message_v2", key, prior)
            return {
                "message_id": key,
                "events": results,
                "status": "done" if all(r["status"] == "done" for r in results) else "needs_attention",
            }


    def _cards_focused(self, actor, text, metadata, observed_at, fallback_category, prior_context, scope,
                       repair_feedback=None):
        topics = {"preference", "constraint", "plan", "profile", "goal"}
        members = sorted(set(self.control.spaces[actor.space_id]["members"]) & self.control.state_subjects)
        if isinstance(metadata, dict) and metadata.get("source_time_unknown") == "true":
            observed_at = None
        payload = {"text": text, "author_id": actor.principal_id, "prior_context": prior_context,
                   "subjects": members, "observed_at": observed_at, "metadata": metadata or {}}
        if repair_feedback:
            payload["format_feedback"] = repair_feedback
            payload["repair_instruction"] = "Return the required card JSON format using the same source information."
        answer = self.core._model("focused_cards", payload)
        if not isinstance(answer, dict) or set(answer) != {"cards"} or not isinstance(answer["cards"], list):
            raise CardFormatError("记忆卡片输出必须包含cards列表")
        category = fallback_category or next(iter(sorted(self.control.categories)))
        if category not in self.control.categories:
            raise ValidationError("回退类别未登记")
        output = []
        for c in answer["cards"]:
            required = {"subject", "claim", "usage", "state_topic", "source"}
            if not isinstance(c, dict) or not required <= set(c) or set(c) - required - {"matter", "participants"}:
                raise CardFormatError("新版记忆卡片字段不合法")
            subject, claim, usage, labels, source = (c[k] for k in ("subject", "claim", "usage", "state_topic", "source"))
            if not isinstance(claim, str) or not claim.strip():
                raise CardFormatError("记忆内容不合法")
            if subject is not None and (not isinstance(subject, str) or subject not in members):
                subject = None
            # An unattributed memory is an ordinary card. Discard a model-supplied
            # topic rather than discard the grounded memory or invent an owner.
            if subject is None and isinstance(labels, list):
                labels = []
            if (not isinstance(usage, str) or len(usage) > 300 or not isinstance(labels, list)
                    or any(not isinstance(t, str) or t not in topics for t in labels)
                    or len(set(labels)) != len(labels) or (subject is None and labels)):
                raise CardFormatError("使用说明或状态主题不合法")
            # Keep the model's source as given. The original message is separately
            # stored on the event, so an imperfect quote does not replace it.
            quote = source.get("quote") if isinstance(source, dict) else None
            quote = quote if isinstance(quote, str) else ""
            matter = c.get("matter", "")
            participants = c.get("participants", [])
            if (not isinstance(matter, str) or len(matter) > 300 or not isinstance(participants, list)
                    or len(participants) > 20 or any(
                        not isinstance(p, dict) or set(p) != {"name", "role"}
                        or not isinstance(p["name"], str) or not p["name"].strip()
                        or len(p["name"]) > 200 or not isinstance(p["role"], str)
                        or len(p["role"]) > 200 for p in participants)):
                raise CardFormatError("事项或人物角色字段不合法")
            fields = {"subject": subject, "claim": claim.strip(), "source_quote": quote,
                      "use_hint": usage.strip(), "state_topic": labels, "source": source,
                      "matter": matter.strip(), "participants": participants,
                      "evidence_kind": "explicit_user_statement", "scope": scope, "conditions": [],
                      "state_key": None, "value": None, "unit": None, "state_aspect": None}
            output.append({"subject_id": subject, "category": category, "source_quote": quote, "fields": fields})
        return output


    @staticmethod
    def _explicit_subject(subject, author, text, members):
        """Conservative evidence check; does not prove correct coreference."""
        if subject == author and re.search(r"(?<!\w)(?:i|me|my|mine)(?!\w)|我|本人", text, re.I):
            return True
        folded = text.casefold()
        if re.search(r"(?<!\w)" + re.escape(subject.casefold()) + r"(?!\w)", folded):
            return True
        ignored = {"user", "patient", "family", "clinician", "nurse", "doctor", "agent", "project",
                   "program", "manager", "staff", "student", "assistant", "adult", "child"}
        parts = [p for p in re.split(r"[_\W]+", subject.casefold()) if len(p) >= 3 and p not in ignored]
        for part in parts:
            if sum(part in re.split(r"[_\W]+", str(member).casefold()) for member in members) != 1:
                continue
            if re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", folded):
                return True
        return False

    @staticmethod
    def _quote_is_user_turn(text, quote):
        """Accept user evidence only when every matching span stays within a user turn."""
        markers = list(re.finditer(r"(?m)^(user|assistant|system):\s*", text))
        if not markers:
            return False
        starts = [m.start() for m in re.finditer(re.escape(quote), text)]
        if not starts:
            return False
        for start in starts:
            end = start + len(quote)
            current = next((m for m in reversed(markers) if m.start() <= start), None)
            if current is None or current.group(1) != "user":
                return False
            if any(start < marker.start() < end for marker in markers):
                return False
        return True

    def read(
        self,
        actor,
        *,
        subject_ids,
        query,
        scope="global",
        target_time=None,
        purpose="current",
        conditions=(),
    ):
        with self.lock:
            self.control.authenticate(actor)
            if (
                purpose not in {"current", "history", "evidence"}
                or not isinstance(query, str)
                or not query.strip()
            ):
                raise ValidationError("无效查询")
            target = target_time or self.clock()
            timestamp(target)
            requested_conditions = set(conditions)
            shared = self.control.access_mode == "shared_pool"
            relevant = None
            self._validate_shared_storage()
            # Subject IDs remain descriptive metadata, not a search filter.
            subject_ids = list(self.control.spaces[actor.space_id]["members"])
            cards, states, candidates = {}, {}, {}
            envelopes = {
                r["id"]: r
                for r in self.store.all("envelope_v2")
                if r["subject"] in subject_ids and self._visible(r, actor)
            }
            for card in self.store.all("card"):
                event = envelopes.get(card["source_event_ids"][0])
                if (
                    event
                    and card["scope"] in {"global", scope}
                    and (self.control.access_mode == "shared_pool"
                         or set(card.get("conditions") or ()) <= requested_conditions)
                ):
                    cards[card["id"]] = card
            # Search only authorized subject/category cubes, before touching native content.
            # Relevance scores are kept: a global cut in subject order would make later
            # subjects invisible whenever earlier subjects have enough cards.
            hits_by_subject: dict[str, list[tuple[float, str]]] = {}
            seen_cards: set[str] = set()
            if shared and cards:
                policy = stable_id("shared_pool", actor.space_id)
                hits = self.backend.search_shared(
                    f"{actor.principal_id} {query}", self._internal(actor, actor.principal_id),
                    policy, self.settings.candidate_limit,
                )
                for hit in hits:
                    mapping = self.store.get("memory", hit.memory_id)
                    card_id = mapping["current_card_id"] if mapping else None
                    if card_id in cards and card_id not in seen_cards:
                        seen_cards.add(card_id)
                        hits_by_subject.setdefault("shared_pool", []).append((hit.score, card_id))
            limit = self.settings.candidate_limit
            ordered = [key for _, key in hits_by_subject.get("shared_pool", [])][:limit]
            candidates = {key: cards[key] for key in ordered}
            candidates = {card_id: candidates[card_id] for card_id in ordered[:limit]}
            states = {}
            # Follow evidence links from retrieved cards to individual state
            # entries, without filtering the public card search by subject.
            known = self.control.state_subjects
            relevant = {c.get("subject") for c in candidates.values() if c.get("subject") in known}
            candidate_ids = set(candidates)
            for subject in sorted(known):
                if subject:
                    key = stable_id("user_state", actor.space_id, subject)
                    state = self.store.get("user_state", key)
                    if state:
                        matched_topics = {
                            topic: [deepcopy(row) for row in rows
                                    if candidate_ids.intersection(row.get("evidence") or [])]
                            for topic, rows in state.get("topics", {}).items()
                        }
                        if any(matched_topics.values()):
                            states[key] = {**state, "topics": matched_topics}
                            relevant.add(subject)
            if purpose == "current" and target_time is None:
                # Reconcile the independent semantic index before every lookup.
                eligible = set()
                for subject in sorted(known):
                    key = stable_id("user_state", actor.space_id, subject)
                    state = self.store.get("user_state", key)
                    if not state:
                        continue
                    for topic, rows in state.get("topics", {}).items():
                        for index, row in enumerate(rows):
                            evidence = row.get("evidence") or []
                            if not evidence or any(card_id not in cards for card_id in evidence):
                                continue
                            eligible.add((subject, topic, row.get("id") or stable_id("state_item", key, topic, str(index))))
                ranked = rank_state_entries(self.store, self.backend, actor.space_id, known, query,
                                            self.settings.independent_state_limit, allowed_ids=eligible,
                                            threshold=self.settings.state_similarity_threshold)
                for _, found in ranked:
                    subject, topic, row, state = found["subject"], found["topic"], found["row"], found["state"]
                    key = stable_id("user_state", actor.space_id, subject)
                    evidence = row["evidence"]
                    # Bound extra evidence separately from the original card Top-K.
                    missing = [card_id for card_id in evidence if card_id not in candidates]
                    if len(candidates) + len(missing) > limit + self.settings.independent_state_limit:
                        continue
                    entry = states.setdefault(key, {**state, "topics": {t: [] for t in state["topics"]}})
                    if row not in entry["topics"].setdefault(topic, []):
                        entry["topics"][topic].append(deepcopy(row))
                    relevant.add(subject)
                    for card_id in missing:
                        candidates[card_id] = cards[card_id]
            selected = list(candidates)
            safe = list(dict.fromkeys(selected))
            excluded = {k: "selector_not_selected" for k in candidates if k not in selected}
            if self.observer:
                retrieved = {card_id for hits in hits_by_subject.values() for _, card_id in hits}
                excluded.update({card_id: "not_retrieved" for card_id in cards if card_id not in retrieved})
                excluded.update(
                    {card_id: "candidate_limit" for card_id in retrieved if card_id not in candidates}
                )
                self.observer(
                    "read",
                    {
                        "actor": asdict(actor),
                        "query": query,
                        "target": target,
                        "authorized_card_ids": list(cards),
                        "retrieval_hits": hits_by_subject,
                        "candidate_card_ids": list(candidates),
                        "selected_card_ids": selected,
                        "returned_card_ids": safe,
                        "excluded": excluded,
                    },
                )
            bundle_id = "bundle_" + secrets.token_hex(16)
            self.store.put(
                "bundle_v2",
                bundle_id,
                {
                    "actor": asdict(actor),
                    "card_ids": safe,
                    "fingerprints": {k: self._digest(cards[k]) for k in safe},
                    "subjects": sorted(subject_ids),
                    "scope": scope,
                    "purpose": purpose,
                    "target": target,
                    "evidence_version": self._evidence_version(actor, subject_ids, scope),
                    "issued_at": self.clock(),
                },
            )
            # Expose only a bounded use hint based on state evidence the actor can read.
            views = []
            for key in safe:
                card = cards[key]
                view = self._card_view(card, target)
                view["usage_note"] = card.get("use_hint") or ""
                views.append(view)
            bundle = self.store.get("bundle_v2", bundle_id)
            bundle["user_states"] = list(states.values())
            bundle["pending_subjects"] = sorted({
                c.get("subject") for c in cards.values()
                if c.get("subject") and (relevant is None or c.get("subject") in relevant)
                and self.store.get("pending_state", c["id"])
            })
            if any(
                not c.get("subject") and self.store.get("pending_state", c["id"])
                for c in candidates.values()
            ):
                # A relevant public change can affect a user's state before
                # joint maintenance determines the actual affected people.
                bundle["pending_subjects"] = sorted(set(bundle["pending_subjects"]) | (relevant or set()))
            bundle["card_annotations"] = {
                view["id"]: {"usage_note": view["usage_note"], "state_status": view["state_status"]}
                for view in views
            }
            bundle["incomplete_message_count"] = sum(
                1 for row in self.store.all("deferred_message_v2") if row.get("space") == actor.space_id)
            bundle["pending_state_count"] = sum(
                1 for row in self.store.all("pending_state")
                if (self.store.get("card", row.get("card_id", "")) or {}).get("team_id") == actor.space_id)
            self.store.put("bundle_v2", bundle_id, bundle)
            result = {"bundle_id": bundle_id, "cards": views}
            result.update(user_states=bundle["user_states"], pending_subjects=bundle["pending_subjects"])
            result["incomplete_message_count"] = bundle["incomplete_message_count"]
            result["pending_state_count"] = bundle["pending_state_count"]
            return result


    def _evidence_version(self, actor, subjects, scope):
        rows = [
            r
            for r in self.store.all("envelope_v2")
            if r["subject"] in subjects and r["scope"] in {"global", scope} and self._visible(r, actor)
        ]
        visible_cards = {k for r in rows for k in r["card_ids"]}
        return self._digest(
            [(r["id"], r["card_ids"]) for r in rows],
            [
                (s["id"], s["revision"])
                for s in self.store.states()
                if s["user_id"] in subjects
                and s["team_id"] == actor.space_id
                and all(k in visible_cards for a in s["assertions"] for k in a["evidence_card_ids"])
            ],
            [self.store.get("user_state", stable_id("user_state", actor.space_id, subject))
             for subject in sorted(subjects)] if self.core.focused_schema else [],
            sorted(r["id"] for r in self.store.all("deferred_message_v2")
                   if r.get("space") == actor.space_id) if self.core.focused_schema else [],
            sorted(r["card_id"] for r in self.store.all("pending_state")
                   if (self.store.get("card", r.get("card_id", "")) or {}).get("team_id") == actor.space_id)
                   if self.core.focused_schema else [],
        )



    def _card_view(self, card, target):
        """Public card plus the state readiness the responder needs to phrase it correctly."""
        view = self._public_card(card)
        if card.get("state_key"):
            state = self.store.state(state_id(card))
            view["state_status"] = state_view(state, target)["status"] if state else "pending"
        else:
            view["state_status"] = None
        return view

    @staticmethod
    def _public_card(card):
        # The caller has already selected cards the actor may read. Preserve
        # the whole card, including observation time and source metadata, so
        # downstream answerers do not lose information stored on the card.
        view = deepcopy(card)
        if "state_topic" in card:
            view["claim"] = card.get("claim_text")
            view["usage"] = card.get("use_hint")
        return view

    def validate_bundle(self, actor, bundle_id):
        with self.lock:
            self.control.authenticate(actor)
            row = self.store.get("bundle_v2", bundle_id)
            if not row or row["actor"] != asdict(actor):
                raise AccessDenied("证据包不属于当前任务身份")
            if row["evidence_version"] != self._evidence_version(actor, row["subjects"], row["scope"]):
                return {"ready": False}
            for key in row["card_ids"]:
                card = self.store.get("card", key)
                if (
                    not card
                    or self._digest(card) != row["fingerprints"][key]
                    or not self._visible(self.store.get("envelope_v2", card["source_event_ids"][0]), actor)
                ):
                    return {"ready": False}
                if (self.control.access_mode != "shared_pool"
                        and row["purpose"] == "current" and not active(card, self.clock())):
                    return {"ready": False}
            return {"ready": True}

    def answer(self, actor, bundle_id, question):
        """Generate from authorized cards without a separate LLM output review."""
        with self.lock:
            if not self.validate_bundle(actor, bundle_id)["ready"]:
                raise AccessDenied("证据已失效，请重新读取")
            row = self.store.get("bundle_v2", bundle_id)
            cards = [self._bundle_card(row, k) for k in row["card_ids"]]
            user_states = row.get("user_states", [])
            incomplete = {"incomplete_message_count": row.get("incomplete_message_count", 0),
                          "pending_state_count": row.get("pending_state_count", 0)}
            if not cards and not user_states:
                return {"action": "no_memory", "text": "当前没有可用于回答的授权记忆。", **incomplete}
            draft = self.core._model(
                "answer", {"question": question, "cards": cards, "user_states": user_states,
                           "pending_subjects": row.get("pending_subjects", []),
                           **incomplete, "asking_user_id": actor.principal_id}
            )
            if (
                set(draft) != {"text", "card_ids"}
                or not isinstance(draft["text"], str)
                or not isinstance(draft["card_ids"], list)
                or not set(draft["card_ids"]) <= set(row["card_ids"])
            ):
                raise ValidationError("回答只能引用当前证据包")
            if not self.validate_bundle(actor, bundle_id)["ready"]:
                raise AccessDenied("证据已失效，请重新读取")
            return {"action": "answer", "text": draft["text"], **incomplete}

    def review_response(self, actor, bundle_id, question, response, *, card_ids=None):
        """Check an externally generated answer against a bound authorized bundle."""
        with self.lock:
            if not isinstance(response, str) or not isinstance(question, str):
                raise ValidationError("问题和回答必须是文本")
            if not self.validate_bundle(actor, bundle_id)["ready"]:
                raise AccessDenied("证据已失效，请重新读取")
            row = self.store.get("bundle_v2", bundle_id)
            if card_ids is None:
                card_ids = row["card_ids"]
            if not isinstance(card_ids, list) or any(not isinstance(k, str) or k not in row["card_ids"] for k in card_ids):
                raise ValidationError("回答审查只能使用当前证据包中的卡片")
            cards = [self._card_view(self.store.get("card", k), row["target"]) for k in card_ids]
            review = self.core._model("output_check", {
                "question": question, "cards": cards, "answer": response,
                "asking_user_id": actor.principal_id,
            })
            return {"supported": review.get("supported") is True and self.validate_bundle(actor, bundle_id)["ready"]}

    def bundle(self, actor, bundle_id):
        with self.lock:
            if not self.validate_bundle(actor, bundle_id)["ready"]:
                raise AccessDenied("证据已失效，请重新读取")
            row = self.store.get("bundle_v2", bundle_id)
            result = {
                "bundle_id": bundle_id,
                "cards": [
                    self._bundle_card(row, key) for key in row["card_ids"]
                ],
            }
            if self.control.access_mode == "shared_pool":
                result.update(user_states=row.get("user_states", []), pending_subjects=row.get("pending_subjects", []))
            return result

    def _bundle_card(self, row, key):
        card = self._card_view(self.store.get("card", key), row["target"])
        # Missing annotations are intentionally not recomputed from potentially
        # inaccessible state, including after handing a bundle to another actor.
        card["state_status"] = None
        card.update(row.get("card_annotations", {}).get(key, {}))
        return card

    def handoff(self, sender, recipient, bundle_id):
        """Recipient gets its own task-bound bundle; no free-text laundering channel."""
        with self.lock:
            if not self.validate_bundle(sender, bundle_id)["ready"]:
                raise AccessDenied("交接证据已失效")
            self.control.authenticate(recipient)
            row = self.store.get("bundle_v2", bundle_id)
            if sender.space_id != recipient.space_id:
                raise AccessDenied("不能跨协作空间交接")
            for key in row["card_ids"]:
                card = self.store.get("card", key)
                if not self._visible(self.store.get("envelope_v2", card["source_event_ids"][0]), recipient):
                    raise AccessDenied("接收者没有原始证据的读取权限")
            new_id = "bundle_" + secrets.token_hex(16)
            self.store.put(
                "bundle_v2",
                new_id,
                {
                    **row,
                    "actor": asdict(recipient),
                    "card_annotations": {},
                    "evidence_version": self._evidence_version(recipient, row["subjects"], row["scope"]),
                    "issued_at": self.clock(),
                },
            )
            return {"bundle_id": new_id, "recipient": asdict(recipient)}

    def sleep(self, actor):
        with self.lock:
            self.control.authenticate(actor)
            if "sleep" not in self.control.agents[actor.agent_id].get("controls", []):
                raise AccessDenied("没有维护职责")
            job_id = stable_id("sleep_v2", asdict(actor))
            job = {"id": job_id, "actor": asdict(actor), "status": "running"}
            self.store.put("sleep_v2", job_id, job)
            # Sleep is a trusted internal maintenance operation. It may inspect
            # live evidence in its own space, but never deleted/blocked evidence.
            original = self.core._visible
            if self.control.agents[actor.agent_id].get("trusted_internal_maintenance") is True:
                self.core._visible = lambda item, caller: (
                    original(item, caller)
                    and all(
                        self._available(self.store.get("envelope_v2", key))
                        and self.store.get("envelope_v2", key).get("space") == actor.space_id
                        for key in item.get("source_event_ids", [])
                    )
                    and not self.store.get("blocked_policy", item.get("policy", ""))
                )
            else:
                self.core._visible = lambda item, caller: (
                    original(item, caller)
                    and all(self._visible(self.store.get("envelope_v2", key), actor)
                            for key in item.get("source_event_ids", []))
                )
            try:
                results = []
                maintenance_subjects = ([actor.principal_id] if self.control.access_mode == "shared_pool"
                                        else self.control.spaces[actor.space_id]["members"])
                for subject in maintenance_subjects:
                    caller = self._internal(actor, subject)
                    results.append(
                        self.scheduler.submit(actor.task_id, caller)
                        if self.scheduler
                        else self.core.finish_task(actor.task_id, caller)
                    )
                job["status"] = (
                    "completed"
                    if all(r["status"] in {"completed", "disabled"} for r in results)
                    else "needs_attention"
                )
                self.store.put("sleep_v2", job_id, job)
                return {"jobs": results, "permission_maintenance": self.control.maintenance_summary(actor)}
            except Exception:
                job["status"] = "failed"
                self.store.put("sleep_v2", job_id, job)
                raise
            finally:
                self.core._visible = original

    def recover(self):
        """Trusted startup worker. Reapply current access rules, never legacy recovery."""
        results = []
        for row in self.store.all("sleep_v2"):
            if row["status"] == "running":
                try:
                    results.append(self.sleep(Actor(**row["actor"])))
                except (AccessDenied, RuntimeError, ValidationError):
                    row["status"] = "needs_attention"
                    self.store.put("sleep_v2", row["id"], row)
        actors = {canonical(j["actor"]): j["actor"] for j in self.store.all("erase_queue_v2")}
        for who in actors.values():
            try:
                results.append(self.retry_deletes(Actor(**who)))
            except AccessDenied:
                # Deleted identities require operator cleanup; data remains blocked.
                results.append({"status": "pending_cleanup"})
        for job in self.store.all("redaction_queue_v2"):
            try:
                results.append(self._resume_redaction(job))
            except (AccessDenied, RuntimeError, ValidationError):
                results.append({"status": "pending_cleanup"})
        return results

    def retry_redactions(self, actor):
        self.control.authenticate(actor)
        jobs = [j for j in self.store.all("redaction_queue_v2") if j["actor"] == asdict(actor)]
        outcomes = [self._resume_redaction(job) for job in jobs]
        return {"status": "pending_cleanup" if any(r.get("status") != "forgotten" for r in outcomes) else "forgotten",
                "pending": sum(r.get("status") != "forgotten" for r in outcomes), "jobs": len(jobs)}

    def propose_redaction(self, actor, event_id, requested_fragment):
        """Ask for a candidate span only after read and delete permission checks."""
        with self.lock:
            row = self.store.get("envelope_v2", event_id)
            event = self.store.get("event", event_id)
            if not row or not event or row.get("space") != actor.space_id:
                raise AccessDenied("删除目标不存在或不在当前空间")
            self.control.require(actor, row["subject"], row["category"], "delete", resource=row.get("item_id"))
            if not self._visible(row, actor):
                raise AccessDenied("无权读取待处理的删除目标")
            result = self.core._model("redaction_plan", {"source_text": event["text"], "requested_fragment": requested_fragment})
            fragment = result.get("remove_text") if isinstance(result, dict) and set(result) == {"remove_text"} else None
            if (not isinstance(fragment, str) or not fragment.strip() or requested_fragment not in fragment
                or event["text"].count(fragment) != 1):
                return None
            return {"event_id": event_id, "remove_text": fragment}

    def rewrite_redacted_event(self, actor, event_id, remove_text, target):
        """Conservatively erase one mixed event and reindex its exact retained text.

        The caller must independently verify the plan against the user's request.
        Refuse known copies/derivatives or an unavailable original writer. This does
        not find hidden retellings or claim all external logs have been erased.
        """
        with self.lock:
            row = self.store.get("envelope_v2", event_id)
            event = self.store.get("event", event_id)
            if not row or not event or row.get("space") != actor.space_id:
                raise AccessDenied("删除目标不存在或不在当前空间")
            self.control.require(actor, row["subject"], row["category"], "delete", resource=row.get("item_id"))
            if not self._visible(row, actor):
                raise AccessDenied("无权读取待处理的删除目标")
            if (not isinstance(remove_text, str) or not isinstance(target, str) or not target.strip()
                or target not in remove_text or event["text"].count(remove_text) != 1):
                return {"status": "needs_review", "reason": "invalid_redaction_span"}
            retained = event["text"].replace(remove_text, "", 1).strip()
            if target.casefold() in retained.casefold() or not retained:
                return {"status": "needs_review", "reason": "ambiguous_or_empty_remainder"}
            for other in self.store.all("event"):
                if other["event_id"] != event_id and target.casefold() in other.get("text", "").casefold():
                    return {"status": "needs_review", "reason": "other_possible_copies"}
            for card in self.store.all("card"):
                if event_id not in card.get("source_event_ids", []) and target.casefold() in (
                    (card.get("claim_text") or "") + " " + (card.get("source_quote") or "")
                ).casefold():
                    return {"status": "needs_review", "reason": "other_possible_copies"}
            if row.get("sources") or any(event_id in r.get("sources", []) for r in self.store.all("envelope_v2")):
                return {"status": "needs_review", "reason": "dependent_sources"}
            if event.get("speaker") != "user" or target.casefold() in canonical(row.get("metadata", {})).casefold():
                return {"status": "needs_review", "reason": "unsupported_provenance"}
            writer = Actor(**row["actor"])
            self.control.require(writer, row["subject"], row["category"], "write", resource=row.get("item_id"))
            job_id = stable_id("redaction", actor.space_id, event_id, self._digest(target))
            job = {"id": job_id, "actor": asdict(actor), "writer": asdict(writer), "event_id": event_id,
                   "subject": row["subject"], "category": row["category"], "item_id": row.get("item_id"),
                   "scope": row["scope"], "observed_at": row.get("observed_at"), "metadata": row.get("metadata", {}),
                   "retained": retained, "status": "pending"}
            self.store.put("redaction_queue_v2", job_id, job)
            return self._resume_redaction(job)

    def _resume_redaction(self, job):
        requester, writer = Actor(**job["actor"]), Actor(**job["writer"])
        row = self.store.get("envelope_v2", job["event_id"])
        if not row:
            return {"status": "pending_cleanup", "reason": "source_missing"}
        try:
            if not row.get("deleted"):
                result = self.forget(requester, [job["event_id"]])
                if result.get("status") not in {"forgotten", "already_forgotten"}:
                    return {"status": "pending_cleanup"}
            fields = {"subject": job["subject"], "evidence_kind": "explicit_user_statement", "scope": job["scope"],
                      "source_quote": job["retained"], "conditions": [], "state_key": None, "value": None,
                      "unit": None, "event_time": None, "valid_from": None, "valid_to": None,
                      "planned_for": None, "correction_of": None, "state_aspect": None, "claim": job["retained"]}
            result = self.ingest(writer, event_id="redacted:" + job["id"], subject_id=job["subject"],
                                 category=job["category"], text=job["retained"], item_id=job["item_id"],
                                 scope=job["scope"], observed_at=job["observed_at"], metadata=job["metadata"],
                                 state_eligible=False, card_fields=fields)
            if result.get("status") != "done":
                return {"status": "pending_cleanup"}
            self.store.remove("redaction_queue_v2", job["id"])
            return {"status": "forgotten", "replacement_event_id": result["event_id"], "scope": "verified_fragment_only"}
        except (AccessDenied, RuntimeError, ValidationError):
            return {"status": "pending_cleanup"}

    def forget_exact_fragment(self, actor, event_ids, fragment):
        """Erase existing whole fragments only; mixed fragments require review.

        The caller must independently verify the request and supply exact event IDs.
        This does not claim to find unknown retellings or redact a mixed event.
        """
        if not isinstance(fragment, str) or not fragment.strip() or not isinstance(event_ids, list) or not event_ids:
            raise ValidationError("部分删除需要明确片段和事件ID")
        with self.lock:
            for key in event_ids:
                row = self.store.get("envelope_v2", key)
                event = self.store.get("event", key)
                if not row or not event or row.get("space") != actor.space_id:
                    raise AccessDenied("删除目标不存在或不在当前空间")
                self.control.require(actor, row["subject"], row["category"], "delete", resource=row.get("item_id"))
                if event.get("text", "").strip() != fragment.strip():
                    return {"status": "needs_review", "reason": "mixed_or_unmatched_fragment"}
            return self.forget(actor, event_ids)

    def forget(self, actor, event_ids):
        with self.lock:
            self.control.authenticate(actor)
            if not isinstance(event_ids, list) or not event_ids:
                raise ValidationError("必须明确指定事件ID")
            roots = set(event_ids)
            for key in roots:
                row = self.store.get("envelope_v2", key)
                if not row or row["space"] != actor.space_id:
                    raise AccessDenied("无权管理目标")
                self.control.require(actor, row["subject"], row["category"], "delete", resource=row.get("item_id"))
            rows = self.store.all("envelope_v2")
            while True:
                shared_fragments = {
                    r["source_fragment_id"] for r in rows if r["id"] in roots and r.get("source_fragment_id")
                }
                expanded = roots | {
                    r["id"]
                    for r in rows
                    if set(r["sources"]) & roots or r.get("source_fragment_id") in shared_fragments
                }
                if expanded == roots:
                    break
                roots = expanded
            # Denial and persistent cleanup queue are committed before native erasure begins.
            with self.store.transaction():
                for row in rows:
                    if row["id"] in roots:
                        row.update(blocked=True)
                        self.store.put("envelope_v2", row["id"], row)
                        self.store.put("erase_queue_v2", row["id"], {"id": row["id"], "actor": asdict(actor)})
                for message in self.store.all("message_v2"):
                    if set(message["events"]) & roots:
                        # Direct-mode messages retain model-produced card fields and
                        # exact quotations too; removing only fragments would leave
                        # the supposedly erased plaintext in the sidecar database.
                        message.update(fragments=[], cards=[], deleted=True)
                        self.store.put("message_v2", message["id"], message)
                        source = self.store.get("source_message_v2", message["id"])
                        if source:
                            source.update(text="", metadata={}, deleted=True, status="deleted")
                            self.store.put("source_message_v2", message["id"], source)
                for card in self.store.all("card"):
                    if set(card["source_event_ids"]) & roots:
                        self.store.remove("pending_state", card["id"])
            self.control._audit(actor, "delete", sorted(roots))
            return self.retry_deletes(actor)

    def retry_deletes(self, actor):
        with self.lock:
            self.control.authenticate(actor)
            pending = 0
            for job in self.store.all("erase_queue_v2"):
                if job["actor"] != asdict(actor):
                    continue
                row = self.store.get("envelope_v2", job["id"])
                caller = self._internal(actor, row["subject"])
                try:
                    marker = self.store.get(
                        "forgotten", stable_id("event", row["subject"], row["space"], row["id"])
                    )
                    # An interrupted pre-ingest envelope may have no core event yet.
                    result = (
                        self.core.forget([row["id"]], caller)
                        if marker or self.store.get("event", row["id"])
                        else {"status": "forgotten"}
                    )
                    if result["status"] == "pending_cleanup":
                        pending += 1
                        continue
                    for key in row["card_ids"]:
                        self.store.remove("pending_state", key)
                    self.store.put(
                        "erased_content_v2",
                        stable_id("erased", row["space"], row["fingerprint"]),
                        {"at": self.clock()},
                    )
                    row.update(deleted=True, blocked=True, card_ids=[])
                    self.store.put("envelope_v2", row["id"], row)
                    self.store.remove("erase_queue_v2", row["id"])
                except (AccessDenied, RuntimeError, ValidationError):
                    pending += 1
            return {"status": "pending_cleanup" if pending else "forgotten", "pending": pending}
