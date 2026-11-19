"""Executable adaptation of dynamic user-agent-resource edges and provenance checks.

Not the authors' released implementation or a claim of reproducing paper scores.
Administrative methods are trusted operations; never expose them as agent tools.
"""

from __future__ import annotations

from .backend import partition
from .models import AccessDenied, Caller, Event, ValidationError, canonical, now, stable_id
from .service import MemoryService


class AccessGraph:
    def __init__(self, store):
        self.store = store

    def provision(self, user_id, team_id, readers, writers, managers=(), category=None):
        readers, writers = sorted(set(readers)), sorted(set(writers))
        if not readers or not set(writers) <= set(readers):
            raise ValidationError("写入者必须属于读取集合")
        policy = partition(
            dict(user_id=user_id, team_id=team_id, allowed_reader_ids=readers, category=category)
        )
        existing = self.store.get("policy", policy)
        if existing:
            # Restart must NOT silently restore revoked permissions from config.
            return policy
        self.store.put(
            "policy",
            policy,
            dict(
                id=policy,
                user_id=user_id,
                team_id=team_id,
                original_readers=readers,
                readers=readers,
                writers=writers,
                managers=sorted(set(managers)),
                revision=1,
            ),
        )
        return policy

    def permits(self, policy, caller, operation="read"):
        row = self.store.get("policy", policy)
        field = {"read": "readers", "write": "writers", "manage": "managers"}[operation]
        return bool(
            row
            and (row["user_id"], row["team_id"]) == (caller.user_id, caller.team_id)
            and caller.agent_id in row[field]
        )

    def change(self, policy, agent_id, *, read=None, write=None):
        """Trusted admin only. Grants cannot expand the original evidence audience."""
        with self.store.transaction():
            row = self.store.get("policy", policy)
            if not row or agent_id not in row["original_readers"]:
                raise AccessDenied("不能通过授权边扩大原始证据的可见集合；须显式发布新证据")
            for field, enabled in (("readers", read), ("writers", write)):
                if enabled is not None:
                    if type(enabled) is not bool:
                        raise ValidationError("权限值必须为布尔值")
                    values = set(row[field])
                    (values.add if enabled else values.discard)(agent_id)
                    row[field] = sorted(values)
            row["writers"] = sorted(set(row["writers"]) & set(row["readers"]))
            row["revision"] += 1
            self.store.put("policy", policy, row)
            self.store.put(
                "audit",
                stable_id("acl", policy, row["revision"]),
                dict(
                    policy=policy,
                    revision=row["revision"],
                    agent_id=agent_id,
                    read=read,
                    write=write,
                    at=now(),
                ),
            )
        return row["revision"]

    def approve_disclosure(self, event: Event):
        """Owner/admin approves exact text + source + audience, never a generic bypass flag."""
        if not event.disclosure_id or not event.source_root_ids:
            raise ValidationError("发布授权必须绑定ID和来源")
        self.store.put(
            "disclosure",
            event.disclosure_id,
            {"event": event.to_dict(), "revoked": False, "approved_at": now()},
        )

    def disclosed(self, event):
        grant = self.store.get("disclosure", event.get("disclosure_id", ""))
        fields = {k: event[k] for k in Event.__dataclass_fields__ if k in event}
        return bool(
            grant
            and not grant["revoked"]
            and canonical(grant["event"]) == canonical(Event.from_dict(fields).to_dict())
        )

    def revoke_disclosure(self, disclosure_id):
        row = self.store.get("disclosure", disclosure_id)
        if not row:
            raise AccessDenied("发布授权不存在")
        row["revoked"] = True
        self.store.put("disclosure", disclosure_id, row)


class GovernedMemoryService(MemoryService):
    def __init__(self, settings, backend, model, store, graph):
        self.graph = graph
        super().__init__(
            settings,
            backend,
            model,
            store,
            disclosure_authorizer=lambda event, source: graph.disclosed(event),
        )

    def _visible(self, item, caller):
        if not MemoryService._visible(item, caller):
            return False
        policy = item.get("policy") or partition(item)
        if not self.graph.permits(policy, caller):
            return False
        # Re-evaluate current permissions along provenance even for old derived copies.
        source_ids = set(item.get("source_event_ids", []))
        for assertion in item.get("assertions", []):
            for card_id in assertion["evidence_card_ids"]:
                card = self.store.get("card", card_id)
                if not card or not self._visible(card, caller):
                    return False
        if "event_id" in item and "text" in item:
            return self._event_visible(item, caller, set())
        return all(self._event_visible(self.store.get("event", key), caller, set()) for key in source_ids)

    def _event_visible(self, event, caller, visited):
        if not event or not MemoryService._visible(event, caller):
            return False
        if not self.graph.permits(event["policy"], caller):
            return False
        if event.get("disclosure_id"):
            return self.graph.disclosed(event)
        key = event["event_id"]
        if key in visited:
            return False
        visited = visited | {key}
        return all(
            root == key or self._event_visible(self.store.get("event", root), caller, visited)
            for root in event["source_root_ids"]
        )

    def _event(self, event, caller):
        policy = partition(event.to_dict())
        if not self.graph.permits(policy, caller, "write"):
            raise AccessDenied("当前Agent没有该记忆域的写入权限")
        # A historical derived state may now depend on a revoked resource. Never
        # place that state into the writer's LLM prompt, even if its cohort still permits reads.
        for state in self.store.states():
            if state.get("policy") == policy and not self._visible(state, caller):
                raise AccessDenied("该域含已撤权的派生状态，须由有权维护者清理后再写入")
        return super()._event(event, caller)

    def forget(self, source_ids, caller):
        for key in source_ids:
            event = self.store.get("event", key)
            if event and not self.graph.permits(event["policy"], caller, "manage"):
                raise AccessDenied("读取权不包含删除权")
        return super().forget(source_ids, caller)

    def get_card(self, card_id, caller):
        self._authorize(caller)
        card = self.store.get("card", card_id)
        if not card or not self._visible(card, caller):
            raise AccessDenied("记录不存在或无权读取")
        return card

    def get_history(self, state_id, caller):
        self._authorize(caller)
        state = self.store.state(state_id)
        if not state or not self._visible(state, caller):
            raise AccessDenied("记录不存在或无权读取")
        return [item for item in self.store.history(state_id) if self._visible(item, caller)]

    def get_trace(self, card_id, caller):
        card = self.get_card(card_id, caller)
        events = []
        for key in sorted(set(card["source_event_ids"] + card["source_root_ids"])):
            event = self.store.get("event", key)
            if event and self._visible(event, caller):
                events.append(event)
        return {"card": card, "readable_source_events": events}

    def _validate_card(self, fields, event, changes):
        if fields.get("evidence_kind") == "execution_result" and event["speaker"] != "tool":
            raise ValidationError("执行成功必须有可信工具结果；Agent自述不能提升为成功证据")
        return super()._validate_card(fields, event, changes)

    def check_plan_dependencies(self, plan_id, caller):
        result = super().check_plan_dependencies(plan_id, caller)
        return result

    def _can_cleanup(self, item, caller):
        return self._visible(item, caller) or (
            item.get("user_id") == caller.user_id
            and item.get("team_id") == caller.team_id
            and caller.agent_id in item.get("cleanup_agent_ids", [])
        )

    def finish_team_task(self, task_id, user_id, team_id):
        """Trusted runner: process separate authorized bundles, never pool private prompts."""
        results = {}
        for agent in self.settings.team_agents.get(team_id, []):
            results[agent] = self.finish_task(task_id, Caller(user_id, team_id, agent))
        return results
