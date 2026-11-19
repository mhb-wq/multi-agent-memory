"""Regression tests for the macro fixes found by the GroupMemBench experiment."""

from __future__ import annotations

import pytest

from ours_memory.backend import FixtureBackend
from ours_memory.config import Settings
from ours_memory.control import Actor
from ours_memory.models import MemoryChange
from ours_memory.multiuser import MultiUserMemory
from ours_memory.relative_dates import resolve_relative
from ours_memory.store import Store

ANCHOR = "2025-07-15T09:00:00+00:00"  # Tuesday


def control_config(people, team=None):
    members = list(people) + ([team] if team else [])
    categories = ["clinical", "logistics"]
    grants = []
    for grantee in members:
        for subject in members:
            for category in categories:
                grants.append(
                    {
                        "id": f"{grantee}-{subject}-{category}",
                        "issuer": subject,
                        "grantee": grantee,
                        "subject": subject,
                        "space": "care",
                        "category": category,
                        "actions": ["read", "write", "confirm", "delete"],
                        "valid_from": None,
                        "valid_to": None,
                    }
                )
    return {
        "principals": members,
        "categories": categories,
        "agents": {
            "clinical": {
                "categories": categories,
                "actions": ["read", "write", "confirm", "delete"],
                "controls": ["grant", "revoke", "sleep"],
            }
        },
        "spaces": {
            "care": {
                "members": members,
                "agents": ["clinical"],
                "grant_managers": {member: [member] for member in members},
            }
        },
        "grants": grants,
    }


class OneMemoryBackend(FixtureBackend):
    def add(self, event, policy):
        self.changes[event.event_id] = [MemoryChange("m_" + event.event_id, event.text)]
        return super().add(event, policy)


class ScoredBackend(OneMemoryBackend):
    def __init__(self):
        super().__init__()
        self.scores = {}

    def search(self, query, caller, policies, limit):
        items = [
            item
            for event, policy, item in self.items.values()
            if event["user_id"] == caller.user_id and policy in policies
        ]
        return sorted(items, key=lambda item: -self.scores.get(item.memory_id, 0.0))[:limit]


class StubModel:
    """Deterministic model with a per-test card factory."""

    def __init__(self, factory):
        self.calls = []
        self.factory = factory

    def complete(self, stage, payload):
        self.calls.append(stage)
        if stage == "cards":
            return {"cards": [self.factory(payload["event"], memory) for memory in payload["memories"]]}
        if stage == "state":
            return {
                "decisions": [
                    {
                        "state_id": key,
                        "expected_revision": group["state"]["revision"],
                        "operations": [
                            {
                                "kind": "UPDATE" if group["state"]["assertions"] else "INIT",
                                "card_id": group["cards"][0]["id"],
                            }
                        ],
                    }
                    for key, group in payload["groups"].items()
                ]
            }
        if stage == "select":
            return {"card_ids": [card["id"] for card in payload["cards"]]}
        if stage == "answer":
            return {"text": "ok", "card_ids": []}
        if stage == "output_check":
            return {"supported": True}
        if stage == "sleep":
            return {"state_id": payload["state"]["id"], "operations": []}
        raise AssertionError(stage)


def plain_card(event, memory):
    return {
        "memory_id": memory["memory_id"],
        "subject": event["user_id"],
        "evidence_kind": "explicit_user_statement",
        "state_key": None,
        "value": None,
        "unit": None,
        "scope": event["scope"],
        "conditions": [],
        "event_time": event["recorded_at"],
        "valid_from": event["recorded_at"],
        "valid_to": None,
        "source_quote": event["text"],
        "correction_of": None,
    }


def build(tmp_path, model, backend, people=("p1", "p2", "p3", "p4"), team=None):
    settings = Settings(
        database=str(tmp_path / "db.sqlite3"),
        team_agents={"care": ["clinical"]},
        state_keys={
            "budget": {"type": "number", "unit": None, "min": 0},
            "goal": {"type": "text", "unit": None},
        },
    )
    store = Store(settings.database)
    memory = MultiUserMemory(settings, backend, model, store, control_config(people, team))
    return memory, store


def actor(person):
    return Actor(person, "clinical", "care", "t")


def test_relative_dates_are_deterministic_and_conservative():
    assert resolve_relative("handoff Thursday", ANCHOR) == "2025-07-17T00:00:00+00:00"
    assert resolve_relative("tomorrow", ANCHOR) == "2025-07-16T00:00:00+00:00"
    assert resolve_relative("next Thursday", ANCHOR) == "2025-07-24T00:00:00+00:00"
    assert resolve_relative("in 3 days", ANCHOR) == "2025-07-18T00:00:00+00:00"
    assert resolve_relative("end of month", ANCHOR) == "2025-07-31T00:00:00+00:00"
    assert resolve_relative("下周四", ANCHOR) == "2025-07-24T00:00:00+00:00"
    assert resolve_relative("月底", ANCHOR) == "2025-07-31T00:00:00+00:00"
    assert resolve_relative("明天", ANCHOR) == "2025-07-16T00:00:00+00:00"
    assert resolve_relative("soon-ish", ANCHOR) is None


def test_plan_gets_scheduled_date_and_is_visible_before_and_after(tmp_path):
    def factory(event, memory):
        card = plain_card(event, memory)
        card.update(evidence_kind="plan", event_time=None, valid_from=None)
        return card

    memory, store = build(tmp_path, StubModel(factory), OneMemoryBackend(), people=("p1",))
    result = memory.ingest(
        actor("p1"),
        event_id="e1",
        subject_id="p1",
        category="logistics",
        text="handoff Thursday",
        observed_at=ANCHOR,
    )
    card = store.get("card", result["card_ids"][0])
    assert card["planned_for"] == "2025-07-17T00:00:00+00:00"
    assert card["valid_from"] == card["event_time"] == ANCHOR

    def read_at(target):
        return memory.read(actor("p1"), subject_ids=["p1"], query="handoff", target_time=target)

    assert card["id"] not in [c["id"] for c in read_at("2025-07-14T00:00:00+00:00")["cards"]]
    assert card["id"] in [c["id"] for c in read_at("2025-07-16T00:00:00+00:00")["cards"]]
    assert card["id"] in [c["id"] for c in read_at("2025-07-20T00:00:00+00:00")["cards"]]


def test_pending_state_card_is_not_dropped_from_current_read(tmp_path):
    def factory(event, memory):
        card = plain_card(event, memory)
        card.update(state_key="budget", value=100, valid_from=None)
        return card

    memory, store = build(tmp_path, StubModel(factory), OneMemoryBackend(), people=("p1",))
    result = memory.ingest(
        actor("p1"),
        event_id="e1",
        subject_id="p1",
        category="logistics",
        text="100",
        observed_at=ANCHOR,
    )
    memory.sleep(actor("p1"))
    state = store.states()[0]
    assert state["assertions"][0]["status"] == "pending"
    read = memory.read(actor("p1"), subject_ids=["p1"], query="budget")
    assert [card["id"] for card in read["cards"]] == [result["card_ids"][0]]
    assert read["cards"][0]["state_status"] == "pending"


def test_candidates_cover_every_subject_and_respect_limit(tmp_path):
    model = StubModel(plain_card)
    backend = ScoredBackend()
    people = ("p1", "p2", "p3", "p4")
    memory, store = build(tmp_path, model, backend, people=people)
    for person in people:
        for index in range(6):
            memory.ingest(
                actor(person),
                event_id=f"{person}-{index}",
                subject_id=person,
                category="logistics",
                text=f"note {index}",
                observed_at=ANCHOR,
            )
    # The last subject owns the highest-scored card; the old subject-ordered cut lost it.
    backend.scores[f"m_{people[-1]}-5"] = 1.0
    read = memory.read(actor("p1"), subject_ids=list(people), query="note")
    subjects = {card["subject"] for card in read["cards"]}
    assert subjects == set(people)
    assert len(read["cards"]) <= memory.settings.candidate_limit


def test_sleep_groups_same_state_into_one_call_and_clears_dirty(tmp_path):
    def factory(event, memory):
        card = plain_card(event, memory)
        card.update(state_key="budget", value=int(event["text"]))
        return card

    model = StubModel(factory)
    memory, store = build(tmp_path, model, OneMemoryBackend(), people=("p1",))
    for index, value in enumerate(("100", "200")):
        memory.ingest(
            actor("p1"),
            event_id=f"e{index}",
            subject_id="p1",
            category="logistics",
            text=value,
            observed_at=ANCHOR,
        )
    memory.sleep(actor("p1"))
    assert model.calls.count("state") == 1
    assert store.dirty_states() == []


def test_registered_team_subject_owns_one_shared_state(tmp_path):
    people = ("p1", "p2", "p3")
    team = "team-x"
    model = StubModel(
        lambda event, memory: {
            **plain_card(event, memory),
            "state_key": "goal",
            "value": "shared plan",
        }
    )
    memory, store = build(tmp_path, model, OneMemoryBackend(), people=people, team=team)
    first = memory.ingest(
        actor("p1"),
        event_id="t1",
        subject_id=team,
        category="logistics",
        text="shared plan",
        observed_at=ANCHOR,
    )
    second = memory.ingest(
        actor("p2"),
        event_id="t2",
        subject_id=team,
        category="logistics",
        text="shared plan",
        observed_at=ANCHOR,
    )
    first_card = store.get("card", first["card_ids"][0])
    second_card = store.get("card", second["card_ids"][0])
    from ours_memory.state import state_id

    assert state_id(first_card) == state_id(second_card)
    memory.sleep(actor("p1"))
    team_states = [state for state in store.states() if state["subject"] == team]
    assert len(team_states) == 1
    read = memory.read(actor("p3"), subject_ids=[team], query="plan")
    assert read["cards"]


def test_conditional_card_requires_matching_conditions(tmp_path):
    def factory(event, memory):
        card = plain_card(event, memory)
        card["conditions"] = ["exam"]
        return card

    memory, _ = build(tmp_path, StubModel(factory), OneMemoryBackend(), people=("p1",))
    result = memory.ingest(
        actor("p1"),
        event_id="e1",
        subject_id="p1",
        category="logistics",
        text="conditional note",
        observed_at=ANCHOR,
    )
    assert memory.read(actor("p1"), subject_ids=["p1"], query="note")["cards"] == []
    read = memory.read(actor("p1"), subject_ids=["p1"], query="note", conditions=["exam"])
    assert [card["id"] for card in read["cards"]] == [result["card_ids"][0]]


def test_confirmed_state_evidence_reaches_candidates_despite_quota(tmp_path):
    def factory(event, memory):
        card = plain_card(event, memory)
        if event["text"] == "100":
            card.update(state_key="budget", value=100)
        return card

    people = ("p1", "p2", "p3", "p4", "p5")
    memory, store = build(tmp_path, StubModel(factory), OneMemoryBackend(), people=people)
    for person in people:
        for index in range(5):
            memory.ingest(
                actor(person),
                event_id=f"{person}-{index}",
                subject_id=person,
                category="logistics",
                text="100" if (person, index) == ("p5", 4) else f"note {index}",
                observed_at=ANCHOR,
            )
    memory.sleep(actor("p1"))
    state_card = next(
        card for card in store.all("card") if card["source_event_ids"] and card.get("state_key")
    )
    read = memory.read(actor("p1"), subject_ids=list(people), query="note")
    assert state_card["id"] in [card["id"] for card in read["cards"]]


def test_many_subjects_still_get_candidate_coverage(tmp_path):
    people = tuple(f"u{index}" for index in range(18))
    memory, _ = build(tmp_path, StubModel(plain_card), OneMemoryBackend(), people=people)
    for person in people:
        for index in range(2):
            memory.ingest(
                actor(person),
                event_id=f"{person}-{index}",
                subject_id=person,
                category="logistics",
                text=f"note {index}",
                observed_at=ANCHOR,
            )
    read = memory.read(actor("u0"), subject_ids=list(people), query="note")
    assert {card["subject"] for card in read["cards"]} == set(people)


def test_unregistered_team_subject_is_rejected(tmp_path):
    from ours_memory.control import ControlPlane
    from ours_memory.models import ValidationError

    config = control_config(("p1",))
    config["spaces"]["care"]["members"] = ["p1", "ghost"]
    with pytest.raises(ValidationError):
        ControlPlane(Store(str(tmp_path / "ghost.sqlite3")), config)
