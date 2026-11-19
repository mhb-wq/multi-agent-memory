"""Regression tests for the 2026-10-27 diagnostic fixes.

1. An unregistered state key must not discard the whole card (evidence retention).
2. The internal selector must respect the model input budget and must not fail a
   whole read when the payload is too large.
"""
import copy

from test_multiuser import Decisions, NativeFixture, actor, env  # noqa: F401

from ours_memory.models import InputTooLarge, canonical


def _cards_reply(event, state_key):
    return {
        "cards": [
            {
                "memory_id": event["text"] and None,
            }
        ]
    }


def test_unregistered_state_key_degrades_to_evidence(env):  # noqa: F811
    m, model, _, _ = env
    original = model.complete

    def complete(stage, payload):
        model.calls.append((stage, copy.deepcopy(payload)))
        if stage == "cards":
            event = payload["event"]
            return {
                "cards": [
                    {
                        "memory_id": payload["memories"][0]["memory_id"],
                        "subject": event["user_id"],
                        "evidence_kind": "plan",
                        "state_key": "plan",
                        "value": "ship on Friday",
                        "unit": None,
                        "scope": event["scope"],
                        "conditions": [],
                        "event_time": event["recorded_at"],
                        "valid_from": event["recorded_at"],
                        "valid_to": None,
                        "source_quote": event["text"],
                        "correction_of": None,
                    }
                ]
            }
        return original(stage, payload)

    model.complete = complete
    result = m.ingest(
        actor(),
        event_id="plan-1",
        subject_id="marcus",
        category="logistics",
        text="we plan to ship on Friday",
    )
    card = m.store.get("card", result["card_ids"][0])
    assert card["state_key"] is None
    assert card["state_key_dropped"] == "plan"
    assert card["value"] is None and card["unit"] is None
    assert not m.store.all("pending_state")


def test_read_trims_oversized_select_payload(env):  # noqa: F811
    m, model, _, _ = env
    for index in range(8):
        m.ingest(
            actor(),
            event_id=f"long-{index}",
            subject_id="marcus",
            category="logistics",
            text=("detail " * 400) + f" item {index}",
        )
    m.settings.max_model_input_chars = 9000
    payloads = []
    original = model.complete

    def complete(stage, payload):
        if stage == "select":
            payloads.append(payload)
        return original(stage, payload)

    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="current information")
    assert payloads, "selector was not called"
    assert len(canonical(payloads[-1])) <= m.settings.max_model_input_chars
    assert bundle["cards"]


def test_read_falls_back_when_select_still_too_large(env):  # noqa: F811
    m, model, _, _ = env
    m.ingest(actor(), event_id="a", subject_id="marcus", category="logistics", text="arrival Friday")
    m.ingest(actor(), event_id="b", subject_id="marcus", category="logistics", text="budget 8000")
    original = model.complete
    seen = []

    def complete(stage, payload):
        if stage == "select":
            seen.append(payload)
            raise InputTooLarge("forced")
        return original(stage, payload)

    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="current information")
    assert bundle["cards"], "fallback must return evidence rather than fail"
    assert len(bundle["cards"]) <= 8


def test_empty_selection_falls_back_to_ranked_candidates(env):  # noqa: F811
    m, model, _, _ = env
    m.ingest(actor(), event_id="a", subject_id="marcus", category="logistics", text="arrival Friday")
    m.ingest(actor(), event_id="b", subject_id="marcus", category="logistics", text="budget 8000")
    original = model.complete

    def complete(stage, payload):
        if stage == "select":
            return {"card_ids": []}
        return original(stage, payload)

    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="current information")
    assert bundle["cards"], "empty selector output must fall back to ranked candidates"
    assert len(bundle["cards"]) <= 5


def test_direct_card_mode_one_call_and_indexes_quote(env):  # noqa: F811
    """card_mode=direct: one message -> one model call -> cards indexed as claim + quote."""
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    text = "arrival Friday at 9 and the budget is 8000"
    original = model.complete

    def complete(stage, payload):
        if stage == "cards_direct":
            model.calls.append((stage, payload))
            return {
                "cards": [
                    {
                        "subject": "marcus",
                        "category": "logistics",
                        "claim": "Marcus arrives Friday at 09:00",
                        "source_quote": "arrival Friday at 9",
                        "evidence_kind": "explicit_user_statement",
                        "state_key": None,
                        "value": None,
                        "unit": None,
                        "conditions": [],
                        "event_time": None,
                        "valid_from": None,
                        "valid_to": None,
                        "planned_for": None,
                        "correction_of": None,
                        "state_aspect": None,
                    },
                    {
                        "subject": "marcus",
                        "category": "logistics",
                        "claim": "The budget is 8000",
                        "source_quote": "the budget is 8000",
                        "evidence_kind": "explicit_user_statement",
                        "state_key": None,
                        "value": None,
                        "unit": None,
                        "conditions": [],
                        "event_time": None,
                        "valid_from": None,
                        "valid_to": None,
                        "planned_for": None,
                        "correction_of": None,
                        "state_aspect": None,
                    },
                ]
            }
        return original(stage, payload)

    model.complete = complete
    result = m.ingest_message(
        actor(),
        message_id="direct-1",
        text=text,
        fallback_category="logistics",
        metadata={"channel": "project-a"},
    )
    stages = [stage for stage, _ in model.calls if isinstance(stage, str)]
    assert stages.count("cards_direct") == 1
    assert "intake" not in stages and "cards" not in stages
    assert result["status"] == "done" and len(result["events"]) == 2
    cards = [
        m.store.get("card", card_id)
        for event in result["events"]
        for card_id in event["card_ids"]
    ]
    assert len(cards) == 2
    indexed = {item.text for _event, _policy, item in m.backend.items.values()}
    for card in cards:
        assert f"{card['claim_text']}\n{card['source_quote']}" in indexed
