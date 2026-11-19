from test_multiuser import actor, env  # noqa: F401

from ours_memory.semantic_views import compact_card, compact_state


def test_compact_views_keep_evidence_and_omit_operational_metadata():
    card = compact_card({"id": "c1", "subject": "Asha", "author_id": "Noah", "claim_text": "EEG Friday",
                         "source_quote": "EEG Friday", "use_hint": "Use EEG Friday", "policy": "private",
                         "source_root_ids": ["root"], "metadata": {"noise": True}})
    assert card == {"id": "c1", "subject": "Asha", "author_id": "Noah", "claim_text": "EEG Friday",
                    "source_quote": "EEG Friday", "use_hint": "Use EEG Friday"}
    state = compact_state({"id": "s1", "subject": "Asha", "state_key": "task_plan", "revision": 1,
                           "policy": "private", "assertions": [{"value": "Friday", "status": "pending",
                           "evidence_card_ids": ["c1"], "source_root_ids": ["root"]}]})
    assert state == {"id": "s1", "subject": "Asha", "state_key": "task_plan", "revision": 1,
                     "assertions": [{"value": "Friday", "status": "pending", "evidence_card_ids": ["c1"]}]}


def test_usage_view_preserves_time_in_claim_without_repeated_time_fields():
    card = {"id": "c", "subject": "zhang", "claim_text": "Zhang attends class on Friday",
            "source_quote": "I attend class on Friday", "planned_for": "2026-10-30T09:00:00+08:00",
            "event_time": "2026-10-29T10:00:00+08:00", "valid_from": "2026-10-29T10:00:00+08:00"}
    view = compact_card(card, for_usage=True)
    assert "Friday" in view["claim_text"] and "Friday" in view["source_quote"]
    assert not {"planned_for", "event_time", "valid_from", "valid_to"} & set(view)
    assert compact_card(card)["valid_from"] == card["valid_from"]


def test_direct_card_persists_hint_without_selection(env):  # noqa: F811
    m, model, _, _ = env
    m.control.access_mode = "shared_pool"
    m.settings.card_mode = "direct"
    original = model.complete
    select_inputs = []
    def complete(stage, payload):
        if stage == "cards_direct":
            return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "EEG Friday",
                    "source_quote": "I have EEG Friday", "use_hint": "Use EEG Friday",
                    "evidence_kind": "explicit_user_statement", "state_key": None, "value": None,
                    "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                    "valid_to": None, "planned_for": None, "correction_of": None, "state_aspect": None}]}
        if stage == "select":
            select_inputs.append(payload)
        return original(stage, payload)
    model.complete = complete
    result = m.ingest_message(actor(), message_id="hint", text="I have EEG Friday", fallback_category="logistics")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["use_hint"] == "Use EEG Friday"
    assert card["source_event_ids"] and card["source_root_ids"]
    bundle = m.read(actor(), subject_ids=["marcus"], query="EEG", purpose="evidence")
    assert select_inputs == []
    assert bundle["cards"][0]["usage_note"] == "Use EEG Friday"
    assert bundle["cards"][0]["use_hint"] == "Use EEG Friday"


def test_extraction_accepts_focused_card_without_optional_fields(env):  # noqa: F811
    m, model, _, _ = env
    m.control.access_mode = "shared_pool"
    m.settings.card_mode = "direct"
    model.complete = lambda stage, payload: {"cards": [{
        "subject": "marcus", "category": "logistics", "claim": "I prefer tea", "source_quote": "I prefer tea",
        "use_hint": "Use for tea preference", "evidence_kind": "explicit_user_statement",
        "state_key": None, "state_aspect": None, "value": None,
    }]}
    result = m.ingest_message(actor(), message_id="compact", text="I prefer tea", fallback_category="logistics")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["claim_text"] == "I prefer tea"
    assert card["use_hint"] == "Use for tea preference"
    assert card["conditions"] == [] and card["state_key"] is None
