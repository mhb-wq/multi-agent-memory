import copy

import pytest
from test_multiuser import actor, env, ingest  # noqa: F401

from ours_memory.models import AccessDenied, ValidationError


def test_resource_rule_and_specific_deny(env):  # noqa: F811
    m, _, _, _ = env
    manager = actor()
    broad = {"id": "resource-allow", "issuer": "marcus", "grantee": "nina", "subject": "marcus",
             "space": "care", "category": "clinical", "actions": ["read"], "valid_from": None,
             "valid_to": None, "resource": "appointment"}
    m.control.grant(manager, broad)
    nina = actor("nina", "clinical")
    assert m.control.permits(nina, "marcus", "clinical", "read", resource="appointment")
    assert not m.control.permits(nina, "marcus", "clinical", "read", resource="medication")
    m.control.grant(manager, {**broad, "id": "resource-deny", "effect": "deny"})
    assert not m.control.permits(nina, "marcus", "clinical", "read", resource="appointment")
    m.control.revoke(manager, "resource-deny")
    assert m.control.permits(nina, "marcus", "clinical", "read", resource="appointment")


def test_resource_denial_blocks_read_before_selector(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m, event_id="appt", category="clinical", item_id="appointment")
    ingest(m, event_id="med", text="secret medicine", category="clinical", item_id="medication")
    spec = {"id": "read-appointment", "issuer": "marcus", "grantee": "nina", "subject": "marcus",
            "space": "care", "category": "clinical", "actions": ["read"], "valid_from": None,
            "valid_to": None, "resource": "appointment"}
    m.control.grant(actor(), spec)
    result = m.read(actor("nina", "clinical"), subject_ids=["marcus"], query="information")
    assert len(result["cards"]) == 1
    assert result["cards"][0]["claim_text"] == "arrival Friday"
    assert result["cards"][0]["usage_note"]
    assert "secret medicine" not in str([p for stage, p in model.calls if stage == "select"])


def test_direct_schema_repairs_once(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    fields = {"subject": "marcus", "category": "logistics", "claim": "Friday", "source_quote": "Friday",
              "evidence_kind": "explicit_user_statement", "state_key": None, "value": None, "unit": None,
              "conditions": [], "event_time": None, "valid_from": None, "valid_to": None,
              "planned_for": None, "correction_of": None, "state_aspect": None}
    calls = []
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        calls.append(copy.deepcopy(payload))
        return {"cards": [{"claim": "Friday"}] if len(calls) == 1 else [fields]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="repair", text="Friday", fallback_category="logistics")
    assert result["status"] == "done" and len(calls) == 2
    assert "validation_feedback" in calls[1]


def test_direct_extraction_keeps_unknown_source_time_unknown(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    calls = []
    fields = {"subject": "marcus", "category": "logistics", "claim": "meeting next Monday",
              "source_quote": "meeting next Monday", "evidence_kind": "explicit_user_statement",
              "state_key": None, "value": None, "unit": None, "conditions": [], "event_time": None,
              "valid_from": None, "valid_to": None, "planned_for": None, "correction_of": None,
              "state_aspect": None}

    def complete(stage, payload):
        if stage == "cards_direct":
            calls.append(copy.deepcopy(payload))
            return {"cards": [fields]}
        return original(stage, payload)

    model.complete = complete
    m.ingest_message(actor(), message_id="undated-1", text="meeting next Monday",
                     fallback_category="logistics", metadata={"source_time_unknown": "true"})
    assert calls[0]["observed_at"] is None
    assert calls[0]["metadata"]["source_time_unknown"] == "true"


def test_direct_cards_receive_past_context_and_record_ids(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    calls = []
    def complete(stage, payload):
        if stage == "cards_direct":
            calls.append(copy.deepcopy(payload))
            return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "Marcus arrives Friday",
                    "source_quote": "arrives Friday", "evidence_kind": "explicit_user_statement", "state_key": None,
                    "value": None, "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                    "valid_to": None, "planned_for": None, "correction_of": None, "state_aspect": None}]}
        return {"card_ids": []}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="context-2", text="arrives Friday",
                              fallback_category="logistics", prior_context=[{"id": "context-1", "text": "Marcus is coming"}])
    assert calls[0]["prior_context"][0]["id"] == "context-1"
    event = m.store.get("event", result["events"][0]["event_id"])
    assert event["metadata"]["context_ids"] == "context-1"


def test_shared_direct_unknown_subject_cannot_create_personal_state(env):  # noqa: F811
    m, model, _, _ = env
    m.control.access_mode = "shared_pool"
    m.settings.card_mode = "direct"
    def complete(stage, payload):
        assert stage == "cards_direct"
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "Marcus arrives Friday",
                "source_quote": "arrives Friday", "evidence_kind": "explicit_user_statement",
                "state_key": "appointment", "value": "Friday", "unit": None, "conditions": [],
                "event_time": None, "valid_from": None, "valid_to": None, "planned_for": None,
                "correction_of": None, "state_aspect": None}]}
    model.complete = complete
    result = m.ingest_message(actor("patel"), message_id="ambiguous", text="arrives Friday",
                              fallback_category="logistics")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] is None and card["state_key"] is None
    assert card["claim_text"] == "arrives Friday"
    assert not card["state_eligible"] and not m.store.all("pending_state")


def test_shared_role_analysis_conflicting_owner_does_not_assign_actor(env):  # noqa: F811
    m, model, _, _ = env
    m.control.access_mode, m.settings.card_mode = "shared_pool", "direct"
    traces = []
    m.observer = lambda stage, data: traces.append((stage, data))
    def complete(stage, payload):
        assert stage == "cards_direct"
        return {"cards": [{"subject": "patel", "category": "logistics",
                "claim": "I moved her appointment", "source_quote": "I moved her appointment",
                "evidence_kind": "explicit_user_statement", "state_key": None, "value": None, "unit": None,
                "conditions": [], "event_time": None, "valid_from": None, "valid_to": None,
                "planned_for": None, "correction_of": None, "state_aspect": None,
                "analysis": {"speaker": "patel", "event": "appointment change", "references": [
                    {"mention": "her", "resolved_to": None, "evidence_message_ids": []}],
                    "actor": "patel", "fact_owner": None, "other_participants": [],
                    "owner_basis": "appointment owner unclear"}}]}
    model.complete = complete
    result = m.ingest_message(actor("patel"), message_id="roles", text="I moved her appointment",
                              fallback_category="logistics")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] is None and card["state_key"] is None
    assert card["subject_analysis"]["actor"] == "patel"
    assert next(d for s, d in traces if s == "subject_analysis")["analysis"]["actor"] == "patel"


@pytest.mark.parametrize("basis,expected", [
    ([{"id": "earlier", "quote": "My EEG appointment is Thursday"}], "marcus"),
    ([{"id": "future", "quote": "My EEG appointment is Thursday"}], None),
    ([{"id": "earlier", "quote": "invented quote"}], None),
    ([], None),
])
def test_shared_context_subject_requires_exact_prior_evidence(env, basis, expected):  # noqa: F811
    m, model, _, _ = env
    m.control.access_mode, m.settings.card_mode = "shared_pool", "direct"
    m.settings.state_keys["task_plan"] = {"type": "text", "unit": None}
    def complete(stage, payload):
        assert stage == "cards_direct"
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "The EEG moved to Friday",
                "source_quote": "I moved the EEG to Friday", "evidence_kind": "explicit_user_statement",
                "state_key": "task_plan", "state_aspect": "eeg_appointment", "value": "Friday",
                "unit": None, "conditions": [], "event_time": None, "valid_from": None, "valid_to": None,
                "planned_for": None, "correction_of": None, "subject_context": basis}]}
    model.complete = complete
    prior = [{"id": "earlier", "speaker": "marcus", "text": "My EEG appointment is Thursday"}]
    result = m.ingest_message(actor("patel"), message_id="later", text="I moved the EEG to Friday",
                              fallback_category="logistics", prior_context=prior)
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] == expected
    assert card["subject_context"] == (basis if expected else [])
    assert card["subject_basis"] == ("prior_context" if expected else "unresolved")
    assert card["source_quote"] == "I moved the EEG to Friday"
    assert bool(card["state_key"]) == bool(expected)


def test_shared_mode_allows_read_across_subjects_but_not_control(env):  # noqa: F811
    m, _, _, _ = env
    m.control.access_mode = "shared_pool"
    ingest(m, text="Martha arrival", who=actor("martha"), subject="martha")
    result = m.read(actor(), subject_ids=["marcus"], query="arrival")
    assert [c["claim_text"] for c in result["cards"]] == ["Martha arrival"]
    assert not m.control.permits(actor(), "martha", "logistics", "delete")
    with pytest.raises(AccessDenied):
        m.control.revoke(actor(), "any-grant")


def test_shared_subject_requires_unambiguous_current_message_evidence():
    from ours_memory.multiuser import MultiUserMemory

    members = ["patient_nadia_lopez", "patient_marta_lopez", "nurse_imani"]
    assert MultiUserMemory._explicit_subject("patient_nadia_lopez", "nurse_imani", "Nadia has an appointment.", members)
    assert not MultiUserMemory._explicit_subject("patient_nadia_lopez", "nurse_imani", "Lopez has an appointment.", members)
    assert not MultiUserMemory._explicit_subject("patient_nadia_lopez", "nurse_imani", "The appointment is Friday.", members)
    assert MultiUserMemory._explicit_subject("nurse_imani", "nurse_imani", "My availability is Friday.", members)


def test_registered_nonpersonal_subject_stays_evidence_without_user_state(env):  # noqa: F811
    m, model, _, _ = env
    m.control.access_mode, m.settings.card_mode = "shared_pool", "direct"
    m.control.state_subjects.discard("martha")
    def complete(stage, payload):
        assert stage == "cards_direct"
        return {"cards": [{"subject": "martha", "category": "logistics", "claim": "Martha arrives Friday",
                "source_quote": "Martha arrives Friday", "evidence_kind": "explicit_user_statement",
                "state_key": "appointment", "value": "Friday", "unit": None, "conditions": [],
                "event_time": None, "valid_from": None, "valid_to": None, "planned_for": None,
                "correction_of": None, "state_aspect": None}]}
    model.complete = complete
    result = m.ingest_message(actor("patel"), message_id="nonperson", text="Martha arrives Friday",
                              fallback_category="logistics")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] == "martha" and card["state_key"] is None
    assert not card["state_eligible"] and not m.store.all("pending_state")


def test_no_cross_subject_delete_grant(env):  # noqa: F811
    m, _, _, _ = env
    result = ingest(m)
    with pytest.raises(AccessDenied):
        m.forget(actor("nina", "controller"), [result["event_id"]])


def test_question_free_text_uses_are_bounded_and_reach_answer(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m)
    original = model.complete

    def complete(stage, payload):
        if stage == "select":
            card_id = payload["cards"][0]["id"]
            return {"card_ids": [card_id], "uses": {card_id: "Use this card as historical evidence."}}
        return original(stage, payload)

    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="when was the old appointment?")
    assert "historical evidence" in bundle["cards"][0]["usage_note"]
    assert "historical evidence" in m.bundle(actor(), bundle["bundle_id"])["cards"][0]["usage_note"]
    m.answer(actor(), bundle["bundle_id"], "when was the old appointment?")
    answer_payload = [payload for stage, payload in model.calls if stage == "answer"][-1]
    assert "historical evidence" in answer_payload["cards"][0]["usage_note"]


def test_untrusted_free_text_use_label_is_not_forwarded(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m)
    original = model.complete

    def complete(stage, payload):
        if stage == "select":
            card_id = payload["cards"][0]["id"]
            return {"card_ids": [card_id], "uses": {card_id: "invented sensitive value"}}
        return original(stage, payload)

    model.complete = complete
    card = m.read(actor(), subject_ids=["marcus"], query="when?")["cards"][0]
    assert "invented sensitive value" not in card["usage_note"]


def test_grounded_free_text_advice_is_forwarded(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m)
    original = model.complete

    def complete(stage, payload):
        if stage == "select":
            card_id = payload["cards"][0]["id"]
            return {"card_ids": [card_id], "uses": {card_id: "Use arrival Friday as direct evidence."}}
        return original(stage, payload)

    model.complete = complete
    card = m.read(actor(), subject_ids=["marcus"], query="when?")["cards"][0]
    assert "Use arrival Friday as direct evidence." in card["usage_note"]


def test_free_text_advice_cannot_echo_private_guess_from_question(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m)
    original = model.complete
    def complete(stage, payload):
        if stage == "select":
            card_id = payload["cards"][0]["id"]
            return {"card_ids": [card_id], "uses": {card_id: "Use secret 6714 as direct evidence."}}
        return original(stage, payload)
    model.complete = complete
    card = m.read(actor(), subject_ids=["marcus"], query="Is the secret 6714?")["cards"][0]
    assert "6714" not in card["usage_note"]


def test_selector_requires_fixed_card_ids_and_uses_mapping(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m)
    recorded = []
    m.observer = lambda stage, payload: recorded.append((stage, payload))
    original = model.complete
    def complete(stage, payload):
        if stage == "select":
            return {"card_ids": [payload["cards"][0]["id"]], "uses": {"other-card": "invented detail"}}
        return original(stage, payload)
    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="when?")
    assert bundle["cards"]
    assert "invented detail" not in bundle["cards"][0]["usage_note"]
    assert any(stage == "select_format_retry" for stage, _ in recorded)
    assert any(stage == "select_fallback" for stage, _ in recorded)


def test_selector_retries_unparseable_json_with_concrete_format_feedback(env):  # noqa: F811
    from ours_memory.models import ModelOutputError
    m, model, _, _ = env
    ingest(m)
    original = model.complete
    calls = []
    def complete(stage, payload):
        if stage != "select":
            return original(stage, payload)
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            raise ModelOutputError("bad JSON")
        card_id = payload["cards"][0]["id"]
        return {"card_ids": [card_id], "uses": {card_id: "Use arrival Friday as direct evidence."}}
    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="when?")
    assert len(calls) == 2 and "JSON" in calls[1]["validation_feedback"]
    assert '"card_ids"' in calls[1]["repair_instruction"] and '"uses"' in calls[1]["repair_instruction"]
    assert "Use arrival Friday as direct evidence." in bundle["cards"][0]["usage_note"]


def test_selector_second_bad_format_returns_candidate_cards_without_any_notes(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m, text="arrival Friday", event_id="one")
    ingest(m, text="appointment Monday", event_id="two")
    original = model.complete
    calls, events = [], []
    m.observer = lambda stage, payload: events.append((stage, payload))
    def complete(stage, payload):
        if stage != "select":
            return original(stage, payload)
        calls.append(copy.deepcopy(payload))
        return {"card_ids": [payload["cards"][0]["id"]], "uses": {"foreign": "wrong"}}
    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="appointment")
    assert len(calls) == 2
    assert "一一对应" in calls[1]["validation_feedback"]
    assert len(bundle["cards"]) == 2
    assert all(card["usage_note"] == "" for card in bundle["cards"])
    assert any(stage == "select_fallback" and data["reason"] == "format_retry_failed" for stage, data in events)


def test_selector_both_unparseable_still_reaches_answer_with_cards_only(env):  # noqa: F811
    from ours_memory.models import ModelOutputError
    m, model, _, _ = env
    ingest(m)
    original = model.complete
    calls = []
    def complete(stage, payload):
        if stage == "select":
            calls.append(payload)
            raise ModelOutputError("bad JSON")
        return original(stage, payload)
    model.complete = complete
    bundle = m.read(actor(), subject_ids=["marcus"], query="when?")
    assert len(calls) == 2 and bundle["cards"][0]["usage_note"] == ""
    assert m.answer(actor(), bundle["bundle_id"], "when?")["action"] == "answer"
    payload = [p for stage, p in model.calls if stage == "answer"][-1]
    assert payload["cards"] and payload["cards"][0]["usage_note"] == ""


def test_direct_forget_clears_cached_card_plaintext(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    secret = "private-test-code-493721"
    original = model.complete
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": secret,
                "source_quote": secret, "evidence_kind": "explicit_user_statement", "state_key": None,
                "value": None, "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                "valid_to": None, "planned_for": None, "correction_of": None, "state_aspect": None}]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="erase-direct", text=secret, fallback_category="logistics")
    assert secret in str(m.store.all("message_v2"))
    erased = m.forget(actor(), [result["events"][0]["event_id"]])
    assert erased["status"] == "forgotten"
    assert secret not in str(m.store.all("message_v2"))
    assert secret not in str(m.store.all("card"))
    assert secret not in str(m.store.all("event"))


def test_trusted_resource_manager_can_grant_scoped_delete(env):  # noqa: F811
    m, _, _, _ = env
    # In production this mapping must come from trusted application bootstrap,
    # never from the message text or a model-generated card.
    m.control.spaces["care"]["resource_managers"] = {"door": ["nina"]}
    nina = actor("nina", "clinical")
    rule = {"id": "door-delete", "issuer": "nina", "grantee": "nina", "subject": "marcus",
            "space": "care", "category": "logistics", "actions": ["delete"], "valid_from": None,
            "valid_to": None, "resource": "door"}
    m.control.grant(nina, rule)
    assert m.control.permits(nina, "marcus", "logistics", "delete", resource="door")
    assert not m.control.permits(nina, "marcus", "logistics", "delete", resource="another-door")
    with pytest.raises(AccessDenied):
        m.control.grant(nina, {**rule, "id": "unscoped", "resource": None})
    event = ingest(m, event_id="managed-door", item_id="door")
    assert m.forget(nina, [event["event_id"]])["status"] == "forgotten"


def test_common_control_entry_verifies_before_executing(env):  # noqa: F811
    m, _, _, _ = env
    grant = {"id": "new-nina", "issuer": "marcus", "grantee": "nina", "subject": "marcus",
             "space": "care", "category": "clinical", "actions": ["read"], "valid_from": None,
             "valid_to": None}
    text = "I authorize Nina to read my clinical records."
    proposal = {"action": "grant", "operation_id": "1", "source_quote": text, "grant": grant}
    m.configure_control_recognition(lambda turn: {"status": "control", "operations": [proposal]},
                                    lambda turn, op: False, agent_id="clinical")
    with pytest.raises(ValidationError):
        m.ingest_message(actor(), message_id="control", text=text, fallback_category="clinical")
    assert not m.control.permits(actor("nina", "clinical"), "marcus", "clinical", "read")
    assert not m.store.all("message_v2")
    m.configure_control_recognition(lambda turn: {"status": "control", "operations": [proposal]},
                                    lambda turn, op: True, agent_id="clinical")
    result = m.ingest_message(actor(), message_id="control", text=text, fallback_category="clinical")
    assert result["status"] == "control_executed"
    assert m.control.permits(actor("nina", "clinical"), "marcus", "clinical", "read")
    assert not m.store.all("message_v2")


def test_forget_exact_fragment_preserves_other_and_refuses_mixed(env):  # noqa: F811
    m, _, _, _ = env
    secret = ingest(m, text="PIN 6714", event_id="secret")
    public = ingest(m, text="Mia arrives Friday", event_id="public")
    mixed = ingest(m, text="PIN 6714 and Mia arrives Friday", event_id="mixed")
    result = m.forget_exact_fragment(actor(), [mixed["event_id"]], "PIN 6714")
    assert result["status"] == "needs_review"
    assert m.store.get("event", mixed["event_id"])
    result = m.forget_exact_fragment(actor(), [secret["event_id"]], "PIN 6714")
    assert result["status"] == "forgotten"
    assert not m.store.get("event", secret["event_id"])
    assert m.store.get("event", public["event_id"])


def test_date_only_plan_is_grounded_and_retains_card(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "Align today",
                "source_quote": "align today", "evidence_kind": "explicit_user_statement", "state_key": None,
                "value": None, "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                "valid_to": None, "planned_for": "2026-10-25", "correction_of": None, "state_aspect": None}]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="date", text="align today", fallback_category="logistics")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["planned_for"] == "2026-10-24T16:00:00+00:00"


def test_source_expansion_finds_authorized_sibling(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    m.settings.source_expansion_enabled = True
    original = model.complete
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        base = {"subject": "marcus", "category": "logistics", "evidence_kind": "explicit_user_statement",
                "state_key": None, "value": None, "unit": None, "conditions": [], "event_time": None,
                "valid_from": None, "valid_to": None, "planned_for": None, "correction_of": None, "state_aspect": None}
        return {"cards": [{**base, "claim": "Old process", "source_quote": "old process"},
                          {**base, "claim": "New process", "source_quote": "new process"}]}
    model.complete = complete
    m.ingest_message(actor(), message_id="two-facts", text="old process; new process", fallback_category="logistics")
    search = m.backend.search
    m.backend.search = lambda *args, **kwargs: search(*args, **kwargs)[:1]
    observed = []
    m.observer = lambda stage, payload: observed.append((stage, payload))
    result = m.read(actor(), subject_ids=["marcus"], query="process", purpose="evidence")
    assert {c["claim_text"] for c in result["cards"]} == {"Old process", "New process"}
    assert any(stage == "source_expansion" for stage, _ in observed)


def test_evidence_read_keeps_candidates_even_when_model_selects_one(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m, text="first relevant fact", event_id="keep-one")
    ingest(m, text="second possibly relevant fact", event_id="keep-two")
    original = model.complete
    def complete(stage, payload):
        if stage == "select":
            key = payload["cards"][0]["id"]
            return {"card_ids": [key], "uses": {key: "Relevant evidence."}}
        return original(stage, payload)
    model.complete = complete
    result = m.read(actor(), subject_ids=["marcus"], query="fact", purpose="evidence")
    assert {c["claim_text"] for c in result["cards"]} == {"first relevant fact", "second possibly relevant fact"}


def test_personal_identity_and_task_plan_are_separate_pending_states(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    m.settings.state_keys.update({"identity": {"type": "text", "unit": None},
                                  "task_plan": {"type": "text", "unit": None}})
    original = model.complete
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        text = payload["text"]
        is_identity = "height" in text
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": text,
                "source_quote": text, "evidence_kind": "explicit_user_statement",
                "state_key": "identity" if is_identity else "task_plan",
                "state_aspect": "height" if is_identity else "friday_appointment",
                "value": "170 cm" if is_identity else "Friday", "unit": None, "conditions": [],
                "event_time": None, "valid_from": None, "valid_to": None,
                "planned_for": None, "correction_of": None}]}
    model.complete = complete
    for ident, text in [("identity", "My height is 170 cm"), ("task", "My appointment is Friday")]:
        result = m.ingest_message(actor(), message_id=ident, text=text, fallback_category="logistics")
        card = m.store.get("card", result["events"][0]["card_ids"][0])
        assert card["state_key"] == ("identity" if ident == "identity" else "task_plan")
        assert m.store.get("pending_state", card["id"])


def test_distinct_state_aspects_do_not_share_address(env):  # noqa: F811
    from ours_memory.state import state_id
    m, _, _, _ = env
    result = ingest(m, text="200")
    card = m.store.get("card", result["card_ids"][0])
    assert state_id({**card, "state_aspect": "budget"}) != state_id({**card, "state_aspect": "study_time"})
    assert state_id(card) == state_id({**card, "state_aspect": None})


def test_direct_content_validation_retries_before_write(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    calls = []
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        calls.append(payload)
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "Friday",
                "source_quote": "invented" if len(calls) == 1 else "Friday",
                "evidence_kind": "explicit_user_statement", "state_key": None, "value": None,
                "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                "valid_to": None, "planned_for": None, "correction_of": None, "state_aspect": None}]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="content-repair", text="Friday", fallback_category="logistics")
    assert result["status"] == "done" and len(calls) == 2
    assert "validation_feedback" in calls[1]
    assert len(m.store.all("card")) == 1 and len(m.store.all("event")) == 1


def test_uncited_numeric_claim_triggers_repair(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    calls = []
    text = "The fare is 1400. I like that option."
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        calls.append(payload)
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "The fare is 1400",
                "source_quote": "I like that option." if len(calls) == 1 else "The fare is 1400.",
                "evidence_kind": "explicit_user_statement", "state_key": None, "state_aspect": None,
                "value": None, "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                "valid_to": None, "planned_for": None, "correction_of": None}]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="numeric-grounding", text=text, fallback_category="logistics")
    assert result["status"] == "done" and len(calls) == 2
    assert "validation_feedback" in calls[1]


def test_failed_numeric_repair_keeps_quote_and_other_cards(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    calls = []
    text = "The fare is 1400. I like that option. I prefer a window seat."
    base = {"subject": "marcus", "category": "logistics", "evidence_kind": "explicit_user_statement",
            "state_key": None, "state_aspect": None, "value": None, "unit": None, "conditions": [],
            "event_time": None, "valid_from": None, "valid_to": None, "planned_for": None,
            "correction_of": None}
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        calls.append(payload)
        return {"cards": [{**base, "claim": "The fare is 1400", "source_quote": "I like that option."},
                          {**base, "claim": "Prefers a window seat", "source_quote": "I prefer a window seat."}]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="numeric-fallback", text=text, fallback_category="logistics")
    cards = [m.store.get("card", key) for e in result["events"] for key in e["card_ids"]]
    assert result["status"] == "done" and len(calls) == 2 and len(cards) == 2
    assert cards[0]["claim_text"] == "I like that option."
    assert cards[0]["grounding_issue"] == "uncited_numeric_claim_replaced_by_quote"
    assert cards[1]["claim_text"] == "Prefers a window seat"


def test_mixed_dialogue_assistant_suggestion_does_not_update_user_state(env):  # noqa: F811
    m, model, _, _ = env
    m.settings.card_mode = "direct"
    original = model.complete
    text = "user: I need an alarm.\nassistant: Would you like gentle music?"
    def complete(stage, payload):
        if stage != "cards_direct":
            return original(stage, payload)
        return {"cards": [{"subject": "marcus", "category": "logistics", "claim": "User prefers gentle music",
                "source_quote": "Would you like gentle music?", "evidence_kind": "explicit_user_statement",
                "state_key": "preference", "state_aspect": "alarm_sound", "value": "gentle music",
                "unit": None, "conditions": [], "event_time": None, "valid_from": None,
                "valid_to": None, "planned_for": None, "correction_of": None}]}
    model.complete = complete
    result = m.ingest_message(actor(), message_id="mixed", text=text, fallback_category="logistics",
                              metadata={"role": "user_assistant_session"})
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["state_key"] is None
    assert card["evidence_kind"] == "agent_inference"
    assert not m.store.all("pending_state")


def test_verified_llm_redaction_preserves_other_fact(env):  # noqa: F811
    m, model, _, _ = env
    original = model.complete
    def complete(stage, payload):
        if stage == "redaction_plan":
            return {"remove_text": "PIN 6714"}
        return original(stage, payload)
    model.complete = complete
    source = ingest(m, text="Mia arrives Friday; PIN 6714; enter east door", event_id="mixed-redact")
    turn = {"turn_id": "erase-request", "speaker_id": "marcus", "text": "Delete PIN 6714."}
    proposal = {"action": "forget", "operation_id": "op", "source_quote": "Delete PIN 6714.",
                "event_ids": [source["event_id"]], "whole_events_verified": False, "exact_fragment": "6714"}
    def verify(raw, proposed):
        return not proposed.get("redaction_plan") or proposed["redaction_plan"]["remove_text"] == "PIN 6714"
    result = m.execute_control_intent(actor(), turn, proposal, verifier=verify)
    assert result["status"] == "forgotten" and result["scope"] == "verified_fragment_only"
    assert not m.store.get("event", source["event_id"])
    retained = m.store.get("event", result["replacement_event_id"])["text"]
    assert "6714" not in retained and "Mia arrives Friday" in retained and "enter east door" in retained
    assert not m.store.all("redaction_queue_v2")


def test_unverified_llm_redaction_never_erases(env):  # noqa: F811
    m, model, _, _ = env
    original = model.complete
    model.complete = lambda stage, payload: ({"remove_text": "PIN 6714"} if stage == "redaction_plan" else original(stage, payload))
    source = ingest(m, text="Mia arrives Friday; PIN 6714", event_id="redact-denied")
    turn = {"turn_id": "erase-request", "speaker_id": "marcus", "text": "Delete PIN 6714."}
    proposal = {"action": "forget", "operation_id": "op", "source_quote": "Delete PIN 6714.",
                "event_ids": [source["event_id"]], "whole_events_verified": False, "exact_fragment": "6714"}
    result = m.execute_control_intent(actor(), turn, proposal, verifier=lambda raw, prop: "redaction_plan" not in prop)
    assert result["status"] == "needs_review"
    assert m.store.get("event", source["event_id"])


def test_redaction_refuses_known_copy_and_keeps_original(env):  # noqa: F811
    m, _, _, _ = env
    source = ingest(m, text="Mia arrives Friday; PIN 6714", event_id="source")
    ingest(m, text="PIN 6714", event_id="other-copy")
    result = m.rewrite_redacted_event(actor(), source["event_id"], "PIN 6714", "6714")
    assert result["status"] == "needs_review"
    assert result["reason"] == "other_possible_copies"
    assert m.store.get("event", source["event_id"])


def test_redaction_retry_recovers_retained_text_after_index_error(env):  # noqa: F811
    m, model, _, _ = env
    original_model = model.complete
    model.complete = lambda stage, payload: ({"remove_text": "PIN 6714"} if stage == "redaction_plan" else original_model(stage, payload))
    source = ingest(m, text="Mia arrives Friday; PIN 6714", event_id="recover-source")
    index = m.backend.index_card
    failures = []
    def flaky_index(*args, **kwargs):
        if not failures:
            failures.append(True)
            raise RuntimeError("temporary index failure")
        return index(*args, **kwargs)
    m.backend.index_card = flaky_index
    turn = {"turn_id": "recover-request", "speaker_id": "marcus", "text": "Delete PIN 6714."}
    proposal = {"action": "forget", "operation_id": "recover", "source_quote": "Delete PIN 6714.",
                "event_ids": [source["event_id"]], "whole_events_verified": False, "exact_fragment": "6714"}
    first = m.execute_control_intent(actor(), turn, proposal, verifier=lambda raw, prop: True)
    assert first["status"] == "pending_cleanup"
    assert m.store.all("redaction_queue_v2")
    assert "6714" not in str(m.store.all("redaction_queue_v2"))
    second = m.execute_control_intent(actor(), turn, proposal, verifier=lambda raw, prop: True)
    assert second["status"] == "forgotten"
    assert not m.store.all("redaction_queue_v2")
    assert any("Mia arrives Friday" in event["text"] for event in m.store.all("event"))


def test_redaction_diagnostic_trace_does_not_save_secret():
    from ours_memory.groupmem import TracedModel
    entries = []
    class Model:
        def complete(self, stage, payload):
            return {"remove_text": "PIN 6714"}
    result = TracedModel(Model(), lambda stage, payload: entries.append((stage, payload))).complete(
        "redaction_plan", {"source_text": "Mia arrives Friday, PIN 6714"})
    assert result["remove_text"] == "PIN 6714"
    assert "6714" not in str(entries)
