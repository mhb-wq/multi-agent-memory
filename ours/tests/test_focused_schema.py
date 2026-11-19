import copy
import json

import pytest
from test_multiuser import actor

from ours_memory.llm import CompatibleJsonModel
from ours_memory.models import ModelOutputError, ValidationError, stable_id, timestamp
from ours_memory.multiuser import MultiUserMemory

pytest_plugins = ["test_multiuser"]

def shared(env):
    old, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    return MultiUserMemory(old.settings, old.backend, model, old.store, control, clock), model


def extraction(subject="marcus", topics=None):
    return {"cards": [{"subject": subject, "claim": "Marcus plans a Friday class",
                       "usage": "Use for his schedule", "state_topic": topics or ["plan", "goal"],
                       "source": {"speaker": "marcus", "quote": "Friday class", "context": []}}]}


def test_focused_extraction_retries_once_on_bad_format(env):
    memory, model = shared(env)
    calls = []
    def complete(stage, payload):
        assert stage == "focused_cards"
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            return {"cards": [{"subject": "marcus", "claim": "Friday class"}]}
        assert payload["format_feedback"]
        return extraction()
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="retry", text="Friday class")
    assert result["status"] == "done"
    assert result["extraction_status"] == "extracted"
    assert result["extraction_attempts"] == 2
    assert len(calls) == 2 and len(memory.store.all("card")) == 1
    saved = memory.store.get("message_v2", result["message_id"])
    assert len(saved["extraction_format_errors"]) == 1


def test_focused_extraction_skips_after_two_bad_outputs_and_continues(env):
    memory, model = shared(env)
    calls = []
    def complete(stage, payload):
        calls.append(payload["text"])
        if payload["text"] == "bad message":
            if len(calls) == 1:
                raise ModelOutputError("invalid JSON")
            return {"cards": [{"subject": "marcus", "claim": "missing fields"}]}
        return extraction()
    model.complete = complete
    first = memory.ingest_message(actor(), message_id="bad", text="bad message")
    assert first["status"] == "accepted_pending" and first["extraction_status"] == "deferred"
    assert first["extraction_attempts"] == 2 and first["events"] == []
    assert not memory.store.all("card") and not memory.store.all("event")
    assert len(memory.store.get("deferred_message_v2", first["message_id"])["issues"]) == 2
    second = memory.ingest_message(actor(), message_id="good", text="Friday class")
    assert second["extraction_status"] == "extracted"
    assert len(memory.store.all("card")) == 1


def test_focused_extraction_does_not_silently_skip_provider_error(env):
    memory, model = shared(env)
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        raise RuntimeError("provider unavailable")
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="provider", text="Friday class")
    assert result["status"] == "accepted_pending" and result["processing_status"] == "extraction_pending"
    assert result["issues"]
    assert calls == ["focused_cards"]
    assert not memory.store.all("message_v2")


def test_one_call_updates_multiple_topics(env):
    memory, model = shared(env)
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        if stage == "focused_cards":
            return extraction()
        assert stage == "focused_state"
        card = payload["cards"][0]
        return {"subject": payload["subject"], "operations": [
            {"action": "add", "topic": topic, "content": card["claim"],
             "status": "confirmed", "evidence": [card["id"]]} for topic in card["state_topic"]]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    assert len(memory.store.all("card")) == 1
    assert memory.store.get("card", card_id)["state_topic"] == ["plan", "goal"]
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["evidence"] == [card_id]
    assert state["topics"]["goal"][0]["evidence"] == [card_id]
    assert calls == ["focused_cards", "focused_state"]
    assert not memory.store.all("pending_state")
    bundle = memory.read(actor(), subject_ids=["marcus"], query="Friday class")
    assert calls == ["focused_cards", "focused_state"]
    assert bundle["user_states"][0]["topics"]["plan"][0]["evidence"] == [card_id]
    assert bundle["cards"][0]["usage_note"] == "Use for his schedule"
    stored_card = memory.store.get("card", card_id)
    assert all(bundle["cards"][0][key] == value for key, value in stored_card.items())
    assert "event_time" in bundle["cards"][0] and "metadata" in bundle["cards"][0]
    bundle["cards"][0]["metadata"]["test"] = "changed"
    assert memory.store.get("card", card_id)["metadata"] == stored_card["metadata"]


def test_shared_answer_receives_states_and_original_usage_without_selector(env):
    memory, model = shared(env)
    calls = []
    def complete(stage, payload):
        calls.append((stage, copy.deepcopy(payload)))
        if stage == "focused_cards":
            return extraction()
        if stage == "focused_state":
            return {"subject": payload["subject"], "operations": [
                {"action": "add", "topic": "plan", "content": "Friday class",
                 "status": "confirmed", "evidence": [payload["cards"][0]["id"]]}]}
        if stage == "answer":
            assert payload["user_states"][0]["topics"]["plan"][0]["content"] == "Friday class"
            assert payload["cards"][0]["usage_note"] == "Use for his schedule"
            return {"text": "Friday class", "card_ids": [payload["cards"][0]["id"]]}
        raise AssertionError(stage)
    model.complete = complete
    memory.ingest_message(actor(), message_id="answer-direct", text="Friday class")
    memory.sleep(actor())
    bundle = memory.read(actor(), subject_ids=["marcus"], query="When is class?")
    assert memory.bundle(actor(), bundle["bundle_id"])["user_states"] == bundle["user_states"]
    assert memory.answer(actor(), bundle["bundle_id"], "When is class?")["text"] == "Friday class"
    assert [stage for stage, _ in calls] == ["focused_cards", "focused_state", "answer"]


def test_state_not_returned_when_no_supporting_cards_are_retrieved(env):
    memory, model = shared(env)
    key = stable_id("user_state", "care", "marcus")
    state = {"subject": "marcus", "topics": {k: [] for k in ("preference", "constraint", "plan", "profile", "goal")}}
    state["topics"]["plan"] = [{"content": "Friday class", "status": "confirmed", "evidence": ["previous-card"]}]
    memory.store.put("user_state", key, state)
    bundle = memory.read(actor(), subject_ids=["marcus"], query="When is class?")
    assert bundle["cards"] == [] and bundle["user_states"] == []


def test_shared_read_does_not_return_unlinked_states_by_name_alone(env):
    memory, model = shared(env)
    for subject in ("marcus", "martha", "patel"):
        state = {"subject": subject, "topics": {k: [] for k in
                 ("preference", "constraint", "plan", "profile", "goal")}}
        memory.store.put("user_state", stable_id("user_state", "care", subject), state)
    result = memory.read(actor("marcus"), subject_ids=[], query="What are Martha's plans?")
    assert result["user_states"] == []
    assert result["cards"] == []


def test_shared_read_follows_card_evidence_to_all_matching_state_entries(env):
    memory, model = shared(env)
    model.complete = lambda stage, payload: extraction(topics=["plan"])
    result = memory.ingest_message(actor(), message_id="linked", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    topics = ("preference", "constraint", "plan", "profile", "goal")
    marcus = {"subject": "marcus", "topics": {t: [] for t in topics}}
    marcus["topics"]["plan"] = [
        {"content": "Friday class", "status": "confirmed", "evidence": [card_id]},
        {"content": "Unrelated", "status": "confirmed", "evidence": ["other"]}]
    marcus["topics"]["goal"] = [{"content": "Learn", "status": "confirmed", "evidence": [card_id]}]
    martha = {"subject": "martha", "topics": {t: [] for t in topics}}
    martha["topics"]["constraint"] = [{"content": "Related", "status": "confirmed", "evidence": [card_id]}]
    memory.store.put("user_state", stable_id("user_state", "care", "marcus"), marcus)
    memory.store.put("user_state", stable_id("user_state", "care", "martha"), martha)
    bundle = memory.read(actor(), subject_ids=[], query="Friday class")
    by_subject = {state["subject"]: state for state in bundle["user_states"]}
    assert set(by_subject) == {"marcus", "martha"}
    assert [x["content"] for x in by_subject["marcus"]["topics"]["plan"]] == ["Friday class"]
    assert by_subject["marcus"]["topics"]["goal"][0]["content"] == "Learn"
    assert by_subject["martha"]["topics"]["constraint"][0]["content"] == "Related"
    assert memory.store.get("user_state", stable_id("user_state", "care", "marcus")) == marcus


def test_invalid_state_keeps_card_and_previous_state(env):
    memory, model = shared(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            return extraction()
        return {"subject": payload["subject"], "operations": [
            {"action": "add", "topic": "plan", "content": "invented",
             "status": "confirmed", "evidence": ["unknown"]}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    key = stable_id("user_state", "care", "marcus")
    original = {"subject": "marcus", "topics": {t: [] for t in ("preference", "constraint", "plan", "profile", "goal")}}
    memory.store.put("user_state", key, original)
    assert memory.sleep(actor())["jobs"][0]["status"] == "needs_attention"
    assert memory.store.get("user_state", key) == original
    assert memory.store.get("card", card_id)
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"
    failure = memory.store.all("focused_state_failure")[0]
    assert failure["input"]["subject"] == "marcus"
    assert failure["input"]["cards"][0]["id"] == card_id
    assert failure["output"]["operations"][0]["evidence"] == ["unknown"]
    assert failure["failed_field"] == "operations[0].evidence"
    assert failure["error"]["code"] == "ValidationError"
    def recovered(stage, payload):
        return {"subject": payload["subject"], "operations": [
            {"action": "add", "topic": "plan", "content": "Friday class",
             "status": "confirmed", "evidence": [card_id]}]}
    model.complete = recovered
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    assert memory.store.get("pending_state", card_id) is None
    assert memory.store.all("focused_state_failure") == [failure]


def test_partial_topics_merge_preserves_others_and_replaces_changed_topics(env):
    memory, model = shared(env)
    key = stable_id("user_state", "care", "marcus")
    old = {"subject": "marcus", "topics": {t: [] for t in ("preference", "constraint", "plan", "profile", "goal")}}
    old["topics"]["preference"] = [{"content": "Tea", "status": "confirmed", "evidence": ["old-tea"]}]
    old["topics"]["plan"] = [{"content": "Old class", "status": "confirmed", "evidence": ["old-class"]}]
    old["topics"]["goal"] = [{"content": "Old goal", "status": "confirmed", "evidence": ["old-goal"]}]
    memory.store.put("user_state", key, copy.deepcopy(old))
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        if stage == "focused_cards":
            return extraction()
        card = payload["cards"][0]
        return {"subject": "marcus", "operations": [
            {"action": "update", "id": payload["state"]["topics"]["plan"][0]["id"],
             "content": "Friday class", "status": "confirmed", "evidence": [card["id"]]},
            {"action": "remove", "id": payload["state"]["topics"]["goal"][0]["id"]}]}
    model.complete = complete
    memory.ingest_message(actor(), message_id="partial", text="Friday class")
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    updated = memory.store.get("user_state", key)
    assert {k:v for k,v in updated["topics"]["preference"][0].items() if k != "id"} == old["topics"]["preference"][0]
    assert updated["topics"]["constraint"] == [] and updated["topics"]["profile"] == []
    assert updated["topics"]["goal"] == []
    assert updated["topics"]["plan"][0]["content"] == "Friday class"
    assert calls == ["focused_cards", "focused_state"]


def test_empty_topics_is_no_change_and_consumes_pending(env):
    memory, model = shared(env)
    key = stable_id("user_state", "care", "marcus")
    old = {"subject": "marcus", "topics": {t: [] for t in ("preference", "constraint", "plan", "profile", "goal")}}
    old["topics"]["preference"] = [{"content": "Tea", "status": "confirmed", "evidence": ["old"]}]
    memory.store.put("user_state", key, copy.deepcopy(old))
    model.complete = lambda stage, payload: extraction() if stage == "focused_cards" else {"subject": "marcus", "operations": []}
    result = memory.ingest_message(actor(), message_id="nochange", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    assert memory.store.get("user_state", key) == old
    assert memory.store.get("pending_state", card_id) is None


def test_item_operations_are_atomic_and_keep_original_on_invalid_entry(env):
    memory, model = shared(env)
    key = stable_id("user_state", "care", "marcus")
    old = {"subject": "marcus", "topics": {t: [] for t in
           ("preference", "constraint", "plan", "profile", "goal")}}
    old["topics"]["plan"] = [{"content": "Old class", "status": "confirmed", "evidence": ["old"]}]
    memory.store.put("user_state", key, copy.deepcopy(old))
    def complete(stage, payload):
        if stage == "focused_cards":
            return extraction(topics=["plan"])
        return {"subject": "marcus", "operations": [
            {"action": "update", "id": payload["state"]["topics"]["plan"][0]["id"],
             "content": "Updated class", "status": "confirmed", "evidence": [payload["cards"][0]["id"]]},
            {"action": "remove", "id": "not-a-state-item"}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="atomic", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    assert memory.sleep(actor())["jobs"][0]["status"] == "needs_attention"
    assert memory.store.get("user_state", key) == old
    assert memory.store.get("pending_state", card_id)
    assert memory.store.all("focused_state_failure")[0]["failed_field"] == "operations[1].id"


def test_one_state_call_updates_multiple_entries_in_same_topic(env):
    memory, model = shared(env)
    key = stable_id("user_state", "care", "marcus")
    old = {"subject": "marcus", "topics": {t: [] for t in
           ("preference", "constraint", "plan", "profile", "goal")}}
    old["topics"]["plan"] = [
        {"content": "First", "status": "confirmed", "evidence": ["old-1"]},
        {"content": "Second", "status": "confirmed", "evidence": ["old-2"]}]
    memory.store.put("user_state", key, old)
    calls = []
    def complete(stage, payload):
        if stage == "focused_cards":
            return extraction(topics=["plan"])
        calls.append(stage)
        return {"subject": "marcus", "operations": [
            {"action": "update", "id": item["id"], "content": item["content"] + " updated",
             "status": "confirmed", "evidence": [payload["cards"][0]["id"]]}
            for item in payload["state"]["topics"]["plan"]]}
    model.complete = complete
    memory.ingest_message(actor(), message_id="multi-update", text="Friday class")
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    updated = memory.store.get("user_state", key)["topics"]["plan"]
    assert [item["content"] for item in updated] == ["First updated", "Second updated"]
    assert len({item["id"] for item in updated}) == 2
    assert calls == ["focused_state"]


def test_ten_card_threshold_updates_only_eligible_user_and_force_flush(env):
    memory, model = shared(env)
    memory.core.focused_min_pending_cards = 10
    state_calls = []
    def complete(stage, payload):
        if stage == "focused_cards":
            return extraction(subject=payload["author_id"], topics=["plan"])
        state_calls.append((payload["subject"], len(payload["cards"])))
        return {"subject": payload["subject"], "operations": []}
    model.complete = complete
    for i in range(10):
        memory.ingest_message(actor(), message_id=f"m{i}", text=f"Friday class {i}")
    memory.ingest_message(actor("martha"), message_id="other", text="Friday class")
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    assert state_calls == [("marcus", 10)]
    assert len(memory.store.all("pending_state")) == 1
    memory.core.focused_min_pending_cards = 1
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    assert state_calls == [("marcus", 10), ("martha", 1)]
    assert not memory.store.all("pending_state")


def test_unknown_topic_cannot_change_existing_state(env):
    memory, model = shared(env)
    model.complete = lambda stage, payload: extraction() if stage == "focused_cards" else {"subject": "marcus", "operations": [{"action": "add", "topic": "unknown", "content": "x", "status": "confirmed", "evidence": ["x"]}]}
    result = memory.ingest_message(actor(), message_id="invalid-topic", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    assert memory.sleep(actor())["jobs"][0]["status"] == "needs_attention"
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"
    assert memory.store.get("user_state", stable_id("user_state", "care", "marcus")) is None


def test_unaffected_topics_and_card_topic_are_not_checked(env):
    memory, model = shared(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            return extraction(topics=["plan"])
        card = payload["cards"][0]
        return {"subject": payload["subject"], "operations": [
            {"action": "add", "topic": "constraint", "content": "Context matters",
             "status": "confirmed", "evidence": [card["id"]]}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["constraint"][0]["evidence"] == [card_id]
    assert not memory.store.all("focused_state_failure")


def test_model_failure_preserves_available_raw_output_and_retries(env):
    memory, model = shared(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            return extraction(topics=["plan"])
        error = RuntimeError("model output incomplete")
        error.model_output = '{"subject":"marcus",'
        raise error
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    assert memory.sleep(actor())["jobs"][0]["status"] == "needs_attention"
    failure = memory.store.all("focused_state_failure")[0]
    assert failure["output"] is None
    assert failure["raw_model_output"] == '{"subject":"marcus",'
    assert failure["failed_field"] == "model_response"
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"


def test_truncated_provider_output_is_kept_in_failure_record(env, monkeypatch):
    memory, model = shared(env)
    original = model.complete
    model.complete = lambda stage, payload: extraction(topics=["plan"]) if stage == "focused_cards" else original(stage, payload)
    memory.ingest_message(actor(), message_id="m1", text="Friday class")
    monkeypatch.setenv(memory.settings.api_key_env, "fixture-key")
    memory.settings.base_url = "https://example.test"
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "{partial"}}]}).encode()
    class Opener:
        def open(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr("ours_memory.llm.urllib.request.build_opener", lambda *args: Opener())
    memory.core.model = CompatibleJsonModel(memory.settings, memory.store)
    assert memory.sleep(actor())["jobs"][0]["status"] == "needs_attention"
    failure = memory.store.all("focused_state_failure")[0]
    assert failure["raw_model_output"] == "{partial"
    assert failure["error"]["code"] == "OutputTruncated"


def test_shared_pool_rejects_old_direct_event_interface(env):
    memory, _ = shared(env)
    with pytest.raises(ValidationError):
        memory.ingest(actor(), event_id="e1", subject_id="marcus", category="logistics", text="Friday class")


def test_unattributed_card_is_not_queued(env):
    memory, model = shared(env)
    def complete(stage, payload):
        result = extraction()
        result["cards"][0]["subject"] = None
        result["cards"][0]["state_topic"] = ["plan"]  # Model supplies a topic despite missing subject.
        return result
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card = memory.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] is None
    assert card["state_topic"] == []
    assert not memory.store.all("pending_state")


def test_candidates_only_include_registered_users_and_nonuser_subject_is_cleared(env):
    old, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    control["state_subjects"] = list(control["principals"])
    control["principals"].append("project_example")
    control["spaces"]["care"]["members"].append("project_example")
    memory = MultiUserMemory(old.settings, old.backend, model, old.store, control, clock)
    def complete(stage, payload):
        assert stage == "focused_cards"
        assert "marcus" in payload["subjects"]
        assert "project_example" not in payload["subjects"]
        result = extraction(subject="project_example")
        return result
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card = memory.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] is None
    assert card["state_topic"] == []
    assert not memory.store.all("pending_state")


@pytest.mark.parametrize("observed_at,metadata,expected_source_time", [
    (None, {"source_time_unknown": "true"}, None),
    ("2026-10-20T10:00:00+08:00", {}, "2026-10-20T10:00:00+08:00"),
    ("2026-10-20T10:00:00+08:00", {"source_time_unknown": "true"}, None),
])
def test_focused_card_preserves_relative_time_without_fabricated_dates(env, observed_at, metadata, expected_source_time):
    memory, model = shared(env)
    captured = {}
    def complete(stage, payload):
        captured[stage] = copy.deepcopy(payload)
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": "张三计划下周六去滑雪",
                               "usage": "用于滑雪计划", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": "下周六我要去滑雪", "context": []}}]}
        card = payload["cards"][0]
        return {"subject": payload["subject"], "operations": [
            {"action": "add", "topic": "plan", "content": card["claim"],
             "status": "confirmed", "evidence": [card["id"]]}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="ski", text="下周六我要去滑雪",
                                   observed_at=observed_at, metadata=metadata)
    card = memory.store.get("card", result["events"][0]["card_ids"][0])
    event = memory.store.get("event", result["events"][0]["event_id"])
    assert card["claim_text"] == "张三计划下周六去滑雪"
    assert "planned_for" not in card
    assert card["valid_from"] is None and card["valid_to"] is None
    assert captured["focused_cards"]["observed_at"] == expected_source_time
    assert event["observed_at"] == expected_source_time
    if expected_source_time is None:
        assert card["event_time"] is None
    else:
        assert timestamp(card["event_time"]) == timestamp(expected_source_time)
    assert memory.sleep(actor())["jobs"][0]["status"] == "completed"
    state_card = captured["focused_state"]["cards"][0]
    assert state_card["claim"] == "张三计划下周六去滑雪"
    assert state_card["observed_at"] == card["event_time"]


def test_source_is_kept_without_verifying_against_current_message_or_context(env):
    memory, model = shared(env)
    def complete(stage, payload):
        result = extraction()
        result["cards"][0]["source"] = {"speaker": "nina", "quote": "a model paraphrase",
                                       "context": [{"id": "unknown", "speaker": "nina", "quote": "x"}]}
        return result
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="m1", text="Friday class")
    card = memory.store.get("card", result["events"][0]["card_ids"][0])
    assert card["source"] == {"speaker": "nina", "quote": "a model paraphrase",
                              "context": [{"id": "unknown", "speaker": "nina", "quote": "x"}]}
    assert card["source_quote"] == "a model paraphrase"
    event = memory.store.get("event", result["events"][0]["event_id"])
    assert event["text"] == "Friday class"


