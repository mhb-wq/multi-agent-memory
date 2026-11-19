import copy

import pytest
from test_immediate_person_matter import setup_memory
from test_multiuser import actor

from ours_memory.joint_validation import JointIds, JointValidationError, validate_joint
from ours_memory.models import ModelOutputError, OutputTruncated, stable_id

pytest_plugins = ["test_multiuser"]


def add(subject="marcus", card="card-new"):
    return {"subject": subject, "action": "add", "state_id": None, "topic": "plan",
            "content": "Friday", "status": "confirmed", "changed_part": "new", "evidence": [card]}


def payload():
    return {"new_card": {"id": "card-new", "claim_text": "Keep card-old literal"},
            "old_cards": [{"id": "card-old"}],
            "states": [{"id": "state-old", "subject": "marcus", "topic": "plan",
                        "evidence": ["card-old"]}],
            "subjects": ["marcus", "patel"], "participant_resolution": {"card-new": [], "card-old": []}}


def test_request_local_short_ids_are_reversible_and_do_not_change_text():
    original = payload()
    codec = JointIds(original)
    wire = codec.encode(original)
    assert wire["new_card"]["id"] == "C1" and wire["old_cards"][0]["id"] == "C2"
    assert wire["new_card"]["claim_text"] == "Keep card-old literal"
    assert wire["states"][0]["id"] == "S1" and wire["states"][0]["evidence"] == ["C2"]
    result = codec.decode({"relations": [{"old_card_id": "C2"}],
                           "operations": [{"state_id": "S1", "evidence": ["C1", "C2"],
                                           "content": "C1 remains text"}]})
    assert result["relations"][0]["old_card_id"] == "card-old"
    assert result["operations"][0]["state_id"] == "state-old"
    assert result["operations"][0]["evidence"] == ["card-new", "card-old"]
    assert result["operations"][0]["content"] == "C1 remains text"


@pytest.mark.parametrize("proposal,code", [
    ({"relations": [{"old_card_id": "card-old", "relation": "update"}], "operations": []}, "missing_fields"),
    ({"relations": [{"old_card_id": "missing", "relation": "update", "same_matter": "same", "changed_part": "x"}], "operations": []}, "old_card_not_candidate"),
    ({"relations": [{"old_card_id": "card-old", "relation": "update", "same_matter": "different", "changed_part": "x"}], "operations": []}, "update_matter_conflict"),
    ({"relations": [], "operations": [add("unknown")]}, "subject_not_candidate"),
    ({"relations": [], "operations": [{**add(), "action": "update", "state_id": "missing"}]}, "state_not_candidate"),
    ({"relations": [], "operations": [{**add("patel"), "action": "update", "state_id": "state-old"}]}, "state_subject_mismatch"),
    ({"relations": [], "operations": [{**add(), "evidence": []}]}, "missing_new_evidence"),
    ({"relations": [], "operations": [{"action": "uncertain"}]}, "state_uncertain"),
    ({"relations": [{"old_card_id": "card-old", "relation": "uncertain", "same_matter": "uncertain", "changed_part": ""}], "operations": []}, "relation_uncertain"),
])
def test_specific_validation_errors(proposal, code):
    with pytest.raises(JointValidationError) as error:
        validate_joint(proposal, payload(), batch=False)
    assert error.value.validation_code == code


def test_single_repair_is_recorded_and_maps_to_real_card_ids(env):
    memory, model = setup_memory(env)
    seen = []
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": request["text"], "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": request["text"]}}]}
        seen.append(copy.deepcopy(request))
        memory.store.call(stage, is_repair="validation_feedback" in request,
                          usage={"total_tokens": 17}, elapsed_seconds=1.25)
        return {"relations": [], "operations": [add("unknown" if len(seen) == 1 else "marcus", request["new_card"]["id"])]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="repair-one", text="Friday")
    assert result["status"] == "done" and len(seen) == 2
    assert seen[0]["new_card"]["id"] == "C1"
    assert seen[1]["validation_feedback"]["code"] == "subject_not_candidate"
    card_id = result["events"][0]["card_ids"][0]
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["evidence"] == [card_id]
    attempt = memory.store.all("joint_attempt")[-1]
    assert attempt["card_ids"] == [card_id]
    assert [a["status"] for a in attempt["attempts"]] == ["invalid", "valid"]
    assert attempt["attempts"][0]["issue"]["validation_code"] == "subject_not_candidate"
    calls = [x for x in memory.store.call_records() if x["stage"] == "focused_joint"]
    assert [x["is_repair"] for x in calls] == [False, True]
    assert calls[1]["usage"]["total_tokens"] == 17 and calls[1]["elapsed_seconds"] == 1.25


def test_unresolved_state_stays_pending_without_repair_pressure(env):
    memory, model = setup_memory(env)
    calls = []
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": request["text"], "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": request["text"]}}]}
        calls.append(request)
        return {"relations": [], "operations": [{"action": "uncertain"}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="uncertain-no-retry", text="unclear")
    assert result["status"] == "accepted_pending" and len(calls) == 1
    card_id = result["events"][0]["card_ids"][0]
    issue = memory.store.get("pending_state", card_id)["last_issues"][0]
    assert issue["validation_code"] == "state_uncertain"
    assert not memory.store.all("user_state")


def test_repair_stops_after_one_retry_and_preserves_both_errors(env):
    memory, model = setup_memory(env)
    seen = []
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": request["text"], "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": request["text"]}}]}
        seen.append(request)
        return {"relations": [], "operations": [add("unknown", request["new_card"]["id"])]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="repair-fails", text="Friday")
    assert result["status"] == "accepted_pending" and len(seen) == 2
    card_id = result["events"][0]["card_ids"][0]
    assert memory.store.get("pending_state", card_id)["last_issues"][0]["validation_code"] == "subject_not_candidate"
    attempt = memory.store.all("joint_attempt")[-1]
    assert [a["status"] for a in attempt["attempts"]] == ["invalid", "invalid"]
    assert not memory.store.all("user_state")


def test_unparseable_json_gets_one_format_repair(env):
    memory, model = setup_memory(env)
    seen = []
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": request["text"], "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": request["text"]}}]}
        seen.append(request)
        if len(seen) == 1:
            raise ModelOutputError("模型未返回可解析的JSON")
        return {"relations": [], "operations": [add(card=request["new_card"]["id"])]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="json-repair", text="Friday")
    assert result["status"] == "done" and len(seen) == 2
    assert seen[1]["validation_feedback"]["code"] == "invalid_json"


def test_uncertain_card_relation_remains_pending_without_model_retry(env):
    memory, model = setup_memory(env)
    joint_calls = []
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": request["text"], "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": request["text"]}}]}
        joint_calls.append(request)
        if request["old_cards"]:
            return {"relations": [{"old_card_id": request["old_cards"][0]["id"], "relation": "uncertain",
                                   "same_matter": "uncertain", "changed_part": ""}], "operations": []}
        return {"relations": [], "operations": [add(card=request["new_card"]["id"])]}
    model.complete = complete
    assert memory.ingest_message(actor(), message_id="rel-first", text="Monday")["status"] == "done"
    prior = copy.deepcopy(memory.store.all("user_state"))
    result = memory.ingest_message(actor(), message_id="rel-second", text="Friday")
    assert result["status"] == "accepted_pending" and len(joint_calls) == 2
    assert memory.store.all("user_state") == prior
    pending = memory.store.get("pending_state", result["events"][0]["card_ids"][0])
    assert pending["last_issues"][0]["validation_code"] == "relation_uncertain"


def test_failed_split_rolls_back_all_states_but_preserves_attempt_logs(env):
    memory, model = setup_memory(env)
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": text}} for text in ("first", "second")]}
        if stage == "focused_joint_batch":
            raise OutputTruncated("truncated")
        memory.store.call(stage, is_repair=False, usage={"total_tokens": 10}, elapsed_seconds=0.5)
        card = request["new_card"]
        if card["claim_text"] == "second":
            return {"relations": [], "operations": [{"action": "uncertain"}]}
        return {"relations": [], "operations": [add(card=card["id"])]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="batch-rollback", text="first second")
    assert result["status"] == "accepted_pending"
    assert not memory.store.all("user_state") and not memory.store.all("joint_state_history")
    assert len(memory.store.all("pending_state")) == 2
    recovery = memory.store.all("joint_batch_recovery")[-1]
    assert recovery["status"] == "rolled_back"
    assert [x["status"] for x in recovery["outcomes"]] == ["completed", "needs_attention"]
    assert sum(x["stage"] == "focused_joint" for x in memory.store.call_records()) == 2
    assert len(memory.store.all("joint_attempt")) == 3


def test_successful_split_commits_complete_batch_and_records_order(env):
    memory, model = setup_memory(env)
    def complete(stage, request):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": text}} for text in ("first", "second")]}
        if stage == "focused_joint_batch":
            raise OutputTruncated("truncated")
        return {"relations": [], "operations": [add(card=request["new_card"]["id"])]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="batch-success", text="first second")
    assert result["status"] == "done" and not memory.store.all("pending_state")
    recovery = memory.store.all("joint_batch_recovery")[-1]
    assert recovery["status"] == "committed"
    assert [x["card_id"] for x in recovery["outcomes"]] == recovery["card_ids"]
    assert all(x["status"] == "completed" for x in recovery["outcomes"])


def test_failed_split_restores_preexisting_state(env):
    memory, model = setup_memory(env)
    def complete(stage, request):
        if stage == "focused_cards":
            parts = ("old",) if request["text"] == "old" else ("first", "second")
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                               "source": {"speaker": "marcus", "quote": text}} for text in parts]}
        if stage == "focused_joint_batch":
            raise OutputTruncated("truncated")
        card = request["new_card"]
        if card["claim_text"] == "second":
            return {"relations": [], "operations": [{"action": "uncertain"}]}
        if card["claim_text"] == "old":
            return {"relations": [], "operations": [add(card=card["id"])]}
        state = request["states"][0]
        return {"relations": [], "operations": [{**add(card=card["id"]), "action": "update",
                "state_id": state["id"], "content": "changed"}]}
    model.complete = complete
    assert memory.ingest_message(actor(), message_id="preexisting", text="old")["status"] == "done"
    prior_state = copy.deepcopy(memory.store.all("user_state"))
    prior_history = copy.deepcopy(memory.store.all("joint_state_history"))
    result = memory.ingest_message(actor(), message_id="split-existing", text="first second")
    assert result["status"] == "accepted_pending"
    assert memory.store.all("user_state") == prior_state
    assert memory.store.all("joint_state_history") == prior_history
    assert memory.store.all("joint_batch_recovery")[-1]["status"] == "rolled_back"
