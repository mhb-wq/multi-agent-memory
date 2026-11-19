import copy

from test_multiuser import actor

from ours_memory.models import stable_id
from ours_memory.multiuser import MultiUserMemory

pytest_plugins = ["current_fixture"]


def test_public_card_can_update_related_user_state_with_history(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    memory = MultiUserMemory(original.settings, original.backend, model, original.store, control, clock)

    def complete(stage, payload):
        if stage == "focused_cards":
            initial = "Monday" in payload["text"]
            return {"cards": [{"subject": "marcus" if initial else None, "claim": payload["text"],
                               "usage": "", "state_topic": ["plan"] if initial else [],
                               "source": {"speaker": "marcus", "quote": payload["text"], "context": []},
                               "matter": "registration interface", "participants": []}]}
        assert stage == "focused_joint"
        card = payload["new_card"]
        if not payload["states"]:
            return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                    "topic": "plan", "content": card["claim_text"], "status": "confirmed", "changed_part": "initial plan",
                    "evidence": [card["id"]]}]}
        old = payload["states"][0]
        assert old["subject"] == "marcus"
        assert payload["old_cards"]
        return {"relations": [{"old_card_id": payload["old_cards"][0]["id"], "relation": "update", "same_matter": "same",
                               "changed_part": "Monday to Friday"}],
                "operations": [{"subject": "marcus", "action": "update", "state_id": old["id"],
                                "topic": "plan", "content": "Marcus registration interface Friday", "status": "confirmed",
                                "changed_part": "Monday to Friday", "evidence": [card["id"]]}]}

    model.complete = complete
    first = memory.ingest_message(actor(), message_id="old", text="Marcus registration interface Monday")
    second = memory.ingest_message(actor(), message_id="new", text="registration interface Friday")
    public_card = memory.store.get("card", second["events"][0]["card_ids"][0])
    assert public_card["subject"] is None
    assert public_card["matter"] == "registration interface"
    key = stable_id("user_state", "care", "marcus")
    state = memory.store.get("user_state", key)
    assert state["topics"]["plan"][0]["content"] == "Marcus registration interface Friday"
    assert set(state["topics"]["plan"][0]["evidence"]) == {
        first["events"][0]["card_ids"][0], second["events"][0]["card_ids"][0]}
    assert len(memory.store.all("joint_state_history")) == 2
    assert len(memory.store.all("card")) == 2


def test_joint_rejects_non_candidate_subject_without_losing_card(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    memory = MultiUserMemory(original.settings, original.backend, model, original.store, control, clock)

    def complete(stage, payload):
        if stage == "focused_cards":
            return {"cards": [{"subject": None, "claim": "Public project deadline Friday", "usage": "",
                    "state_topic": [], "source": {"speaker": "marcus", "quote": "deadline Friday", "context": []}}]}
        return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                "topic": "plan", "content": "Marcus deadline Friday", "status": "confirmed",
                "changed_part": "new", "evidence": [payload["new_card"]["id"]]}]}

    model.complete = complete
    result = memory.ingest_message(actor(), message_id="unattributed", text="deadline Friday")
    card_id = result["events"][0]["card_ids"][0]
    assert result["status"] == "accepted_pending"
    assert memory.store.get("card", card_id)
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"
    assert memory.store.get("user_state", stable_id("user_state", "care", "marcus")) is None


def test_joint_retrieves_old_state_semantically_when_card_search_misses(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    memory = MultiUserMemory(original.settings, original.backend, model, original.store, control, clock)
    memory.backend.embed_state_texts = lambda space, texts: [[1.0, 0.0] for _ in texts]

    def complete(stage, payload):
        if stage == "focused_cards":
            first = payload["text"] == "car"
            return {"cards": [{"subject": "marcus" if first else None, "claim": payload["text"],
                    "usage": "", "state_topic": ["plan"] if first else [],
                    "source": {"speaker": "marcus", "quote": payload["text"], "context": []}}]}
        card = payload["new_card"]
        if card["claim_text"] == "car":
            op = {"subject": "marcus", "action": "add", "state_id": None, "topic": "plan",
                  "content": "car", "status": "confirmed", "changed_part": "new", "evidence": [card["id"]]}
        else:
            assert not payload["old_cards"]
            assert payload["states"][0]["content"] == "car"
            old = payload["states"][0]
            op = {"subject": "marcus", "action": "update", "state_id": old["id"], "topic": "plan",
                  "content": "automobile", "status": "confirmed", "changed_part": "change",
                  "evidence": old["evidence"] + [card["id"]]}
        return {"relations": [], "operations": [op]}

    model.complete = complete
    memory.ingest_message(actor(), message_id="semantic-old", text="car")
    memory.backend.search_shared = lambda *args: []
    memory.ingest_message(actor(), message_id="semantic-new", text="automobile")
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["content"] == "automobile"
