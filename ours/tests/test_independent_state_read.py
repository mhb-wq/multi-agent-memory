import copy

from test_multiuser import actor

from ours_memory.models import stable_id
from ours_memory.multiuser import MultiUserMemory

pytest_plugins = ["test_multiuser"]


def test_independent_read_recovers_supporting_card_outside_card_search(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    original.settings.independent_state_read_enabled = True
    memory = MultiUserMemory(original.settings, original.backend, model, original.store, control, clock)
    model.complete = lambda stage, payload: {
        "cards": [{"subject": "marcus", "claim": "Marcus plans Friday class", "usage": "",
                   "state_topic": ["plan"], "source": {"speaker": "marcus", "quote": "Friday class", "context": []}}]
    }
    result = memory.ingest_message(actor(), message_id="state-evidence", text="Friday class")
    card_id = result["events"][0]["card_ids"][0]
    key = stable_id("user_state", "care", "marcus")
    memory.store.put("user_state", key, {"subject": "marcus", "topics": {
        "plan": [{"id": "s1", "content": "Marcus plans Friday class", "status": "confirmed",
                  "evidence": [card_id]}]}})
    memory.backend.search_shared = lambda *args: []
    bundle = memory.read(actor(), subject_ids=["marcus"], query="When is Friday class?")
    assert [c["id"] for c in bundle["cards"]] == [card_id]
    assert bundle["user_states"][0]["topics"]["plan"][0]["id"] == "s1"
    past = memory.read(actor(), subject_ids=["marcus"], query="Friday class", target_time="2026-10-24T10:00:00+08:00")
    assert past["user_states"] == []


def test_independent_read_uses_vectors_without_word_overlap(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    original.settings.independent_state_read_enabled = True
    memory = MultiUserMemory(original.settings, original.backend, model, original.store, control, clock)
    model.complete = lambda stage, payload: {
        "cards": [{"subject": "marcus", "claim": "car delivery", "usage": "",
                   "state_topic": ["plan"], "source": {"speaker": "marcus", "quote": "car delivery", "context": []}}]
    }
    result = memory.ingest_message(actor(), message_id="synonym", text="car delivery")
    card_id = result["events"][0]["card_ids"][0]
    key = stable_id("user_state", "care", "marcus")
    memory.store.put("user_state", key, {"subject": "marcus", "topics": {"plan": [
        {"id": "s1", "content": "car delivery", "status": "confirmed", "evidence": [card_id]}]}})
    memory.backend.search_shared = lambda *args: []
    memory.backend.embed_state_texts = lambda space, texts: [[1.0, 0.0] for _ in texts]
    bundle = memory.read(actor(), subject_ids=["marcus"], query="automobile")
    assert bundle["user_states"][0]["topics"]["plan"][0]["id"] == "s1"


def test_independent_read_never_exposes_missing_evidence(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    original.settings.independent_state_read_enabled = True
    memory = MultiUserMemory(original.settings, original.backend, model, original.store, control, clock)
    key = stable_id("user_state", "care", "marcus")
    memory.store.put("user_state", key, {"subject": "marcus", "topics": {
        "plan": [{"id": "s1", "content": "Friday class", "status": "confirmed", "evidence": ["missing"]}]}})
    assert memory.read(actor(), subject_ids=["marcus"], query="Friday class")["user_states"] == []
