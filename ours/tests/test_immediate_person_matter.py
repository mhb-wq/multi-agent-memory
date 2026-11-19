import copy

from test_multiuser import actor

from ours_memory.models import stable_id
from ours_memory.multiuser import MultiUserMemory
from ours_memory.participants import resolve_participants

pytest_plugins = ["current_fixture"]


def setup_memory(env, aliases=None):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    control["person_aliases"] = aliases or {}
    return MultiUserMemory(original.settings, original.backend, model, original.store, control, clock), model


def extracted(text, participants):
    return {"cards": [{"subject": None, "claim": text, "usage": "", "state_topic": [],
                       "matter": "registration interface", "participants": participants,
                       "source": {"speaker": "patel", "quote": text, "context": []}}]}


def test_unambiguous_participant_becomes_candidate_and_updates_immediately(env):
    memory, model = setup_memory(env, {"marcus": ["小李"]})
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        if stage == "focused_cards":
            return extracted(payload["text"], [{"name": "小李", "role": "owner"}])
        person = payload["participant_resolution"][payload["new_card"]["id"]][0]
        assert person["resolved_id"] == "marcus" and person["role"] == "owner"
        assert payload["subjects"] == ["marcus"]
        return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                "topic": "plan", "content": "Marcus owns interface", "status": "confirmed",
                "changed_part": "new responsibility", "evidence": [payload["new_card"]["id"]]}]}
    model.complete = complete
    result = memory.ingest_message(actor("patel"), message_id="person", text="小李负责接口")
    assert result["status"] == "done" and result["events"][0]["state_update"] == "completed"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["content"] == "Marcus owns interface"
    assert calls == ["focused_cards", "focused_joint"]
    assert memory.store.all("pending_state") == []
    memory.ingest_message(actor("patel"), message_id="person", text="小李负责接口")
    assert calls == ["focused_cards", "focused_joint"]


def test_ambiguous_alias_is_retained_without_committing_to_user(env):
    memory, model = setup_memory(env, {"marcus": ["小李"], "patel": ["小李"]})
    def complete(stage, payload):
        if stage == "focused_cards":
            return extracted(payload["text"], [{"name": "小李", "role": "owner"}])
        person = payload["participant_resolution"][payload["new_card"]["id"]][0]
        assert person["status"] == "ambiguous" and person["resolved_id"] is None
        assert person["candidate_ids"] == ["marcus", "patel"]
        assert payload["subjects"] == []
        return {"relations": [], "operations": []}
    model.complete = complete
    result = memory.ingest_message(actor("patel"), message_id="ambiguous", text="小李负责接口")
    decision = memory.store.get("joint_card_decision", stable_id(
        "joint_card_decision", "care", result["events"][0]["card_ids"][0]))
    assert decision["participant_resolution"]
    assert not memory.store.all("user_state")


def test_failed_immediate_update_is_retryable_without_duplicate_extraction(env):
    memory, model = setup_memory(env)
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": payload["text"], "usage": "",
                    "state_topic": ["plan"], "source": {"speaker": "marcus", "quote": payload["text"]}}]}
        if calls.count("focused_joint") == 1:
            raise RuntimeError("unavailable")
        return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                "topic": "plan", "content": "Friday", "status": "confirmed", "changed_part": "new",
                "evidence": [payload["new_card"]["id"]]}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="retry-immediate", text="Friday")
    assert result["status"] == "accepted_pending"
    card_id = result["events"][0]["card_ids"][0]
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"
    result = memory.ingest_message(actor(), message_id="retry-immediate", text="Friday")
    assert result["status"] == "done" and not memory.store.get("pending_state", card_id)
    assert calls == ["focused_cards", "focused_joint", "focused_joint"]


def test_participant_resolution_requires_exact_alias():
    card = {"participants": [{"name": "Lee", "role": "participant"}, {"name": "M", "role": "owner"}]}
    result = resolve_participants(card, {"marcus", "patel"}, {"marcus": ["Lee"], "patel": ["Lee"]})
    assert result[0]["status"] == "ambiguous"
    assert result[1]["status"] == "unresolved"


def test_multiple_cards_are_maintained_in_order_before_ingest_returns(env):
    memory, model = setup_memory(env)
    calls = []
    memory.backend.embed_state_texts = lambda space, texts: [[1.0, 0.0] for _ in texts]
    def complete(stage, payload):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                    "source": {"speaker": "marcus", "quote": text}} for text in ("task Monday", "task Friday")]}
        assert stage == "focused_joint_batch"
        cards = payload["new_cards"]
        calls.append([c["claim_text"] for c in cards])
        ids = [c["id"] for c in cards]
        return {"cards": [{"card_id": c["id"], "relations": []} for c in cards],
                "operations": [{"subject": "marcus", "action": "add",
                "state_id": None, "topic": "plan", "content": "task Friday",
                "status": "confirmed", "changed_part": "plan",
                "source_card_ids": ids, "evidence": ids}]}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="two-cards", text="task Monday then task Friday")
    assert result["status"] == "done" and calls == [["task Monday", "task Friday"]]
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["content"] == "task Friday"
    assert set(state["topics"]["plan"][0]["evidence"]) == {r["card_ids"][0] for r in result["events"]}
    assert not memory.store.all("pending_state")


def test_matter_is_used_to_retrieve_candidates_and_inconsistent_relation_is_rejected(env):
    memory, model = setup_memory(env)
    queries = []
    search = memory.backend.search_shared
    def search_with_trace(query, *args):
        queries.append(query)
        return search(query, *args)
    memory.backend.search_shared = search_with_trace
    def complete(stage, payload):
        if stage == "focused_cards":
            return extracted(payload["text"], [{"name": "marcus", "role": "owner"}])
        if payload["old_cards"]:
            return {"relations": [{"old_card_id": payload["old_cards"][0]["id"], "same_matter": "different",
                                   "relation": "update", "changed_part": "deadline"}], "operations": []}
        return {"relations": [], "operations": []}
    model.complete = complete
    first = memory.ingest_message(actor(), message_id="matter-first", text="first notice")
    assert first["status"] == "done" and "registration interface" in queries[0]
    second = memory.ingest_message(actor(), message_id="matter-second", text="second notice")
    assert second["status"] == "accepted_pending"
    card_id = second["events"][0]["card_ids"][0]
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"


def test_update_without_same_matter_is_not_accepted_as_confirmed(env):
    memory, model = setup_memory(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            return extracted(payload["text"], [{"name": "marcus", "role": "owner"}])
        if payload["old_cards"]:
            return {"relations": [{"old_card_id": payload["old_cards"][0]["id"],
                                   "relation": "update", "changed_part": "deadline"}],
                    "operations": []}
        return {"relations": [], "operations": []}
    model.complete = complete
    assert memory.ingest_message(actor(), message_id="missing-matter-first", text="first notice")["status"] == "done"
    result = memory.ingest_message(actor(), message_id="missing-matter-second", text="second notice")
    assert result["status"] == "accepted_pending"
    card_id = result["events"][0]["card_ids"][0]
    assert memory.store.get("pending_state", card_id)["status"] == "needs_attention"


def test_failed_batch_commits_no_state_and_retry_resumes_without_reextracting(env):
    memory, model = setup_memory(env)
    attempts = []
    def complete(stage, payload):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                    "source": {"speaker": "marcus", "quote": text}} for text in ("first", "second")]}
        assert stage == "focused_joint_batch"
        attempts.append([card["claim_text"] for card in payload["new_cards"]])
        if len(attempts) == 1:
            raise RuntimeError("offline")
        return {"cards": [{"card_id": c["id"], "relations": []} for c in payload["new_cards"]], "operations": []}
    model.complete = complete
    result = memory.ingest_message(actor(), message_id="ordered-retry", text="first second")
    assert result["status"] == "accepted_pending"
    assert attempts == [["first", "second"]] and len(memory.store.all("card")) == 2
    assert len(memory.store.all("pending_state")) == 2 and not memory.store.all("user_state")
    result = memory.ingest_message(actor(), message_id="ordered-retry", text="first second")
    assert result["status"] == "done" and attempts == [["first", "second"], ["first", "second"]]
    assert len(memory.store.all("card")) == 2
    assert not memory.store.all("pending_state")


def test_batch_rejects_competing_updates_without_partial_commit(env):
    memory, model = setup_memory(env)
    memory.backend.embed_state_texts = lambda space, texts: [[1.0, 0.0] for _ in texts]
    def complete(stage, payload):
        if stage == "focused_cards":
            texts = ("initial",) if payload["text"] == "initial" else ("first", "second")
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                    "source": {"speaker": "marcus", "quote": text}} for text in texts]}
        if stage == "focused_joint":
            return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                    "topic": "plan", "content": "initial", "status": "confirmed", "changed_part": "new",
                    "evidence": [payload["new_card"]["id"]]}]}
        assert stage == "focused_joint_batch"
        old = payload["states"][0]
        cards = payload["new_cards"]
        operations = [{"subject": "marcus", "action": "update", "state_id": old["id"], "topic": "plan",
                       "content": card["claim_text"], "status": "confirmed", "changed_part": "change",
                       "source_card_ids": [card["id"]], "evidence": old["evidence"] + [card["id"]]} for card in cards]
        return {"cards": [{"card_id": card["id"], "relations": []} for card in cards], "operations": operations}
    model.complete = complete
    assert memory.ingest_message(actor(), message_id="original", text="initial")["status"] == "done"
    result = memory.ingest_message(actor(), message_id="competing", text="first second")
    assert result["status"] == "accepted_pending"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["content"] == "initial"
    assert len(memory.store.all("pending_state")) == 2


def test_batch_combines_two_cards_into_one_existing_state_update(env):
    memory, model = setup_memory(env)
    memory.backend.embed_state_texts = lambda space, texts: [[1.0, 0.0] for _ in texts]
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        if stage == "focused_cards":
            texts = ("initial",) if payload["text"] == "initial" else ("Friday", "room 2")
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": ["plan"],
                    "source": {"speaker": "marcus", "quote": text}} for text in texts]}
        if stage == "focused_joint":
            return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                    "topic": "plan", "content": "initial", "status": "confirmed", "changed_part": "new",
                    "evidence": [payload["new_card"]["id"]]}]}
        assert stage == "focused_joint_batch"
        old = payload["states"][0]
        cards = payload["new_cards"]
        ids = [c["id"] for c in cards]
        return {"cards": [{"card_id": c["id"], "relations": []} for c in cards],
                "operations": [{"subject": "marcus", "action": "update", "state_id": old["id"], "topic": "plan",
                "content": "Friday, room 2", "status": "confirmed", "changed_part": "time and room",
                "source_card_ids": ids, "evidence": ids}]}
    model.complete = complete
    original = memory.ingest_message(actor(), message_id="original-for-batch", text="initial")
    result = memory.ingest_message(actor(), message_id="batch-update", text="Friday room 2")
    assert result["status"] == "done"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))["topics"]["plan"]
    assert len(state) == 1 and state[0]["content"] == "Friday, room 2"
    assert {r["card_ids"][0] for r in result["events"]} <= set(state[0]["evidence"])
    assert original["events"][0]["card_ids"][0] in state[0]["evidence"]
    assert calls.count("focused_joint_batch") == 1


def test_prior_evidence_merge_filters_missing_deleted_and_other_space(env):
    memory, _ = setup_memory(env)
    caller = memory._internal(actor(), "marcus")
    memory.store.put("card", "valid", {"id": "valid", "team_id": "care"})
    memory.store.put("card", "deleted", {"id": "deleted", "team_id": "care", "deleted": True})
    memory.store.put("card", "blocked", {"id": "blocked", "team_id": "care", "blocked": True})
    memory.store.put("card", "other", {"id": "other", "team_id": "elsewhere"})
    assert memory.core._merge_state_evidence(["valid", "missing", "deleted", "blocked", "other"],
                                             ["valid"], caller) == ["valid"]


def test_index_sync_failure_after_commit_retries_index_without_duplicate_state(env):
    memory, model = setup_memory(env)
    calls = []
    def complete(stage, payload):
        calls.append(stage)
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": "Friday", "usage": "", "state_topic": ["plan"],
                    "source": {"speaker": "marcus", "quote": "Friday"}}]}
        return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                "topic": "plan", "content": "Friday", "status": "confirmed", "changed_part": "new",
                "evidence": [payload["new_card"]["id"]]}]}
    model.complete = complete
    original_embed = memory.backend.embed_state_texts
    def fail(space, texts):
        raise RuntimeError("index unavailable")
    memory.backend.embed_state_texts = fail
    result = memory.ingest_message(actor(), message_id="sync-retry", text="Friday")
    assert result["status"] == "accepted_pending"
    assert memory.store.get("state_vector_issue", "care")
    assert len(memory.store.get("user_state", stable_id("user_state", "care", "marcus"))["topics"]["plan"]) == 1
    memory.backend.embed_state_texts = original_embed
    result = memory.ingest_message(actor(), message_id="sync-retry", text="Friday")
    assert result["status"] == "done" and not memory.store.get("state_vector_issue", "care")
    assert calls == ["focused_cards", "focused_joint"]
    assert len(memory.store.get("user_state", stable_id("user_state", "care", "marcus"))["topics"]["plan"]) == 1


def test_later_card_can_use_prior_unresolved_mention_as_evidence(env):
    memory, model = setup_memory(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            first = payload["text"] == "小李负责接口"
            return {"cards": [{"subject": None if first else "marcus", "claim": payload["text"], "usage": "",
                    "state_topic": [] if first else ["plan"], "matter": "接口",
                    "participants": [{"name": "小李" if first else "marcus", "role": "owner"}],
                    "source": {"speaker": "marcus", "quote": payload["text"]}}]}
        if payload["new_card"]["claim_text"] == "小李负责接口":
            return {"relations": [], "operations": []}
        assert payload["old_cards"]
        old = payload["old_cards"][0]
        assert payload["participant_resolution"][old["id"]][0]["status"] == "unresolved"
        return {"relations": [{"old_card_id": old["id"], "same_matter": "same",
                               "relation": "supplement", "changed_part": "identity"}],
                "operations": [{"subject": "marcus", "action": "add", "state_id": None, "topic": "plan",
                                "content": "marcus 负责接口", "status": "confirmed", "changed_part": "identity",
                                "evidence": [old["id"], payload["new_card"]["id"]]}]}
    model.complete = complete
    memory.ingest_message(actor(), message_id="unresolved-first", text="小李负责接口")
    assert not memory.store.all("user_state")
    result = memory.ingest_message(actor(), message_id="resolved-later", text="marcus 即小李负责接口")
    assert result["status"] == "done"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["plan"][0]["content"] == "marcus 负责接口"
    assert len(state["topics"]["plan"][0]["evidence"]) == 2


def test_uncertain_state_does_not_block_later_message_or_overwrite_it(env):
    memory, model = setup_memory(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": payload["text"], "usage": "",
                    "state_topic": ["preference"], "source": {"speaker": "marcus", "quote": payload["text"]}}]}
        card = payload["new_card"]
        if card["claim_text"] == "May I share details?":
            return {"relations": [], "operations": [{"action": "uncertain"}]}
        return {"relations": [], "operations": [{"subject": "marcus", "action": "add", "state_id": None,
                "topic": "preference", "content": "Do not share", "status": "confirmed", "changed_part": "new",
                "evidence": [card["id"]]}]}
    model.complete = complete
    first = memory.ingest_message(actor(), message_id="first-uncertain", text="May I share details?")
    assert first["status"] == "accepted_pending" and first["processing_status"] == "state_maintenance_pending"
    card_id = first["events"][0]["card_ids"][0]
    assert memory.store.get("pending_state", card_id)
    assert not memory.store.all("user_state")
    second = memory.ingest_message(actor(), message_id="next-certain", text="Do not share")
    assert second["status"] == "done"
    state = memory.store.get("user_state", stable_id("user_state", "care", "marcus"))
    assert state["topics"]["preference"][0]["content"] == "Do not share"
    retry = memory.ingest_message(actor(), message_id="first-uncertain", text="May I share details?")
    assert retry["processing_status"] == "ordered_replay_required"
    assert memory.store.get("pending_state", card_id)
    assert memory.store.get("user_state", stable_id("user_state", "care", "marcus")) == state


def test_extraction_failure_keeps_source_and_later_messages_continue(env):
    memory, model = setup_memory(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            if payload["text"] == "temporary failure":
                raise RuntimeError("model offline")
            return {"cards": [{"subject": "marcus", "claim": payload["text"], "usage": "",
                    "state_topic": [], "source": {"speaker": "marcus", "quote": payload["text"]}}]}
        return {"relations": [], "operations": []}
    model.complete = complete
    first = memory.ingest_message(actor(), message_id="failed-extraction", text="temporary failure")
    assert first["status"] == "accepted_pending" and first["processing_status"] == "extraction_pending"
    assert memory.store.get("source_message_v2", first["message_id"])["text"] == "temporary failure"
    assert memory.store.get("deferred_message_v2", first["message_id"])
    second = memory.ingest_message(actor(), message_id="later", text="new message")
    assert second["status"] == "done"
    retry = memory.ingest_message(actor(), message_id="failed-extraction", text="temporary failure")
    assert retry["processing_status"] == "ordered_replay_required"


def test_one_card_storage_failure_does_not_stop_other_cards(env):
    memory, model = setup_memory(env)
    def complete(stage, payload):
        if stage == "focused_cards":
            return {"cards": [{"subject": "marcus", "claim": text, "usage": "", "state_topic": [],
                    "source": {"speaker": "marcus", "quote": text}} for text in ("first", "second")]}
        return {"cards": [{"card_id": c["id"], "relations": []} for c in payload["new_cards"]], "operations": []}
    model.complete = complete
    original_ingest = memory.ingest
    def flaky(*args, **kwargs):
        if kwargs["event_id"].endswith(":0"):
            raise RuntimeError("first card failed")
        return original_ingest(*args, **kwargs)
    memory.ingest = flaky
    result = memory.ingest_message(actor(), message_id="partial-cards", text="first second")
    assert result["status"] == "accepted_pending" and result["processing_status"] == "card_storage_pending"
    assert len(result["events"]) == 2 and result["events"][1]["status"] == "done"
    assert memory.store.get("deferred_message_v2", result["message_id"])
