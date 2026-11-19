# ruff: noqa: F811
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_multiuser import Clock, Decisions, NativeFixture, actor, env, ingest  # noqa: F401,F811

from ours_memory.config import Settings
from ours_memory.control import Actor
from ours_memory.control_intents import ControlIntentExecutor
from ours_memory.groupmem import control_config
from ours_memory.models import AccessDenied, ValidationError
from ours_memory.multiuser import MultiUserMemory
from ours_memory.store import Store


def grant_proposal(config):
    grant = copy.deepcopy(config["control"]["grants"][-1])
    grant.update(id="control-intent-grant", issuer="marcus", category="contact")
    return {"action": "grant", "operation_id": "1", "source_quote": "I authorize Nina",
            "grant": grant}


def test_verified_grant_and_revoke_are_idempotent(env):
    memory, _, _, config = env
    turn = {"turn_id": "t1", "speaker_id": "marcus", "text": "I authorize Nina to receive contact information."}
    proposal = grant_proposal(config)
    def execute(t, p):
        return memory.execute_control_intent(actor(), t, p, verifier=lambda turn, proposal: True)
    assert execute(turn, proposal)["status"] == "active"
    assert execute(turn, proposal)["status"] == "active"
    assert memory.control.permits(actor("nina", "clinical"), "marcus", "contact", "read")
    revoke = {"action": "revoke", "operation_id": "2", "source_quote": "I revoke Nina",
              "grant_id": "control-intent-grant"}
    assert execute({"turn_id": "t2", "speaker_id": "marcus", "text": "I revoke Nina's access."}, revoke)["status"] == "revoked"
    assert not memory.control.permits(actor("nina", "clinical"), "marcus", "contact", "read")


def test_unverified_and_forged_requests_never_execute(env):
    memory, _, _, config = env
    turn = {"turn_id": "t1", "speaker_id": "marcus", "text": "I authorize Nina."}
    proposal = grant_proposal(config)
    assert ControlIntentExecutor(memory, lambda t, p: False).execute(actor(), turn, proposal)["status"] == "needs_review"
    assert not memory.control.permits(actor("nina", "clinical"), "marcus", "contact", "read")
    executor = ControlIntentExecutor(memory, lambda t, p: True)
    with pytest.raises(AccessDenied):
        executor.execute(actor("nina"), turn, proposal)
    with pytest.raises(ValidationError):
        executor.execute(actor(), turn, {**proposal, "source_quote": "invented quote"})


def test_partial_delete_is_not_expanded_to_whole_event(env):
    memory, _, _, _ = env
    result = ingest(memory, "Contact is 123 and unrelated information", event_id="mixed")
    executor = ControlIntentExecutor(memory, lambda t, p: True)
    turn = {"turn_id": "t2", "speaker_id": "marcus", "text": "Delete the number 123."}
    proposal = {"action": "forget", "operation_id": "1", "source_quote": turn["text"],
                "event_ids": [result["event_id"]], "whole_events_verified": False}
    assert executor.execute(actor(), turn, proposal)["status"] == "needs_review"
    assert not memory.store.get("envelope_v2", result["event_id"])["blocked"]


def test_verified_whole_event_deletion_routes_to_native_forget(env):
    memory, _, _, _ = env
    result = ingest(memory, "123", event_id="delete-number")
    executor = ControlIntentExecutor(memory, lambda t, p: True)
    turn = {"turn_id": "t3", "speaker_id": "marcus", "text": "Please delete 123."}
    proposal = {"action": "forget", "operation_id": "1", "source_quote": turn["text"],
                "event_ids": [result["event_id"]], "whole_events_verified": True}
    assert executor.execute(actor(), turn, proposal)["status"] == "forgotten"
    assert memory.store.get("envelope_v2", result["event_id"])["deleted"]


def test_gate_bridge_stops_on_uncertain_controls(env):
    memory, _, _, _ = env
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gate_control_bridge", root / "experiments/gatemem/control_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episode_path = root / "experiments/gatemem/official/data/gatemem/data/medical/episodes.jsonl"
    episode = json.loads(episode_path.open(encoding="utf8").readline())
    raw = episode["turns"][4]
    assert "logistics" in raw["text"]
    # Use the official turn text; the fixture supplies an authenticated local ID.
    raw = {**raw, "speaker": {"principal_id": "marcus", "role": "patient"}}
    bridge = module.GateControlBridge(memory, lambda _: {"status": "uncertain", "operations": []},
        lambda t, p: True, agent_id="clinical", space_id="care", task_id="case")
    with pytest.raises(ValidationError, match="暂停回放"):
        bridge.process_turn(raw)


def test_external_answer_review_rejects_unverified_and_revoked_evidence(env):
    memory, model, _, _ = env
    ingest(memory, "arrival Friday", event_id="review")
    nina = actor("nina", "logistics")
    bundle = memory.read(nina, subject_ids=["marcus"], query="arrival time")
    assert memory.review_response(nina, bundle["bundle_id"], "arrival time", "Friday")["supported"]
    model.allow_output = False
    assert not memory.review_response(nina, bundle["bundle_id"], "arrival time", "secret")["supported"]
    memory.control.revoke(actor(), "nina-logistics")
    with pytest.raises(AccessDenied):
        memory.review_response(nina, bundle["bundle_id"], "arrival time", "Friday")


@pytest.mark.parametrize("space", ["GroupMemBench:Technology", "PERMA:user108"])
def test_common_benchmark_config_exposes_verified_control_and_self_delete(space):
    users = ["User_1", "User_2"]
    agents = ["ingestor", "maintainer", "answerer", "controller"]
    store = Store(":memory:")
    try:
        memory = MultiUserMemory(Settings(team_agents={space: agents}), NativeFixture(), Decisions(),
                                 store, control_config(users, space), Clock())
        owner = Actor("User_1", "controller", space, "task")
        other = Actor("User_2", "controller", space, "task")
        assert memory.control.permits(owner, "User_1", "conversation", "delete")
        assert not memory.control.permits(other, "User_1", "conversation", "delete")
        grant = {"id": "extra", "issuer": "User_1", "grantee": "User_2", "subject": "User_1",
                 "space": space, "category": "conversation", "actions": ["read"],
                 "valid_from": None, "valid_to": None}
        proposal = {"action": "grant", "operation_id": "1", "source_quote": "I authorize User_2", "grant": grant}
        turn = {"turn_id": "turn1", "speaker_id": "User_1", "text": "I authorize User_2."}
        assert memory.execute_control_intent(owner, turn, proposal, verifier=lambda t, p: True)["status"] == "active"
        with pytest.raises(ValidationError):
            memory.execute_control_intent(owner, turn, proposal, verifier=None)
    finally:
        store.close()
