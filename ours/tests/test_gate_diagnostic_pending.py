import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "experiments/_diag_20261027/run_gatemem.py"
    spec = importlib.util.spec_from_file_location("gate_diagnostic_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_identity_candidates_exclude_relationship_project_ids(runner):
    episode = {"entities": {"principals": [{"principal_id": "alice"}, {"principal_id": "agent_helper"}],
                            "relationships": [{"principal_id": "alice", "project_id": "project_example"}]}}
    principals, subjects, relations = runner.episode_principals(episode, access_mode="shared_pool")
    assert principals == ["alice", "agent_helper"]
    assert set(subjects) == {"alice", "agent_helper"}
    assert relations[0]["project_id"] == "project_example"


def test_ambiguous_control_is_recorded_and_later_turn_is_ingested(runner):
    saved, messages = [], []
    memory = SimpleNamespace(store=SimpleNamespace(put=lambda *args: saved.append(args)),
                             ingest_message=lambda *args, **kwargs: messages.append(kwargs) or {"status": "done"})
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.memory, agent.space, agent.subjects = memory, "space", ["alice"]
    agent.clock, agent.log, agent.pending_controls = runner.Clock(), [], []
    agent.message_count, agent.prior_messages = 0, []
    agent.access_mode = "governed"
    agent.undated, agent.last_day, agent.last_moment, agent.unslept = True, None, None, False
    agent._control = lambda *args: (_ for _ in ()).throw(runner.PendingControl("unknown", "revoke", "bob"))
    control = SimpleNamespace(timestamp=None, turn_id="1", speaker_principal_id="alice",
                              speaker_role="user", text="revoke Bob's access")
    ordinary = SimpleNamespace(timestamp=None, turn_id="2", speaker_principal_id="alice",
                               speaker_role="user", text="We meet next Monday")
    agent.ingest(control)
    agent.ingest(ordinary)
    assert saved[0][0] == "pending_control_v2"
    assert agent.pending_controls[0]["status"] == "needs_review"
    assert messages[0]["observed_at"] is None
    assert messages[0]["metadata"]["source_time_unknown"] == "true"


def test_pending_deletion_does_not_globally_block_unrelated_query(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.access_mode = "governed"
    agent.undated, agent.unslept = False, False
    agent.space, agent.subjects, agent.clock = "space", ["alice"], runner.Clock()
    agent.clock.value = "2026-01-01T00:00:00+00:00"
    agent.pending_controls = [{"action": "forget", "issuer": "alice", "status": "needs_review"}]
    agent.memory = SimpleNamespace(control=SimpleNamespace(categories=["conversation"], permits=lambda *a: True),
                                   read=lambda *a, **kw: {"cards": [], "bundle_id": "b"})
    output = agent.query(SimpleNamespace(asker_principal_id="alice", query_text="unrelated topic"))
    assert output["action"] == "no_memory"


def test_replay_context_is_past_bounded_and_sleep_every_twenty(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.prior_messages, agent.message_count = [], 0
    agent.access_mode = "governed"
    seen, sleeps = [], []
    agent._ingest_turn = lambda turn: seen.append([x["id"] for x in agent.prior_messages])
    agent._sleep = lambda label: sleeps.append(label)
    for i in range(1, 42):
        agent.ingest(SimpleNamespace(timestamp=None, turn_id=str(i), speaker_principal_id="a",
                                     speaker_role="user", text=f"message {i}"))
    assert seen[0] == []
    assert seen[20] == [str(i) for i in range(11, 21)]
    assert sleeps == ["after-20-messages", "after-40-messages"]
    assert len(agent.prior_messages) == 10


def test_context_does_not_exceed_6000_characters(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.prior_messages, agent.message_count = [], 0
    agent.access_mode = "governed"
    agent._ingest_turn = lambda turn: None
    agent._sleep = lambda label: None
    agent.ingest(SimpleNamespace(timestamp=None, turn_id="1", speaker_principal_id="a",
                                 speaker_role="user", text="x" * 7000))
    assert sum(len(x["text"]) for x in agent.prior_messages) <= 6000


def test_scaffold_only_bootstraps_explicit_bounded_scope(runner):
    rels = [
        {"principal_id": "nina", "patient_id": "marcus", "access_scope": "scheduling_only"},
        {"principal_id": "doctor", "patient_id": "marcus", "type": "assigned_clinician"},
        {"principal_id": "other", "patient_id": "marcus", "access_scope": "scheduling_only_initially"},
    ]
    grants = runner.scaffold_read_grants(rels, {"nina", "doctor", "other", "marcus"}, "care")
    assert len(grants) == 1
    assert grants[0]["grantee"] == "nina" and grants[0]["subject"] == "marcus"
    assert grants[0]["category"] == "schedule" and grants[0]["actions"] == ["read"]


def test_exact_authorization_creates_only_verified_logistics_grant(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.space, agent.control_events, agent.log = "care", 0, []
    agent.episode = {"entities": {"principals": [{"principal_id": "nina", "display_name": "Nina"}]}}
    captured = []
    def execute(actor, turn, proposal, *, verifier):
        assert verifier(turn, proposal)
        captured.append(proposal["grant"])
        return {"status": "active"}
    agent.memory = SimpleNamespace(execute_control_intent=execute)
    turn = SimpleNamespace(turn_id="t1", speaker_principal_id="marcus", text="I authorize Nina to read my schedule.")
    assert agent._control(turn, "2026-01-01T00:00:00+00:00", "task")
    assert captured[0]["subject"] == "marcus"
    assert captured[0]["grantee"] == "nina"
    assert captured[0]["category"] == "schedule" and captured[0]["actions"] == ["read"]


def test_pending_revocation_does_not_block_other_known_people(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.episode = {"entities": {"principals": [{"principal_id": "nina", "display_name": "Nina"},
                                                    {"principal_id": "doctor", "display_name": "Dr Patel"}]}}
    pending = {"action": "revoke", "issuer": "marcus", "target": "Nina"}
    assert agent._pending_revoke_applies(pending, "nina", "marcus")
    assert not agent._pending_revoke_applies(pending, "doctor", "marcus")
    assert not agent._pending_revoke_applies(pending, "marcus", "marcus")
    pending["target"] = "unclear"
    assert agent._pending_revoke_applies(pending, "doctor", "marcus")


def test_shared_mode_control_text_is_ingested_as_normal_message(runner):
    calls = []
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.memory = SimpleNamespace(ingest_message=lambda *a, **kw: calls.append(kw) or {"status": "done"})
    agent.space, agent.subjects, agent.clock, agent.log = "space", ["alice"], runner.Clock(), []
    agent.undated, agent.last_day, agent.last_moment, agent.unslept = True, None, None, False
    agent.access_mode, agent.prior_messages, agent.message_count = "shared_pool", [], 0
    agent._control = lambda *a: (_ for _ in ()).throw(AssertionError("must not execute control"))
    turn = SimpleNamespace(timestamp=None, turn_id="1", speaker_principal_id="alice",
                           speaker_role="user", text="Please delete the old note.")
    agent.ingest(turn)
    assert calls[0]["text"] == turn.text and calls[0]["prior_context"] == []


def test_shared_mode_records_skipped_format_and_continues(runner):
    calls = []
    def ingest_message(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "done", "extraction_status": "skipped_format", "extraction_attempts": 2}
        return {"status": "done", "extraction_status": "extracted", "extraction_attempts": 1}
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.memory = SimpleNamespace(ingest_message=ingest_message)
    agent.space, agent.subjects, agent.clock, agent.log = "space", ["alice"], runner.Clock(), []
    agent.undated, agent.last_day, agent.last_moment, agent.unslept = True, None, None, False
    agent.access_mode, agent.prior_messages, agent.message_count = "shared_pool", [], 0
    for turn_id in ("1", "2"):
        agent.ingest(SimpleNamespace(timestamp=None, turn_id=turn_id, speaker_principal_id="alice",
                                     speaker_role="user", text="message " + turn_id))
    assert len(calls) == 2 and agent.message_count == 2
    assert agent.log == [{"stage": "ingest", "turn": "1", "status": "skipped_format", "attempts": 2}]


def test_shared_mode_empty_memory_uses_official_answer_path(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.access_mode, agent.undated, agent.unslept = "shared_pool", False, False
    agent.space, agent.subjects, agent.pending_controls = "space", ["alice"], []
    agent.clock = runner.Clock()
    agent.clock.value = "2026-01-01T00:00:00+00:00"
    agent._base = SimpleNamespace(_format_retrieved_memory_block=lambda cards: str(cards))
    agent.memory = SimpleNamespace(control=SimpleNamespace(categories=["conversation"], permits=lambda *a: True),
                                   read=lambda *a, **kw: {"cards": [], "bundle_id": "b"},
                                   validate_bundle=lambda *a: {"ready": True})
    calls = []
    agent._run_llm = lambda **kw: calls.append(kw) or {"action": "no_memory", "answer": "none"}
    checkpoint = SimpleNamespace(asker_principal_id="alice", query_text="question")
    assert agent.query(checkpoint)["action"] == "no_memory"
    assert calls[0]["retrieved_memory"] == [] and calls[0]["checkpoint"] is checkpoint


def test_shared_gate_answer_receives_state_and_extraction_usage(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.access_mode, agent.unslept = "shared_pool", False
    agent.space, agent.subjects, agent.pending_controls = "space", ["alice"], []
    agent.clock = runner.Clock()
    agent.clock.value = "2026-01-01T00:00:00+00:00"
    agent._base = SimpleNamespace(_format_retrieved_memory_block=lambda cards: str(cards))
    bundle = {"bundle_id": "b", "cards": [{"id": "m1", "claim_text": "class Friday",
              "usage": "use for schedule", "author_id": "alice"}],
              "user_states": [{"subject": "alice", "topics": {"plan": [{"content": "class Friday"}]}}],
              "pending_subjects": ["alice"]}
    agent.memory = SimpleNamespace(control=SimpleNamespace(categories=["conversation"], permits=lambda *a: True),
                                   read=lambda *a, **kw: bundle,
                                   validate_bundle=lambda *a: {"ready": True})
    calls = []
    agent._run_llm = lambda **kw: calls.append(kw) or {"action": "answer"}
    agent.query(SimpleNamespace(asker_principal_id="alice", query_text="When?"))
    assert "use for schedule" in calls[0]["retrieved_memory"][0]["text"]
    assert '"subject": "alice"' in calls[0]["rendered_memory_block_override"]
    assert '"pending_subjects": ["alice"]' in calls[0]["rendered_memory_block_override"]


def test_sleep_partial_is_logged_without_interrupting_replay(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.space, agent.subjects, agent.log, agent.unslept = "space", ["alice"], [], True
    agent.memory = SimpleNamespace(sleep=lambda actor: {"jobs": [
        {"status": "needs_attention", "result": {"issues": [{"code": "ValidationError"}]}}
    ]})
    agent._sleep("after-20-messages")
    assert agent.log[0]["status"] == "needs_attention"
    assert agent.log[0]["jobs"][0]["result"]["issues"][0]["code"] == "ValidationError"
    assert agent.unslept is False


def test_sleep_system_failure_still_surfaces(runner):
    agent = runner.OursGateAgent.__new__(runner.OursGateAgent)
    agent.space, agent.subjects, agent.log = "space", ["alice"], []
    agent.memory = SimpleNamespace(sleep=lambda actor: {"jobs": [{"status": "failed"}]})
    with pytest.raises(RuntimeError, match="Sleep"):
        agent._sleep("after-20-messages")
