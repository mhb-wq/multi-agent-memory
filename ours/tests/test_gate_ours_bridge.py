import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "experiments/gatemem/ours_bridge.py"
    spec = importlib.util.spec_from_file_location("gate_ours_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ours_bridge_stops_before_ingest_if_control_uncertain(gate):
    class Control:
        def process_turn(self, raw):
            raise gate.ValidationError("ambiguous control")

    class Memory:
        def ingest_message(self, *args, **kwargs):
            raise AssertionError("ambiguous control was written as an ordinary message")

    clock = SimpleNamespace(value="")
    agent = gate.GateOursAgent(lambda episode: (Memory(), clock), lambda memory, episode: Control(),
                               llm_router=SimpleNamespace(provider="fake"), space_id="care", maintainer_id="owner")
    agent.reset({"episode_id": "case", "entities": {}})
    turn = SimpleNamespace(timestamp="2026-03-03T08:12", turn_id="t005", speaker_principal_id="owner",
                           speaker_role="patient", text="Only logistics; not medications.")
    with pytest.raises(gate.ValidationError, match="ambiguous control"):
        agent.ingest(turn)


def test_gate_clock_requires_timestamp(gate):
    assert gate.replay_time("2026-03-03T08:12") == "2026-03-03T08:12:00+00:00"
    with pytest.raises(gate.ValidationError):
        gate.replay_time(None)


def test_ours_bridge_returns_answer_without_llm_review(gate):
    class Permission:
        spaces = {"care": {"members": ["owner", "other"]}}
        categories = {"conversation"}

        def authenticate(self, who):
            assert who.principal_id == "owner"

        def permits(self, who, subject, category, action):
            return subject == "owner"

    class Memory:
        control = Permission()

        def read(self, who, *, subject_ids, **kwargs):
            assert subject_ids == ["owner"]
            return {"bundle_id": "b", "cards": [{"id": "c1", "claim_text": "Friday"}]}

        def validate_bundle(self, *args):
            return {"ready": True}

        def review_response(self, who, bundle_id, question, response, *, card_ids):
            raise AssertionError("LLM output review must not run")

    class Router:
        provider = "fake"

        def complete_result(self, **kwargs):
            return SimpleNamespace(text='{"action":"answer","answer":"Friday"}', usage=None,
                                   latency_s=0, provider="fake", model="fake")

    clock = SimpleNamespace(value="2026-03-03T08:12:00+00:00")
    agent = gate.GateOursAgent(lambda episode: (Memory(), clock), lambda memory, episode: None,
                               llm_router=Router(), space_id="care", maintainer_id="owner")
    agent.reset({"episode_id": "case", "entities": {}})
    from bench.agents.base import Checkpoint

    checkpoint = Checkpoint(checkpoint_id="q", episode_id="case", as_of_turn_id="t1",
                            asker_principal_id="owner", asker_role="patient", query_type="utility",
                            query_text="When?", expected_action="answer")
    result = agent.query(checkpoint)
    assert result["action"] == "answer"
    assert "Friday" in result["answer"]
