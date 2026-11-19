import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ours_memory.baselines.base import Passage
from ours_memory.benchmark_inputs import gate_events, jsonl, perma_answer_context, perma_case

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def test_gate_interleaves_real_checkpoints_without_labels():
    root = ROOT / "gatemem/official/data/gatemem/data/medical"
    episode = jsonl(root / "episodes.jsonl")[0]
    checks = [c for c in jsonl(root / "checkpoints.jsonl") if c["episode_id"] == episode["episode_id"]]
    events = list(gate_events(episode, checks))
    observed = []
    for kind, value in events:
        if kind == "ingest":
            observed.append(value.id.rsplit("/", 1)[-1])
        else:
            assert observed[-1] == value["as_of_turn_id"]
            assert not {"expected_action", "query_type", "judge_spec", "leak_targets"} & value.keys()
    assert sum(kind == "query" for kind, _ in events) == len(checks)


def test_perma_real_case_hides_gold_and_uses_provided_history():
    root = ROOT / "perma/official/data"
    path = root / "tasks/user108/input_data_multi_c.json"
    data = json.loads(path.read_text(encoding="utf8"))
    event = data["overall"][0]
    meta_path = root / "evaluation/user108/meta/overall" / f"{event['task_id']}_{event['type']}.json"
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    messages, query = perma_case(event, meta, "user108")
    assert len(messages) == sum(len(s[0]) for s in event["context"])
    assert "gold_label" not in query and "preferences" not in query
    assert query["options"] == meta["options"]
    assert perma_answer_context({**event, "type": 1}, []) == "No memory found."
    with pytest.raises(ValueError):
        perma_case(event, {**meta, "task_type": 99}, "user108")


def test_official_gate_bridge_hides_gold_and_obeys_top_k():
    path = ROOT / "gatemem/adapter.py"
    spec = importlib.util.spec_from_file_location("gate_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from bench.eval.runner import checkpoint_from_raw, turn_from_raw

    class Memory:
        def __init__(self):
            self.turns = []

        def ingest(self, message):
            self.turns.append(message)

        def retrieve(self, query, asking_user_id, top_k):
            return [Passage(text=self.turns[-1].text, source_ids=(self.turns[-1].id,))]

    class Router:
        provider = "fake_test"

        def complete_result(self, *, system_prompt, user_prompt):
            assert "SECRET_TEST_GOLD" not in system_prompt + user_prompt
            return SimpleNamespace(text='{"action":"refuse","answer":"No"}', usage=None,
                                   latency_s=0, provider="fake_test", model="fake")

    gate = ROOT / "gatemem/official/data/gatemem/data/medical"
    episode = jsonl(gate / "episodes.jsonl")[0]
    check = next(c for c in jsonl(gate / "checkpoints.jsonl") if c["episode_id"] == episode["episode_id"])
    agent = module.GateBaselineBridge(lambda _: Memory(), llm_router=Router())
    agent.reset(episode)
    agent.ingest(turn_from_raw(episode["turns"][0]))
    checkpoint = checkpoint_from_raw({**check, "expected_action": "SECRET_TEST_GOLD"})
    assert agent.query(checkpoint)["action"] == "refuse"
    with pytest.raises(ValueError):
        module.GateBaselineBridge(lambda _: Memory(), llm_router=None)


def test_perma_bridge_uses_official_prompt_and_type_one_no_memory():
    spec = importlib.util.spec_from_file_location("perma_bridge_test", ROOT / "perma/adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeAdapter:
        def reset(self):
            self.messages = []

        def ingest_many(self, messages):
            self.messages.extend(messages)

        def retrieve(self, query, user, top_k):
            return [Passage(text="PRIVATE_MEMORY_MARKER")]

    root = ROOT / "perma/official/data"
    event = json.loads((root / "tasks/user108/input_data_multi_c.json").read_text(encoding="utf8"))["overall"][0]
    meta = json.loads((root / f"evaluation/user108/meta/overall/{event['task_id']}_{event['type']}.json").read_text(encoding="utf8"))

    def answer(prompt):
        assert "PRIVATE_MEMORY_MARKER" not in prompt
        assert "No memory found." in prompt
        assert meta["question"] in prompt
        return "A"

    assert int(event["type"]) == 1
    result = module.run_mcq(FakeAdapter(), event, meta, "user108", answer)
    assert result["answer"] == "A"
