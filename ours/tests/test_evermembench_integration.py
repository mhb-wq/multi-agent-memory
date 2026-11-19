import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "experiments" / "evermembench" / (name + ".py")
    spec = importlib.util.spec_from_file_location("evermem_" + name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_preparation_works_with_official_loaders(tmp_path, monkeypatch):
    prep = load("prepare_project")
    result = prep.prepare("01", tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "experiments" / "evermembench" / "official" / "code"))
    from eval.src.core.loaders import load_groupchat_dataset
    from eval.src.core.qa_loader import load_qa
    dataset = load_groupchat_dataset(str(tmp_path / "dialogue_official_adapter.json"))
    qa = load_qa(str(tmp_path / "qa_official_adapter.json"))
    assert (dataset.total_messages, len(qa), result["speakers"]) == (10222, 488, 38)
    assert qa[0].question_id == "F_SH_Top01_001"


def test_same_group_context_uses_only_prior_messages():
    module = load("integration")
    M = module.SourceMessage
    messages = [M("1", "alice", "g1", "2025-01-01 01:00:00", "one"),
                M("2", "bob", "g2", "2025-01-01 01:01:00", "other"),
                M("3", "bob", "g1", "2025-01-01 01:02:00", "three")]
    rows = list(module.histories(messages))
    assert rows[0][1] == [] and rows[1][1] == []
    assert [x["id"] for x in rows[2][1]] == ["1"]
    baseline = module.baseline_message(messages[2])
    assert all(s in baseline.text for s in ("bob", "g1", "2025-01-01", "three"))


def test_ours_ingest_supplies_speaker_and_only_prior_same_group():
    from types import SimpleNamespace
    module = load("integration")
    M = module.SourceMessage
    messages = [M("1", "alice", "g1", "2025-01-01 01:00:00", "one"),
                M("2", "bob", "g1", "2025-01-01 01:01:00", "two")]
    calls, sleeps = [], []
    class Memory:
        def ingest_message(self, actor, **kwargs):
            calls.append((actor, kwargs))
            return {"status": "done", "extraction_status": "extracted"}
        def sleep(self, actor):
            sleeps.append(actor)
            return {"jobs": [{"status": "completed"}]}
    counts = module.ingest_ours(Memory(), SimpleNamespace(value=""), "space", messages, sleep_every=20)
    assert counts["messages"] == 2 and len(sleeps) == 1
    assert calls[1][0].principal_id == "bob"
    assert calls[1][1]["prior_context"][0]["speaker"] == "alice"
    assert "answer" not in calls[1][1] and "gold" not in calls[1][1]


def test_resume_reconstructs_history_and_preserves_progress():
    from types import SimpleNamespace
    module = load("integration")
    M = module.SourceMessage
    messages = [M("1", "alice", "g1", "2025-01-01 01:00:00", "one"),
                M("2", "bob", "g1", "2025-01-01 01:01:00", "two")]
    calls = []
    class Memory:
        def ingest_message(self, actor, **kwargs):
            calls.append(kwargs)
            return {"status": "done"}
        def sleep(self, actor):
            return {"jobs": [{"status": "completed"}]}
    counts = module.ingest_ours(Memory(), SimpleNamespace(value=""), "space", messages,
                                start_index=1, initial_counts={"messages": 1, "sleep_issues": 3})
    assert counts == {"messages": 2, "skipped_format": 0, "sleep_issues": 3}
    assert [p["id"] for p in calls[0]["prior_context"]] == ["1"]


def test_threshold_triggers_on_tenth_user_card_and_flushes_remainder():
    from types import SimpleNamespace
    module = load("integration")
    messages = [module.SourceMessage(str(i), "alice", "g", f"2025-01-01 01:00:{i:02}", "x") for i in range(11)]
    class Store:
        def __init__(self): self.cards = {}
        def all(self, kind): return [{"card_id": k} for k in self.cards]
        def get(self, kind, key): return self.cards.get(key)
    class Memory:
        def __init__(self): self.store, self.core, self.batches = Store(), SimpleNamespace(), []
        def ingest_message(self, actor, **kw):
            key = kw["message_id"]
            self.store.cards[key] = {"subject": actor.principal_id}
            return {"status": "done", "events": [{"card_ids": [key]}]}
        def sleep(self, actor):
            self.batches.append((self.core.focused_min_pending_cards, len(self.store.cards)))
            self.store.cards.clear()
            return {"jobs": [{"status": "completed"}]}
    memory = Memory()
    module.ingest_ours(memory, SimpleNamespace(value=""), "space", messages, state_card_threshold=10)
    assert memory.batches == [(10, 10), (1, 1)]


def test_amem_checkpoint_restores_notes_and_embeddings(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    module = load("run_amem_resilient")
    monkeypatch.syspath_prepend(str(ROOT / ".cache/baselines/agentic-memory-repro"))
    from memory_layer import MemoryNote
    note = MemoryNote(content="hello", id="id1", context="context", keywords=["hello"], tags=["tag"], category="x")
    retriever = SimpleNamespace(corpus=["hello"], document_ids={"hello": 0}, embeddings=np.array([[1.0, 2.0]]))
    system = SimpleNamespace(memories={"id1": note}, evo_cnt=3, retriever=retriever)
    path = tmp_path / "checkpoint.npz"
    module.checkpoint(path, system, 1)
    target = SimpleNamespace(memories={}, evo_cnt=0, retriever=SimpleNamespace(corpus=[], document_ids={}, embeddings=None))
    assert module.restore(path, target) == 1
    assert target.memories["id1"].content == "hello" and target.evo_cnt == 3
    np.testing.assert_array_equal(target.retriever.embeddings, [[1.0, 2.0]])
