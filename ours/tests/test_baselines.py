import hashlib
import json
from pathlib import Path

from ours_memory.baselines import (
    Message,
    NaiveRAGAdapter,
    Passage,
    format_passage,
    load_messages,
    qa_user_prompt,
)
from ours_memory.baselines.amem import AMEMAdapter
from ours_memory.baselines.mem0 import Mem0Adapter
from ours_memory.baselines.memos import MemOSAdapter, MemOSNativeAdapter

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "experiments/groupmembench/baseline.lock.json"


def test_lock_pins_official_sources_and_licenses():
    lock = json.loads(LOCK.read_text("utf8"))
    assert lock["selection"]["lower_bound"] == "naive_rag"
    assert lock["selection"]["memory_systems"] == ["mem0", "a-mem", "memos"]
    assert "mirix" in lock["selection"]["deferred"]
    assert lock["systems"]["mem0"]["license"] == "Apache-2.0"
    assert lock["systems"]["a-mem"]["license"] == "MIT"
    assert lock["systems"]["memos"]["license"] == "Apache-2.0"
    assert len(lock["systems"]["mem0"]["commit"]) == 40
    assert len(lock["systems"]["a-mem"]["commit"]) == 40
    assert len(lock["systems"]["memos"]["commit"]) == 40


def test_official_naive_rag_files_match_lock():
    lock = json.loads(LOCK.read_text("utf8"))
    base = ROOT / "experiments/groupmembench/official"
    for relative, expected in lock["systems"]["naive_rag"]["file_sha256"].items():
        actual = hashlib.sha256((base / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def message(
    identifier="Technology/A/M1", text="alpha budget 5000", author="User_1", timestamp="2025-01-01T10:00:00"
):
    return Message(
        id=identifier,
        text=text,
        author=author,
        timestamp=timestamp,
        channel="A",
        role="PM",
        topic="budget",
    )


def test_load_messages_preserves_metadata_and_order(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "A": [
                    {
                        "msg_node": "M2",
                        "content": "later",
                        "author": "U2",
                        "timestamp": "2025-01-02T10:00:00",
                    },
                    {
                        "msg_node": "M1",
                        "content": "earlier",
                        "author": "U1",
                        "timestamp": "2025-01-01T10:00:00",
                    },
                ],
                "B": [
                    {
                        "msg_node": "M3",
                        "content": "middle",
                        "author": "U3",
                        "timestamp": "2025-01-01T12:00:00",
                    }
                ],
            }
        ),
        encoding="utf8",
    )
    rows = load_messages(corpus, "Technology")
    assert [row.text for row in rows] == ["earlier", "middle", "later"]
    assert rows[0].id == "Technology/A/M1"
    assert rows[0].metadata["channel"] == "A"


def test_naive_rag_adapter_returns_ranked_source_messages():
    class FakeEncoder:
        def encode(self, texts, **kwargs):
            # Two-dimensional fixture: budget -> [1, 0], otherwise [0, 1].
            return [[1.0, 0.0] if "budget" in text.lower() else [0.0, 1.0] for text in texts]

    adapter = NaiveRAGAdapter(FakeEncoder())
    adapter.ingest_many(
        [
            message("Technology/A/M1", "the budget is 5000 dollars"),
            message("Technology/A/M2", "the launch is on Tuesday"),
        ]
    )
    results = adapter.retrieve("budget", "User_1", top_k=1)
    assert results[0].text == "the budget is 5000 dollars"
    assert results[0].source_ids == ("Technology/A/M1",)
    assert "user=User_1" in format_passage(results[0])


def test_mem0_adapter_uses_official_add_search_contract():
    class FakeMem0:
        def __init__(self):
            self.added, self.searched = [], []

        def add(self, messages, **kwargs):
            self.added.append((messages, kwargs))

        def search(self, query, **kwargs):
            self.searched.append((query, kwargs))
            return {"results": [{"id": "m1", "memory": "budget 5000", "metadata": {"author": "User_1"}}]}

        def reset(self):
            pass

    fake = FakeMem0()
    adapter = Mem0Adapter(fake, native_users=True)
    adapter.ingest(message())
    assert fake.added[0][1]["user_id"] == "User_1"
    assert fake.added[0][1]["agent_id"] == "group"
    results = adapter.retrieve("budget", "User_1")
    assert results[0].text == "budget 5000"
    assert fake.searched[0][1]["filters"] == {"agent_id": "group"}


def test_amem_adapter_uses_add_note_and_search_agentic():
    class FakeAMEM:
        def __init__(self):
            self.notes, self.queries = [], []

        def add_note(self, content, **kwargs):
            self.notes.append((content, kwargs))

        def search_agentic(self, query, k):
            self.queries.append((query, k))
            return [{"id": "n1", "content": "budget 5000"}]

    fake = FakeAMEM()
    adapter = AMEMAdapter(fake)
    adapter.ingest(message())
    assert fake.notes[0][0] == "alpha budget 5000"
    assert fake.notes[0][1]["time"] == "202501011000"
    results = adapter.retrieve("budget", "User_1", top_k=5)
    assert results[0].text == "budget 5000"
    assert fake.queries[0][0] == "User_1 budget"


def test_amem_adapter_supports_paper_reproduction_api():
    class Note:
        def __init__(self, identifier, content):
            self.id = identifier
            self.content = content
            self.timestamp = "202501011000"
            self.context = "budget"
            self.tags = ["A"]

    class FakePaperAMEM:
        def __init__(self):
            self.memories = {"n1": Note("n1", "budget 5000")}

        def add_note(self, content, **kwargs):
            pass

        def find_related_memories(self, query, k):
            return "serialized", [0]

        def find_related_memories_raw(self, query, k):
            return "talk start time:202501011000 memory content: budget 5000"

    adapter = AMEMAdapter(FakePaperAMEM())
    results = adapter.retrieve("budget", "User_1", top_k=5)
    assert "budget 5000" in results[0].text
    assert results[0].metadata["format"] == "raw"


def test_memos_adapter_uses_mos_add_search_contract():
    class FakeMOS:
        def __init__(self):
            self.added, self.searched = [], []

        def add(self, messages, **kwargs):
            self.added.append((messages, kwargs))

        def search(self, query, **kwargs):
            self.searched.append((query, kwargs))
            return {"text_mem": [{"memories": [{"id": "x", "memory": "budget 5000", "metadata": {}}]}]}

    fake = FakeMOS()
    adapter = MemOSAdapter(fake)
    adapter.ingest(message())
    assert fake.added[0][1]["user_id"] == "group"
    results = adapter.retrieve("budget", "User_1")
    assert results[0].text == "budget 5000"


def test_memos_native_adapter_uses_extract_add_search():
    class Item:
        id = "m1"
        memory = "budget 5000"
        metadata = {}

    class FakeText:
        def __init__(self):
            self.added = []

        def extract(self, messages):
            return [{"memory": messages[0]["content"]}]

        def add(self, items):
            self.added.append(items)

        def search(self, query, top_k):
            return [Item()]

        def delete_all(self):
            pass

    fake = FakeText()
    adapter = MemOSNativeAdapter(fake)
    adapter.ingest(message())
    assert fake.added
    assert adapter.retrieve("budget", "User_1")[0].text == "budget 5000"


def test_qa_prompt_contains_asker_but_not_gold():
    prompt = qa_user_prompt(
        {"question": "What is the budget?", "answer": "gold-only-secret", "asking_user_id": "User_1"},
        [Passage("budget 5000", metadata={"author": "User_2", "channel": "A"})],
    )
    assert "Asking user: User_1" in prompt
    assert "user=User_2" in prompt
    assert "gold-only-secret" not in prompt
