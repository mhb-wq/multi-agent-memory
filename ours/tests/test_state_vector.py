import pytest

from ours_memory.models import stable_id
from ours_memory.state_vector import rank_state_entries
from ours_memory.store import Store


class Embeddings:
    identity = "model-v1"

    def __init__(self):
        self.calls = []
        self.fail = False

    def state_embedding_identity(self, space):
        return self.identity

    def embed_state_texts(self, space, texts):
        self.calls.append((space, list(texts)))
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0, 0.0] if any(word in text.lower() for word in ("car", "automobile"))
                else [0.0, 1.0] for text in texts]


def put_state(store, space, subject, rows):
    store.put("user_state", stable_id("user_state", space, subject),
              {"subject": subject, "topics": {"plan": rows}})


def test_semantic_rank_cache_update_remove_and_model_change(tmp_path):
    store = Store(str(tmp_path / "state.sqlite"))
    backend = Embeddings()
    put_state(store, "team", "alice", [{"id": "s1", "content": "car delivery Monday", "evidence": ["c1"]}])
    first = rank_state_entries(store, backend, "team", {"alice"}, "automobile", 5)
    assert [entry["id"] for _, entry in first] == ["s1"]
    assert len(backend.calls) == 2
    rank_state_entries(store, backend, "team", {"alice"}, "automobile", 5)
    assert len(backend.calls) == 3  # query only; stored state vector reused
    put_state(store, "team", "alice", [{"id": "s1", "content": "train delivery Friday", "evidence": ["c2"]}])
    assert rank_state_entries(store, backend, "team", {"alice"}, "automobile", 5) == []
    assert len(backend.calls) == 5
    backend.identity = "model-v2"
    rank_state_entries(store, backend, "team", {"alice"}, "train", 5)
    assert len(backend.calls) == 7
    put_state(store, "team", "alice", [])
    assert rank_state_entries(store, backend, "team", {"alice"}, "train", 5) == []
    assert store.all("state_vector") == []
    store.close()


def test_embedding_failure_does_not_replace_cached_index_or_silently_fallback(tmp_path):
    store = Store(str(tmp_path / "state.sqlite"))
    backend = Embeddings()
    put_state(store, "team", "alice", [{"id": "s1", "content": "car", "evidence": ["c1"]}])
    rank_state_entries(store, backend, "team", {"alice"}, "car", 5)
    previous = store.all("state_vector")
    put_state(store, "team", "alice", [{"id": "s1", "content": "train", "evidence": ["c2"]}])
    backend.fail = True
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        rank_state_entries(store, backend, "team", {"alice"}, "train", 5)
    assert store.all("state_vector") == previous
    backend.fail = False
    assert rank_state_entries(store, backend, "team", {"alice"}, "train", 5)[0][1]["id"] == "s1"
    store.close()


def test_index_isolated_by_space_and_eligible_entries(tmp_path):
    store = Store(str(tmp_path / "state.sqlite"))
    backend = Embeddings()
    put_state(store, "a", "alice", [{"id": "s1", "content": "car", "evidence": ["c1"]}])
    put_state(store, "b", "alice", [{"id": "s1", "content": "train", "evidence": ["c2"]}])
    assert rank_state_entries(store, backend, "a", {"alice"}, "car", 5, allowed_ids=set()) == []
    assert rank_state_entries(store, backend, "b", {"alice"}, "car", 5) == []
    assert rank_state_entries(store, backend, "a", {"alice"}, "car", 5)[0][1]["id"] == "s1"
    store.close()
