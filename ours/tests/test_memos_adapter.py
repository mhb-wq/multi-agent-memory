from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ours_memory import Caller, Event
from ours_memory.memos_backend import MemOSBackend
from ours_memory.store import Store


class Item:
    def model_dump(self):
        return {"id": "random-upstream-id", "memory": "budget 5000", "metadata": {}}


def test_native_failure_retries_exact_extracted_batch_without_reextract(tmp_path):
    store = Store(str(tmp_path / "state.sqlite3"))
    memory = Mock()
    memory.extract.return_value = [Item()]
    memory.add.side_effect = [RuntimeError("temporary Qdrant failure"), None]
    backend = MemOSBackend(
        {"directory": str(tmp_path / "qdrant")},
        store,
        cube_factory=lambda policy, user: SimpleNamespace(text_mem=memory),
    )
    event = Event("e", "u", "t", "a", "trip", "budget 5000", "2026-10-20T00:00:00Z", ("a",))
    with pytest.raises(RuntimeError):
        backend.add(event, "policy")
    first_batch = memory.add.call_args.args[0]
    result = backend.add(event, "policy")
    assert memory.extract.call_count == 1
    assert memory.add.call_args.args[0] == first_batch
    assert result[0].memory_id == first_batch[0]["id"]
    backend.add(event, "policy")
    assert memory.add.call_count == 2
    memory.delete.side_effect = None
    backend.forget(result[0].memory_id, "policy", "u")
    assert backend.find_event_memories("e", "policy", "u") == []
    assert not store.all("native_event")
    store.close()


def test_search_only_opens_provided_authorized_cubes(tmp_path):
    seen = []
    memory = Mock()
    memory.search.return_value = []

    def factory(policy, user):
        seen.append((policy, user))
        return SimpleNamespace(text_mem=memory)

    store = Store(":memory:")
    backend = MemOSBackend({"directory": str(tmp_path)}, store, factory)
    backend.search("budget", Caller("u", "t", "a"), ["shared"], 5)
    assert seen == [("shared", "u")]
    store.close()
