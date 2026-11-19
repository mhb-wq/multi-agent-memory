import json
from pathlib import Path

import pytest
from test_multiuser import Clock, Decisions, NativeFixture

from ours_memory.config import Settings
from ours_memory.multiuser import MultiUserMemory
from ours_memory.store import Store


@pytest.fixture
def env(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/multiuser.json").read_text(encoding="utf8"))
    settings = Settings(**config["method"])
    store = Store(str(tmp_path / "db.sqlite3"))
    model, backend, clock = Decisions(), NativeFixture(), Clock()
    memory = MultiUserMemory(settings, backend, model, store, config["control"], clock)
    yield memory, model, clock, config
    store.close()
