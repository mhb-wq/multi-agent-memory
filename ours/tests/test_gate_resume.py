import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def resume(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/_diag_20261027"
    monkeypatch.syspath_prepend(str(directory))
    path = directory / "resume_gatemem_episode.py"
    spec = importlib.util.spec_from_file_location("resume_gatemem_episode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_requires_exact_contiguous_message_prefix(resume):
    turns = [{"turn_id": "t1"}, {"turn_id": "t2"}, {"turn_id": "t3"}]
    assert resume.reconstruct_prefix(turns, {"t1", "t2"}) == 1
    with pytest.raises(ValueError):
        resume.reconstruct_prefix(turns, {"t1", "t3"})
    with pytest.raises(ValueError):
        resume.reconstruct_prefix(turns, {"t1", "foreign"})


def test_resume_uses_only_missing_questions_in_original_order(resume):
    checkpoints = [{"checkpoint_id": "q1", "as_of_turn_id": "t1"},
                   {"checkpoint_id": "q2", "as_of_turn_id": "t2"},
                   {"checkpoint_id": "q3", "as_of_turn_id": "t3"}]
    pos = {"t1": 0, "t2": 1, "t3": 2}
    remaining = resume.check_answer_prefix(checkpoints, pos, [{"checkpoint_id": "q1"}],
                                           [{"checkpoint_id": "q2"}])
    assert [r["checkpoint_id"] for r in remaining] == ["q3"]
    with pytest.raises(ValueError):
        resume.check_answer_prefix(checkpoints, pos, [{"checkpoint_id": "q1"}], [{"checkpoint_id": "q1"}])
    with pytest.raises(ValueError):
        resume.check_answer_prefix(checkpoints, pos, [{"checkpoint_id": "q2"}], [])


def test_copy_creates_independent_state_and_never_overwrites_it(resume, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "ours-run").mkdir(parents=True)
    (source / "ours-run" / "state.sqlite3").write_bytes(b"source-snapshot")
    copied = resume.copy_memory_once(source, target)
    (copied / "state.sqlite3").write_bytes(b"continued-work")
    assert (source / "ours-run" / "state.sqlite3").read_bytes() == b"source-snapshot"
    resume.copy_memory_once(source, target)
    assert (copied / "state.sqlite3").read_bytes() == b"continued-work"
