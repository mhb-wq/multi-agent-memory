import copy
import json

import pytest
from test_multiuser import Decisions, NativeFixture, actor, env, ingest, read  # noqa: F401

from ours_memory import groupmem as g
from ours_memory.config import Settings
from ours_memory.models import ValidationError
from ours_memory.multiuser import MultiUserMemory
from ours_memory.store import Store


def test_asker_reaches_search_select_answer_and_review(env):  # noqa: F811
    m, model, _, _ = env
    ingest(m)
    queries = []
    original = m.backend.search

    def search(query, *args):
        queries.append(query)
        return original(query, *args)

    m.backend.search = search
    bundle = read(m, actor("nina", "logistics"))
    m.answer(actor("nina", "logistics"), bundle["bundle_id"], "my appointment?")
    assert queries[0].startswith("nina ")
    for stage in ("select", "answer"):
        assert [p for s, p in model.calls if s == stage][-1]["asking_user_id"] == "nina"


def test_unresolved_subject_keeps_evidence_without_personal_state(env):  # noqa: F811
    m, model, _, _ = env
    model.fragments = []
    result = m.ingest_message(
        actor(),
        message_id="unresolved",
        text="8000",
        fallback_category="logistics",
        metadata={"channel": "project-a", "role": "Engineer"},
    )
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["state_key"] is None and card["state_eligible"] is False
    assert card["metadata"]["channel"] == "project-a"
    m.sleep(actor())
    assert not m.store.states()
    assert read(m)["cards"][0]["source_quote"] == "8000"


def test_recognized_subject_is_not_author_and_provenance_survives(env):  # noqa: F811
    m, model, _, _ = env
    model.fragments = [{"subject_id": "marcus", "category": "logistics", "quote": "8000"}]
    result = m.ingest_message(
        actor("patel"), message_id="report", text="8000", metadata={"reply_to": "Msg_2"}
    )
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] == "marcus" and card["author_id"] == "patel"
    assert read(m)["cards"][0]["metadata"]["reply_to"] == "Msg_2"


def test_different_project_states_do_not_overwrite(env):  # noqa: F811
    m, _, _, _ = env
    ingest(m, text="8000", item_id="one", metadata={"channel": "A"})
    ingest(m, text="5000", event_id="second", item_id="two", metadata={"channel": "B"})
    m.sleep(actor())
    assert len(m.store.states()) == 2


def test_recent_context_is_prior_channel_bounded_and_control_cleared():
    rows = [{"id": str(i), "author": "u", "text": f"message {i}", "time": None,
             "metadata": {"channel": "a", "role": "user"}} for i in range(15)]
    rows[12]["text"] = "Please delete that contact"
    rows[13]["metadata"]["channel"] = "b"
    context = g.recent_source_context(rows, 15 - 1)
    assert [x["id"] for x in context] == []
    rows[12]["text"] = "ordinary message"
    context = g.recent_source_context(rows, 14)
    assert all(x["id"] != "13" for x in context)
    assert all(int(x["id"]) < 14 for x in context)
    assert len(context) <= 10 and sum(len(x["text"]) for x in context) <= 6000


def test_current_diagnostic_allows_provider_reasoning(experiment):
    run, _, _, _, _, _ = experiment
    spec = g.read_json(run / "protocol.json")
    spec.update(reasoning_policy="provider_default", reasoning_mode="provider default")
    g.validate_protocol(spec)
    spec["method"] = {"joint_state_update_enabled": "true"}
    with pytest.raises(ValidationError):
        g.validate_protocol(spec)


def test_focused_successful_zero_card_message_is_not_a_failure(experiment, monkeypatch):
    run, cfg, memory, clock, trace, _ = experiment
    memory.core.focused_schema = True

    def empty_message(*args, **kwargs):
        memory.store.put("source_message_v2", kwargs["message_id"], {"text": kwargs["text"]})
        return {"status": "done", "events": [], "extraction_status": "extracted"}

    monkeypatch.setattr(memory, "ingest_message", empty_message)
    result = g.ingest_run(run, cfg, memory, clock, trace)
    assert result["ready"] and result["next_index"] == 3
    assert len(memory.store.all("source_message_v2")) == 3


def test_group_recording_does_not_grant_cross_subject_confirmation():
    config = g.control_config(["alice", "bob"], "group")
    grant = next(x for x in config["grants"] if x["grantee"] == "alice" and x["subject"] == "bob")
    assert "write" in grant["actions"]
    assert "confirm" not in grant["actions"] and "delete" not in grant["actions"]


class IntakeDecisions(Decisions):
    def complete(self, stage, payload):
        if stage == "intake":
            self.calls.append((stage, copy.deepcopy(payload)))
            return {
                "fragments": [{"subject_id": "User_2", "category": "conversation", "quote": payload["text"]}]
            }
        return super().complete(stage, payload)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    official = assets / "official"
    (official / "questions/Technology").mkdir(parents=True)
    (official / "qa.txt").write_text("fixture QA", "utf8")
    (official / "judge.txt").write_text("fixture judge", "utf8")
    q = [
        {
            "id": "q1",
            "question": "What is my budget?",
            "answer": "gold-only-secret",
            "asking_user_id": "User_1",
        },
        {"id": "q2", "question": "What changed?", "answer": "5000", "asking_user_id": "User_2"},
    ]
    (official / "questions/Technology/knowledge_update.jsonl").write_text(
        "\n".join(json.dumps(x) for x in q), "utf8"
    )
    protocol = {
        "benchmark": "GroupMemBench",
        "current_parameter_profile": "test",
        "domain": "Technology",
        "memory_model": "deepseek-flash",
        "embedding": {"model": "fixture", "dimensions": 4, "execution": "local"},
        "reasoning_mode": "disabled explicitly",
        "qa": {
            "model": "test",
            "system_prompt": "official/qa.txt",
            "temperature": 0.2,
            "max_tokens": 512,
            "include_asking_user_id": True,
            "gold_visible": False,
        },
        "judge": {
            "model": "test",
            "system_prompt": "official/judge.txt",
            "temperature": 1.0,
            "max_tokens": 2048,
            "input_fields": ["question", "gold_answer", "agent_final_answer"],
            "feedback_to_qa": False,
        },
        "retrieval": {
            "candidate_limit": 20,
            "include_asking_user_id": True,
            "question_type_visible": False,
        },
        "sleep": {"boundary": "UTC data day end", "final_maintenance": True, "qa_writes_memory": False},
    }
    g.save(assets / "protocol.json", protocol)
    g.save(
        assets / "manifest.json",
        {
            "splits": {
                "development": [{"domain": "Technology", "type": "knowledge_update", "id": "q1"}],
                "holdout": [{"domain": "Technology", "type": "knowledge_update", "id": "q2"}],
            }
        },
    )
    corpus = {
        "A": [
            {
                "msg_node": "Msg_1",
                "author": "User_1",
                "content": "8000",
                "timestamp": "2025-01-01T10:00:00",
                "topic": "budget",
            },
            {
                "msg_node": "Msg_2",
                "author": "User_2",
                "content": "5000",
                "timestamp": "2025-01-02T10:00:00",
                "topic": "budget",
            },
        ],
        "B": [
            {
                "msg_node": "Msg_1",
                "author": "User_2",
                "content": "other evidence",
                "timestamp": "2025-01-01T11:00:00",
                "topic": "handoff",
            }
        ],
    }
    g.save(assets / "corpus.json", corpus)
    monkeypatch.setattr(g, "EXPERIMENT", assets)
    monkeypatch.setattr(g, "OFFICIAL", official)
    run = tmp_path / "run"
    cfg = g.prepare(run, assets / "corpus.json", assets / "protocol.json", assets / "manifest.json")
    store = Store(str(run / "state.sqlite3"))
    model, backend = IntakeDecisions(), NativeFixture()
    clock, trace = g.ReplayClock(cfg["final_time"]), g.Trace(run / "diagnostics")
    settings = Settings(
        team_agents={cfg["space"]: ["ingestor", "maintainer", "answerer", "controller"]},
        state_keys={"budget": {"type": "number", "unit": "CNY"}},
    )
    memory = MultiUserMemory(
        settings, backend, model, store, g.control_config(cfg["users"], cfg["space"]), clock
    )
    memory.observer = trace
    yield run, cfg, memory, clock, trace, model
    store.close()


def test_daily_replay_resume_and_fixed_snapshot_qa(experiment):
    run, cfg, m, clock, trace, model = experiment
    result = g.ingest_run(run, cfg, m, clock, trace)
    assert result["completed_days"] == ["2025-01-01", "2025-01-02"]
    assert result["next_index"] == 3 and result["ready"]
    # Fewer than twenty messages must not trigger day-end or query-time Sleep.
    assert not m.store.all("sleep_v2")
    calls = len(model.calls)
    g.ingest_run(run, cfg, m, clock, trace)
    assert len(model.calls) == calls
    before = g.snapshot(m)
    observed = []

    def chat(stage, prompt):
        observed.append((stage, prompt))
        if stage == "qa":
            assert "Asking user: User_1" in prompt and "gold-only-secret" not in prompt
            return "Final: 5000"
        assert "gold-only-secret" in prompt
        return "Final: Correct"

    report = g.evaluate(run, cfg, m, clock, trace, chat, "development")
    assert report["overall"]["correct"] == 1
    assert g.snapshot(m) == before
    assert [s for s, _ in observed] == ["qa", "judge"]
    g.evaluate(run, cfg, m, clock, trace, chat, "development")
    assert len(observed) == 2  # No judge reroll on resume.
    assert not (run / "results/holdout").exists()
    data = g.load_corpus(cfg["corpus"], cfg["domain"])
    assert len({r["id"] for r in data}) == 3  # Same Msg_1 in different channels.


def test_partial_ingestion_stops_and_resumes_without_reingesting(experiment, monkeypatch):
    run, cfg, m, clock, trace, _ = experiment
    original = m.ingest_message
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(kwargs["message_id"])
        if len(calls) == 2:
            return {"status": "needs_attention", "events": []}
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "ingest_message", interrupted)
    with pytest.raises(ValidationError):
        g.ingest_run(run, cfg, m, clock, trace)
    checkpoint = g.read_json(run / "checkpoint.json")
    assert checkpoint["next_index"] == 1 and not checkpoint["ready"]
    with pytest.raises(ValidationError):
        g.evaluate(run, cfg, m, clock, trace, None, "development")
    monkeypatch.setattr(m, "ingest_message", original)
    assert g.ingest_run(run, cfg, m, clock, trace)["ready"]
    assert len(m.store.all("event")) == 3


def test_smoke_cannot_be_reported_as_benchmark(experiment):
    run, cfg, m, clock, trace, _ = experiment
    g.ingest_run(run, cfg, m, clock, trace)
    with pytest.raises(ValidationError):
        g.evaluate(run, {**cfg, "mode": "smoke"}, m, clock, trace, None, "development")


def test_question_or_config_change_invalidates_run(experiment):
    run, _, _, _, _, _ = experiment
    assert g.verify_run(run)
    g.save(run / "protocol.json", {"changed": True})
    with pytest.raises(ValidationError):
        g.verify_run(run)


def test_scorer_unclear_and_failures_are_not_hidden(tmp_path):
    rows = [
        ("completed", "Correct"),
        ("completed", "Incorrect"),
        ("completed", "Unclear"),
        ("request_failure", None),
        ("running", None),
    ]
    for i, (status, verdict) in enumerate(rows):
        g.save(
            tmp_path / "results/development" / f"{i}.json",
            {"type": "temporal", "status": status, "verdict": verdict},
        )
    totals = g.summarize(tmp_path, "development")["overall"]
    assert totals["paper_accuracy"] == 0.5
    assert totals["correct_over_completed"] == pytest.approx(1 / 3)
    assert totals["request_failure"] == totals["interrupted"] == totals["unclear"] == 1
    assert g.parse_judgment("Final: not correct") == "Incorrect"
    assert g.parse_judgment("Final: undecidable") == "Unclear"


def test_failure_is_persisted_without_automatic_judge_retry(experiment):
    run, cfg, m, clock, trace, _ = experiment
    g.ingest_run(run, cfg, m, clock, trace)
    calls = []

    def chat(stage, prompt):
        calls.append(stage)
        if stage == "judge":
            raise TimeoutError("test")
        return "Final: answer"

    report = g.evaluate(run, cfg, m, clock, trace, chat, "development")
    assert report["overall"]["request_failure"] == 1
    g.evaluate(run, cfg, m, clock, trace, chat, "development")
    assert calls == ["qa", "judge"]


def test_invalid_subject_falls_back_to_evidence_only(env):  # noqa: F811
    m, model, _, _ = env
    model.fragments = [{"subject_id": "ghost", "category": "logistics", "quote": "8000"}]
    result = m.ingest_message(
        actor(),
        message_id="ghost-subject",
        text="8000",
        fallback_category="logistics",
        metadata={"channel": "project-a"},
    )
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["subject"] == "marcus" and card["state_eligible"] is False
    assert card["state_key"] is None


def test_done_without_cards_is_a_write_failure(experiment, monkeypatch):
    run, cfg, m, clock, trace, _ = experiment

    def empty(*args, **kwargs):
        return {"status": "done", "events": [{"status": "done", "card_ids": []}]}

    monkeypatch.setattr(m, "ingest_message", empty)
    with pytest.raises(ValidationError):
        g.ingest_run(run, cfg, m, clock, trace)
    assert g.read_json(run / "checkpoint.json")["next_index"] == 0


def test_judge_invalid_output_is_unclear_not_request_failure(experiment):
    run, cfg, m, clock, trace, _ = experiment
    g.ingest_run(run, cfg, m, clock, trace)

    def chat(stage, prompt):
        if stage == "qa":
            return "Final: 5000"
        raise g.ModelOutputError("judge truncated", "Final: Cor")

    report = g.evaluate(run, cfg, m, clock, trace, chat, "development")
    record = g.read_json(run / "results/development/knowledge_update__q1.json")
    assert record["status"] == "completed" and record["verdict"] == "Unclear"
    assert record["judge_failure"] == "invalid_output"
    assert report["overall"]["request_failure"] == 0
    assert report["overall"]["unclear"] == 1


def test_interrupted_question_is_resumed_but_decided_one_is_not(experiment):
    run, cfg, m, clock, trace, _ = experiment
    g.ingest_run(run, cfg, m, clock, trace)
    path = run / "results/development/knowledge_update__q1.json"
    g.save(path, {"id": "q1", "type": "knowledge_update", "status": "running", "attempt": 1})
    calls = []

    def chat(stage, prompt):
        calls.append(stage)
        return "Final: 5000" if stage == "qa" else "Final: Correct"

    report = g.evaluate(run, cfg, m, clock, trace, chat, "development")
    record = g.read_json(path)
    assert record["status"] == "completed" and record["attempt"] == 2
    assert report["overall"]["correct"] == 1
    g.evaluate(run, cfg, m, clock, trace, chat, "development")
    assert len(calls) == 2


def test_oracle_diagnostic_is_not_a_score(experiment):
    run, cfg, m, clock, trace, _ = experiment
    g.ingest_run(run, cfg, m, clock, trace)
    card_id = m.store.all("card")[0]["id"]
    calls = []

    def chat(stage, prompt):
        calls.append(stage)
        assert "gold-only-secret" not in prompt
        return "Final: 5000"

    result = g.diagnose_oracle(run, cfg, "development", "q1", [card_id], chat, trace)
    assert result["scope"] == "oracle_diagnostic_not_score"
    assert calls == ["qa"]
    assert (run / "diagnostics/oracle/development__q1.json").exists()
    assert not (run / "results/development").exists()


def test_protocol_mismatch_is_rejected_before_ingestion(experiment):
    run, _, _, _, _, _ = experiment
    protocol = g.read_json(run / "protocol.json")
    protocol["judge"]["max_tokens"] = 256
    broken = run.parent / "broken-protocol.json"
    g.save(broken, protocol)
    with pytest.raises(ValidationError):
        g.prepare(
            run.parent / "second-run",
            run.parent / "assets/corpus.json",
            broken,
            run.parent / "assets/manifest.json",
        )
