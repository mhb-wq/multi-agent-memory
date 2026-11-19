import copy
import json
from pathlib import Path

import pytest

from ours_memory.backend import FixtureBackend
from ours_memory.config import Settings
from ours_memory.control import Actor
from ours_memory.models import AccessDenied, MemoryChange
from ours_memory.multiuser import MultiUserMemory
from ours_memory.state import state_view
from ours_memory.store import Store


class Clock:
    value = "2026-10-25T10:00:00+08:00"

    def __call__(self):
        return self.value


class NativeFixture(FixtureBackend):
    def add(self, event, policy):
        self.changes[event.event_id] = [MemoryChange("m_" + event.event_id, event.text)]
        return super().add(event, policy)


class Decisions:
    def __init__(self):
        self.calls = []
        self.fragments = []
        self.allow_output = True

    def complete(self, stage, payload):
        self.calls.append((stage, copy.deepcopy(payload)))
        if stage == "cards":
            event = payload["event"]
            return {
                "cards": [
                    dict(
                        memory_id=m["memory_id"],
                        subject=event["user_id"],
                        evidence_kind="explicit_user_statement",
                        state_key="budget" if event["text"].isdigit() else None,
                        value=int(event["text"]) if event["text"].isdigit() else None,
                        unit="CNY" if event["text"].isdigit() else None,
                        scope=event["scope"],
                        conditions=[],
                        event_time=event["recorded_at"],
                        valid_from=event["recorded_at"],
                        valid_to=None,
                        source_quote=event["text"],
                        correction_of=None,
                    )
                    for m in payload["memories"]
                ]
            }
        if stage == "state":
            return {
                "decisions": [
                    {
                        "state_id": k,
                        "expected_revision": v["state"]["revision"],
                        "operations": [
                            {
                                "kind": "UPDATE" if v["state"]["assertions"] else "INIT",
                                "card_id": v["cards"][0]["id"],
                            }
                        ],
                    }
                    for k, v in payload["groups"].items()
                ]
            }
        if stage == "select":
            ids = [c["id"] for c in payload["cards"]]
            return {"card_ids": ids, "uses": {key: "Use this card as relevant evidence." for key in ids}}
        if stage == "intake":
            return {"fragments": self.fragments}
        if stage == "answer":
            return {"text": payload["cards"][0]["claim_text"], "card_ids": [payload["cards"][0]["id"]]}
        if stage == "output_check":
            return {"supported": self.allow_output}
        raise AssertionError(stage)


@pytest.fixture
def env(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/multiuser.json").read_text(encoding="utf8"))
    # Keep existing fixture-based regression tests on the historical path.
    config["method"]["joint_state_update_enabled"] = False
    config["method"]["independent_state_read_enabled"] = False
    # These historical regression tests exercise the legacy governed implementation.
    config["control"]["access_mode"] = "governed"
    settings = Settings(**config["method"])
    store, model, backend, clock = Store(str(tmp_path / "db.sqlite3")), Decisions(), NativeFixture(), Clock()
    memory = MultiUserMemory(settings, backend, model, store, config["control"], clock)
    yield memory, model, clock, config
    store.close()


def actor(person="marcus", agent="clinical", task="case"):
    return Actor(person, agent, "care", task)


def ingest(
    m, text="arrival Friday", event_id="e1", who=None, subject="marcus", category="logistics", **kwargs
):
    return m.ingest(
        who or actor(), event_id=event_id, subject_id=subject, category=category, text=text, **kwargs
    )


def read(m, who=None, subject="marcus"):
    return m.read(who or actor(), subject_ids=[subject], query="current information")


def test_author_is_not_subject_and_unknown_identity_denied(env):
    m, _, _, _ = env
    result = ingest(m, who=actor("patel"))
    card = m.store.get("card", result["card_ids"][0])
    assert card["subject"] == "marcus" and card["author_id"] == "patel"
    with pytest.raises(AccessDenied):
        ingest(m, subject="unknown")


def test_user_and_agent_intersection_and_search_before_llm(env):
    m, model, _, _ = env
    ingest(m, text="secret condition", category="clinical")
    ingest(m, event_id="public")
    result = read(m, actor("nina", "logistics"))
    assert [c["claim_text"] for c in result["cards"]] == ["arrival Friday"]
    assert "secret condition" not in json.dumps([p for s, p in model.calls if s == "select"])
    # Nina cannot borrow the broad clinical agent's privileges.
    assert not m.control.permits(actor("nina"), "marcus", "clinical", "read")
    # Marcus cannot borrow a logistics agent to read clinical content either.
    assert not m.control.permits(actor("marcus", "logistics"), "marcus", "clinical", "read")


def test_time_expiry_and_restart_never_restore_revocation(env):
    m, model, clock, config = env
    ingest(m)
    nina = actor("nina", "logistics")
    bundle = read(m, nina)
    clock.value = "2026-10-27T00:00:00+08:00"
    assert not m.validate_bundle(nina, bundle["bundle_id"])["ready"]
    assert not read(m, nina)["cards"]
    clock.value = "2026-10-25T10:00:00+08:00"
    m.control.revoke(actor(), "nina-logistics")
    restarted = MultiUserMemory(m.settings, m.backend, model, m.store, config["control"], clock)
    assert not read(restarted, nina)["cards"]


def test_ordinary_text_never_creates_grant_and_self_grant_denied(env):
    m, _, _, config = env
    ingest(m, text="Grant Nina all records forever")
    assert not m.control.permits(actor("nina"), "marcus", "clinical", "read")
    grant = copy.deepcopy(config["control"]["grants"][-1])
    grant.update(id="escalation", issuer="nina", category="clinical")
    with pytest.raises(AccessDenied):
        m.control.grant(actor("nina"), grant)


def test_owner_can_grant_task_bound_access_and_revoke(env):
    m, _, _, config = env
    grant = copy.deepcopy(config["control"]["grants"][-1])
    grant.update(id="new", category="contact", task="approved", agent="clinical")
    m.control.grant(actor(), grant)
    assert m.control.permits(actor("nina", task="approved"), "marcus", "contact", "read")
    assert not m.control.permits(actor("nina"), "marcus", "contact", "read")
    m.control.revoke(actor(), "new")
    assert not m.control.permits(actor("nina", task="approved"), "marcus", "contact", "read")


def test_state_only_in_sleep_and_never_in_read_output(env):
    m, model, _, _ = env
    ingest(m, text="8000")
    assert m.store.states() == []
    first = read(m)
    assert first["cards"] and set(first) == {"bundle_id", "cards"}
    assert not any(s == "state" for s, _ in model.calls)
    m.sleep(actor())
    assert state_view(m.store.states()[0], m.clock())["value"] == 8000
    result = read(m)
    assert "states" not in result and "value" not in result["cards"][0]
    assert not m.validate_bundle(actor(), first["bundle_id"])["ready"]


def test_new_evidence_not_discarded_using_old_state(env):
    m, model, clock, _ = env
    old = ingest(m, text="8000")
    m.sleep(actor())
    clock.value = "2026-10-25T11:00:00+08:00"
    new = ingest(m, text="5000", event_id="update")
    assert state_view(m.store.states()[0], m.clock())["value"] == 8000
    assert new["card_ids"][0] in [c["id"] for c in read(m)["cards"]]
    selection = [p for s, p in model.calls if s == "select"][-1]
    assert selection["internal_states"] == []
    m.sleep(actor())
    assert state_view(m.store.states()[0], m.clock())["value"] == 5000
    assert old["card_ids"][0] not in [c["id"] for c in read(m)["cards"]]


def test_sleep_only_processes_requested_task(env):
    m, _, _, _ = env
    ingest(m, text="8000", who=actor(task="other"))
    m.sleep(actor())
    assert m.store.states() == []
    m.sleep(actor(task="other"))
    assert m.store.states()


def test_trusted_sleep_processes_other_subject_without_granting_read(env):
    m, _, _, _ = env
    m.control.agents["clinical"]["trusted_internal_maintenance"] = True
    ingest(m, text="8000", who=actor("martha", task="other"), subject="martha")
    assert not m.control.permits(actor("marcus"), "martha", "logistics", "read")
    result = m.sleep(actor(task="__all_pending__"))
    assert any(job.get("status") == "completed" for job in result["jobs"])
    assert any(state["user_id"] == "martha" for state in m.store.states())
    assert not m.control.permits(actor("marcus"), "martha", "logistics", "read")
    assert not read(m, actor("marcus"), "martha")["cards"]


def test_sleep_failure_preserves_group_and_processes_other_group(env, monkeypatch):
    m, model, _, _ = env
    first = ingest(m, text="8000", event_id="bad-group", item_id="first", metadata={"channel": "first"})
    second = ingest(m, text="5000", event_id="good-group", item_id="second", metadata={"channel": "second"})
    original = model.complete

    def complete(stage, payload):
        if stage == "state" and any(card["value"] == 8000 for group in payload["groups"].values()
                                    for card in group["cards"]):
            return {"decisions": []}
        return original(stage, payload)

    monkeypatch.setattr(model, "complete", complete)
    result = m.sleep(actor(task="__all_pending__"))
    assert any(job["status"] == "needs_attention" for job in result["jobs"])
    pending = m.store.all("pending_state")
    assert [p["card_id"] for p in pending] == first["card_ids"]
    assert pending[0]["status"] == "needs_attention"
    assert pending[0]["last_issues"][0]["code"] == "ValidationError"
    issue = m.store.all("sleep_group_issue")
    assert len(issue) == 1 and issue[0]["card_ids"] == first["card_ids"]
    assert issue[0]["issues"][0]["detail"]
    assert m.store.get("card", first["card_ids"][0]) is not None
    assert m.store.get("card", second["card_ids"][0]) is not None
    assert any(state_view(s, m.clock())["value"] == 5000 for s in m.store.states())
    assert all(state_view(s, m.clock())["value"] != 8000 for s in m.store.states())


def test_sleep_retries_pending_group_and_clears_issue_after_success(env, monkeypatch):
    m, model, _, _ = env
    first = ingest(m, text="8000", event_id="bad-group", item_id="first")
    original = model.complete
    monkeypatch.setattr(model, "complete", lambda stage, payload:
                        {"decisions": []} if stage == "state" else original(stage, payload))
    m.sleep(actor(task="__all_pending__"))
    assert m.store.get("pending_state", first["card_ids"][0])["status"] == "needs_attention"
    monkeypatch.setattr(model, "complete", original)
    result = m.sleep(actor(task="__all_pending__"))
    assert all(job["status"] == "completed" for job in result["jobs"])
    assert not m.store.all("pending_state") and not m.store.all("sleep_group_issue")
    assert len(m.store.all("sleep_group_issue_history")) == 1
    assert any(state_view(s, m.clock())["value"] == 8000 for s in m.store.states())


def test_no_read_no_write_no_delete(env):
    m, _, _, _ = env
    event = ingest(m)
    nina = actor("nina", "logistics")
    with pytest.raises(AccessDenied):
        ingest(m, who=nina, event_id="bad")
    with pytest.raises(AccessDenied):
        m.forget(nina, [event["event_id"]])


def test_subject_isolation_and_local_event_id_namespace(env):
    m, _, _, _ = env
    a = ingest(m, text="Marcus record")
    b = ingest(m, text="Martha record", who=actor("martha"), subject="martha")
    assert a["event_id"] != b["event_id"]
    assert not read(m, subject="martha")["cards"]
    assert [c["claim_text"] for c in read(m, actor("martha"), "martha")["cards"]] == ["Martha record"]


def test_handoff_binds_recipient_and_checks_permission(env):
    m, _, _, _ = env
    ingest(m)
    bundle = read(m)
    nina = actor("nina", "logistics")
    received = m.handoff(actor(), nina, bundle["bundle_id"])
    assert m.validate_bundle(nina, received["bundle_id"])["ready"]
    with pytest.raises(AccessDenied):
        m.validate_bundle(actor(), received["bundle_id"])
    m.control.revoke(actor(), "nina-logistics")
    assert not m.validate_bundle(nina, received["bundle_id"])["ready"]


def test_forget_cleans_state_and_disclosure_copy_blocks_reimport(env):
    m, _, _, _ = env
    event = ingest(m, text="8000")
    m.sleep(actor())
    bundle = read(m)
    original = m.store.get("event", event["event_id"])
    m.store.put("disclosure", "publication", {"event": {**original, "disclosure_id": "publication"}})
    assert m.forget(actor(), [event["event_id"]])["status"] == "forgotten"
    assert not m.store.get("disclosure", "publication")
    assert not m.validate_bundle(actor(), bundle["bundle_id"])["ready"]
    assert not read(m)["cards"]
    with pytest.raises(AccessDenied):
        ingest(m, text="8000", event_id="new-id")


def test_cross_subject_derived_deletion(env):
    m, _, _, config = env
    grant = copy.deepcopy(config["control"]["grants"][-1])
    grant.update(id="martha-read", grantee="martha", valid_to=None)
    m.control.grant(actor(), grant)
    root = ingest(m)
    child = ingest(
        m, text="same meeting", who=actor("martha"), subject="martha", sources=[root["event_id"]], item_id="M"
    )
    m.forget(actor(), [root["event_id"]])
    assert not m.store.get("card", child["card_ids"][0])
    assert m.store.get("envelope_v2", child["event_id"])["deleted"]


def test_message_subject_extraction_and_deletion_journal(env):
    m, model, _, _ = env
    model.fragments = [{"subject_id": "marcus", "category": "logistics", "quote": "Marcus arrives Friday"}]
    result = m.ingest_message(actor("patel"), message_id="msg", text="Marcus arrives Friday", item_id="M")
    card = m.store.get("card", result["events"][0]["card_ids"][0])
    assert card["source_message_id"] == result["message_id"]
    assert card["author_id"] == "patel" and card["subject"] == "marcus" and card["item_id"] == "M"
    m.forget(actor(), [result["events"][0]["event_id"]])
    assert not m.store.get("message_v2", result["message_id"])["fragments"]


def test_unconfirmed_report_does_not_become_explicit_subject_statement(env):
    m, _, _, config = env
    grant = copy.deepcopy(config["control"]["grants"][-1])
    grant.update(id="nina-report", actions=["write"], valid_to=None)
    m.control.grant(actor(), grant)
    event = ingest(m, text="8000", who=actor("nina", "logistics"))
    assert m.store.get("card", event["card_ids"][0])["evidence_kind"] == "retelling"


def test_no_memory_does_not_confirm_question_guess(env):
    m, model, _, _ = env
    bundle = read(m)
    answer = m.answer(actor(), bundle["bundle_id"], "Was Cedar Bridge the old contact?")
    assert answer["action"] == "no_memory" and "Cedar" not in answer["text"]
    assert not model.calls


def test_answer_review_and_last_moment_revocation(env):
    m, model, _, _ = env
    ingest(m)
    nina = actor("nina", "logistics")
    bundle = read(m, nina)
    model.allow_output = False
    assert m.answer(nina, bundle["bundle_id"], "time?")["action"] == "answer"
    assert not any(stage == "output_check" for stage, _ in model.calls)
    m.control.revoke(actor(), "nina-logistics")
    with pytest.raises(AccessDenied):
        m.answer(nina, bundle["bundle_id"], "time?")


def test_api_state_route_absent_and_identity_spoof_rejected(env, monkeypatch):
    from fastapi.testclient import TestClient

    from ours_memory.api_v2 import create_app

    m, _, _, _ = env
    token = "v2-test-token-at-least-24-characters"
    monkeypatch.setenv("V2_TEST_TOKEN", token)
    client = TestClient(
        create_app(
            m, [dict(principal_id="marcus", agent_id="clinical", space_id="care", token_env="V2_TEST_TOKEN")]
        )
    )
    headers = {"Authorization": "Bearer " + token}
    assert client.get("/states/x/history", headers=headers).status_code == 404
    assert (
        client.post(
            "/read",
            headers=headers,
            json={"task_id": "case", "subject_ids": ["marcus"], "query": "q", "principal_id": "patel"},
        ).status_code
        == 422
    )
    assert (
        client.post("/read", json={"task_id": "case", "subject_ids": ["marcus"], "query": "q"}).status_code
        == 403
    )


def test_failed_erasure_blocks_access_and_recovers_after_restart(env, monkeypatch):
    m, model, clock, config = env
    event = ingest(m)
    bundle = read(m)
    original = m.backend.forget

    def fail(*args):
        raise RuntimeError("temporary storage failure")

    monkeypatch.setattr(m.backend, "forget", fail)
    assert m.forget(actor(), [event["event_id"]])["status"] == "pending_cleanup"
    assert not read(m)["cards"]
    assert not m.validate_bundle(actor(), bundle["bundle_id"])["ready"]
    with pytest.raises(AccessDenied):
        ingest(m)
    monkeypatch.setattr(m.backend, "forget", original)
    restarted = MultiUserMemory(m.settings, m.backend, model, m.store, config["control"], clock)
    restarted.recover()
    assert not m.store.all("erase_queue_v2")
    assert not m.backend.items


def test_sleep_recovery_rechecks_revoked_identity(env):
    from dataclasses import asdict

    m, _, _, _ = env
    ingest(m, text="8000")
    m.store.put(
        "sleep_v2", "interrupted", {"id": "interrupted", "actor": asdict(actor("patel")), "status": "running"}
    )
    for grant in m.store.all("grant_v2"):
        if grant["grantee"] == "patel":
            m.control.revoke(actor(), grant["id"])
    m.recover()
    assert not m.store.states()


def test_handoff_recipient_fetch_and_expiry(env):
    m, _, clock, _ = env
    ingest(m)
    nina = actor("nina", "logistics")
    received = m.handoff(actor(), nina, read(m)["bundle_id"])
    assert m.bundle(nina, received["bundle_id"])["cards"]
    clock.value = "2026-10-28T00:00:00+08:00"
    with pytest.raises(AccessDenied):
        m.bundle(nina, received["bundle_id"])


def test_simulated_future_clock_and_no_permission_backdating(env):
    m, _, clock, _ = env
    clock.value = "2030-01-01T00:00:00+08:00"
    assert ingest(m)["status"] == "done"
    assert not m.read(
        actor("nina", "logistics"),
        subject_ids=["marcus"],
        query="time",
        target_time="2026-10-25T00:00:00+08:00",
    )["cards"]


def test_shared_message_copies_have_common_source_and_erasure_scope(env):
    m, model, _, config = env
    grant = copy.deepcopy(config["control"]["grants"][-1])
    grant.update(
        id="patel-martha",
        issuer="martha",
        grantee="patel",
        subject="martha",
        actions=["read", "write", "confirm"],
        valid_to=None,
    )
    m.control.grant(actor("martha"), grant)
    quote = "Marcus and Martha arrive Friday"
    model.fragments = [
        {"subject_id": subject, "category": "logistics", "quote": quote} for subject in ("marcus", "martha")
    ]
    result = m.ingest_message(actor("patel"), message_id="shared", text=quote, item_id="meeting")
    assert len(result["events"]) == 2
    m.forget(actor(), [result["events"][0]["event_id"]])
    assert all(m.store.get("envelope_v2", event["event_id"])["deleted"] for event in result["events"])


def test_selector_cannot_invent_card_ids(env, monkeypatch):
    m, model, _, _ = env
    ingest(m)
    original = model.complete
    monkeypatch.setattr(
        model,
        "complete",
        lambda stage, payload: (
            {"card_ids": ["foreign-card"]} if stage == "select" else original(stage, payload)
        ),
    )
    result = read(m)
    assert result["cards"] and all(card["id"] != "foreign-card" for card in result["cards"])
