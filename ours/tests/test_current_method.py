import json

import pytest

from ours_memory.config import Settings, validate_current_method
from ours_memory.groupmem import build_memory
from ours_memory.models import ValidationError
from ours_memory.multiuser import MultiUserMemory
from ours_memory.runtime import build

pytest_plugins = ["current_fixture"]


@pytest.mark.parametrize("field", ["sleep_enabled", "state_enabled"])
def test_current_method_cannot_disable_core_features(field):
    settings = Settings()
    validate_current_method(settings)
    setattr(settings, field, False)
    with pytest.raises(ValidationError):
        validate_current_method(settings)


@pytest.mark.parametrize("field", ["card_mode", "joint_state_update_enabled",
                                   "immediate_state_maintenance_enabled", "independent_state_read_enabled"])
def test_old_version_switches_are_not_settings(field):
    with pytest.raises(TypeError):
        Settings(**{field: False})


def test_direct_construction_rejects_legacy_access_mode():
    with pytest.raises(ValidationError):
        MultiUserMemory(Settings(), None, None, None, {"access_mode": "governed"})


@pytest.mark.parametrize("mode,access", [("legacy", "shared_pool"), ("multiuser-v2", "governed")])
def test_runtime_rejects_legacy_mode_before_opening_database(tmp_path, mode, access):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"method": {}, "mode": mode, "control": {"access_mode": access}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        build(path)
    assert not list(tmp_path.rglob("*.sqlite3"))


def test_gatemem_builder_rejects_old_variant_before_opening_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    (tmp_path / "protocol.json").write_text(json.dumps({"memory_model": "fixture", "embedding": {},
        "method": {"joint_state_update_enabled": False}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        build_memory(tmp_path, {"space": "case", "users": ["alice"]}, None, None)
    assert not (tmp_path / "state.sqlite3").exists()


def test_current_api_binds_identity_and_rejects_forged_requests(env, monkeypatch):
    from fastapi.testclient import TestClient

    from ours_memory.api_v2 import create_app

    memory, _, _, _ = env
    token = "current-test-token-at-least-24-characters"
    monkeypatch.setenv("CURRENT_TEST_TOKEN", token)
    identity = {"principal_id": "marcus", "agent_id": "clinical", "space_id": "care",
                "token_env": "CURRENT_TEST_TOKEN"}
    client = TestClient(create_app(memory, [identity]))
    payload = {"task_id": "case", "subject_ids": ["marcus"], "query": "q"}
    assert client.post("/read", json=payload).status_code == 403
    assert client.post("/read", headers={"Authorization": "Bearer " + token},
                       json={**payload, "principal_id": "patel"}).status_code == 422
    with pytest.raises(ValidationError):
        create_app(memory, [identity, identity])
