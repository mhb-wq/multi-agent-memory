import copy

import pytest
from test_multiuser import actor

from ours_memory.models import ValidationError, stable_id
from ours_memory.multiuser import MultiUserMemory

pytest_plugins = ["test_multiuser"]


def shared_memory(env):
    original, model, clock, config = env
    control = copy.deepcopy(config["control"])
    control["access_mode"] = "shared_pool"
    return MultiUserMemory(original.settings, original.backend, model, original.store, control, clock), model


def test_zero_card_message_still_preserves_original_and_metadata(env):
    memory, model = shared_memory(env)
    model.complete = lambda stage, payload: {"cards": []}
    result = memory.ingest_message(actor(), message_id="empty", text="OK.",
                                   observed_at="2026-10-25T09:00:00+08:00",
                                   metadata={"channel": "project", "reply_to": "previous"})
    source = memory.store.get("source_message_v2", result["message_id"])
    assert source["text"] == "OK."
    assert (source["author_id"], source["channel"], source["reply_to"]) == ("marcus", "project", "previous")
    assert source["observed_at"] == "2026-10-25T09:00:00+08:00"
    assert source["status"] == "extracted" and result["events"] == []
    assert memory.ingest_message(actor(), message_id="empty", text="OK.",
                                 observed_at="2026-10-25T09:00:00+08:00",
                                 metadata={"channel": "project", "reply_to": "previous"}) == result
    with pytest.raises(ValidationError):
        memory.ingest_message(actor(), message_id="empty", text="Changed")


def test_provider_failure_keeps_source_and_retry_can_extract(env):
    memory, model = shared_memory(env)
    def fail(stage, payload):
        raise RuntimeError("offline")
    model.complete = fail
    result = memory.ingest_message(actor(), message_id="retry-source", text="Original message")
    assert result["status"] == "accepted_pending" and result["processing_status"] == "extraction_pending"
    key = stable_id("message_v2", "care", "marcus", "retry-source")
    assert memory.store.get("source_message_v2", key)["text"] == "Original message"
    assert memory.store.get("source_message_v2", key)["status"] == "extraction_pending"
    model.complete = lambda stage, payload: {"cards": []}
    assert memory.ingest_message(actor(), message_id="retry-source", text="Original message")["message_id"] == key
    assert memory.store.get("source_message_v2", key)["status"] == "extracted"
