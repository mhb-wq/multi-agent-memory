from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


class ValidationError(ValueError):
    pass


class MigrationRequired(ValidationError):
    pass


class ModelOutputError(ValidationError):
    pass


class OutputTruncated(ModelOutputError):
    pass


class InputTooLarge(ValidationError):
    pass


class AccessDenied(PermissionError):
    pass


class StaleRevision(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_key(value: Any) -> str:
    """Numeric equality for facts and plan fingerprints; booleans stay distinct."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int) or value.is_integer():
            return canonical(int(value))
    return canonical(unicodedata.normalize("NFC", value) if isinstance(value, str) else value)


def stable_id(prefix: str, *parts: Any) -> str:
    return prefix + "_" + hashlib.sha256(canonical(parts).encode()).hexdigest()[:28]


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError("时间必须为带时区的ISO字符串")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("无法识别的时间") from exc
    if result.tzinfo is None:
        raise ValidationError("时间必须含时区")
    return result.astimezone(timezone.utc)


def iso(value: str) -> str:
    return timestamp(value).isoformat()


def observation_date(value: str) -> str:
    """Calendar references belong to the observation's original offset, not its UTC date."""
    timestamp(value)  # Apply the same timezone/format validation, without replacing the offset.
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def interval(start: str | None, end: str | None) -> None:
    if start:
        timestamp(start)
    if end:
        timestamp(end)
    if start and end and timestamp(start) >= timestamp(end):
        raise ValidationError("有效区间必须前闭后开且起点早于终点")


def active(item: dict, at: str) -> bool:
    time = timestamp(at)
    return (not item.get("valid_from") or timestamp(item["valid_from"]) <= time) and (
        not item.get("valid_to") or time < timestamp(item["valid_to"])
    )


@dataclass(frozen=True)
class Caller:
    """Created by the trusted team runner, never taken from model output."""

    user_id: str
    team_id: str
    agent_id: str


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    team_id: str
    writer_agent_id: str
    task_id: str
    text: str
    recorded_at: str
    allowed_reader_ids: tuple[str, ...]
    speaker: str = "user"
    scope: str = "global"
    source_root_ids: tuple[str, ...] = ()
    # Trusted observation time for imported records; absence is not evidence of 'now'.
    observed_at: str | None = None
    # Opaque capability minted by a trusted runner, never accepted from model output.
    disclosure_id: str | None = None
    author_id: str | None = None
    category: str | None = None
    item_id: str | None = None
    source_message_id: str | None = None
    metadata: dict = field(default_factory=dict)
    state_eligible: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["allowed_reader_ids"] = sorted(set(self.allowed_reader_ids))
        data["source_root_ids"] = sorted(set(self.source_root_ids or (self.event_id,)))
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        copied = dict(data)
        copied["allowed_reader_ids"] = tuple(copied["allowed_reader_ids"])
        copied["source_root_ids"] = tuple(copied.get("source_root_ids", ()))
        return cls(**copied)

    @classmethod
    def record_with_defaults(cls, record: dict) -> dict:
        """Preserve stored metadata while adding dataclass defaults to legacy event records."""
        event = cls.from_dict(
            {key: value for key, value in record.items() if key in cls.__dataclass_fields__}
        )
        return {**record, **event.to_dict()}


@dataclass(frozen=True)
class ReadRequest:
    task_id: str
    query: str
    target_time: str
    scope: str = "global"
    role: str = ""
    purpose: str = "current"
    conditions: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryChange:
    memory_id: str
    text: str
    event: str = "ADD"
    score: float = 0.0


@dataclass
class ReadResult:
    states: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    uncertain: list[dict] = field(default_factory=list)
    revisions: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
