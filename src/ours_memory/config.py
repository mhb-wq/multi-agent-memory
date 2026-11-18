from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import ValidationError

PERSONAL_STATE_TEXT_KEYS = (
    "study_goal", "meeting_preference", "response_language_preference", "available_time",
    "dietary_restriction", "travel_constraint", "work_location", "recurring_schedule", "ongoing_commitment",
)


def boolean(value: str) -> bool:
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    if value.lower() in {"false", "0", "no", "off"}:
        return False
    raise ValidationError("Sleep开关必须为true或false")


@dataclass
class Settings:
    database: str = "runtime/ours_memory.sqlite3"
    sleep_enabled: bool = True
    state_enabled: bool = True
    routing_enabled: bool = True
    source_expansion_enabled: bool = False
    independent_state_read_enabled: bool = False
    independent_state_limit: int = 10
    joint_state_update_enabled: bool = False
    immediate_state_maintenance_enabled: bool = True
    joint_candidate_limit: int = 10
    state_similarity_threshold: float = 0.2
    candidate_limit: int = 20
    card_batch_size: int = 8
    card_repair_attempts: int = 0
    # "extract": MemOS extractor -> one ENRICH call per native memory (upstream path).
    # "direct": one cards_direct call per message; the card text itself is indexed.
    card_mode: str = "extract"
    max_model_input_chars: int = 192000
    answer_max_input_chars: int = 384000
    max_context_chars: int = 16000
    team_agents: dict[str, list[str]] = field(default_factory=dict)
    state_keys: dict[str, dict] = field(
        default_factory=lambda: {
            "available_study_hours_per_week": {"unit": "hours/week", "type": "number", "min": 0},
            "budget": {"unit": None, "type": "number", "min": 0},
            "goal": {"unit": None, "type": "text"},
            "preference": {"unit": None, "type": "text"},
            "constraint": {"unit": None, "type": "text"},
            "identity": {"unit": None, "type": "text"},
            "task_plan": {"unit": None, "type": "text"},
            **{key: {"unit": None, "type": "text"} for key in PERSONAL_STATE_TEXT_KEYS},
        }
    )
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: int = 90
    max_tokens: int = 4096
    answer_max_tokens: int = 8192
    state_max_tokens: int = 65536
    retries: int = 2
    request_options: dict = field(default_factory=dict)
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    mem0: dict = field(default_factory=dict)

    def validate(self) -> None:
        if (type(self.state_similarity_threshold) not in (int, float)
                or not math.isfinite(self.state_similarity_threshold)
                or not -1 <= self.state_similarity_threshold <= 1):
            raise ValidationError("state_similarity_threshold 必须在 -1 到 1 之间")
        for key in ("sleep_enabled", "state_enabled", "routing_enabled", "source_expansion_enabled", "independent_state_read_enabled", "joint_state_update_enabled", "immediate_state_maintenance_enabled"):
            if type(getattr(self, key)) is not bool:
                raise ValidationError(f"{key}必须是布尔值")
        for key in ("candidate_limit", "independent_state_limit", "joint_candidate_limit", "card_batch_size", "max_model_input_chars", "answer_max_input_chars", "max_context_chars", "answer_max_tokens", "state_max_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValidationError(f"{key}必须是正整数")
        if type(self.card_repair_attempts) is not int or self.card_repair_attempts not in {0, 1}:
            raise ValidationError("card_repair_attempts只允许0或1，禁止无限重试")
        if self.card_mode not in {"extract", "direct"}:
            raise ValidationError("card_mode只允许extract或direct")
        if not self.state_enabled and self.sleep_enabled:
            raise ValidationError("当前Sleep复核用户状态；禁用状态时须同时关闭Sleep")
        if (self.joint_state_update_enabled and self.immediate_state_maintenance_enabled
                and (not self.state_enabled or not self.sleep_enabled)):
            raise ValidationError("逐卡即时维护需要启用状态和维护功能")

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        settings = cls(**data)
        if "TEAM_MEMORY_SLEEP_ENABLED" in os.environ:
            settings.sleep_enabled = boolean(os.environ["TEAM_MEMORY_SLEEP_ENABLED"])
        settings.model = os.environ.get("TEAM_MEMORY_MODEL", settings.model)
        settings.base_url = os.environ.get("TEAM_MEMORY_BASE_URL", settings.base_url)
        settings.validate()
        return settings

    def to_dict(self) -> dict:
        return asdict(self)
