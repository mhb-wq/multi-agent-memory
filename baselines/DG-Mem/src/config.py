"""TOML experiment configuration with switchable LLM profiles."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from .memory_graph import MemoryGraphConfig
from .user_context_graph import UserContextGraphConfig


T = TypeVar("T")


def _from_mapping(cls: type[T], values: Mapping[str, Any] | None) -> T:
    values = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {', '.join(unknown)}")
    return cls(**values)


@dataclass
class RunSettings:
    retriever: str = "memory_graph"
    conversation_json: str = ""
    questions_jsonl: str = ""
    output_jsonl: str = ""
    index_dir: str = ""
    reuse_index: bool = False
    top_k: int = 10
    question_limit: int = 0
    evaluate: bool = False
    judge: bool = True
    qa_workers: int = 4
    resume: bool = False
    agent_prompt: str = "../GroupMemBench/prompts/hipporag_agent_system.txt"
    judge_prompt: str = "../GroupMemBench/prompts/hipporag_judge_system.txt"


@dataclass
class EmbeddingSettings:
    provider: str = "sentence_transformer"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 64


@dataclass
class EntitySettings:
    provider: str = "regex"
    model: str = "en_core_web_sm"
    device: str = "cpu"
    batch_size: int = 64


@dataclass
class RerankerSettings:
    provider: str = "none"
    model: str = ""
    device: str = "auto"
    batch_size: int = 64


@dataclass
class LLMSettings:
    provider: str = "openai_compatible"
    model: str = ""
    api_key_env: str = ""
    base_url: str = ""
    base_url_env: str = ""
    api_version: str = "2024-02-15-preview"
    timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_initial_seconds: float = 1.0
    max_tokens: int = 512
    max_tokens_field: str = "max_tokens"
    temperature: float = 0.0
    send_temperature: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def api_key(self, required: bool = False) -> str:
        value = os.getenv(self.api_key_env, "") if self.api_key_env else ""
        if required and not value:
            name = self.api_key_env or "<api_key_env>"
            raise ValueError(f"Missing API key environment variable: {name}")
        return value

    def resolved_base_url(self, default: str = "", required: bool = False) -> str:
        value = os.getenv(self.base_url_env, "") if self.base_url_env else ""
        value = value or self.base_url or default
        if required and not value:
            name = self.base_url_env or "base_url"
            raise ValueError(f"Missing model endpoint: {name}")
        return value.rstrip("/")


@dataclass
class LLMProfile:
    agent: LLMSettings
    judge: LLMSettings


@dataclass
class ExperimentConfig:
    name: str = "groupmemrag"
    llm_profile_name: str = ""
    run: RunSettings = field(default_factory=RunSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    entity: EntitySettings = field(default_factory=EntitySettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    memory_graph: MemoryGraphConfig = field(default_factory=MemoryGraphConfig)
    user_context_graph: UserContextGraphConfig = field(default_factory=UserContextGraphConfig)
    llm_profile: LLMProfile | None = None
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: str | Path, profile: str | None = None) -> "ExperimentConfig":
        source = Path(path).resolve()
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
        experiment = dict(raw.get("experiment", {}))
        selected_profile = str(profile or experiment.get("llm_profile", "")).strip()
        profile_data = dict(raw.get("llm_profiles", {}).get(selected_profile, {})) if selected_profile else {}
        llm_profile = None
        if selected_profile:
            if not profile_data:
                available = ", ".join(sorted(raw.get("llm_profiles", {}))) or "<none>"
                raise ValueError(f"Unknown LLM profile {selected_profile!r}. Available: {available}")
            llm_profile = LLMProfile(
                agent=_from_mapping(LLMSettings, profile_data.get("agent")),
                judge=_from_mapping(LLMSettings, profile_data.get("judge")),
            )
        return cls(
            name=str(experiment.get("name", source.stem)),
            llm_profile_name=selected_profile,
            run=_from_mapping(RunSettings, raw.get("run")),
            embedding=_from_mapping(EmbeddingSettings, raw.get("embedding")),
            entity=_from_mapping(EntitySettings, raw.get("entity")),
            reranker=_from_mapping(RerankerSettings, raw.get("reranker")),
            memory_graph=_from_mapping(MemoryGraphConfig, raw.get("memory_graph")),
            user_context_graph=_from_mapping(UserContextGraphConfig, raw.get("user_context_graph")),
            llm_profile=llm_profile,
            source_path=source,
        )

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or self.source_path is None:
            return path
        return (self.source_path.parent / path).resolve()

    def safe_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "experiment": {"name": self.name, "llm_profile": self.llm_profile_name},
            "run": asdict(self.run),
            "embedding": asdict(self.embedding),
            "entity": asdict(self.entity),
            "reranker": asdict(self.reranker),
            "memory_graph": asdict(self.memory_graph),
            "user_context_graph": asdict(self.user_context_graph),
        }
        if self.llm_profile:
            agent = asdict(self.llm_profile.agent)
            judge = asdict(self.llm_profile.judge)
            for settings in (agent, judge):
                settings["extra_headers"] = {
                    key: ("<redacted>" if any(marker in key.lower() for marker in ("key", "token", "auth")) else value)
                    for key, value in settings.get("extra_headers", {}).items()
                }
            data["llm"] = {
                "agent": agent,
                "judge": judge,
            }
        return data
