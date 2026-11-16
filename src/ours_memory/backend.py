from __future__ import annotations

import inspect
import os
import time
from typing import Protocol, runtime_checkable

from .models import Caller, Event, MemoryChange, ValidationError, observation_date, stable_id


def partition(event: dict) -> str:
    # Policy cohorts, not mandatory Agent-private/shared dual stores. Native extraction cannot see
    # memories from a different reader set, even while comparing existing facts at write time.
    parts = [event["user_id"], event["team_id"], sorted(event["allowed_reader_ids"])]
    if event.get("category"):
        parts.append(event["category"])
    return stable_id("policy", *parts)


@runtime_checkable
class Backend(Protocol):
    def add(self, event: Event, policy: str) -> list[MemoryChange]: ...
    def search(self, query: str, caller: Caller, policies: list[str], limit: int) -> list[MemoryChange]: ...
    def delete(self, memory_id: str): ...
    def find_event_memories(self, event_id: str, policy: str, user_id: str) -> list[str]: ...
    def forget(self, memory_id: str, policy: str, user_id: str): ...
    def clear_event_cache(self, policy: str, user_id: str): ...


def validate_backend(backend) -> None:
    required = ("add", "search", "delete", "find_event_memories", "forget", "clear_event_cache")
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise ValidationError("记忆后端缺少必需接口：" + ", ".join(missing))


class Mem0Backend:
    def __init__(self, memory, store=None):
        self.memory, self.store = memory, store
        # Experiment adapters set this for structured tool/reference records. It is
        # prepended to the native extraction message only; the stored event text,
        # trace and card quotes remain the original bytes.
        self.native_message_prefix = ""
        self.top_k_name = "top_k" if "top_k" in inspect.signature(memory.search).parameters else "limit"
        if store:
            self._observe_provider_calls(getattr(memory, "llm", None), "chat", "mem0_llm")
            self._observe_provider_calls(
                getattr(memory, "embedding_model", None), "embeddings", "mem0_embedding"
            )

    def _observe_provider_calls(self, provider, resource_name: str, stage: str):
        client = getattr(provider, "client", None)
        resource = getattr(client, resource_name, None)
        if resource_name == "chat":
            resource = getattr(resource, "completions", None)
        original = getattr(resource, "create", None)
        if not original or getattr(original, "_ours_memory_observed", False):
            return

        def observed(*args, **kwargs):
            started = time.perf_counter()
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                self.store.call(
                    stage,
                    success=False,
                    model=kwargs.get("model"),
                    error=type(exc).__name__,
                    elapsed_seconds=time.perf_counter() - started,
                    measurement="SDK call; provider-internal retries may be hidden",
                )
                raise
            usage = getattr(response, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            self.store.call(
                stage,
                success=True,
                model=kwargs.get("model"),
                served_model=getattr(response, "model", None),
                usage=usage,
                elapsed_seconds=time.perf_counter() - started,
                measurement="SDK call; provider-internal retries may be hidden",
            )
            return response

        observed._ours_memory_observed = True
        resource.create = observed

    @classmethod
    def from_config(cls, config: dict, store=None):
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        from pathlib import Path

        os.environ.setdefault("MEM0_DIR", str(Path("runtime/mem0-home").resolve()))
        from mem0 import Memory

        return cls(Memory.from_config(config), store)

    def add(self, event: Event, policy: str) -> list[MemoryChange]:
        filters = {"user_id": event.user_id, "run_id": policy, "tm_event_id": event.event_id}
        # Recover a native success followed by a sidecar crash without extracting twice.
        restored = self.memory.get_all(filters=filters, **self._all_options())
        restored = restored.get("results", []) if isinstance(restored, dict) else restored
        if restored:
            return [MemoryChange(item["id"], item["memory"], "ADD") for item in restored]
        content = event.text
        if self.native_message_prefix and event.speaker == "tool" and content.lstrip().startswith("["):
            content = self.native_message_prefix + "\n" + content
        messages = [
            {"role": "user" if event.speaker in {"user", "tool"} else "assistant", "content": content}
        ]
        # OSS rejects timestamp=. Its supported prompt= adds custom instructions while
        # retaining the native extraction prompt and native decision about fact count.
        observation = event.observed_at or event.recorded_at
        instruction = (
            f"For THIS imported event, the authoritative Observation Date is "
            f"{observation_date(observation)} ({observation}). "
            "Resolve relative dates in these new messages against this date, not the system date. "
            "Previously cached messages retain their own dates. This is time metadata, not a fact to store."
        )
        custom = getattr(self.memory, "custom_instructions", None)
        if custom:
            instruction = custom + "\n" + instruction
        result = self.memory.add(
            messages,
            user_id=event.user_id,
            run_id=policy,
            prompt=instruction,
            metadata={
                "tm_event_id": event.event_id,
                "tm_team_id": event.team_id,
                "tm_writer_id": event.writer_agent_id,
                "tm_observed_at": observation,
                "tm_source_root_ids": list(event.source_root_ids),
                **({"tm_disclosure_id": event.disclosure_id} if event.disclosure_id else {}),
            },
        )
        if self.store:
            self.store.call(
                "mem0_add",
                success=True,
                event_id=event.event_id,
                usage="see_native_provider_usage; not included in extra-module tokens",
            )
        return [
            MemoryChange(item["id"], item.get("memory", ""), item.get("event", "ADD").upper())
            for item in result.get("results", [])
            if item.get("event", "ADD").upper() != "NONE"
        ]

    def search(self, query: str, caller: Caller, policies: list[str], limit: int) -> list[MemoryChange]:
        results = []
        for policy in policies:
            # Authorization is applied BEFORE native search/re-ranking/any model context.
            response = self.memory.search(
                query, filters={"user_id": caller.user_id, "run_id": policy}, **{self.top_k_name: limit}
            )
            for item in response.get("results", []):
                results.append(MemoryChange(item["id"], item["memory"], score=float(item.get("score") or 0)))
        if self.store:
            self.store.call(
                "mem0_search",
                success=True,
                partitions=len(policies),
                usage="native embedding/retrieval usage separate",
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def delete(self, memory_id: str):
        self.memory.delete(memory_id)

    def find_event_memories(self, event_id: str, policy: str, user_id: str) -> list[str]:
        # Fail closed on truncation rather than silently leaving undiscovered records behind.
        result = self.memory.get_all(
            filters={"user_id": user_id, "run_id": policy, "tm_event_id": event_id}, **self._all_options()
        )
        items = result.get("results", []) if isinstance(result, dict) else result
        if len(items) >= 10000:
            raise RuntimeError("事件原生记忆数量达到删除查询上限，须分批清理")
        return [item["id"] for item in items]

    def _all_options(self):
        size = "top_k" if "top_k" in inspect.signature(self.memory.get_all).parameters else "limit"
        return {size: 10000, "show_expired": True}

    def forget(self, memory_id: str, policy: str, user_id: str):
        # Pinned Mem0 stores original messages and deletion history too. Its public delete()
        # alone leaves those records readable. Erase the affected session cache conservatively.
        if self.memory.get(memory_id) is not None:
            self.memory.delete(memory_id)
        database = self.memory.db
        with database._lock, database.connection:
            database.connection.execute("PRAGMA secure_delete=ON")
            database.connection.execute("DELETE FROM history WHERE memory_id=?", (memory_id,))
        self.clear_event_cache(policy, user_id)

    def clear_event_cache(self, policy: str, user_id: str):
        from mem0.memory.main import _build_session_scope

        database = self.memory.db
        with database._lock, database.connection:
            database.connection.execute("PRAGMA secure_delete=ON")
            database.connection.execute(
                "DELETE FROM messages WHERE session_scope=?",
                (_build_session_scope({"user_id": user_id, "run_id": policy}),),
            )


class FixtureBackend:
    """Local fixture for tests only; does not imitate or replace Mem0 extraction."""

    def __init__(self, changes: dict[str, list[MemoryChange]] | None = None):
        self.changes = changes or {}
        self.items: dict[str, tuple[dict, str, MemoryChange]] = {}
        self.search_policies: list[list[str]] = []
        self.add_count = 0

    def add(self, event: Event, policy: str) -> list[MemoryChange]:
        self.add_count += 1
        changes = self.changes.get(event.event_id, [])
        for change in changes:
            if change.event == "DELETE":
                self.items.pop(change.memory_id, None)
            else:
                self.items[change.memory_id] = (event.to_dict(), policy, change)
        return changes

    def index_card(self, event: Event, policy: str, memory_id: str, text: str) -> None:
        """Direct-card path: no extraction, the supplied text is the indexed unit."""
        self.items[memory_id] = (event.to_dict(), policy, MemoryChange(memory_id, text))

    def search(self, query: str, caller: Caller, policies: list[str], limit: int) -> list[MemoryChange]:
        self.search_policies.append(list(policies))
        return [
            item
            for event, policy, item in self.items.values()
            if event["user_id"] == caller.user_id and policy in policies
        ][:limit]

    def search_shared(self, query: str, caller: Caller, policy: str, limit: int) -> list[MemoryChange]:
        self.search_policies.append([policy])
        return [item for event, group, item in self.items.values()
                if event["team_id"] == caller.team_id and group == policy][:limit]

    def delete(self, memory_id: str):
        self.items.pop(memory_id, None)

    def forget(self, memory_id: str, policy: str, user_id: str):
        self.delete(memory_id)

    def clear_event_cache(self, policy: str, user_id: str):
        # This explicit fixture has no message or history cache.
        pass

    def find_event_memories(self, event_id: str, policy: str, user_id: str) -> list[str]:
        return [
            key
            for key, (event, group, _) in self.items.items()
            if event["event_id"] == event_id and group == policy and event["user_id"] == user_id
        ]


def resolve_mem0_config(config: dict, settings) -> dict:
    """Require an explicit embedding provider; DeepSeek chat does not imply embeddings."""
    import json

    result = json.loads(json.dumps(config))
    llm = result.setdefault("llm", {"provider": "deepseek", "config": {}})
    base_fields = {"deepseek": "deepseek_base_url", "openai": "openai_base_url"}
    if llm["provider"] not in base_fields:
        raise ValidationError("当前统一接口支持deepseek或openai兼容服务；其他原生接口须单独适配")
    cfg = llm.setdefault("config", {})
    for field in base_fields.values():
        cfg.pop(field, None)
    cfg.update(model=settings.model, api_key=os.environ.get(settings.api_key_env))
    cfg[base_fields[llm["provider"]]] = settings.base_url
    if not llm["config"]["api_key"]:
        raise ValidationError(f"未配置{settings.api_key_env}")
    if "embedder" not in result:
        raise ValidationError("须显式配置向量模型，不默认调用另一家付费服务")
    if result["embedder"]["provider"] == "openai":
        cfg = result["embedder"].setdefault("config", {})
        mapping = {
            "model": "TEAM_MEMORY_EMBEDDING_MODEL",
            "api_key": "TEAM_MEMORY_EMBEDDING_API_KEY",
            "openai_base_url": "TEAM_MEMORY_EMBEDDING_BASE_URL",
        }
        for key, env in mapping.items():
            cfg[key] = os.environ.get(env) or cfg.get(key)
            if not cfg[key]:
                raise ValidationError(f"须配置{env}")
        dimension = os.environ.get("TEAM_MEMORY_EMBEDDING_DIMS") or cfg.get("embedding_dims")
        if not dimension:
            raise ValidationError("须配置向量维度TEAM_MEMORY_EMBEDDING_DIMS")
        cfg["embedding_dims"] = int(dimension)
        result.setdefault("vector_store", {}).setdefault("config", {})["embedding_model_dims"] = int(
            dimension
        )
    return result
