"""Pinned, native MemOS general_text adapter. No replacement extraction/retrieval algorithm.

Shared-pool mode uses one physical Qdrant directory and MemCube per space.
Governed mode uses one per immutable access cohort. Policy is
checked BEFORE selecting cubes, not after results have entered a model prompt.
The SQLite outbox bridges native Qdrant writes and sidecar transactions.
"""

from __future__ import annotations

import os
import time
import uuid
from copy import deepcopy
from pathlib import Path

from .models import Caller, Event, MemoryChange, ValidationError, canonical, stable_id


class MemOSBackend:
    def __init__(self, config: dict, store, cube_factory=None):
        self.config, self.store = deepcopy(config), store
        self.root = Path(config["directory"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MEMOS_BASE_PATH", str(self.root))
        # Avoid retaining forgotten plaintext in the optional provider embedding cache.
        os.environ["MEMOS_EMBEDDING_OPTIMIZATION_ENABLED"] = "false"
        self.cubes = {}
        self.cube_factory = cube_factory

    def _cube(self, policy: str, user_id: str):
        if policy.startswith("shared_pool_"):
            # A stable public owner, independent of who writes/queries first.
            user_id = policy
        if policy in self.cubes:
            return self.cubes[policy]
        if self.cube_factory:
            cube = self.cube_factory(policy, user_id)
        else:
            from memos import GeneralMemCube, GeneralMemCubeConfig

            text_config = deepcopy(self.config["text_memory"])
            if text_config.get("backend") != "general_text":
                raise ValidationError("本适配器固定 general_text；不能静默替换为 TreeMemory")
            vector = text_config["config"]["vector_db"]
            if vector["backend"] != "qdrant":
                raise ValidationError("当前已验收存储路径仅支持 Qdrant")
            vector["config"].update(collection_name=policy, path=str(self.root / policy))
            # A local database per policy prevents unrestricted upstream searches crossing ACLs.
            for key in ("host", "port", "url", "api_key"):
                vector["config"].pop(key, None)
            cube = GeneralMemCube(
                GeneralMemCubeConfig(
                    user_id=user_id,
                    cube_id=policy,
                    text_mem=text_config,
                    act_mem={"backend": "uninitialized", "config": {}},
                    para_mem={"backend": "uninitialized", "config": {}},
                    pref_mem={"backend": "uninitialized", "config": {}},
                )
            )
            # Log API usage, never prompt text or credentials.
            self._observe(cube.text_mem.extractor_llm.client.chat.completions, "memos_extract")
            embedder = cube.text_mem.embedder
            client = getattr(embedder, "client", None)
            if client:
                self._observe(client.embeddings, "memos_embed")
        self.cubes[policy] = cube
        return cube

    def _observe(self, resource, stage):
        original = resource.create
        if getattr(original, "_ours_observed", False):
            return

        def create(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                self.store.call(stage, success=False, error=type(exc).__name__)
                raise
            usage = getattr(result, "usage", None)
            self.store.call(
                stage,
                success=True,
                model=kwargs.get("model"),
                served_model=getattr(result, "model", None),
                usage=usage.model_dump() if usage else None,
                elapsed_seconds=time.perf_counter() - start,
            )
            return result

        create._ours_observed = True
        resource.create = create

    def add(self, event: Event, policy: str) -> list[MemoryChange]:
        key = stable_id("native", event.user_id, policy, event.event_id)
        journal = self.store.get("native_event", key)
        memory = self._cube(policy, event.user_id).text_mem
        if journal is None:
            role = "user" if event.speaker == "user" else "assistant"
            # The native extractor selects fact count/content. Identity, permissions,
            # exact source and trusted observation time are supplied in the Card stage.
            items = memory.extract([{"role": role, "content": event.text}])
            rows = []
            for index, item in enumerate(items):
                row = item.model_dump()
                if not isinstance(row.get("memory"), str) or not row["memory"].strip():
                    raise ValidationError("MemOS 返回了空记忆")
                row["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, canonical([key, index, row["memory"]])))
                row.setdefault("metadata", {}).update(
                    ours_event_id=event.event_id,
                    ours_policy=policy,
                    ours_user_id=event.user_id,
                    ours_source_roots=list(event.source_root_ids or (event.event_id,)),
                    ours_recorded_at=event.recorded_at,
                )
                rows.append(row)
            journal = {
                "id": key,
                "policy": policy,
                "user_id": event.user_id,
                "event_id": event.event_id,
                "rows": rows,
                "status": "prepared",
            }
            self.store.put("native_event", key, journal)
        if journal["status"] != "committed":
            if journal["rows"]:
                memory.add(journal["rows"])  # native embedding + Qdrant upsert; UUIDs make retries idempotent
            journal["status"] = "committed"
            self.store.put("native_event", key, journal)
        return [MemoryChange(row["id"], row["memory"]) for row in journal["rows"]]

    def index_card(self, event: Event, policy: str, memory_id: str, text: str) -> None:
        """Index a card's own text (claim + exact source quote) without extraction.

        Used by ``card_mode=direct``: the model already produced the memory unit, so the
        vector store must hold that unit (and the original wording), not a paraphrase.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("直接卡片索引文本不能为空")
        key = stable_id("native", event.user_id, policy, event.event_id)
        journal = self.store.get("native_event", key)
        if journal is None:
            journal = {
                "id": key,
                "policy": policy,
                "user_id": event.user_id,
                "event_id": event.event_id,
                "rows": [],
                "status": "prepared",
            }
        if not any(row.get("id") == memory_id for row in journal["rows"]):
            row = {
                "id": memory_id,
                "memory": text,
                "metadata": {
                    "ours_event_id": event.event_id,
                    "ours_policy": policy,
                    "ours_user_id": event.user_id,
                    "ours_card_id": memory_id,
                    "ours_source_roots": list(event.source_root_ids or (event.event_id,)),
                    "ours_recorded_at": event.recorded_at,
                },
            }
            journal["rows"].append(row)
            self.store.put("native_event", key, journal)
            self._cube(policy, event.user_id).text_mem.add([row])
        journal["status"] = "committed"
        self.store.put("native_event", key, journal)

    def search(self, query: str, caller: Caller, policies: list[str], limit: int) -> list[MemoryChange]:
        hits = []
        for policy in policies:
            # GeneralTextMemory does not expose scores: use explicit reciprocal-rank
            # interleaving across authorized cubes, not invented cosine scores.
            items = self._cube(policy, caller.user_id).text_mem.search(query, top_k=limit)
            hits.extend(
                MemoryChange(item.id, item.memory, score=1 / (60 + rank))
                for rank, item in enumerate(items, 1)
            )
        self.store.call("memos_search", success=True, authorized_cubes=len(policies))
        return sorted(hits, key=lambda item: (-item.score, item.memory_id))[:limit]

    def search_shared(self, query: str, caller: Caller, policy: str, limit: int) -> list[MemoryChange]:
        """One native search over the space's public index, like the MemOS baseline.

        GeneralTextMemory returns ordered items without similarity scores. Do
        not fabricate cross-cube scores or reorder these native results.
        """
        if policy != stable_id("shared_pool", caller.team_id):
            raise ValidationError("公共记忆池必须匹配当前空间")
        items = self._cube(policy, caller.team_id).text_mem.search(query, top_k=limit)
        self.store.call("memos_search", success=True, public_cubes=1, ranking="native")
        return [MemoryChange(item.id, item.memory) for item in items[:limit]]

    def state_embedding_identity(self, space: str) -> str:
        config = self.config["text_memory"]["config"]["embedder"]
        def public(value):
            if isinstance(value, dict):
                return {k: public(v) for k, v in value.items()
                        if not any(secret in k.lower() for secret in ("key", "token", "password", "secret"))}
            if isinstance(value, list):
                return [public(v) for v in value]
            return value
        return stable_id("state_embedding", canonical(public(config)))

    def embed_state_texts(self, space: str, texts: list[str]) -> list[list[float]]:
        policy = stable_id("shared_pool", space)
        return self._cube(policy, space).text_mem.embedder.embed(texts)

    def find_event_memories(self, event_id, policy, user_id):
        journal = self.store.get("native_event", stable_id("native", user_id, policy, event_id))
        return [row["id"] for row in journal["rows"]] if journal else []

    def forget(self, memory_id, policy, user_id):
        self._cube(policy, user_id).text_mem.delete([memory_id])
        for row in self.store.all("native_event"):
            if row["policy"] != policy or row["user_id"] != user_id:
                continue
            kept = [item for item in row["rows"] if item["id"] != memory_id]
            if len(kept) != len(row["rows"]):
                if kept:
                    row["rows"] = kept
                    self.store.put("native_event", row["id"], row)
                else:
                    self.store.remove("native_event", row["id"])

    def delete(self, memory_id):
        for row in self.store.all("native_event"):
            if any(item["id"] == memory_id for item in row["rows"]):
                self.forget(memory_id, row["policy"], row["user_id"])
                return

    def clear_event_cache(self, policy, user_id):
        # This native path has no conversation cache; purge empty recovery journals.
        for row in self.store.all("native_event"):
            if row["policy"] == policy and row["user_id"] == user_id and not row["rows"]:
                self.store.remove("native_event", row["id"])

    def close(self):
        for cube in self.cubes.values():
            client = getattr(getattr(cube.text_mem, "vector_db", None), "client", None)
            if client:
                client.close()
        self.cubes.clear()
