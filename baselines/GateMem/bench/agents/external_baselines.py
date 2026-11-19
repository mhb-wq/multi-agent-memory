"""GateMem bridges to the locally collected baseline implementations.

These adapters never read checkpoint labels. Backends still need their own
dependencies and model configuration; failures are deliberately not hidden.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .base import BaseMemoryAgent, Checkpoint, Turn

_ROOT = Path(__file__).resolve().parents[3]


def _source(name: str) -> None:
    path = str(_ROOT / name)
    if not Path(path).is_dir():
        raise RuntimeError(f"Baseline source directory missing: {path}")
    if path not in sys.path:
        sys.path.insert(0, path)


def _row(turn: Turn) -> dict[str, Any]:
    # No record_refs or memory_ops: these can be benchmark annotations.
    return {"msg_node": turn.turn_id, "content": turn.text,
            "author": turn.speaker_principal_id, "role": turn.speaker_role,
            "timestamp": turn.timestamp or "", "_channel": "main"}


def _hits(rows: list[Any]) -> list[dict[str, Any]]:
    output = []
    for hit in rows:
        m = hit.message
        output.append({"record_id": hit.message_id, "turn_id": hit.message_id,
                       "principal_id": m.get("author", ""), "role": m.get("role", ""),
                       "text": m.get("content", ""), "score": float(hit.score)})
    return output


class _IncrementalAgent(BaseMemoryAgent):
    def reset(self, episode: dict[str, Any]) -> None:
        super().reset(episode)
        self.messages: list[dict[str, Any]] = []

    def ingest(self, turn: Turn) -> None:
        self.messages.append(_row(turn))

    def _answer(self, checkpoint: Checkpoint, memories: list[dict[str, Any]],
                started: float, native: str | None = None) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        # The upstream stub consumes gold labels. Strip them even in smoke tests.
        visible_checkpoint = replace(checkpoint, query_type="", expected_action="",
                                     attack_type=None, judge_spec=None, leak_targets=None,
                                     gold_answer_structured=None, gold_refusal_category=None,
                                     policy_snapshot=None)
        result = self._run_llm(checkpoint=visible_checkpoint, retrieved_memory=memories,
                               rendered_memory_block_override=native)
        result["retrieval_s"] = elapsed
        return result


class BM25Agent(_IncrementalAgent):
    def query(self, checkpoint: Checkpoint) -> dict[str, Any]:
        started = time.perf_counter()
        _source("DG-Mem")
        from src.bm25 import BM25Retriever
        hits = BM25Retriever(self.messages).retrieve(checkpoint.query_text, self.top_k) if self.messages else []
        return self._answer(checkpoint, _hits(hits), started)


class DGMemAgent(_IncrementalAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._retriever = None
        self._dirty = True
        self._components = None

    def reset(self, episode: dict[str, Any]) -> None:
        super().reset(episode)
        self._retriever = None
        self._dirty = True

    def ingest(self, turn: Turn) -> None:
        super().ingest(turn)
        self._dirty = True

    def _build(self) -> None:
        _source("DG-Mem")
        from src.embeddings import HashingEmbedder, SentenceTransformerEmbedder
        from src.entities import RegexEntityExtractor, SpacyEntityExtractor
        from src.memory_graph import MemoryGraph
        from src.rerankers import build_reranker
        from src.user_context_graph import DualGraphRetriever, UserContextGraph
        mode = os.environ.get("GATEMEM_DGMEM_MODE", "reference")
        if self._components is None:
            if mode == "smoke":
                self._components = HashingEmbedder(), RegexEntityExtractor(), None
            elif mode == "reference":
                device = os.environ.get("GATEMEM_DGMEM_DEVICE", "auto")
                self._components = (
                    SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device=device),
                    SpacyEntityExtractor("en_core_web_sm"),
                    build_reranker("sentence_transformer", os.environ.get(
                        "GATEMEM_DGMEM_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2"), device=device),
                )
            else:
                raise ValueError("GATEMEM_DGMEM_MODE must be reference or smoke")
        embedder, entities, reranker = self._components
        content = MemoryGraph(embedder, entities).build(self.messages)
        user = UserContextGraph(reranker=reranker).build(self.messages)
        self._retriever = DualGraphRetriever(content, user)
        self._dirty = False

    def query(self, checkpoint: Checkpoint) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.messages:
            return self._answer(checkpoint, [], started)
        if self._dirty:
            self._build()
        hits = self._retriever.retrieve(checkpoint.query_text, top_k=self.top_k,
                                        asking_user_id=checkpoint.asker_principal_id)
        out = self._answer(checkpoint, _hits(hits), started)
        out["dgmem_mode"] = os.environ.get("GATEMEM_DGMEM_MODE", "reference")
        return out


class MemGPTAgent(_IncrementalAgent):
    """Keep the upstream MemGPT-style state across checkpoint queries."""

    def reset(self, episode: dict[str, Any]) -> None:
        super().reset(episode)
        _source("MemSyco-Bench")
        from baselines import memgpt
        from baselines.config_loader import build_baseline_eval_config
        self.impl = memgpt
        self.cfg = build_baseline_eval_config("MemGPT", top_k=self.top_k)
        self.state = memgpt._initial_state()
        self.processed = 0
        self.clients = None
        self.failed = False

    def query(self, checkpoint: Checkpoint) -> dict[str, Any]:
        started = time.perf_counter()
        if self.failed:
            raise RuntimeError("MemGPT state may be partially updated; reset the episode before retrying")
        if not self.messages:
            return self._answer(checkpoint, [], started)
        m = self.impl
        try:
            if self.clients is None:
                self.clients = m._make_clients(self.cfg)
            chat, embed = self.clients
            pending = self.messages[self.processed:]
            batch_size = m._ingest_batch_size()
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                turns = [{"role": "assistant" if item["role"] == "agent" else "user",
                          "content": json.dumps(item, ensure_ascii=False)} for item in batch]
                prompt = ("Here is part of a prior conversation between a user and an assistant. Read it and store "
                          "any important, durable information about the user into your memory: concise facts go into "
                          "the 'human' core memory block via core_memory_append (or core_memory_replace to update an "
                          "existing fact); longer evidence or verbatim passages go into archival memory via "
                          "archival_memory_insert. Do not ask me questions; briefly confirm once you have stored what "
                          "matters.\n\n" + m._transcript(turns))
                m._run_agent_loop(chat, m._resolve_llm_model(self.cfg), prompt, self.state, embed,
                                  m._resolve_embedding_model(self.cfg), m._resolve_embedding_dims(self.cfg), m._max_steps())
                self.processed += len(batch)
            index = m._MemoryIndex(user_id=checkpoint.episode_id, save_dir="", core_blocks=self.state.core_blocks,
                                   archival_docs=self.state.archival_docs, archival_vectors=self.state.archival_vectors)
            hits = m._search_archival(index, checkpoint.query_text, embed, m._resolve_embedding_model(self.cfg),
                                      m._resolve_embedding_dims(self.cfg), self.top_k)
            memories = [{"record_id": f"core:{k}", "text": v} for k, v in self.state.core_blocks.items() if v]
            memories.extend({"record_id": f"archive:{i}", "text": str(h.get("used_content") or h.get("content") or "")}
                            for i, h in enumerate(hits))
            native = m._build_context_text(self.state.core_blocks, hits)
        except Exception:
            self.failed = True
            raise
        out = self._answer(checkpoint, memories, started, native=native)
        out["baseline_state_mode"] = "incremental_memsyco_reimplementation"
        return out


class LightMemAgent(_IncrementalAgent):
    """Keep one native LightMemory store per episode, writing new turns once."""

    def close(self) -> None:
        if getattr(self, "memory", None) is not None:
            for name in ("embedding_retriever", "summary_retriever"):
                client = getattr(getattr(self.memory, name, None), "client", None)
                if client is not None:
                    client.close()
            self.memory = None
        if getattr(self, "_temporary", None) is not None:
            self._temporary.cleanup()
            self._temporary = None

    def reset(self, episode: dict[str, Any]) -> None:
        self.close()
        super().reset(episode)
        _source("MemSyco-Bench")
        from baselines import lightmem
        from baselines.config_loader import build_baseline_eval_config
        self.impl = lightmem
        self.cfg = build_baseline_eval_config("LightMem", top_k=self.top_k)
        self.memory = None
        self._temporary = tempfile.TemporaryDirectory(prefix="gatemem_lightmem_")
        self.processed = 0
        self.failed = False

    def query(self, checkpoint: Checkpoint) -> dict[str, Any]:
        started = time.perf_counter()
        if self.failed:
            raise RuntimeError("LightMem state may be partially updated; reset the episode before retrying")
        if not self.messages:
            return self._answer(checkpoint, [], started)
        try:
            m = self.impl
            if self.memory is None:
                m._prepare_import_path()
                from lightmem.memory.lightmem import LightMemory
                cfg = m._load_native_lightmem_config(self.cfg, user_id="episode", save_dir=self._temporary.name)
                if cfg.get("graph_mem"):
                    raise NotImplementedError("LightMem graph_mem is not implemented in upstream")
                self.memory = LightMemory.from_config(cfg)
            pending = self.messages[self.processed:]
            if pending:
                for index, item in enumerate(pending, self.processed):
                    role = "assistant" if item["role"] == "agent" else "user"
                    timestamp = m._native_lightmem_timestamp(index)
                    content = json.dumps(item, ensure_ascii=False)
                    batch = [{"role": role, "content": content, "time_stamp": timestamp}]
                    if role == "user":
                        batch.append({"role": "assistant", "content": "", "time_stamp": timestamp})
                    self.memory.add_memory(messages=batch, force_segment=True, force_extract=True)
                    self.processed += 1
                self.memory.construct_update_queue_all_entries(
                    top_k=int(os.getenv("LIGHTMEM_FULL_UPDATE_TOP_K", "20")),
                    keep_top_n=int(os.getenv("LIGHTMEM_FULL_UPDATE_KEEP_TOP_N", "10")),
                    max_workers=int(os.getenv("LIGHTMEM_FULL_UPDATE_QUEUE_WORKERS", "8")))
                self.memory.offline_update_all_entries(
                    score_threshold=float(os.getenv("LIGHTMEM_FULL_UPDATE_SCORE_THRESHOLD", "0.8")),
                    max_workers=int(os.getenv("LIGHTMEM_FULL_OFFLINE_UPDATE_WORKERS", "5")))
            texts = self.memory.retrieve(checkpoint.query_text, limit=self.top_k)
            memories = [{"record_id": f"lightmem:{i}", "text": str(value)} for i, value in enumerate(texts)]
        except Exception:
            self.failed = True
            raise
        out = self._answer(checkpoint, memories, started)
        out["baseline_state_mode"] = "incremental_lightmem_checkpoint_update"
        return out


class MemoryBankAgent(_IncrementalAgent):
    """Incremental multi-party adaptation of the MemSyco MemoryBank baseline."""

    def reset(self, episode: dict[str, Any]) -> None:
        super().reset(episode)
        _source("MemSyco-Bench")
        from baselines import memorybank
        from baselines.config_loader import build_baseline_eval_config
        self.impl = memorybank
        self.cfg = build_baseline_eval_config("MemoryBank", top_k=self.top_k)
        self.index = None
        self.clients = None
        self.processed = 0
        self.failed = False
        self.history_summaries = []
        self.personality_summaries = []

    def query(self, checkpoint: Checkpoint) -> dict[str, Any]:
        started = time.perf_counter()
        if self.failed:
            raise RuntimeError("MemoryBank state may be partially updated; reset the episode before retrying")
        if not self.messages:
            return self._answer(checkpoint, [], started)
        import numpy as np
        m = self.impl
        try:
            if self.clients is None:
                self.clients = m._make_clients(self.cfg)
            pending = self.messages[self.processed:]
            if pending:
                chat, embed = self.clients
                messages = [{"role": "assistant" if item["role"] == "agent" else "user",
                             "content": json.dumps(item, ensure_ascii=False)} for item in pending]
                pairs = m._build_dialog_pairs(messages)
                language = m._language()
                memory = m._build_user_memory(pairs, chat, m._resolve_llm_model(self.cfg), language)
                docs = m._build_documents(memory, language)
                vectors = m._embed_texts(embed, m._resolve_embedding_model(self.cfg),
                                         [d["page_content"] for d in docs], m._resolve_embedding_dims(self.cfg))
                if self.index is None:
                    self.index = m._MemoryIndex(user_id=checkpoint.episode_id, save_dir="", docs=[],
                                                vectors=vectors[:0], overall_history="", overall_personality="")
                self.index.docs.extend(docs)
                self.index.vectors = np.concatenate([self.index.vectors, vectors], axis=0)
                date = m.DEFAULT_DATE
                if memory.get("summary", {}).get(date):
                    self.history_summaries.append((f"batch-{len(self.history_summaries) + 1}", memory["summary"][date]))
                if memory.get("personality", {}).get(date):
                    self.personality_summaries.append((f"batch-{len(self.personality_summaries) + 1}", memory["personality"][date]))
                if len(self.history_summaries) > 1:
                    _, _, overall_prompt, personality_prompt = m._import_memorybank_prompts()
                    model = m._resolve_llm_model(self.cfg)
                    self.index.overall_history = m._chat_summarize(chat, model, overall_prompt(self.history_summaries, language=language), language)
                    self.index.overall_personality = m._chat_summarize(chat, model, personality_prompt(self.personality_summaries, language=language), language)
                else:
                    self.index.overall_history = str(memory.get("overall_history") or "")
                    self.index.overall_personality = str(memory.get("overall_personality") or "")
                self.processed += len(pending)
            retrieved = m._retrieve(self.index, checkpoint.query_text, self.cfg, self.top_k)
            native = m._format_context(self.index, retrieved, m._language())
            memories = [{"record_id": f"memorybank:{i}", "text": str(item.get("used_content") or item.get("content") or "")}
                        for i, item in enumerate(retrieved)]
        except Exception:
            self.failed = True
            raise
        out = self._answer(checkpoint, memories, started, native=native)
        out["baseline_state_mode"] = "incremental_memorybank_multiparticipant_adaptation"
        return out


class EGMemoryAgent(_IncrementalAgent):
    def close(self) -> None:
        if getattr(self, "memory", None) is not None:
            self.memory.close()
            self.memory = None
        if getattr(self, "_temporary", None) is not None:
            self._temporary.cleanup()
            self._temporary = None

    def reset(self, episode: dict[str, Any]) -> None:
        super().reset(episode)
        # The file must specify real model/provider settings. Never substitute
        # a pretend memory writer when configuration is missing.
        path = os.environ.get("GATEMEM_EGMEMORY_CONFIG")
        if not path:
            raise RuntimeError("Set GATEMEM_EGMEMORY_CONFIG to a GroupMemory JSON config containing llm settings")
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("llm"):
            raise ValueError("EGMemory config must be an object with a configured llm")
        _source("EGMemory")
        from groupmem import GroupMemory
        self.close()
        self._temporary = tempfile.TemporaryDirectory(prefix="gatemem_egmemory_")
        # Isolate each episode, regardless of IDs or prior runs.
        config["name"] = "episode"
        if config.get("record_store", {"backend": "jsonl"}).get("backend") != "jsonl":
            raise ValueError("EGMemory adapter currently requires isolated jsonl record_store")
        config["record_store"] = {"backend": "jsonl", "root": self._temporary.name}
        self.memory = GroupMemory.from_config(config)
        # Smoke-only limits are explicitly configured by the runner; the
        # upstream default remains unchanged for normal runs.
        timeout = os.environ.get("GATEMEM_EGMEMORY_HTTP_TIMEOUT")
        if timeout:
            import groupmem.llms.openai_compat as transport
            transport.HTTP_TIMEOUT = float(timeout)
        cap = os.environ.get("GATEMEM_EGMEMORY_MAX_TOKENS")
        if cap:
            original = self.memory.llm.call_tools
            def bounded_tools(messages, tools=None, max_tokens=0, **kwargs):
                return original(messages, tools=tools, max_tokens=max_tokens or int(cap), **kwargs)
            self.memory.llm.call_tools = bounded_tools

    def ingest(self, turn: Turn) -> None:
        self.memory.add([{"msg_id": turn.turn_id, "author": turn.speaker_principal_id,
                          "role": turn.speaker_role, "timestamp": turn.timestamp or "",
                          "content": turn.text}], channel="main")
        super().ingest(turn)

    def query(self, checkpoint: Checkpoint) -> dict[str, Any]:
        started = time.perf_counter()
        mode = os.environ.get("GATEMEM_EGMEMORY_READ_MODE", "native")
        if mode == "native":
            from bench.agents.egmemory_budget import bounded_read_tools
            from bench.domains import detect_domain_from_episode, get_domain_label, get_query_policy_block
            template = self._load_query_template()
            system, _ = self._split_system_user(template)
            domain = detect_domain_from_episode(self.episode)
            policy = system.format(domain_label=get_domain_label(domain),
                                   global_access_policy_block=get_query_policy_block(domain),
                                   asker_principal_id=checkpoint.asker_principal_id,
                                   asker_role=checkpoint.asker_role,
                                   relationship_facts_block=self._format_relationship_facts(checkpoint),
                                   retrieved_memory_block="", query_text=checkpoint.query_text)
            context = (f"{policy}\n\nRequester principal_id: {checkpoint.asker_principal_id}\n"
                       f"Requester role: {checkpoint.asker_role}\n"
                       f"Requester-relevant relationships:\n{self._format_relationship_facts(checkpoint)}")
            read_side = self.memory.load()
            read_side["tools"] = bounded_read_tools(read_side["tools"], self.top_k)
            budget_tools = read_side["tools"]
            _, trace = self.memory.answer(checkpoint.query_text, asking_user=checkpoint.asker_principal_id,
                                          user_context=context, with_trace=True)
            # The native loop is used only to select evidence. Do not pass its
            # provisional answer, reasoning or tool instructions to the final
            # model: the latter must receive the same GateMem answer template
            # and memory rendering as the other baseline adapters.
            ids = budget_tools.evidence_order
            if len(ids) != len(budget_tools._seen) or set(ids) != budget_tools._seen:
                raise RuntimeError("EGMemory evidence ordering does not match retrieved nodes")
            memories = []
            for mid in ids:
                mem = budget_tools.store.memories[mid]
                memories.append({"record_id": mid, "principal_id": mem.author,
                                 "role": mem.role, "text": budget_tools._passage(mem)})
            out = self._answer(checkpoint, memories, started)
            out.update(egmemory_tool_trace=self._json_safe(trace), egmemory_read_mode="native",
                       egmemory_answer_mode="gatemem_unified", egmemory_evidence_limit=self.top_k,
                       egmemory_evidence_count=len(ids), egmemory_evidence_ids=list(ids))
            return out
        if mode != "search":
            raise ValueError("GATEMEM_EGMEMORY_READ_MODE must be native or search")
        hits = self.memory.search(checkpoint.query_text, k=self.top_k) if self.messages else []
        memories = [{"record_id": h.get("memory_id"), "principal_id": h.get("author"),
                     "text": h.get("text", "")} for h in hits]
        out = self._answer(checkpoint, memories, started)
        out["egmemory_read_mode"] = "search_then_gatemem_answer"
        return out
