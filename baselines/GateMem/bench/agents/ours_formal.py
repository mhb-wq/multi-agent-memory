"""GateMem adapter for the current shared-pool Card–State–Sleep method."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .base import BaseMemoryAgent, Checkpoint, Turn

_OURS_ROOT = Path(__file__).resolve().parents[4] / "ours"
if str(_OURS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_OURS_ROOT / "src"))


class OursFormalAgent(BaseMemoryAgent):
    """One isolated memory instance per episode; no gold enters the method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm_router is None or getattr(self.llm_router, "provider", "stub") == "stub":
            raise ValueError("Ours formal evaluation requires a real configured model, not the gold-aware stub")
        config = getattr(self.llm_router, "config", None)
        self.model = getattr(config, "model", None)
        if not self.model or self.model != os.environ.get("OURS_FORMAL_MODEL"):
            raise ValueError("Ours memory and GateMem answer must use the same explicitly requested model")
        answer_base = str(getattr(config, "api_base", "") or "").rstrip("/")
        memory_base = os.environ.get("OURS_LLM_BASE_URL", "").rstrip("/")
        if not answer_base or answer_base != memory_base:
            raise ValueError("Ours memory and GateMem answer must use the same relay endpoint")
        directory = os.environ.get("OURS_FORMAL_OUTPUT")
        if not directory:
            raise ValueError("OURS_FORMAL_OUTPUT must name the isolated run directory")
        self.root = Path(directory).resolve() / "ours_episodes"
        self.embedding_base = os.environ.get("MEMORY_EMBEDDING_BASE_URL", "")
        if urlparse(self.embedding_base).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ours formal evaluation requires the shared local MiniLM embedding service")
        self.memory = None

    def reset(self, episode: dict) -> None:
        if self.memory is not None:
            raise RuntimeError("The previous Ours episode must be closed before reset")
        from ours_memory.groupmem import ReplayClock, Trace, build_memory, save

        super().reset(episode)
        eid = str(episode.get("episode_id") or "")
        users = sorted({str(p.get("principal_id")) for p in (episode.get("entities") or {}).get("principals", [])
                        if p.get("principal_id")})
        if not eid or not users:
            raise ValueError("GateMem episode needs an ID and registered principals")
        self.users = users
        self.space = "GateMem:" + eid
        self.run = self.root / hashlib.sha256(eid.encode("utf-8")).hexdigest()[:24]
        self.run.mkdir(parents=True, exist_ok=False)
        self.clock = ReplayClock("2026-01-01T00:00:00+00:00")
        self.trace = Trace(self.run / "trace")
        protocol = {"memory_model": self.model,
                    "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimensions": 384,
                                  "base_url": self.embedding_base, "max_seq_length": 256, "normalized": True},
                    "reasoning_policy": "provider_default",
                    "method": {"joint_state_update_enabled": True,
                               "immediate_state_maintenance_enabled": True,
                               "independent_state_read_enabled": True},
                    "access_mode": "shared_pool", "top_k": self.top_k, "episode_id": eid}
        save(self.run / "protocol.json", protocol)
        self.memory = build_memory(self.run, {"space": self.space, "users": users, "access_mode": "shared_pool"},
                                   self.trace, self.clock)
        self.previous = []

    def ingest(self, turn: Turn) -> None:
        from ours_memory.control import Actor

        observed = turn.timestamp
        if observed:
            moment = datetime.fromisoformat(observed.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            observed = moment.isoformat()
            self.clock.value = observed
        metadata = {"role": turn.speaker_role, "msg_node": turn.turn_id}
        if observed:
            metadata["timestamp"] = observed
        else:
            metadata["source_time_unknown"] = "true"
        self.trace.context = "turn_" + hashlib.sha256(turn.turn_id.encode("utf-8")).hexdigest()[:20]
        result = self.memory.ingest_message(
            Actor(turn.speaker_principal_id, "ingestor", self.space, "gatemem"),
            message_id=turn.turn_id, text=turn.text, observed_at=observed,
            metadata=metadata, fallback_category="conversation", prior_context=self.previous[-10:])
        if result.get("status") not in {"done", "accepted_pending"}:
            raise RuntimeError("Ours message ingestion did not reach a recorded state")
        self.previous.append({"id": result["message_id"], "speaker": turn.speaker_principal_id,
                              "role": turn.speaker_role, "text": turn.text, "timestamp": observed})
        self.previous = self.previous[-10:]
        while sum(len(item["text"]) for item in self.previous) > 6000:
            self.previous.pop(0)
        audit = {"turn_id": turn.turn_id, "message_id": result["message_id"], "status": result.get("status"),
                 "processing_status": result.get("processing_status"), "issues": result.get("issues", []),
                 "events": result.get("events", [])}
        with (self.run / "turns.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._json_safe(audit), ensure_ascii=False) + "\n")

    def query(self, checkpoint: Checkpoint) -> dict:
        from ours_memory.control import Actor

        started = time.perf_counter()
        self.trace.context = "checkpoint_" + hashlib.sha256(checkpoint.checkpoint_id.encode("utf-8")).hexdigest()[:20]
        actor = Actor(checkpoint.asker_principal_id, "answerer", self.space, "query")
        bundle = self.memory.read(actor, subject_ids=self.users, query=checkpoint.query_text, purpose="current")
        if not self.memory.validate_bundle(actor, bundle["bundle_id"])["ready"]:
            raise RuntimeError("Ours evidence bundle became invalid before the answer")
        cards = bundle["cards"][:self.top_k]
        retrieved = [{"text": c.get("claim_text", "") + "\nSource: " + str((c.get("source") or {}).get("quote", "")),
                      "record_id": c["id"], "principal_id": c.get("author_id", "")} for c in cards]
        block = self._format_retrieved_memory_block(retrieved)
        block += "\n\nCurrent user states:\n" + json.dumps(bundle.get("user_states", []), ensure_ascii=False)
        # Strip benchmark scoring annotations even when a different router is used.
        visible = replace(checkpoint, query_type="", expected_action="", attack_type=None,
                          judge_spec=None, leak_targets=None, gold_answer_structured=None,
                          gold_refusal_category=None, policy_snapshot=None)
        retrieval_s = time.perf_counter() - started
        output = self._run_llm(checkpoint=visible, retrieved_memory=retrieved,
                               rendered_memory_block_override=block)
        if not self.memory.validate_bundle(actor, bundle["bundle_id"])["ready"]:
            raise RuntimeError("Ours evidence bundle became invalid during the answer")
        output.update(ours_incomplete_message_count=bundle.get("incomplete_message_count", 0),
                      ours_pending_state_count=bundle.get("pending_state_count", 0),
                      ours_pending_subjects=bundle.get("pending_subjects", []),
                      ours_state_count=len(bundle.get("user_states", [])), retrieval_s=retrieval_s)
        return output

    def close(self) -> None:
        if self.memory is None:
            return
        from ours_memory.groupmem import close_memory, save

        try:
            save(self.run / "calls.json", self.memory.store.call_records())
            save(self.run / "pending.json", {"pending_state": self.memory.store.all("pending_state"),
                                              "deferred_messages": [
                                                  {k: row.get(k) for k in ("id", "message_id", "stage", "status")}
                                                  for row in self.memory.store.all("deferred_message_v2")],
                                              "joint_attempt": self.memory.store.all("joint_attempt"),
                                              "joint_batch_recovery": self.memory.store.all("joint_batch_recovery")})
        finally:
            close_memory(self.memory)
            self.memory = None
