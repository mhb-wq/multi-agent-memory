"""Bridge official GateMem runner to an existing BaselineAdapter instance.

The caller supplies a fresh adapter factory per episode and the official LLM
router. This layer does not fabricate authorization/deletion support.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/gatemem/official/code"))

from bench.agents.a_mem import AMemAgent  # noqa: E402
from bench.agents.base import BaseMemoryAgent  # noqa: E402

from ours_memory.baselines.base import Message  # noqa: E402


class GateOfficialAMemAgent(AMemAgent):
    """Use the official algorithm with separate memory and answer settings."""

    def __init__(self, *args, memory_llm_router, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_llm_router = memory_llm_router

    def _analyze_turn_metadata_llm(self, turn):
        answer_router = self.llm_router
        self.llm_router = self.memory_llm_router
        try:
            return super()._analyze_turn_metadata_llm(turn)
        finally:
            self.llm_router = answer_router


class GateBaselineBridge(BaseMemoryAgent):
    def __init__(self, factory, *, llm_router, top_k=10):
        if llm_router is None or getattr(llm_router, "provider", "stub") == "stub":
            raise ValueError("official stub uses hidden labels and is unsuitable for system evaluation")
        super().__init__(top_k=top_k, llm_router=llm_router)
        self.factory = factory
        self.adapter = None

    def reset(self, episode):
        super().reset(episode)
        self.adapter = self.factory(episode)
        self.ingested_count = 0

    def ingest(self, turn):
        self.adapter.ingest(Message(
            id=f"{self.episode['episode_id']}/{turn.turn_id}", text=turn.text,
            author=turn.speaker_principal_id, role=turn.speaker_role,
            timestamp=turn.timestamp or "", channel=self.episode["episode_id"],
        ))
        self.ingested_count += 1
        if self.ingested_count % 10 == 0:
            print(f"GateMem ingested {self.ingested_count} turns", flush=True)

    def query(self, checkpoint):
        passages = self.adapter.retrieve(checkpoint.query_text, checkpoint.asker_principal_id, top_k=self.top_k)
        if len(passages) > self.top_k:
            raise ValueError("backend returned more than top_k passages")
        retrieved = [{"text": p.text, "principal_id": (p.metadata or {}).get("author", ""),
                      "record_id": ",".join(p.source_ids), "score": p.score} for p in passages]
        return self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved)
