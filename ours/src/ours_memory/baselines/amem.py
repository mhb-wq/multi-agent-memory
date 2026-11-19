"""Adapter for the official A-MEM open-source package.

A-MEM owns note generation, keyword/context/tag analysis, linking and its
ChromaDB hybrid retrieval. The adapter only calls ``add_note`` and
``search_agentic``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .base import BaselineAdapter, Message, Passage, search_text


def _compact_time(value: str) -> str:
    """A-MEM documents YYYYMMDDHHMM; fall back to the raw string if malformed."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%d%H%M")
    except (TypeError, ValueError):
        return value


class AMEMAdapter(BaselineAdapter):
    name = "a-mem"

    def __init__(self, system: Any, *, factory: Callable[[], Any] | None = None) -> None:
        """Wrap an already configured ``AgenticMemorySystem``.

        A-MEM has no public reset API, so pass a factory if the runner needs to
        recreate a clean store for the next domain.
        """
        self.system = system
        self.factory = factory

    def reset(self) -> None:
        if self.factory is None:
            raise NotImplementedError(
                "A-MEM adapter needs a factory to reset; create a new run directory instead"
            )
        self.system = self.factory()

    def ingest(self, message: Message) -> None:
        self.system.add_note(
            message.text,
            time=_compact_time(message.timestamp),
            category=message.topic or "General",
            tags=[tag for tag in (message.channel, message.author) if tag],
        )

    def count(self) -> int:
        return len(getattr(self.system, "memories", {}))

    def retrieve(self, query: str, asking_user_id: str, top_k: int = 10) -> list[Passage]:
        text = search_text(query, asking_user_id)
        if hasattr(self.system, "find_related_memories_raw"):
            # Paper reproduction evaluation uses this raw string, including the
            # linked neighbours. Keep it as one upstream retrieval unit.
            raw = self.system.find_related_memories_raw(text, k=top_k)
            return (
                [Passage(text=raw, metadata={"retriever": "a-mem", "format": "raw"})]
                if isinstance(raw, str) and raw.strip()
                else []
            )
        if hasattr(self.system, "search_agentic"):
            rows = self.system.search_agentic(text, k=top_k)
        elif hasattr(self.system, "find_related_memories"):
            # Compatibility path when only the index-returning API is present.
            _, indices = self.system.find_related_memories(text, k=top_k)
            memories = list(self.system.memories.values())
            rows = [
                {
                    "id": memories[index].id,
                    "content": memories[index].content,
                    "metadata": {
                        "timestamp": memories[index].timestamp,
                        "context": memories[index].context,
                        "tags": memories[index].tags,
                    },
                }
                for index in indices
                if 0 <= index < len(memories)
            ]
        else:
            raise TypeError(
                "A-MEM system exposes none of find_related_memories_raw/search_agentic/find_related_memories"
            )
        passages: list[Passage] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            text = row.get("content") or row.get("memory")
            if not isinstance(text, str) or not text.strip():
                continue
            identifier = row.get("id")
            passages.append(
                Passage(
                    text=text,
                    source_ids=(str(identifier),) if identifier else (),
                    metadata=row.get("metadata") or {},
                    score=row.get("score"),
                )
            )
        return passages[:top_k]
