"""Adapter for the official MemOS high-level ``MOS`` API.

This is the stock MemOS control, not the Card--State--Sleep layer. The adapter
uses MemOS's public ``add`` / ``search`` methods; it does not add cards, states,
permissions or sleep consolidation.
"""

from __future__ import annotations

from typing import Any

from .base import BaselineAdapter, Message, Passage, search_text


class MemOSAdapter(BaselineAdapter):
    name = "memos"

    def __init__(self, mos: Any, *, user_id: str = "group") -> None:
        """Wrap an already configured ``memos.mem_os.main.MOS`` instance."""
        self.mos = mos
        self.user_id = user_id

    def reset(self) -> None:
        # MOS has no public reset across all cubes. The runner gives each
        # domain/run its own cube and directory.
        raise NotImplementedError("Create a fresh MemOS cube for each run")

    def ingest(self, message: Message) -> None:
        self.mos.add(
            messages=[{"role": "user", "content": message.text}],
            user_id=self.user_id,
        )

    def retrieve(self, query: str, asking_user_id: str, top_k: int = 10) -> list[Passage]:
        result = self.mos.search(
            search_text(query, asking_user_id),
            user_id=self.user_id,
            top_k=top_k,
            mode="fast",
        )
        passages: list[Passage] = []
        for block in (result or {}).get("text_mem", []):
            if not isinstance(block, dict):
                continue
            for item in block.get("memories", []):
                text = getattr(item, "memory", None)
                metadata = getattr(item, "metadata", None)
                identifier = getattr(item, "id", None)
                if text is None and isinstance(item, dict):
                    text, metadata, identifier = item.get("memory"), item.get("metadata"), item.get("id")
                if not isinstance(text, str) or not text.strip():
                    continue
                passages.append(
                    Passage(
                        text=text,
                        source_ids=(str(identifier),) if identifier else (),
                        metadata=metadata if isinstance(metadata, dict) else {},
                    )
                )
        return passages[:top_k]


class MemOSNativeAdapter(BaselineAdapter):
    """Stock MemOS ``GeneralTextMemory`` (extract/add/search), no Ours card layer."""

    name = "memos"

    def __init__(self, text_mem: Any) -> None:
        self.text_mem = text_mem

    def reset(self) -> None:
        self.text_mem.delete_all()

    def ingest(self, message: Message) -> None:
        items = self.text_mem.extract([{"role": "user", "content": message.text}])
        if items:
            self.text_mem.add(items)

    def count(self) -> int:
        return len(self.text_mem.get_all())

    def retrieve(self, query: str, asking_user_id: str, top_k: int = 10) -> list[Passage]:
        items = self.text_mem.search(search_text(query, asking_user_id), top_k=top_k)
        passages: list[Passage] = []
        for item in items or []:
            text = getattr(item, "memory", None)
            metadata = getattr(item, "metadata", None)
            identifier = getattr(item, "id", None)
            if not isinstance(text, str) or not text.strip():
                continue
            passages.append(
                Passage(
                    text=text,
                    source_ids=(str(identifier),) if identifier else (),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )
        return passages[:top_k]
