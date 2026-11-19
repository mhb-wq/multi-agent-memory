"""Adapter for the official Mem0 open-source package.

The adapter only calls Mem0's public ``add`` / ``search`` / ``reset`` methods.
It does not replace Mem0's extraction prompt, vector store, graph store or
memory update logic.
"""

from __future__ import annotations

from typing import Any

from .base import BaselineAdapter, Message, Passage, search_text


class Mem0Adapter(BaselineAdapter):
    name = "mem0"

    def __init__(
        self,
        memory: Any,
        *,
        group_id: str = "group",
        native_users: bool = False,
    ) -> None:
        """Wrap an already configured ``mem0.Memory`` instance.

        ``native_users=False`` uses one group user, the baseline's shared-group
        mode. ``native_users=True`` writes each message under its author and
        tags it with ``agent_id=group_id``; retrieval filters by the group tag.
        Both modes keep Mem0's own extraction and search in charge.
        """
        self.memory = memory
        self.group_id = group_id
        self.native_users = native_users

    def reset(self) -> None:
        self.memory.reset()

    def ingest(self, message: Message) -> None:
        kwargs: dict[str, Any] = {
            "metadata": {
                **message.provenance(),
                "message_id": message.id,
            }
        }
        if self.native_users:
            kwargs["user_id"] = message.author
            kwargs["agent_id"] = self.group_id
        else:
            kwargs["user_id"] = self.group_id
        self.memory.add([{"role": "user", "content": message.text}], **kwargs)

    def retrieve(self, query: str, asking_user_id: str, top_k: int = 10) -> list[Passage]:
        filters = {"agent_id": self.group_id} if self.native_users else {"user_id": self.group_id}
        result = self.memory.search(
            search_text(query, asking_user_id),
            top_k=top_k,
            filters=filters,
        )
        rows = result.get("results", []) if isinstance(result, dict) else result
        passages: list[Passage] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            text = row.get("memory") or row.get("text") or row.get("data")
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
