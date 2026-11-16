"""Thin adapters for the official baseline implementations.

No memory algorithm is implemented here. Each adapter translates the benchmark
message/query stream into the upstream project's public API and translates its
public output back into passages. Extraction, storage, consolidation, indexing
and retrieval stay inside the upstream project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Message:
    """One source message, preserving the metadata the dataset provides."""

    id: str
    text: str
    author: str
    timestamp: str
    channel: str = ""
    role: str = ""
    topic: str = ""
    phase_name: str = ""
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "author": self.author,
                "role": self.role,
                "channel": self.channel,
                "phase_name": self.phase_name,
                "topic": self.topic,
                "timestamp": self.timestamp,
                "reply_to": self.reply_to,
                "msg_node": self.id.rsplit("/", 1)[-1],
            }.items()
            if value not in (None, "")
        }


@dataclass(frozen=True)
class Passage:
    """One retrieval unit returned by a baseline."""

    text: str
    source_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class BaselineAdapter(ABC):
    """The only interface the evaluator knows.

    ``ingest`` receives messages in replay order. ``retrieve`` receives the
    original question and the asking user; each adapter decides how to use the
    asker, exactly as the upstream system would.
    """

    name = "baseline"

    @abstractmethod
    def reset(self) -> None:
        """Clear this run's store. The runner uses a fresh directory per run."""

    @abstractmethod
    def ingest(self, message: Message) -> None:
        """Store one source message through the upstream public API."""

    @abstractmethod
    def retrieve(self, query: str, asking_user_id: str, top_k: int = 10) -> list[Passage]:
        """Return up to ``top_k`` upstream retrieval units, best first."""

    def ingest_many(self, messages: Iterable[Message]) -> None:
        for message in messages:
            self.ingest(message)


def search_text(query: str, asking_user_id: str) -> str:
    """The official RAG baselines prepend the asker to the retrieval query."""
    asker = (asking_user_id or "").strip()
    return f"{asker} {query}".strip() if asker else query


def format_passage(passage: Passage) -> str:
    """The official passage header shape, for a memory unit or a message."""
    metadata = passage.metadata or {}
    tags: list[str] = []
    author = metadata.get("author") or metadata.get("user_id")
    if author:
        tags.append(f"user={author}")
    for source, label in (
        ("role", "speaker_role"),
        ("speaker_role", "speaker_role"),
        ("channel", "channel"),
        ("phase_name", "phase_name"),
        ("topic", "topic"),
        ("timestamp", "timestamp"),
        ("reply_to", "reply_to"),
        ("msg_node", "msg_node"),
    ):
        value = metadata.get(source)
        if value not in (None, "") and f"{label}={value}" not in tags:
            tags.append(f"{label}={value}")
    if not tags and passage.source_ids:
        tags.append("memory=" + ",".join(passage.source_ids))
    header = f"[{' / '.join(tags)}]" if tags else "[memory]"
    return f"{header}\n{passage.text}".strip()


