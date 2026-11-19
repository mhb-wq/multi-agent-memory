"""Optional second-stage semantic rerankers."""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

import numpy as np


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
QUERY_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("owner", re.compile(
        r"\b(?:who|whose|owner|ownership|owned by|lead|point person|accountable|assigned to)\b",
        re.IGNORECASE,
    )),
    ("deadline", re.compile(
        r"\b(?:when|deadline|due|date|time|eod|end of day|today|tomorrow|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    )),
    ("status", re.compile(
        r"\b(?:status|state|freeze|frozen|lock|locked|approved|approval|pending|"
        r"finalized|confirmed|open|closed|paused|blocked)\b",
        re.IGNORECASE,
    )),
    ("responsibility", re.compile(
        r"\b(?:responsibility|responsibilities|responsible for|task|action|duty|"
        r"required to do|supposed to do|role)\b|\bhow\b.*\bhandl(?:e|ed|ing)\b|"
        r"\bwork remains\b",
        re.IGNORECASE,
    )),
    ("quantity", re.compile(
        r"\b(?:how many|how much|number|count|quantity|amount|total|percentage|percent)\b",
        re.IGNORECASE,
    )),
    ("outcome", re.compile(
        r"\b(?:outcome|result|impact|effect|consequence|what happened|lesson|caused)\b",
        re.IGNORECASE,
    )),
    ("requirement", re.compile(
        r"\b(?:requirement|condition|criterion|criteria|prerequisite|depends on|needed for)\b",
        re.IGNORECASE,
    )),
)
FIELD_CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "owner": re.compile(
        r"\b(?:owner|owned by|ownership|lead|point person|accountable|assigned to|"
        r"designated|will own|owns|responsible party)\b",
        re.IGNORECASE,
    ),
    "status": re.compile(
        r"\b(?:status|state|frozen|locked|approved|pending|finalized|confirmed|"
        r"open|closed|paused|blocked|on hold|in progress|complete|completed)\b",
        re.IGNORECASE,
    ),
    "deadline": re.compile(
        r"\b(?:deadline|due|by\s+(?:eod|end of day)|eod|end of day|today|tomorrow|yesterday|"
        r"last\s+(?:night|week|weekend|month|year|spring|summer|fall|autumn|winter|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"(?:day|week|month|year)s?\s+(?:ago|before|after)|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"\d{4}-\d{2}-\d{2}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\b",
        re.IGNORECASE,
    ),
    "responsibility": re.compile(
        r"\b(?:responsible for|responsibility|responsibilities|task|action|duty|"
        r"role|must|needs? to|will handle|will post|will prepare|will review)\b",
        re.IGNORECASE,
    ),
    "quantity": re.compile(
        r"\b(?:number|count|quantity|amount|total|percentage|percent|\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    "outcome": re.compile(
        r"\b(?:outcome|result|impact|effect|consequence|resulted in|led to|lesson|caused)\b",
        re.IGNORECASE,
    ),
    "requirement": re.compile(
        r"\b(?:requirement|condition|criterion|criteria|prerequisite|required|"
        r"only if|provided that|depends on|needed for)\b",
        re.IGNORECASE,
    ),
}
STATUS_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:freeze|frozen)\b",
        r"\b(?:lock|locked)\b",
        r"\b(?:approve|approved|approval)\b",
        r"\b(?:confirm|confirmed|confirmation)\b",
        r"\b(?:final|finalized)\b",
        r"\bpending\b",
        r"\b(?:open|closed)\b",
        r"\b(?:pause|paused|on hold)\b",
        r"\b(?:block|blocked)\b",
        r"\b(?:complete|completed)\b",
    )
)


def infer_query_field(query: str) -> str:
    """Infer the answer slot requested by a natural-language question."""
    text = str(query or "")
    for name, pattern in QUERY_FIELD_PATTERNS:
        if pattern.search(text):
            return name
    return "general"


def split_sentences(content: str) -> list[str]:
    sentences = [value.strip() for value in SENTENCE_SPLIT_RE.split(str(content or "")) if value.strip()]
    return sentences or ([str(content).strip()] if str(content).strip() else [])


def field_match_score(query_field: str, sentence: str, query: str = "") -> float:
    if query_field == "general":
        return 0.0
    pattern = FIELD_CUE_PATTERNS.get(query_field)
    sentence_text = str(sentence or "")
    if pattern is None or not pattern.search(sentence_text):
        return 0.0
    if query_field == "status":
        requested_anchors = [anchor for anchor in STATUS_ANCHOR_PATTERNS if anchor.search(str(query or ""))]
        if requested_anchors:
            return 1.0 if requested_anchors[0].search(sentence_text) else 0.25
    return 1.0


def format_reranker_sentence(message: dict[str, Any], sentence: str, query_field: str) -> str:
    metadata: list[str] = []
    authors = message.get("authors")
    if isinstance(authors, (list, tuple)) and authors:
        metadata.append(f"speakers={','.join(str(value) for value in authors)}")
    elif message.get("author"):
        metadata.append(f"speaker={message['author']}")
    for key, label in (
        ("date_time", "date"),
        ("timestamp", "timestamp"),
        ("role", "role"),
        ("_channel", "channel"),
        ("phase_name", "phase"),
        ("topic", "topic"),
    ):
        value = str(message.get(key, "") or "").strip()
        if value:
            metadata.append(f"{label}={value}")
    metadata.append(f"requested_field={query_field}")
    return f"[{' / '.join(metadata)}] {sentence}".strip()


class Reranker(Protocol):
    provider: str
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("SentenceTransformers is required for cross-encoder reranking.") from exc

        requested = str(device or "auto").lower()
        self.model = CrossEncoder(model_name, device=None if requested == "auto" else requested)
        self.provider = "sentence_transformer"
        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.zeros(0, dtype=np.float32)
        pairs = [(str(query), str(document)) for document in documents]
        values = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) >= 1000,
        )
        return np.asarray(values, dtype=np.float32).reshape(-1)


def build_reranker(provider: str, model_name: str, device: str = "auto", batch_size: int = 64) -> Reranker | None:
    normalized = str(provider or "none").strip().lower()
    if normalized in {"", "none", "disabled", "off"}:
        return None
    if normalized in {"sentence_transformer", "sentence-transformer", "cross_encoder", "cross-encoder"}:
        if not str(model_name).strip():
            raise ValueError("A reranker model is required when reranking is enabled.")
        return CrossEncoderReranker(model_name, device=device, batch_size=batch_size)
    raise ValueError(f"Unknown reranker provider: {provider}")
