"""BM25-only retrieval for apples-to-apples GroupMemBench comparisons."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


class _LocalBM25Okapi:
    """Small rank_bm25-compatible fallback when the optional package is absent."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(document) for document in corpus) / max(1, self.corpus_size)
        self.doc_len = [len(document) for document in corpus]
        self.doc_freqs = [Counter(document) for document in corpus]
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for index, frequencies in enumerate(self.doc_freqs):
            for term, frequency in frequencies.items():
                self.postings[term][index] = frequency
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)
        self.idf: dict[str, float] = {}
        for term, documents in self.postings.items():
            self.idf[term] = math.log(self.corpus_size - len(documents) + 0.5) - math.log(len(documents) + 0.5)
        average_idf = sum(self.idf.values()) / max(1, len(self.idf))
        floor = self.epsilon * average_idf
        for term, value in list(self.idf.items()):
            if value < 0:
                self.idf[term] = floor

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in query:
            postings = self.postings.get(term, {})
            idf = self.idf.get(term, 0.0)
            for index, frequency in postings.items():
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[index] / max(1e-9, self.avgdl)
                )
                scores[index] += idf * frequency * (self.k1 + 1.0) / max(1e-9, denominator)
        return scores


try:  # Prefer the benchmark's reference implementation when installed.
    from rank_bm25 import BM25Okapi as _RankBM25Okapi
except ImportError:  # pragma: no cover - depends on the active environment
    _RankBM25Okapi = None


@dataclass
class BM25SearchResult:
    message_id: str
    score: float
    message: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "score": round(float(self.score), 8),
            "dense_score": 0.0,
            "ppr_score": 0.0,
            "bm25_score": round(float(self.score), 8),
            "entity_hits": [],
            "message": self.message,
        }


class BM25Retriever:
    def __init__(self, messages: Sequence[dict[str, Any]], show_progress: bool = False):
        self.messages = [dict(message) for message in messages]
        self.message_ids = [str(message.get("msg_node", f"Msg_{index}")) for index, message in enumerate(self.messages)]
        texts = [str(message.get("content", "")) for message in self.messages]
        text_items: Any = texts
        if show_progress and len(texts) >= 1000:
            from tqdm.auto import tqdm

            text_items = tqdm(texts, desc="BM25 index", unit="message", dynamic_ncols=True)
        corpus = [tokenize(text) for text in text_items]
        backend = _RankBM25Okapi or _LocalBM25Okapi
        self.backend_name = "rank_bm25" if _RankBM25Okapi else "local_okapi"
        self.index = backend(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[BM25SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.messages:
            return []
        scores = np.asarray(self.index.get_scores(query_tokens), dtype=np.float32)
        # Python's stable sort matches the official baseline's tie behavior.
        order = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        return [
            BM25SearchResult(
                message_id=self.message_ids[index],
                score=float(scores[index]),
                message=self.messages[index],
            )
            for index in order[: max(0, int(top_k))]
        ]
