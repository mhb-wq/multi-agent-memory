"""NaiveRAG: official GroupMemBench dense-RAG shape with a fixed local embedder.

The official dense baseline indexes one document per message (content only)
and retrieves by exact cosine similarity, top-k. This adapter keeps that
retrieval shape and replaces only the embedding model with the protocol's
fixed local MiniLM model; it does not add chunking, reranking or an LLM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaselineAdapter, Message, Passage, search_text


class NaiveRAGAdapter(BaselineAdapter):
    name = "naive_rag"

    def __init__(
        self, encoder: Any = None, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.encoder = encoder
        self.model_name = model_name
        self.messages: list[Message] = []
        self._matrix: np.ndarray | None = None

    def reset(self) -> None:
        self.messages.clear()
        self._matrix = None

    def ingest(self, message: Message) -> None:
        self.messages.append(message)
        self._matrix = None

    def count(self) -> int:
        return len(self.messages)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name)
        vectors = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _build(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = self._encode([message.text for message in self.messages])
        return self._matrix

    def retrieve(self, query: str, asking_user_id: str, top_k: int = 10) -> list[Passage]:
        if not self.messages:
            return []
        matrix = self._build()
        query_vector = self._encode([search_text(query, asking_user_id)])[0]
        scores = matrix @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            Passage(
                text=self.messages[index].text,
                source_ids=(self.messages[index].id,),
                metadata=self.messages[index].provenance(),
                score=float(scores[index]),
            )
            for index in order
        ]
