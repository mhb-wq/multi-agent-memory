"""Embedding backends used by the initial memory graph.

The graph itself only depends on the small ``encode`` protocol. This keeps
unit tests and offline experiments independent from a model download while
the real benchmark can use SentenceTransformers.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Deterministic local fallback, useful for tests and smoke runs only."""

    def __init__(self, dimension: int = 256):
        self.dimension = int(dimension)
        self.provider = "hashing"
        self.model_name = f"hashing-{self.dimension}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, index] += sign
            if not tokens:
                out[row, 0] = 1.0
        return normalize_rows(out)


class SentenceTransformerEmbedder:
    """Lazy SentenceTransformers adapter for benchmark runs."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "SentenceTransformers is unavailable. Install requirements.txt "
                "or use --embedding-backend hashing for a smoke run."
            ) from exc

        requested = str(device or "auto").lower()
        if requested == "auto":
            try:
                import torch

                requested = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                requested = "cpu"
        self.model = SentenceTransformer(model_name, device=requested)
        self.provider = "sentence_transformer"
        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        values = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= 1000,
            batch_size=self.batch_size,
        )
        return normalize_rows(np.asarray(values, dtype=np.float32))


def build_embedder(
    backend: str,
    model_name: str,
    device: str = "auto",
    batch_size: int = 64,
) -> Embedder:
    backend = str(backend or "sentence_transformer").strip().lower()
    if backend in {"hashing", "local", "test"}:
        return HashingEmbedder()
    if backend in {"sentence_transformer", "sentence-transformer", "st"}:
        return SentenceTransformerEmbedder(model_name, device=device, batch_size=batch_size)
    raise ValueError(f"Unknown embedding backend: {backend}")
