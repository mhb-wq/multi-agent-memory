"""Memory Graph retrieval for GroupMemBench.

This is the first graph in the proposed two-graph architecture. It models
message content and semantic entities plus reply/time structure. User identity
is retained as metadata only; it is deliberately not used for ranking yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .embeddings import Embedder, normalize_rows
from .entities import normalize_entity


INDEX_FORMAT_VERSION = 3


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def row_dot(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute row-wise dot products without triggering heavyweight BLAS startup."""
    return np.einsum("ij,j->i", matrix, vector, optimize=False)


@dataclass
class MemoryGraphConfig:
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 40
    ppr_tolerance: float = 1e-7
    reply_edge_weight: float = 1.50
    temporal_edge_weight: float = 0.20
    same_phase_edge_weight: float = 0.10
    temporal_half_life_hours: float = 168.0
    dense_weight: float = 0.25
    ppr_weight: float = 0.50
    bm25_weight: float = 0.25
    entity_seed_weight: float = 1.0
    dense_seed_top_k: int = 32
    entity_seed_top_k: int = 3
    entity_similarity_threshold: float = 0.45
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    candidate_pool: int = 64


@dataclass
class SearchResult:
    message_id: str
    score: float
    dense_score: float
    ppr_score: float
    bm25_score: float
    entity_hits: list[str]
    message: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "score": round(float(self.score), 8),
            "dense_score": round(float(self.dense_score), 8),
            "ppr_score": round(float(self.ppr_score), 8),
            "bm25_score": round(float(self.bm25_score), 8),
            "entity_hits": list(self.entity_hits),
            "message": self.message,
        }


class MemoryGraph:
    def __init__(self, embedder: Embedder, entity_extractor: Any, config: MemoryGraphConfig | None = None):
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.config = config or MemoryGraphConfig()
        self.messages: list[dict[str, Any]] = []
        self.message_ids: list[str] = []
        self.message_index: dict[str, int] = {}
        self.message_embeddings = np.zeros((0, getattr(embedder, "dimension", 1)), dtype=np.float32)
        self.entity_texts: list[str] = []
        self.entity_index: dict[str, int] = {}
        self.entity_embeddings = np.zeros((0, getattr(embedder, "dimension", 1)), dtype=np.float32)
        self.message_entities: list[list[str]] = []
        self.node_names: list[str] = []
        self.node_index: dict[str, int] = {}
        self.adjacency: sparse.csr_matrix | None = None
        self.transition: sparse.csr_matrix | None = None
        self.dangling_mask = np.zeros(0, dtype=bool)
        self._bm25_tokens: list[list[str]] = []
        self._bm25_tf: list[Counter[str]] = []
        self._bm25_df: Counter[str] = Counter()
        self._bm25_avgdl = 0.0
        self._bm25_postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._built = False

    def build(self, messages: Sequence[dict[str, Any]], show_progress: bool = False) -> "MemoryGraph":
        self._reset()
        self.messages = self._normalize_messages(messages)
        self.message_ids = [str(item["msg_node"]) for item in self.messages]
        self.message_index = {message_id: idx for idx, message_id in enumerate(self.message_ids)}
        texts = [str(item.get("content", "")) for item in self.messages]
        self.message_embeddings = normalize_rows(self.embedder.encode(texts))

        extracted = self.entity_extractor.batch_extract(texts)
        entity_surfaces: dict[str, str] = {}
        self.message_entities = []
        for values in extracted:
            keys: list[str] = []
            for value in values:
                key = normalize_entity(value)
                if re.fullmatch(r"user_\d+", key):
                    continue
                if key and key not in keys:
                    keys.append(key)
                    entity_surfaces.setdefault(key, str(value).strip())
            self.message_entities.append(keys)
        self.entity_texts = [entity_surfaces[key] for key in sorted(entity_surfaces)]
        self.entity_index = {normalize_entity(text): idx for idx, text in enumerate(self.entity_texts)}
        self.entity_embeddings = normalize_rows(self.embedder.encode(self.entity_texts))

        self.node_names = [f"message:{message_id}" for message_id in self.message_ids]
        self.node_names.extend(f"entity:{normalize_entity(text)}" for text in self.entity_texts)
        self.node_index = {name: idx for idx, name in enumerate(self.node_names)}
        edge_weights: dict[tuple[int, int], float] = defaultdict(float)
        directed_edge_weights: dict[tuple[int, int], float] = defaultdict(float)

        def add_edge(left: int, right: int, weight: float) -> None:
            if left == right or weight <= 0:
                return
            key = (min(left, right), max(left, right))
            edge_weights[key] += float(weight)

        def add_directed_edge(source: int, target: int, weight: float) -> None:
            if source == target or weight <= 0:
                return
            directed_edge_weights[(source, target)] += float(weight)

        message_count = len(self.messages)
        entity_items = enumerate(self.message_entities)
        if show_progress and len(self.message_entities) >= 1000:
            from tqdm.auto import tqdm

            entity_items = tqdm(
                entity_items,
                total=len(self.message_entities),
                desc="Memory Graph edges",
                unit="message",
                dynamic_ncols=True,
            )
        for message_idx, entity_keys in entity_items:
            normalized_content = " ".join(str(self.messages[message_idx].get("content", "")).lower().split())
            occurrences_by_key = {key: max(1, normalized_content.count(key)) for key in entity_keys}
            denominator = max(1, sum(occurrences_by_key.values()))
            for key in entity_keys:
                entity_idx = self.entity_index[key]
                occurrences = occurrences_by_key[key]
                add_edge(message_idx, message_count + entity_idx, occurrences / denominator)

        for message_idx, message in enumerate(self.messages):
            parent = str(message.get("reply_to") or "").strip()
            if parent in self.message_index:
                add_edge(message_idx, self.message_index[parent], self.config.reply_edge_weight)

        by_channel: dict[str, list[int]] = defaultdict(list)
        for idx, message in enumerate(self.messages):
            by_channel[str(message.get("_channel", ""))].append(idx)
        for channel_indices in by_channel.values():
            channel_indices.sort(key=lambda idx: (str(self.messages[idx].get("timestamp", "")), idx))
            for left, right in zip(channel_indices, channel_indices[1:]):
                weight = self.config.temporal_edge_weight * self._temporal_decay(
                    self.messages[left].get("timestamp"), self.messages[right].get("timestamp")
                )
                # Forward-only time edges preserve chronological direction. Same-phase
                # edges remain symmetric so related messages are still mutually reachable.
                add_directed_edge(left, right, weight)
                if self.messages[left].get("phase_name") and self.messages[left].get("phase_name") == self.messages[right].get("phase_name"):
                    add_edge(left, right, self.config.same_phase_edge_weight)

        self.adjacency = self._build_sparse_adjacency(
            len(self.node_names),
            edge_weights,
            directed_edge_weights,
        )
        degrees = np.asarray(self.adjacency.sum(axis=1)).ravel()
        inverse = np.zeros_like(degrees, dtype=np.float32)
        nonzero = degrees > 0
        self.dangling_mask = ~nonzero
        inverse[nonzero] = 1.0 / degrees[nonzero]
        self.transition = sparse.diags(inverse) @ self.adjacency
        self._build_bm25(texts, show_progress=show_progress)
        self._built = True
        return self

    def retrieve(self, question: str, top_k: int = 10) -> list[SearchResult]:
        self._require_built()
        if not self.messages:
            return []
        query = str(question or "").strip()
        q_embedding = normalize_rows(self.embedder.encode([query]))[0]
        dense = row_dot(self.message_embeddings, q_embedding)
        bm25 = self._bm25_scores(query)
        query_entities = [normalize_entity(value) for value in self.entity_extractor.batch_extract([query])[0]]
        entity_hits: dict[int, list[str]] = defaultdict(list)
        reset = np.zeros(len(self.node_names), dtype=np.float32)

        dense_seed_count = min(len(dense), max(1, int(self.config.dense_seed_top_k)))
        dense_order = np.argsort(dense)[::-1][:dense_seed_count]
        dense_values = np.maximum(dense[dense_order], 0.0)
        for idx, value in zip(dense_order, dense_values):
            reset[int(idx)] += float(value) + 1e-4

        for query_entity in query_entities:
            matches: list[tuple[int, float]] = []
            exact = self.entity_index.get(query_entity)
            if exact is not None:
                matches.append((exact, 1.0))
            elif len(self.entity_embeddings):
                query_vec = normalize_rows(self.embedder.encode([query_entity]))[0]
                similarities = row_dot(self.entity_embeddings, query_vec)
                order = np.argsort(similarities)[::-1][: max(1, int(self.config.entity_seed_top_k))]
                matches.extend((int(idx), float(similarities[idx])) for idx in order if similarities[idx] >= self.config.entity_similarity_threshold)
            for entity_idx, similarity in matches:
                node_idx = len(self.messages) + entity_idx
                reset[node_idx] += self.config.entity_seed_weight * max(0.0, similarity)
                entity_hits[entity_idx].append(query_entity)

        if reset.sum() <= 0:
            reset[: len(self.messages)] = 1.0
        reset /= reset.sum()
        ppr = self._personalized_pagerank(reset)
        candidates = set(int(idx) for idx in dense_order)
        candidates.update(int(idx) for idx in np.argsort(ppr[: len(self.messages)])[::-1][: self.config.candidate_pool])
        candidates.update(int(idx) for idx in np.argsort(bm25)[::-1][: self.config.candidate_pool])
        candidate_ids = sorted(candidates)
        dense_n = minmax(dense[candidate_ids])
        ppr_n = minmax(ppr[candidate_ids])
        bm25_n = minmax(bm25[candidate_ids])
        combined = (
            self.config.dense_weight * dense_n
            + self.config.ppr_weight * ppr_n
            + self.config.bm25_weight * bm25_n
        )
        order = np.argsort(combined)[::-1][: max(0, int(top_k))]
        results: list[SearchResult] = []
        for local_idx in order:
            message_idx = candidate_ids[int(local_idx)]
            hits = [self.entity_texts[idx] for idx, values in entity_hits.items() if values and self._message_has_entity(message_idx, idx)]
            results.append(SearchResult(
                message_id=self.message_ids[message_idx],
                score=float(combined[local_idx]),
                dense_score=float(dense[message_idx]),
                ppr_score=float(ppr[message_idx]),
                bm25_score=float(bm25[message_idx]),
                entity_hits=hits,
                message=self.messages[message_idx],
            ))
        return results

    def save(self, directory: str | Path) -> None:
        self._require_built()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "messages.json").write_text(json.dumps(self.messages, ensure_ascii=False), encoding="utf-8")
        (target / "entities.json").write_text(json.dumps(self.entity_texts, ensure_ascii=False), encoding="utf-8")
        np.save(target / "message_embeddings.npy", self.message_embeddings)
        np.save(target / "entity_embeddings.npy", self.entity_embeddings)
        sparse.save_npz(target / "adjacency.npz", self.adjacency)
        metadata = {
            "index_format_version": INDEX_FORMAT_VERSION,
            "config": asdict(self.config),
            "message_ids": self.message_ids,
            "message_entities": self.message_entities,
            "fingerprint": self.fingerprint(),
            "embedding_dimension": int(self.message_embeddings.shape[1]) if self.message_embeddings.ndim == 2 else 0,
            "embedder": self._component_signature(self.embedder),
            "entity_extractor": self._component_signature(self.entity_extractor),
        }
        (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        embedder: Embedder,
        entity_extractor: Any,
        show_progress: bool = False,
    ) -> "MemoryGraph":
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        cached_version = int(metadata.get("index_format_version", 0))
        if cached_version != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"Cached Memory Graph index format is {cached_version}; expected {INDEX_FORMAT_VERSION}. "
                "Rebuild the index instead of using --reuse-index."
            )
        graph = cls(embedder, entity_extractor, MemoryGraphConfig(**metadata["config"]))
        graph.messages = json.loads((source / "messages.json").read_text(encoding="utf-8"))
        graph.entity_texts = json.loads((source / "entities.json").read_text(encoding="utf-8"))
        graph.message_ids = [str(value) for value in metadata["message_ids"]]
        graph.message_index = {message_id: idx for idx, message_id in enumerate(graph.message_ids)}
        graph.message_entities = [list(values) for values in metadata["message_entities"]]
        graph.entity_index = {normalize_entity(text): idx for idx, text in enumerate(graph.entity_texts)}
        graph.message_embeddings = normalize_rows(np.load(source / "message_embeddings.npy"))
        graph.entity_embeddings = normalize_rows(np.load(source / "entity_embeddings.npy"))
        expected_dimension = int(metadata.get("embedding_dimension", 0))
        if expected_dimension and graph.message_embeddings.shape[1] != expected_dimension:
            raise ValueError("Cached message embedding dimension does not match metadata.")
        runtime_dimension = int(getattr(embedder, "dimension", 0) or 0)
        if runtime_dimension and expected_dimension and runtime_dimension != expected_dimension:
            raise ValueError(
                f"Cached embedding dimension is {expected_dimension}, but the active embedder returns {runtime_dimension}."
            )
        for key, component in (("embedder", embedder), ("entity_extractor", entity_extractor)):
            cached_signature = metadata.get(key)
            if cached_signature and cached_signature != graph._component_signature(component):
                raise ValueError(
                    f"Cached {key} does not match the active configuration: "
                    f"cached={cached_signature}, active={graph._component_signature(component)}"
                )
        graph.node_names = [f"message:{message_id}" for message_id in graph.message_ids]
        graph.node_names.extend(f"entity:{normalize_entity(text)}" for text in graph.entity_texts)
        graph.node_index = {name: idx for idx, name in enumerate(graph.node_names)}
        graph.adjacency = sparse.load_npz(source / "adjacency.npz").tocsr().astype(np.float32)
        if graph.adjacency.shape != (len(graph.node_names), len(graph.node_names)):
            raise ValueError("Cached adjacency shape does not match cached nodes.")
        degrees = np.asarray(graph.adjacency.sum(axis=1)).ravel()
        inverse = np.zeros_like(degrees, dtype=np.float32)
        nonzero = degrees > 0
        inverse[nonzero] = 1.0 / degrees[nonzero]
        graph.dangling_mask = ~nonzero
        graph.transition = sparse.diags(inverse) @ graph.adjacency
        graph._build_bm25(
            [str(message.get("content", "")) for message in graph.messages],
            show_progress=show_progress,
        )
        graph._built = True
        if graph.fingerprint() != metadata.get("fingerprint"):
            raise ValueError("Cached Memory Graph fingerprint does not match cached messages.")
        return graph

    def fingerprint(self) -> str:
        return self.corpus_fingerprint(self.messages)

    @staticmethod
    def corpus_fingerprint(messages: Sequence[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for message in messages:
            for key in (
                "msg_node",
                "content",
                "author",
                "authors",
                "role",
                "timestamp",
                "reply_to",
                "phase_name",
                "topic",
                "_channel",
            ):
                digest.update(str(message.get(key, "")).encode("utf-8"))
                digest.update(b"\0")
        return digest.hexdigest()

    def _reset(self) -> None:
        self._built = False
        self.messages = []
        self.message_ids = []
        self.message_index = {}
        self.entity_texts = []
        self.entity_index = {}
        self.message_entities = []
        self.node_names = []
        self.node_index = {}
        self.adjacency = None
        self.transition = None
        self.dangling_mask = np.zeros(0, dtype=bool)

    @staticmethod
    def _normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, raw in enumerate(messages):
            item = dict(raw)
            message_id = str(item.get("msg_node") or f"Msg_{idx}")
            if message_id in seen:
                raise ValueError(f"Duplicate msg_node: {message_id}")
            seen.add(message_id)
            item["msg_node"] = message_id
            item["content"] = str(item.get("content", ""))
            normalized.append(item)
        return normalized

    @staticmethod
    def _build_sparse_adjacency(
        node_count: int,
        edge_weights: dict[tuple[int, int], float],
        directed_edge_weights: dict[tuple[int, int], float] | None = None,
    ) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for (left, right), weight in edge_weights.items():
            rows.extend((left, right))
            cols.extend((right, left))
            data.extend((weight, weight))
        for (source, target), weight in (directed_edge_weights or {}).items():
            rows.append(source)
            cols.append(target)
            data.append(weight)
        return sparse.csr_matrix((data, (rows, cols)), shape=(node_count, node_count), dtype=np.float32)

    @staticmethod
    def _component_signature(component: Any) -> dict[str, Any]:
        return {
            "class": component.__class__.__name__,
            "provider": str(getattr(component, "provider", "")),
            "model_name": str(getattr(component, "model_name", "")),
        }

    def _temporal_decay(self, left: Any, right: Any) -> float:
        try:
            from datetime import datetime

            first = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
            second = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
            hours = max(0.0, (second - first).total_seconds() / 3600.0)
            half_life = max(1e-6, float(self.config.temporal_half_life_hours))
            return float(math.exp(-math.log(2.0) * hours / half_life))
        except Exception:
            return 1.0

    def _build_bm25(self, texts: Sequence[str], show_progress: bool = False) -> None:
        self._bm25_tokens = [tokenize(text) for text in texts]
        self._bm25_tf = []
        self._bm25_df = Counter()
        self._bm25_postings = defaultdict(dict)
        total_length = 0
        token_items = enumerate(self._bm25_tokens)
        if show_progress and len(self._bm25_tokens) >= 1000:
            from tqdm.auto import tqdm

            token_items = tqdm(
                token_items,
                total=len(self._bm25_tokens),
                desc="BM25 index",
                unit="message",
                dynamic_ncols=True,
            )
        for idx, tokens in token_items:
            frequencies = Counter(tokens)
            self._bm25_tf.append(frequencies)
            total_length += len(tokens)
            for token in frequencies:
                self._bm25_df[token] += 1
                self._bm25_postings[token][idx] = frequencies[token]
        self._bm25_avgdl = total_length / max(1, len(self._bm25_tokens))

    def _bm25_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.messages), dtype=np.float32)
        query_terms = set(tokenize(query))
        document_count = max(1, len(self.messages))
        for term in query_terms:
            postings = self._bm25_postings.get(term, {})
            document_frequency = self._bm25_df.get(term, 0)
            if not postings or not document_frequency:
                continue
            idf = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for idx, frequency in postings.items():
                document_length = len(self._bm25_tokens[idx])
                denominator = frequency + self.config.bm25_k1 * (
                    1.0 - self.config.bm25_b + self.config.bm25_b * document_length / max(1e-9, self._bm25_avgdl)
                )
                scores[idx] += idf * frequency * (self.config.bm25_k1 + 1.0) / max(1e-9, denominator)
        return scores

    def _personalized_pagerank(self, reset: np.ndarray) -> np.ndarray:
        if self.transition is None:
            return reset
        rank = reset.copy()
        damping = float(self.config.ppr_damping)
        for _ in range(max(1, int(self.config.ppr_max_iterations))):
            dangling_mass = float(rank[self.dangling_mask].sum()) if self.dangling_mask.size else 0.0
            updated = (1.0 - damping) * reset + damping * ((self.transition.T @ rank) + dangling_mass * reset)
            updated = np.asarray(updated).ravel()
            total = float(updated.sum())
            if total > 0:
                updated /= total
            if np.linalg.norm(updated - rank, ord=1) <= self.config.ppr_tolerance:
                rank = updated
                break
            rank = updated
        return rank

    def _message_has_entity(self, message_idx: int, entity_idx: int) -> bool:
        key = normalize_entity(self.entity_texts[entity_idx])
        return key in self.message_entities[message_idx]

    def _require_built(self) -> None:
        if not self._built or self.adjacency is None:
            raise RuntimeError("MemoryGraph is not built. Call build(messages) first.")
