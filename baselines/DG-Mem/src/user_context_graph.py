"""Speaker- and context-aware graph retrieval for GroupMemBench.

The User-Context Graph is deliberately separate from the Memory Graph. It
stores lightweight message anchors and structural identity/context edges, then
uses shared ``msg_node`` identifiers to rerank Memory Graph candidates.

Only deployment-realistic fields are used: author, role, channel, phase,
topic, reply target, timestamp, and message text. Synthetic generation labels
such as ``is_noise`` and ``decision_change_metadata`` are never indexed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .bm25 import BM25Retriever
from .memory_graph import MemoryGraph, SearchResult, minmax, tokenize
from .rerankers import (
    Reranker,
    field_match_score,
    format_reranker_sentence,
    infer_query_field,
    split_sentences,
)


USER_ID_RE = re.compile(r"\buser_\d+\b", re.IGNORECASE)
FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.IGNORECASE)
TEMPORAL_QUERY_RE = re.compile(
    r"\b(?:when|deadline|date|time|latest|current|currently|today|before|after)\b",
    re.IGNORECASE,
)
STATE_QUERY_RE = re.compile(
    r"\b(?:status|freeze|frozen|lock|locked|approve|approved|approval|final|finalized|"
    r"owner|ownership|designated|decision|decided|confirmed|exact|specified|trigger|triggers|"
    r"wording|sla|approver|name of)\b",
    re.IGNORECASE,
)
EXACT_QUERY_RE = re.compile(
    r"\b(?:exact|designated|specified|final|confirmed|who owns|"
    r"what is the (?:deadline|status)|trigger|triggers|wording|sla|approver|name of)\b",
    re.IGNORECASE,
)
CONTEXT_QUERY_RE = re.compile(r"\b(?:phase|topic|channel|during)\b", re.IGNORECASE)
CONFIRMED_EVIDENCE_RE = re.compile(
    r"\b(?:confirmed|finalized|approved|agreed|decided|designated|locked|frozen|"
    r"signed off|deadline is|owner is|has been)\b",
    re.IGNORECASE,
)
STRONG_CONFIRMED_EVIDENCE_RE = re.compile(
    r"\b(?:confirmed|finalized|approved|agreed|decided|designated|locked|frozen|signed off|has been)\b",
    re.IGNORECASE,
)
UNAVAILABLE_EVIDENCE_RE = re.compile(
    r"\b(?:tbd|to be (?:confirmed|decided|finalized|specified)|not (?:yet )?(?:confirmed|"
    r"finalized|specified|decided)|awaiting|pending|still need|needs? to (?:confirm|decide|"
    r"provide|specify|name|designate|mark|post|define)|asked to (?:confirm|decide|provide|"
    r"specify|name|designate|mark|post|define)|not resolved|"
    r"push back on|without (?:forcing|creating) (?:a )?false (?:sense|impression) of finality)\b",
    re.IGNORECASE,
)
TENTATIVE_EVIDENCE_RE = re.compile(
    r"\b(?:proposed|proposal|suggested|suggestion|draft|candidate|tentative|target|"
    r"should|could|might|may|recommend|option)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "which", "who", "why", "with",
    "phase", "status",
}
STRUCTURAL_CONFIG_FIELDS = (
    "author_edge_weight",
    "context_edge_weight",
    "user_context_edge_weight",
    "role_edge_weight",
    "reply_edge_weight",
    "mention_edge_weight",
)


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass
class UserContextGraphConfig:
    ppr_damping: float = 0.85
    ppr_max_iterations: int = 40
    ppr_tolerance: float = 1e-7
    author_edge_weight: float = 1.50
    context_edge_weight: float = 1.00
    user_context_edge_weight: float = 0.35
    role_edge_weight: float = 0.50
    reply_edge_weight: float = 1.20
    mention_edge_weight: float = 1.00
    memory_seed_weight: float = 1.00
    lexical_seed_weight: float = 0.80
    asker_seed_weight: float = 1.50
    self_query_asker_seed_scale: float = 0.75
    non_self_asker_seed_weight: float = 0.15
    context_seed_weight: float = 0.60
    memory_weight: float = 0.50
    context_bm25_weight: float = 0.25
    user_ppr_weight: float = 0.20
    compatibility_weight: float = 0.05
    context_match_weight: float = 0.08
    evidence_state_weight: float = 0.07
    reranker_weight: float = 0.35
    field_match_weight: float = 0.20
    sentence_reranking: bool = True
    diversity_weight: float = 0.08
    same_author_penalty: float = 0.02
    query_aware_fusion: bool = True
    cross_graph_retrieval: bool = True
    candidate_refinement: bool = True
    self_query_ppr_scale: float = 0.75
    self_query_compatibility_scale: float = 1.50
    contextual_query_context_scale: float = 1.50
    state_query_evidence_scale: float = 1.50
    candidate_pool: int = 64


@dataclass
class UserContextSearchResult:
    message_id: str
    score: float
    memory_score: float
    user_context_score: float
    user_ppr_score: float
    context_bm25_score: float
    asker_compatibility: float
    context_match_score: float
    evidence_state_score: float
    evidence_state_label: str
    reranker_score: float
    reranker_sentence: str
    query_field: str
    field_match_score: float
    context_hits: list[str]
    query_profile: list[str]
    fusion_weights: dict[str, float]
    message: dict[str, Any]
    memory_result: SearchResult | None = None

    def to_dict(self) -> dict[str, Any]:
        memory = self.memory_result
        return {
            "message_id": self.message_id,
            "score": round(float(self.score), 8),
            "memory_score": round(float(self.memory_score), 8),
            "user_context_score": round(float(self.user_context_score), 8),
            "dense_score": round(float(memory.dense_score), 8) if memory else 0.0,
            "ppr_score": round(float(memory.ppr_score), 8) if memory else 0.0,
            "bm25_score": round(float(memory.bm25_score), 8) if memory else 0.0,
            "entity_hits": list(memory.entity_hits) if memory else [],
            "user_ppr_score": round(float(self.user_ppr_score), 8),
            "context_bm25_score": round(float(self.context_bm25_score), 8),
            "asker_compatibility": round(float(self.asker_compatibility), 8),
            "context_match_score": round(float(self.context_match_score), 8),
            "evidence_state_score": round(float(self.evidence_state_score), 8),
            "evidence_state_label": self.evidence_state_label,
            "reranker_score": round(float(self.reranker_score), 8),
            "reranker_sentence": self.reranker_sentence,
            "query_field": self.query_field,
            "field_match_score": round(float(self.field_match_score), 8),
            "context_hits": list(self.context_hits),
            "query_profile": list(self.query_profile),
            "fusion_weights": {key: round(float(value), 8) for key, value in self.fusion_weights.items()},
            "message": self.message,
        }


class UserContextGraph:
    """Heterogeneous user, role, context, and message-anchor graph."""

    def __init__(self, config: UserContextGraphConfig | None = None, reranker: Reranker | None = None):
        self.config = config or UserContextGraphConfig()
        self.reranker = reranker
        self._reset()

    def build(self, messages: Sequence[dict[str, Any]]) -> "UserContextGraph":
        self._reset()
        self.messages = self._normalize_messages(messages)
        self.message_ids = [str(message["msg_node"]) for message in self.messages]
        self.message_index = {message_id: index for index, message_id in enumerate(self.message_ids)}

        self.user_texts = sorted({author for message in self.messages for author in self._message_authors(message)})
        self.role_texts = sorted({str(message.get("role", "")).strip() for message in self.messages if message.get("role")})
        self.context_keys = sorted({self._context_key(message) for message in self.messages})
        self._initialize_nodes()

        edge_weights: dict[tuple[int, int], float] = defaultdict(float)
        user_roles: set[tuple[str, str]] = set()
        user_context_counts: Counter[tuple[str, tuple[str, str, str]]] = Counter()

        def add_edge(left: int, right: int, weight: float) -> None:
            if left == right or weight <= 0:
                return
            key = (min(left, right), max(left, right))
            edge_weights[key] += float(weight)

        for message_idx, message in enumerate(self.messages):
            author_keys = {normalize_label(author) for author in self._message_authors(message)}
            role_key = normalize_label(message.get("role"))
            context_key = self._context_key(message)
            for author_key in author_keys:
                if author_key not in self.user_index:
                    continue
                add_edge(message_idx, self._user_node(author_key), self.config.author_edge_weight)
                user_context_counts[(author_key, context_key)] += 1
                if role_key in self.role_index:
                    user_roles.add((author_key, role_key))
            add_edge(message_idx, self._context_node(context_key), self.config.context_edge_weight)

            parent_id = str(message.get("reply_to") or "").strip()
            if parent_id in self.message_index:
                add_edge(message_idx, self.message_index[parent_id], self.config.reply_edge_weight)

            for mentioned_user in USER_ID_RE.findall(str(message.get("content", ""))):
                mentioned_key = normalize_label(mentioned_user)
                if mentioned_key in self.user_index and mentioned_key not in author_keys:
                    add_edge(message_idx, self._user_node(mentioned_key), self.config.mention_edge_weight)

        for author_key, role_key in user_roles:
            add_edge(self._user_node(author_key), self._role_node(role_key), self.config.role_edge_weight)
        for (author_key, context_key), count in user_context_counts.items():
            weight = self.config.user_context_edge_weight * math.log1p(count)
            add_edge(self._user_node(author_key), self._context_node(context_key), weight)

        self.adjacency = self._build_sparse_adjacency(len(self.node_names), edge_weights)
        self._initialize_transition()
        self._initialize_lexical_index()
        self._built = True
        return self

    def retrieve(
        self,
        question: str,
        asking_user_id: str = "",
        *,
        memory_results: Sequence[SearchResult] | None = None,
        top_k: int = 10,
    ) -> list[UserContextSearchResult]:
        self._require_built()
        if not self.messages:
            return []

        query = str(question or "").strip()
        asker_key = normalize_label(asking_user_id)
        is_self_query = bool(FIRST_PERSON_RE.search(query)) and asker_key in self.user_index
        query_profile = self._query_profile(query, is_self_query)
        query_field = infer_query_field(query)
        cross_graph_enabled = bool(self.config.cross_graph_retrieval)
        refinement_enabled = bool(self.config.candidate_refinement)
        personalized_query = (
            f"{self._author_token(asking_user_id)} {query}".strip()
            if is_self_query and cross_graph_enabled
            else query
        )
        pool_size = max(int(top_k), int(self.config.candidate_pool))
        lexical_results = self._context_bm25.retrieve(personalized_query, top_k=pool_size)
        memory_results = list(memory_results or [])

        reset = np.zeros(len(self.node_names), dtype=np.float32)
        if cross_graph_enabled:
            for rank, result in enumerate(memory_results[:pool_size]):
                message_idx = self.message_index.get(str(result.message_id))
                if message_idx is not None:
                    reset[message_idx] += self.config.memory_seed_weight / (rank + 1.0)
        for rank, result in enumerate(lexical_results):
            message_idx = self.message_index.get(str(result.message_id))
            if message_idx is not None:
                reset[message_idx] += self.config.lexical_seed_weight / (rank + 1.0)

        if cross_graph_enabled:
            if asker_key in self.user_index:
                weight = self.config.asker_seed_weight if is_self_query else self.config.non_self_asker_seed_weight
                if is_self_query:
                    weight *= self.config.self_query_asker_seed_scale
                reset[self._user_node(asker_key)] += weight
            for explicit_user in USER_ID_RE.findall(query):
                explicit_key = normalize_label(explicit_user)
                if explicit_key in self.user_index:
                    reset[self._user_node(explicit_key)] += self.config.asker_seed_weight
            self._seed_matching_contexts(reset, query)

        if reset.sum() <= 0:
            reset[: len(self.messages)] = 1.0
        reset /= reset.sum()
        user_ppr = self._personalized_pagerank(reset)

        memory_by_id = {str(result.message_id): result for result in memory_results}
        lexical_by_id = {str(result.message_id): float(result.score) for result in lexical_results}
        candidates = {self.message_index[message_id] for message_id in memory_by_id if message_id in self.message_index}
        candidates.update(self.message_index[message_id] for message_id in lexical_by_id if message_id in self.message_index)
        ppr_order = np.argsort(user_ppr[: len(self.messages)])[::-1][:pool_size]
        candidates.update(int(index) for index in ppr_order)
        candidate_ids = sorted(candidates)
        if not candidate_ids:
            return []

        memory_raw = np.asarray([
            max(0.0, float(memory_by_id[self.message_ids[index]].score))
            if self.message_ids[index] in memory_by_id else 0.0
            for index in candidate_ids
        ], dtype=np.float32)
        lexical_raw = np.asarray([
            max(0.0, lexical_by_id.get(self.message_ids[index], 0.0))
            for index in candidate_ids
        ], dtype=np.float32)
        ppr_raw = np.asarray([user_ppr[index] for index in candidate_ids], dtype=np.float32)
        if refinement_enabled:
            compatibility = np.asarray([
                self._asker_compatibility(self.messages[index], asker_key, is_self_query)
                for index in candidate_ids
            ], dtype=np.float32)
            context_match = np.asarray([
                self._context_match_score(query, self.messages[index])
                for index in candidate_ids
            ], dtype=np.float32)
            evidence_states = [
                self._evidence_state(self.messages[index], query_profile, query)
                for index in candidate_ids
            ]
            evidence_state = np.asarray([value for value, _ in evidence_states], dtype=np.float32)
            reranker_raw, reranker_sentences, field_matches = self._rerank_candidates(
                query, query_field, candidate_ids,
            )
        else:
            compatibility = np.zeros(len(candidate_ids), dtype=np.float32)
            context_match = np.zeros(len(candidate_ids), dtype=np.float32)
            evidence_states = [(0.0, "disabled") for _ in candidate_ids]
            evidence_state = np.zeros(len(candidate_ids), dtype=np.float32)
            reranker_raw = np.zeros(len(candidate_ids), dtype=np.float32)
            reranker_sentences = [str(self.messages[index].get("content", "")) for index in candidate_ids]
            field_matches = np.zeros(len(candidate_ids), dtype=np.float32)

        memory_n = minmax(memory_raw)
        lexical_n = minmax(lexical_raw)
        ppr_n = minmax(ppr_raw)
        reranker_n = minmax(reranker_raw)
        fusion_weights = self._fusion_weights(query_profile, query_field)
        user_context_score = (
            fusion_weights["context_bm25"] * lexical_n
            + fusion_weights["user_ppr"] * ppr_n
            + fusion_weights["compatibility"] * compatibility
            + fusion_weights["context_match"] * context_match
            + fusion_weights["evidence_state"] * evidence_state
            + fusion_weights["reranker"] * reranker_n
            + fusion_weights["field_match"] * field_matches
        )
        combined = fusion_weights["memory"] * memory_n + user_context_score
        order = self._diverse_order(
            candidate_ids, combined, max(0, int(top_k)), apply_penalties=refinement_enabled,
        )

        output: list[UserContextSearchResult] = []
        for local_idx in order:
            message_idx = candidate_ids[int(local_idx)]
            message_id = self.message_ids[message_idx]
            output.append(UserContextSearchResult(
                message_id=message_id,
                score=float(combined[local_idx]),
                memory_score=float(memory_n[local_idx]),
                user_context_score=float(user_context_score[local_idx]),
                user_ppr_score=float(ppr_raw[local_idx]),
                context_bm25_score=float(lexical_raw[local_idx]),
                asker_compatibility=float(compatibility[local_idx]),
                context_match_score=float(context_match[local_idx]),
                evidence_state_score=float(evidence_state[local_idx]),
                evidence_state_label=evidence_states[int(local_idx)][1],
                reranker_score=float(reranker_raw[local_idx]),
                reranker_sentence=reranker_sentences[int(local_idx)],
                query_field=query_field,
                field_match_score=float(field_matches[local_idx]),
                context_hits=self._context_hits(query, self.messages[message_idx]),
                query_profile=query_profile,
                fusion_weights=fusion_weights,
                message=self.messages[message_idx],
                memory_result=memory_by_id.get(message_id),
            ))
        return output

    def _rerank_candidates(
        self,
        query: str,
        query_field: str,
        candidate_ids: Sequence[int],
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Score query-aligned sentences and retain one auditable sentence per message."""
        selected_sentences: list[str] = []
        selected_field_scores: list[float] = []
        sentence_groups: list[list[tuple[str, float]]] = []
        flattened: list[str] = []

        for message_idx in candidate_ids:
            message = self.messages[message_idx]
            content = str(message.get("content", "") or "")
            sentences = split_sentences(content) if self.config.sentence_reranking else [content]
            # LoCoMo passages carry DATE/CONVERSATION headers. The date is useful
            # as metadata on every utterance, but a bare header is not an answer
            # sentence and otherwise wins field filtering for temporal queries.
            content_sentences = [
                sentence for sentence in sentences
                if not re.match(r"^(?:DATE|CONVERSATION)\s*:", sentence, re.IGNORECASE)
            ]
            if content_sentences:
                sentences = content_sentences
            scored = [(sentence, field_match_score(query_field, sentence, query)) for sentence in sentences]
            if query_field != "general" and any(score > 0 for _, score in scored):
                best_field_match = max(score for _, score in scored)
                scored = [(sentence, score) for sentence, score in scored if score == best_field_match]
            sentence_groups.append(scored)
            flattened.extend(format_reranker_sentence(message, sentence, query_field) for sentence, _ in scored)

        reranking_enabled = self.reranker is not None and self.config.reranker_weight > 0
        flat_scores = (
            self.reranker.score(f"Requested field: {query_field}. {query}", flattened)
            if reranking_enabled and flattened
            else np.zeros(len(flattened), dtype=np.float32)
        )
        if len(flat_scores) != len(flattened):
            raise ValueError("Reranker returned a score count that does not match the sentence count.")

        candidate_scores: list[float] = []
        offset = 0
        for scored in sentence_groups:
            group_scores = flat_scores[offset:offset + len(scored)]
            if scored:
                if reranking_enabled:
                    best = max(range(len(scored)), key=lambda index: (float(group_scores[index]), scored[index][1], -index))
                else:
                    best = max(range(len(scored)), key=lambda index: (scored[index][1], -index))
                selected_sentences.append(scored[best][0])
                selected_field_scores.append(scored[best][1])
                candidate_scores.append(float(group_scores[best]))
            else:
                selected_sentences.append("")
                selected_field_scores.append(0.0)
                candidate_scores.append(0.0)
            offset += len(scored)

        return (
            np.asarray(candidate_scores, dtype=np.float32),
            selected_sentences,
            np.asarray(selected_field_scores, dtype=np.float32),
        )

    def _diverse_order(
        self,
        candidate_ids: Sequence[int],
        scores: np.ndarray,
        top_k: int,
        *,
        apply_penalties: bool = True,
    ) -> list[int]:
        """Select high-scoring evidence without filling Top-K with paraphrased replies."""
        remaining = set(range(len(candidate_ids)))
        selected: list[int] = []
        token_sets = [
            self._meaningful_tokens(str(self.messages[index].get("content", "") or ""))
            for index in candidate_ids
        ]
        dialog_sets = [
            {str(value) for value in self.messages[index].get("dialog_ids", []) if str(value)}
            for index in candidate_ids
        ]
        authors = [
            {normalize_label(author) for author in self._message_authors(self.messages[index])}
            for index in candidate_ids
        ]
        diversity_weight = max(0.0, float(self.config.diversity_weight)) if apply_penalties else 0.0
        author_penalty = max(0.0, float(self.config.same_author_penalty)) if apply_penalties else 0.0

        while remaining and len(selected) < top_k:
            best_index = max(
                remaining,
                key=lambda index: (
                    float(scores[index])
                    - diversity_weight * max(
                        (
                            self._jaccard(dialog_sets[index], dialog_sets[chosen])
                            if dialog_sets[index] and dialog_sets[chosen]
                            else self._jaccard(token_sets[index], token_sets[chosen])
                            for chosen in selected
                        ),
                        default=0.0,
                    )
                    - author_penalty * sum(
                        bool(authors[index]) and bool(authors[index] & authors[chosen])
                        for chosen in selected
                    ),
                    float(scores[index]),
                    -index,
                ),
            )
            selected.append(best_index)
            remaining.remove(best_index)
        return selected

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def save(self, directory: str | Path) -> None:
        self._require_built()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(target / "adjacency.npz", self.adjacency)
        metadata = {
            "index_format_version": 1,
            "config": asdict(self.config),
            "message_ids": self.message_ids,
            "user_texts": self.user_texts,
            "role_texts": self.role_texts,
            "context_keys": [list(key) for key in self.context_keys],
            "fingerprint": self.fingerprint(),
        }
        (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        messages: Sequence[dict[str, Any]],
        config: UserContextGraphConfig | None = None,
        reranker: Reranker | None = None,
    ) -> "UserContextGraph":
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        stored_config = UserContextGraphConfig(**metadata["config"])
        active_config = config or stored_config
        changed_structure = [
            name for name in STRUCTURAL_CONFIG_FIELDS
            if not math.isclose(float(getattr(stored_config, name)), float(getattr(active_config, name)))
        ]
        if changed_structure:
            raise ValueError(
                "Cached User-Context Graph was built with different structural settings: "
                + ", ".join(changed_structure)
                + ". Rebuild the index instead of using --reuse-index."
            )
        graph = cls(active_config, reranker=reranker)
        graph.messages = graph._normalize_messages(messages)
        graph.message_ids = [str(message["msg_node"]) for message in graph.messages]
        if graph.message_ids != [str(value) for value in metadata["message_ids"]]:
            raise ValueError("Cached User-Context Graph message order does not match the input corpus.")
        if graph.fingerprint() != metadata.get("fingerprint"):
            raise ValueError("Cached User-Context Graph belongs to a different conversation corpus.")
        graph.message_index = {message_id: index for index, message_id in enumerate(graph.message_ids)}
        graph.user_texts = [str(value) for value in metadata["user_texts"]]
        graph.role_texts = [str(value) for value in metadata["role_texts"]]
        graph.context_keys = [tuple(str(part) for part in value) for value in metadata["context_keys"]]
        graph._initialize_nodes()
        graph.adjacency = sparse.load_npz(source / "adjacency.npz").tocsr().astype(np.float32)
        if graph.adjacency.shape != (len(graph.node_names), len(graph.node_names)):
            raise ValueError("Cached User-Context Graph adjacency shape does not match cached nodes.")
        graph._initialize_transition()
        graph._initialize_lexical_index()
        graph._built = True
        return graph

    def fingerprint(self) -> str:
        return MemoryGraph.corpus_fingerprint(self.messages)

    def _reset(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.message_ids: list[str] = []
        self.message_index: dict[str, int] = {}
        self.user_texts: list[str] = []
        self.role_texts: list[str] = []
        self.context_keys: list[tuple[str, str, str]] = []
        self.user_index: dict[str, int] = {}
        self.role_index: dict[str, int] = {}
        self.context_index: dict[tuple[str, str, str], int] = {}
        self.node_names: list[str] = []
        self.node_index: dict[str, int] = {}
        self.adjacency: sparse.csr_matrix | None = None
        self.transition: sparse.csr_matrix | None = None
        self.dangling_mask = np.zeros(0, dtype=bool)
        self._context_bm25: BM25Retriever | None = None
        self._built = False

    def _initialize_nodes(self) -> None:
        self.user_index = {normalize_label(value): index for index, value in enumerate(self.user_texts)}
        self.role_index = {normalize_label(value): index for index, value in enumerate(self.role_texts)}
        self.context_index = {key: index for index, key in enumerate(self.context_keys)}
        self.node_names = [f"message:{message_id}" for message_id in self.message_ids]
        self.node_names.extend(f"user:{normalize_label(value)}" for value in self.user_texts)
        self.node_names.extend(f"role:{normalize_label(value)}" for value in self.role_texts)
        self.node_names.extend(f"context:{'|'.join(key)}" for key in self.context_keys)
        self.node_index = {name: index for index, name in enumerate(self.node_names)}

    def _initialize_lexical_index(self) -> None:
        documents = []
        for message in self.messages:
            authors = self._message_authors(message)
            safe_context = " ".join([
                *(self._author_token(author) for author in authors),
                *authors,
                *(str(message.get(key, "") or "") for key in ("role", "_channel", "phase_name", "topic")),
            ])
            documents.append({
                "msg_node": str(message["msg_node"]),
                "content": f"{safe_context} {message.get('content', '')}".strip(),
            })
        self._context_bm25 = BM25Retriever(documents)

    def _initialize_transition(self) -> None:
        if self.adjacency is None:
            return
        degrees = np.asarray(self.adjacency.sum(axis=1)).ravel()
        inverse = np.zeros_like(degrees, dtype=np.float32)
        nonzero = degrees > 0
        inverse[nonzero] = 1.0 / degrees[nonzero]
        self.dangling_mask = ~nonzero
        self.transition = sparse.diags(inverse) @ self.adjacency

    def _seed_matching_contexts(self, reset: np.ndarray, query: str) -> None:
        query_terms = self._meaningful_tokens(query)
        if not query_terms:
            return
        matches: list[tuple[int, float]] = []
        for context_key, context_idx in self.context_index.items():
            terms = self._meaningful_tokens(" ".join(context_key))
            overlap = len(query_terms & terms)
            if overlap:
                matches.append((context_idx, overlap / max(1.0, math.sqrt(len(terms)))))
        for context_idx, score in sorted(matches, key=lambda item: item[1], reverse=True)[:8]:
            reset[self._context_node_by_index(context_idx)] += self.config.context_seed_weight * score

        for role_key, role_idx in self.role_index.items():
            role_terms = self._meaningful_tokens(role_key)
            overlap = len(query_terms & role_terms)
            if overlap:
                reset[self._role_node_by_index(role_idx)] += self.config.context_seed_weight * overlap

    def _asker_compatibility(self, message: dict[str, Any], asker_key: str, is_self_query: bool) -> float:
        if not asker_key:
            return 0.0
        author_keys = {normalize_label(author) for author in self._message_authors(message)}
        if is_self_query and asker_key in author_keys:
            return 1.0
        score = 0.0
        parent_id = str(message.get("reply_to") or "").strip()
        parent_idx = self.message_index.get(parent_id)
        if parent_idx is not None and asker_key in {
            normalize_label(author) for author in self._message_authors(self.messages[parent_idx])
        }:
            score = max(score, 0.60 if is_self_query else 0.30)
        mentioned = {normalize_label(value) for value in USER_ID_RE.findall(str(message.get("content", "")))}
        if asker_key in mentioned:
            score = max(score, 0.40 if is_self_query else 0.20)
        if not is_self_query and asker_key in author_keys:
            score = max(score, 0.10)
        return score

    def _query_profile(self, query: str, is_self_query: bool) -> list[str]:
        profile: list[str] = []
        if is_self_query:
            profile.append("self")
        if CONTEXT_QUERY_RE.search(query) or any(
            self._meaningful_tokens(query) & self._meaningful_tokens(" ".join(context_key))
            for context_key in self.context_keys
        ):
            profile.append("contextual")
        if TEMPORAL_QUERY_RE.search(query):
            profile.append("temporal")
        if STATE_QUERY_RE.search(query):
            profile.append("state")
        if EXACT_QUERY_RE.search(query):
            profile.append("exact")
        return profile or ["general"]

    def _fusion_weights(self, query_profile: Sequence[str], query_field: str = "general") -> dict[str, float]:
        weights = {
            "memory": max(0.0, float(self.config.memory_weight)),
            "context_bm25": max(0.0, float(self.config.context_bm25_weight)),
            "user_ppr": max(0.0, float(self.config.user_ppr_weight)),
            "compatibility": max(0.0, float(self.config.compatibility_weight)),
            "context_match": max(0.0, float(self.config.context_match_weight)),
            "evidence_state": max(0.0, float(self.config.evidence_state_weight)),
            "reranker": max(0.0, float(self.config.reranker_weight)) if self.reranker is not None else 0.0,
            "field_match": max(0.0, float(self.config.field_match_weight)) if query_field != "general" else 0.0,
        }
        if not self.config.candidate_refinement:
            for name in ("compatibility", "context_match", "evidence_state", "reranker", "field_match"):
                weights[name] = 0.0
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("Candidate refinement needs a positive base retrieval weight.")
            return {name: value / total for name, value in weights.items()}
        if self.config.query_aware_fusion:
            if "self" in query_profile:
                weights["user_ppr"] *= max(0.0, float(self.config.self_query_ppr_scale))
                weights["compatibility"] *= max(0.0, float(self.config.self_query_compatibility_scale))
            if "contextual" in query_profile:
                weights["context_match"] *= max(0.0, float(self.config.contextual_query_context_scale))
            else:
                weights["context_match"] = 0.0
            if {"temporal", "state", "exact"} & set(query_profile):
                weights["evidence_state"] *= max(0.0, float(self.config.state_query_evidence_scale))
            else:
                weights["evidence_state"] = 0.0
        total = sum(weights.values())
        if total <= 0:
            return {name: 0.0 for name in weights}
        return {name: value / total for name, value in weights.items()}

    def _context_match_score(self, query: str, message: dict[str, Any]) -> float:
        query_terms = self._meaningful_tokens(query)
        if not query_terms:
            return 0.0
        field_scores: list[float] = []
        for key in ("role", "_channel", "phase_name", "topic"):
            terms = self._meaningful_tokens(str(message.get(key, "") or ""))
            if terms:
                field_scores.append(len(query_terms & terms) / len(terms))
        return min(1.0, max(field_scores, default=0.0))

    @classmethod
    def _evidence_state(
        cls,
        message: dict[str, Any],
        query_profile: Sequence[str],
        query: str = "",
    ) -> tuple[float, str]:
        if not ({"temporal", "state", "exact"} & set(query_profile)):
            return 0.5, "not_applicable"
        content = str(message.get("content", "") or "")
        evidence_text = cls._query_aligned_evidence_text(content, query)
        if UNAVAILABLE_EVIDENCE_RE.search(evidence_text):
            # Preserve explicit unavailability as a traceable state, but do not
            # rank it above a message that contains a confirmed value.
            return 0.25, "explicitly_unavailable"
        if TENTATIVE_EVIDENCE_RE.search(evidence_text) and not STRONG_CONFIRMED_EVIDENCE_RE.search(evidence_text):
            return 0.0, "tentative"
        if CONFIRMED_EVIDENCE_RE.search(evidence_text):
            return 1.0, "confirmed"
        return 0.5, "neutral"

    @classmethod
    def _query_aligned_evidence_text(cls, content: str, query: str) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", content)
            if sentence.strip()
        ]
        query_terms = cls._meaningful_tokens(query)
        if not sentences or not query_terms:
            return content
        ranked: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            terms = cls._meaningful_tokens(sentence)
            overlap = len(query_terms & terms)
            if overlap:
                ranked.append((overlap / max(1.0, math.sqrt(len(terms))), -index, sentence))
        if not ranked:
            return content
        return sorted(ranked, reverse=True)[0][2]

    def _context_hits(self, query: str, message: dict[str, Any]) -> list[str]:
        query_terms = self._meaningful_tokens(query)
        hits: list[str] = []
        for key, label in (("role", "role"), ("_channel", "channel"), ("phase_name", "phase"), ("topic", "topic")):
            value = str(message.get(key, "") or "").strip()
            if value and query_terms & self._meaningful_tokens(value):
                hits.append(f"{label}:{value}")
        return hits

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

    def _user_node(self, user_key: str) -> int:
        return len(self.messages) + self.user_index[user_key]

    def _role_node(self, role_key: str) -> int:
        return self._role_node_by_index(self.role_index[role_key])

    def _role_node_by_index(self, role_idx: int) -> int:
        return len(self.messages) + len(self.user_texts) + role_idx

    def _context_node(self, context_key: tuple[str, str, str]) -> int:
        return self._context_node_by_index(self.context_index[context_key])

    def _context_node_by_index(self, context_idx: int) -> int:
        return len(self.messages) + len(self.user_texts) + len(self.role_texts) + context_idx

    @staticmethod
    def _context_key(message: dict[str, Any]) -> tuple[str, str, str]:
        return (
            normalize_label(message.get("_channel")),
            normalize_label(message.get("phase_name")),
            normalize_label(message.get("topic")),
        )

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}

    @staticmethod
    def _author_token(value: Any) -> str:
        normalized = normalize_label(value).replace(" ", "_")
        return f"author_{normalized}" if normalized else ""

    @staticmethod
    def _message_authors(message: dict[str, Any]) -> list[str]:
        values = message.get("authors")
        if isinstance(values, (list, tuple, set)):
            authors = [str(value).strip() for value in values if str(value).strip()]
            if authors:
                return list(dict.fromkeys(authors))
        author = str(message.get("author", "") or "").strip()
        return [author] if author else []

    @staticmethod
    def _normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(messages):
            item = dict(raw)
            message_id = str(item.get("msg_node") or f"Msg_{index}")
            if message_id in seen:
                raise ValueError(f"Duplicate msg_node: {message_id}")
            seen.add(message_id)
            item["msg_node"] = message_id
            item["content"] = str(item.get("content", ""))
            normalized.append(item)
        return normalized

    @staticmethod
    def _build_sparse_adjacency(node_count: int, edge_weights: dict[tuple[int, int], float]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for (left, right), weight in edge_weights.items():
            rows.extend((left, right))
            cols.extend((right, left))
            values.extend((weight, weight))
        return sparse.csr_matrix((values, (rows, cols)), shape=(node_count, node_count), dtype=np.float32)

    def _require_built(self) -> None:
        if not self._built or self.adjacency is None or self._context_bm25 is None:
            raise RuntimeError("UserContextGraph is not built. Call build(messages) first.")


class DualGraphRetriever:
    """Fuse Memory Graph candidates with User-Context Graph evidence."""

    def __init__(self, memory_graph: MemoryGraph, user_context_graph: UserContextGraph):
        if memory_graph.fingerprint() != user_context_graph.fingerprint():
            raise ValueError("Memory Graph and User-Context Graph must index the same corpus.")
        self.memory_graph = memory_graph
        self.user_context_graph = user_context_graph
        self.messages = memory_graph.messages
        self.message_ids = memory_graph.message_ids
        self.entity_texts = memory_graph.entity_texts

    def retrieve(self, question: str, top_k: int = 10, asking_user_id: str = "") -> list[UserContextSearchResult]:
        pool_size = max(int(top_k), int(self.user_context_graph.config.candidate_pool))
        memory_results = self.memory_graph.retrieve(question, top_k=pool_size)
        return self.user_context_graph.retrieve(
            question,
            asking_user_id,
            memory_results=memory_results,
            top_k=top_k,
        )
