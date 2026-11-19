"""Entity extraction adapters for the memory graph."""

from __future__ import annotations

import re
from typing import Iterable, Sequence


def normalize_entity(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


class RegexEntityExtractor:
    """Dependency-free fallback for smoke runs.

    It intentionally favors explicit identifiers, dates, acronyms, and
    capitalized names. It is not a replacement for a trained NER model.
    """

    _patterns = (
        re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}(?:[T ][0-9:.-]+)?\b"),
        re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b"),
        re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)+\b"),
        re.compile(r"\b[A-Z]{2,}(?:[0-9]+)?\b"),
        re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"),
    )

    provider = "regex"
    model_name = "regex-v1"

    def extract(self, text: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for pattern in self._patterns:
            for match in pattern.findall(str(text or "")):
                value = str(match).strip()
                key = normalize_entity(value)
                if re.fullmatch(r"user_\d+", key):
                    continue
                if key and key not in seen:
                    seen.add(key)
                    found.append(value)
        return found

    def batch_extract(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.extract(text) for text in texts]


class SpacyEntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm", device: str = "cpu", batch_size: int = 64):
        try:
            import spacy
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("spaCy is unavailable; use --entity-extractor regex for a smoke run.") from exc
        if device == "cuda":
            spacy.require_gpu()
        elif device == "auto":
            spacy.prefer_gpu()
        self.nlp = spacy.load(model_name)
        self.provider = "spacy"
        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))

    def batch_extract(self, texts: Sequence[str]) -> list[list[str]]:
        output: list[list[str]] = []
        documents = self.nlp.pipe(list(texts), batch_size=self.batch_size)
        if len(texts) >= 1000:
            from tqdm.auto import tqdm

            documents = tqdm(documents, total=len(texts), desc="spaCy NER", unit="message", dynamic_ncols=True)
        for doc in documents:
            values: list[str] = []
            seen: set[str] = set()
            for ent in doc.ents:
                if ent.label_ in {"CARDINAL", "ORDINAL"}:
                    continue
                value = ent.text.strip()
                key = normalize_entity(value)
                if key and key not in seen:
                    seen.add(key)
                    values.append(value)
            output.append(values)
        return output


def build_entity_extractor(kind: str, model_name: str, device: str = "cpu", batch_size: int = 64):
    kind = str(kind or "regex").strip().lower()
    if kind in {"regex", "local", "test"}:
        return RegexEntityExtractor()
    if kind in {"spacy", "spaCy".lower()}:
        return SpacyEntityExtractor(model_name, device=device, batch_size=batch_size)
    raise ValueError(f"Unknown entity extractor: {kind}")
