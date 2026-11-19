"""LoCoMo data conversion, reader prompting, and string-only metrics."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from nltk.stem import PorterStemmer


CATEGORY_NAMES = {
    4: "SingleHop",
    1: "MultiHop",
    2: "Temporal",
    3: "OpenDomain",
}
CATEGORY_ORDER = (4, 1, 2, 3)
PROMPT_FILES = {
    1: "locomo_multihop_temporal.txt",
    2: "locomo_multihop_temporal.txt",
    3: "locomo_opendomain.txt",
    4: "locomo_singlehop.txt",
}
SESSION_RE = re.compile(r"^session_(\d+)$")
_STEMMER = PorterStemmer()


def load_locomo(path: str | Path) -> list[dict[str, Any]]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("A LoCoMo data file must contain a JSON list.")
    return [dict(sample) for sample in data]


def _session_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%I:%M %p on %d %B, %Y").isoformat()
    except ValueError:
        return text


def _turn_text(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker", "Unknown")).strip() or "Unknown"
    dialog_id = str(turn.get("dia_id", "")).strip()
    text = str(turn.get("text", "")).strip()
    prefix = f"[{dialog_id}] " if dialog_id else ""
    lines = [f"{prefix}{speaker}: {text}".strip()]
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        lines.append(f"[Shared image: {caption}]")
    return "\n".join(lines)


def conversation_windows(
    sample: dict[str, Any],
    *,
    window_size: int = 6,
    stride: int = 3,
    min_tail_size: int = 4,
) -> list[dict[str, Any]]:
    """Convert one LoCoMo conversation into session-local sliding windows.

    A final partial window is kept only when it contains at least
    ``min_tail_size`` turns. Sessions shorter than that are retained as one
    window so no source conversation disappears.
    """

    if window_size < 1 or stride < 1 or min_tail_size < 1:
        raise ValueError("window_size, stride, and min_tail_size must be positive")
    if min_tail_size > window_size:
        raise ValueError("min_tail_size cannot exceed window_size")

    sample_id = str(sample.get("sample_id", "sample")).strip() or "sample"
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError(f"LoCoMo sample {sample_id!r} has no conversation object")

    sessions: list[tuple[int, str, list[dict[str, Any]]]] = []
    for key, value in conversation.items():
        match = SESSION_RE.fullmatch(str(key))
        if not match or not isinstance(value, list) or not value:
            continue
        sessions.append((int(match.group(1)), str(key), [dict(turn) for turn in value if isinstance(turn, dict)]))
    sessions.sort(key=lambda item: item[0])

    windows: list[dict[str, Any]] = []
    for session_number, session_key, turns in sessions:
        if not turns:
            continue
        previous_id = ""
        date_text = str(conversation.get(f"{session_key}_date_time", "")).strip()
        if len(turns) < min_tail_size:
            starts = [0]
        else:
            starts = list(range(0, len(turns) - min_tail_size + 1, stride))
        for start in starts:
            chunk = turns[start:start + window_size]
            if not chunk:
                continue
            dialog_ids = [str(turn.get("dia_id", "")).strip() for turn in chunk if turn.get("dia_id")]
            speakers = list(dict.fromkeys(
                str(turn.get("speaker", "")).strip() for turn in chunk if str(turn.get("speaker", "")).strip()
            ))
            window_id = f"{sample_id}:S{session_number}:W{start // stride + 1}"
            content = f"DATE: {date_text}\nCONVERSATION:\n" + "\n".join(_turn_text(turn) for turn in chunk)
            message = {
                "msg_node": window_id,
                "content": content,
                "author": speakers[0] if speakers else "",
                "authors": speakers,
                "role": "conversation participant",
                "timestamp": _session_timestamp(date_text),
                "reply_to": previous_id,
                "phase_name": session_key,
                "topic": "LoCoMo conversation",
                "_channel": session_key,
                "sample_id": sample_id,
                "session_number": session_number,
                "window_start": start,
                "dialog_ids": dialog_ids,
                "date_time": date_text,
            }
            windows.append(message)
            previous_id = window_id
    return windows


def prompt_path(prompt_dir: str | Path, category: int) -> Path:
    try:
        filename = PROMPT_FILES[int(category)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported LoCoMo category: {category}") from exc
    return Path(prompt_dir) / filename


def build_reader_user_prompt(question: str, passages: Sequence[str], category: int) -> str:
    context = "\n\n".join(f"Evidence {index}:\n{passage}" for index, passage in enumerate(passages, 1))
    temporal = (
        "\nUse the DATE attached to each conversation as well as relative time expressions in the dialogue."
        if int(category) == 2
        else ""
    )
    return (
        f"Question:\n{question}{temporal}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Return only the final short answer."
    )


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower().replace(",", "")
    text = "".join(character for character in text if character not in set(string.punctuation))
    tokens = [token for token in text.split() if token not in {"a", "an", "the", "and"}]
    return " ".join(tokens)


def _stemmed_tokens(value: Any) -> list[str]:
    return [_STEMMER.stem(token) for token in normalize_answer(value).split()]


def token_f1(prediction: Any, reference: Any) -> float:
    prediction_tokens = _stemmed_tokens(prediction)
    reference_tokens = _stemmed_tokens(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def locomo_f1(prediction: Any, reference: Any, category: int) -> float:
    reference_text = str(reference or "")
    if int(category) == 3:
        reference_text = reference_text.split(";", 1)[0].strip()
    if int(category) != 1:
        return token_f1(prediction, reference_text)
    predictions = [item.strip() for item in str(prediction or "").split(",") if item.strip()] or [""]
    references = [item.strip() for item in reference_text.split(",") if item.strip()] or [""]
    return sum(max(token_f1(candidate, gold) for candidate in predictions) for gold in references) / len(references)


def bleu1(prediction: Any, reference: Any, category: int) -> float:
    """Standard sentence BLEU-1: clipped unigram precision with brevity penalty."""

    reference_text = str(reference or "")
    if int(category) == 3:
        reference_text = reference_text.split(";", 1)[0].strip()
    prediction_tokens = _stemmed_tokens(prediction)
    reference_tokens = _stemmed_tokens(reference_text)
    if not prediction_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(prediction_tokens)
    if precision <= 0.0:
        return 0.0
    brevity_penalty = 1.0 if len(prediction_tokens) >= len(reference_tokens) else math.exp(
        1.0 - len(reference_tokens) / len(prediction_tokens)
    )
    return brevity_penalty * precision


def score_answer(prediction: Any, reference: Any, category: int) -> dict[str, float]:
    return {
        "f1": locomo_f1(prediction, reference, category),
        "bleu1": bleu1(prediction, reference, category),
    }


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        record for record in records
        if record.get("status") == "ok" and int(record.get("category", 0) or 0) in CATEGORY_NAMES
    ]
    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        subset = [record for record in completed if int(record.get("category", 0)) == category]
        by_category[CATEGORY_NAMES[category]] = {
            "category": category,
            "count": len(subset),
            "f1": sum(float(record.get("f1", 0.0)) for record in subset) / len(subset) if subset else 0.0,
            "bleu1": sum(float(record.get("bleu1", 0.0)) for record in subset) / len(subset) if subset else 0.0,
        }
    micro = {
        "count": len(completed),
        "f1": sum(float(record.get("f1", 0.0)) for record in completed) / len(completed) if completed else 0.0,
        "bleu1": sum(float(record.get("bleu1", 0.0)) for record in completed) / len(completed) if completed else 0.0,
    }
    available = [value for value in by_category.values() if value["count"]]
    macro = {
        "count": len(available),
        "f1": sum(value["f1"] for value in available) / len(available) if available else 0.0,
        "bleu1": sum(value["bleu1"] for value in available) / len(available) if available else 0.0,
    }
    return {"categories": by_category, "Average": micro, "macro_average": macro}


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo results",
        "",
        "Average is the question-weighted mean over categories 1–4; category 5 is excluded.",
        "",
        "| Split | Count | F1 | BLEU-1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("SingleHop", "MultiHop", "Temporal", "OpenDomain"):
        value = summary["categories"][name]
        lines.append(f"| {name} | {value['count']} | {value['f1']:.4f} | {value['bleu1']:.4f} |")
    average = summary["Average"]
    lines.append(f"| **Average** | **{average['count']}** | **{average['f1']:.4f}** | **{average['bleu1']:.4f}** |")
    lines.append("")
    return "\n".join(lines)
