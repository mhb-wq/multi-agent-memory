"""Shared GroupMemBench reader, judge, and result aggregation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .llm import ChatModel


DATE_OUTPUT_REQUEST_RE = re.compile(
    r"(?:yyyy\s*-\s*mm\s*-\s*dd|iso(?:-?8601)?\s+date|exact\s+(?:calendar\s+)?date)",
    re.IGNORECASE,
)
DATE_QUESTION_RE = re.compile(
    r"\b(?:when|deadline|due\s+date|what\s+date|which\s+date|on\s+what\s+date|by\s+what\s+date)\b",
    re.IGNORECASE,
)
DATETIME_OUTPUT_REQUEST_RE = re.compile(r"yyyy\s*-\s*mm\s*-\s*dd\s+hh\s*:\s*mm", re.IGNORECASE)
RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
TIMESTAMP_TAG_RE = re.compile(r"\btimestamp=([^/\]\r\n]+)", re.IGNORECASE)
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def complete_with_metadata(model: ChatModel, messages: Sequence[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    method = getattr(model, "complete_with_metadata", None)
    if callable(method):
        completion = method(messages)
        return str(completion.text), dict(completion.metadata)
    return model.complete(messages), {}


def extract_final(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in str(text).splitlines()]
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        lower = line.lower()
        if lower.startswith("final answer:") or lower.startswith("final:"):
            first_line = line.split(":", 1)[1].strip()
            continuation = "\n".join(value.strip() for value in lines[index + 1:] if value.strip())
            return "\n".join(value for value in (first_line, continuation) if value).strip()
    nonempty = [line.strip() for line in lines if line.strip()]
    return nonempty[-1] if nonempty else ""


def split_reasoning_and_final(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    lines = [line.rstrip() for line in str(text).splitlines()]
    final_index = None
    for index in range(len(lines) - 1, -1, -1):
        lower = lines[index].strip().lower()
        if lower.startswith("final:") or lower.startswith("final answer:"):
            final_index = index
            break
    if final_index is None:
        return str(text).strip(), extract_final(text)
    reasoning = "\n".join(lines[:final_index]).strip()
    first_line = lines[final_index].split(":", 1)[1].strip()
    continuation = "\n".join(line.strip() for line in lines[final_index + 1:] if line.strip())
    final = "\n".join(value for value in (first_line, continuation) if value).strip()
    return reasoning, final


def parse_judgment(text: str) -> bool | None:
    final = extract_final(text).strip().lower()
    if "incorrect" in final or "wrong" in final or "not correct" in final:
        return False
    if "correct" in final:
        return True
    return None


def _resolve_relative_date(expression: str, timestamp: datetime) -> str:
    normalized = " ".join(str(expression).lower().split())
    if normalized == "today":
        value = timestamp.date()
    elif normalized == "tomorrow":
        value = timestamp.date() + timedelta(days=1)
    else:
        weekday = normalized.removeprefix("this ")
        offset = (WEEKDAYS[weekday] - timestamp.weekday()) % 7
        value = timestamp.date() + timedelta(days=offset)
    return value.isoformat()


def normalize_relative_date_answer(
    question: str,
    answer: str,
    retrieved_docs: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Resolve a relative answer only when matching timestamped evidence is unique."""
    trace: dict[str, Any] = {
        "status": "not_requested",
        "expression": "",
        "resolved_dates": [],
        "source_timestamps": [],
    }
    question_text = str(question or "")
    if DATETIME_OUTPUT_REQUEST_RE.search(question_text):
        trace["status"] = "datetime_not_supported"
        return answer, trace
    if not (DATE_OUTPUT_REQUEST_RE.search(question_text) or DATE_QUESTION_RE.search(question_text)):
        return answer, trace

    match = RELATIVE_DATE_RE.search(str(answer or ""))
    if match is None:
        trace["status"] = "answer_not_relative"
        return answer, trace

    expression = " ".join(match.group(1).lower().split())
    lookup = expression.removeprefix("this ")
    evidence_expression_re = re.compile(rf"\b{re.escape(lookup)}\b", re.IGNORECASE)
    resolved: set[str] = set()
    timestamps: list[str] = []
    for document in retrieved_docs:
        text = str(document or "")
        timestamp_match = TIMESTAMP_TAG_RE.search(text)
        content = text.split("]", 1)[1] if "]" in text else text
        if timestamp_match is None or not evidence_expression_re.search(content):
            continue
        timestamp_text = timestamp_match.group(1).strip()
        try:
            timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        except ValueError:
            continue
        timestamps.append(timestamp_text)
        resolved.add(_resolve_relative_date(expression, timestamp))

    trace.update({
        "expression": match.group(0),
        "resolved_dates": sorted(resolved),
        "source_timestamps": timestamps,
    })
    if len(resolved) == 1:
        normalized_answer = next(iter(resolved))
        trace["status"] = "normalized"
        trace["normalized_answer"] = normalized_answer
        return normalized_answer, trace
    trace["status"] = "ambiguous" if resolved else "matching_evidence_not_found"
    return answer, trace


def evaluate_record(
    record: dict[str, Any],
    *,
    agent: ChatModel,
    judge: ChatModel | None,
    agent_system_prompt: str,
    judge_system_prompt: str,
) -> dict[str, Any]:
    question = str(record.get("question", ""))
    gold_answer = str(record.get("answer", ""))
    asking_user_id = str(record.get("asking_user_id", ""))
    retrieved_docs = list(record.get("retrieved_docs", []))
    passages = "\n\n".join(f"[{index}] {document}" for index, document in enumerate(retrieved_docs, start=1))
    asker_line = f"Asking user: {asking_user_id}\n\n" if asking_user_id else ""
    agent_user_prompt = (
        f"{asker_line}Question:\n{question}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Answer the question using the retrieved passages."
    )
    agent_output, agent_response = complete_with_metadata(agent, [
        {"role": "system", "content": agent_system_prompt},
        {"role": "user", "content": agent_user_prompt},
    ])
    agent_reasoning, agent_answer_raw = split_reasoning_and_final(agent_output)
    if not agent_answer_raw.strip():
        raise ValueError(f"Agent returned no final answer; response metadata: {agent_response}")
    agent_answer, answer_normalization = normalize_relative_date_answer(
        question,
        agent_answer_raw,
        retrieved_docs,
    )
    result = dict(record)
    result.update({
        "query": question,
        "agent_reasoning": agent_reasoning,
        "agent_answer_raw": agent_answer_raw,
        "agent_answer": agent_answer,
        "answer_normalization": answer_normalization,
        "agent_response": agent_response,
        "judge_reasoning": "",
        "judge_answer": "",
        "judge_correct": None,
        "judge_response": {},
        "status": "ok",
    })
    if judge is not None:
        judge_user_prompt = (
            f"Question:\n{question}\n\n"
            f"Gold Answer:\n{gold_answer}\n\n"
            f"Agent Answer:\n{agent_answer}\n"
        )
        judge_output, judge_response = complete_with_metadata(judge, [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt},
        ])
        judge_reasoning, judge_answer = split_reasoning_and_final(judge_output)
        if not judge_answer.strip():
            raise ValueError(f"Judge returned no final verdict; response metadata: {judge_response}")
        result["judge_reasoning"] = judge_reasoning
        result["judge_answer"] = judge_answer
        result["judge_correct"] = parse_judgment(judge_output)
        result["judge_response"] = judge_response
    return result


def load_completed_ids(path: str | Path, require_judge: bool) -> set[str]:
    source = Path(path)
    if not source.exists():
        return set()
    completed: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") != "ok" or not record.get("agent_answer"):
            continue
        if require_judge and not str(record.get("judge_answer", "")).strip():
            continue
        completed.add(str(record.get("id", "")))
    return completed


def sort_and_deduplicate_results(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_id[str(record.get("id", ""))] = record
    records = sorted(by_id.values(), key=lambda item: int(item.get("question_index", 0)))
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def summarize_judgments(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("status") == "ok" and str(record.get("judge_answer", "")).strip()]
    correct = sum(record.get("judge_correct") is True for record in evaluated)
    incorrect = sum(record.get("judge_correct") is False for record in evaluated)
    unclear = len(evaluated) - correct - incorrect
    return {
        "correct": correct,
        "incorrect": incorrect,
        "unclear": unclear,
        "total": len(evaluated),
        "accuracy": correct / len(evaluated) if evaluated else 0.0,
    }
