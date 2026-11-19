"""Read official benchmark inputs without exposing evaluation labels to agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .baselines.base import Message


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gate_events(episode: dict, checkpoints: list[dict]) -> Iterator[tuple[str, object]]:
    """Interleave checkpoints after their turn; exclude evaluator-only fields."""
    turns = episode["turns"]
    positions = {str(turn["turn_id"]): i for i, turn in enumerate(turns)}
    if len(positions) != len(turns):
        raise ValueError("duplicate GateMem turn ID")
    by_turn: dict[str, list[dict]] = {}
    for ck in checkpoints:
        if ck["episode_id"] != episode["episode_id"]:
            raise ValueError("checkpoint belongs to another episode")
        if ck["as_of_turn_id"] not in positions:
            raise ValueError("checkpoint turn is missing")
        by_turn.setdefault(ck["as_of_turn_id"], []).append(ck)
    for turn in turns:
        speaker = turn["speaker"]
        yield "ingest", Message(
            id=f"{episode['episode_id']}/{turn['turn_id']}", text=turn["text"],
            author=speaker["principal_id"], role=speaker.get("role", ""),
            timestamp=turn.get("timestamp") or "", channel=episode["episode_id"],
            metadata={"turn_kind": turn.get("turn_kind")},
        )
        for ck in by_turn.get(turn["turn_id"], []):
            yield "query", {
                "checkpoint_id": ck["checkpoint_id"], "as_of_turn_id": ck["as_of_turn_id"],
                "asking_user_id": ck["asker"]["principal_id"],
                "asking_user_role": ck["asker"].get("role", ""),
                "question": ck["query_text"],
            }


def perma_case(event: dict, meta: dict, user_id: str) -> tuple[list[Message], dict]:
    """Use only the supplied as-of context; preferences and gold remain hidden."""
    if str(event["task_id"]) != str(meta["task_id"]) or str(event["type"]) != str(meta["task_type"]):
        raise ValueError("PERMA event and question do not match")
    messages = []
    for session_index, session in enumerate(event.get("context", [])):
        if not isinstance(session, list) or len(session) < 2 or not isinstance(session[0], list):
            raise ValueError("invalid PERMA session")
        for turn_index, turn in enumerate(session[0]):
            role = turn["role"]
            messages.append(Message(
                id=f"{user_id}/{event['task_id']}/{event['type']}/{session_index}/{turn_index}",
                author=user_id if role == "user" else "assistant", role=role,
                timestamp=str(session[1]), text=turn["content"], channel=str(session_index),
            ))
    query = {"task_id": meta["task_id"], "asking_user_id": user_id,
             "question": meta["question"], "question_date": meta["question_date"],
             "options": meta["options"]}
    return messages, query


def perma_answer_context(event: dict, passages: list) -> str:
    """The official answer stage substitutes no memory for type 1."""
    if int(event["type"]) == 1:
        return "No memory found."
    return "\n\n".join(p.text for p in passages) if passages else "No memory found."
