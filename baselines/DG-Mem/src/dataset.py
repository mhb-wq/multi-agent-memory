"""GroupMemBench data loading and passage formatting."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def load_conversation_messages(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    messages: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        raise ValueError("A GroupMemBench conversation file must be an object keyed by channel.")
    for channel, channel_messages in data.items():
        if not isinstance(channel_messages, list):
            continue
        for message in channel_messages:
            if not isinstance(message, dict):
                continue
            item = dict(message)
            item["_channel"] = str(channel)
            item["content"] = html.unescape(str(item.get("content", ""))).strip()
            messages.append(item)
    messages.sort(key=lambda item: (
        str(item.get("_channel", "")),
        str(item.get("timestamp", "")),
        str(item.get("msg_node", "")),
    ))
    return messages


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def format_message(message: dict[str, Any]) -> str:
    tags = [f"user={message.get('author') or '?'}"]
    for key, label in (
        ("role", "speaker_role"),
        ("_channel", "channel"),
        ("phase_name", "phase_name"),
        ("topic", "topic"),
        ("timestamp", "timestamp"),
        ("reply_to", "reply_to"),
        ("msg_node", "msg_node"),
    ):
        value = message.get(key)
        if value not in (None, ""):
            tags.append(f"{label}={value}")
    return f"[{ ' / '.join(tags) }]\n{message.get('content', '')}".strip()
