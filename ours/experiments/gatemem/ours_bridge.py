"""GateMem runner bridge for an explicitly configured Ours memory instance.

The caller must supply isolated memory, a replay clock and an independently
verified control bridge; this module never grants permissions automatically.
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/gatemem/official/code")]

from bench.agents.base import BaseMemoryAgent  # noqa: E402

from ours_memory.control import Actor  # noqa: E402
from ours_memory.models import ValidationError  # noqa: E402


def replay_time(value):
    if not isinstance(value, str) or not value:
        raise ValidationError("GateMem事件缺少时间，不能安全回放")
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class GateOursAgent(BaseMemoryAgent):
    def __init__(self, memory_factory, control_factory, *, llm_router, space_id, maintainer_id, top_k=10):
        if llm_router is None or getattr(llm_router, "provider", "stub") == "stub":
            raise ValueError("不能使用读取题目标准答案的stub")
        super().__init__(top_k=top_k, llm_router=llm_router)
        self.memory_factory, self.control_factory = memory_factory, control_factory
        self.space_id, self.maintainer_id = space_id, maintainer_id

    def reset(self, episode):
        if hasattr(self, "memory"):
            raise ValidationError("每个GateMem episode须使用独立Ours实例并关闭旧存储")
        super().reset(episode)
        self.memory, self.clock = self.memory_factory(episode)
        self.control_bridge = self.control_factory(self.memory, episode)
        self.last_day = None
        self.last_event_time = None

    def _sleep(self, day):
        self.clock.value = f"{day}T23:59:59.999999+00:00"
        result = self.memory.sleep(Actor(self.maintainer_id, "maintainer", self.space_id, day))
        if any(job.get("status") not in {"completed", "disabled"} for job in result.get("jobs", [])):
            raise ValidationError("Sleep未完成，停止回放")
        if any(row.get("task_id") == day for row in self.memory.store.all("pending_state")):
            raise ValidationError("Sleep仍有待处理证据，停止回放")

    def ingest(self, turn):
        moment = replay_time(turn.timestamp)
        day = moment[:10]
        if self.last_event_time and moment < self.last_event_time:
            raise ValidationError("GateMem事件时间倒退，停止回放")
        if self.last_day and day < self.last_day:
            raise ValidationError("GateMem事件不按时间排序")
        if self.last_day and day != self.last_day:
            self._sleep(self.last_day)
        self.clock.value, self.last_day, self.last_event_time = moment, day, moment
        raw = {"turn_id": turn.turn_id, "timestamp": turn.timestamp,
               "speaker": {"principal_id": turn.speaker_principal_id, "role": turn.speaker_role},
               "text": turn.text}
        # A control request is never written as ordinary memory before it is
        # verified and acted on. Ambiguity halts replay in the control bridge.
        if self.control_bridge.process_turn(raw):
            return
        result = self.memory.ingest_message(
            Actor(turn.speaker_principal_id, "ingestor", self.space_id, day),
            message_id=turn.turn_id, text=turn.text, scope="global", observed_at=moment,
            metadata={"role": turn.speaker_role, "timestamp": moment, "msg_node": turn.turn_id},
            fallback_category="conversation",
        )
        if result.get("status") != "done":
            raise ValidationError("GateMem写入任务未完成")

    def query(self, checkpoint):
        who = Actor(checkpoint.asker_principal_id, "answerer", self.space_id, "gate-query")
        self.memory.control.authenticate(who)
        subjects = [s for s in self.memory.control.spaces[self.space_id]["members"]
                    if any(self.memory.control.permits(who, s, c, "read")
                           for c in self.memory.control.categories)]
        if not subjects:
            return {"action": "no_memory", "answer": "No authorized memory is available.",
                    "answer_structured": {}, "used_record_ids": [], "retrieved_memory": []}
        bundle = self.memory.read(who, subject_ids=subjects, query=checkpoint.query_text,
                                  purpose="evidence", target_time=self.clock.value)
        cards = bundle["cards"][:self.top_k]
        if not cards and not bundle.get("user_states"):
            return {"action": "no_memory", "answer": "No authorized memory is available.",
                    "answer_structured": {}, "used_record_ids": [], "retrieved_memory": []}
        if not self.memory.validate_bundle(who, bundle["bundle_id"])["ready"]:
            raise ValidationError("回答前证据包已失效")
        retrieved = [{"text": c.get("claim_text", "") + "\nMemory usage: " + (c.get("usage") or c.get("usage_note") or ""), "record_id": c["id"],
                      "principal_id": c.get("author_id", "")} for c in cards]
        if self.memory.control.access_mode == "shared_pool":
            block = self._format_retrieved_memory_block(retrieved)
            block += "\n\nCurrent user states (pending subjects may have newer cards):\n" + json.dumps(
                {"user_states": bundle.get("user_states", []), "pending_subjects": bundle.get("pending_subjects", [])}, ensure_ascii=False)
            output = self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved, rendered_memory_block_override=block)
        else:
            output = self._run_llm(checkpoint=checkpoint, retrieved_memory=retrieved)
        if not self.memory.validate_bundle(who, bundle["bundle_id"])["ready"]:
            raise ValidationError("回答时证据包已失效")
        return output
