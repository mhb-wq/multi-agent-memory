"""Translate GateMem turns into independently verified Ours control actions.

Proposal generation and verification are injected by the deployment. Neither
the checkpoint labels nor expected actions are used by this bridge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ours_memory.control import Actor  # noqa: E402
from ours_memory.models import ValidationError  # noqa: E402


class GateControlBridge:
    def __init__(self, memory, propose, verify, *, agent_id, space_id, task_id):
        self.memory = memory
        self.propose = propose
        self.verify = verify
        self.agent_id, self.space_id, self.task_id = agent_id, space_id, task_id

    def process_turn(self, raw):
        speaker = raw.get("speaker") or {}
        speaker_id = speaker.get("principal_id")
        text, turn_id = raw.get("text"), raw.get("turn_id")
        if not isinstance(speaker_id, str) or not isinstance(text, str) or not isinstance(turn_id, str):
            raise ValidationError("GateMem事件缺少认证主体、文本或ID")
        turn = {"turn_id": turn_id, "speaker_id": speaker_id, "text": text,
                "timestamp": raw.get("timestamp"), "role": speaker.get("role")}
        response = self.propose(turn)
        if not isinstance(response, dict) or set(response) != {"status", "operations"}:
            raise ValidationError("控制识别器须返回明确状态和操作")
        if response["status"] == "uncertain":
            raise ValidationError("控制事件含义不明确；暂停回放，禁止按普通消息继续")
        if response["status"] not in {"none", "control"} or not isinstance(response["operations"], list):
            raise ValidationError("控制识别器返回无效状态")
        if (response["status"] == "none") != (not response["operations"]):
            raise ValidationError("控制事件状态与操作列表不一致")
        if len(response["operations"]) > 1:
            raise ValidationError("同一事件包含多个控制操作；暂停回放等待安全拆分")
        actor = Actor(speaker_id, self.agent_id, self.space_id, self.task_id)
        results = []
        for proposal in response["operations"]:
            result = self.memory.execute_control_intent(actor, turn, proposal, verifier=self.verify)
            if result.get("status") in {"needs_review", "pending_cleanup"}:
                raise ValidationError("控制事件未完整执行；暂停回放")
            results.append(result)
        return results
