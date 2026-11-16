"""Execute verified control requests derived from authenticated source turns.

The verifier is a trusted component supplied by the integration. A model proposal
is never sufficient by itself to create or revoke a grant or erase a memory.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from .control import Actor
from .models import AccessDenied, ValidationError, stable_id


class ControlIntentExecutor:
    def __init__(self, memory: Any, verify: Callable[[dict, dict], bool]):
        self.memory = memory
        self.verify = verify

    def execute(self, actor: Actor, turn: dict, proposal: dict) -> dict:
        """Validate authenticated provenance, exact quote and trusted verification.

        Fail closed for unresolved actions; do not interpret a missing grant as
        permission. Partial deletions must first be resolved to exact event IDs.
        """
        if not isinstance(turn, dict) or not isinstance(proposal, dict):
            raise ValidationError("控制事件和提议必须为对象")
        text, speaker, turn_id = turn.get("text"), turn.get("speaker_id"), turn.get("turn_id")
        if not isinstance(text, str) or not isinstance(turn_id, str) or not turn_id:
            raise ValidationError("控制事件缺少原文或事件ID")
        if speaker != actor.principal_id:
            raise AccessDenied("控制请求说话人与认证身份不一致")
        quote = proposal.get("source_quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValidationError("控制请求必须引用原始事件的连续片段")
        action = proposal.get("action")
        if action not in {"grant", "revoke", "forget"}:
            raise ValidationError("不支持的控制操作")
        operation_id = proposal.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValidationError("控制请求缺少稳定操作ID")
        self.memory.control.authenticate(actor)
        # External verification must check intent, ownership, scope, and each
        # deletion target against the original turn and prior evidence.
        if self.verify(turn, proposal) is not True:
            return {"status": "needs_review", "action": action, "turn_id": turn_id}

        request_id = stable_id("control_turn", actor.space_id, turn_id, operation_id, asdict(actor))
        proposal_digest = stable_id("control_proposal", proposal)
        with self.memory.lock:
            previous = self.memory.store.get("control_intent_v2", request_id)
            if previous:
                if previous["proposal_digest"] != proposal_digest:
                    raise ValidationError("同一事件已绑定不同的控制请求")
                if previous["result"].get("status") == "pending_cleanup":
                    retried = self.memory.retry_redactions(actor)
                    previous["result"] = retried if retried["jobs"] else self.memory.retry_deletes(actor)
                    self.memory.store.put("control_intent_v2", request_id, previous)
                return previous["result"]

            if action == "grant":
                spec = proposal.get("grant")
                if not isinstance(spec, dict) or spec.get("issuer") != actor.principal_id:
                    raise AccessDenied("授权主体与认证身份不一致")
                result = self.memory.control.grant(actor, spec)
            elif action == "revoke":
                grant_id = proposal.get("grant_id")
                if not isinstance(grant_id, str) or not grant_id:
                    raise ValidationError("撤权需要明确的授权ID")
                result = self.memory.control.revoke(actor, grant_id)
            else:
                event_ids = proposal.get("event_ids")
                if not isinstance(event_ids, list) or not event_ids or not all(isinstance(x, str) and x for x in event_ids):
                    raise ValidationError("删除需要明确的事件ID")
                # Exact scope has to be checked by the trusted verifier. The
                # native API deletes whole events, not arbitrary substrings.
                if proposal.get("whole_events_verified") is True:
                    result = self.memory.forget(actor, event_ids)
                elif isinstance(proposal.get("exact_fragment"), str):
                    result = self.memory.forget_exact_fragment(actor, event_ids, proposal["exact_fragment"])
                    if result.get("status") == "needs_review" and len(event_ids) == 1:
                        plan = self.memory.propose_redaction(actor, event_ids[0], proposal["exact_fragment"])
                        if plan and self.verify(turn, {**proposal, "redaction_plan": plan}) is True:
                            result = self.memory.rewrite_redacted_event(actor, event_ids[0], plan["remove_text"], proposal["exact_fragment"])
                else:
                    return {"status": "needs_review", "action": action, "turn_id": turn_id}
            self.memory.store.put("control_intent_v2", request_id,
                                  {"turn_id": turn_id, "proposal_digest": proposal_digest, "result": result})
            return result
