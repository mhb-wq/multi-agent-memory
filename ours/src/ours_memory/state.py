from __future__ import annotations

from copy import deepcopy

from .models import ValidationError, active, canonical, stable_id, timestamp, value_key

CONFIRMED_KINDS = {"explicit_user_statement", "tool_observation", "execution_result"}


def support_source(card: dict) -> dict:
    return {
        "roots": card["source_root_ids"],
        "time": card.get("event_time"),
        "trusted": card["evidence_kind"] in CONFIRMED_KINDS,
    }


def refresh_support(assertion: dict, cards: dict[str, dict]) -> dict | None:
    remaining = [
        cards[key] for key in assertion["evidence_card_ids"] if key in cards and not cards[key].get("deleted")
    ]
    if not remaining:
        return None
    item = deepcopy(assertion)
    item["evidence_card_ids"] = sorted(card["id"] for card in remaining)
    item["source_root_ids"] = sorted({root for card in remaining for root in card["source_root_ids"]})
    item["support_sources"] = {card["id"]: support_source(card) for card in remaining}
    trusted = [
        card for card in remaining if card["evidence_kind"] in CONFIRMED_KINDS and card.get("event_time")
    ]
    if not trusted and item["status"] in {"confirmed", "unknown"}:
        item["status"] = "pending"
    times = [card["event_time"] for card in trusted if card.get("event_time")]
    item["event_time"] = max(times, key=timestamp) if times else None
    # Free-form reasons can quote withdrawn evidence; do not retain them across erasure.
    item["reason"] = "support_recomputed"
    return item


def state_id(card: dict) -> str:
    parts = [
        card["user_id"],
        card["team_id"],
        card["subject"],
        card["state_key"],
        card["scope"],
        card["conditions"],
        card["unit"],
        card["policy"],
    ]
    if card.get("state_aspect"):
        parts.append(card["state_aspect"])
    # New provenance-rich imports distinguish a person's deadlines across projects.
    # Legacy cards keep their existing addresses; no implicit migration of old stores.
    if card.get("metadata") and card.get("item_id"):
        parts.append(card["item_id"])
    return stable_id("state", *parts)


def empty_state(card: dict) -> dict:
    readers = tuple(sorted(set(card.get("allowed_reader_ids") or ())))
    visibility = card.get("visibility") or ("private" if len(readers) == 1 else "shared")
    owner = card.get("owner_agent_id")
    state = {
        "id": state_id(card),
        "user_id": card["user_id"],
        "team_id": card["team_id"],
        "subject": card["subject"],
        "state_key": card["state_key"],
        "scope": card["scope"],
        "conditions": card["conditions"],
        "unit": card["unit"],
        "policy": card["policy"],
        "allowed_reader_ids": card["allowed_reader_ids"],
        "visibility": visibility,
        "owner_agent_id": owner,
        "assertions": [],
        "revision": 0,
    }
    if card.get("state_aspect"):
        state["state_aspect"] = card["state_aspect"]
    if card.get("item_id"):
        state["item_id"] = card["item_id"]
    return state


def _overlap(a: dict, b: dict) -> bool:
    return not (
        (a.get("valid_to") and b.get("valid_from") and timestamp(a["valid_to"]) <= timestamp(b["valid_from"]))
        or (
            b.get("valid_to")
            and a.get("valid_from")
            and timestamp(b["valid_to"]) <= timestamp(a["valid_from"])
        )
    )


def _subtract(old: dict, new: dict) -> list[dict]:
    """Preserve historical/future portions outside a scoped replacement interval."""
    if not _overlap(old, new):
        return [old]
    pieces = []
    if new.get("valid_from") and (
        not old.get("valid_from") or timestamp(old["valid_from"]) < timestamp(new["valid_from"])
    ):
        pieces.append({**deepcopy(old), "valid_to": new["valid_from"]})
    if new.get("valid_to") and (
        not old.get("valid_to") or timestamp(new["valid_to"]) < timestamp(old["valid_to"])
    ):
        pieces.append({**deepcopy(old), "valid_from": new["valid_to"]})
    return pieces


def apply_operations(previous: dict, operations: list[dict], cards: dict[str, dict]) -> dict:
    result = deepcopy(previous)
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValidationError("状态操作须为对象")
        kind = operation.get("kind")
        if kind not in {"INIT", "CONFIRM", "UPDATE", "CONFLICT", "INVALIDATE", "IGNORE"}:
            raise ValidationError("未知状态操作")
        card = cards.get(operation.get("card_id"))
        if not card or state_id(card) != result["id"]:
            raise ValidationError("状态操作引用了不存在或地址不匹配的卡片")
        if kind == "IGNORE":
            continue
        if kind in {"CONFIRM", "UPDATE"} and card["value"] is None:
            raise ValidationError("CONFIRM/UPDATE必须有明确值；明确撤销而无新值时使用INVALIDATE")
        if kind == "CONFIRM" and not any(
            item["status"] == "confirmed"
            and value_key(item["value"]) == value_key(card["value"])
            and _overlap(item, card)
            for item in result["assertions"]
        ):
            raise ValidationError("CONFIRM须与现有已确认值及有效区间相容；新值使用INIT/UPDATE或保留冲突")
        reason = operation.get("reason", "")
        if not isinstance(reason, str) or len(reason) > 1000:
            raise ValidationError("状态变更理由须为简短文本")
        if kind == "INVALIDATE" and card["value"] is not None:
            raise ValidationError("失效但未知新值时，证据值必须为空")
        if kind in {"UPDATE", "INVALIDATE"} and not card.get("valid_from"):
            raise ValidationError("覆盖状态须有明确生效起点；未知时保留待确认")
        evidence_ids = [card["id"]]
        proposed = {
            "value": card["value"],
            "unit": card["unit"],
            "valid_from": card["valid_from"],
            "valid_to": card["valid_to"],
            "event_time": card["event_time"],
            "evidence_card_ids": evidence_ids,
            "source_root_ids": card["source_root_ids"],
            "support_sources": {card["id"]: support_source(card)},
            "operation": kind,
            "reason": reason,
            "status": "confirmed" if card["evidence_kind"] in CONFIRMED_KINDS else "pending",
        }
        if card["value"] is None and proposed["status"] == "confirmed":
            proposed["status"] = "unknown"
        elif not proposed["valid_from"]:
            proposed["status"] = "pending"
        if not card.get("event_time"):
            proposed["status"] = "pending"
        if kind == "CONFLICT":
            proposed["status"] = "conflict"
        assertions = result["assertions"]
        # Retrying an accepted operation or seeing an Agent repeat the same source cannot multiply evidence.
        own = [item for item in assertions if card["id"] in item["evidence_card_ids"]]
        if own and all(item.get("operation") == kind for item in own):
            continue
        same_origin = [
            item
            for item in assertions
            if value_key(item["value"]) == value_key(proposed["value"])
            and set(item["source_root_ids"]) == set(proposed["source_root_ids"])
            and _overlap(item, proposed)
        ]
        if same_origin and not own:
            continue
        # Reject an older replacement BEFORE removing any existing evidence.
        newer = [
            item
            for item in assertions
            if item["status"] in {"confirmed", "unknown", "conflict"}
            and item.get("event_time")
            and card.get("event_time")
            and _overlap(item, proposed)
            and timestamp(item["event_time"]) > timestamp(card["event_time"])
        ]
        if kind in {"UPDATE", "INVALIDATE"} and newer and proposed["status"] != "pending":
            continue
        if own and kind in {"UPDATE", "INVALIDATE", "CONFLICT"}:
            # Sleep may resolve a previously pending/conflicting interpretation using the same
            # validated card. It still cannot alter card text or invent a replacement value.
            retained = []
            for item in assertions:
                if card["id"] not in item["evidence_card_ids"]:
                    retained.append(item)
                else:
                    other_ids = set(item["evidence_card_ids"]) - {card["id"]}
                    if other_ids:
                        other = {**deepcopy(item), "evidence_card_ids": sorted(other_ids)}
                        support = {
                            key: source
                            for key, source in item.get("support_sources", {}).items()
                            if key in other_ids
                        }
                        if support:
                            other["support_sources"] = support
                            other["source_root_ids"] = sorted(
                                {root for source in support.values() for root in source["roots"]}
                            )
                            times = [
                                source["time"]
                                for source in support.values()
                                if source.get("time") and source["trusted"]
                            ]
                            other["event_time"] = max(times, key=timestamp) if times else None
                        other["reason"] = "support_recomputed"
                        retained.append(other)
            result["assertions"] = retained
            assertions = result["assertions"]
        if kind == "CONFIRM" and proposed["status"] == "confirmed":
            compatible = [
                item
                for item in assertions
                if item["status"] == "confirmed"
                and value_key(item["value"]) == value_key(proposed["value"])
                and item["valid_from"] == proposed["valid_from"]
                and item["valid_to"] == proposed["valid_to"]
            ]
            if compatible:
                item = compatible[0]
                item["evidence_card_ids"] = sorted(set(item["evidence_card_ids"] + evidence_ids))
                item["source_root_ids"] = sorted(set(item["source_root_ids"] + card["source_root_ids"]))
                item.setdefault("support_sources", {}).update(proposed["support_sources"])
                item["event_time"] = max([item["event_time"], card["event_time"]], key=timestamp)
                continue
        if kind in {"UPDATE", "INVALIDATE"}:
            if not proposed["valid_to"]:
                future = [
                    item["valid_from"]
                    for item in assertions
                    if item["status"] in {"confirmed", "unknown", "conflict"}
                    and item.get("valid_from")
                    and timestamp(item["valid_from"]) > timestamp(proposed["valid_from"])
                    and card.get("correction_of") not in item["evidence_card_ids"]
                ]
                if future:
                    # A new current value does not silently cancel an already confirmed future plan.
                    proposed["valid_to"] = min(future, key=timestamp)
            # Old source events cannot supersede newer evidence simply because imported later.
            if proposed["status"] == "pending":
                # A model's inference/plan/retelling is not a confirmed overwrite.
                assertions.append(proposed)
                continue
            result["assertions"] = [
                piece
                for old in assertions
                for piece in (
                    [old]
                    if old["status"] == "pending"
                    or (own and value_key(old["value"]) == value_key(proposed["value"]))
                    else _subtract(old, proposed)
                )
            ]
        result["assertions"].append(proposed)
    result["assertions"].sort(key=lambda item: canonical(item))
    return result


def state_view(state: dict, at: str) -> dict:
    current = [deepcopy(item) for item in state["assertions"] if active(item, at)]
    confirmed = [item for item in current if item["status"] == "confirmed"]
    values = {value_key(item["value"]) for item in confirmed}
    if any(item["status"] == "conflict" for item in current) or len(values) > 1:
        status, value = "conflict", None
    elif any(item["status"] == "unknown" for item in current):
        status, value = "unknown", None
    elif len(values) == 1:
        status, value = "confirmed", confirmed[0]["value"]
    else:
        status, value = ("pending" if current else "unknown"), None
    return {k: deepcopy(v) for k, v in state.items() if k != "assertions"} | {
        "target_time": at,
        "status": status,
        "value": value,
        "assertions": current,
        "has_pending": any(item["status"] == "pending" for item in current),
    }
