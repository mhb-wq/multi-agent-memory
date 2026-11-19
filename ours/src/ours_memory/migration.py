"""Offline, operator-supplied historical ACL repair. Never called by the Agent service."""

from __future__ import annotations

from .models import ValidationError, stable_id
from .store import Store


def migrate_legacy_forget(store: Store, metadata: list[dict], team_agents: dict[str, list[str]]) -> dict:
    """Restore only historical identity/ACL metadata; never remove an erasure marker.

    The local operator must obtain metadata from a trusted archive/configuration. Missing
    ACLs cannot be inferred from a hash or from the Agent requesting a repeat deletion.
    """
    if not isinstance(metadata, list) or not metadata:
        raise ValidationError("迁移资料须为非空列表")
    proposals, current, seen = [], 0, set()
    with store.transaction():
        for row in metadata:
            keys = {"event_id", "user_id", "team_id", "allowed_reader_ids"}
            if not isinstance(row, dict) or set(row) != keys:
                raise ValidationError("迁移项须仅包含event_id/user_id/team_id/allowed_reader_ids")
            if any(
                not isinstance(row[key], str) or not row[key] for key in ("event_id", "user_id", "team_id")
            ):
                raise ValidationError("迁移身份与事件ID须为非空文本")
            readers = row["allowed_reader_ids"]
            if (
                not isinstance(readers, list)
                or not readers
                or not all(isinstance(value, str) for value in readers)
                or not set(readers) <= set(team_agents.get(row["team_id"], []))
            ):
                raise ValidationError("原始读取范围必须明确，且须属于已配置团队")
            key = stable_id("event", row["user_id"], row["team_id"], row["event_id"])
            if key in seen:
                raise ValidationError("迁移资料不能重复列出同一事件")
            seen.add(key)
            marker = store.get("forgotten", key)
            if not marker:
                raise ValidationError("迁移目标不是已有遗忘标记，拒绝创建或改动其他记录")
            replacement = {name: row[name] for name in ("user_id", "team_id")}
            replacement["allowed_reader_ids"] = sorted(set(readers))
            for name, value in replacement.items():
                if name in marker and marker[name] != value:
                    raise ValidationError("迁移资料与已有身份或权限冲突，拒绝覆盖")
            if all(name in marker for name in replacement):
                current += 1
            else:
                proposals.append((key, {**marker, **replacement}))
        for key, marker in proposals:
            store.put("forgotten", key, marker)
    return {"migrated": len(proposals), "already_current": current}
