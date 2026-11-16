"""Trusted authorization plane; never derive grants from ordinary memory text."""

from dataclasses import asdict, dataclass

from .models import AccessDenied, ValidationError, active, interval, now, stable_id, timestamp


@dataclass(frozen=True)
class Actor:
    principal_id: str
    agent_id: str
    space_id: str
    task_id: str


class ControlPlane:
    def __init__(self, store, config, clock=now):
        self.store, self.config, self.clock = store, config, clock
        self.access_mode = config.get("access_mode", "governed")
        if self.access_mode not in {"governed", "shared_pool"}:
            raise ValidationError("未知记忆访问模式")
        self.people = set(config["principals"])
        self.state_subjects = set(config.get("state_subjects", config["principals"]))
        if not self.state_subjects <= self.people:
            raise ValidationError("状态主体必须是已登记主体")
        self.categories = set(config["categories"])
        self.agents = config["agents"]
        self.spaces = config["spaces"]
        # A team/project subject must be registered like any person before it can own
        # memories or grants; otherwise a typo silently creates an unreachable subject.
        for space in self.spaces.values():
            unknown = set(space.get("members", [])) - self.people
            if unknown:
                raise ValidationError("协作空间包含未登记主体")
            for subject, managers in space.get("grant_managers", {}).items():
                if subject not in space.get("members", []) or not set(managers) <= self.people:
                    raise ValidationError("授权管理者配置引用未登记主体")
            for resource, managers in space.get("resource_managers", {}).items():
                if (not isinstance(resource, str) or not resource or not isinstance(managers, list)
                    or not set(managers) <= set(space.get("members", []))):
                    raise ValidationError("资源管理者配置引用未登记主体或无效资源")
        # Insert-once bootstrap: restarting cannot reinstate a revoked grant.
        for grant in config.get("grants", []):
            self._validate(grant)
            key = stable_id("grant", grant["id"])
            if not store.get("grant_v2", key):
                store.put("grant_v2", key, {**grant, "revoked": False})

    def authenticate(self, actor):
        space = self.spaces.get(actor.space_id, {})
        if (
            actor.principal_id not in self.people
            or actor.principal_id not in space.get("members", [])
            or actor.agent_id not in self.agents
            or actor.agent_id not in space.get("agents", [])
            or not actor.task_id
        ):
            raise AccessDenied("未知身份或协作空间")

    def _validate(self, grant):
        required = {
            "id",
            "issuer",
            "grantee",
            "subject",
            "space",
            "category",
            "actions",
            "valid_from",
            "valid_to",
        }
        if (
            not isinstance(grant, dict)
            or not required <= set(grant)
            or set(grant) - required - {"task", "agent", "revoked", "resource", "effect", "source"}
        ):
            raise ValidationError("授权字段不完整或包含未知字段")
        if any(
            not isinstance(grant[k], str) or not grant[k]
            for k in ("id", "issuer", "grantee", "subject", "space", "category")
        ):
            raise ValidationError("授权身份和范围须为非空字符串")
        if any(
            grant.get(k) is not None and (not isinstance(grant[k], str) or not grant[k])
            for k in ("task", "agent", "resource", "source")
        ):
            raise ValidationError("任务和Agent限制须为非空字符串")
        if (
            not isinstance(grant["id"], str)
            or not grant["id"]
            or grant["issuer"] not in self.people
            or grant["grantee"] not in self.people
            or grant["subject"] not in self.people
            or grant["space"] not in self.spaces
            or grant["category"] not in self.categories
        ):
            raise ValidationError("授权引用未知主体或范围")
        if (
            not isinstance(grant["actions"], list)
            or not grant["actions"]
            or not all(isinstance(a, str) for a in grant["actions"])
            or not set(grant["actions"]) <= {"read", "write", "confirm", "delete"}
        ):
            raise ValidationError("授权操作不合法")
        if grant.get("agent") and grant["agent"] not in self.agents:
            raise ValidationError("未知Agent")
        if grant.get("effect", "allow") not in {"allow", "deny"}:
            raise ValidationError("授权效果只允许allow或deny")
        interval(grant["valid_from"], grant["valid_to"])

    def permits(self, actor, subject, category, operation, *, resource=None):
        self.authenticate(actor)
        if subject not in self.spaces[actor.space_id].get("members", []) or category not in self.categories:
            return False
        agent = self.agents[actor.agent_id]
        if self.access_mode == "shared_pool":
            # Research mode: no user-to-user read/write ACL; destructive
            # control operations are intentionally not enabled.
            return operation in {"read", "write", "confirm"} and operation in agent.get("actions", [])
        if category not in agent.get("categories", []) or operation not in agent.get("actions", []):
            return False
        matching = [g for g in self.store.all("grant_v2") if (
            not g["revoked"]
            and g["grantee"] == actor.principal_id
            and g["subject"] == subject
            and g["space"] == actor.space_id
            and g["category"] == category
            and operation in g["actions"]
            and (not g.get("task") or g["task"] == actor.task_id)
            and (not g.get("agent") or g["agent"] == actor.agent_id)
            and (not g.get("resource") or g["resource"] == resource)
            and active(g, self.clock())
        )]
        return any(g.get("effect", "allow") == "allow" for g in matching) and not any(
            g.get("effect") == "deny" for g in matching
        )

    def require(self, actor, subject, category, operation, *, resource=None):
        if not self.permits(actor, subject, category, operation, resource=resource):
            raise AccessDenied("该请求未获授权")

    def _manager(self, actor, subject, resource=None):
        self.authenticate(actor)
        space = self.spaces[actor.space_id]
        rules = space.get("grant_managers", {})
        resource_rules = space.get("resource_managers", {})
        return (actor.principal_id in rules.get(subject, []) or
                bool(resource and actor.principal_id in resource_rules.get(resource, [])))

    def maintenance_summary(self, actor):
        """Summarize rules without changing their authority or retaining private facts."""
        self.authenticate(actor)
        if "sleep" not in self.agents[actor.agent_id].get("controls", []):
            raise AccessDenied("没有维护职责")
        rows = [g for g in self.store.all("grant_v2") if g["space"] == actor.space_id]
        current = self.clock()
        return {
            "active": sum(not g["revoked"] and active(g, current) for g in rows),
            "revoked": sum(bool(g["revoked"]) for g in rows),
            "expired": sum(not g["revoked"] and bool(g.get("valid_to")) and timestamp(g["valid_to"]) <= timestamp(current) for g in rows),
        }

    def grant(self, actor, spec):
        if self.access_mode == "shared_pool":
            raise AccessDenied("共享池模式不执行授权操作")
        self._validate(spec)
        if (
            spec["issuer"] != actor.principal_id
            or spec["space"] != actor.space_id
            or not self._manager(actor, spec["subject"], spec.get("resource"))
        ):
            raise AccessDenied("没有授予该主体权限的资格")
        if "grant" not in self.agents[actor.agent_id].get("controls", []):
            raise AccessDenied("Agent不具备授权操作职责")
        key = stable_id("grant", spec["id"])
        if self.store.get("grant_v2", key):
            raise ValidationError("授权ID已使用；恢复授权必须建立新记录")
        self.store.put("grant_v2", key, {**spec, "revoked": False})
        self._audit(actor, "grant", spec["id"])
        return {"grant_id": spec["id"], "status": "active"}

    def revoke(self, actor, grant_id):
        if self.access_mode == "shared_pool":
            raise AccessDenied("共享池模式不执行撤权操作")
        self.authenticate(actor)
        key = stable_id("grant", grant_id)
        row = self.store.get("grant_v2", key)
        if (
            not row
            or row["space"] != actor.space_id
            or not self._manager(actor, row["subject"], row.get("resource"))
            or "revoke" not in self.agents[actor.agent_id].get("controls", [])
        ):
            raise AccessDenied("无权撤销该授权")
        row.update(revoked=True, revoked_at=self.clock())
        self.store.put("grant_v2", key, row)
        self._audit(actor, "revoke", grant_id)
        return {"grant_id": grant_id, "status": "revoked"}

    def _audit(self, actor, operation, target):
        at = self.clock()
        timestamp(at)
        key = stable_id("control_audit", asdict(actor), operation, target, at)
        self.store.put(
            "control_audit", key, {"actor": asdict(actor), "operation": operation, "target": target, "at": at}
        )
