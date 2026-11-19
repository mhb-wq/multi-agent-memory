"""Trusted authorization plane; never derive grants from ordinary memory text."""

from dataclasses import dataclass

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
        self.access_mode = config.get("access_mode", "shared_pool")
        if self.access_mode != "shared_pool":
            raise ValidationError("仅支持共享记忆池")
        self.people = set(config["principals"])
        self.state_subjects = set(config.get("state_subjects", config["principals"]))
        if not self.state_subjects <= self.people:
            raise ValidationError("状态主体必须是已登记主体")
        self.person_aliases = config.get("person_aliases", {})
        if (not isinstance(self.person_aliases, dict) or any(
            subject not in self.people or not isinstance(names, list)
            or any(not isinstance(name, str) or not name.strip() for name in names)
            for subject, names in self.person_aliases.items()
        )):
            raise ValidationError("人物别名必须映射已登记主体到非空文本列表")
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
        return operation in {"read", "write", "confirm"} and operation in agent.get("actions", [])

    def require(self, actor, subject, category, operation, *, resource=None):
        if not self.permits(actor, subject, category, operation, resource=resource):
            raise AccessDenied("该请求未获授权")

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
        raise AccessDenied("共享池模式不执行授权操作")

    def revoke(self, actor, grant_id):
        raise AccessDenied("共享池模式不执行撤权操作")
