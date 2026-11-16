from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Settings
from .governance import AccessGraph, GovernedMemoryService
from .llm import CompatibleJsonModel
from .memos_backend import MemOSBackend
from .models import ValidationError
from .store import Store


def expand_env(value):
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        if not os.environ.get(name):
            raise ValidationError(f"缺少环境变量 {name}；未使用替代模型")
        return os.environ[name]
    return value


def build(config_path):
    path = Path(config_path).resolve()
    data = expand_env(json.loads(path.read_text(encoding="utf-8-sig")))
    root = path.parent.parent
    settings = Settings(**data["method"])
    settings.database = str(root / settings.database)
    settings.validate()
    native = data["memos"]
    native["directory"] = str(root / native["directory"])
    store = Store(settings.database)
    if data.get("mode") == "multiuser-v2":
        from .multiuser import MultiUserMemory
        from .scheduler import MemOSSleepScheduler

        backend = MemOSBackend(native, store)
        service = MultiUserMemory(
            settings, backend, CompatibleJsonModel(settings, store), store, data["control"]
        )
        service.scheduler = MemOSSleepScheduler(service.core)
        return service, {}, data.get("identities", [])
    graph = AccessGraph(store)
    policies = {}
    for name, spec in data["policies"].items():
        known = set(settings.team_agents.get(spec["team_id"], []))
        if not set(spec["readers"] + spec["writers"] + spec.get("managers", [])) <= known:
            raise ValidationError("权限配置包含未注册Agent")
        policies[name] = graph.provision(**spec)
    backend = MemOSBackend(native, store)
    service = GovernedMemoryService(settings, backend, CompatibleJsonModel(settings, store), store, graph)
    from .scheduler import MemOSSleepScheduler

    service.scheduler = MemOSSleepScheduler(service)
    return service, policies, data.get("identities", [])
