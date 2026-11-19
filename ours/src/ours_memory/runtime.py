from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Settings, validate_current_method
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
    validate_current_method(settings)
    if data.get("mode") != "multiuser-v2" or data.get("control", {}).get("access_mode") != "shared_pool":
        raise ValidationError("仅支持最新版多用户共享记忆池")
    native = data["memos"]
    native["directory"] = str(root / native["directory"])
    store = Store(settings.database)
    from .multiuser import MultiUserMemory
    from .scheduler import MemOSSleepScheduler

    backend = MemOSBackend(native, store)
    service = MultiUserMemory(settings, backend, CompatibleJsonModel(settings, store), store, data["control"])
    service.scheduler = MemOSSleepScheduler(service.core)
    return service, {}, data.get("identities", [])
