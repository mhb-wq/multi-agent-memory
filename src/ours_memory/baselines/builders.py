"""Builders for official baseline objects.

Each builder constructs the upstream object; it does not wrap or replace the
upstream algorithm. Credentials and model configuration are passed in by the
runner so a run can be reproduced from its config file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def build_mem0(config: dict[str, Any] | None = None):
    """Create an official ``mem0.Memory`` from a Mem0 ``MemoryConfig`` dict."""
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    return Memory(MemoryConfig(**config)) if config else Memory()


def _fix_temperature(system: Any, temperature: float) -> Any:
    """Freeze the A-MEM controller call and adapt its JSON-schema request.

    DeepSeek accepts ``json_object`` but not OpenAI's ``json_schema`` response
    format. The prompt already contains the requested JSON shape; replacing the
    response format is an API compatibility shim, not a change to A-MEM logic.
    """
    original = system.llm_controller.llm.get_completion

    def fixed_temperature(prompt, response_format, _temperature=None, _original=original):
        if response_format:
            response_format = {"type": "json_object"}
        return _original(prompt, response_format, temperature=temperature)

    system.llm_controller.llm.get_completion = fixed_temperature
    return system


def _point_amem_at_base(system: Any, api_key: str | None, base_url: str | None) -> Any:
    """A-MEM's controller ignores api_base; point its OpenAI client explicitly."""
    if not base_url:
        return system
    os.environ["OPENAI_BASE_URL"] = base_url
    controller = getattr(system, "llm_controller", None)
    llm = getattr(controller, "llm", None)
    if llm is not None and hasattr(llm, "client"):
        from openai import OpenAI

        llm.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        if str(getattr(llm, "model", "")).startswith("deepseek"):
            create = llm.client.chat.completions.create

            def create_without_thinking(*args, **kwargs):
                extra = dict(kwargs.get("extra_body") or {})
                extra["thinking"] = {"type": "disabled"}
                kwargs["extra_body"] = extra
                return create(*args, **kwargs)

            llm.client.chat.completions.create = create_without_thinking
    return system


def build_amem(
    *,
    model_name: str = "all-MiniLM-L6-v2",
    llm_model: str = "deepseek-flash",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = 0.0,
):
    """Create an ``AgenticMemorySystem`` from the official A-MEM library.

    This library (``agiresearch/A-mem``) uses ChromaDB and exposes
    ``search_agentic``. For the paper's reproduction layer use
    :func:`build_amem_paper`.
    """
    from agentic_memory.memory_system import AgenticMemorySystem

    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
    system = AgenticMemorySystem(
        model_name=model_name,
        llm_backend="openai",
        llm_model=llm_model,
        api_key=api_key,
    )
    system = _point_amem_at_base(system, api_key, base_url)
    return _fix_temperature(system, temperature) if temperature is not None else system


def build_amem_paper(
    repo_path: str,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    llm_model: str = "deepseek-flash",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = 0.0,
):
    """Create the paper reproduction ``AgenticMemorySystem`` from ``memory_layer.py``.

    ``repo_path`` is the checkout of ``WujiangXu/AgenticMemory`` pinned in
    ``baseline-sources.json``. The upstream module is imported as-is; only the
    OpenAI-compatible endpoint and inference temperature are supplied.
    """
    sys.path.insert(0, str(Path(repo_path).resolve()))
    from memory_layer import AgenticMemorySystem

    system = AgenticMemorySystem(
        model_name=model_name,
        llm_backend="openai",
        llm_model=llm_model,
        api_key=api_key,
        api_base=base_url,
    )
    system = _point_amem_at_base(system, api_key, base_url)
    return _fix_temperature(system, temperature) if temperature is not None else system


def build_memos(config: dict[str, Any] | None = None):
    """Create an official ``memos.mem_os.main.MOS`` from a ``MOSConfig`` dict."""
    from memos.configs.mem_os import MOSConfig
    from memos.mem_os.main import MOS

    return MOS(MOSConfig(**config)) if config else MOS()


def build_memos_native(
    directory: str,
    *,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-flash",
    embedding_model: str = "all-MiniLM-L6-v2",
    cube_id: str = "group",
):
    """Create the native MemOS ``GeneralTextMemory`` used as the stock control.

    This is the same upstream text-memory module used by Ours, but without the
    Card--State--Sleep layer. It is not a reimplementation of MemOS.
    """
    from pathlib import Path

    from memos import GeneralMemCube, GeneralMemCubeConfig

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MEMOS_BASE_PATH", str(root))
    extractor = {
        "backend": "openai",
        "config": {
            "model_name_or_path": model,
            "api_key": api_key,
            "api_base": base_url,
            "temperature": 0.0,
            "max_tokens": 4096,
            "extra_body": {"thinking": {"type": "disabled"}},
            "remove_think_prefix": True,
        },
    }
    text_mem = {
        "backend": "general_text",
        "config": {
            "cube_id": cube_id,
            "memory_filename": "textual_memory.json",
            "extractor_llm": extractor,
            "embedder": {
                "backend": "sentence_transformer",
                "config": {"model_name_or_path": embedding_model, "trust_remote_code": False},
            },
            "vector_db": {
                "backend": "qdrant",
                "config": {
                    "collection_name": cube_id,
                    "path": str(root / "qdrant"),
                    "distance_metric": "cosine",
                    "vector_dimension": 384,
                },
            },
        },
    }
    return GeneralMemCube(
        GeneralMemCubeConfig(
            user_id=cube_id,
            cube_id=cube_id,
            text_mem=text_mem,
            act_mem={"backend": "uninitialized", "config": {}},
            para_mem={"backend": "uninitialized", "config": {}},
            pref_mem={"backend": "uninitialized", "config": {}},
        )
    )
