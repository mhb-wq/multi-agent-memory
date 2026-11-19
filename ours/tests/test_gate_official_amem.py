"""Verify GateMem uses its benchmark-maintained A-MEM implementation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_gate_uses_official_amem():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gatemem_smoke", root / "experiments/gatemem/smoke_naive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.GateOfficialAMemAgent(top_k=10, llm_router=SimpleNamespace(provider="fake"),
        memory_llm_router=SimpleNamespace(provider="fake"), metadata_mode="llm")
    assert agent.__class__.__mro__[1].__module__ == "bench.agents.a_mem"
    assert agent.metadata_mode == "llm"
    assert agent.top_k == 10
