from __future__ import annotations

import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASELINES_ROOT = Path(__file__).resolve().parents[2] / "MemSyco-Bench"
sys.path.insert(0, str(BASELINES_ROOT))
from baselines import lightmem
from baselines.base import BaselineEvalConfig

lightmem._prepare_import_path()
from lightmem.memory.lightmem import LightMemory
from lightmem.memory.graph import GraphMem


class FakeMemory:
    def __init__(self):
        self.calls = []

    def add_memory(self, **kwargs):
        self.calls.append(("add", kwargs))

    def construct_update_queue_all_entries(self, **kwargs):
        self.calls.append(("queue", kwargs))

    def offline_update_all_entries(self, **kwargs):
        self.calls.append(("update", kwargs))

    def retrieve(self, query, limit):
        self.calls.append(("retrieve", query, limit))
        return ["retrieved fact"]


class LightMemBridgeTest(unittest.TestCase):
    def test_graph_module_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            GraphMem(True)

    def test_default_configuration_disables_graph(self):
        config = BaselineEvalConfig(method="LightMem")
        full = lightmem._default_full_lightmem_config(
            config, user_id="test", save_dir="test", embedder_provider="openai",
            text_embedder_configs={"model": "test", "embedding_dims": 2}, embedding_dims=2)
        raw = lightmem._default_raw_lightmem_config(
            config, user_id="test", save_dir="test", embedder_provider="openai",
            text_embedder_configs={"model": "test", "embedding_dims": 2}, embedding_dims=2)
        self.assertIs(full["graph_mem"], False)
        self.assertIs(raw["graph_mem"], False)

    def test_full_wrapper_invokes_write_update_and_retrieval(self):
        fake = FakeMemory()
        with tempfile.TemporaryDirectory() as tmp:
            config = BaselineEvalConfig(method="LightMem", top_k=2, save_root=Path(tmp))
            with patch.object(lightmem, "_load_native_lightmem_config", return_value={"graph_mem": False}), \
                 patch.object(LightMemory, "from_config", return_value=fake):
                result = lightmem.build_context("User: Alice chose tea\nAssistant: OK", "drink?", config)
        self.assertEqual([call[0] for call in fake.calls], ["add", "queue", "update", "retrieve"])
        self.assertEqual(fake.calls[-1], ("retrieve", "drink?", 2))
        self.assertEqual(result.retrieved_memories[0]["used_content"], "retrieved fact")

    def test_compressor_device_default_and_override(self):
        config = BaselineEvalConfig(method="LightMem")
        def build():
            return lightmem._default_full_lightmem_config(
                config, user_id="test", save_dir="test", embedder_provider="openai",
                text_embedder_configs={"model": "test", "embedding_dims": 2}, embedding_dims=2)
        with patch.dict(os.environ, {"LIGHTMEM_LLMLINGUA_DEVICE": "", "MEMORY_EMBEDDING_DEVICE": ""}), \
             patch.object(lightmem, "_default_compressor_device", return_value="cuda:0"):
            full = build()
            self.assertEqual(full["pre_compressor"]["configs"]["llmlingua_config"]["device_map"], "cuda:0")
            self.assertEqual(full["topic_segmenter"]["configs"]["device_map"], "cuda:0")
        with patch.dict(os.environ, {"LIGHTMEM_LLMLINGUA_DEVICE": "cpu", "MEMORY_EMBEDDING_DEVICE": "cuda:0"}):
            self.assertEqual(build()["topic_segmenter"]["configs"]["device_map"], "cpu")

    def test_graph_config_fails_before_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = BaselineEvalConfig(method="LightMem", save_root=Path(tmp))
            with patch.object(lightmem, "_load_native_lightmem_config", return_value={"graph_mem": True}), \
                 patch.object(LightMemory, "from_config") as construct:
                with self.assertRaisesRegex(NotImplementedError, "graph_mem"):
                    lightmem.build_context("User: hello", "question", config)
                construct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
