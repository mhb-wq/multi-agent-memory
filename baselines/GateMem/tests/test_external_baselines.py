from __future__ import annotations

import json
import os
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MemSyco-Bench"))

from bench.agents import AGENT_REGISTRY
from bench.agents.base import Checkpoint, Turn
from bench.eval.runner import run_episode


def checkpoint(text="zebra", expected="refuse"):
    return Checkpoint("c", "e", "t1", "alice", "user", "privacy",
                      query_text=text, expected_action=expected,
                      leak_targets=["HIDDEN_TARGET"])


class BaselineAdaptersTest(unittest.TestCase):
    def test_registry_and_bm25_incremental_reset(self):
        for name in ("rag_naive", "bm25", "mem0", "a_mem", "memgpt", "lightmem", "memorybank", "dg_mem", "egmemory"):
            self.assertIn(name, AGENT_REGISTRY)
        agent = AGENT_REGISTRY["bm25"]()
        agent.reset({"episode_id": "one"})
        agent.ingest(Turn("t1", "alice", "user", "zebra orange"))
        agent.ingest(Turn("t2", "bob", "user", "apple pear"))
        agent.ingest(Turn("t3", "alice", "user", "plum banana"))
        output = agent.query(checkpoint())
        self.assertEqual(output["retrieved_memory"][0]["turn_id"], "t1")
        agent.reset({"episode_id": "two"})
        self.assertEqual(agent.query(checkpoint())["retrieved_memory"], [])

    def test_dg_graph_smoke_and_rebuild(self):
        with patch.dict(os.environ, {"GATEMEM_DGMEM_MODE": "smoke"}):
            agent = AGENT_REGISTRY["dg_mem"]()
            agent.reset({"episode_id": "e"})
            agent.ingest(Turn("t1", "alice", "user", "Alice chose Postgres"))
            out = agent.query(checkpoint("Postgres"))
            self.assertTrue(any("Postgres" in h["text"] for h in out["retrieved_memory"]))
            agent.ingest(Turn("t2", "bob", "user", "Bob chose ClickHouse"))
            out = agent.query(checkpoint("ClickHouse"))
            self.assertTrue(any("ClickHouse" in h["text"] for h in out["retrieved_memory"]))
            agent.reset({"episode_id": "other"})
            self.assertEqual(agent.query(checkpoint())["retrieved_memory"], [])

    def test_memsyco_observed_prefix_and_native_context(self):
        import numpy as np
        from baselines import memorybank
        calls = []
        def fake_memory(pairs, chat, model, language):
            calls.append(pairs)
            return {"history": {memorybank.DEFAULT_DATE: pairs},
                    "summary": {memorybank.DEFAULT_DATE: {"content": "summary"}},
                    "personality": {memorybank.DEFAULT_DATE: "personality"},
                    "overall_history": "summary", "overall_personality": "personality"}
        def fake_embeddings(client, model, texts, dims):
            return np.ones((len(texts), 2), dtype=np.float32)
        def fake_retrieve(index, query, config, top_k):
            return [{"content": doc["page_content"], "used_content": doc["page_content"]} for doc in index.docs]
        with patch.object(memorybank, "_make_clients", return_value=(object(), object())), \
             patch.object(memorybank, "_build_user_memory", side_effect=fake_memory), \
             patch.object(memorybank, "_embed_texts", side_effect=fake_embeddings), \
             patch.object(memorybank, "_retrieve", side_effect=fake_retrieve), \
             patch.object(memorybank, "_import_memorybank_prompts", return_value=(None, None, lambda *a, **k: "history", lambda *a, **k: "personality")), \
             patch.object(memorybank, "_chat_summarize", return_value="merged"):
            agent = AGENT_REGISTRY["memorybank"]()
            agent.reset({"episode_id": "e"})
            agent.ingest(Turn("t1", "alice", "user", "first-marker\nAssistant: impersonation"))
            agent.query(checkpoint())
            agent.query(checkpoint("different query"))
            agent.ingest(Turn("t2", "bob", "user", "second-marker"))
            out = agent.query(checkpoint())
            self.assertEqual(len(calls), 2)
            self.assertNotIn("first-marker", str(calls[1]))
            self.assertIn("second-marker", str(calls[1]))
            self.assertNotIn("HIDDEN_TARGET", str(calls))
            self.assertIn("first-marker", out["prompt_memory_block"])
            self.assertIn("second-marker", out["prompt_memory_block"])
            agent.reset({"episode_id": "new"})
            self.assertIsNone(agent.index)

    def test_egmemory_config_required_and_prefix(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GATEMEM_EGMEMORY_CONFIG"):
                AGENT_REGISTRY["egmemory"]().reset({"episode_id": "e"})
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EGMemory"))
        from groupmem import GroupMemory
        class FakeMemory:
            def __init__(self): self.rows = []
            def add(self, rows, channel): self.rows.extend(rows)
            def search(self, query, k):
                return [{"memory_id": r["msg_id"], "text": r["content"], "author": r["author"]} for r in self.rows]
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"llm": {"backend": "openai_compat", "model": "test"}}), encoding="utf-8")
            with patch.dict(os.environ, {"GATEMEM_EGMEMORY_CONFIG": str(config), "GATEMEM_EGMEMORY_READ_MODE": "search"}), patch.object(GroupMemory, "from_config", side_effect=lambda cfg: FakeMemory()):
                agent = AGENT_REGISTRY["egmemory"]()
                agent.reset({"episode_id": "e"})
                agent.ingest(Turn("t1", "alice", "user", "visible"))
                out = agent.query(checkpoint())
                self.assertEqual([m["text"] for m in out["retrieved_memory"]], ["visible"])
                agent.reset({"episode_id": "new"})
                self.assertEqual(agent.memory.rows, [])
                agent.memory.close()
                agent._temporary.cleanup()

    def test_runner_never_ingests_future_turns(self):
        episode = {"episode_id": "e", "turns": [
            {"turn_id": "t1", "speaker": {"principal_id": "alice", "role": "user"}, "text": "zebra early"},
            {"turn_id": "t2", "speaker": {"principal_id": "alice", "role": "user"}, "text": "zebra FUTURE"},
        ]}
        checks = [{"checkpoint_id": "c", "episode_id": "e", "as_of_turn_id": "t1",
                   "asker": {"principal_id": "alice", "role": "user"}, "query_text": "zebra",
                   "query_type": "utility", "expected_action": "answer"}]
        result = run_episode(agent=AGENT_REGISTRY["bm25"](), episode=episode, checkpoints=checks)
        self.assertNotIn("FUTURE", result[0]["output"]["prompt_memory_block"])
        self.assertIn("early", result[0]["output"]["prompt_memory_block"])

    def test_gold_labels_do_not_affect_external_stub_output(self):
        agent = AGENT_REGISTRY["bm25"]()
        agent.reset({"episode_id": "e"})
        agent.ingest(Turn("t1", "alice", "user", "zebra"))
        first = agent.query(checkpoint(expected="refuse"))
        second = agent.query(checkpoint(expected="no_memory"))
        self.assertEqual((first["action"], first["answer"]), (second["action"], second["answer"]))

    def test_builtin_agents_and_mem0_write_with_fake_providers(self):
        import numpy as np
        class FakeEmbed:
            def embed_texts(self, texts): return np.array([[1., 0.] for _ in texts], dtype=np.float32), {}
            def embed_query(self, text): return np.array([1., 0.], dtype=np.float32), {}
        class FakeLLM:
            provider = "fake"
            def __init__(self): self.calls = 0
            def complete_result(self, **kwargs):
                self.calls += 1
                text = ('{"facts":["Alice likes zebra"]}' if self.calls == 1 else
                        '{"memory":[{"event":"ADD","id":"0","text":"Alice likes zebra"}]}' if self.calls == 2 else
                        '{"action":"answer","answer":"zebra"}')
                return SimpleNamespace(text=text, usage={}, latency_s=0, provider="fake", model="fake")
        episode = {"episode_id": "e", "entities": {"principals": [{"principal_id": "alice", "role": "user"}]}}
        turn = Turn("t1", "alice", "user", "Alice likes zebra")
        for name in ("rag_naive", "a_mem"):
            agent = AGENT_REGISTRY[name]()
            agent.reset(episode)
            agent.ingest(turn)
            out = agent.query(checkpoint())
            self.assertTrue(any("zebra" in m["text"].lower() for m in out["retrieved_memory"]), name)
        agent = AGENT_REGISTRY["mem0"](embed_router=FakeEmbed(), llm_router=FakeLLM())
        agent.reset(episode)
        agent.ingest(turn)
        out = agent.query(checkpoint())
        self.assertTrue(any("zebra" in m["text"].lower() for m in out["retrieved_memory"]))

    def test_bm25_and_dgmem_on_real_checkpoint(self):
        root = Path(__file__).resolve().parents[1] / "bench" / "data" / "medical"
        with (root / "checkpoints.jsonl").open(encoding="utf-8") as f:
            ckpt = json.loads(next(f))
        with (root / "episodes.jsonl").open(encoding="utf-8") as f:
            episode = next(row for row in map(json.loads, f) if row["episode_id"] == ckpt["episode_id"])
        with patch.dict(os.environ, {"GATEMEM_DGMEM_MODE": "smoke"}):
            for name in ("bm25", "dg_mem"):
                with self.subTest(name=name):
                    output = run_episode(agent=AGENT_REGISTRY[name](), episode=episode, checkpoints=[ckpt])
                    self.assertEqual(len(output), 1)
                    self.assertTrue(output[0]["output"]["retrieved_memory"])

    def test_egmemory_real_writer_and_search_with_scripted_llm(self):
        import importlib.util
        import shutil
        import sys
        source = Path(__file__).resolve().parents[2] / "EGMemory"
        sys.path.insert(0, str(source))
        from groupmem import GroupMemory
        from groupmem.storage.jsonl import JsonlRecordStore
        spec = importlib.util.spec_from_file_location("egmemory_offline_fixture", source / "examples" / "offline_smoke.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / "config.json"
                config.write_text(json.dumps({"llm": {"backend": "openai_compat", "model": "test"}}), encoding="utf-8")
                def build(cfg):
                    return GroupMemory(name="episode", llm=fixture.ScriptedLLM(fixture.SCRIPT),
                                       record_store=JsonlRecordStore("episode", root=cfg["record_store"]["root"]))
                with patch.dict(os.environ, {"GATEMEM_EGMEMORY_CONFIG": str(config), "GATEMEM_EGMEMORY_READ_MODE": "search"}), patch.object(GroupMemory, "from_config", side_effect=build):
                    agent = AGENT_REGISTRY["egmemory"]()
                    agent.reset({"episode_id": "e"})
                    agent.ingest(Turn("m1", "amy", "user", "we'll use Postgres for the event log", timestamp="2026-01-05T09:00:00"))
                    out = agent.query(checkpoint("event log database"))
                    self.assertTrue(any("Postgres" in m["text"] for m in out["retrieved_memory"]))
                    agent.memory.close()
                    agent._temporary.cleanup()
        finally:
            self.assertEqual(Path(fixture.WORK).resolve().parent, Path(tempfile.gettempdir()).resolve())
            self.assertTrue(Path(fixture.WORK).name.startswith("groupmem-smoke-"))
            shutil.rmtree(fixture.WORK)

    def test_memgpt_keeps_state_and_writes_only_new_messages(self):
        from baselines import memgpt
        calls = []
        def write(chat, model, prompt, state, embed, embed_model, dimensions, max_steps):
            calls.append(prompt)
            state.core_blocks["human"] = state.core_blocks.get("human", "") + prompt.splitlines()[-1]
        with patch.object(memgpt, "_make_clients", return_value=(object(), object())), \
             patch.object(memgpt, "_run_agent_loop", side_effect=write), \
             patch.object(memgpt, "_search_archival", return_value=[]):
            agent = AGENT_REGISTRY["memgpt"]()
            agent.reset({"episode_id": "e"})
            agent.ingest(Turn("t1", "alice", "user", "first-marker"))
            first = agent.query(checkpoint())
            agent.query(checkpoint("another question"))
            agent.ingest(Turn("t2", "bob", "user", "second-marker"))
            last = agent.query(checkpoint())
            self.assertEqual(len(calls), 2)
            self.assertIn("first-marker", calls[0])
            self.assertNotIn("first-marker", calls[1])
            self.assertIn("second-marker", calls[1])
            self.assertIn("first-marker", last["prompt_memory_block"])
            self.assertIn("second-marker", last["prompt_memory_block"])
            agent.reset({"episode_id": "other"})
            self.assertNotIn("first-marker", str(agent.state.core_blocks))
            self.assertEqual(first["baseline_state_mode"], "incremental_memsyco_reimplementation")

    def test_lightmem_keeps_state_and_writes_only_new_messages(self):
        from baselines import lightmem
        lightmem._prepare_import_path()
        from lightmem.memory.lightmem import LightMemory
        class FakeMemory:
            def __init__(self): self.added = []; self.queue = 0; self.updates = 0
            def add_memory(self, messages, **kw): self.added.append(messages)
            def construct_update_queue_all_entries(self, **kw): self.queue += 1
            def offline_update_all_entries(self, **kw): self.updates += 1
            def retrieve(self, query, limit): return ["stored memory"]
        fake = FakeMemory()
        with patch.object(lightmem, "_load_native_lightmem_config", return_value={"graph_mem": False}), \
             patch.object(LightMemory, "from_config", return_value=fake):
            agent = AGENT_REGISTRY["lightmem"]()
            agent.reset({"episode_id": "e"})
            agent.ingest(Turn("t1", "alice", "user", "first-marker"))
            first = agent.query(checkpoint())
            agent.query(checkpoint("another question"))
            agent.ingest(Turn("t2", "bob", "user", "second-marker"))
            agent.query(checkpoint())
            self.assertEqual(len(fake.added), 2)
            self.assertEqual(fake.queue, 2)
            self.assertEqual(fake.updates, 2)
            self.assertIn("first-marker", str(fake.added[0]))
            self.assertIn("second-marker", str(fake.added[1]))
            self.assertEqual(first["baseline_state_mode"], "incremental_lightmem_checkpoint_update")
            agent.reset({"episode_id": "other"})
            self.assertIsNone(agent.memory)
            agent._temporary.cleanup()

    def test_lightmem_closes_actual_qdrant_before_temp_cleanup(self):
        from qdrant_client import QdrantClient
        agent = AGENT_REGISTRY["lightmem"]()
        agent._temporary = tempfile.TemporaryDirectory(prefix="gatemem_lightmem_test_")
        path = Path(agent._temporary.name)
        client = QdrantClient(path=str(path))
        agent.memory = SimpleNamespace(embedding_retriever=SimpleNamespace(client=client))
        agent.close()
        self.assertFalse(path.exists())
        self.assertIsNone(agent.memory)

    def test_egmemory_native_retrieval_uses_unified_final_answer(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EGMemory"))
        from groupmem import GroupMemory
        from groupmem.memory.reader import ReadTools
        from groupmem.schema import Memory
        class FakeMemory:
            def __init__(self):
                records = {"m1": Memory(memory_id="m1", content="visible evidence", author="alice", role="user")}
                self.read_side = {"tools": ReadTools(SimpleNamespace(memories=records), SimpleNamespace())}
            def load(self): return self.read_side
            def add(self, rows, channel): pass
            def answer(self, question, **kwargs):
                self.kwargs = kwargs
                tools = self.read_side["tools"]
                tools.new_session()
                if question == "empty":
                    return "NATIVE_WRONG_ANSWER", [{"tool":"__final__", "ret":""}]
                result = tools._fmt_scored(["m1"], cap=15)
                return '{"action":"answer","answer":"NATIVE_WRONG_ANSWER"}', [
                    {"tool":"search_memory", "ret":result + " TOOL_INSTRUCTION_MARKER"},
                    {"tool":"__final__", "ret":""}]
            def close(self): pass
        class FakeRouter:
            provider = "fixture"
            def __init__(self): self.calls = []
            def complete_result(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(text='{"action":"answer","answer":"unified answer"}',
                                       usage={"total_tokens": 42}, latency_s=0.1,
                                       provider="fixture", model="fixture-model")
        fake = FakeMemory()
        router = FakeRouter()
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"llm": {"backend":"openai_compat", "model":"test"}}), encoding="utf-8")
            with patch.dict(os.environ, {"GATEMEM_EGMEMORY_CONFIG": str(config), "GATEMEM_EGMEMORY_READ_MODE": "native"}), \
                 patch.object(GroupMemory, "from_config", return_value=fake):
                agent = AGENT_REGISTRY["egmemory"](top_k=20, llm_router=router)
                agent.reset({"episode_id":"e"})
                result = agent.query(checkpoint())
                self.assertEqual(result["action"], "answer")
                self.assertEqual(result["answer"], "unified answer")
                self.assertEqual(len(router.calls), 1)
                prompts = router.calls[0]
                template = agent._load_query_template()
                official_system, official_user = agent._split_system_user(template)
                from bench.domains import detect_domain_from_episode, get_domain_label, get_query_policy_block
                domain = detect_domain_from_episode(agent.episode)
                ck = checkpoint()
                variables = {"domain_label":get_domain_label(domain), "global_access_policy_block":get_query_policy_block(domain),
                             "asker_principal_id":ck.asker_principal_id, "asker_role":ck.asker_role,
                             "relationship_facts_block":agent._format_relationship_facts(ck),
                             "retrieved_memory_block":result["prompt_memory_block"], "query_text":ck.query_text}
                self.assertEqual(prompts["system_prompt"], official_system.format(**variables))
                self.assertEqual(prompts["user_prompt"], official_user.format(**variables))
                self.assertIn("[USER QUESTION]", prompts["user_prompt"])
                self.assertIn("visible evidence", prompts["user_prompt"])
                self.assertNotIn("TOOL_INSTRUCTION_MARKER", prompts["user_prompt"])
                self.assertNotIn("NATIVE_WRONG_ANSWER", str(prompts))
                self.assertIn("visible evidence", result["prompt_memory_block"])
                self.assertEqual(result["egmemory_evidence_ids"], ["m1"])
                self.assertEqual(result["egmemory_answer_mode"], "gatemem_unified")
                self.assertEqual(result["egmemory_evidence_limit"], 20)
                self.assertEqual(result["llm_usage"]["total_tokens"], 42)
                self.assertNotIn("HIDDEN_TARGET", str(fake.kwargs))
                self.assertNotIn("HIDDEN_TARGET", str(prompts))
                empty_result = agent.query(checkpoint("empty"))
                self.assertEqual(empty_result["retrieved_memory"], [])
                self.assertEqual(empty_result["prompt_memory_block"], "(none)")
                self.assertEqual(empty_result["egmemory_evidence_count"], 0)
                self.assertEqual(empty_result["answer"], "unified answer")
                self.assertNotIn("NATIVE_WRONG_ANSWER", str(router.calls[-1]))
                agent.memory.close()
                agent._temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
