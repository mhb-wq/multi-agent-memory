from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench.agents import AGENT_REGISTRY
from bench.agents.base import Checkpoint, Turn
from bench.domains import detect_domain_from_episode, get_domain_label, get_query_policy_block
from bench.eval.runner import run_episode
from bench.eval.validator import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_relay_experiment as launcher  # noqa: E402


class FakeStore:
    def call_records(self):
        return []

    def all(self, kind):
        return []


class FakeMemory:
    def __init__(self):
        self.turns = []
        self.store = FakeStore()

    def ingest_message(self, actor, **kwargs):
        self.turns.append((actor, kwargs))
        return {"message_id": "message-" + kwargs["message_id"],
                "status": "accepted_pending" if kwargs["message_id"] == "t1" else "done",
                "processing_status": "state_maintenance_pending" if kwargs["message_id"] == "t1" else None,
                "events": []}

    def read(self, actor, *, subject_ids, query, purpose):
        last = self.turns[-1][1]["text"] if self.turns else "none"
        return {"bundle_id": "bundle", "cards": [{"id": "card", "author_id": "alice",
                "claim_text": last, "source": {"quote": last}}], "user_states": [{"subject": "alice"}],
                "pending_state_count": 1, "incomplete_message_count": 1, "pending_subjects": ["alice"]}

    def validate_bundle(self, actor, bundle):
        return {"ready": True}


class FakeRouter:
    provider = "relay"

    def __init__(self, model):
        self.config = SimpleNamespace(model=model, api_base="http://127.0.0.1:4568/v1")
        self.calls = []

    def complete_result(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text='{"action":"answer","answer":"fixture"}',
                               usage={"total_tokens": 12}, latency_s=0.01,
                               provider="relay", model=self.config.model)


def episode():
    return {"episode_id": "episode", "entities": {"principals": [{"principal_id": "alice"}]},
            "turns": [{"turn_id": "t1", "timestamp": "2026-01-01T00:00:00Z",
                       "speaker": {"principal_id": "alice", "role": "user"}, "text": "first fact"},
                      {"turn_id": "t2", "timestamp": "2026-01-01T00:01:00Z",
                       "speaker": {"principal_id": "alice", "role": "user"}, "text": "future fact"}]}


class OursFormalTest(unittest.TestCase):
    def test_three_explicit_model_ids_and_matching_official_answers(self):
        from ours_memory import groupmem

        for model in ("deepseek-example", "gpt-example", "glm-example"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                fake = FakeMemory()
                router = FakeRouter(model)
                captured = []
                def build(path, cfg, trace, clock):
                    captured.append((json.loads((path / "protocol.json").read_text(encoding="utf-8")), cfg))
                    return fake
                env = {"OURS_FORMAL_MODEL": model, "OURS_FORMAL_OUTPUT": tmp,
                       "OURS_LLM_BASE_URL": "http://127.0.0.1:4568/v1",
                       "MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:4567/v1"}
                with patch.dict(os.environ, env), patch.object(groupmem, "build_memory", side_effect=build), \
                     patch.object(groupmem, "close_memory") as close:
                    agent = AGENT_REGISTRY["ours"](top_k=20, llm_router=router)
                    agent.reset(episode())
                    agent.ingest(Turn("t1", "alice", "user", "first fact", "2026-01-01T00:00:00Z"))
                    ck = Checkpoint("c", "episode", "t1", "alice", "user", "utility", query_text="question",
                                    expected_action="HIDDEN_ACTION", judge_spec={"gold":"HIDDEN_GOLD"})
                    out = agent.query(ck)
                    agent.ingest(Turn("t2", "alice", "user", "future fact", "2026-01-01T00:01:00Z"))
                    agent.close()
                    self.assertEqual(captured[0][0]["memory_model"], model)
                    self.assertEqual(captured[0][0]["reasoning_policy"], "provider_default")
                    self.assertEqual(captured[0][0]["embedding"]["base_url"], env["MEMORY_EMBEDDING_BASE_URL"])
                    self.assertTrue(all(captured[0][0]["method"].values()))
                    self.assertIn("first fact", router.calls[0]["user_prompt"])
                    self.assertNotIn("future fact", router.calls[0]["user_prompt"])
                    self.assertNotIn("HIDDEN_GOLD", str(router.calls))
                    self.assertNotIn("HIDDEN_ACTION", str(router.calls))
                    self.assertIn("GLOBAL ACCESS POLICY", router.calls[0]["system_prompt"])
                    domain = detect_domain_from_episode(agent.episode)
                    system, user = agent._split_system_user((ROOT / "bench/prompts/query_prompt.txt").read_text(encoding="utf8"))
                    variables = {"domain_label": get_domain_label(domain), "global_access_policy_block": get_query_policy_block(domain),
                                 "asker_principal_id": ck.asker_principal_id, "asker_role": ck.asker_role,
                                 "relationship_facts_block": agent._format_relationship_facts(ck),
                                 "retrieved_memory_block": out["prompt_memory_block"], "query_text": ck.query_text}
                    self.assertEqual(router.calls[0]["system_prompt"], system.format(**variables))
                    self.assertEqual(router.calls[0]["user_prompt"], user.format(**variables))
                    self.assertEqual(out["ours_pending_state_count"], 1)
                    self.assertGreaterEqual(out["retrieval_s"], 0)
                    self.assertEqual(fake.turns[1][1]["prior_context"][0]["text"], "first fact")
                    self.assertTrue((agent.run / "calls.json").exists())
                    self.assertTrue((agent.run / "pending.json").exists())
                    self.assertEqual(json.loads((agent.run / "turns.jsonl").read_text(encoding="utf8").splitlines()[0])["status"], "accepted_pending")
                    close.assert_called_once_with(fake)

    def test_formal_agent_reads_real_current_memory_with_fixture_model(self):
        from ours_memory import groupmem
        from ours_memory.backend import FixtureBackend
        from ours_memory.config import Settings
        from ours_memory.multiuser import MultiUserMemory
        from ours_memory.store import Store

        class Decisions:
            def complete(self, stage, payload):
                if stage == "focused_cards":
                    return {"cards": [{"subject": "alice", "claim": payload["text"], "usage": "",
                                       "state_topic": ["plan"],
                                       "source": {"speaker": "alice", "quote": payload["text"]}}]}
                if stage == "focused_joint":
                    return {"relations": [], "operations": [{"subject": "alice", "action": "add",
                            "state_id": None, "topic": "plan", "content": payload["new_card"]["claim_text"],
                            "status": "confirmed", "changed_part": "new", "evidence": [payload["new_card"]["id"]]}]}
                raise AssertionError(stage)

        router = FakeRouter("fixture-id")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OURS_FORMAL_MODEL": "fixture-id", "OURS_FORMAL_OUTPUT": tmp,
            "OURS_LLM_BASE_URL": "http://127.0.0.1:4568/v1",
            "MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:4567/v1"}):
            stores = []
            def build(path, cfg, trace, clock):
                store = Store(str(path / "test.sqlite3"))
                stores.append(store)
                settings = Settings(team_agents={cfg["space"]: ["ingestor", "maintainer", "answerer", "controller"]})
                return MultiUserMemory(settings, FixtureBackend(), Decisions(), store,
                                       groupmem.control_config(cfg["users"], cfg["space"]), clock)
            with patch.object(groupmem, "build_memory", side_effect=build), \
                 patch.object(groupmem, "close_memory", side_effect=lambda m: m.store.close()):
                agent = AGENT_REGISTRY["ours"](top_k=20, llm_router=router)
                agent.reset(episode())
                agent.ingest(Turn("t1", "alice", "user", "Alice plans a Friday class", "2026-01-01T00:00:00Z"))
                ck = Checkpoint("c", "episode", "t1", "alice", "user", "utility", query_text="When is class?")
                result = agent.query(ck)
                self.assertIn("Alice plans a Friday class", router.calls[0]["user_prompt"])
                self.assertEqual(result["ours_pending_state_count"], 0)
                self.assertGreater(result["ours_state_count"], 0)
                agent.close()

    def test_rejects_wrong_model_and_nonlocal_embedding(self):
        router = FakeRouter("real-id")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OURS_FORMAL_MODEL": "different-id", "OURS_FORMAL_OUTPUT": tmp,
            "OURS_LLM_BASE_URL": "http://127.0.0.1:4568/v1",
            "MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1"}):
            with self.assertRaisesRegex(ValueError, "same explicitly requested model"):
                AGENT_REGISTRY["ours"](llm_router=router)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OURS_FORMAL_MODEL": "real-id", "OURS_FORMAL_OUTPUT": tmp,
            "OURS_LLM_BASE_URL": "http://127.0.0.1:4568/v1",
            "MEMORY_EMBEDDING_BASE_URL": "https://remote.example/v1"}):
            with self.assertRaisesRegex(ValueError, "local MiniLM"):
                AGENT_REGISTRY["ours"](llm_router=router)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OURS_FORMAL_MODEL": "real-id", "OURS_FORMAL_OUTPUT": tmp,
            "OURS_LLM_BASE_URL": "http://127.0.0.1:9999/v1",
            "MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1"}):
            with self.assertRaisesRegex(ValueError, "same relay endpoint"):
                AGENT_REGISTRY["ours"](llm_router=router)

    def test_all_domains_launcher_passes_all_official_records_to_shared_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(domain="all", agent="ours", model="fixture-model", smoke=False,
                                   smoke_turns=None, ours_python=None, run_name="unit-check")
            result_dir = Path(tmp) / "result"
            result_dir.mkdir()
            def run(cmd, cwd, env, check):
                data = Path(cmd[cmd.index("--data_dir") + 1])
                episodes = [json.loads(line) for line in (data / "episodes.jsonl").read_text(encoding="utf8").splitlines()]
                checkpoints = [json.loads(line) for line in (data / "checkpoints.jsonl").read_text(encoding="utf8").splitlines()]
                self.assertEqual((len(episodes), len(checkpoints)), (91, 2218))
                self.assertEqual(validate_dataset(episodes=episodes, checkpoints=checkpoints, strict=True)[0], [])
                self.assertEqual(cmd[cmd.index("--agent") + 1], "ours")
                self.assertEqual(cmd[cmd.index("--llm_model") + 1], "fixture-model")
                self.assertEqual(cmd[cmd.index("--judge_model") + 1], "fixture-model")
                self.assertIn("--use_llm_judge", cmd)
                return SimpleNamespace(returncode=0)
            with patch.object(launcher.subprocess, "run", side_effect=run):
                code = launcher.run_child(args, {}, "https://invalid.example/v1", "http://127.0.0.1:1/v1",
                                          "http://127.0.0.1:2/v1", result_dir, ROOT / "bench" / "data")
            self.assertEqual(code, 0)
            manifest = json.loads((result_dir / "experiment_manifest.json").read_text(encoding="utf8"))
            self.assertTrue(manifest["ours_method"]["joint_state_update_enabled"])
            self.assertEqual(manifest["domain"], "all")
            self.assertEqual(len(manifest["dataset_sha256"]), 8)
            self.assertEqual(len(manifest["query_prompt_sha256"]), 64)
            self.assertEqual(len(manifest["judge_prompt_sha256"]), 64)

    def test_official_runner_respects_checkpoint_order_with_pending_messages(self):
        from ours_memory import groupmem
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OURS_FORMAL_MODEL": "fixture", "OURS_FORMAL_OUTPUT": tmp,
            "OURS_LLM_BASE_URL": "http://127.0.0.1:4568/v1",
            "MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1"}), \
             patch.object(groupmem, "build_memory", return_value=FakeMemory()), \
             patch.object(groupmem, "close_memory"):
            router = FakeRouter("fixture")
            agent = AGENT_REGISTRY["ours"](top_k=20, llm_router=router)
            checks = [{"checkpoint_id": "c1", "episode_id": "episode", "as_of_turn_id": "t1",
                       "asker": {"principal_id": "alice", "role": "user"},
                       "query_type": "utility", "query_text": "first question", "expected_action": "answer"},
                      {"checkpoint_id": "c2", "episode_id": "episode", "as_of_turn_id": "t2",
                       "asker": {"principal_id": "alice", "role": "user"},
                       "query_type": "utility", "query_text": "second question", "expected_action": "answer"}]
            predictions = run_episode(agent=agent, episode=episode(), checkpoints=checks)
            agent.close()
            self.assertEqual([p["checkpoint_id"] for p in predictions], ["c1", "c2"])
            self.assertIn("first fact", router.calls[0]["user_prompt"])
            self.assertNotIn("future fact", router.calls[0]["user_prompt"])
            self.assertIn("future fact", router.calls[1]["user_prompt"])
            self.assertTrue(all(p["output"]["ours_pending_state_count"] == 1 for p in predictions))

    def test_existing_episode_directory_is_never_overwritten(self):
        from ours_memory import groupmem
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OURS_FORMAL_MODEL": "fixture", "OURS_FORMAL_OUTPUT": tmp,
            "OURS_LLM_BASE_URL": "http://127.0.0.1:4568/v1",
            "MEMORY_EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1"}), \
             patch.object(groupmem, "build_memory", return_value=FakeMemory()), \
             patch.object(groupmem, "close_memory"):
            agent = AGENT_REGISTRY["ours"](llm_router=FakeRouter("fixture"))
            agent.reset(episode())
            agent.close()
            with self.assertRaises(FileExistsError):
                agent.reset(episode())

    def test_formal_cli_rejects_missing_or_mismatched_judge_and_resume(self):
        base = [sys.executable, str(ROOT / "bench/scripts/run_eval.py"), "--agent", "ours",
                "--llm_provider", "relay", "--llm_model", "model-a"]
        variants = [([], "requires a real model and the GateMem LLM judge"),
                    (["--use_llm_judge", "--judge_model", "model-b"], "same requested model and provider"),
                    (["--resume", "--run_name", "no-run"], "does not yet support safe automatic resume")]
        for extra, message in variants:
            with self.subTest(message=message):
                completed = subprocess.run(base + extra, cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr + completed.stdout)

    def test_formal_cli_never_overwrites_existing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "old-run"
            run.mkdir()
            prediction = run / "predictions.jsonl"
            prediction.write_text("previous-results\n", encoding="utf8")
            cmd = [sys.executable, str(ROOT / "bench/scripts/run_eval.py"), "--agent", "ours",
                   "--llm_provider", "relay", "--llm_model", "model-a", "--use_llm_judge",
                   "--out_dir", tmp, "--run_name", "old-run"]
            completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Ours output already exists", completed.stderr + completed.stdout)
            self.assertEqual(prediction.read_text(encoding="utf8"), "previous-results\n")

    def test_launcher_routes_memory_answer_and_judge_to_requested_model(self):
        class Server:
            def shutdown(self): pass
            def server_close(self): pass
        for model in ("deepseek-fixture", "gpt-fixture", "glm-fixture"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / ".env").write_text("CIYUAN_API_KEY=fixture-key\nCIYUAN_BASE_URL=https://invalid.example/v1\n", encoding="utf8")
                payload = json.dumps({"data": [{"id": model}]}).encode()
                def fake_child(args, env, base, relay, embedding, result, data):
                    self.assertEqual(args.model, model)
                    self.assertEqual(env["OURS_FORMAL_MODEL"], model)
                    self.assertEqual(env["OURS_LLM_BASE_URL"], relay)
                    self.assertEqual(env["MEMORY_EMBEDDING_BASE_URL"], embedding)
                    self.assertEqual(env["DEEPSEEK_API_KEY"], env["CIYUAN_API_KEY"])
                    self.assertIn("127.0.0.1", env["NO_PROXY"])
                    return 0
                argv = ["runner", "--agent", "ours", "--model", model, "--domain", "all", "--run-name", "case"]
                with patch.object(launcher, "ROOT", root), patch.object(sys, "argv", argv), \
                     patch.object(launcher.urllib.request, "urlopen", return_value=io.BytesIO(payload)), \
                     patch.object(launcher, "start_relay", return_value=(Server(), "http://127.0.0.1:5/v1")), \
                     patch.object(launcher, "start_server", return_value=(Server(), "http://127.0.0.1:6/v1")), \
                     patch.object(launcher, "run_child", side_effect=fake_child):
                    self.assertEqual(launcher.main(), 0)


if __name__ == "__main__":
    unittest.main()
