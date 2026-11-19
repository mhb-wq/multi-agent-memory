"""Run a GateMem baseline using one relay model and local MiniLM embeddings.

Use --smoke for one real checkpoint; without it the requested domain is run.
Credentials are read from ignored .env and passed only via child environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from local_embedding_server import MODEL, start_server
from local_relay_proxy import start_relay

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("rag_naive", "bm25", "mem0", "a_mem", "memgpt", "lightmem", "memorybank", "dg_mem", "egmemory", "ours")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=AGENTS)
    parser.add_argument("--model", required=True, help="Exact model ID shown by the currently active relay group")
    parser.add_argument("--domain", default="medical", choices=("medical", "office", "education", "household", "all"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ours-python", help="Python executable with Ours/MemOS dependencies (only for --agent ours)")
    parser.add_argument("--smoke", action="store_true", help="Use one checkpoint; outputs are not formal results")
    parser.add_argument("--smoke-turns", type=int, help="For smoke only, move the boundary earlier to limit paid calls")
    args = parser.parse_args()
    if args.smoke_turns is not None and (not args.smoke or args.smoke_turns < 1):
        parser.error("--smoke-turns requires --smoke and must be positive")
    if args.domain == "all" and args.agent != "ours":
        parser.error("--domain all is currently supported only for --agent ours")
    if args.agent == "ours" and args.smoke:
        parser.error("Ours formal entry does not implement smoke runs")
    if args.ours_python and args.agent != "ours":
        parser.error("--ours-python is only valid for --agent ours")
    if args.ours_python and not Path(args.ours_python).is_file():
        parser.error("--ours-python must point to an existing executable")
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("--run-name must be a simple directory name")
    if (ROOT / "outputs" / args.run_name).exists():
        raise SystemExit("Run directory already exists; choose a new --run-name to avoid overwriting results.")
    if args.agent == "dg_mem":
        model_file = ROOT / "models" / "dgmem-reranker" / "model.safetensors"
        if not model_file.is_file() or hashlib.sha256(model_file.read_bytes()).hexdigest() != "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae":
            raise SystemExit("DG-Mem reranker is missing or failed SHA-256 verification")

    local = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            local[name.strip()] = value.strip()
    key, base = local["CIYUAN_API_KEY"], local["CIYUAN_BASE_URL"].rstrip("/")
    req = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + key,
                                                              "User-Agent": "GateMem-Research-Client/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        catalog = json.load(response)
    allowed = {x.get("id") for x in catalog.get("data", []) if isinstance(x, dict)}
    if args.model not in allowed:
        raise SystemExit("The requested model ID is not available in the current relay group; check /v1/models and switch groups if necessary.")
    data_dir = ROOT / "bench" / "data" if args.domain == "all" else ROOT / "bench" / "data" / args.domain
    env = os.environ.copy()
    result_dir = ROOT / "outputs" / args.run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    relay_server, relay_url = start_relay(base, key, args.model, result_dir / "model_calls.jsonl")
    env.update({"CIYUAN_API_KEY": key, "OPENAI_API_KEY": key, "OPENAI_BASE_URL": relay_url,
                "MEMORY_API_KEY": key, "MEMORY_BASE_URL": relay_url, "MEMORY_LLM_MODEL": args.model,
                "MEMORY_EMBEDDING_MODEL": MODEL, "MEMORY_EMBEDDING_DIMS": "384",
                "MEMORY_EMBEDDING_API_KEY": "local-only", "MEMORY_EMBEDDER_PROVIDER": "openai",
                "LIGHTMEM_MEMORY_MANAGER": "openai", "GATEMEM_DGMEM_MODE": "reference",
                "GATEMEM_DGMEM_RERANKER": str(ROOT / "models" / "dgmem-reranker"),
                "GATEMEM_EGMEMORY_READ_MODE": "native", "HF_HUB_OFFLINE": "1"})
    if args.agent == "ours":
        env.update({"DEEPSEEK_API_KEY": key, "OURS_LLM_BASE_URL": relay_url,
                    "OURS_FORMAL_MODEL": args.model, "OURS_FORMAL_OUTPUT": str(result_dir)})
        no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
        env["NO_PROXY"] = ",".join(x for x in (no_proxy, "127.0.0.1", "localhost", "::1") if x)
        env["no_proxy"] = env["NO_PROXY"]
    try:
        server, embedding_url = start_server()
        env["MEMORY_EMBEDDING_BASE_URL"] = embedding_url
        try:
            return run_child(args, env, base, relay_url, embedding_url, result_dir, data_dir)
        finally:
            server.shutdown()
            server.server_close()
    finally:
        relay_server.shutdown()
        relay_server.server_close()


def run_child(args, env, base, relay_url, embedding_url, result_dir, data_dir):
        with tempfile.TemporaryDirectory(prefix="gatemem_run_") as temp:
            temp_root = Path(temp)
            selected_domains = ("medical", "office", "education", "household") if args.domain == "all" else (args.domain,)
            dataset_hashes = {}
            for domain in selected_domains:
                folder = data_dir / domain if args.domain == "all" else data_dir
                for name in ("episodes.jsonl", "checkpoints.jsonl"):
                    dataset_hashes[f"{domain}/{name}"] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            if args.domain == "all":
                merged = temp_root / "all_domains"
                merged.mkdir()
                for name in ("episodes.jsonl", "checkpoints.jsonl"):
                    with (merged / name).open("wb") as target:
                        for domain in ("medical", "office", "education", "household"):
                            source = (data_dir / domain / name).read_bytes()
                            target.write(source)
                            if source and not source.endswith(b"\n"):
                                target.write(b"\n")
                data_dir = merged
            if args.smoke:
                with (data_dir / "checkpoints.jsonl").open(encoding="utf-8") as f:
                    checkpoint = json.loads(next(f))
                with (data_dir / "episodes.jsonl").open(encoding="utf-8") as f:
                    episode = next(row for row in map(json.loads, f) if row["episode_id"] == checkpoint["episode_id"])
                if args.smoke_turns is not None:
                    boundary = min(args.smoke_turns, len(episode["turns"])) - 1
                    checkpoint["as_of_turn_id"] = episode["turns"][boundary]["turn_id"]
                    print("Smoke-only modified checkpoint boundary; judge scores from this run are not valid experiment results.", flush=True)
                small = temp_root / "data"
                small.mkdir()
                for name, row in (("checkpoints.jsonl", checkpoint), ("episodes.jsonl", episode)):
                    (small / name).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
                data_dir = small
            eg_config = {"llm": {"backend": "openai_compat", "model": args.model},
                         "embedder": {"backend": "openai", "model": MODEL, "dim": 384,
                                      "base_url": embedding_url, "api_key": "local-only"},
                         "reranker": {"backend": "noop"}, "vector_store": {"backend": "inmemory"}}
            if args.smoke:
                eg_config["llm"].update({"retries": 1, "hard_timeout": 35})
                env["GATEMEM_EGMEMORY_HTTP_TIMEOUT"] = "25"
                env["GATEMEM_EGMEMORY_MAX_TOKENS"] = "1024"
            config_file = temp_root / "egmemory.json"
            config_file.write_text(json.dumps(eg_config), encoding="utf-8")
            env["GATEMEM_EGMEMORY_CONFIG"] = str(config_file)
            venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            interpreter = str(venv_python) if args.agent == "dg_mem" and venv_python.is_file() else sys.executable
            if args.agent == "ours":
                ours_root = ROOT.parents[1] / "ours"
                ours_python = (Path(args.ours_python).resolve() if args.ours_python else
                               ours_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
                if args.ours_python and not ours_python.is_file():
                    raise ValueError("--ours-python does not point to an existing executable")
                if ours_python.is_file():
                    interpreter = str(ours_python)
            cmd = [interpreter, "bench/scripts/run_eval.py", "--agent", args.agent,
                   "--data_dir", str(data_dir), "--llm_provider", "relay", "--llm_model", args.model,
                   "--api_base", relay_url, "--api_key_env", "CIYUAN_API_KEY", "--use_llm_judge",
                   "--judge_provider", "relay", "--judge_model", args.model,
                   "--retrieval_backend", "embedding", "--embed_provider", "hf", "--embed_model", MODEL,
                   "--embed_device", "cpu", "--embed_max_length", "256", "--mem0_backend", "builtin",
                   "--top_k", "20", "--answer_protocol", "standard", "--temperature", "0.2",
                   "--max_output_tokens", "4096", "--judge_temperature", "0", "--judge_max_output_tokens", "4096",
                   "--episode_concurrency", "1", "--judge_concurrency", "1", "--run_name", args.run_name]
            if args.smoke:
                cmd.extend(["--max_output_tokens", "1024", "--judge_max_output_tokens", "1024"])
            manifest = {"schema_version": 1, "agent": args.agent, "domain": args.domain,
                        "requested_write_answer_judge_model": args.model, "relay_base_url": base,
                        "embedding_model": MODEL, "embedding_dimensions": 384, "embedding_max_tokens": 256,
                        "embedding_device": "cpu", "embedding_token_usage": "not_measured",
                        "smoke_only": args.smoke, "modified_checkpoint_boundary": args.smoke_turns,
                        "top_k": 20, "episode_concurrency": 1, "judge_concurrency": 1,
                        "answer_protocol": "standard", "answer_temperature": 0.2,
                        "answer_max_output_tokens": 1024 if args.smoke else 4096,
                        "judge_temperature": 0, "judge_max_output_tokens": 1024 if args.smoke else 4096,
                        "gate_by_action": False,
                        "mem0_backend": "builtin", "egmemory_retrieval_mode": "native",
                        "egmemory_answer_mode": "gatemem_unified",
                        "dataset_sha256": dataset_hashes,
                        "query_prompt_sha256": hashlib.sha256((ROOT / "bench/prompts/query_prompt.txt").read_bytes()).hexdigest(),
                        "judge_prompt_sha256": hashlib.sha256((ROOT / "bench/prompts/judge_prompt.txt").read_bytes()).hexdigest(),
                        "lightmem_update": "checkpoint", "memorybank_mode": "incremental_multiparticipant_adaptation"}
            if args.agent == "ours":
                manifest["ours_method"] = {"access_mode": "shared_pool", "joint_state_update_enabled": True,
                                           "immediate_state_maintenance_enabled": True,
                                           "independent_state_read_enabled": True,
                                           "reasoning_policy": "provider_default"}
            (result_dir / "experiment_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Running {args.agent}; requested model={args.model}; domain={args.domain}; smoke={args.smoke}; local embedding={MODEL}", flush=True)
            return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
