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
AGENTS = ("rag_naive", "bm25", "mem0", "a_mem", "memgpt", "lightmem", "memorybank", "dg_mem", "egmemory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=AGENTS)
    parser.add_argument("--model", required=True, help="Exact model ID shown by the currently active relay group")
    parser.add_argument("--domain", default="medical", choices=("medical", "office", "education", "household"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true", help="Use one checkpoint; outputs are not formal results")
    parser.add_argument("--smoke-turns", type=int, help="For smoke only, move the boundary earlier to limit paid calls")
    args = parser.parse_args()
    if args.smoke_turns is not None and (not args.smoke or args.smoke_turns < 1):
        parser.error("--smoke-turns requires --smoke and must be positive")
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
    data_dir = ROOT / "bench" / "data" / args.domain
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
            cmd = [interpreter, "bench/scripts/run_eval.py", "--agent", args.agent,
                   "--data_dir", str(data_dir), "--llm_provider", "relay", "--llm_model", args.model,
                   "--api_base", relay_url, "--api_key_env", "CIYUAN_API_KEY", "--use_llm_judge",
                   "--judge_provider", "relay", "--judge_model", args.model,
                   "--retrieval_backend", "embedding", "--embed_provider", "hf", "--embed_model", MODEL,
                   "--embed_device", "cpu", "--embed_max_length", "256", "--mem0_backend", "builtin",
                   "--episode_concurrency", "1", "--judge_concurrency", "1", "--run_name", args.run_name]
            if args.smoke:
                cmd.extend(["--max_output_tokens", "1024", "--judge_max_output_tokens", "1024"])
            manifest = {"schema_version": 1, "agent": args.agent, "domain": args.domain,
                        "requested_write_answer_judge_model": args.model, "relay_base_url": base,
                        "embedding_model": MODEL, "embedding_dimensions": 384, "embedding_max_tokens": 256,
                        "embedding_device": "cpu", "embedding_token_usage": "not_measured",
                        "smoke_only": args.smoke, "modified_checkpoint_boundary": args.smoke_turns,
                        "top_k": 20, "episode_concurrency": 1, "judge_concurrency": 1,
                        "mem0_backend": "builtin", "egmemory_answer_mode": "native",
                        "lightmem_update": "checkpoint", "memorybank_mode": "incremental_multiparticipant_adaptation"}
            (result_dir / "experiment_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Running {args.agent}; requested model={args.model}; domain={args.domain}; smoke={args.smoke}; local embedding={MODEL}", flush=True)
            return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
