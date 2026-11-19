"""Actual GateMem NaiveRAG smoke with official incremental runner and judge."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/gatemem/official/code"), str(Path(__file__).parent)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from adapter import GateBaselineBridge, GateOfficialAMemAgent
from bench.embeddings import EmbeddingConfig, EmbeddingRouter
from bench.eval.judge import run_llm_judge
from bench.eval.runner import run_episode
from bench.llm.router import LLMRouter
from bench.llm.types import LLMConfig

from ours_memory.benchmark_inputs import jsonl

sys.path.insert(0, str(ROOT / "experiments/groupmembench"))
from smoke_all import build_baseline


class Router(LLMRouter):
    def complete_result(self, **kwargs):
        kwargs["provider_options"] = {"thinking": {"type": "disabled"}}
        result = super().complete_result(**kwargs)
        if not result.text or (result.usage or {}).get("reasoning_tokens", 0):
            raise RuntimeError("empty response or unexpected reasoning")
        choices = (result.raw or {}).get("choices") or []
        if not choices or choices[0].get("finish_reason") != "stop":
            raise RuntimeError("model output did not finish normally")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["naive_rag", "mem0", "amem", "memos"], default="naive_rag")
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()
    root = ROOT / "experiments/gatemem/official/data/gatemem/data/medical"
    episode = jsonl(root / "episodes.jsonl")[0]
    ckpts = [c for c in jsonl(root / "checkpoints.jsonl") if c["episode_id"] == episode["episode_id"]]
    ckpts.sort(key=lambda c: int(c["as_of_turn_id"][1:]))
    ckpts = ckpts[:10]
    out = ROOT / f"runtime/gatemem-smoke-{args.system}{args.run_suffix}"
    out.mkdir(parents=True, exist_ok=False)
    if args.system == "mem0":
        os.environ["MEM0_DIR"] = str(out / "mem0-internal")
    answer = Router(LLMConfig(provider="deepseek", model="deepseek-flash", temperature=0.2, max_output_tokens=512, max_retries=1, timeout_s=90))
    judge = Router(LLMConfig(provider="deepseek", model="deepseek-flash", temperature=1, max_output_tokens=2048, max_retries=1, timeout_s=90))
    if args.system == "amem":
        embed = EmbeddingRouter(EmbeddingConfig(provider="hf", model="sentence-transformers/all-MiniLM-L6-v2"))
        memory_router = Router(LLMConfig(provider="deepseek", model="deepseek-flash", temperature=0.0,
                                         max_output_tokens=1000, max_retries=1, timeout_s=90))
        agent = GateOfficialAMemAgent(top_k=10, llm_router=answer, memory_llm_router=memory_router,
            retrieval_backend="embedding", embed_router=embed,
            metadata_mode="llm", link_top_m=3, link_score_threshold=0.15,
            graph_expand_hops=1, graph_expand_per_hit=2, rerank_role_bonus=0.08, rerank_entity_bonus=0.06)
    else:
        agent = GateBaselineBridge(lambda _: build_baseline(args.system, out, []), llm_router=answer)
    predictions = run_episode(agent=agent, episode=episode, checkpoints=ckpts, prediction_path=str(out / "predictions.jsonl"))
    rows, stats = run_llm_judge(episodes=[episode], checkpoints=ckpts, predictions=predictions, judge_router=judge,
        prompt_path=str(ROOT / "experiments/gatemem/official/code/bench/prompts/judge_prompt.txt"),
        out_path=str(out / "judgments.jsonl"), concurrency=1)
    result = {"dataset": "GateMem", "system": args.system, "checkpoints": len(ckpts),
              "predictions": len(predictions), "judgments": len(rows), "smoke_only": True,
              "implementation": "GateMem official A-MEM-style" if args.system == "amem" else "existing baseline adapter",
              "official_judge_stats": stats}
    (out / "status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "official_judge_stats"}))


if __name__ == "__main__":
    main()
