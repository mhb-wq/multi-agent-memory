"""Apply the official GateMem LLM judge to already frozen predictions."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/gatemem/official/code"))
from bench.eval.judge import run_llm_judge  # noqa: E402
from bench.llm.router import LLMRouter  # noqa: E402
from bench.llm.types import LLMConfig  # noqa: E402

OUT = ROOT / "runtime/gatemem-compare-20261117"
DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/medical"
EP = "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line"
parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["medical", "household"], default="medical")
parser.add_argument("--card-only", action="store_true")
args = parser.parse_args()
if args.case == "household":
    OUT = OUT / "household"
    DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/household"
    EP = "household_episode_custom_en_010_rowan_wardrobe_rowan_wash_tailor_linen_private_note"
read = lambda path: [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]  # noqa: E731
episode = next(x for x in read(DATA / "episodes.jsonl") if x["episode_id"] == EP)
checkpoints = [x for x in read(DATA / "checkpoints.jsonl") if x["checkpoint_id"] in {EP + f"_ckpt_{i:02d}" for i in (1, 2, 3)}]
router = LLMRouter(LLMConfig(provider="deepseek", model="gpt-6-sol", api_base=os.environ["OURS_LLM_BASE_URL"],
                             temperature=1.0, max_output_tokens=2048, max_retries=1, timeout_s=120))
for system in (["naive", "mem0", "amem", "memos", "ours"] if args.case == "household" else
               ["naive", "mem0", "amem", "memos", "ours_reuse"]):
    if args.card_only and system not in {"ours", "ours_reuse"}:
        continue
    path = OUT / system / "results.json"
    if not path.exists():
        continue
    anchored = OUT / system / ("card_only_answers.json" if args.card_only else "anchored_answers.json")
    if not anchored.exists():
        print(json.dumps({"system":system,"status":"no_anchored_predictions"}), flush=True)
        continue
    result = {"answers":json.loads(anchored.read_text(encoding="utf8"))}
    predictions = [{"checkpoint_id":x["checkpoint_id"], "episode_id":EP,
                    "output":{"action":"answer", "answer":x["answer"], "answer_structured":{}}}
                   for x in result.get("answers", [])]
    if not predictions:
        print(json.dumps({"system":system,"status":"no_predictions"}), flush=True)
        continue
    try:
        rows, stats = run_llm_judge(episodes=[episode], checkpoints=checkpoints, predictions=predictions,
            judge_router=router, prompt_path=str(ROOT / "experiments/gatemem/official/code/bench/prompts/judge_prompt.txt"),
            out_path=str(OUT / system / ("card_only_judgments.jsonl" if args.card_only else "judgments.jsonl")), concurrency=1, resume=True)
        (OUT / system / ("card_only_judge_summary.json" if args.card_only else "judge_summary.json")).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({"system":system,"judged":len(rows)}), flush=True)
    except Exception as exc:
        print(json.dumps({"system":system,"status":"judge_failed","error":type(exc).__name__}), flush=True)
        break
