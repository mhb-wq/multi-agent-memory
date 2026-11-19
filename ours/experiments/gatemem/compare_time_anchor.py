"""Re-answer frozen retrieved contexts with the dataset checkpoint time explicit."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/gatemem/official/code")]
from bench.eval.scorer import score_one  # noqa: E402

from ours_memory.groupmem import TextChat, Trace  # noqa: E402

OUT = ROOT / "runtime/gatemem-compare-20261117"
DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/medical"
EP = "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line"
parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["medical", "household"], default="medical")
args = parser.parse_args()
if args.case == "household":
    OUT = OUT / "household"
    DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/household"
    EP = "household_episode_custom_en_010_rowan_wardrobe_rowan_wash_tailor_linen_private_note"
lines = lambda p: [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()]  # noqa: E731
episode = next(x for x in lines(DATA / "episodes.jsonl") if x["episode_id"] == EP)
checkpoints = {x["checkpoint_id"]:x for x in lines(DATA / "checkpoints.jsonl") if x["episode_id"] == EP}
times = {x["turn_id"]:x["timestamp"] for x in episode["turns"]}
for system in (["naive", "mem0", "amem", "memos", "ours"] if args.case == "household" else
               ["naive", "mem0", "amem", "memos", "ours_reuse"]):
    run = OUT / system
    path = run / "results.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text(encoding="utf8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf8"))
    chat = TextChat(protocol, run, Trace(run / "anchored_diagnostics"))
    output_path = run / "anchored_answers.json"
    out = json.loads(output_path.read_text(encoding="utf8")) if output_path.exists() else []
    done = {row["checkpoint_id"] for row in out}
    try:
        for original in result.get("answers", []):
            if original["checkpoint_id"] in done:
                continue
            ck = checkpoints[original["checkpoint_id"]]
            asker = ck["asker"]["principal_id"]
            moment = times[ck["as_of_turn_id"]]
            user = f"The question is asked at this conversation time: {moment}. Interpret 'now' relative to that time.\n"
            user += f"Asking user: {asker}\nQuestion: {ck['query_text']}\nRetrieved passages:\n" + "\n\n".join(original["passages"])
            user += "\nCurrent user states:\n" + json.dumps(original["user_states"], ensure_ascii=False)
            raw = chat("qa", user)
            score = score_one(checkpoint=ck,pred_output={"action":"answer","answer":raw,"answer_structured":{}})
            out.append({"checkpoint_id":ck["checkpoint_id"],"as_of":ck["as_of_turn_id"],"time":moment,
                        "answer":raw,"usage":chat.last_meta,"auxiliary_rule_score":score})
            output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
            print(json.dumps({"system":system,"checkpoint":ck["as_of_turn_id"],"hits":score["aux"]["include_hits"],
                              "needed":score["aux"]["include_needed"]}), flush=True)
    except Exception as exc:
        print(json.dumps({"system":system,"status":"failed","error":type(exc).__name__}), flush=True)
        break
