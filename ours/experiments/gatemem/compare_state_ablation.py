"""Re-answer frozen Ours cards without state; QA-only exploratory ablation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/gatemem/official/code")]
from bench.eval.scorer import score_one  # noqa: E402
from ours_memory.groupmem import TextChat, Trace  # noqa: E402

OUT = ROOT / "runtime/gatemem-compare-20261117"
cases = [
    (OUT / "ours_reuse", "medical", "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line"),
    (OUT / "household/ours", "household", "household_episode_custom_en_010_rowan_wardrobe_rowan_wash_tailor_linen_private_note"),
]
for run, domain, episode_id in cases:
    if not (run / "results.json").exists():
        continue
    result = json.loads((run / "results.json").read_text(encoding="utf8"))
    data = ROOT / "experiments/gatemem/official/data/gatemem/data" / domain
    read = lambda p: [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()]  # noqa: E731
    episode = next(x for x in read(data / "episodes.jsonl") if x["episode_id"] == episode_id)
    times = {x["turn_id"]:x["timestamp"] for x in episode["turns"]}
    checkpoints = {x["checkpoint_id"]:x for x in read(data / "checkpoints.jsonl") if x["episode_id"] == episode_id}
    chat = TextChat(json.loads((run / "protocol.json").read_text(encoding="utf8")), run, Trace(run / "card_only_diagnostics"))
    output_path = run / "card_only_answers.json"
    rows = json.loads(output_path.read_text(encoding="utf8")) if output_path.exists() else []
    done = {row["checkpoint_id"] for row in rows}
    for original in result.get("answers", []):
        if original["checkpoint_id"] in done:
            continue
        ck = checkpoints[original["checkpoint_id"]]
        user = f"The question is asked at this conversation time: {times[ck['as_of_turn_id']]}. Interpret 'now' relative to that time.\n"
        user += f"Asking user: {ck['asker']['principal_id']}\nQuestion: {ck['query_text']}\nRetrieved passages:\n"
        user += "\n\n".join(original["passages"]) + "\nCurrent user states:\n[]"
        try:
            raw = chat("qa", user)
        except Exception as exc:
            print(json.dumps({"domain":domain,"status":"failed","error":type(exc).__name__}),flush=True)
            break
        score = score_one(checkpoint=ck,pred_output={"action":"answer","answer":raw,"answer_structured":{}})
        rows.append({"checkpoint_id":ck["checkpoint_id"],"as_of":ck["as_of_turn_id"],"answer":raw,
                     "usage":chat.last_meta,"auxiliary_rule_score":score})
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({"domain":domain,"checkpoint":ck["as_of_turn_id"],"hits":score["aux"]["include_hits"],
                          "needed":score["aux"]["include_needed"]}),flush=True)
