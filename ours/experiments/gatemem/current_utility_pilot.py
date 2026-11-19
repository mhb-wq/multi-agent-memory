"""One official GateMem utility checkpoint, replayed at its published boundary.

Shared-pool utility only; not a privacy, deletion, or full benchmark evaluation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/gatemem/official/code")]
from bench.eval.scorer import score_one  # noqa: E402

from ours_memory.control import Actor  # noqa: E402
from ours_memory.groupmem import (  # noqa: E402
    ReplayClock,
    TextChat,
    Trace,
    build_memory,
    close_memory,
    format_card,
    qa_prompt,
)

DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/medical"
EPISODE = "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line"
CHECKPOINT = EPISODE + "_ckpt_01"
RUN = ROOT / "runtime/gatemem-current-utility-20261117"


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def save(name, data):
    (RUN / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def main():
    episode = next(x for x in read_lines(DATA / "episodes.jsonl") if x["episode_id"] == EPISODE)
    checkpoint = next(x for x in read_lines(DATA / "checkpoints.jsonl") if x["checkpoint_id"] == CHECKPOINT)
    stop = next(i for i, x in enumerate(episode["turns"]) if x["turn_id"] == checkpoint["as_of_turn_id"])
    if stop != 16 or checkpoint["query_type"] != "utility":
        raise ValueError("unexpected checkpoint boundary")
    RUN.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / "experiments/groupmembench/protocol-current-diagnostic.json").read_text(encoding="utf8"))
    save("protocol.json", protocol)
    (RUN / "qa.txt").write_text("Answer the user's question using only the retrieved memories and states. "
                                "If the evidence is insufficient, say so. Be concise. End with Final: and your answer.", encoding="utf8")
    (RUN / "judge.txt").write_text("unused", encoding="utf8")
    users = sorted(p["principal_id"] for p in episode["entities"]["principals"])
    cfg = {"space": "GateMem:" + EPISODE, "users": users, "access_mode": "shared_pool"}
    clock = ReplayClock("2026-01-01T00:00:00+00:00")
    trace = Trace(RUN / "diagnostics")
    progress = {"episode_id": EPISODE, "checkpoint_id": CHECKPOINT, "boundary": checkpoint["as_of_turn_id"],
                "turns": [], "status": "running", "note": "utility-only shared pool; no access or deletion evaluation"}
    memory = build_memory(RUN, cfg, trace, clock)
    try:
        previous = []
        for turn in episode["turns"][:stop + 1]:
            moment = turn["timestamp"] + "+00:00" if len(turn["timestamp"]) == 16 else turn["timestamp"]
            clock.value = moment
            trace.context = turn["turn_id"]
            author = turn["speaker"]["principal_id"]
            result = memory.ingest_message(Actor(author, "ingestor", cfg["space"], "pilot"),
                      message_id=turn["turn_id"], text=turn["text"], scope="global", observed_at=moment,
                      metadata={"role":turn["speaker"]["role"], "timestamp":moment, "msg_node":turn["turn_id"]},
                      fallback_category="conversation", prior_context=previous[-10:])
            progress["turns"].append({"turn_id":turn["turn_id"],"status":result.get("status"),
                                       "extraction_status":result.get("extraction_status"),
                                       "cards":sum(len(e.get("card_ids", [])) for e in result.get("events", []))})
            save("progress.json", progress)
            print(json.dumps(progress["turns"][-1]), flush=True)
            if result.get("status") != "done" or result.get("extraction_status") == "skipped_format":
                progress["status"] = "ingestion_failed"
                save("progress.json", progress)
                return
            previous.append({"id": result["message_id"], "speaker":author,"role":turn["speaker"]["role"],
                             "text":turn["text"],"timestamp":moment})
            while sum(len(x["text"]) for x in previous[-10:]) > 6000:
                previous.pop(0)
        actor = Actor(checkpoint["asker"]["principal_id"], "answerer", cfg["space"], "query")
        bundle = memory.read(actor, subject_ids=users, query=checkpoint["query_text"], purpose="current")
        if not memory.validate_bundle(actor, bundle["bundle_id"])["ready"]:
            raise RuntimeError("retrieved bundle not ready")
        question = {"question": checkpoint["query_text"], "asking_user_id":checkpoint["asker"]["principal_id"]}
        chat = TextChat(protocol, RUN, trace)
        raw = chat("qa", qa_prompt(question, bundle["cards"], bundle.get("user_states"), bundle.get("pending_subjects")))
        # Reference fields are consulted only after the answer is generated.
        output = {"action":"answer", "answer":raw, "answer_structured":{}}
        scoring = score_one(checkpoint=checkpoint, pred_output=output)
        save("answer.json", {"checkpoint_id":CHECKPOINT, "question":question, "answer":raw,
                             "cards":[format_card(c) for c in bundle["cards"]], "user_states":bundle.get("user_states", []),
                             "official_auxiliary_rule_score":scoring, "usage":chat.last_meta})
        save("calls.json", memory.store.call_records())
        progress["status"] = "completed"
        save("progress.json", progress)
        print(json.dumps({"status":"completed", "checkpoint":CHECKPOINT, "auxiliary_score":scoring, "answer":raw}, ensure_ascii=False), flush=True)
    finally:
        close_memory(memory)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
