"""Frozen GateMem utility checkpoints; one isolated method per process."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/gatemem/official/code")]
from bench.eval.scorer import score_one  # noqa: E402

from ours_memory.baselines import (  # noqa: E402
    AMEMAdapter,
    Mem0Adapter,
    MemOSNativeAdapter,
    Message,
    NaiveRAGAdapter,
    build_amem_paper,
    build_memos_native,
)
from ours_memory.baselines.base import format_passage  # noqa: E402
from ours_memory.control import Actor  # noqa: E402
from ours_memory.groupmem import ReplayClock, TextChat, Trace, build_memory, close_memory, format_card  # noqa: E402

OUT = ROOT / "runtime/gatemem-compare-20261117"
DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/medical"
EP = "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line"
IDS = [EP + f"_ckpt_{i:02d}" for i in (1, 2, 3)]


def lines(p):
    return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()]


def write(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(p)


def main():
    global DATA, EP, IDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["ours", "naive", "mem0", "amem", "memos"], required=True)
    parser.add_argument("--case", choices=["medical", "household"], default="medical")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-pilot", action="store_true")
    args = parser.parse_args()
    system = args.system
    if args.case == "household":
        if args.resume or args.reuse_pilot:
            raise ValueError("household case uses fresh runs")
        DATA = ROOT / "experiments/gatemem/official/data/gatemem/data/household"
        EP = "household_episode_custom_en_010_rowan_wardrobe_rowan_wash_tailor_linen_private_note"
        IDS = [EP + "_ckpt_01"]
    run = (OUT / "household" if args.case == "household" else OUT) / ("ours_reuse" if args.reuse_pilot else system)
    run.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_pilot:
        if system != "ours" or args.resume:
            raise ValueError("pilot reuse only supports a fresh Ours continuation")
        shutil.copytree(ROOT / "runtime/gatemem-current-utility-20261117", run)
    if args.resume:
        if system != "ours" or not (run / "results.json").exists():
            raise ValueError("resume only supports an existing Ours run")
    elif not args.reuse_pilot:
        run.mkdir(parents=True, exist_ok=False)
    episode = next(x for x in lines(DATA / "episodes.jsonl") if x["episode_id"] == EP)
    checkpoints = [next(x for x in lines(DATA / "checkpoints.jsonl") if x["checkpoint_id"] == key) for key in IDS]
    positions = {x["turn_id"]: i for i, x in enumerate(episode["turns"])}
    expected_positions = [23] if args.case == "household" else [16, 20, 24]
    if [positions[x["as_of_turn_id"]] for x in checkpoints] != expected_positions:
        raise ValueError("unexpected checkpoint boundary")
    protocol = json.loads((ROOT / "experiments/groupmembench/protocol-current-diagnostic.json").read_text(encoding="utf8"))
    write(run / "protocol.json", protocol)
    (run / "qa.txt").write_text("Answer using only the supplied retrieved evidence and states. If evidence is insufficient, say so. Be concise and end with Final: followed by your answer.", encoding="utf8")
    (run / "judge.txt").write_text("unused", encoding="utf8")
    trace = Trace(run / "diagnostics")
    chat = TextChat(protocol, run, trace)
    key, base = os.environ["DEEPSEEK_API_KEY"], os.environ["OURS_LLM_BASE_URL"]
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = base
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    users = sorted(p["principal_id"] for p in episode["entities"]["principals"])
    clock = ReplayClock("2026-01-01T00:00:00+00:00")
    cfg = {"space": "GateMem:" + EP, "users": users, "access_mode": "shared_pool"}
    memory = None
    data = (json.loads((run / "results.json").read_text(encoding="utf8")) if args.resume else
            {"system":system, "episode_id":EP, "checkpoint_ids":IDS, "status":"running", "turns":[], "answers":[],
             "scope":"shared-pool utility only; fixed answer action, no privacy/deletion scoring"})
    if args.resume:
        data.setdefault("resume_history", []).append({"status": data.get("status"), "error": data.get("error"),
                                                       "detail": data.get("detail"), "completed_turns": len(data["turns"])})
    if args.reuse_pilot:
        pilot = json.loads((run / "progress.json").read_text(encoding="utf8"))
        if pilot.get("status") != "completed" or len(pilot.get("turns", [])) != 17:
            raise ValueError("pilot snapshot incomplete")
        data["turns"] = [{"id": x["turn_id"], "status": x["status"], "cards": x["cards"], "reused": True}
                         for x in pilot["turns"]]
        data["scope"] += "; continuation from previously successful, non-independent pilot snapshot"
    data["status"] = "running"
    data.pop("error", None)
    data.pop("detail", None)
    write(run / "results.json", data)
    try:
        if system == "ours":
            memory = build_memory(run, cfg, trace, clock)
            adapter = None
        elif system == "naive":
            adapter = NaiveRAGAdapter()
        elif system == "amem":
            adapter = AMEMAdapter(build_amem_paper(str(ROOT / ".cache/baselines/agentic-memory-repro"),
                       llm_model="gpt-6-sol", api_key=key, base_url=base, temperature=0.0))
        elif system == "memos":
            cube = build_memos_native(str(run / "memos"), api_key=key, base_url=base, model="gpt-6-sol",
                                      embedding_model="sentence-transformers/all-MiniLM-L6-v2")
            cube.text_mem.extractor_llm.config.extra_body = {}
            adapter = MemOSNativeAdapter(cube.text_mem)
        else:
            import mem0.memory.main as mem0_main
            import mem0.memory.setup as mem0_setup
            from mem0 import Memory
            from mem0.configs.base import MemoryConfig
            home = run / "mem0-home"
            home.mkdir()
            mem0_main.mem0_dir = str(home)
            mem0_setup.mem0_dir = str(home)
            config = {"llm":{"provider":"openai","config":{"model":"gpt-6-sol","temperature":0.0,
                        "api_key":key,"openai_base_url":base,"max_tokens":2048}},
                      "embedder":{"provider":"huggingface","config":{"model":"sentence-transformers/all-MiniLM-L6-v2"}},
                      "vector_store":{"provider":"qdrant","config":{"path":str(run/"qdrant"),
                                      "collection_name":"gate","embedding_model_dims":384}},
                      "history_db_path":str(run/"history.db")}
            adapter = Mem0Adapter(Memory(MemoryConfig(**config)), group_id="group")
        previous = []
        for i, turn in enumerate(episode["turns"][:max(expected_positions) + 1]):
            moment = turn["timestamp"] + "+00:00" if len(turn["timestamp"]) == 16 else turn["timestamp"]
            clock.value = moment
            trace.context = turn["turn_id"]
            author = turn["speaker"]["principal_id"]
            if (args.resume or args.reuse_pilot) and i < len(data["turns"]):
                source = next((x for x in memory.store.all("source_message_v2") if x.get("external_id") == turn["turn_id"]), None)
                if not source:
                    raise RuntimeError("missing source for resume: " + turn["turn_id"])
                previous.append({"id":source["id"],"speaker":author,"role":turn["speaker"]["role"],
                                 "text":turn["text"],"timestamp":moment})
                while sum(len(x["text"]) for x in previous[-10:]) > 6000:
                    previous.pop(0)
                if not args.reuse_pilot or i != 16:
                    continue
                asker = checkpoints[0]["asker"]["principal_id"]
                question = checkpoints[0]["query_text"]
                actor = Actor(asker,"answerer",cfg["space"],"query")
                bundle = memory.read(actor,subject_ids=users,query=question,purpose="current")
                if not memory.validate_bundle(actor,bundle["bundle_id"])["ready"]:
                    raise RuntimeError("pilot evidence bundle not ready")
                passages = [format_card(c) for c in bundle["cards"]]
                state = bundle.get("user_states", [])
                user = f"Asking user: {asker}\nQuestion: {question}\nRetrieved passages:\n" + "\n\n".join(passages)
                user += "\nCurrent user states:\n" + json.dumps(state, ensure_ascii=False)
                raw = chat("qa", user)
                score = score_one(checkpoint=checkpoints[0],pred_output={"action":"answer","answer":raw,"answer_structured":{}})
                data["answers"].append({"checkpoint_id":checkpoints[0]["checkpoint_id"],"as_of":"t017", "question":question,
                    "answer":raw,"passages":passages,"user_states":state,"qa_meta":chat.last_meta,"auxiliary_rule_score":score})
                write(run / "results.json", data)
                print(json.dumps({"system":"ours_reuse","checkpoint":checkpoints[0]["checkpoint_id"],
                                  "hits":score["aux"]["include_hits"],"needed":score["aux"]["include_needed"]}), flush=True)
                continue
            start = time.perf_counter()
            if memory:
                for attempt in range(2):
                    result = memory.ingest_message(Actor(author,"ingestor",cfg["space"],"pilot"),
                        message_id=turn["turn_id"], text=turn["text"], scope="global", observed_at=moment,
                        metadata={"role":turn["speaker"]["role"],"timestamp":moment,"msg_node":turn["turn_id"]},
                        fallback_category="conversation", prior_context=previous[-10:])
                    if result.get("status") in {"done", "accepted_pending"}:
                        break
                    data.setdefault("retries", []).append({"turn_id": turn["turn_id"], "attempt": attempt + 1,
                                                         "status": result.get("status")})
                    write(run / "results.json", data)
                if result.get("status") not in {"done", "accepted_pending"} or result.get("extraction_status") == "skipped_format":
                    raise RuntimeError("memory ingest incomplete at " + turn["turn_id"])
                count = sum(len(x.get("card_ids", [])) for x in result.get("events", []))
                previous.append({"id":result["message_id"],"speaker":author,"role":turn["speaker"]["role"],
                                 "text":turn["text"],"timestamp":moment})
                while sum(len(x["text"]) for x in previous[-10:]) > 6000:
                    previous.pop(0)
            else:
                adapter.ingest(Message(id=turn["turn_id"],text=turn["text"],author=author,timestamp=moment,
                                       channel=EP,role=turn["speaker"]["role"]))
                count = None
            turn_status = result.get("status") if memory else "done"
            data["turns"].append({"id":turn["turn_id"],"status":turn_status,"cards":count,
                                  "processing_status": result.get("processing_status") if memory else None,
                                  "elapsed":round(time.perf_counter()-start,3)})
            write(run / "results.json", data)
            print(json.dumps({"system":system,"turn":i+1,"status":turn_status}), flush=True)
            for ck in [x for x in checkpoints if x["as_of_turn_id"] == turn["turn_id"]]:
                question = ck["query_text"]
                asker = ck["asker"]["principal_id"]
                if memory:
                    actor = Actor(asker,"answerer",cfg["space"],"query")
                    bundle = memory.read(actor,subject_ids=users,query=question,purpose="current")
                    if not memory.validate_bundle(actor,bundle["bundle_id"])["ready"]:
                        raise RuntimeError("evidence bundle not ready")
                    passages = [format_card(c) for c in bundle["cards"]]
                    state = bundle.get("user_states", [])
                    data.setdefault("incomplete_checkpoints", {})[ck["checkpoint_id"]] = {
                        "incomplete_message_count": bundle.get("incomplete_message_count", 0),
                        "pending_state_count": bundle.get("pending_state_count", 0)}
                else:
                    passages = [format_passage(p) for p in adapter.retrieve(question,asker,top_k=10)]
                    state = []
                user = f"Asking user: {asker}\nQuestion: {question}\nRetrieved passages:\n" + "\n\n".join(passages)
                user += "\nCurrent user states:\n" + json.dumps(state, ensure_ascii=False)
                raw = chat("qa", user)
                answer = {"action":"answer","answer":raw,"answer_structured":{}}
                # Gold is used only after this independent answer has been generated.
                score = score_one(checkpoint=ck, pred_output=answer)
                data["answers"].append({"checkpoint_id":ck["checkpoint_id"],"as_of":ck["as_of_turn_id"],
                    "question":question,"answer":raw,"passages":passages,"user_states":state,
                    "qa_meta":chat.last_meta,"auxiliary_rule_score":score})
                write(run / "results.json", data)
                print(json.dumps({"system":system,"checkpoint":ck["checkpoint_id"],"hits":score["aux"]["include_hits"],
                                  "needed":score["aux"]["include_needed"]}), flush=True)
        data["status"] = "completed"
    except Exception as exc:
        data.update(status="failed", error=type(exc).__name__, detail=str(exc)[:300].replace(key,"[redacted]"))
    finally:
        if memory:
            write(run / "calls.json", memory.store.call_records())
            close_memory(memory)
        write(run / "results.json", data)
        print(json.dumps({"system":system,"status":data["status"],"turns":len(data["turns"]),"answers":len(data["answers"]),"error":data.get("error")}), flush=True)


if __name__ == "__main__":
    main()
