"""Pinned GroupMemBench pilot: real memory, daily replay, immutable QA snapshot.

No gold is loaded by ingestion. Diagnostic logs are private research artifacts,
not service responses. External model calls occur only in ingest/evaluate/probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .control import Actor
from .llm import CompatibleJsonModel, RejectRedirects
from .models import ValidationError, canonical, stable_id

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT / "experiments/groupmembench"
OFFICIAL = EXPERIMENT / "official"


class ModelRequestError(RuntimeError):
    """Network/HTTP/timeout failure: the model was never given a valid answer chance."""


class ModelOutputError(RuntimeError):
    """The model answered, but the output is empty, truncated, or unusable."""

    def __init__(self, message, output=""):
        super().__init__(message)
        self.output = output


class ModelConfigurationError(RuntimeError):
    """The provider ignored a required protocol setting (e.g. reasoning disabled)."""


def read_json(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def normalized_time(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Synthetic source timestamps have no zone: UTC is an experiment convention.
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()


def load_corpus(path, domain):
    data = read_json(path)
    rows = []
    seen = set()
    for channel, messages in data.items():
        for message in messages:
            key = stable_id("gmb", domain, channel, message["msg_node"])
            if key in seen:
                raise ValidationError("重复来源ID")
            seen.add(key)
            # Only conversational metadata is used. No gold, noise or decision labels.
            metadata = {
                k: message.get(k)
                for k in ("role", "reply_to", "topic", "phase_name", "msg_node", "timestamp")
            }
            metadata.update(domain=domain, channel=channel)
            rows.append(
                {
                    "id": key,
                    "author": message["author"],
                    "text": message["content"],
                    "time": normalized_time(message["timestamp"]),
                    "metadata": metadata,
                    "item_id": stable_id("item", domain, channel, message.get("topic")),
                }
            )
    # Replay globally by time; stable ties do not depend on gold or question order.
    rows.sort(key=lambda r: (r["time"], r["metadata"]["channel"], r["metadata"]["msg_node"]))
    if not rows:
        raise ValidationError("语料为空")
    return rows


def recent_source_context(rows, index):
    """Earlier messages in the same channel; control turns end the window."""
    if not 0 <= index < len(rows):
        raise ValidationError("前文索引无效")
    channel = rows[index].get("metadata", {}).get("channel")
    chosen, size = [], 0
    controls = re.compile(r"\b(delete|forget|erase|remove|wipe|purge|revoke|authorize|grant|permit)\b", re.I)
    for row in reversed(rows[max(0, index - 10):index]):
        if row.get("metadata", {}).get("channel") != channel:
            continue
        text = row.get("text") or ""
        if controls.search(text):
            break
        if len(text) + size > 6000:
            break
        chosen.append({"id": row["id"], "speaker": row["author"],
                       "role": row.get("metadata", {}).get("role"), "text": text,
                       "timestamp": row.get("time")})
        size += len(text)
    return list(reversed(chosen))


def control_config(users, space, access_mode="shared_pool"):
    if access_mode != "shared_pool":
        raise ValidationError("仅支持共享记忆池")
    actions = ["read", "write", "confirm"]
    return {
        "access_mode": access_mode,
        "principals": users,
        "categories": ["conversation"],
        "agents": {
            "ingestor": {"categories": ["conversation"], "actions": actions, "controls": []},
            "maintainer": {"categories": ["conversation"], "actions": ["read"], "controls": ["sleep"],
                           "trusted_internal_maintenance": True},
            "answerer": {"categories": ["conversation"], "actions": ["read"], "controls": []},
            "controller": {"categories": ["conversation"], "actions": ["read", "delete"], "controls": ["grant", "revoke"]},
        },
        "spaces": {
            space: {
                "members": users,
                "agents": ["ingestor", "maintainer", "answerer", "controller"],
                "grant_managers": {u: [u] for u in users},
            }
        },
        "grants": [],
    }


def validate_protocol(spec: dict) -> None:
    """Fail before spending tokens if the approved protocol is not the one implemented."""
    qa, judge = spec.get("qa") or {}, spec.get("judge") or {}
    retrieval, sleep = spec.get("retrieval") or {}, spec.get("sleep") or {}
    method = spec.get("method") or {}
    for key in ("joint_state_update_enabled", "immediate_state_maintenance_enabled", "independent_state_read_enabled"):
        if key in method and type(method[key]) is not bool:
            raise ValidationError(f"方法配置 {key} 必须为布尔值")
    checks = [
        (qa.get("max_tokens") == 512, "QA max_tokens 必须为 512"),
        (qa.get("temperature") == 0.2, "QA temperature 必须为 0.2"),
        (qa.get("gold_visible") is False, "QA 不得看到标准答案"),
        (qa.get("include_asking_user_id") is True, "QA 必须携带 asking_user_id"),
        (judge.get("max_tokens") == 2048, "judge max_tokens 必须为 2048"),
        (judge.get("temperature") == 1.0, "judge temperature 必须为 1.0"),
        (judge.get("feedback_to_qa") is False, "judge 不得把意见回传给 QA"),
        (
            judge.get("input_fields") == ["question", "gold_answer", "agent_final_answer"],
            "judge 输入字段不符",
        ),
        (retrieval.get("candidate_limit") == 20, "候选上限必须为 20"),
        (retrieval.get("include_asking_user_id") is True, "检索必须携带 asking_user_id"),
        (retrieval.get("question_type_visible") is False, "QA 不得看到题型标签"),
        (sleep.get("qa_writes_memory") is False, "答题不得写入记忆"),
        (spec.get("reasoning_policy", "disabled") in {"disabled", "provider_default"}, "推理策略不合法"),
        (spec.get("reasoning_policy", "disabled") != "disabled" or "disabled" in str(spec.get("reasoning_mode", "")).lower(), "必须显式记录推理模式"),
    ]
    for ok, message in checks:
        if not ok:
            raise ValidationError("协议不符：" + message)
    for stage, value in (("qa", qa), ("judge", judge)):
        if not value.get("model") or not value.get("system_prompt"):
            raise ValidationError(f"协议缺少{stage}模型或提示词")


def load_question_row(domain: str, qtype: str, question_id: str) -> dict:
    path = OFFICIAL / "questions" / domain / (qtype + ".jsonl")
    matches = [
        json.loads(line)
        for line in path.read_text("utf8").splitlines()
        if line.strip() and json.loads(line).get("id") == question_id
    ]
    if len(matches) != 1:
        raise ValidationError(f"题目不存在或ID不唯一：{domain}/{qtype}/{question_id}")
    return matches[0]


def prepare(run, corpus, protocol, manifest, message_limit=None):
    run = Path(run).resolve()
    if message_limit is not None and message_limit < 1:
        raise ValidationError("消息数量须为正整数")
    spec = read_json(protocol)
    validate_protocol(spec)
    domain = spec["domain"]
    rows = load_corpus(corpus, domain)
    selection = read_json(manifest)
    keys = [(q["domain"], q["type"], q["id"]) for values in selection["splits"].values() for q in values]
    if len(keys) != len(set(keys)) or any(k[0] != domain for k in keys):
        raise ValidationError("抽样重复、交叉或域不匹配")
    users = sorted({r["author"] for r in rows})
    for split, items in selection["splits"].items():
        for item in items:
            question = load_question_row(item["domain"], item["type"], item["id"])
            if question.get("asking_user_id") not in users:
                raise ValidationError(f"{split} 中的 asking_user_id 未登记：{item['id']}")
    if run.exists():
        raise ValidationError("准备需要全新输出目录；已有运行请直接恢复")
    run.mkdir(parents=True)
    save(run / "protocol.json", spec)
    save(run / "manifest.json", selection)
    prompts = {}
    for stage in ("qa", "judge"):
        source = EXPERIMENT / spec[stage]["system_prompt"]
        target = run / f"{stage}.txt"
        target.write_bytes(source.read_bytes())
        prompts[stage] = file_hash(target)
    question_files = {
        str(p.relative_to(OFFICIAL)): file_hash(p)
        for p in sorted((OFFICIAL / "questions" / domain).glob("*.jsonl"))
    }
    selected_rows = rows[:message_limit] if message_limit else rows
    config = {
        "domain": domain,
        "space": "GroupMemBench:" + domain,
        "corpus": str(Path(corpus).resolve()),
        "corpus_sha256": file_hash(corpus),
        "protocol_sha256": file_hash(run / "protocol.json"),
        "manifest_sha256": file_hash(run / "manifest.json"),
        "prompts": prompts,
        "code_sha256": code_hash(),
        "question_files": question_files,
        "users": users,
        "protocol_profile": spec.get("current_parameter_profile"),
        "message_limit": message_limit,
        "message_count": len(selected_rows),
        "full_message_count": len(rows),
        "mode": "smoke" if message_limit else "pilot",
        "final_time": selected_rows[-1]["time"][:10] + "T23:59:59.999999+00:00",
        "sleep": ("immediate after each card" if (spec.get("method", {}).get("joint_state_update_enabled")
                    and spec.get("method", {}).get("immediate_state_maintenance_enabled"))
                  else "after each 20 successfully ingested messages; no query-triggered maintenance"),
        "method": spec.get("method", {}),
        "access_mode": "shared_pool",
        "query_memory_writes": False,
    }
    save(run / "run.json", config)
    save(run / "checkpoint.json", {"next_index": 0, "completed_days": [], "ready": False})
    return config


def verify_run(run):
    run = Path(run)
    cfg = read_json(run / "run.json")
    checks = [
        file_hash(cfg["corpus"]) == cfg["corpus_sha256"],
        code_hash() == cfg["code_sha256"],
        file_hash(run / "protocol.json") == cfg["protocol_sha256"],
        file_hash(run / "manifest.json") == cfg["manifest_sha256"],
    ]
    checks.extend(file_hash(run / f"{k}.txt") == v for k, v in cfg["prompts"].items())
    checks.extend(file_hash(OFFICIAL / name) == value for name, value in cfg["question_files"].items())
    if not all(checks):
        raise ValidationError("运行配置、代码、语料或提示词已改变；请建立新运行，不能混用旧结果")
    return cfg


class Trace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.context = "startup"

    def __call__(self, stage, payload):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / (self.context + ".jsonl")).open("a", encoding="utf8") as stream:
            stream.write(json.dumps({"stage": stage, "data": payload}, ensure_ascii=False) + "\n")


class TracedModel:
    def __init__(self, model, trace):
        self.model, self.trace = model, trace

    def complete(self, stage, payload):
        start = time.perf_counter()
        if stage == "redaction_plan":
            self.trace(stage + ":input", {"redacted": True})
        else:
            self.trace(stage + ":input", payload)
        try:
            result = self.model.complete(stage, payload)
            self.trace(stage + ":output", {"result": "redacted" if stage == "redaction_plan" else result,
                                           "elapsed": time.perf_counter() - start})
            return result
        except Exception as exc:
            self.trace(
                stage + ":failure", {"error": type(exc).__name__, "elapsed": time.perf_counter() - start}
            )
            raise


class ReplayClock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def build_memory(run, cfg, trace, clock):
    from .config import PERSONAL_STATE_TEXT_KEYS, validate_current_method
    from .memos_backend import MemOSBackend
    from .multiuser import MultiUserMemory
    from .scheduler import MemOSSleepScheduler
    from .store import Store

    protocol = read_json(run / "protocol.json")
    method = protocol.get("method") or {}
    if any(method.get(key, True) is not True for key in (
        "joint_state_update_enabled", "immediate_state_maintenance_enabled", "independent_state_read_enabled"
    )):
        raise ValidationError("仅支持最新版联合判断、即时维护和独立状态读取")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValidationError("缺少DEEPSEEK_API_KEY；没有发出模型请求")
    base = os.environ.get("OURS_LLM_BASE_URL", "https://api.deepseek.com")
    settings = Settings(
        database=str(run / "state.sqlite3"),
        team_agents={cfg["space"]: ["ingestor", "maintainer", "answerer", "controller"]},
        model=protocol["memory_model"],
        base_url=base,
        card_repair_attempts=1,
        retries=1,
        request_options=({"thinking": {"type": "disabled"}} if protocol.get("reasoning_policy", "disabled") == "disabled" else {}),
        state_keys={
            k: {"type": "text", "unit": None}
            for k in ("deadline", "goal", "constraint", "preference", "identity", "task_plan", "commitment", *PERSONAL_STATE_TEXT_KEYS)
        },
    )
    settings.validate()
    validate_current_method(settings)
    if cfg.get("access_mode", "shared_pool") != "shared_pool":
        raise ValidationError("仅支持共享记忆池")
    store = Store(settings.database)
    native = {
        "directory": str(run / "cubes"),
        "text_memory": {
            "backend": "general_text",
            "config": {
                "extractor_llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": protocol["memory_model"],
                        "api_key": key,
                        "api_base": base,
                        "temperature": 0,
                        "max_tokens": 4096,
                        "extra_body": ({"thinking": {"type": "disabled"}} if protocol.get("reasoning_policy", "disabled") == "disabled" else {}),
                        "remove_think_prefix": True,
                    },
                },
                "embedder": {
                    "backend": "sentence_transformer",
                    "config": {
                        "model_name_or_path": protocol["embedding"]["model"],
                        "trust_remote_code": False,
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": "overridden-per-policy",
                        "distance_metric": "cosine",
                        "vector_dimension": protocol["embedding"]["dimensions"],
                    },
                },
            },
        },
    }
    embedding_base = protocol["embedding"].get("base_url")
    if embedding_base:
        # The formal GateMem runner supplies the same loopback MiniLM service
        # used by the baselines. Historical runs without this field are unchanged.
        native["text_memory"]["config"]["embedder"] = {
            "backend": "universal_api",
            "config": {"provider": "openai", "model_name_or_path": protocol["embedding"]["model"],
                       "embedding_dims": protocol["embedding"]["dimensions"],
                       "base_url": embedding_base, "api_key": "local-only"},
        }
    backend = MemOSBackend(native, store)
    memory = MultiUserMemory(
        settings,
        backend,
        TracedModel(CompatibleJsonModel(settings, store), trace),
        store,
        control_config(cfg["users"], cfg["space"], cfg.get("access_mode", "shared_pool")),
        clock,
    )
    memory.scheduler = MemOSSleepScheduler(memory.core)
    memory.observer = trace
    return memory


def close_memory(memory):
    memory.scheduler.close()
    memory.backend.close()
    memory.store.close()


def snapshot(memory):
    # Evidence, state and controls only; bundles and accounting may change during QA.
    return digest(
        {
            "objects": {
                k: memory.store.all(k)
                for k in ("event", "card", "memory", "envelope_v2", "message_v2", "source_message_v2",
                          "pending_state", "grant_v2", "user_state", "state_vector", "joint_card_decision", "joint_state_history")
            },
            "states": memory.store.states(),
            "dirty": memory.store.dirty_states(),
        }
    )


def pending_card_ids(memory):
    return sorted({row["card_id"] for row in memory.store.all("pending_state")})


def reasoning_tokens(memory):
    total = 0
    for record in memory.store.call_records():
        usage = record.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        total += details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
    return total


def ingest_run(run, cfg, memory, clock, trace):
    immediate = True
    checkpoint = read_json(run / "checkpoint.json")
    if checkpoint["ready"]:
        if snapshot(memory) != checkpoint["snapshot"]:
            raise ValidationError("冻结记忆快照发生变化")
        return checkpoint
    rows = load_corpus(cfg["corpus"], cfg["domain"])
    if cfg["message_limit"]:
        rows = rows[: cfg["message_limit"]]
    days = defaultdict(list)
    for index, row in enumerate(rows):
        days[row["time"][:10]].append((index, row))

    def maintain(day, final=False):
        trace.context = ("final-" if final else "sleep-") + day
        before = pending_card_ids(memory)
        trace(
            "before",
            {
                "pending_card_ids": before,
                "pending": memory.store.all("pending_state"),
                "states": memory.store.states(),
            },
        )
        # Daily maintenance covers that day's task. The final pass must drain every
        # remaining task id, otherwise evidence from an earlier day is silently kept.
        tasks = ["__all_pending__"]
        results = [
            memory.sleep(Actor(cfg["users"][0], "maintainer", cfg["space"], task)) for task in tasks
        ] or [{"jobs": []}]
        after = pending_card_ids(memory)
        trace(
            "after",
            {
                "results": results,
                "pending_card_ids": after,
                "pending": memory.store.all("pending_state"),
                "states": memory.store.states(),
                "processed_card_ids": sorted(set(before) - set(after)),
                "failed_card_ids": sorted(set(before) & set(after)),
            },
        )
        if any(
            job["status"] not in {"completed", "disabled", "needs_attention"}
            for result in results
            for job in result.get("jobs", [])
        ):
            raise ValidationError("Sleep未完成；保持断点，禁止正式答题")

    for day, items in days.items():
        if day in checkpoint["completed_days"]:
            continue
        for index, row in items:
            if index < checkpoint["next_index"]:
                continue
            clock.value = row["time"]
            trace.context = row["id"]
            trace("source", row)
            started = time.perf_counter()
            result = memory.ingest_message(
                Actor(row["author"], "ingestor", cfg["space"], day),
                message_id=row["id"],
                text=row["text"],
                item_id=row["item_id"],
                scope="global",
                observed_at=row["time"],
                metadata=row["metadata"],
                fallback_category="conversation",
                prior_context=recent_source_context(rows, index),
            )
            trace("ingest_result", {"result": result, "elapsed": time.perf_counter() - started})
            # A task status is the contract, not the absence of an exception.
            # Some extractors emit an empty fragment alongside real cards; the
            # message is still covered. Only an all-empty message is a lost source.
            events = result.get("events") or []
            if result.get("extraction_status") == "skipped_format":
                trace("extraction_skipped", {"message_id": row["id"], "attempts": result["extraction_attempts"]})
                if not immediate and (index + 1) % 20 == 0:
                    maintain(day)
                checkpoint["next_index"] = index + 1
                save(run / "checkpoint.json", checkpoint)
                continue
            if result.get("status") == "accepted_pending":
                trace("ingest_pending", {"message_id": row["id"], "processing_status": result.get("processing_status")})
                checkpoint.setdefault("pending_messages", []).append({"id": row["id"],
                    "processing_status": result.get("processing_status")})
                checkpoint["next_index"] = index + 1
                save(run / "checkpoint.json", checkpoint)
                continue
            zero_card = (memory.core.focused_schema and result.get("extraction_status") == "extracted" and not events)
            if result.get("status") != "done" or (not events and not zero_card):
                raise ValidationError("记忆写入任务未成功；没有推进断点")
            if any(event.get("status") != "done" for event in events):
                raise ValidationError("存在未完成的片段；没有推进断点")
            if events and not any(event.get("card_ids") for event in events):
                raise ValidationError("全部片段都是空卡片；没有推进断点")
            if not immediate and (index + 1) % 20 == 0:
                maintain(day)
                print(json.dumps({"ingested": index + 1, "total": len(rows)}), flush=True)
            checkpoint["next_index"] = index + 1
            save(run / "checkpoint.json", checkpoint)
        checkpoint["completed_days"].append(day)
        save(run / "checkpoint.json", checkpoint)
    clock.value = cfg["final_time"]
    observed_reasoning = reasoning_tokens(memory)
    if observed_reasoning and read_json(run / "protocol.json").get("reasoning_policy", "disabled") == "disabled":
        trace("reasoning_incompatible", {"reasoning_tokens": observed_reasoning})
        raise ValidationError("推理模式未关闭；不得静默把配置失败计入答题成绩")
    checkpoint.update(ready=True, snapshot=snapshot(memory))
    save(run / "checkpoint.json", checkpoint)
    save(
        run / "ingest_summary.json",
        {
            "messages": len(rows),
            "cards": len(memory.store.all("card")),
            "legacy_states": len(memory.store.states()),
            "user_state_subjects": len(memory.store.all("user_state")),
            "user_state_entries": sum(len(rows) for state in memory.store.all("user_state") for rows in state.get("topics", {}).values()),
            "days": len(days),
            "mode": cfg["mode"],
            "reasoning_tokens": observed_reasoning,
            "reasoning_mode_effective": observed_reasoning == 0,
            "method_call_records": memory.store.call_records(),
            "cost_note": "call_records include memos_extract and method calls; local embedding has no API fee. USD requires provider pricing; not an invoice.",
        },
    )
    return checkpoint


def extract_final(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith(("final:", "final answer:")):
            return line.split(":", 1)[1].strip()
    return lines[-1] if lines else ""


def parse_judgment(text):
    final = extract_final(text).lower()
    if any(word in final for word in ("incorrect", "wrong", "not correct")):
        return "Incorrect"
    return "Correct" if "correct" in final else "Unclear"


class TextChat:
    def __init__(self, protocol, run, trace):
        self.protocol, self.run, self.trace = protocol, run, trace
        self.last_meta = {}
        self.last_output = ""

    def __call__(self, stage, user):
        spec = self.protocol[stage]
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValidationError("缺少DEEPSEEK_API_KEY")
        body = {
            "model": spec["model"],
            "messages": [
                {"role": "system", "content": (self.run / f"{stage}.txt").read_text("utf8")},
                {"role": "user", "content": user},
            ],
            "max_tokens": spec["max_tokens"],
            "temperature": spec["temperature"],
            "stream": False,
        }
        if self.protocol.get("reasoning_policy", "disabled") == "disabled":
            body["thinking"] = {"type": "disabled"}
        self.trace(stage + ":input", body)
        started = time.perf_counter()
        request = urllib.request.Request(
            os.environ.get("OURS_LLM_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
            data=canonical(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "User-Agent": "python-requests/2.32.3"},
        )
        # One judge call, no self-consistency or feedback loop. Failed requests remain failures.
        try:
            with urllib.request.build_opener(RejectRedirects()).open(request, timeout=120) as response:
                result = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_meta = {
                "error": type(exc).__name__,
                "http_status": getattr(exc, "code", None),
                "elapsed": time.perf_counter() - started,
            }
            self.trace(stage + ":request_failure", self.last_meta)
            raise ModelRequestError(f"{stage}请求失败") from None
        except ValueError:
            self.last_meta = {"error": "invalid_json", "elapsed": time.perf_counter() - started}
            self.trace(stage + ":invalid_output", self.last_meta)
            raise ModelOutputError(f"{stage}响应不是可解析的JSON") from None
        try:
            choice = result["choices"][0]
            output = choice["message"].get("content")
        except (KeyError, IndexError, TypeError):
            self.last_meta = {"error": "malformed_response", "elapsed": time.perf_counter() - started}
            self.trace(stage + ":invalid_output", self.last_meta)
            raise ModelOutputError(f"{stage}响应缺少必要字段") from None
        usage = result.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
        self.last_meta = {
            "usage": usage,
            "finish_reason": choice.get("finish_reason"),
            "served_model": result.get("model"),
            "reasoning_tokens": reasoning,
            "elapsed": time.perf_counter() - started,
        }
        self.last_output = output if isinstance(output, str) else ""
        self.trace(
            stage + ":output",
            {
                "content": output,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "served_model": result.get("model"),
                "reasoning_tokens": reasoning,
                "elapsed": self.last_meta["elapsed"],
            },
        )
        if choice.get("finish_reason") != "stop":
            self.trace(stage + ":truncated", self.last_meta)
            raise ModelOutputError(f"{stage}输出未完整完成", self.last_output)
        if not isinstance(output, str) or not output.strip():
            self.trace(stage + ":empty", self.last_meta)
            raise ModelOutputError(f"{stage}输出为空", self.last_output)
        if reasoning and self.protocol.get("reasoning_policy", "disabled") == "disabled":
            # A disabled reasoning mode that the provider ignored is a configuration
            # incompatibility, not a wrong answer. Never hide it inside accuracy.
            self.trace(stage + ":reasoning_not_disabled", self.last_meta)
            raise ModelConfigurationError(f"{stage}推理模式未关闭（reasoning_tokens={reasoning}）")
        return output


def format_card(card):
    metadata = card.get("metadata", {})
    tags = {
        "user": card.get("author_id"),
        "speaker_role": metadata.get("role"),
        "channel": metadata.get("channel"),
        "phase_name": metadata.get("phase_name"),
        "topic": metadata.get("topic"),
        "timestamp": metadata.get("timestamp"),
        "reply_to": metadata.get("reply_to"),
        "subject": card.get("subject") if card.get("state_eligible", True) else "unresolved/evidence-only",
    }
    header = " / ".join(f"{k}={v}" for k, v in tags.items() if v)
    note = f"\nUse for this question: {card['usage_note']}" if card.get("usage_note") else ""
    return f"[{header}]\n{card['claim_text']}\nSource quote: {card.get('source_quote', '')}{note}"


def qa_prompt(question, cards, user_states=None, pending_subjects=None):
    passages = "\n\n".join(f"[{i}] {format_card(c)}" for i, c in enumerate(cards, 1))
    return (
        f"Asking user: {question['asking_user_id']}\n\nQuestion:\n{question['question']}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        f"Current user states (pending subjects may have newer cards):\n"
        f"{json.dumps({'user_states': user_states or [], 'pending_subjects': pending_subjects or []}, ensure_ascii=False)}\n\n"
        "Answer using the retrieved passages and user states."
    )


def selected_questions(run, split, limit=None):
    manifest = read_json(run / "manifest.json")
    selected = manifest["splits"][split]
    if limit:
        # 12-question smoke is balanced: first two from each category, not first two categories.
        groups = defaultdict(list)
        for row in selected:
            groups[row["type"]].append(row)
        selected = [
            q
            for i in range(max(map(len, groups.values())))
            for group in groups.values()
            for q in group[i : i + 1]
        ][:limit]
    result = []
    for row in selected:
        path = OFFICIAL / "questions" / row["domain"] / (row["type"] + ".jsonl")
        questions = [json.loads(line) for line in path.read_text("utf8").splitlines() if line.strip()]
        matches = [q for q in questions if q["id"] == row["id"]]
        if len(matches) != 1:
            raise ValidationError("题目不存在或ID不唯一")
        result.append({**matches[0], "type": row["type"], "domain": row["domain"]})
    return result


def evaluate(run, cfg, memory, clock, trace, chat, split, limit=None):
    checkpoint = read_json(run / "checkpoint.json")
    if cfg["mode"] != "pilot" or not checkpoint["ready"]:
        raise ValidationError("语料未完整写入并完成Sleep，禁止正式评测")
    clock.value = cfg["final_time"]
    if snapshot(memory) != checkpoint["snapshot"]:
        raise ValidationError("答题前记忆快照不一致")
    output = run / "results" / split
    output.mkdir(parents=True, exist_ok=True)
    for question in selected_questions(run, split, limit):
        identifier = question["type"] + "__" + question["id"]
        path = output / (identifier + ".json")
        attempt = 1
        if path.exists():
            prior = read_json(path)
            # Completed/decided/known failures are never rerolled automatically.
            # A crash leaves "running"; only that interrupted attempt is resumed.
            if prior.get("status") != "running":
                continue
            attempt = int(prior.get("attempt", 1)) + 1
        trace.context = split + "-" + identifier
        record = {
            "id": question["id"],
            "type": question["type"],
            "split": split,
            "query": question["question"],
            "asking_user_id": question["asking_user_id"],
            "attempt": attempt,
            "status": "running",
            "stage": "retrieve",
        }
        # Persist attempt before calls: a process crash cannot silently trigger another judge.
        save(path, record)
        start = time.perf_counter()
        stage = "retrieve"
        qa_meta = judge_meta = None
        try:
            actor = Actor(question["asking_user_id"], "answerer", cfg["space"], identifier)
            use_current = True
            bundle = memory.read(
                actor,
                subject_ids=cfg["users"],
                query=question["question"],
                purpose="current" if use_current else "evidence",
                target_time=None if use_current else clock.value,
            )
            record.update(
                bundle_id=bundle["bundle_id"],
                evidence_count=len(bundle["cards"]),
                retrieved_docs=[format_card(c) for c in bundle["cards"]],
                card_ids=[c["id"] for c in bundle["cards"]],
                user_states=bundle.get("user_states", []),
                pending_subjects=bundle.get("pending_subjects", []),
                stage="qa",
            )
            if not memory.validate_bundle(actor, bundle["bundle_id"])["ready"]:
                raise ValidationError("证据包已失效")
            stage = "qa"
            raw = chat("qa", qa_prompt(question, bundle["cards"], bundle.get("user_states"), bundle.get("pending_subjects")))
            qa_meta = getattr(chat, "last_meta", None)
            record.update(qa_meta=qa_meta)
            if not memory.validate_bundle(actor, bundle["bundle_id"])["ready"]:
                raise ValidationError("回答交付前证据包已失效")
            record.update(agent_output=raw, agent_answer=extract_final(raw), stage="judge")
            save(path, record)
            stage = "judge"
            try:
                verdict = chat(
                    "judge",
                    f"Question:\n{question['question']}\n\nGold Answer:\n{question['answer']}\n\nAgent Answer:\n{record['agent_answer']}\n",
                )
            except ModelOutputError as exc:
                # Judge produced unusable/truncated output: undecidable, not "incorrect"
                # and not a request failure. Keep the raw output and the reason.
                judge_meta = getattr(chat, "last_meta", None)
                record.update(
                    judge_output=exc.output,
                    judge_answer=extract_final(exc.output) if exc.output else "",
                    verdict="Unclear",
                    judge_failure="invalid_output",
                    status="completed",
                    failure_stage="judge",
                    error=type(exc).__name__,
                )
            except ModelRequestError as exc:
                judge_meta = getattr(chat, "last_meta", None)
                record.update(status="request_failure", failure_stage="judge", error=type(exc).__name__)
            except ModelConfigurationError as exc:
                judge_meta = getattr(chat, "last_meta", None)
                record.update(status="configuration_failure", failure_stage="judge", error=type(exc).__name__)
            else:
                judge_meta = getattr(chat, "last_meta", None)
                record.update(
                    judge_output=verdict,
                    judge_answer=extract_final(verdict),
                    verdict=parse_judgment(verdict),
                    status="completed",
                )
        except ModelOutputError as exc:
            record.update(
                status="invalid_output",
                failure_stage=stage,
                error=type(exc).__name__,
                agent_output=exc.output or record.get("agent_output", ""),
            )
        except ModelRequestError as exc:
            record.update(status="request_failure", failure_stage=stage, error=type(exc).__name__)
        except ModelConfigurationError as exc:
            record.update(status="configuration_failure", failure_stage=stage, error=type(exc).__name__)
        except (TimeoutError, OSError) as exc:
            record.update(status="request_failure", failure_stage=stage, error=type(exc).__name__)
        except Exception as exc:
            record.update(status="implementation_failure", failure_stage=stage, error=type(exc).__name__)
        finally:
            qa_meta = qa_meta or getattr(chat, "last_meta", None)
            judge_meta = judge_meta or getattr(chat, "last_meta", None)
            if qa_meta:
                record["qa_meta"] = qa_meta
            if judge_meta:
                record["judge_meta"] = judge_meta
            record["elapsed_seconds"] = time.perf_counter() - start
            save(path, record)
        if snapshot(memory) != checkpoint["snapshot"]:
            raise ValidationError("问答改变了记忆，停止实验")
    return summarize(run, split)


def summarize(run, split):
    rows = [read_json(p) for p in sorted((run / "results" / split).glob("*.json"))]

    def counts(values):
        c = sum(r.get("verdict") == "Correct" for r in values)
        i = sum(r.get("verdict") == "Incorrect" for r in values)
        u = sum(r.get("verdict") == "Unclear" for r in values)
        completed = c + i + u
        by_stage = defaultdict(int)
        for row in values:
            if row.get("failure_stage"):
                by_stage[row["failure_stage"]] += 1
        return {
            "attempted": len(values),
            "correct": c,
            "incorrect": i,
            "unclear": u,
            "completed": completed,
            "request_failure": sum(r.get("status") == "request_failure" for r in values),
            "invalid_output": sum(r.get("status") == "invalid_output" for r in values),
            "configuration_failure": sum(r.get("status") == "configuration_failure" for r in values),
            "implementation_failure": sum(r.get("status") == "implementation_failure" for r in values),
            "interrupted": sum(r.get("status") == "running" for r in values),
            "failure_by_stage": dict(sorted(by_stage.items())),
            "paper_accuracy": c / (c + i) if c + i else None,
            "correct_over_completed": c / completed if completed else None,
            "judge_coverage": completed / len(values) if values else None,
            "paper_denominator": c + i,
            "completed_denominator": completed,
        }

    report = {
        "split": split,
        "overall": counts(rows),
        "by_type": {
            kind: counts([r for r in rows if r["type"] == kind]) for kind in sorted({r["type"] for r in rows})
        },
    }
    save(run / f"summary-{split}.json", report)
    return report


def diagnose_oracle(run, cfg, split, question_id, card_ids, chat, trace):
    """Give human-confirmed evidence to the same QA model; never counted as a score."""
    from .store import Store

    question = next(
        (q for q in selected_questions(run, split) if q["id"] == question_id),
        None,
    )
    if question is None:
        raise ValidationError("诊断题不存在")
    store = Store(str(run / "state.sqlite3"))
    try:
        cards = []
        for card_id in card_ids:
            card = store.get("card", card_id)
            if not card or card.get("deleted"):
                raise ValidationError(f"诊断证据不存在或已删除：{card_id}")
            cards.append(card)
    finally:
        store.close()
    if not cards:
        raise ValidationError("诊断需要至少一条人工确认的证据")
    trace.context = "oracle-" + split + "-" + question_id
    answer = chat("qa", qa_prompt(question, cards))
    result = {
        "scope": "oracle_diagnostic_not_score",
        "id": question_id,
        "split": split,
        "asking_user_id": question["asking_user_id"],
        "card_ids": card_ids,
        "answer": extract_final(answer),
        "raw": answer,
        "qa_meta": getattr(chat, "last_meta", None),
    }
    save(run / "diagnostics" / "oracle" / (split + "__" + question_id + ".json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "ingest", "evaluate", "summarize", "probe", "diagnose"]
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=OFFICIAL / "data/final/Technology/synthetic_domain_channels_rolevariants_Technology.json",
    )
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "protocol.json")
    parser.add_argument("--manifest", type=Path, default=OFFICIAL / "pilot_manifest.json")
    parser.add_argument("--message-limit", type=int)
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id")
    parser.add_argument("--card-ids", default="")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.command == "prepare":
        result = prepare(run, args.corpus, args.protocol, args.manifest, args.message_limit)
    else:
        cfg = verify_run(run)
        trace = Trace(run / "diagnostics")
        if args.command == "summarize":
            result = summarize(run, args.split)
        elif args.command == "probe":
            chat = TextChat(read_json(run / "protocol.json"), run, trace)
            result = {"scope": "connectivity_only_not_benchmark", "status": "running"}
            try:
                answer = chat(
                    "qa",
                    "Asking user: Probe\nQuestion: What is the code?\nRetrieved passages: The code is blue.",
                )
                result["qa"] = extract_final(answer)
                result["qa_meta"] = chat.last_meta
                verdict = chat(
                    "judge",
                    f"Question: What is the code?\nGold Answer: blue\nAgent Answer: {extract_final(answer)}",
                )
                result["judge"] = parse_judgment(verdict)
                result["judge_meta"] = chat.last_meta
                result["status"] = "completed"
            except (ModelRequestError, ModelOutputError, ModelConfigurationError) as exc:
                result.update(status="failed", error=type(exc).__name__, last_meta=chat.last_meta)
            save(run / "probe.json", result)
        elif args.command == "diagnose":
            if not args.question_id or not args.card_ids:
                raise ValidationError("诊断需要 --question-id 与 --card-ids")
            card_ids = [value.strip() for value in args.card_ids.split(",") if value.strip()]
            result = diagnose_oracle(
                run,
                cfg,
                args.split,
                args.question_id,
                card_ids,
                TextChat(read_json(run / "protocol.json"), run, trace),
                trace,
            )
        else:
            clock = ReplayClock(cfg["final_time"])
            memory = build_memory(run, cfg, trace, clock)
            try:
                result = (
                    ingest_run(run, cfg, memory, clock, trace)
                    if args.command == "ingest"
                    else evaluate(
                        run,
                        cfg,
                        memory,
                        clock,
                        trace,
                        TextChat(read_json(run / "protocol.json"), run, trace),
                        args.split,
                        args.limit,
                    )
                )
            finally:
                close_memory(memory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
