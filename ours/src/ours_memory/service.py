from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from threading import RLock

from .backend import Backend, partition, validate_backend
from .calendar_bounds import complete_month_end
from .config import Settings
from .llm import JsonModel
from .models import (
    AccessDenied,
    Caller,
    Event,
    InputTooLarge,
    MemoryChange,
    MigrationRequired,
    OutputTruncated,
    ReadRequest,
    ReadResult,
    StaleRevision,
    ValidationError,
    active,
    canonical,
    interval,
    iso,
    now,
    stable_id,
    timestamp,
    value_key,
)
from .relative_dates import resolve_relative
from .semantic_views import compact_card, compact_event, compact_state
from .state import apply_operations, empty_state, refresh_support, state_id, state_view
from .store import Store


class MemoryService:
    """Trusted in-process service. The team runner authenticates callers; LLMs cannot set identity.

    One serialized writer is the initial deployment mode. Revisions also guard independent processes.
    The CLI is a local research tool, not a public authenticated API server.
    """

    def __init__(
        self,
        settings: Settings,
        backend: Backend,
        model: JsonModel,
        store: Store | None = None,
        *,
        disclosure_authorizer=None,
    ):
        settings.validate()
        if backend is not None:
            validate_backend(backend)
        self.settings, self.backend, self.model = settings, backend, model
        self.store = store or Store(settings.database)
        self.lock = RLock()
        self.clock = now
        self.disclosure_authorizer = disclosure_authorizer

    def _require_backend(self):
        validate_backend(self.backend)

    def _storage_partition(self, data):
        return partition(data)

    def _authorize(self, caller: Caller):
        agents = self.settings.team_agents.get(caller.team_id, [])
        if not all((caller.user_id, caller.team_id, caller.agent_id)) or caller.agent_id not in agents:
            raise AccessDenied("调用者不属于配置中的团队")

    @staticmethod
    def _visible(item: dict, caller: Caller) -> bool:
        return (
            item.get("user_id") == caller.user_id
            and item.get("team_id") == caller.team_id
            and caller.agent_id in item.get("allowed_reader_ids", [])
            and not item.get("deleted")
        )

    def _event(self, event: Event, caller: Caller) -> dict:
        self._authorize(caller)
        if (event.user_id, event.team_id, event.writer_agent_id) != (
            caller.user_id,
            caller.team_id,
            caller.agent_id,
        ):
            raise AccessDenied("写入身份必须来自已验证调用者")
        timestamp(event.recorded_at)
        if timestamp(event.recorded_at) > timestamp(self.clock()) + timedelta(minutes=5):
            raise ValidationError("记录时间不能晚于当前时间；未来安排使用有效时间")
        if event.observed_at and timestamp(event.observed_at) > timestamp(event.recorded_at):
            raise ValidationError("观察时间不能晚于记录时间")
        if not event.event_id or not event.task_id or not event.text.strip():
            raise ValidationError("事件ID、任务ID和内容不能为空")
        if event.speaker not in {"user", "agent", "tool"}:
            raise ValidationError("未知记录类型")
        data = event.to_dict()
        readers = set(data["allowed_reader_ids"])
        if (
            not readers
            or not readers <= set(self.settings.team_agents[caller.team_id])
            or caller.agent_id not in readers
        ):
            raise AccessDenied("读取范围无效，写入者须有权读取其写入")
        for root_id in data["source_root_ids"]:
            if root_id == event.event_id:
                continue
            source = self.store.get("event", root_id)
            if not source or not self._visible(source, caller):
                raise AccessDenied("不能转述未知或无权读取的来源，也不能扩大来源权限")
            if not readers <= set(source["allowed_reader_ids"]):
                # Default remains deny. The trusted publication gate must bind exact
                # content, source IDs, writer and destination; a flag from an LLM is insufficient.
                if self.disclosure_authorizer is None or not self.disclosure_authorizer(data, source):
                    raise AccessDenied("不能扩大来源权限；缺少可信披露授权")
        data["policy"] = self._storage_partition(data)
        if self.store.get("blocked_policy", data["policy"]):
            raise RuntimeError("相关索引遗忘尚未完成，暂不在该范围写入")
        return data

    def _unsettled(self, caller: Caller, scope: str | None = None) -> list[dict]:
        return [
            job
            for job in self.store.all("job")
            if self._visible(job, caller)
            and job["status"] != "done"
            and (scope is None or job.get("scope", "global") in {"global", scope})
        ]

    def _state_unsettled(self, state: dict, caller: Caller, query_scope: str | None = None) -> bool:
        for job in self._unsettled(caller, query_scope or state["scope"]):
            if job["policy"] != state["policy"]:
                continue
            if any(issue["stage"] in {"native", "cards"} for issue in job.get("issues", [])):
                return True
            if not job.get("issues") or any(issue.get("state_id") == state["id"] for issue in job["issues"]):
                return True
        return False

    @staticmethod
    def _issue(stage: str, exc: Exception, **ids) -> dict:
        # Do not persist arbitrary provider messages or private source text.
        detail = str(exc) if isinstance(exc, (ValidationError, StaleRevision)) else "调用失败，需重试"
        issue = {"stage": stage, "code": type(exc).__name__, "detail": detail, **ids}
        if getattr(exc, "validation_code", None):
            issue["validation_code"] = exc.validation_code
        return issue

    def ingest_event(self, event: Event, caller: Caller, card_fields: dict | None = None) -> list[str]:
        """Store one event. ``card_fields`` selects the direct-card path: the caller has
        already produced the card (one model call for the whole message) and the event
        text is that card's exact source quote, so no extractor/enrich call is made.
        """
        with self.lock:
            self._require_backend()
            data = self._event(event, caller)
            event_key = stable_id("event", event.user_id, event.team_id, event.event_id)
            if self.store.get("forgotten", event_key):
                raise AccessDenied("已遗忘事件不可重放")
            previous = self.store.get("event", event.event_id)
            if previous:
                previous = Event.record_with_defaults(previous)
            if previous and canonical(previous) != canonical(data):
                raise ValidationError("事件ID已绑定不同内容或权限")
            self.store.put("event", event.event_id, data)
            job = self.store.get("job", event_key)
            if job and job["status"] == "done":
                return job["card_ids"]
            job = job or {
                **{k: data[k] for k in ("user_id", "team_id", "allowed_reader_ids", "policy", "scope")},
                "id": event_key,
                "event_id": event.event_id,
                "status": "pending",
                "card_ids": [],
                "states_done": [],
            }
            job["status"], job["issues"] = "pending", []
            self.store.put("job", event_key, job)
            stage = "native"
            try:
                if card_fields is not None:
                    stage = "cards"
                    self._store_direct_card(event, data, job, card_fields)
                elif "changes" not in job:
                    job["changes"] = [asdict(change) for change in self.backend.add(event, data["policy"])]
                    self.store.put("job", event_key, job)
                if card_fields is None:
                    changes = [MemoryChange(**change) for change in job["changes"]]
                    for change in changes:
                        if change.event == "DELETE":
                            self._retire_memory(change.memory_id, event.task_id)
                    additions = [change for change in changes if change.event in {"ADD", "UPDATE"}]
                    stage = "cards"
                    saved = {
                        self.store.get("card", key)["memory_id"]
                        for key in job["card_ids"]
                        if self.store.get("card", key)
                    }
                    remaining = [item for item in additions if item.memory_id not in saved]
                    for start in range(0, len(remaining), self.settings.card_batch_size):
                        self._enrich_batch(remaining[start : start + self.settings.card_batch_size], data, job)
                job["cards_saved"] = not any(issue["stage"] == "cards" for issue in job["issues"])
                stage = "state"
                if self.settings.state_enabled:
                    job["issues"].extend(self._update_cards(job["card_ids"], event.task_id, job))
                job["status"] = "partial" if job["issues"] else "done"
                job.pop("error", None)
                self.store.put("job", event_key, job)
                if job["issues"] and not job["card_ids"]:
                    raise ValidationError("本事件卡片均未通过，已隔离；见job-status并可retry-job")
                return job["card_ids"]
            except Exception as exc:
                job["status"], job["error"] = "failed", type(exc).__name__
                if not job["issues"]:
                    job["issues"].append(self._issue(stage, exc))
                self.store.put("job", event_key, job)
                raise

    def _store_direct_card(self, event: Event, data: dict, job: dict, card_fields: dict) -> None:
        """Validate and store one already-extracted card, then index claim + quote.

        The indexed text is exactly what retrieval sees, so the vector store holds the
        evidence wording instead of an extractor paraphrase.
        """
        index_card = getattr(self.backend, "index_card", None)
        if index_card is None:
            raise ValidationError("当前后端不支持直接卡片索引（card_mode=direct）")
        fields = dict(card_fields)
        claim = fields.pop("claim", None) or fields.pop("claim_text", None) or ""
        if not isinstance(claim, str) or not claim.strip():
            raise ValidationError("直接卡片缺少 claim 文本")
        # MemOS TextualMemoryItem ids must be UUID strings, so the native vector id is a
        # deterministic UUID derived from event + claim; the card keeps the same id.
        fields.setdefault(
            "memory_id", str(uuid.uuid5(uuid.NAMESPACE_URL, f"{event.event_id}|{claim.strip()}"))
        )
        card = self._validate_card(fields, data, [MemoryChange(fields["memory_id"], claim.strip())])
        with self.store.transaction():
            self.store.put("card", card["id"], card)
            mapping = self.store.get("memory", card["memory_id"]) or {"card_ids": []}
            mapping.update(
                card_ids=sorted(set(mapping["card_ids"] + [card["id"]])),
                current_card_id=card["id"],
                policy=card["policy"],
            )
            self.store.put("memory", card["memory_id"], mapping)
            job["card_ids"] = sorted(set(job["card_ids"] + [card["id"]]))
            self.store.put("job", job["id"], job)
        indexed = f"{card['claim_text']}\n{card['source_quote']}"
        index_card(event, data["policy"], card["memory_id"], indexed)

    def _enrich_batch(self, batch: list[MemoryChange], data: dict, job: dict, repair=None):
        try:
            payload = {
                "event": data,
                "memories": [asdict(item) for item in batch],
                "state_registry": self.settings.state_keys,
                "allowed_evidence_kinds": self._evidence_kinds(data["speaker"]),
            }
            if repair:
                payload["validation_feedback"] = repair
            answer = self._model("cards", payload)
            items = answer.get("cards")
            if not isinstance(items, list):
                raise ValidationError("卡片输出须为列表")
        except (InputTooLarge, OutputTruncated) as exc:
            if len(batch) > 1:
                mid = len(batch) // 2
                self._enrich_batch(batch[:mid], data, job, repair=repair)
                self._enrich_batch(batch[mid:], data, job, repair=repair)
                return
            job["issues"].append(self._issue("cards", exc, memory_id=batch[0].memory_id))
            return
        except (ValidationError, RuntimeError) as exc:
            job["issues"].append(self._issue("cards", exc))
            return
        retry, feedback = [], []
        for change in batch:
            matches = [
                item for item in items if isinstance(item, dict) and item.get("memory_id") == change.memory_id
            ]
            try:
                if len(matches) != 1:
                    raise ValidationError("模型须为该记忆恰好返回一张卡片")
                card = self._validate_card(matches[0], data, batch)
                with self.store.transaction():
                    self.store.put("card", card["id"], card)
                    mapping = self.store.get("memory", card["memory_id"]) or {"card_ids": []}
                    mapping.update(
                        card_ids=sorted(set(mapping["card_ids"] + [card["id"]])),
                        current_card_id=card["id"],
                        policy=card["policy"],
                    )
                    self.store.put("memory", card["memory_id"], mapping)
                    job["card_ids"] = sorted(set(job["card_ids"] + [card["id"]]))
                    self.store.put("job", job["id"], job)
            except ValidationError as exc:
                if repair is None and self.settings.card_repair_attempts:
                    retry.append(change)
                    feedback.append({"memory_id": change.memory_id, "error": str(exc)})
                else:
                    job["issues"].append(self._issue("cards", exc, memory_id=change.memory_id))
        if retry:
            # Repair metadata only. Never repeat native extraction/writes or alter
            # accepted cards, and keep the failed attempt visible in the job log.
            job.setdefault("card_repairs", []).append({"feedback": feedback, "attempt": 1})
            self.store.put("job", job["id"], job)
            self._enrich_batch(retry, data, job, repair=feedback)

    @staticmethod
    def _evidence_kinds(speaker):
        base = ["agent_inference", "retelling", "plan", "execution_result"]
        return base + (
            ["explicit_user_statement"]
            if speaker == "user"
            else ["tool_observation"]
            if speaker == "tool"
            else []
        )

    def job_status(self, event_id: str, caller: Caller) -> dict:
        self._authorize(caller)
        key = stable_id("event", caller.user_id, caller.team_id, event_id)
        job = self.store.get("job", key)
        if not job or not self._visible(job, caller):
            raise AccessDenied("任务不存在或无权读取")
        return {
            k: job.get(k) for k in ("event_id", "status", "card_ids", "states_done", "issues", "card_repairs")
        }

    def retry_job(self, event_id: str, caller: Caller) -> list[str]:
        with self.lock:
            self.job_status(event_id, caller)
            data = self.store.get("event", event_id)
            if data["writer_agent_id"] != caller.agent_id:
                raise AccessDenied("仅原写入者可重试事件；其他Agent可查询处理状态")
            fields = {key: data[key] for key in Event.__dataclass_fields__ if key in data}
            return self.ingest_event(Event.from_dict(fields), caller)

    def discard_job(self, event_id: str, caller: Caller) -> dict:
        self.job_status(event_id, caller)
        return self.forget([event_id], caller)

    def _model(self, stage: str, payload: dict) -> dict:
        if len(canonical(payload)) > self.settings.max_model_input_chars:
            raise InputTooLarge("单份材料超限；已隔离，须调整材料预算或提交更小的原始事件")
        return self.model.complete(stage, payload)

    def _validate_card(self, fields: dict, event: dict, changes: list[MemoryChange]) -> dict:
        allowed_fields = {
            "memory_id",
            "subject",
            "evidence_kind",
            "state_key",
            "value",
            "unit",
            "scope",
            "conditions",
            "event_time",
            "valid_from",
            "valid_to",
            "planned_for",
            "source_quote",
            "correction_of",
            "state_aspect",
            "grounding_issue",
            "subject_context",
            "subject_basis",
            "subject_analysis",
            "use_hint",
            "state_topic",
            "source",
            "matter",
            "participants",
        }
        if set(fields) - allowed_fields:
            raise ValidationError("模型卡片包含未授权字段")
        change = next((item for item in changes if item.memory_id == fields.get("memory_id")), None)
        quote = (fields.get("source_quote", "") if getattr(self, "focused_schema", False)
                 else self._source_quote(fields.get("source_quote"), event["text"]))
        if not change or quote is None:
            raise ValidationError("卡片来源引用不存在于本轮原文")
        fields = {**fields, "source_quote": quote}
        kind = fields.get("evidence_kind")
        if kind not in {
            "explicit_user_statement",
            "tool_observation",
            "agent_inference",
            "retelling",
            "plan",
            "execution_result",
        }:
            raise ValidationError("未知证据类型")
        if kind == "explicit_user_statement" and event["speaker"] != "user":
            raise ValidationError("非用户来源不能伪装为用户明确表达")
        if kind == "tool_observation" and event["speaker"] != "tool":
            raise ValidationError("非工具来源不能伪装为工具观察")
        scope = fields.get("scope", event["scope"])
        if scope != event["scope"]:
            raise ValidationError("模型不能扩大程序确定的任务范围")
        conditions = fields.get("conditions", [])
        if not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions):
            raise ValidationError("适用条件须为字符串列表")
        key = fields.get("state_key")
        aspect = fields.get("state_aspect")
        if aspect is not None:
            import re

            if not isinstance(aspect, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", aspect):
                raise ValidationError("状态事项标识须为小写英文、数字或下划线")
        if key is not None and (fields.get("subject") != event["user_id"] or key not in self.settings.state_keys):
            # Evidence retention beats state purity.  An unregistered state key (models
            # routinely invent "plan"/"decision") must not discard the whole card: keep
            # the claim as an ordinary card so it stays retrievable, but never let it
            # create or update a state assertion.
            fields = {**fields, "state_key": None, "value": None, "unit": None, "state_key_dropped": key}
            key = None
        if key is not None:
            rule = self.settings.state_keys[key]
            value = fields.get("value")
            if value is not None:
                if rule["type"] == "number" and (
                    isinstance(value, bool) or not isinstance(value, (float, int))
                ):
                    raise ValidationError("状态值须为数值")
                if rule["type"] == "number":
                    import math

                    if not math.isfinite(value):
                        raise ValidationError("状态数值必须有限")
                    if isinstance(value, float) and value.is_integer():
                        fields["value"] = int(value)
                if rule["type"] == "text" and not isinstance(value, str):
                    raise ValidationError("状态值须为文本")
                if rule["type"] == "text":
                    import unicodedata

                    fields["value"] = unicodedata.normalize("NFC", value.strip())
                if rule.get("min") is not None and value < rule["min"]:
                    raise ValidationError("状态值超出允许范围")
            if fields.get("unit") != rule.get("unit"):
                raise ValidationError("状态单位与字段表不一致")
        focused = getattr(self, "focused_schema", False)
        plan_time_issue = None
        if focused:
            # The claim retains the source wording; recording time is operational
            # metadata and must never be used as the message's original time.
            start, end, time_normalization, planned_for = None, None, None, None
            event_time = event.get("observed_at")
        else:
            start, end = fields.get("valid_from"), fields.get("valid_to")
            end, time_normalization = complete_month_end(
                quote, start, end, event.get("observed_at") or event["recorded_at"]
            )
            event_time = fields.get("event_time") or event.get("observed_at")
            if event_time and timestamp(event_time) > timestamp(event["recorded_at"]):
                raise ValidationError("已发生事件的时间不能晚于记录时间；未来安排使用有效时间")
            planned_for = fields.get("planned_for")
            if planned_for is not None:
                try:
                    timestamp(planned_for)
                except ValidationError:
                    raw_plan = planned_for
                    resolved = resolve_relative(quote, event.get("observed_at") or event["recorded_at"])
                    try:
                        day = date.fromisoformat(raw_plan) if isinstance(raw_plan, str) and len(raw_plan) == 10 else None
                    except ValueError:
                        day = None
                    if day and resolved and resolved[:10] == day.isoformat():
                        planned_for = resolved
                    elif day and day.isoformat() in quote:
                        anchor = datetime.fromisoformat((event.get("observed_at") or event["recorded_at"]).replace("Z", "+00:00"))
                        planned_for = datetime.combine(day, time.min, anchor.tzinfo).isoformat()
                    else:
                        planned_for = None
                        plan_time_issue = "unverified_or_invalid_planned_for"
            else:
                planned_for = resolve_relative(quote, event.get("observed_at") or event["recorded_at"])
            if kind == "plan" and start is None and event_time:
                start = event_time
        interval(start, end)
        correction = fields.get("correction_of")
        if correction:
            corrected = self.store.get("card", correction)
            if (
                not corrected
                or corrected["user_id"] != event["user_id"]
                or corrected["policy"] != event["policy"]
            ):
                raise ValidationError("更正目标不存在或读取范围不对应")
            if key and (corrected.get("state_key") != key or corrected["scope"] != scope):
                raise ValidationError("更正目标须属于同一状态字段与范围")
        card = {
            **fields,
            "id": stable_id("card", change.memory_id, event["event_id"], change.text),
            "memory_id": change.memory_id,
            "claim_text": change.text,
            "scope": scope,
            "conditions": sorted(set(conditions)),
            "event_time": iso(event_time) if event_time else None,
            "valid_from": iso(start) if start else None,
            "valid_to": iso(end) if end else None,
            "planned_for": iso(planned_for) if planned_for else None,
            "user_id": event["user_id"],
            "team_id": event["team_id"],
            "task_id": event["task_id"],
            "writer_agent_id": event["writer_agent_id"],
            "recorded_at": event["recorded_at"],
            "source_event_ids": [event["event_id"]],
            "source_root_ids": event["source_root_ids"],
            "allowed_reader_ids": event["allowed_reader_ids"],
            "visibility": "private" if len(event["allowed_reader_ids"]) == 1 else "shared",
            "owner_agent_id": event["allowed_reader_ids"][0]
            if len(event["allowed_reader_ids"]) == 1
            else None,
            "policy": event["policy"],
            "deleted": False,
            "author_id": event.get("author_id") or event["user_id"],
            "category": event.get("category"),
            "item_id": event.get("item_id"),
            "source_message_id": event.get("source_message_id"),
            "metadata": event.get("metadata", {}),
            "state_eligible": event.get("state_eligible", True),
        }
        if focused:
            card.pop("planned_for", None)

        if time_normalization:
            card["time_normalization"] = time_normalization
        if plan_time_issue:
            card["planned_for_issue"] = plan_time_issue
        # Normalize optional fields so downstream logic is never dependent on omitted model keys.
        for name in ("state_key", "value", "unit", "subject"):
            card.setdefault(name, None)
        return card

    @staticmethod
    def _source_quote(quote, text: str) -> str | None:
        """Only normalize presentation, then return the exact ORIGINAL source span."""
        if not isinstance(quote, str) or not quote.strip():
            return None
        if quote in text:
            return quote
        import unicodedata

        def normalize(value):
            chars, positions = [], []
            for index, char in enumerate(value):
                char = {"“": '"', "”": '"', "‘": "'", "’": "'"}.get(char, char)
                for part in unicodedata.normalize("NFD", char):
                    part = " " if part.isspace() else part
                    if part == " " and chars and chars[-1] == " ":
                        positions[-1] = (positions[-1][0], index + 1)
                    else:
                        chars.append(part)
                        positions.append((index, index + 1))
            return "".join(chars), positions

        normalized, positions = normalize(text)
        target, _ = normalize(quote.strip())
        offset = normalized.find(target)
        if offset >= 0:
            return text[positions[offset][0] : positions[offset + len(target) - 1][1]]
        # Recover a case-only presentation error only when the source span is
        # unique. Store the exact source bytes, never the model's altered quote.
        folded, mapping = [], []
        for i, char in enumerate(normalized):
            for part in char.casefold():
                folded.append(part)
                mapping.append(i)
        haystack, needle = "".join(folded), target.casefold()
        start = haystack.find(needle)
        if start < 0 or haystack.find(needle, start + 1) >= 0:
            return None
        left, right = mapping[start], mapping[start + len(needle) - 1]
        return text[positions[left][0] : positions[right][1]]

    def _update_cards(self, card_ids: list[str], task_id: str, job: dict | None = None):
        groups: dict[str, dict] = {}
        for card_id in card_ids:
            card = self.store.get("card", card_id)
            if card and card.get("state_key") and not card.get("deleted"):
                key = state_id(card)
                group = groups.setdefault(
                    key, {"state": self.store.state(key) or empty_state(card), "cards": []}
                )
                group["cards"].append(card)
        if job:
            groups = {key: group for key, group in groups.items() if key not in job["states_done"]}
        # Independent small state batches; no one-LLM-call-per-card policy.
        issues = []
        entries = list(groups.items())
        for start in range(0, len(entries), self.settings.card_batch_size):
            batch = dict(entries[start : start + self.settings.card_batch_size])
            issues.extend(self._state_batch(batch, task_id, job))
        return issues

    def _state_batch(self, batch: dict, task_id: str, job: dict | None, repair_feedback=None):
        issues = []
        try:
            shared = getattr(getattr(self, "control", None), "access_mode", None) == "shared_pool"
            payload = {"groups": ({key: {"state": compact_state(group["state"]),
                                         "cards": [compact_card(card) for card in group["cards"]]}
                                   for key, group in batch.items()} if shared else batch)}
            if repair_feedback:
                payload["validation_feedback"] = repair_feedback
                payload["repair_instruction"] = "Repair invalid operations or return IGNORE to retain evidence without changing the state."
            answer = self._model("state", payload)
        except (InputTooLarge, OutputTruncated) as exc:
            if len(batch) > 1:
                entries = list(batch.items())
                middle = len(entries) // 2
                return self._state_batch(dict(entries[:middle]), task_id, job) + self._state_batch(
                    dict(entries[middle:]), task_id, job
                )
            return [self._issue("state", exc, state_id=next(iter(batch)))]
        except (ValidationError, RuntimeError) as exc:
            return [self._issue("state", exc, state_id=key) for key in batch]
        try:
            decisions = answer.get("decisions")
            if not isinstance(decisions, list):
                raise ValidationError("状态决策须为列表")
        except ValidationError as exc:
            return [self._issue("state", exc, state_id=key) for key in batch]
        for key, group in batch.items():
            try:
                matches = [d for d in decisions if isinstance(d, dict) and d.get("state_id") == key]
                if len(matches) != 1:
                    raise ValidationError("模型必须为每个状态组恰好返回一个决策")
                decision = matches[0]
                group = batch[key]
                previous = group["state"]
                supplied_revision = decision.get("expected_revision")
                if supplied_revision is None:
                    supplied_revision = previous["revision"]
                if type(supplied_revision) is not int or supplied_revision != previous["revision"]:
                    raise StaleRevision(f"模型状态版本不匹配；期望版本{previous['revision']}")
                operations = decision.get("operations")
                if not isinstance(operations, list):
                    raise ValidationError("状态操作须为列表")
                allowed = {card["id"]: card for card in group["cards"]}
                proposed = apply_operations(previous, operations, allowed)
                revision = self.store.commit_state(proposed, previous["revision"], task_id=task_id)
                if revision == previous["revision"]:
                    self.store.mark_dirty(key, task_id)
                if job:
                    job["states_done"].append(key)
                    self.store.put("job", job["id"], job)
            except (ValidationError, StaleRevision) as exc:
                if isinstance(exc, ValidationError) and not isinstance(exc, StaleRevision) and repair_feedback is None:
                    issues.extend(self._state_batch({key: group}, task_id, job, repair_feedback=str(exc)))
                else:
                    issues.append(self._issue("state", exc, state_id=key))
        return issues

    def update_state(self, card_ids: list[str], caller: Caller, task_id: str):
        with self.lock:
            self._authorize(caller)
            if not self.settings.state_enabled:
                return {"status": "disabled"}
            cards = [self.store.get("card", key) for key in card_ids]
            if any(not card or not self._visible(card, caller) for card in cards):
                raise AccessDenied("无权更新这些证据支持的状态")
            issues = self._update_cards(card_ids, task_id)
            return {"status": "partial" if issues else "updated", "issues": issues}

    def read_memory(self, request: ReadRequest, caller: Caller) -> ReadResult:
        with self.lock:
            self._require_backend()
            self._authorize(caller)
            timestamp(request.target_time)
            if request.purpose not in {"current", "history"}:
                raise ValidationError("读取用途须为current或history")
            visible_cards = {
                card["id"]: card
                for card in self.store.all("card")
                if self._visible(card, caller)
                and card["scope"] in {"global", request.scope}
                and set(card["conditions"]) <= set(request.conditions)
            }
            policies = sorted(
                {
                    card["policy"]
                    for card in visible_cards.values()
                    if not self.store.get("blocked_policy", card["policy"])
                }
            )
            # An authorized policy with incomplete erasure must not be searched at all.
            if any(self.store.get("blocked_policy", card["policy"]) for card in visible_cards.values()):
                raise RuntimeError("相关索引清理未完成，暂不读取")
            hits = (
                self.backend.search(request.query, caller, policies, self.settings.candidate_limit)
                if policies
                else []
            )
            candidates = {}
            for hit in hits:
                mapping = self.store.get("memory", hit.memory_id)
                if mapping:
                    card = visible_cards.get(mapping["current_card_id"])
                    if card:
                        candidates[card["id"]] = card
            states = []
            if self.settings.state_enabled:
                for state in self.store.states():
                    if (
                        self._visible(state, caller)
                        and state["scope"] in {"global", request.scope}
                        and set(state["conditions"]) <= set(request.conditions)
                    ):
                        view = state_view(state, request.target_time)
                        if self._state_unsettled(state, caller, request.scope):
                            view.update(status="pending", value=None, processing_issue=True)
                        if view["assertions"]:
                            states.append(view)
                            for assertion in view["assertions"]:
                                for key in assertion["evidence_card_ids"]:
                                    if key in visible_cards:
                                        candidates[key] = visible_cards[key]
            unresolved = self._unsettled(caller, request.scope)
            if not candidates and not states:
                result = ReadResult(
                    issues=["有相关事件尚未处理成功，不能据此确认用户状态"] if unresolved else []
                )
                self._save_read(request, caller, result)
                return result
            allowed_states = {state["id"]: state for state in states}
            if self.settings.routing_enabled:
                decision = self._model(
                    "route",
                    {
                        "task": request.to_dict(),
                        "caller": asdict(caller),
                        "states": states,
                        "cards": list(candidates.values()),
                    },
                )
            else:
                decision = {
                    "state_ids": list(allowed_states),
                    "cards": [{"card_id": key, "use": "current"} for key in candidates],
                    "missing": [],
                }
            if not isinstance(decision.get("state_ids"), list) or not isinstance(decision.get("cards"), list):
                raise ValidationError("路由输出格式错误")
            if any(key not in allowed_states for key in decision["state_ids"]):
                raise ValidationError("路由创建了不存在或无权读取的状态")
            result = ReadResult()
            missing = decision.get("missing", [])
            if not isinstance(missing, list) or not all(
                isinstance(item, str) and len(item) <= 1000 for item in missing
            ):
                raise ValidationError("缺失证据须为简短文本列表")
            result.missing = missing
            if unresolved:
                result.issues.append("有相关事件尚未处理成功，受影响状态暂不作为确定依据；可查询job-status")
            for key in dict.fromkeys(decision["state_ids"]):
                state = allowed_states[key]
                (result.states if self._view_ready(state) else result.uncertain).append(state)
                result.revisions[key] = state["revision"]
            seen = set()
            for selected in decision["cards"]:
                key, usage = selected.get("card_id"), selected.get("use")
                if key not in candidates or usage not in {"current", "history", "conflict"}:
                    raise ValidationError("路由引用了未提供的卡片或未知用途")
                if key in seen:
                    continue
                seen.add(key)
                card = candidates[key]
                if card["scope"] not in {"global", request.scope} or not set(card["conditions"]) <= set(
                    request.conditions
                ):
                    result.issues.append("已拦截范围或条件不满足的卡片")
                    continue
                if usage == "history" and request.purpose != "history":
                    result.issues.append("历史卡片未作为当前依据返回")
                    continue
                if usage == "current" and not self._currently_usable(card, request, caller):
                    result.issues.append("已拦截失效、冲突或待确认卡片")
                    continue
                result.cards.append({**card, "use": usage})
            # Account for implicit dependencies in selected state cards as well as explicit state IDs.
            for card in result.cards:
                if card.get("state_key"):
                    state = self.store.state(state_id(card))
                    if state:
                        result.revisions[state["id"]] = state["revision"]
            if len(canonical(result.to_dict())) > self.settings.max_context_chars:
                raise ValidationError("返回资料超过预算，未静默截断关键证据")
            self._save_read(request, caller, result)
            return result

    def _save_read(self, request: ReadRequest, caller: Caller, result: ReadResult):
        read_id = stable_id("read", caller.user_id, caller.team_id, caller.agent_id, request.task_id)
        old = self.store.get("read", read_id) or {}
        context = stable_id("context", iso(request.target_time), request.scope, sorted(request.conditions))
        # Re-reading replaces the snapshot for this context; other task contexts stay independent.
        basis = [item for item in old.get("basis", []) if item.get("context") != context]
        returned_views = {view["id"]: view for view in result.states + result.uncertain}
        for key, revision in result.revisions.items():
            state = self.store.state(key)
            if state:
                view = returned_views.get(key)
                if view is None:
                    view = state_view(state, request.target_time)
                    if self._state_unsettled(state, caller, request.scope):
                        view.update(status="pending", value=None, processing_issue=True)
                basis.append(
                    {
                        "kind": "state",
                        "context": context,
                        "state_id": key,
                        "revision": revision,
                        "target_time": request.target_time,
                        "scope": request.scope,
                        "requires_reread": not self._view_ready(view),
                        "value": self._fingerprint(view),
                    }
                )
        for card in result.cards:
            basis.append(
                {
                    "kind": "card",
                    "context": context,
                    "card_id": card["id"],
                    "use": card["use"],
                    "target_time": request.target_time,
                    "value": self._card_fingerprint(card),
                }
            )
        unsettled = self._unsettled(caller, request.scope)
        if unsettled:
            readers = set(self.settings.team_agents[caller.team_id])
            for job in unsettled:
                readers &= set(job["allowed_reader_ids"])
            basis.append(
                {
                    "kind": "read_guard",
                    "context": context,
                    "target_time": request.target_time,
                    "allowed_reader_ids": sorted(readers),
                    "requires_reread": True,
                }
            )
        self.store.put(
            "read",
            read_id,
            {**asdict(caller), "task_id": request.task_id, "scope": request.scope, "basis": basis},
        )

    @staticmethod
    def _card_fingerprint(card: dict) -> str:
        return stable_id("card_basis", {key: value for key, value in card.items() if key != "use"})

    def _currently_usable(self, card: dict, request: ReadRequest, caller: Caller) -> bool:
        if not active(card, request.target_time):
            return False
        if not card.get("valid_from") and timestamp(request.target_time) < timestamp(
            card.get("event_time") or card["recorded_at"]
        ):
            return False
        if not card.get("state_key") or not self.settings.state_enabled:
            return True
        state = self.store.state(state_id(card))
        if not state:
            return False
        if self._state_unsettled(state, caller, request.scope):
            return False
        view = state_view(state, request.target_time)
        return self._view_ready(view) and any(
            card["id"] in item["evidence_card_ids"] and item["status"] == "confirmed"
            for item in view["assertions"]
        )

    def finish_task(self, task_id: str, caller: Caller) -> dict:
        with self.lock:
            self._authorize(caller)
            # Completion covers only what this authorized caller could review.
            key = stable_id("completion", caller.user_id, caller.team_id, caller.agent_id, task_id)
            previous = self.store.get("completion", key)
            dirty = any(
                self._visible(self.store.state(item["state_id"]) or {}, caller)
                for item in self.store.dirty_states()
            )
            if (
                self.settings.sleep_enabled
                and previous
                and previous.get("result") == "completed"
                and not dirty
                and not self._unsettled(caller)
            ):
                return {"status": "already_completed"}
            result = self.review_changed_state(caller, boundary_task_id=task_id)
            if result["status"] == "completed":
                self.store.put(
                    "completion", key, {**asdict(caller), "task_id": task_id, "result": "completed"}
                )
            return result

    def review_changed_state(self, caller: Caller, boundary_task_id: str) -> dict:
        """Master gate applies to automatic AND manual entry points; no force-bypass parameter."""
        with self.lock:
            self._authorize(caller)
            self.settings.validate()
            if not self.settings.sleep_enabled:
                return {"status": "disabled", "reviewed": 0, "changed": 0}
            reviewed = changed = 0
            issues = [
                {"code": "UnsettledWrite", "event_id": job["event_id"], "job_status": job["status"]}
                for job in self._unsettled(caller)
            ]
            for item in self.store.dirty_states():
                state = self.store.state(item["state_id"])
                if not state or not self._visible(state, caller):
                    continue
                if self._state_unsettled(state, caller):
                    issues.append({"state_id": state["id"], "code": "UnsettledWrite"})
                    continue
                failed = self.store.get("sleep_issue", state["id"])
                if failed and failed["generation"] == item["generation"]:
                    issues.append({"state_id": state["id"], "code": "QuarantinedSleep"})
                    continue
                # One key/scope/reader policy per bundle. No cross-permission evidence mixing.
                cards = {
                    card["id"]: card
                    for card in self.store.all("card")
                    if self._visible(card, caller) and card.get("state_key") and state_id(card) == state["id"]
                }
                if not cards:
                    self.store.clean_dirty(state["id"], item["generation"])
                    continue
                # Independent confirmed non-overlapping assertions with no missing evidence need no LLM.
                boundaries = {self.clock()} | {
                    a[key] for a in state["assertions"] for key in ("valid_from", "valid_to") if a.get(key)
                }
                views = [state_view(state, point) for point in boundaries]
                evidence_ids = {
                    key for assertion in state["assertions"] for key in assertion["evidence_card_ids"]
                }
                all_accounted = all(
                    card["id"] in evidence_ids
                    or any(
                        set(card["source_root_ids"]) == set(a["source_root_ids"])
                        and value_key(card["value"]) == value_key(a["value"])
                        for a in state["assertions"]
                    )
                    for card in cards.values()
                )
                needs_semantics = (
                    not all_accounted
                    or any(view["status"] in {"conflict", "pending"} or view["has_pending"] for view in views)
                    or any(a["status"] in {"pending", "conflict"} for a in state["assertions"])
                )
                if needs_semantics:
                    events = [
                        event
                        for event in self.store.all("event")
                        if self._visible(event, caller)
                        and event["event_id"]
                        in {e for card in cards.values() for e in card["source_event_ids"]}
                    ]
                    payload = {
                        "state": state,
                        "history": self.store.history(state["id"])[-4:],
                        # Sleep review only needs recent, relevant evidence for this state.
                        "cards": [
                            {
                                **card,
                                "claim_text": (card.get("claim_text") or "")[:1000],
                                "source_quote": (card.get("source_quote") or "")[:1000],
                            }
                            for card in sorted(cards.values(), key=lambda c: c.get("event_time") or "")[-8:]
                        ],
                        "events": [
                            {**event, "text": (event.get("text") or "")[:1000]}
                            for event in events[-8:]
                        ],
                        "boundary_task_id": boundary_task_id,
                    }
                    if getattr(getattr(self, "control", None), "access_mode", None) == "shared_pool":
                        payload["state"] = compact_state(payload["state"])
                        payload["history"] = [compact_state(state) for state in payload["history"]]
                        payload["cards"] = [compact_card(card) for card in payload["cards"]]
                        payload["events"] = [compact_event(event) for event in payload["events"]]
                    try:
                        for attempt in range(2):
                            proposal = self._model("sleep", payload)
                            supplied_revision = proposal.get("expected_revision")
                            if supplied_revision is None:
                                supplied_revision = state["revision"]
                            if (
                                proposal.get("state_id") != state["id"]
                                or type(supplied_revision) is not int
                                or supplied_revision != state["revision"]
                            ):
                                raise StaleRevision(f"Sleep版本不匹配；期望版本{state['revision']}")
                            try:
                                operations = proposal.get("operations")
                                if not isinstance(operations, list):
                                    raise ValidationError("Sleep操作须为列表")
                                next_state = apply_operations(state, operations, cards)
                                break
                            except ValidationError as exc:
                                if attempt:
                                    raise
                                payload = {**payload, "validation_feedback": str(exc),
                                           "repair_instruction": "Repair the invalid operations or return an empty list to preserve uncertainty."}
                        revision = self.store.commit_state(
                            next_state, state["revision"], task_id=boundary_task_id, dirty=False
                        )
                        changed += int(revision != state["revision"])
                    except (ValidationError, StaleRevision, RuntimeError) as exc:
                        issue = self._issue("sleep", exc, state_id=state["id"])
                        self.store.put(
                            "sleep_issue",
                            state["id"],
                            {
                                **issue,
                                "generation": item["generation"],
                                **{k: state[k] for k in ("user_id", "team_id", "allowed_reader_ids")},
                            },
                        )
                        issues.append(issue)
                        continue
                reviewed += 1
                self.store.clean_dirty(state["id"], item["generation"])
                self.store.remove("sleep_issue", state["id"])
            return {
                "status": "needs_attention" if issues else "completed",
                "reviewed": reviewed,
                "changed": changed,
                "issues": issues,
            }

    def retry_sleep(self, caller: Caller, task_id: str) -> dict:
        with self.lock:
            self._authorize(caller)
            if not self.settings.sleep_enabled:
                return {"status": "disabled", "reviewed": 0, "changed": 0}
            for item in self.store.all("sleep_issue"):
                if self._visible(item, caller):
                    self.store.remove("sleep_issue", item["state_id"])
            return self.review_changed_state(caller, task_id)

    def register_plan(self, plan_id: str, task_id: str, caller: Caller) -> dict:
        with self.lock:
            self._authorize(caller)
            reads = [
                item
                for item in self.store.all("read")
                if item["user_id"] == caller.user_id
                and item["team_id"] == caller.team_id
                and item["agent_id"] == caller.agent_id
                and item["task_id"] == task_id
            ]
            dependencies = {}
            readers = set(self.settings.team_agents[caller.team_id])
            for read in reads:
                for basis in read.get("basis", []):
                    if basis.get("kind") == "read_guard":
                        readers &= set(basis["allowed_reader_ids"])
                        dependencies[stable_id("guard", basis["context"])] = basis
                        continue
                    key = basis.get("state_id") or basis["card_id"]
                    source = (
                        self.store.state(key)
                        if basis.get("kind", "state") == "state"
                        else self.store.get("card", key)
                    )
                    if source and self._visible(source, caller):
                        readers &= set(source["allowed_reader_ids"])
                    else:
                        # A missing source must not widen a previously private plan.
                        readers &= {caller.agent_id}
                    dependencies[
                        stable_id(
                            "dep", basis.get("kind", "state"), key, basis["target_time"], basis.get("context")
                        )
                    ] = basis
            key = stable_id("plan", caller.user_id, caller.team_id, plan_id)
            previous = self.store.get("plan", key)
            if previous and (previous["agent_id"] != caller.agent_id or previous["task_id"] != task_id):
                raise AccessDenied("方案ID已由其他创建者或任务使用，不能覆盖")
            plan = {
                **asdict(caller),
                "plan_id": plan_id,
                "task_id": task_id,
                "dependencies": dependencies,
                "allowed_reader_ids": sorted(readers),
            }
            self.store.put("plan", key, plan)
            return {"plan_id": plan_id, "dependency_count": len(dependencies)}

    @staticmethod
    def _fingerprint(view: dict) -> str:
        # No plaintext user state is copied into plan/read logs.
        return stable_id(
            "basis",
            view["status"],
            value_key(view["value"]),
            view["unit"],
            view["conditions"],
            bool(view.get("has_pending")),
            bool(view.get("processing_issue")),
        )

    @staticmethod
    def _view_ready(view: dict) -> bool:
        return (
            view["status"] == "confirmed" and not view.get("has_pending") and not view.get("processing_issue")
        )

    def check_plan_dependencies(self, plan_id: str, caller: Caller) -> dict:
        with self.lock:
            self._authorize(caller)
            plan = self.store.get("plan", stable_id("plan", caller.user_id, caller.team_id, plan_id))
            if not plan:
                raise ValidationError("方案不存在")
            if caller.agent_id not in plan["allowed_reader_ids"]:
                raise AccessDenied("当前Agent无权执行依赖这些资料的方案")
            stale = []
            for dependency in plan["dependencies"].values():
                if dependency.get("kind") == "read_guard":
                    stale.append(dependency["context"])
                    continue
                if dependency.get("kind") == "card":
                    key = dependency["card_id"]
                    card = self.store.get("card", key)
                    if (
                        not card
                        or not self._visible(card, caller)
                        or self._card_fingerprint(card) != dependency["value"]
                        or (dependency["use"] == "current" and not active(card, dependency["target_time"]))
                    ):
                        stale.append(key)
                    continue
                key = dependency["state_id"]
                state = self.store.state(key)
                view = state_view(state, dependency["target_time"]) if state else None
                if (
                    not state
                    or not self._visible(state, caller)
                    or not self._view_ready(view)
                    or dependency.get("requires_reread")
                    or self._state_unsettled(state, caller, dependency.get("scope"))
                    or self._fingerprint(view) != dependency["value"]
                ):
                    stale.append(key)
            return {"ready": not stale, "requires_recheck": stale}

    def _retire_memory(self, memory_id: str, task_id: str):
        mapping = self.store.get("memory", memory_id)
        if not mapping:
            return
        removed = set(mapping["card_ids"])
        with self.store.transaction():
            affected = {
                state["id"]
                for state in self.store.states()
                if any(set(item["evidence_card_ids"]) & removed for item in state["assertions"])
            }
            import json

            for row in self.store.connection.execute("SELECT id,data FROM state_history"):
                if any(
                    set(item["evidence_card_ids"]) & removed for item in json.loads(row["data"])["assertions"]
                ):
                    affected.add(row["id"])
            for key in removed:
                card = self.store.get("card", key)
                if card:
                    card["deleted"] = True
                    self.store.put("card", key, card)
            self._remove_support(removed, task_id)
            for key in affected:
                self.store.connection.execute("DELETE FROM state_history WHERE id=?", (key,))
                self.store.remove("sleep_issue", key)
            self.store.remove("memory", memory_id)

    def _remove_support(self, removed: set[str], task_id: str):
        cards = {card["id"]: card for card in self.store.all("card") if card["id"] not in removed}
        for state in self.store.states():
            next_state = deepcopy(state)
            changed = False
            for assertion in next_state["assertions"]:
                old = set(assertion["evidence_card_ids"])
                if old & removed:
                    changed = True
                    assertion["evidence_card_ids"] = sorted(old - removed)
                    replacement = refresh_support(assertion, cards)
                    assertion.clear()
                    assertion.update(replacement or {"evidence_card_ids": []})
            if changed:
                next_state["assertions"] = [
                    item for item in next_state["assertions"] if item["evidence_card_ids"]
                ]
                self.store.commit_state(next_state, state["revision"], task_id=task_id)

    def forget(self, source_ids: list[str], caller: Caller) -> dict:
        """Atomically erase sidecar evidence and persist resumable native-erasure work."""
        with self.lock:
            self._authorize(caller)
            self._require_backend()
            if not source_ids or not all(isinstance(key, str) and key for key in source_ids):
                raise ValidationError("遗忘来源须为非空ID列表")
            requested = set(source_ids)
            events = self.store.all("event")
            known = {event["event_id"]: event for event in events}
            for source in requested:
                event = known.get(source)
                marker = self.store.get(
                    "forgotten", stable_id("event", caller.user_id, caller.team_id, source)
                )
                if event and self._visible(event, caller):
                    continue
                if marker and not all(key in marker for key in ("user_id", "team_id", "allowed_reader_ids")):
                    raise MigrationRequired(
                        "旧库遗忘标记缺少权限元数据；须用原始权限资料运行migrate_legacy_forget.py，不能自动猜测权限"
                    )
                if marker and self._visible(marker, caller):
                    continue
                raise AccessDenied("遗忘目标不存在或无权管理")
            sources = requested & set(known)
            while True:
                expanded = sources | {
                    event["event_id"]
                    for event in events
                    if event["user_id"] == caller.user_id
                    and event["team_id"] == caller.team_id
                    and set(event["source_root_ids"]) & sources
                }
                if expanded == sources:
                    break
                sources = expanded
            selected = [event for event in events if event["event_id"] in sources]
            cards = [
                card
                for card in self.store.all("card")
                if card["user_id"] == caller.user_id
                and card["team_id"] == caller.team_id
                and set(card["source_event_ids"] + card["source_root_ids"]) & sources
            ]
            removed = {card["id"] for card in cards}
            affected = {state_id(card) for card in cards if card.get("state_key")}
            # A forgotten card may survive ONLY in historical versions after a replacement.
            import json

            for row in self.store.connection.execute("SELECT id,data FROM state_history"):
                history = json.loads(row["data"])
                if history["user_id"] != caller.user_id or history["team_id"] != caller.team_id:
                    continue
                if any(
                    set(item["evidence_card_ids"]) & removed or set(item["source_root_ids"]) & sources
                    for item in history["assertions"]
                ):
                    affected.add(row["id"])
            with self.store.transaction():
                # Publication approvals contain exact event plaintext too.
                for grant in self.store.all("disclosure"):
                    approved = grant.get("event", {})
                    if (
                        approved.get("event_id") in sources
                        or set(approved.get("source_root_ids", [])) & sources
                    ):
                        self.store.remove("disclosure", approved["disclosure_id"])
                for event in selected:
                    policy = event["policy"]
                    blocked = self.store.get("blocked_policy", policy) or {"memory_ids": [], "event_ids": []}
                    key = stable_id("event", event["user_id"], event["team_id"], event["event_id"])
                    job = self.store.get("job", key) or {}
                    ids = {item["memory_id"] for item in job.get("changes", [])}
                    ids |= {card["memory_id"] for card in cards if card["policy"] == policy}
                    self.store.put(
                        "blocked_policy",
                        policy,
                        {
                            **{k: event[k] for k in ("user_id", "team_id", "allowed_reader_ids", "policy")},
                            "memory_ids": sorted(ids | set(blocked["memory_ids"])),
                            "event_ids": sorted(set(blocked.get("event_ids", [])) | {event["event_id"]}),
                            # Source-erasure authority does NOT grant access to private derivative contents.
                            "cleanup_agent_ids": sorted(
                                set(blocked.get("cleanup_agent_ids", [])) | {caller.agent_id}
                            ),
                            "cache_pending": True,
                        },
                    )
                    # The marker, erasure queue and removal are one SQLite transaction.
                    self.store.put(
                        "forgotten",
                        key,
                        {
                            "at": self.clock(),
                            **{k: event[k] for k in ("user_id", "team_id", "allowed_reader_ids", "policy")},
                        },
                    )
                self._remove_support(removed, "forget")
                for key in affected:
                    self.store.connection.execute("DELETE FROM state_history WHERE id=?", (key,))
                    self.store.remove("sleep_issue", key)
                for card in cards:
                    self.store.remove("card", card["id"])
                    self.store.remove("memory", card["memory_id"])
                for event in selected:
                    self.store.remove("event", event["event_id"])
                    self.store.remove(
                        "job", stable_id("event", event["user_id"], event["team_id"], event["event_id"])
                    )
            maintenance = self.retry_index_deletes(caller)
            checkpoint_pending = False
            try:
                result = self.store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                checkpoint_pending = bool(result and result[0])
            except Exception as exc:
                import sqlite3

                if not isinstance(exc, sqlite3.Error):
                    raise
                checkpoint_pending = True
            status = (
                "pending_cleanup"
                if maintenance["pending_index_deletes"]
                else ("forgotten" if sources else "already_forgotten")
            )
            return {
                "status": status,
                "forgotten_events": len(sources),
                "forgotten_cards": len(removed),
                **maintenance,
                "checkpoint_pending": checkpoint_pending,
            }

    def retry_index_deletes(self, caller: Caller) -> dict:
        with self.lock:
            self._authorize(caller)
            self._require_backend()
            pending = 0
            rows = self.store.connection.execute(
                "SELECT id,data FROM objects WHERE kind='blocked_policy'"
            ).fetchall()
            import json

            for row in rows:
                blocked = json.loads(row["data"])
                if not self._can_cleanup(blocked, caller):
                    continue
                remaining = set(blocked["memory_ids"])
                undiscovered = []
                for event_id in blocked.get("event_ids", []):
                    try:
                        remaining.update(
                            self.backend.find_event_memories(event_id, row["id"], caller.user_id)
                        )
                    except Exception:
                        undiscovered.append(event_id)
                # Persist discovered IDs BEFORE any native deletion, allowing recovery after a crash.
                blocked.update(memory_ids=sorted(remaining), event_ids=undiscovered)
                self.store.put("blocked_policy", row["id"], blocked)
                for memory_id in list(remaining):
                    try:
                        self._erase_native(memory_id, row["id"], caller.user_id)
                    except Exception:
                        continue
                    remaining.discard(memory_id)
                cache_pending = False
                if blocked.get("cache_pending"):
                    try:
                        self._clear_native_cache(row["id"], caller.user_id)
                    except Exception:
                        cache_pending = True
                if remaining or cache_pending or undiscovered:
                    self.store.put(
                        "blocked_policy",
                        row["id"],
                        {**blocked, "memory_ids": sorted(remaining), "cache_pending": cache_pending},
                    )
                    pending += len(remaining) + int(cache_pending) + len(undiscovered)
                else:
                    self.store.remove("blocked_policy", row["id"])
            return {"pending_index_deletes": pending}

    @classmethod
    def _can_cleanup(cls, item: dict, caller: Caller) -> bool:
        return cls._visible(item, caller) or (
            item.get("user_id") == caller.user_id
            and item.get("team_id") == caller.team_id
            and caller.agent_id in item.get("cleanup_agent_ids", [])
        )

    def _erase_native(self, memory_id: str, policy: str, user_id: str):
        self.backend.forget(memory_id, policy, user_id)

    def _clear_native_cache(self, policy: str, user_id: str):
        self.backend.clear_event_cache(policy, user_id)
