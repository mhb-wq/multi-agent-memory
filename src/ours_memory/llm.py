from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from typing import Protocol

from . import prompts
from .config import Settings
from .models import InputTooLarge, ModelOutputError, OutputTruncated, ValidationError, canonical
from .store import Store


class JsonModel(Protocol):
    def complete(self, stage: str, payload: dict) -> dict: ...


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CompatibleJsonModel:
    """DeepSeek by config; changing the base model does not change memory logic."""

    def __init__(self, settings: Settings, store: Store):
        self.settings, self.store = settings, store

    def complete(self, stage: str, payload: dict) -> dict:
        settings = self.settings
        text = canonical(payload)
        input_limit = settings.answer_max_input_chars if stage == "answer" else settings.max_model_input_chars
        if len(text) > input_limit:
            raise InputTooLarge("材料超过上限；请拆批，不截断冲突证据")
        key = os.environ.get(settings.api_key_env)
        if not key:
            raise ValidationError(f"未配置环境变量{settings.api_key_env}；未发出网络请求")
        instructions = {
            "cards": prompts.ENRICH,
            "cards_direct": prompts.CARDS_DIRECT,
            "focused_cards": prompts.FOCUSED_CARDS,
            "focused_state": prompts.FOCUSED_STATE,
            "question_subjects": prompts.QUESTION_SUBJECTS,
            "intake": prompts.INTAKE,
            "select": prompts.SELECT,
            "answer": prompts.ANSWER,
            "output_check": prompts.OUTPUT_CHECK,
            "redaction_plan": prompts.REDACTION_PLAN,
            "state": prompts.UPDATE,
            "route": prompts.ROUTE,
            "sleep": prompts.SLEEP,
        }[stage]
        options = dict(settings.request_options)
        if any(k in options for k in ("model", "messages", "response_format", "stream")):
            raise ValidationError("request_options不能覆盖模型身份、消息或JSON输出约定")
        body = {
            "model": settings.model,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": (settings.state_max_tokens if stage == "focused_state" else
                           settings.answer_max_tokens if stage == "answer" else settings.max_tokens),
            "response_format": {"type": "json_object"},
            "stream": False,
            **options,
        }
        request = urllib.request.Request(
            settings.base_url.rstrip("/") + "/chat/completions",
            data=canonical(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        for attempt in range(settings.retries + 1):
            started = time.perf_counter()
            usage = {}
            model_output = None
            try:
                with urllib.request.build_opener(RejectRedirects()).open(
                    request, timeout=settings.timeout_seconds
                ) as response:
                    raw_response = response.read().decode("utf-8")
                    if stage == "focused_state":
                        model_output = raw_response
                    try:
                        result = json.loads(raw_response)
                    except ValueError:
                        raise ModelOutputError("接口响应不是可解析的JSON") from None
                if not isinstance(result, dict):
                    raise ModelOutputError("接口响应须为JSON对象")
                usage = result.get("usage", {})
                choice = result["choices"][0]
                model_output = (choice.get("message") or {}).get("content")
                if choice.get("finish_reason") == "length":
                    raise OutputTruncated("模型输出被截断；须缩小材料批次")
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ModelOutputError("模型content须为JSON文本，不能为null")
                content = content.strip()
                if content.startswith("```") and content.endswith("```"):
                    content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    data = json.loads(content)
                except (ValueError, TypeError):
                    raise ModelOutputError("模型未返回可解析的JSON") from None
                if not isinstance(data, dict):
                    raise ModelOutputError("模型必须输出JSON对象")
                self._log(stage, started, attempt, usage, True, served_model=result.get("model"))
                return data
            except ModelOutputError as exc:
                if stage == "focused_state":
                    exc.model_output = model_output
                self._log(stage, started, attempt, usage, False, type(exc).__name__)
                raise
            except (KeyError, IndexError, TypeError) as exc:
                self._log(stage, started, attempt, usage, False, type(exc).__name__)
                error = ModelOutputError("模型响应缺少必要字段或字段类型错误")
                if stage == "focused_state":
                    error.model_output = model_output
                raise error from None
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                # Response bodies may echo private prompts or secrets: persist only exception class/status.
                self._log(
                    stage, started, attempt, usage, False, type(exc).__name__, getattr(exc, "code", None)
                )
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"模型请求失败，HTTP {exc.code}") from None
                if attempt == settings.retries:
                    raise RuntimeError(f"{stage}模型调用失败；未提交结果，详见调用日志") from None
                time.sleep(min(2**attempt, 4))
        raise AssertionError("unreachable")

    def _log(self, stage, started, attempt, usage, success, error=None, status=None, served_model=None):
        estimated = None
        s = self.settings
        if usage and s.input_price_per_million is not None and s.output_price_per_million is not None:
            estimated = (
                usage.get("prompt_tokens", 0) * s.input_price_per_million
                + usage.get("completion_tokens", 0) * s.output_price_per_million
            ) / 1_000_000
        self.store.call(
            stage,
            model=s.model,
            served_model=served_model,
            attempt=attempt,
            success=success,
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
            http_status=status,
            estimated_uncached_cost=estimated,
            billing="estimate_only_not_provider_invoice",
        )


class ScriptedModel:
    """Explicit test fixture, never used as the research model or a claimed LLM result."""

    def __init__(self, responses: dict[str, list[dict]] | None = None):
        self.responses = defaultdict(deque)
        self.calls: list[tuple[str, dict]] = []
        for stage, values in (responses or {}).items():
            self.responses[stage].extend(values)

    def complete(self, stage: str, payload: dict) -> dict:
        self.calls.append((stage, json.loads(canonical(payload))))
        if not self.responses[stage]:
            raise AssertionError(f"测试未提供{stage}响应")
        result = self.responses[stage].popleft()
        if callable(result):
            return result(payload)
        return result
