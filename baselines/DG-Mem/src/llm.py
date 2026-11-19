"""Provider-neutral chat model adapters.

The adapters use HTTP directly so an experiment does not need four vendor
SDKs installed at once. New providers can be added through
``register_chat_provider`` without changing callers.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import quote

from .config import LLMSettings


Message = dict[str, str]


@dataclass(frozen=True)
class ChatCompletion:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EmptyCompletionError(RuntimeError):
    def __init__(self, metadata: dict[str, Any], *, retryable: bool):
        self.metadata = metadata
        self.retryable = retryable
        super().__init__(f"Model returned empty content: {metadata}")


class ChatModel(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        ...


class HTTPChatModel:
    def __init__(self, settings: LLMSettings, http_client: Any = None):
        self.settings = settings
        self._client = http_client
        self._client_lock = threading.Lock()

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def parse_response(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.complete_with_metadata(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ).text

    def complete_with_metadata(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion:
        client = self._client
        if client is None:
            with self._client_lock:
                client = self._client
                if client is None:
                    try:
                        import httpx
                    except Exception as exc:  # pragma: no cover - depends on local install
                        raise RuntimeError("httpx is required for LLM calls") from exc
                    client = httpx.Client(timeout=self.settings.timeout_seconds, trust_env=False)
                    self._client = client
        url, headers, body = self.build_request(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        last_error: Exception | None = None
        attempts = max(1, int(self.settings.max_retries) + 1)
        for attempt in range(attempts):
            try:
                response = client.post(url, headers=headers, json=body, timeout=self.settings.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                completion = ChatCompletion(
                    text=self.parse_response(payload).strip(),
                    metadata=self.response_metadata(payload),
                )
                if not completion.text:
                    finish_reason = str(completion.metadata.get("finish_reason", "")).lower()
                    raise EmptyCompletionError(
                        completion.metadata,
                        retryable=finish_reason not in {"length", "max_tokens", "max_output_tokens"},
                    )
                return completion
            except Exception as exc:
                last_error = exc
                if (
                    attempt >= attempts - 1
                    or isinstance(exc, EmptyCompletionError) and not exc.retryable
                ):
                    break
                delay = max(0.0, self.settings.retry_initial_seconds) * (2**attempt)
                if delay:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def response_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in ("id", "model", "modelVersion", "created", "system_fingerprint", "service_tier"):
            value = payload.get(key)
            if value is not None:
                metadata[key] = value

        usage = payload.get("usage", payload.get("usageMetadata"))
        if isinstance(usage, dict):
            metadata["usage"] = usage

        finish_reason = payload.get("stop_reason")
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            finish_reason = choice.get("finish_reason", finish_reason)
            message = choice.get("message")
            if isinstance(message, dict):
                refusal = message.get("refusal")
                if refusal:
                    metadata["refusal"] = str(refusal)
                reasoning_content = message.get("reasoning_content")
                if reasoning_content:
                    metadata["reasoning_content_chars"] = len(str(reasoning_content))

        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            finish_reason = candidates[0].get("finishReason", finish_reason)
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        return metadata

    def _limits(self, max_tokens: int | None, temperature: float | None) -> tuple[int, float]:
        return (
            int(max_tokens if max_tokens is not None else self.settings.max_tokens),
            float(temperature if temperature is not None else self.settings.temperature),
        )


class OpenAICompatibleChatModel(HTTPChatModel):
    default_base_url = "https://api.openai.com/v1"

    def build_request(self, messages, *, max_tokens=None, temperature=None):
        base_url = self.settings.resolved_base_url(default=self.default_base_url)
        api_key = self.settings.api_key(required=False)
        headers = {"Content-Type": "application/json", **self.settings.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        token_limit, temp = self._limits(max_tokens, temperature)
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            self.settings.max_tokens_field: token_limit,
            **self.settings.extra_body,
        }
        if self.settings.send_temperature:
            body["temperature"] = temp
        return f"{base_url}/chat/completions", headers, body

    def parse_response(self, payload):
        content = payload["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content or "")


class OpenAIChatModel(OpenAICompatibleChatModel):
    def build_request(self, messages, *, max_tokens=None, temperature=None):
        self.settings.api_key(required=True)
        return super().build_request(messages, max_tokens=max_tokens, temperature=temperature)


class QwenChatModel(OpenAICompatibleChatModel):
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def build_request(self, messages, *, max_tokens=None, temperature=None):
        self.settings.api_key(required=True)
        return super().build_request(messages, max_tokens=max_tokens, temperature=temperature)


class RelayChatModel(OpenAICompatibleChatModel):
    """Generic third-party relay implementing the OpenAI chat API."""

    def build_request(self, messages, *, max_tokens=None, temperature=None):
        self.settings.resolved_base_url(required=True)
        self.settings.api_key(required=True)
        return super().build_request(messages, max_tokens=max_tokens, temperature=temperature)


class AzureOpenAIChatModel(OpenAICompatibleChatModel):
    def build_request(self, messages, *, max_tokens=None, temperature=None):
        endpoint = self.settings.resolved_base_url(required=True)
        api_key = self.settings.api_key(required=True)
        token_limit, temp = self._limits(max_tokens, temperature)
        headers = {"Content-Type": "application/json", "api-key": api_key, **self.settings.extra_headers}
        body: dict[str, Any] = {
            "messages": list(messages),
            self.settings.max_tokens_field: token_limit,
            **self.settings.extra_body,
        }
        if self.settings.send_temperature:
            body["temperature"] = temp
        deployment = quote(self.settings.model, safe="-_.")
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={quote(self.settings.api_version)}"
        return url, headers, body


class AnthropicChatModel(HTTPChatModel):
    def build_request(self, messages, *, max_tokens=None, temperature=None):
        base_url = self.settings.resolved_base_url(default="https://api.anthropic.com/v1")
        api_key = self.settings.api_key(required=True)
        token_limit, temp = self._limits(max_tokens, temperature)
        system_parts = [item.get("content", "") for item in messages if item.get("role") == "system"]
        conversation = [
            {"role": "assistant" if item.get("role") == "assistant" else "user", "content": item.get("content", "")}
            for item in messages
            if item.get("role") != "system"
        ]
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            **self.settings.extra_headers,
        }
        body: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": token_limit,
            "messages": conversation,
            **self.settings.extra_body,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if self.settings.send_temperature:
            body["temperature"] = temp
        return f"{base_url}/messages", headers, body

    def parse_response(self, payload):
        return "".join(str(item.get("text", "")) for item in payload.get("content", []) if item.get("type") == "text")


class GeminiChatModel(HTTPChatModel):
    def build_request(self, messages, *, max_tokens=None, temperature=None):
        base_url = self.settings.resolved_base_url(default="https://generativelanguage.googleapis.com/v1beta")
        api_key = self.settings.api_key(required=True)
        token_limit, temp = self._limits(max_tokens, temperature)
        system_parts = [item.get("content", "") for item in messages if item.get("role") == "system"]
        contents = [
            {
                "role": "model" if item.get("role") == "assistant" else "user",
                "parts": [{"text": item.get("content", "")}],
            }
            for item in messages
            if item.get("role") != "system"
        ]
        generation: dict[str, Any] = {"maxOutputTokens": token_limit}
        if self.settings.send_temperature:
            generation["temperature"] = temp
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation, **self.settings.extra_body}
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        model = quote(self.settings.model, safe="-_.")
        headers = {"x-goog-api-key": api_key, **self.settings.extra_headers}
        return f"{base_url}/models/{model}:generateContent", headers, body

    def parse_response(self, payload):
        parts = payload["candidates"][0]["content"].get("parts", [])
        return "".join(str(item.get("text", "")) for item in parts)


ChatFactory = Callable[[LLMSettings, Any], ChatModel]
_PROVIDERS: dict[str, ChatFactory] = {}


def register_chat_provider(name: str, factory: ChatFactory, *, overwrite: bool = False) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Provider name cannot be empty")
    if key in _PROVIDERS and not overwrite:
        raise ValueError(f"Chat provider is already registered: {key}")
    _PROVIDERS[key] = factory


def build_chat_model(settings: LLMSettings, http_client: Any = None) -> ChatModel:
    provider = str(settings.provider).strip().lower()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown chat provider {provider!r}. Available: {available}")
    return factory(settings, http_client)


register_chat_provider("openai_compatible", OpenAICompatibleChatModel)
register_chat_provider("openai", OpenAIChatModel)
register_chat_provider("qwen", QwenChatModel)
register_chat_provider("dashscope", QwenChatModel)
register_chat_provider("relay", RelayChatModel)
register_chat_provider("relay2", RelayChatModel)
register_chat_provider("siliconflow", RelayChatModel)
register_chat_provider("azure_openai", AzureOpenAIChatModel)
register_chat_provider("anthropic", AnthropicChatModel)
register_chat_provider("gemini", GeminiChatModel)
register_chat_provider("google", GeminiChatModel)
