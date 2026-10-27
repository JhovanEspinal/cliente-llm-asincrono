"""Clientes concretos: OpenAI y Anthropic con sus SDKs asincronos oficiales."""

import time
from collections.abc import AsyncIterator

import anthropic
import openai
from pydantic import SecretStr

from .base import BaseLLMClient, LLMError
from .schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse, Usage


def _status_error(status: int, message: str) -> ErrorInfo:
    if status >= 500:
        return ErrorInfo(kind="server", message=message, retryable=True)
    return ErrorInfo(kind="bad_request", message=message, retryable=False)


def _normalize_openai_error(exc: Exception) -> LLMError:
    # APITimeoutError hereda de APIConnectionError: ambos son fallos de red.
    if isinstance(exc, openai.RateLimitError):
        info = ErrorInfo(kind="rate_limit", message=str(exc), retryable=True)
    elif isinstance(exc, openai.APIConnectionError):
        info = ErrorInfo(kind="network", message=str(exc), retryable=True)
    elif isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        info = ErrorInfo(kind="auth", message=str(exc), retryable=False)
    elif isinstance(exc, openai.APIStatusError):
        info = _status_error(exc.status_code, str(exc))
    else:
        info = ErrorInfo(kind="unknown", message=f"{type(exc).__name__}: {exc}")
    return LLMError(info)


def _normalize_anthropic_error(exc: Exception) -> LLMError:
    if isinstance(exc, anthropic.RateLimitError):
        info = ErrorInfo(kind="rate_limit", message=str(exc), retryable=True)
    elif isinstance(exc, anthropic.APIConnectionError):
        info = ErrorInfo(kind="network", message=str(exc), retryable=True)
    elif isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        info = ErrorInfo(kind="auth", message=str(exc), retryable=False)
    elif isinstance(exc, anthropic.APIStatusError):
        info = _status_error(exc.status_code, str(exc))
    else:
        info = ErrorInfo(kind="unknown", message=f"{type(exc).__name__}: {exc}")
    return LLMError(info)


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, api_key: SecretStr, model: str) -> None:
        self.model = model
        # max_retries=0: los reintentos los controla el manager, no el SDK.
        self._client = openai.AsyncOpenAI(
            api_key=api_key.get_secret_value(), max_retries=0
        )

    @staticmethod
    def _payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=self._payload(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_s,
            )
        except openai.OpenAIError as exc:
            raise _normalize_openai_error(exc) from exc
        usage = (
            Usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
            if resp.usage
            else None
        )
        return ModelResponse(
            provider=self.provider,
            model=self.model,
            text=resp.choices[0].message.content or "",
            usage=usage,
            latency_s=time.perf_counter() - start,
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=self._payload(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_s,
                stream=True,
            )
            async for chunk in stream:
                # Algunos chunks (p. ej. el ultimo) llegan sin choices o sin texto.
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as exc:
            raise _normalize_openai_error(exc) from exc


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"
    MAX_TEMPERATURE = 1.0  # la API de Anthropic acepta 0-1, no 0-2

    def __init__(self, api_key: SecretStr, model: str) -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key.get_secret_value(), max_retries=0
        )

    def _kwargs(self, messages: list[ChatMessage], config: ModelConfig) -> dict:
        # Anthropic recibe el system prompt aparte, no dentro de messages.
        system = "\n".join(m.content for m in messages if m.role == "system")
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
            "temperature": min(config.temperature, self.MAX_TEMPERATURE),
            "max_tokens": config.max_tokens,
            "timeout": config.timeout_s,
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(**self._kwargs(messages, config))
        except anthropic.AnthropicError as exc:
            raise _normalize_anthropic_error(exc) from exc
        return ModelResponse(
            provider=self.provider,
            model=self.model,
            text="".join(b.text for b in resp.content if b.type == "text"),
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            latency_s=time.perf_counter() - start,
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(
                **self._kwargs(messages, config)
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as exc:
            raise _normalize_anthropic_error(exc) from exc
