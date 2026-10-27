import asyncio
import logging
import os
import random
from collections.abc import AsyncIterator

from pydantic import SecretStr

from .base import BaseLLMClient, LLMError
from .providers import AnthropicClient, OpenAIClient
from .schemas import (
    ChatMessage,
    ErrorInfo,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamChunk,
)

log = logging.getLogger("llm_client")

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
API_KEY_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
JITTER_RANGE = (0.5, 1.0)


def error_info_from(exc: Exception) -> ErrorInfo:
    if isinstance(exc, LLMError):
        return exc.info
    log.exception("Error inesperado en el cliente LLM")
    return ErrorInfo(kind="unknown", message=f"{type(exc).__name__}: {exc}")


def build_client(provider: str) -> BaseLLMClient:
    if provider not in API_KEY_VARS:
        raise ValueError(
            f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(API_KEY_VARS)}"
        )
    key_var = API_KEY_VARS[provider]
    api_key = os.getenv(key_var)
    if not api_key:
        raise ValueError(f"Falta la variable de entorno {key_var} para {provider}.")
    model = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    client_class = OpenAIClient if provider == "openai" else AnthropicClient
    return client_class(
        api_key=SecretStr(api_key),
        model=model,
        base_url=os.getenv(f"{provider.upper()}_BASE_URL") or None,
    )


def exponential_backoff_with_jitter(attempt: int, config: ModelConfig) -> float:
    return config.retry_base_delay_s * 2 ** (attempt - 1) * random.uniform(*JITTER_RANGE)


def should_retry(
    info: ErrorInfo, attempt: int, config: ModelConfig, text_already_emitted: bool = False
) -> bool:
    return info.retryable and not text_already_emitted and attempt <= config.max_retries


class AsyncLLMManager:
    def __init__(
        self,
        provider: Provider | None = None,
        client: BaseLLMClient | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.client = client or build_client(
            provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        )

    async def generate(
        self, prompt: str | list[ChatMessage], config: ModelConfig | None = None
    ) -> ModelResponse:
        config = config or self.config
        messages = as_messages(prompt)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.generate(messages, config)
                response.attempts = attempt
                return response
            except Exception as exc:
                info = error_info_from(exc)
                if should_retry(info, attempt, config):
                    await self._wait_before_retry(info, attempt, config)
                    continue
                log.error(
                    "%s fallo definitivo (%s): %s",
                    self.client.provider, info.kind, info.message,
                )
                return ModelResponse(
                    provider=self.client.provider,
                    model=self.client.model,
                    attempts=attempt,
                    error=info,
                )

    async def stream(
        self, prompt: str | list[ChatMessage], config: ModelConfig | None = None
    ) -> AsyncIterator[StreamChunk]:
        config = config or self.config
        messages = as_messages(prompt)
        attempt = 0
        while True:
            attempt += 1
            text_already_emitted = False
            try:
                async for text in self.client.stream(messages, config):
                    text_already_emitted = True
                    yield StreamChunk(text=text)
                return
            except Exception as exc:
                info = error_info_from(exc)
                if should_retry(info, attempt, config, text_already_emitted):
                    await self._wait_before_retry(info, attempt, config)
                    continue
                log.error(
                    "%s stream fallo definitivo (%s): %s",
                    self.client.provider, info.kind, info.message,
                )
                yield StreamChunk(error=info)
                return

    async def _wait_before_retry(
        self, info: ErrorInfo, attempt: int, config: ModelConfig
    ) -> None:
        delay = exponential_backoff_with_jitter(attempt, config)
        log.warning(
            "%s fallo (%s); reintento %d/%d en %.1fs",
            self.client.provider, info.kind, attempt, config.max_retries, delay,
        )
        await asyncio.sleep(delay)


def as_messages(prompt: str | list[ChatMessage]) -> list[ChatMessage]:
    if isinstance(prompt, str):
        return [ChatMessage(role="user", content=prompt)]
    return prompt
