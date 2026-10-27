"""AsyncLLMManager: elige el proveedor por configuracion y agrega resiliencia."""

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

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
API_KEY_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _to_info(exc: Exception) -> ErrorInfo:
    """Errores del proveedor ya vienen normalizados; el resto se registra con
    traceback y se devuelve como `unknown` (no reintentable) en vez de tumbar el loop."""
    if isinstance(exc, LLMError):
        return exc.info
    log.exception("Error inesperado en el cliente LLM")
    return ErrorInfo(kind="unknown", message=f"{type(exc).__name__}: {exc}")


def build_client(provider: str) -> BaseLLMClient:
    """Factory: crea el cliente concreto leyendo las variables de entorno."""
    if provider not in API_KEY_VARS:
        raise ValueError(
            f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(API_KEY_VARS)}"
        )
    key_var = API_KEY_VARS[provider]
    api_key = os.getenv(key_var)
    if not api_key:
        raise ValueError(f"Falta la variable de entorno {key_var} para {provider}.")
    model = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    cls = OpenAIClient if provider == "openai" else AnthropicClient
    return cls(
        api_key=SecretStr(api_key),
        model=model,
        base_url=os.getenv(f"{provider.upper()}_BASE_URL") or None,
    )


class AsyncLLMManager:
    """Interfaz unica sobre cualquier proveedor.

    Los fallos del proveedor (red, rate limit, API key invalida...) nunca salen
    como excepcion: generate() devuelve un ModelResponse con `error` y stream()
    emite un StreamChunk final con `error`. Solo la configuracion invalida al
    construir el manager lanza ValueError, para fallar de forma clara y temprana.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        client: BaseLLMClient | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        if client is not None:
            self.client = client
        else:
            self.client = build_client(provider or os.getenv("LLM_PROVIDER", "anthropic"))

    @staticmethod
    def _as_messages(prompt: str | list[ChatMessage]) -> list[ChatMessage]:
        return [ChatMessage(role="user", content=prompt)] if isinstance(prompt, str) else prompt

    def _delay(self, attempt: int, config: ModelConfig) -> float:
        # Backoff exponencial con jitter para no sincronizar reintentos.
        return config.retry_base_delay_s * 2 ** (attempt - 1) * random.uniform(0.5, 1.0)

    async def generate(
        self, prompt: str | list[ChatMessage], config: ModelConfig | None = None
    ) -> ModelResponse:
        config = config or self.config
        messages = self._as_messages(prompt)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.generate(messages, config)
                response.attempts = attempt
                return response
            except Exception as exc:
                info = _to_info(exc)
                if info.retryable and attempt <= config.max_retries:
                    delay = self._delay(attempt, config)
                    log.warning(
                        "%s fallo (%s); reintento %d/%d en %.1fs",
                        self.client.provider, info.kind, attempt, config.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                log.error("%s fallo definitivo (%s): %s", self.client.provider, info.kind, info.message)
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
        messages = self._as_messages(prompt)
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                async for text in self.client.stream(messages, config):
                    emitted = True
                    yield StreamChunk(text=text)
                return
            except Exception as exc:
                info = _to_info(exc)
                # Solo se reintenta si aun no se emitio texto: reintentar a mitad
                # del stream duplicaria el contenido que el usuario ya vio.
                if info.retryable and not emitted and attempt <= config.max_retries:
                    delay = self._delay(attempt, config)
                    log.warning(
                        "%s stream fallo (%s); reintento %d/%d en %.1fs",
                        self.client.provider, info.kind, attempt, config.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                log.error("%s stream fallo definitivo (%s): %s", self.client.provider, info.kind, info.message)
                yield StreamChunk(error=info)
                return
