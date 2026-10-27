"""Esquemas Pydantic: mensajes de entrada, configuracion del modelo y respuestas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]
Provider = Literal["openai", "anthropic"]
ErrorKind = Literal["rate_limit", "network", "auth", "server", "bad_request", "unknown"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parametros de la llamada. Pydantic rechaza valores fuera de rango."""

    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0, le=32_000)
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay_s: float = Field(default=1.0, ge=0)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ErrorInfo(BaseModel):
    """Error controlado: lo que el llamador recibe en vez de una excepcion."""

    kind: ErrorKind
    message: str
    retryable: bool = False


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None  # p. ej. "stop"/"end_turn"; "length"/"max_tokens" = truncada
    latency_s: float = 0.0
    attempts: int = 1
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento de streaming: texto, o un error controlado que cierra el flujo."""

    text: str = ""
    error: ErrorInfo | None = None
