"""Interfaz comun para todos los proveedores."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse, Provider


class LLMError(Exception):
    """Error normalizado de un proveedor. Lo capturan el manager y los reintentos."""

    def __init__(self, info: ErrorInfo) -> None:
        super().__init__(info.message)
        self.info = info


class BaseLLMClient(ABC):
    provider: Provider
    model: str

    @abstractmethod
    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        """Respuesta completa. Lanza LLMError ante fallos del proveedor."""

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        """Generador asincrono de fragmentos de texto. Lanza LLMError ante fallos."""
