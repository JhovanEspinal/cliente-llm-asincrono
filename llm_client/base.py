from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse, Provider


class LLMError(Exception):
    def __init__(self, info: ErrorInfo) -> None:
        super().__init__(info.message)
        self.info = info


class BaseLLMClient(ABC):
    provider: Provider
    model: str

    @abstractmethod
    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse: ...

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]: ...
