"""Manager con un cliente falso: reintentos, fallos controlados y streaming."""

import pytest

from llm_client import (
    AsyncLLMManager,
    BaseLLMClient,
    ErrorInfo,
    LLMError,
    ModelConfig,
    ModelResponse,
)

RATE_LIMIT = ErrorInfo(kind="rate_limit", message="429 Too Many Requests", retryable=True)
AUTH = ErrorInfo(kind="auth", message="API key invalida", retryable=False)


class FakeClient(BaseLLMClient):
    """Falla con los errores de `errors` (en orden) y luego responde bien."""

    provider = "anthropic"
    model = "fake"

    def __init__(self, errors=(), chunks=("Hola", " mundo"), fail_after_chunks=None):
        self.errors = list(errors)
        self.chunks = chunks
        self.fail_after_chunks = fail_after_chunks
        self.calls = 0

    async def generate(self, messages, config):
        self.calls += 1
        if self.errors:
            raise LLMError(self.errors.pop(0))
        return ModelResponse(provider=self.provider, model=self.model, text="ok")

    async def stream(self, messages, config):
        self.calls += 1
        if self.errors and self.fail_after_chunks is None:
            raise LLMError(self.errors.pop(0))
        for i, chunk in enumerate(self.chunks):
            if self.fail_after_chunks == i:
                raise LLMError(self.errors.pop(0))
            yield chunk


FAST = ModelConfig(max_retries=3, retry_base_delay_s=0)


def manager(client):
    return AsyncLLMManager(client=client, config=FAST)


async def test_generate_ok():
    resp = await manager(FakeClient()).generate("hola")
    assert resp.ok and resp.text == "ok" and resp.attempts == 1


async def test_rate_limit_se_reintenta_y_se_recupera():
    client = FakeClient(errors=[RATE_LIMIT, RATE_LIMIT])
    resp = await manager(client).generate("hola")
    assert resp.ok and resp.attempts == 3 and client.calls == 3


async def test_rate_limit_persistente_devuelve_error_sin_excepcion():
    client = FakeClient(errors=[RATE_LIMIT] * 10)
    resp = await manager(client).generate("hola")
    assert not resp.ok
    assert resp.error.kind == "rate_limit"
    assert client.calls == FAST.max_retries + 1  # intento inicial + reintentos


async def test_error_no_reintentable_no_se_reintenta():
    client = FakeClient(errors=[AUTH])
    resp = await manager(client).generate("hola")
    assert resp.error.kind == "auth" and client.calls == 1


async def test_stream_ok():
    chunks = [c async for c in manager(FakeClient()).stream("hola")]
    assert "".join(c.text for c in chunks) == "Hola mundo"
    assert all(c.error is None for c in chunks)


async def test_stream_reintenta_si_falla_antes_del_primer_token():
    client = FakeClient(errors=[RATE_LIMIT])
    chunks = [c async for c in manager(client).stream("hola")]
    assert "".join(c.text for c in chunks) == "Hola mundo"
    assert client.calls == 2


async def test_stream_no_reintenta_a_mitad_y_emite_error_controlado():
    client = FakeClient(errors=[RATE_LIMIT], fail_after_chunks=1)
    chunks = [c async for c in manager(client).stream("hola")]
    assert chunks[0].text == "Hola"
    assert chunks[-1].error.kind == "rate_limit"
    assert client.calls == 1  # reintentar duplicaria "Hola"


def test_proveedor_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        AsyncLLMManager(provider="gemini")


def test_falta_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AsyncLLMManager(provider="openai")


def test_proveedor_se_elige_por_variable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert AsyncLLMManager().client.provider == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert AsyncLLMManager().client.provider == "anthropic"
