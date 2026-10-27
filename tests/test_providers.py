from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import openai
import pytest
from pydantic import SecretStr

from llm_client import AnthropicClient, ChatMessage, LLMError, ModelConfig, OpenAIClient

CONFIG = ModelConfig()
MSGS = [
    ChatMessage(role="system", content="Responde breve."),
    ChatMessage(role="user", content="Hola"),
]
REQ = httpx.Request("POST", "https://api.test/v1")


def http_error(sdk, cls_name, status):
    resp = httpx.Response(status, request=REQ)
    return getattr(sdk, cls_name)("boom", response=resp, body=None)


async def agen(items):
    for item in items:
        yield item


def openai_client(create):
    client = OpenAIClient(api_key=SecretStr("sk-test"), model="gpt-test")
    client._client = NS(chat=NS(completions=NS(create=create)))
    return client


async def test_openai_generate_normaliza_respuesta():
    async def create(**kw):
        assert kw["messages"][0] == {"role": "system", "content": "Responde breve."}
        return NS(
            choices=[NS(message=NS(content="Hola!"), finish_reason="stop")],
            usage=NS(prompt_tokens=5, completion_tokens=2),
        )

    resp = await openai_client(create).generate(MSGS, CONFIG)
    assert resp.text == "Hola!" and resp.usage.output_tokens == 2 and resp.provider == "openai"


async def test_openai_stream_ignora_chunks_vacios():
    chunks = [
        NS(choices=[NS(delta=NS(content="Ho"))]),
        NS(choices=[]),
        NS(choices=[NS(delta=NS(content=None))]),
        NS(choices=[NS(delta=NS(content="la"))]),
    ]

    async def create(**kw):
        assert kw["stream"] is True
        return agen(chunks)

    out = [t async for t in openai_client(create).stream(MSGS, CONFIG)]
    assert out == ["Ho", "la"]


@pytest.mark.parametrize(
    "exc, kind, retryable",
    [
        (http_error(openai, "RateLimitError", 429), "rate_limit", True),
        (openai.APIConnectionError(request=REQ), "network", True),
        (openai.APITimeoutError(request=REQ), "network", True),
        (http_error(openai, "AuthenticationError", 401), "auth", False),
        (http_error(openai, "InternalServerError", 500), "server", True),
        (http_error(openai, "BadRequestError", 400), "bad_request", False),
    ],
)
async def test_openai_errores_se_normalizan(exc, kind, retryable):
    async def create(**kw):
        raise exc

    with pytest.raises(LLMError) as info:
        await openai_client(create).generate(MSGS, CONFIG)
    assert info.value.info.kind == kind and info.value.info.retryable is retryable


class FakeStream:
    def __init__(self, texts):
        self.text_stream = agen(texts)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def anthropic_client(create=None, stream=None):
    client = AnthropicClient(api_key=SecretStr("sk-ant-test"), model="claude-test")
    client._client = NS(messages=NS(create=create, stream=stream))
    return client


async def test_anthropic_generate_separa_system_y_no_envia_temperature():
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return NS(
            content=[NS(type="text", text="Hola!")],
            usage=NS(input_tokens=4, output_tokens=1),
            stop_reason="end_turn",
        )

    resp = await anthropic_client(create=create).generate(MSGS, ModelConfig(temperature=1.8))
    assert resp.text == "Hola!" and resp.usage.input_tokens == 4
    assert seen["system"] == "Responde breve."
    assert all(m["role"] != "system" for m in seen["messages"])
    assert "temperature" not in seen


async def test_anthropic_stream_entrega_fragmentos():
    client = anthropic_client(stream=lambda **kw: FakeStream(["Ho", "la"]))
    assert [t async for t in client.stream(MSGS, CONFIG)] == ["Ho", "la"]


@pytest.mark.parametrize(
    "exc, kind, retryable",
    [
        (http_error(anthropic, "RateLimitError", 429), "rate_limit", True),
        (anthropic.APIConnectionError(request=REQ), "network", True),
        (http_error(anthropic, "AuthenticationError", 401), "auth", False),
        (http_error(anthropic, "InternalServerError", 529), "server", True),
    ],
)
async def test_anthropic_errores_se_normalizan(exc, kind, retryable):
    async def create(**kw):
        raise exc

    with pytest.raises(LLMError) as info:
        await anthropic_client(create=create).generate(MSGS, CONFIG)
    assert info.value.info.kind == kind and info.value.info.retryable is retryable
