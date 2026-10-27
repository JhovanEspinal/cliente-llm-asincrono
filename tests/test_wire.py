"""Los SDKs REALES (openai / anthropic) contra un servidor HTTP local.

A diferencia de test_providers.py, aqui NO se simula el SDK: se ejecuta de
verdad (construccion del request, headers, parseo de la respuesta, SSE,
excepciones) y solo se sustituye la red por un servidor local. Sirve para
comprobar que nuestro codigo usa los SDKs como realmente funcionan.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import SecretStr

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    ChatMessage,
    ModelConfig,
    OpenAIClient,
)

MSGS = [
    ChatMessage(role="system", content="Responde breve."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]
FAST = ModelConfig(max_retries=2, retry_base_delay_s=0)


class FakeAPI:
    """Servidor local con respuestas con el formato real de cada API."""

    def __init__(self):
        self.requests: list[dict] = []  # {"path", "headers", "body"}
        self.failures: list[int] = []  # codigos HTTP a devolver antes de responder bien
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _make_handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body: bytes, ctype="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                if api.failures:
                    status = api.failures.pop(0)
                    err = {"error": {"message": f"fallo {status}", "type": "x"}}
                    if self.path.endswith("/v1/messages"):
                        err = {"type": "error", "error": {"type": "x", "message": f"fallo {status}"}}
                    return self._send(status, json.dumps(err).encode())

                if self.path.endswith("/chat/completions"):
                    return self._openai(body)
                if self.path.endswith("/v1/messages"):
                    return self._anthropic(body)
                self._send(404, b"{}")

            def _openai(self, body):
                if body.get("stream"):
                    chunks = [
                        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                        {"choices": [{"index": 0, "delta": {"content": "Desorden"}}]},
                        {"choices": [{"index": 0, "delta": {"content": " molecular"}}]},
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    ]
                    sse = "".join(
                        f"data: {json.dumps({'id': 'c1', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm', **c})}\n\n"
                        for c in chunks
                    ) + "data: [DONE]\n\n"
                    return self._send(200, sse.encode(), "text/event-stream")
                resp = {
                    "id": "c1", "object": "chat.completion", "created": 1, "model": body["model"],
                    "choices": [{
                        "index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Desorden molecular"},
                    }],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
                self._send(200, json.dumps(resp).encode())

            def _anthropic(self, body):
                if body.get("stream"):
                    def ev(name, data):
                        return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"

                    msg = {"id": "m1", "type": "message", "role": "assistant", "model": body["model"],
                           "content": [], "stop_reason": None, "stop_sequence": None,
                           "usage": {"input_tokens": 7, "output_tokens": 0}}
                    sse = (
                        ev("message_start", {"message": msg})
                        + ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
                        + ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Desorden"}})
                        + ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": " molecular"}})
                        + ev("content_block_stop", {"index": 0})
                        + ev("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                               "usage": {"output_tokens": 2}})
                        + ev("message_stop", {})
                    )
                    return self._send(200, sse.encode(), "text/event-stream")
                resp = {
                    "id": "m1", "type": "message", "role": "assistant", "model": body["model"],
                    "content": [{"type": "text", "text": "Desorden molecular"}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 7, "output_tokens": 2},
                }
                self._send(200, json.dumps(resp).encode())

        return Handler


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()


def openai_client(api):
    return OpenAIClient(SecretStr("sk-test"), "gpt-4o-mini", base_url=f"{api.url}/v1")


def anthropic_client(api):
    return AnthropicClient(SecretStr("sk-ant-test"), "claude-haiku-5-5", base_url=api.url)


# ---------- OpenAI con el SDK real ----------


async def test_openai_generate(api):
    resp = await openai_client(api).generate(MSGS, ModelConfig(temperature=1.5, max_tokens=300))
    assert resp.text == "Desorden molecular"
    assert resp.finish_reason == "stop" and resp.usage.input_tokens == 7
    sent = api.requests[0]["body"]
    assert sent["temperature"] == 1.5 and sent["max_completion_tokens"] == 300
    assert sent["messages"][0] == {"role": "system", "content": "Responde breve."}
    assert api.requests[0]["headers"]["authorization"] == "Bearer sk-test"


async def test_openai_stream(api):
    out = [t async for t in openai_client(api).stream(MSGS, ModelConfig())]
    assert "".join(out) == "Desorden molecular"
    assert api.requests[0]["body"]["stream"] is True


# ---------- Anthropic con el SDK real ----------


async def test_anthropic_generate(api):
    resp = await anthropic_client(api).generate(MSGS, ModelConfig(temperature=1.5, max_tokens=300))
    assert resp.text == "Desorden molecular"
    assert resp.finish_reason == "end_turn" and resp.usage.output_tokens == 2
    sent = api.requests[0]["body"]
    assert sent["system"] == "Responde breve."
    assert sent["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert sent["max_tokens"] == 300
    assert "temperature" not in sent
    assert api.requests[0]["headers"]["authorization"] == "Bearer sk-ant-test"
    # Desde esta version de la API, enviar temperature/top_p/top_k da error 400.
    assert api.requests[0]["headers"]["anthropic-version"] == "2026-10-21"


async def test_anthropic_stream(api):
    out = [t async for t in anthropic_client(api).stream(MSGS, ModelConfig())]
    assert "".join(out) == "Desorden molecular"
    assert api.requests[0]["body"]["stream"] is True


# ---------- Manager + SDK real: errores HTTP reales ----------


@pytest.mark.parametrize("make_client", [openai_client, anthropic_client])
async def test_rate_limit_real_se_reintenta_y_se_recupera(api, make_client):
    api.failures = [429, 429]
    manager = AsyncLLMManager(client=make_client(api), config=FAST)
    resp = await manager.generate(MSGS)
    assert resp.ok and resp.attempts == 3 and resp.text == "Desorden molecular"


@pytest.mark.parametrize("make_client", [openai_client, anthropic_client])
async def test_rate_limit_persistente_devuelve_error_controlado(api, make_client):
    api.failures = [429] * 10
    manager = AsyncLLMManager(client=make_client(api), config=FAST)
    resp = await manager.generate(MSGS)
    assert not resp.ok and resp.error.kind == "rate_limit"
    assert len(api.requests) == FAST.max_retries + 1


@pytest.mark.parametrize("make_client", [openai_client, anthropic_client])
async def test_api_key_invalida_no_se_reintenta(api, make_client):
    api.failures = [401]
    manager = AsyncLLMManager(client=make_client(api), config=FAST)
    resp = await manager.generate(MSGS)
    assert resp.error.kind == "auth" and len(api.requests) == 1


@pytest.mark.parametrize("make_client", [openai_client, anthropic_client])
async def test_stream_con_rate_limit_real_se_recupera(api, make_client):
    api.failures = [429]
    manager = AsyncLLMManager(client=make_client(api), config=FAST)
    chunks = [c async for c in manager.stream(MSGS)]
    assert "".join(c.text for c in chunks) == "Desorden molecular"
    assert all(c.error is None for c in chunks)


async def test_servidor_caido_es_error_de_red_controlado():
    # Puerto sin servidor: conexion rechazada -> APIConnectionError real del SDK.
    client = AnthropicClient(SecretStr("sk-ant-test"), "m", base_url="http://127.0.0.1:9")
    resp = await AsyncLLMManager(client=client, config=FAST).generate(MSGS)
    assert not resp.ok and resp.error.kind == "network"
