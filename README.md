# Unified Async LLM Client

Cliente asíncrono para LLMs con una interfaz común sobre **OpenAI** y **Anthropic**. Pre-Entrega 1 del módulo 1 de AI Engineering (Python 3.12).

## Qué incluye

| Archivo | Rol |
|---|---|
| `llm_client/schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelConfig` (temperatura 0–2, `max_tokens > 0`, timeout, reintentos), `ModelResponse`, `StreamChunk`, `ErrorInfo` |
| `llm_client/base.py` | `BaseLLMClient` (ABC) con `generate()` y `stream()`, y `LLMError` |
| `llm_client/providers.py` | `OpenAIClient` (`AsyncOpenAI`) y `AnthropicClient` (`AsyncAnthropic`) |
| `llm_client/manager.py` | `AsyncLLMManager`: elige el proveedor por configuración y añade reintentos |
| `main.py` | Script de prueba: "¿Qué es la entropía?" en modo normal y en streaming |
| `tests/` | Tests con SDKs simulados (no usan red ni API keys) |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # en Windows: copy .env.example .env
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `anthropic` u `openai` (por defecto `anthropic`) |
| `ANTHROPIC_API_KEY` | Clave de Anthropic (obligatoria si el proveedor es `anthropic`) |
| `OPENAI_API_KEY` | Clave de OpenAI (obligatoria si el proveedor es `openai`) |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | Opcionales. Por defecto `claude-haiku-5-5` / `gpt-4o-mini` |

Solo hace falta la clave del proveedor que uses. `.env` está en `.gitignore`: nunca se sube. Las claves se manejan como `SecretStr` para que no aparezcan en logs.

## Ejecutar

```bash
python main.py       # prueba real contra el proveedor de LLM_PROVIDER
python -m pytest     # tests sin red
```

## Uso desde código

```python
from llm_client import AsyncLLMManager, ModelConfig

manager = AsyncLLMManager(provider="anthropic", config=ModelConfig(temperature=0.2))

resp = await manager.generate("¿Qué es la entropía?")
print(resp.text if resp.ok else resp.error)

async for chunk in manager.stream("¿Qué es la entropía?"):
    print(chunk.text, end="", flush=True)
```

## Resiliencia

- **Nunca se escapa una excepción del proveedor.** `generate()` devuelve un `ModelResponse` con `error` (`kind`, `message`, `retryable`) y `stream()` emite un `StreamChunk` final con `error`. Solo una configuración inválida al crear el manager (proveedor desconocido, falta la API key) lanza `ValueError`, para fallar pronto y con un mensaje claro.
- **Tipos de error:** `rate_limit`, `network`, `server` (reintentables); `auth`, `bad_request`, `unknown` (no reintentables).
- **Reintentos** con backoff exponencial y jitter (`max_retries`, `retry_base_delay_s` en `ModelConfig`). Se desactivan los reintentos internos de los SDKs para que el control sea único.
- **Streaming:** solo se reintenta si el error ocurre *antes* del primer token; a mitad de la respuesta se emite el error, porque reintentar duplicaría texto que el usuario ya vio.

## Notas

- **Temperatura en Anthropic:** la API acepta 0–1, así que `AnthropicClient` limita el valor a 1.0 aunque `ModelConfig` permita hasta 2 (rango de OpenAI).
- **System prompt:** Anthropic lo recibe aparte de `messages`; `AnthropicClient` lo separa automáticamente.
- **Estado de las pruebas:** ambos clientes se verifican con los SDKs simulados (`python -m pytest`). Aún no se ejecutó `main.py` contra una API real; para hacerlo, añade una API key en `.env`.
