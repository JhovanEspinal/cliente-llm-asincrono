import asyncio
import logging

from dotenv import load_dotenv

from llm_client import AsyncLLMManager

PREGUNTA = "¿Qué es la entropía?"


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        manager = AsyncLLMManager()
    except ValueError as exc:
        print(f"Configuracion invalida: {exc}")
        return
    print(f"Proveedor: {manager.client.provider} | modelo: {manager.client.model}\n")

    print("=== Modo normal ===")
    resp = await manager.generate(PREGUNTA)
    if resp.ok:
        print(resp.text)
        print(f"\n[{resp.latency_s:.2f}s | intentos={resp.attempts} | tokens={resp.usage}]")
    else:
        print(f"Error controlado ({resp.error.kind}): {resp.error.message}")

    print("\n=== Streaming ===")
    async for chunk in manager.stream(PREGUNTA):
        if chunk.error:
            print(f"\nError controlado ({chunk.error.kind}): {chunk.error.message}")
        else:
            print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
