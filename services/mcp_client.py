"""
Cliente MCP síncrono para llamar herramientas del servidor MCP de observabilidad
desde Streamlit (código síncrono) y desde el agente LangGraph.

Cada llamada lanza el servidor MCP como subproceso stdio, ejecuta la herramienta
y cierra la conexión. Esto es correcto para baja frecuencia de llamadas (sidebar).
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

# Ruta al servidor MCP (junto a este archivo en la raíz del proyecto)
_PROJECT_ROOT = Path(__file__).parent.parent
_SERVER_SCRIPT = str(_PROJECT_ROOT / "mcp_observability_server.py")


async def _call_tool_async(tool_name: str, args: dict) -> Any:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_SERVER_SCRIPT],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            if result.content:
                text = result.content[0].text
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, AttributeError):
                    return text
            return {}


def call_mcp_tool(tool_name: str, args: dict | None = None) -> Any:
    """
    Llama a una herramienta del servidor MCP de observabilidad de forma síncrona.
    Compatible con Streamlit (crea un event loop nuevo en un hilo auxiliar).

    Args:
        tool_name: nombre de la herramienta MCP a invocar
        args: argumentos para la herramienta (dict)

    Returns:
        Resultado de la herramienta como dict o string.

    Raises:
        RuntimeError: si el servidor MCP no responde o falla.
    """
    if args is None:
        args = {}

    result: list[Any] = [None]
    exc: list[Exception | None] = [None]

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result[0] = loop.run_until_complete(_call_tool_async(tool_name, args))
        except Exception as e:
            exc[0] = e
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=20)

    if thread.is_alive():
        raise RuntimeError("El servidor MCP de observabilidad no respondió a tiempo.")
    if exc[0]:
        raise RuntimeError(f"Error al llamar herramienta MCP '{tool_name}': {exc[0]}") from exc[0]

    return result[0]
