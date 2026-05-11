"""
Servidor MCP de Observabilidad — Agente Conversacional
Expone métricas de uso del sistema (tokens, costos, servicios) como herramientas MCP.

Inicio automático: lo lanza mcp_client.py como subproceso stdio.
También puede iniciarse manualmente:
    python mcp_observability_server.py
"""

from mcp.server.fastmcp import FastMCP
from services.finops_service import get_session_total, get_global_stats, _load_log

mcp = FastMCP(
    name="observabilidad-agente",
    instructions=(
        "Proporciona métricas de observabilidad del agente conversacional: "
        "consumo de tokens, costos en USD y desglose de llamadas por servicio."
    ),
)


@mcp.tool()
def get_session_usage(session_id: str) -> dict:
    """
    Devuelve el consumo acumulado de tokens y costo en USD para una sesión específica.

    Parámetros:
    - session_id: identificador de la sesión (thread_id de LangGraph)

    Devuelve:
    - total_cost_usd: costo total de la sesión en dólares
    - total_tokens_in: tokens de entrada consumidos
    - total_tokens_out: tokens de salida generados
    - calls_count: número de llamadas al LLM en esta sesión
    """
    return get_session_total(session_id)


@mcp.tool()
def get_global_usage() -> dict:
    """
    Devuelve las estadísticas globales acumuladas de todas las sesiones históricas.

    Devuelve:
    - total_cost_usd: costo total histórico en dólares
    - total_calls: número total de llamadas al LLM de todas las sesiones
    """
    return get_global_stats()


@mcp.tool()
def get_service_breakdown(session_id: str) -> dict:
    """
    Devuelve el desglose de uso por servicio para una sesión específica.
    Muestra cuántas veces se usó cada herramienta (sentimiento, resumen, propagación, etc.)
    y cuántos tokens/costo generó cada una.

    Parámetros:
    - session_id: identificador de la sesión a analizar

    Devuelve:
    - breakdown: dict con stats por servicio (calls, tokens_in, tokens_out, cost_usd)
    - total_calls: total de llamadas en la sesión
    - session_id: el session_id analizado
    """
    log = _load_log()
    session = log["sessions"].get(session_id, {})
    calls = session.get("calls", [])

    breakdown: dict = {}
    for call in calls:
        svc = call.get("service", "unknown")
        if svc not in breakdown:
            breakdown[svc] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        breakdown[svc]["calls"] += 1
        breakdown[svc]["tokens_in"] += call.get("tokens_in", 0)
        breakdown[svc]["tokens_out"] += call.get("tokens_out", 0)
        breakdown[svc]["cost_usd"] = round(breakdown[svc]["cost_usd"] + call.get("cost_usd", 0.0), 8)

    return {
        "session_id": session_id,
        "breakdown": breakdown,
        "total_calls": len(calls),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
