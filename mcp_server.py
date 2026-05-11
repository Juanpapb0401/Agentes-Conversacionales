"""
Servidor MCP — Análisis de Conversaciones Digitales
Expone las 4 herramientas analíticas del proyecto como herramientas MCP estándar.

Uso (stdio, compatible con Claude Desktop y Claude Code):
    python mcp_server.py

El servidor FastAPI debe estar corriendo en API_BASE_URL antes de iniciar este servidor.
"""

import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

_API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_TIMEOUT = 30.0

mcp = FastMCP(
    name="analisis-conversaciones",
    instructions=(
        "Servidor de análisis de conversaciones digitales en redes sociales. "
        "Proporciona herramientas para analizar sentimientos, resumir conversaciones, "
        "medir la propagación viral de mensajes y obtener métricas generales del dataset."
    ),
)


def _get(endpoint: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(f"{_API_BASE}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()


def _post(endpoint: str, body: dict) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(f"{_API_BASE}{endpoint}", json=body)
        resp.raise_for_status()
        return resp.json()


# =============================================================================
# Tool 1 — Análisis de Sentimientos
# =============================================================================

@mcp.tool()
def analizar_sentimiento(text: str) -> dict:
    """
    Analiza el clima emocional o semántico de un texto.

    Devuelve:
    - clima: "positivo", "negativo" o "neutral"
    - score: número entre 0 y 1 (1 = máxima intensidad del clima detectado)
    - justificacion: explicación del análisis

    Úsala para saber si un comentario, publicación o conversación tiene
    un tono positivo, negativo o neutral.
    """
    if not text or not text.strip():
        return {"clima": "neutral", "score": 0.5, "justificacion": "Texto vacío."}
    return _post("/analisis/sentimientos", {"text": text})


# =============================================================================
# Tool 2 — Resumen de Conversación
# =============================================================================

@mcp.tool()
def resumir_conversacion(textos: list[str], max_palabras: int = 120, idioma: str = "es") -> dict:
    """
    Genera un resumen de una lista de textos o comentarios de una conversación digital.

    Parámetros:
    - textos: lista de mensajes o publicaciones a resumir
    - max_palabras: límite aproximado de palabras del resumen (default 120)
    - idioma: idioma del resumen, "es" para español (default)

    Devuelve:
    - resumen: texto síntesis de la conversación
    - temas_principales: lista de temas detectados
    - posturas_clave: posturas o posiciones identificadas en el debate
    - alcance_textos: cantidad de textos procesados
    """
    limpios = [t for t in textos if t and t.strip()]
    if not limpios:
        return {"resumen": "No se proporcionaron textos.", "temas_principales": [], "posturas_clave": [], "alcance_textos": 0}
    return _post("/analisis/resumen", {"textos": limpios, "max_palabras": max_palabras, "idioma": idioma})


# =============================================================================
# Tool 3 — Análisis de Propagación
# =============================================================================

@mcp.tool()
def analizar_propagacion(post_id: str) -> dict:
    """
    Mide la propagación e impacto viral de un mensaje específico en la red.

    Usa BFS sobre el grafo de conversaciones para encontrar todos los
    mensajes que respondieron (directa o indirectamente) al post_id dado.

    Parámetros:
    - post_id: identificador único del mensaje a analizar

    Devuelve métricas como:
    - alcance: nodos totales en la cadena de propagación
    - replies_directas: respuestas inmediatas al post original
    - cadena_total_nodos: descendientes totales (BFS completo)
    - velocidad_media_min: tiempo medio entre réplicas en minutos
    - porcentaje_contenido_replicado: % de nodos que repiten palabras del original
    - score_impacto: puntuación de 0 a 100
    - nivel_impacto: "Muy Alto", "Alto", "Medio" o "Bajo"
    """
    if not post_id or not post_id.strip():
        return {"encontrado": False, "error": "Debes proporcionar un post_id válido."}
    return _get("/analisis/propagacion", params={"post_id": post_id.strip()})


# =============================================================================
# Tool 4 — Métricas Generales de la Red
# =============================================================================

@mcp.tool()
def analizar_metricas() -> dict:
    """
    Obtiene las métricas globales de toda la red de conversaciones del dataset.

    No requiere parámetros. Analiza el dataset completo y devuelve:
    - total_likes: suma total de likes en la red
    - total_mensajes: número total de publicaciones en el dataset
    - top_influencers: top 5 usuarios con más likes acumulados
    - top_posts_por_likes: top 5 publicaciones con más likes (con fragmento de texto)
    - plataformas: plataformas de redes sociales presentes en el dataset
    """
    return _post("/analisis/metricas", {})


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
