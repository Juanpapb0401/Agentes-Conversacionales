"""
Herramientas (Tools) del Agente Conversacional — Rol 4
Cada función envuelve un endpoint de la API FastAPI local y expone
una descripción clara para que el LLM decida cuándo invocarla.
"""

import os
import httpx
from typing import Any, Dict, List
from langchain_core.tools import tool
from dotenv import load_dotenv

from services.finops_service import SESSION_CTX as _SESSION_CTX

load_dotenv()

# URL base de la API (configurable por variable de entorno)
_API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Timeout de las llamadas HTTP en segundos
_HTTP_TIMEOUT = 30.0

# Session ID set by wrap_tool_call before each tool execution
_active_session_id: str = "unknown"


def _tool_call_wrapper(request, execute):
    """
    Intercepts every tool call to propagate session_id from LangGraph config
    into the module-level _active_session_id, bypassing ContextVar threading issues.
    Reads thread_id / session_id from request.runtime.config.configurable.
    """
    global _active_session_id
    runtime = request.runtime
    if isinstance(runtime, dict):
        configurable = runtime.get("configurable") or (runtime.get("config") or {}).get("configurable")
    else:
        configurable = getattr(getattr(runtime, "config", None), "configurable", None) or {}
    session_id = configurable.get("session_id") or configurable.get("thread_id") or "unknown"
    _active_session_id = session_id
    try:
        return execute(request)
    finally:
        _active_session_id = "unknown"


def _session_headers() -> Dict[str, str]:
    return {"X-Session-ID": _active_session_id}


def _get(endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Realiza una petición GET a la API y devuelve el JSON de respuesta."""
    url = f"{_API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.get(url, params=params, headers=_session_headers())
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise RuntimeError(
            "No se puede conectar con la API. Asegúrate de que el servidor FastAPI "
            f"esté corriendo en {_API_BASE_URL}."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Error HTTP {exc.response.status_code} en {endpoint}: {exc.response.text}"
        ) from exc


def _post(endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Realiza una petición POST a la API y devuelve el JSON de respuesta."""
    url = f"{_API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.post(url, json=body, headers=_session_headers())
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise RuntimeError(
            "No se puede conectar con la API. Asegúrate de que el servidor FastAPI "
            f"esté corriendo en {_API_BASE_URL}."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Error HTTP {exc.response.status_code} en {endpoint}: {exc.response.text}"
        ) from exc


# =============================================================================
# TOOL 1 — Análisis de Sentimientos
# =============================================================================

@tool
def tool_analizar_sentimiento(text: str) -> Dict[str, Any]:
    """
    Analiza el clima emocional o semántico de un texto dado.
    Úsala cuando el usuario quiera saber si una conversación, comentario o publicación
    es positiva, negativa o neutral. Devuelve el clima (positivo/negativo/neutral),
    un score numérico entre 0 y 1, y una justificación del análisis.

    Ejemplos de consultas que activan esta herramienta:
    - "¿Cómo está el clima de la conversación?"
    - "¿Este mensaje es positivo o negativo?"
    - "Analiza el sentimiento de este texto: ..."
    - "¿Cuál es el tono emocional de estos comentarios?"
    """
    if not text or not text.strip():
        return {
            "clima": "neutral",
            "score": 0.5,
            "justificacion": "No se proporcionó texto para analizar.",
        }
    return _post("/analisis/sentimientos", {"text": text})


# =============================================================================
# TOOL 2 — Resumen General de Conversación
# =============================================================================

@tool
def tool_resumir_conversacion(textos: List[str], max_palabras: int = 120, idioma: str = "es") -> Dict[str, Any]:
    """
    Genera un resumen conciso de una lista de textos o comentarios de una conversación digital.
    Identifica los temas principales y las posturas clave presentes en el debate.
    Úsala cuando el usuario quiera entender de qué trata una conversación o un conjunto
    de publicaciones sin leerlas todas.

    Argumentos:
    - textos: lista de strings con los mensajes o comentarios a resumir.
    - max_palabras: límite aproximado de palabras del resumen (por defecto 120).
    - idioma: idioma esperado del resumen (por defecto "es" para español).

    Ejemplos de consultas que activan esta herramienta:
    - "Resume esta conversación"
    - "¿De qué hablan estos comentarios?"
    - "¿Cuál es la temática principal del hilo?"
    - "Hazme un resumen de estas publicaciones"
    """
    textos_limpios = [t for t in textos if t and t.strip()]
    if not textos_limpios:
        return {
            "resumen": "No se proporcionaron textos para resumir.",
            "temas_principales": [],
            "posturas_clave": [],
            "alcance_textos": 0,
        }
    return _post(
        "/analisis/resumen",
        {"textos": textos_limpios, "max_palabras": max_palabras, "idioma": idioma},
    )


# =============================================================================
# TOOL 3 — Análisis de Propagación de un Mensaje (OBLIGATORIO)
# =============================================================================

@tool
def tool_analizar_propagacion(post_id: str) -> Dict[str, Any]:
    """
    Analiza la propagación e impacto mediático de un mensaje específico dentro de la red,
    identificado por su post_id. Calcula el alcance total (número de nodos que respondieron
    al mensaje), las respuestas directas, la velocidad de propagación (en minutos),
    el porcentaje de contenido replicado y un score de impacto del 0 al 100.

    Úsala cuando el usuario pregunte por la viralidad, el impacto o la difusión de
    un mensaje concreto usando su identificador.

    Argumentos:
    - post_id: identificador único del mensaje a analizar (string).

    Ejemplos de consultas que activan esta herramienta:
    - "¿Qué tan viral fue el mensaje con ID abc123?"
    - "Analiza la propagación del post 456"
    - "¿Cuál fue el alcance del mensaje XYZ?"
    - "¿Cuántas respuestas generó el post con ID 789?"
    - "¿Qué impacto mediático tuvo este mensaje?"
    """
    if not post_id or not post_id.strip():
        return {"encontrado": False, "error": "Debes proporcionar un post_id válido."}
    return _get("/analisis/propagacion", params={"post_id": post_id.strip()})


# =============================================================================
# TOOL 4 — Métricas Generales de la Red
# =============================================================================

@tool
def tool_analizar_metricas() -> Dict[str, Any]:
    """
    Obtiene las métricas generales de toda la red de conversaciones: el total de likes
    acumulados y los top 5 usuarios más influyentes (con mayor cantidad de likes).
    Úsala cuando el usuario quiera un panorama general de la actividad, los actores
    clave o las estadísticas globales del dataset.

    Esta herramienta no requiere ningún argumento; analiza todo el dataset disponible.

    Ejemplos de consultas que activan esta herramienta:
    - "¿Quiénes son los usuarios más influyentes?"
    - "¿Cuáles son las métricas generales de la red?"
    - "¿Cuántos likes tiene la red en total?"
    - "Dame un resumen de los actores más activos"
    - "¿Qué publicaciones tienen más repercusión?"
    """
    return _post("/analisis/metricas", {})


# =============================================================================
# Lista exportable para el Agente + wrapper de sesion
# =============================================================================

TOOLS = [
    tool_analizar_sentimiento,
    tool_resumir_conversacion,
    tool_analizar_propagacion,
    tool_analizar_metricas,
]

wrap_tool_call = _tool_call_wrapper
