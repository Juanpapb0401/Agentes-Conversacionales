"""
Grafo Conversacional LangGraph — Rol 4: Agentic Framework Builder

Arquitectura del grafo:
    [START]
       │
       ▼
  llamar_modelo  ──(tool_calls?)──► ejecutar_herramienta
       ▲                                      │
       └──────────────────────────────────────┘
       │
   (sin tool calls)
       │
       ▼
     [END]

El grafo usa MemorySaver para persistir el historial por session (thread_id),
lo que habilita conversaciones multi-turno con memoria de contexto.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from agent.tools import TOOLS
from services.finops_service import log_call as _finops_log

load_dotenv()

# =============================================================================
# 1. Selección del LLM según LLM_PROVIDER (espeja la lógica de nlp_service.py)
# =============================================================================

def _build_llm():
    """
    Instancia el LLM configurado en la variable de entorno LLM_PROVIDER.
    Admite 'openai' y 'gemini'. Lanza un error descriptivo si falta configuración.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta OPENAI_API_KEY en el archivo .env para usar el proveedor 'openai'."
            )
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta GEMINI_API_KEY en el archivo .env para usar el proveedor 'gemini'."
            )
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)

    raise RuntimeError(
        "LLM_PROVIDER no está configurado en el .env. "
        "Establece LLM_PROVIDER=openai o LLM_PROVIDER=gemini."
    )


# =============================================================================
# 2. Prompt del sistema que define el rol y el alcance del agente
# =============================================================================

SYSTEM_PROMPT = SystemMessage(content="""Eres un Agente Conversacional especializado en el análisis de conversaciones digitales.
Tienes acceso a cuatro herramientas analíticas que debes invocar cuando el usuario lo necesite:

1. tool_analizar_sentimiento — para determinar el clima emocional (positivo, negativo, neutral) de un texto.
2. tool_resumir_conversacion — para generar un resumen con temas y posturas clave de una lista de textos.
3. tool_analizar_propagacion — para medir el alcance, velocidad e impacto de un mensaje identificado por su post_id.
4. tool_analizar_metricas — para obtener estadísticas generales: total de likes y top influencers de la red.

Instrucciones de comportamiento:
- Identifica la intención del usuario y selecciona la herramienta más adecuada.
- Usa siempre el idioma del usuario (responde en español si el usuario habla en español).
- Cuando el usuario pida analizar un mensaje específico, solicita el post_id si no lo ha proporcionado.
- Presenta los resultados de manera clara, estructurada y en lenguaje natural.
- Si la pregunta está fuera de tu alcance, explícalo brevemente y sugiere qué análisis sí puedes realizar.
- Nunca inventes datos; solo reporta lo que las herramientas devuelven.

IMPORTANTE (Análisis de Propagación):
- Al usar tool_analizar_propagacion, debes explicar de forma narrativa el **Arquetipo de la Conversación** (Estrella, Hilo Crítico, Explosión Viral, etc.) y la **Profundidad Máxima** del hilo.
- Usa estos datos para explicar la dinámica social (ej: si es un hilo profundo, destaca que hubo un debate intenso; si es una estrella, destaca que fue ruido momentáneo).

Manejo de errores de propagación:
- Los post_id son hashes hexadecimales largos (ej: "c6adb4630994bdee807d387382d526bc"), NO números simples como "1001".
- Si la pregunta está fuera de tu alcance, explícalo brevemente y sugiere qué análisis sí puedes realizar.
- Nunca inventes datos; solo reporta lo que las herramientas devuelven.
- Si tool_analizar_propagacion devuelve un error 404, explica al usuario que ese ID no existe en el dataset.
- Nunca intentes adivinar o inventar un post_id válido.
""")


# =============================================================================
# 3. Nodos del Grafo
# =============================================================================

# Instanciar el LLM con las tools vinculadas (lazy: se hace al construir el grafo)
def _crear_nodo_modelo(llm_con_tools, model_name: str):
    """Fábrica del nodo llamar_modelo con el LLM ya inyectado."""

    def llamar_modelo(state: MessagesState, config: RunnableConfig) -> dict:
        """
        Nodo principal: invoca el LLM con el historial completo de mensajes.
        El sistema siempre encabeza la conversación para mantener el contexto del agente.
        """
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        mensajes = [SYSTEM_PROMPT] + state["messages"]
        respuesta = llm_con_tools.invoke(mensajes)

        # Extraer y registrar uso de tokens del agente
        # 1) LangChain standardized attribute (funciona con todos los providers)
        usage_lc = getattr(respuesta, "usage_metadata", None) or {}
        # 2) Fallback: Gemini raw response_metadata
        meta = respuesta.response_metadata or {}
        usage_gemini = meta.get("usage_metadata", {})
        # 3) Fallback: OpenAI raw response_metadata
        usage_openai = meta.get("token_usage", {})
        tokens_in = (
            usage_lc.get("input_tokens")
            or usage_gemini.get("prompt_token_count")
            or usage_openai.get("prompt_tokens")
            or 0
        )
        tokens_out = (
            usage_lc.get("output_tokens")
            or usage_gemini.get("candidates_token_count")
            or usage_openai.get("completion_tokens")
            or 0
        )
        if tokens_in or tokens_out:
            _finops_log(thread_id, "agente", model_name, tokens_in, tokens_out)

        return {"messages": [respuesta]}

    return llamar_modelo


# =============================================================================
# 4. Construcción y compilación del Grafo
# =============================================================================

def construir_grafo():
    """
    Construye y compila el grafo LangGraph con MemorySaver.

    Returns:
        CompiledGraph listo para invocar con `graph.invoke(input, config)`.
    """
    # Instanciar LLM y vincular tools (tool-calling)
    llm = _build_llm()
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    llm_con_tools = llm.bind_tools(TOOLS)

    # Nodo de ejecución de herramientas (ToolNode maneja errores internamente)
    nodo_herramientas = ToolNode(tools=TOOLS)

    # Crear el nodo llamar_modelo con el LLM inyectado
    nodo_modelo = _crear_nodo_modelo(llm_con_tools, model_name)

    # Definir el grafo con MessagesState (lista de mensajes acumulada)
    builder = StateGraph(MessagesState)

    # Registrar nodos
    builder.add_node("llamar_modelo", nodo_modelo)
    builder.add_node("ejecutar_herramienta", nodo_herramientas)

    # Punto de entrada
    builder.set_entry_point("llamar_modelo")

    # Edge condicional: si el LLM generó tool_calls → ejecutar herramienta, si no → END
    builder.add_conditional_edges(
        "llamar_modelo",
        tools_condition,  # devuelve "tools" o END según si hay tool_calls
        {
            "tools": "ejecutar_herramienta",
            END: END,
        },
    )

    # Después de ejecutar la herramienta, siempre volver al modelo
    builder.add_edge("ejecutar_herramienta", "llamar_modelo")

    # Checkpointer para memoria de sesión por thread_id
    memoria = MemorySaver()

    return builder.compile(checkpointer=memoria)


# =============================================================================
# 5. Instancia singleton del grafo (importable por app.py y tests)
# =============================================================================

# Se construye en el momento de importar el módulo.
# Si el .env no está configurado se lanzará RuntimeError con mensaje claro.
grafo = construir_grafo()
