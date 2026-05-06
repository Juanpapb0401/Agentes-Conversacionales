"""
Interfaz Streamlit — Agente Conversacional de Análisis Digital
Rol 4: Agentic Framework Builder

Cómo correr:
    streamlit run app.py

Requiere que el servidor FastAPI esté activo:
    uvicorn main:app --reload
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from services.finops_service import SESSION_CTX, get_session_total, get_global_stats

load_dotenv()

SESSION_BUDGET = float(os.getenv("SESSION_BUDGET", "0.05"))

# =============================================================================
# Configuración de la página
# =============================================================================

st.set_page_config(
    page_title="Agente Conversacional — Reto ICESI",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agente Conversacional")
st.caption("Análisis de conversaciones digitales · Reto ICESI · Powered by Gemini 2.5 Flash Lite + LangGraph")

# =============================================================================
# Carga del grafo (una sola vez por sesión via st.cache_resource)
# =============================================================================

@st.cache_resource(show_spinner="Inicializando el agente...")
def cargar_grafo():
    from agent.graph import construir_grafo
    return construir_grafo()


try:
    grafo = cargar_grafo()
except RuntimeError as e:
    st.error(f"**Error al inicializar el agente:** {e}")
    st.info("Revisa el archivo `.env` y asegúrate de tener `LLM_PROVIDER` y la API key configurada.")
    st.stop()

# =============================================================================
# Estado de la sesión
# =============================================================================

# Generar un thread_id único por sesión de navegador (memoria multi-turno)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Historial de mensajes para renderizar en la UI
if "historial_ui" not in st.session_state:
    st.session_state.historial_ui = []

# Controlar visibilidad de pasos intermedios
if "mostrar_pasos" not in st.session_state:
    st.session_state.mostrar_pasos = False

# =============================================================================
# Sidebar — configuración y ayuda
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    st.session_state.mostrar_pasos = st.toggle(
        "Mostrar pasos intermedios",
        value=st.session_state.mostrar_pasos,
        help="Muestra qué herramienta usó el agente y los datos que retornó.",
    )

    st.divider()

    # FinOps Dashboard
    st.subheader("💰 FinOps Dashboard")
    fin_stats = get_session_total(st.session_state.thread_id)
    pct = fin_stats["total_cost_usd"] / SESSION_BUDGET if SESSION_BUDGET > 0 else 0.0

    st.progress(min(pct, 1.0))
    st.caption(
        f"Sesión: **${fin_stats['total_cost_usd']:.5f}** / ${SESSION_BUDGET:.2f} USD"
    )
    if pct >= 1.0:
        st.error("⛔ Presupuesto agotado")
    elif pct >= 0.8:
        st.warning("⚠️ 80% del presupuesto consumido")

    col1, col2 = st.columns(2)
    col1.metric("Tokens entrada", f"{fin_stats['total_tokens_in']:,}")
    col2.metric("Tokens salida", f"{fin_stats['total_tokens_out']:,}")
    st.caption(f"Llamadas LLM esta sesión: **{fin_stats['calls_count']}**")

    with st.expander("📊 Estadísticas globales"):
        g = get_global_stats()
        st.write(f"Costo total acumulado: **${g['total_cost_usd']:.5f}**")
        st.write(f"Total llamadas históricas: **{g['total_calls']}**")

    st.divider()
    st.subheader("💡 Ejemplos de consultas")
    ejemplos = [
        "¿Quiénes son los usuarios más influyentes?",
        "Analiza el sentimiento: 'El gobierno decepcionó a todos con esta decisión'",
        "¿Qué tan viral fue el mensaje con ID 199219160505_1274366331365120?",
        "Resume esta conversación: ['El gobierno anunció nuevas medidas', 'La gente está molesta', 'Habrá protestas mañana']",
    ]
    for ejemplo in ejemplos:
        if st.button(ejemplo, use_container_width=True, key=f"btn_{ejemplo[:20]}"):
            st.session_state["input_inyectado"] = ejemplo

    st.divider()
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.historial_ui = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")

# =============================================================================
# Renderizar historial de mensajes
# =============================================================================

for turno in st.session_state.historial_ui:
    rol = turno["rol"]
    contenido = turno["contenido"]

    if rol == "user":
        with st.chat_message("user"):
            st.markdown(contenido)

    elif rol == "assistant":
        with st.chat_message("assistant"):
            st.markdown(contenido)

    elif rol == "tool_step" and st.session_state.mostrar_pasos:
        with st.chat_message("assistant", avatar="🔧"):
            with st.expander(f"🔧 Herramienta usada: `{turno['tool_name']}`", expanded=False):
                st.json(turno["tool_output"])
                
                # --- Visualización Especial Rol 3 (Propagación) ---
                if turno["tool_name"] == "tool_analizar_propagacion":
                    try:
                        data = turno["tool_output"]
                        if isinstance(data, str):
                            import json
                            data = json.loads(data)
                        
                        if data.get("encontrado"):
                            st.divider()
                            st.subheader("🔍 Estructura Detectada")
                            c1, c2 = st.columns(2)
                            c1.metric("Arquetipo", data.get("arquetipo", "N/A"))
                            c2.metric("Profundidad", f"{data.get('profundidad_maxima', 0)} niveles")
                            st.info(f"**Score de Impacto:** {data.get('score_impacto', 0)}% — {data.get('nivel_impacto', 'N/A')}")
                    except:
                        pass

# =============================================================================
# Input del usuario
# =============================================================================

# Soporta tanto el st.chat_input como los botones de ejemplo del sidebar
input_from_button = st.session_state.pop("input_inyectado", None)
pregunta = st.chat_input("Escribe tu consulta...") or input_from_button

if pregunta:
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.historial_ui.append({"rol": "user", "contenido": pregunta})

    # Configuración del grafo (thread_id para memoria multi-turno)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # FinOps: verificar presupuesto antes de invocar
    stats_now = get_session_total(st.session_state.thread_id)
    if stats_now["total_cost_usd"] >= SESSION_BUDGET:
        with st.chat_message("assistant"):
            st.error(
                f"⛔ Presupuesto de sesión agotado "
                f"(${stats_now['total_cost_usd']:.5f} / ${SESSION_BUDGET:.2f} USD). "
                "Limpia la conversación para iniciar una nueva sesión."
            )
        st.stop()

    # FinOps: propagar session_id al stack de llamadas (tools → FastAPI → nlp_service)
    SESSION_CTX.set(st.session_state.thread_id)

    # Invocar el grafo con el mensaje del usuario
    with st.spinner("El agente está analizando..."):
        try:
            resultado = grafo.invoke(
                {"messages": [HumanMessage(content=pregunta)]},
                config=config,
            )
        except Exception as e:
            st.error(f"**Error al ejecutar el agente:** {e}")
            st.session_state.historial_ui.append({
                "rol": "assistant",
                "contenido": f"⚠️ Ocurrió un error: {e}",
            })
            st.rerun()

    # Procesar los mensajes del resultado para extraer respuesta final y pasos
    mensajes_respuesta = resultado.get("messages", [])
    respuesta_final = None
    pasos_intermedios = []

    for msg in mensajes_respuesta:
        # Capturar resultados de tools (pasos intermedios)
        if isinstance(msg, ToolMessage):
            pasos_intermedios.append({
                "tool_name": msg.name,
                "tool_output": msg.content,
            })
        # La última AIMessage sin tool_calls es la respuesta final
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            respuesta_final = msg.content

    # Mostrar pasos intermedios si está activado el toggle
    if st.session_state.mostrar_pasos:
        for paso in pasos_intermedios:
            with st.chat_message("assistant", avatar="🔧"):
                with st.expander(f"🔧 Herramienta usada: `{paso['tool_name']}`", expanded=False):
                    st.json(paso["tool_output"])
                    
                    # --- Visualización Especial Rol 3 (Propagación) ---
                    if paso["tool_name"] == "tool_analizar_propagacion":
                        try:
                            import json
                            data = json.loads(paso["tool_output"])
                            if data.get("encontrado"):
                                st.divider()
                                st.subheader("🔍 Estructura Detectada")
                                c1, c2 = st.columns(2)
                                c1.metric("Arquetipo", data.get("arquetipo", "N/A"))
                                c2.metric("Profundidad", f"{data.get('profundidad_maxima', 0)} niveles")
                                st.info(f"**Score de Impacto:** {data.get('score_impacto', 0)}% — {data.get('nivel_impacto', 'N/A')}")
                        except:
                            pass

            st.session_state.historial_ui.append({
                "rol": "tool_step",
                "tool_name": paso["tool_name"],
                "tool_output": paso["tool_output"],
            })

    # Normalizar content a string (Gemini puede devolver lista de partes)
    def _content_str(content) -> str:
        if isinstance(content, list):
            return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content) if content is not None else ""

    # Mostrar la respuesta final del agente
    texto_final = _content_str(respuesta_final)
    if texto_final.strip():
        with st.chat_message("assistant"):
            st.markdown(texto_final)
        st.session_state.historial_ui.append({
            "rol": "assistant",
            "contenido": texto_final,
        })
    else:
        # Intentar re-leer si el modelo retornó contenido vacío tras tool call
        ai_msgs = [m for m in mensajes_respuesta if isinstance(m, AIMessage)]
        contenido_recuperado = next(
            (_content_str(m.content) for m in reversed(ai_msgs) if _content_str(m.content).strip()),
            None,
        )
        fallback_text = contenido_recuperado or "No obtuve una respuesta. Intenta reformular tu consulta."
        with st.chat_message("assistant"):
            st.markdown(fallback_text)
        st.session_state.historial_ui.append({"rol": "assistant", "contenido": fallback_text})
