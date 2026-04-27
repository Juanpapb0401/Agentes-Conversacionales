"""
Interfaz Streamlit — Agente Conversacional de Análisis Digital
Rol 4: Agentic Framework Builder

Cómo correr:
    streamlit run app.py

Requiere que el servidor FastAPI esté activo:
    uvicorn main:app --reload
"""

import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

# =============================================================================
# Configuración de la página
# =============================================================================

st.set_page_config(
    page_title="Agente Conversacional — Reto ICESI",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agente Conversacional")
st.caption("Análisis de conversaciones digitales · Reto ICESI · Powered by Gemini 2.5 Flash + LangGraph")

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
    st.subheader("💡 Ejemplos de consultas")
    ejemplos = [
        "¿Quiénes son los usuarios más influyentes?",
        "Analiza el sentimiento de este texto: 'Este producto es increíble, lo recomiendo a todos'",
        "¿Qué tan viral fue el mensaje con ID 1001?",
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
            st.session_state.historial_ui.append({
                "rol": "tool_step",
                "tool_name": paso["tool_name"],
                "tool_output": paso["tool_output"],
            })

    # Mostrar la respuesta final del agente
    if respuesta_final:
        with st.chat_message("assistant"):
            st.markdown(respuesta_final)
        st.session_state.historial_ui.append({
            "rol": "assistant",
            "contenido": respuesta_final,
        })
    else:
        fallback = "No obtuve una respuesta. Intenta reformular tu consulta."
        with st.chat_message("assistant"):
            st.markdown(fallback)
        st.session_state.historial_ui.append({"rol": "assistant", "contenido": fallback})
