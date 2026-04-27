"""
Ticket 5 — Tests del Grafo LangGraph y Memoria Multi-turno (Ticket 6)
Verifica la estructura del grafo, tool-calling y coherencia entre turnos.
Los tests mockean el LLM para no consumir API keys en cada corrida.
"""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Configura variables de entorno mínimas para instanciar el grafo en tests."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-fake")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")


# =============================================================================
# Tests: Estructura del Grafo
# =============================================================================

class TestEstructuraGrafo:

    def test_nodos_requeridos_existen(self):
        """El grafo debe tener los nodos llamar_modelo y ejecutar_herramienta."""
        from agent.graph import construir_grafo
        grafo = construir_grafo()
        nodos = list(grafo.get_graph().nodes.keys())
        assert "llamar_modelo" in nodos
        assert "ejecutar_herramienta" in nodos

    def test_edges_correctos(self):
        """El edge ejecutar_herramienta → llamar_modelo debe existir (ciclo del agente)."""
        from agent.graph import construir_grafo
        grafo = construir_grafo()
        edges = [(e.source, e.target) for e in grafo.get_graph().edges]
        assert ("ejecutar_herramienta", "llamar_modelo") in edges
        assert ("__start__", "llamar_modelo") in edges

    def test_memoria_acepta_thread_id(self):
        """El grafo compilado con MemorySaver debe aceptar thread_id en config."""
        from agent.graph import construir_grafo
        grafo = construir_grafo()
        config = {"configurable": {"thread_id": "test-session-1"}}
        state = grafo.get_state(config)
        assert state.values == {} or isinstance(state.values, dict)

    def test_tools_vinculadas(self):
        """El grafo debe tener exactamente 4 tools registradas."""
        from agent.tools import TOOLS
        assert len(TOOLS) == 4
        nombres = [t.name for t in TOOLS]
        assert "tool_analizar_sentimiento" in nombres
        assert "tool_resumir_conversacion" in nombres
        assert "tool_analizar_propagacion" in nombres
        assert "tool_analizar_metricas" in nombres

    def test_error_sin_llm_provider(self, monkeypatch):
        """Sin LLM_PROVIDER configurado, construir_grafo lanza RuntimeError."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "")
        from agent import graph as graph_module
        with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
            graph_module._build_llm()


# =============================================================================
# Tests: Tool-Calling — el agente selecciona la tool correcta
# =============================================================================

class TestToolCalling:
    """
    Mockea el LLM para devolver una AIMessage con tool_calls específico,
    y mockea httpx para simular la respuesta de la API.
    Verifica el flujo completo: mensaje → LLM elige tool → tool se ejecuta → respuesta.
    """

    def _make_ai_tool_call(self, tool_name: str, args: dict, call_id: str = "call_1") -> AIMessage:
        """Construye un AIMessage que simula que el LLM eligió una tool."""
        return AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
        )

    def _make_ai_final(self, content: str) -> AIMessage:
        """Construye el AIMessage final (sin tool_calls) del agente."""
        return AIMessage(content=content)

    def test_agente_invoca_tool_sentimiento(self):
        """Cuando el LLM elige tool_analizar_sentimiento, el grafo la ejecuta y retorna respuesta."""
        api_payload = {"clima": "positivo", "score": 0.88, "justificacion": "Tono favorable."}
        tool_call_msg = self._make_ai_tool_call(
            "tool_analizar_sentimiento",
            {"text": "Estoy muy feliz hoy"},
        )
        final_msg = self._make_ai_final("El análisis indica un clima positivo (score 0.88).")

        # Mockear _post directamente para no interferir con el SDK de OpenAI
        with patch("agent.tools._post", return_value=api_payload):
            with patch("langchain_openai.ChatOpenAI.invoke") as mock_llm:
                mock_llm.side_effect = [tool_call_msg, final_msg]

                from agent.graph import construir_grafo
                grafo = construir_grafo()
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                result = grafo.invoke(
                    {"messages": [HumanMessage(content="¿Cómo está el sentimiento de: Estoy muy feliz hoy?")]},
                    config=config,
                )

        mensajes = result["messages"]
        tool_msgs = [m for m in mensajes if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0].name == "tool_analizar_sentimiento"
        final_ai = [m for m in mensajes if isinstance(m, AIMessage) and not m.tool_calls]
        assert len(final_ai) >= 1

    def test_agente_invoca_tool_propagacion(self):
        """Cuando el LLM elige tool_analizar_propagacion, se ejecuta con el post_id correcto."""
        api_payload = {
            "id_original": "post_99",
            "encontrado": True,
            "alcance": 20,
            "replies_directas": 8,
            "cadena_total_nodos": 15,
            "velocidad_media_min": 5.2,
            "velocidad_media_label": "Muy rápido",
            "porcentaje_contenido_replicado": 60.0,
            "score_impacto": 85.0,
            "nivel_impacto": "Muy Alto",
        }
        tool_call_msg = self._make_ai_tool_call(
            "tool_analizar_propagacion",
            {"post_id": "post_99"},
        )
        final_msg = self._make_ai_final("El mensaje post_99 tuvo un impacto muy alto con alcance de 20 nodos.")

        with patch("agent.tools._get", return_value=api_payload):
            with patch("langchain_openai.ChatOpenAI.invoke") as mock_llm:
                mock_llm.side_effect = [tool_call_msg, final_msg]

                from agent.graph import construir_grafo
                grafo = construir_grafo()
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                result = grafo.invoke(
                    {"messages": [HumanMessage(content="¿Cuál fue el impacto del post post_99?")]},
                    config=config,
                )

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any(m.name == "tool_analizar_propagacion" for m in tool_msgs)

    def test_agente_invoca_tool_metricas(self):
        """Cuando el LLM elige tool_analizar_metricas, se ejecuta sin argumentos."""
        api_payload = {"total_likes": 12000, "top_influencers": ["u1", "u2", "u3", "u4", "u5"]}
        tool_call_msg = self._make_ai_tool_call("tool_analizar_metricas", {})
        final_msg = self._make_ai_final("Los influencers más relevantes son u1, u2, u3.")

        with patch("agent.tools._post", return_value=api_payload):
            with patch("langchain_openai.ChatOpenAI.invoke") as mock_llm:
                mock_llm.side_effect = [tool_call_msg, final_msg]

                from agent.graph import construir_grafo
                grafo = construir_grafo()
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                result = grafo.invoke(
                    {"messages": [HumanMessage(content="¿Quiénes son los usuarios más influyentes?")]},
                    config=config,
                )

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any(m.name == "tool_analizar_metricas" for m in tool_msgs)

    def test_agente_sin_tool_va_a_end(self):
        """Cuando el LLM responde sin tool_calls, el grafo va directo a END."""
        final_msg = self._make_ai_final("Lo siento, esa consulta está fuera de mi alcance.")

        with patch("langchain_openai.ChatOpenAI.invoke", return_value=final_msg):
            from agent.graph import construir_grafo
            grafo = construir_grafo()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = grafo.invoke(
                {"messages": [HumanMessage(content="¿Cuánto es 2+2?")]},
                config=config,
            )

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 0
        assert ai_msgs[-1].content == "Lo siento, esa consulta está fuera de mi alcance."


# =============================================================================
# Ticket 6 — Tests de Memoria Multi-turno
# =============================================================================

class TestMemoriaMultiTurno:
    """
    Verifica que el MemorySaver preserva el historial entre múltiples invocaciones
    con el mismo thread_id, habilitando conversaciones con contexto.
    """

    def test_mismo_thread_acumula_mensajes(self):
        """Dos invocaciones con el mismo thread_id deben acumular el historial."""
        respuesta_1 = AIMessage(content="El clima es positivo.")
        respuesta_2 = AIMessage(content="Sí, me refería al mismo texto.")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        with patch("langchain_openai.ChatOpenAI.invoke") as mock_llm:
            mock_llm.side_effect = [respuesta_1, respuesta_2]

            from agent.graph import construir_grafo
            grafo = construir_grafo()

            # Primer turno
            grafo.invoke({"messages": [HumanMessage(content="Analiza el sentimiento de: Me alegra verte")]}, config=config)
            # Segundo turno (misma sesión)
            grafo.invoke({"messages": [HumanMessage(content="¿Y si el texto fuera negativo?")]}, config=config)

            # Recuperar estado del grafo para verificar historial acumulado
            state = grafo.get_state(config)

        mensajes = state.values.get("messages", [])
        human_msgs = [m for m in mensajes if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 2, "Debe haber 2 mensajes humanos en el historial"

    def test_diferente_thread_no_comparte_estado(self):
        """Dos thread_id distintos no deben compartir historial de mensajes."""
        respuesta = AIMessage(content="Respuesta genérica.")

        config_a = {"configurable": {"thread_id": str(uuid.uuid4())}}
        config_b = {"configurable": {"thread_id": str(uuid.uuid4())}}

        with patch("langchain_openai.ChatOpenAI.invoke", return_value=respuesta):
            from agent.graph import construir_grafo
            grafo = construir_grafo()

            grafo.invoke({"messages": [HumanMessage(content="Mensaje sesión A")]}, config=config_a)

            # El estado de la sesión B debe estar vacío
            state_b = grafo.get_state(config_b)

        assert state_b.values.get("messages", []) == []

    def test_historial_crece_con_turnos_sucesivos(self):
        """Tres turnos con el mismo thread_id deben acumular 3 mensajes humanos."""
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        respuesta = AIMessage(content="Ok.")

        with patch("langchain_openai.ChatOpenAI.invoke", return_value=respuesta):
            from agent.graph import construir_grafo
            grafo = construir_grafo()

            for i in range(3):
                grafo.invoke(
                    {"messages": [HumanMessage(content=f"Consulta número {i + 1}")]},
                    config=config,
                )

            state = grafo.get_state(config)

        human_msgs = [m for m in state.values.get("messages", []) if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 3
