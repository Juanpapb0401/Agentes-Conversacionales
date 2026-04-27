"""
Ticket 5 — Tests de las Tools del Agente
Verifica que cada tool llama al endpoint correcto y maneja errores de conexión.
Usa mocks de httpx para no depender de la API corriendo.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Helpers de respuesta HTTP mock
# =============================================================================

def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Crea un objeto Response falso que imita httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    return mock_resp


def _mock_connect_error():
    """Simula un fallo de conexión (API no está corriendo)."""
    import httpx
    mock = MagicMock()
    mock.__enter__ = MagicMock(side_effect=httpx.ConnectError("Connection refused"))
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# =============================================================================
# Tests: tool_analizar_sentimiento
# =============================================================================

class TestToolAnalizarSentimiento:

    def test_sentimiento_positivo(self):
        """La tool retorna el JSON del endpoint cuando el texto es válido."""
        payload = {"clima": "positivo", "score": 0.9, "justificacion": "Lenguaje muy favorable."}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_response(payload)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_analizar_sentimiento
            result = tool_analizar_sentimiento.invoke({"text": "¡Este producto es increíble!"})

        assert result["clima"] == "positivo"
        assert result["score"] == 0.9
        assert "justificacion" in result
        # Verificar que llamó al endpoint correcto
        call_args = mock_client.post.call_args
        assert "/analisis/sentimientos" in call_args[0][0]
        assert call_args[1]["json"]["text"] == "¡Este producto es increíble!"

    def test_sentimiento_texto_vacio(self):
        """La tool maneja texto vacío sin llamar a la API."""
        with patch("httpx.Client") as mock_client_cls:
            from agent.tools import tool_analizar_sentimiento
            result = tool_analizar_sentimiento.invoke({"text": ""})

        assert result["clima"] == "neutral"
        assert result["score"] == 0.5
        mock_client_cls.assert_not_called()

    def test_sentimiento_api_no_disponible(self):
        """La tool lanza RuntimeError cuando la API no está corriendo."""
        import httpx
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_analizar_sentimiento
            with pytest.raises(RuntimeError, match="No se puede conectar con la API"):
                tool_analizar_sentimiento.invoke({"text": "Texto de prueba"})


# =============================================================================
# Tests: tool_resumir_conversacion
# =============================================================================

class TestToolResumirConversacion:

    def test_resumen_exitoso(self):
        """La tool retorna resumen, temas y posturas cuando recibe textos válidos."""
        payload = {
            "resumen": "Debate sobre política económica.",
            "temas_principales": ["inflación", "empleo"],
            "posturas_clave": ["A favor de reforma", "En contra de ajuste"],
            "alcance_textos": 3,
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_response(payload)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_resumir_conversacion
            textos = ["El gobierno sube impuestos", "La oposición protesta", "Ciudadanos preocupados"]
            result = tool_resumir_conversacion.invoke({"textos": textos})

        assert "resumen" in result
        assert isinstance(result["temas_principales"], list)
        assert isinstance(result["posturas_clave"], list)
        assert result["alcance_textos"] == 3
        call_args = mock_client.post.call_args
        assert "/analisis/resumen" in call_args[0][0]

    def test_resumen_lista_vacia(self):
        """La tool maneja lista vacía sin llamar a la API."""
        with patch("httpx.Client") as mock_client_cls:
            from agent.tools import tool_resumir_conversacion
            result = tool_resumir_conversacion.invoke({"textos": []})

        assert result["resumen"] == "No se proporcionaron textos para resumir."
        assert result["alcance_textos"] == 0
        mock_client_cls.assert_not_called()

    def test_resumen_filtra_textos_vacios(self):
        """La tool filtra strings vacíos antes de enviar a la API."""
        payload = {
            "resumen": "Un solo comentario.",
            "temas_principales": [],
            "posturas_clave": [],
            "alcance_textos": 1,
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_response(payload)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_resumir_conversacion
            result = tool_resumir_conversacion.invoke({"textos": ["", "  ", "Texto real", ""]})

        sent_body = mock_client.post.call_args[1]["json"]
        assert sent_body["textos"] == ["Texto real"]


# =============================================================================
# Tests: tool_analizar_propagacion
# =============================================================================

class TestToolAnalizarPropagacion:

    def test_propagacion_encontrado(self):
        """La tool retorna métricas completas cuando el post_id existe."""
        payload = {
            "id_original": "post_42",
            "encontrado": True,
            "alcance": 15,
            "replies_directas": 5,
            "cadena_total_nodos": 12,
            "velocidad_media_min": 8.3,
            "velocidad_media_label": "Rápido",
            "porcentaje_contenido_replicado": 45.0,
            "score_impacto": 72.5,
            "nivel_impacto": "Alto",
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = _mock_response(payload)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_analizar_propagacion
            result = tool_analizar_propagacion.invoke({"post_id": "post_42"})

        assert result["encontrado"] is True
        assert result["alcance"] == 15
        assert result["score_impacto"] == 72.5
        # Verificar GET con query param
        call_args = mock_client.get.call_args
        assert "/analisis/propagacion" in call_args[0][0]
        assert call_args[1]["params"]["post_id"] == "post_42"

    def test_propagacion_post_id_vacio(self):
        """La tool maneja post_id vacío sin llamar a la API."""
        with patch("httpx.Client") as mock_client_cls:
            from agent.tools import tool_analizar_propagacion
            result = tool_analizar_propagacion.invoke({"post_id": ""})

        assert result["encontrado"] is False
        assert "post_id válido" in result["error"]
        mock_client_cls.assert_not_called()

    def test_propagacion_post_id_con_espacios(self):
        """La tool limpia espacios del post_id antes de enviarlo."""
        payload = {"id_original": "abc", "encontrado": True, "alcance": 0,
                   "replies_directas": 0, "cadena_total_nodos": 0,
                   "velocidad_media_min": None, "velocidad_media_label": "Sin datos",
                   "porcentaje_contenido_replicado": 0.0, "score_impacto": 0.0,
                   "nivel_impacto": "Nulo"}

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = _mock_response(payload)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_analizar_propagacion
            tool_analizar_propagacion.invoke({"post_id": "  abc  "})

        sent_params = mock_client.get.call_args[1]["params"]
        assert sent_params["post_id"] == "abc"


# =============================================================================
# Tests: tool_analizar_metricas
# =============================================================================

class TestToolAnalizarMetricas:

    def test_metricas_exitoso(self):
        """La tool retorna total_likes y top_influencers correctamente."""
        payload = {
            "total_likes": 48320,
            "top_influencers": ["user_A", "user_B", "user_C", "user_D", "user_E"],
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_response(payload)
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_analizar_metricas
            result = tool_analizar_metricas.invoke({})

        assert result["total_likes"] == 48320
        assert len(result["top_influencers"]) == 5
        call_args = mock_client.post.call_args
        assert "/analisis/metricas" in call_args[0][0]

    def test_metricas_api_error_http(self):
        """La tool lanza RuntimeError con el código HTTP cuando la API falla."""
        import httpx

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="500", request=MagicMock(), response=mock_resp
            )
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            from agent.tools import tool_analizar_metricas
            with pytest.raises(RuntimeError, match="Error HTTP 500"):
                tool_analizar_metricas.invoke({})
