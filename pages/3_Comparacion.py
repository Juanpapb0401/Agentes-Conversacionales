"""
Comparador de Modelos IA — Side-by-Side
Página 3 del sistema multi-página de Streamlit.

Compara respuestas de hasta 4 modelos simultáneamente en tres modos:
  1. Análisis de sentimiento (salida estructurada: clima, score, justificación)
  2. Resumen de conversación (salida estructurada: resumen, temas, posturas)
  3. Prompt libre (texto libre, sin restricciones de formato)

Los modelos disponibles se detectan automáticamente según las API keys configuradas en .env.
"""

import os
import time
import json
from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from services.nlp_service import call_model_direct, _safe_json_parse
from services.finops_service import calculate_cost

load_dotenv()

st.set_page_config(
    page_title="Comparador de Modelos IA",
    page_icon="⚖️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# REGISTRO DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: List[Dict[str, Any]] = [
    # Google Gemini — GRATIS (Google AI Studio: 500 req/día, 15 RPM)
    {"provider": "gemini",    "id": "gemini-2.5-flash",           "label": "Gemini 2.5 Flash",       "color": "#4285F4", "env_key": "GEMINI_API_KEY",    "free": True},
    {"provider": "gemini",    "id": "gemini-2.0-flash",           "label": "Gemini 2.0 Flash",       "color": "#4285F4", "env_key": "GEMINI_API_KEY",    "free": True},
    {"provider": "gemini",    "id": "gemini-1.5-flash",           "label": "Gemini 1.5 Flash",       "color": "#4285F4", "env_key": "GEMINI_API_KEY",    "free": True},
    # Groq — GRATIS (tier gratuito generoso, sin tarjeta)
    {"provider": "groq",      "id": "llama-3.3-70b-versatile",   "label": "LLaMA 3.3 70B (Groq)",  "color": "#F97316", "env_key": "GROQ_API_KEY",      "free": True},
    {"provider": "groq",      "id": "llama-3.1-8b-instant",      "label": "LLaMA 3.1 8B (Groq)",   "color": "#F97316", "env_key": "GROQ_API_KEY",      "free": True},
    {"provider": "groq",      "id": "mixtral-8x7b-32768",        "label": "Mixtral 8×7B (Groq)",   "color": "#F97316", "env_key": "GROQ_API_KEY",      "free": True},
    # OpenAI — PAGO
    {"provider": "openai",    "id": "gpt-4o-mini",               "label": "GPT-4o Mini",            "color": "#00A67E", "env_key": "OPENAI_API_KEY",    "free": False},
    {"provider": "openai",    "id": "gpt-4o",                    "label": "GPT-4o",                 "color": "#00A67E", "env_key": "OPENAI_API_KEY",    "free": False},
    # Anthropic Claude — PAGO
    {"provider": "anthropic", "id": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku",       "color": "#D97706", "env_key": "ANTHROPIC_API_KEY", "free": False},
    {"provider": "anthropic", "id": "claude-3-5-sonnet-20241022","label": "Claude 3.5 Sonnet",      "color": "#D97706", "env_key": "ANTHROPIC_API_KEY", "free": False},
    {"provider": "anthropic", "id": "claude-opus-4-5",           "label": "Claude Opus 4.5",        "color": "#D97706", "env_key": "ANTHROPIC_API_KEY", "free": False},
]

_PAID_PLACEHOLDERS = {
    "sk-your-openai-key-here",
    "sk-ant-your-anthropic-key-here",
    "gsk_your-groq-key-here",
    "",
}

def _available_models() -> List[Dict[str, Any]]:
    """Devuelve solo los modelos cuya API key esté configurada y no sea un placeholder."""
    available = []
    for m in MODEL_REGISTRY:
        key_val = os.environ.get(m["env_key"], "")
        if key_val and key_val not in _PAID_PLACEHOLDERS:
            available.append(m)
    return available


def _provider_badge(provider: str, color: str) -> str:
    labels = {"gemini": "Google", "openai": "OpenAI", "anthropic": "Anthropic", "groq": "Groq"}
    return f":{labels.get(provider, provider)}"


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS POR MODO
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_SENTIMIENTO = (
    "Eres un analista de sentimientos para conversaciones digitales. "
    "Devuelve SOLO JSON valido en español con exactamente estas claves: clima, score, justificacion. "
    "clima debe ser: positivo, negativo o neutral. score debe estar entre 0.0 y 1.0 (float). "
    "No incluyas markdown ni texto adicional."
)

_SYSTEM_RESUMEN = (
    "Eres un analista de conversaciones digitales. "
    "Devuelve SOLO JSON valido en español con exactamente estas claves: "
    "resumen (string), temas_principales (lista de strings), posturas_clave (lista de strings). "
    "No incluyas markdown ni texto adicional."
)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.title("⚖️ Comparador de Modelos IA")
st.caption("Ejecuta el mismo prompt en múltiples modelos y compara respuestas, velocidad y costo side-by-side.")

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO DE PROVEEDORES
# ──────────────────────────────────────────────────────────────────────────────

available = _available_models()
all_providers = {"gemini", "openai", "anthropic", "groq"}
configured_providers = {m["provider"] for m in available}

with st.expander("🔑 Estado de proveedores configurados", expanded=len(available) == 0):
    provider_info = {
        "gemini":    ("Google Gemini",    "GEMINI_API_KEY",    "https://aistudio.google.com/apikey"),
        "openai":    ("OpenAI",           "OPENAI_API_KEY",    "https://platform.openai.com/api-keys"),
        "anthropic": ("Anthropic Claude", "ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
        "groq":      ("Groq",             "GROQ_API_KEY",      "https://console.groq.com/keys"),
    }
    cols = st.columns(4)
    for i, (prov, (name, env_key, url)) in enumerate(provider_info.items()):
        with cols[i]:
            if prov in configured_providers:
                st.success(f"✅ **{name}**\n\nConfigurado")
            else:
                st.error(f"❌ **{name}**\n\n`{env_key}` no configurada\n\n[Obtener key]({url})")

if not available:
    st.warning("⚠️ No hay ningún proveedor configurado. Agrega al menos una API key en el archivo `.env` y reinicia Streamlit.")
    st.stop()

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL EXPERIMENTO
# ──────────────────────────────────────────────────────────────────────────────

col_cfg1, col_cfg2 = st.columns([1, 2])

with col_cfg1:
    modo = st.radio(
        "Tipo de análisis",
        options=["Sentimiento", "Resumen", "Prompt libre"],
        index=0,
        help=(
            "**Sentimiento**: compara clima emocional, score y justificación.\n\n"
            "**Resumen**: compara resumen, temas y posturas.\n\n"
            "**Prompt libre**: respuesta de texto libre sin formato forzado."
        ),
    )

with col_cfg2:
    model_options = {
        f"{'🟢 GRATIS' if m['free'] else '🔴 PAGO'} · {m['label']} ({m['provider']})": m
        for m in available
    }
    seleccionados_labels = st.multiselect(
        "Modelos a comparar (máx. 4)",
        options=list(model_options.keys()),
        default=list(model_options.keys())[:min(2, len(model_options))],
        max_selections=4,
        help="Solo aparecen modelos con API key configurada en `.env`.",
    )
    modelos_sel = [model_options[l] for l in seleccionados_labels]

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# INPUT DEL PROMPT
# ──────────────────────────────────────────────────────────────────────────────

if modo == "Sentimiento":
    texto_input = st.text_area(
        "Texto a analizar",
        height=120,
        placeholder="Ej: El gobierno decepcionó a todos con esta decisión. La ciudadanía está furiosa.",
        help="El mismo texto se enviará a todos los modelos seleccionados.",
    )
    user_prompt_base = f"Texto a analizar:\n{texto_input}\n\nResponde solo JSON."
    system_prompt    = _SYSTEM_SENTIMIENTO
    json_mode        = True
    max_tokens       = 300

elif modo == "Resumen":
    textos_raw = st.text_area(
        "Textos a resumir (uno por línea)",
        height=160,
        placeholder="El gobierno anunció nuevas medidas.\nLa gente está molesta con la reforma.\nHabrá protestas mañana en la capital.",
        help="Cada línea se trata como un mensaje/comentario separado.",
    )
    textos_lista = [t.strip() for t in textos_raw.splitlines() if t.strip()]
    contenido    = "\n---\n".join(textos_lista)
    user_prompt_base = (
        f"Textos de la conversación:\n{contenido}\n\n"
        "Responde solo JSON con resumen, temas_principales y posturas_clave."
    )
    system_prompt = _SYSTEM_RESUMEN
    json_mode     = True
    max_tokens    = 500

else:  # Prompt libre
    col_sys, col_usr = st.columns([1, 2])
    with col_sys:
        system_prompt = st.text_area(
            "System prompt (opcional)",
            height=100,
            value="Eres un asistente experto en análisis de redes sociales y conversaciones digitales.",
            help="Instrucción de sistema que recibirán todos los modelos.",
        )
    with col_usr:
        user_prompt_base = st.text_area(
            "Pregunta o instrucción",
            height=100,
            placeholder="Ej: ¿Cuáles son las mejores estrategias para detectar desinformación en redes sociales?",
        )
    json_mode  = False
    max_tokens = 800

# ──────────────────────────────────────────────────────────────────────────────
# BOTÓN DE COMPARACIÓN
# ──────────────────────────────────────────────────────────────────────────────

input_valido = bool(user_prompt_base.strip()) and len(modelos_sel) >= 1
comparar_btn = st.button(
    f"⚡ Comparar {len(modelos_sel)} modelo{'s' if len(modelos_sel) != 1 else ''}",
    type="primary",
    disabled=not input_valido,
    use_container_width=True,
)

if not input_valido and not comparar_btn:
    if not modelos_sel:
        st.info("Selecciona al menos un modelo para comparar.")
    elif not user_prompt_base.strip():
        st.info("Escribe el texto o prompt que quieres analizar.")

# ──────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN Y RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────

if comparar_btn and input_valido:
    st.divider()
    st.subheader("📊 Resultados de la Comparación")

    resultados: Dict[str, Any] = {}
    progreso = st.progress(0, text="Iniciando comparación…")

    for i, modelo in enumerate(modelos_sel):
        progreso.progress(
            (i) / len(modelos_sel),
            text=f"Consultando **{modelo['label']}** ({i+1}/{len(modelos_sel)})…",
        )
        t_inicio = time.perf_counter()
        try:
            raw = call_model_direct(
                system_prompt=system_prompt,
                user_prompt=user_prompt_base,
                provider=modelo["provider"],
                model=modelo["id"],
                max_tokens=max_tokens,
                session_id="comparacion",
                service="comparacion",
                json_mode=json_mode,
            )
            t_fin = time.perf_counter()
            resultados[modelo["id"]] = {
                "ok":       True,
                "raw":      raw,
                "parsed":   _safe_json_parse(raw) if json_mode else None,
                "tiempo_s": round(t_fin - t_inicio, 2),
                "modelo":   modelo,
            }
        except Exception as exc:
            t_fin = time.perf_counter()
            resultados[modelo["id"]] = {
                "ok":       False,
                "error":    str(exc),
                "tiempo_s": round(t_fin - t_inicio, 2),
                "modelo":   modelo,
            }

    progreso.progress(1.0, text="✅ Comparación completada")

    # ── Columnas de resultados ────────────────────────────────────────────────
    cols_res = st.columns(len(modelos_sel))
    scores_chart: Dict[str, float] = {}
    tiempos_chart: Dict[str, float] = {}

    for col, modelo in zip(cols_res, modelos_sel):
        mid   = modelo["id"]
        color = modelo["color"]
        res   = resultados[mid]
        tiempos_chart[modelo["label"]] = res["tiempo_s"]

        with col:
            # Header del modelo
            badge_text  = "🟢 GRATIS" if modelo["free"] else "🔴 PAGO"
            badge_color = "#16a34a"  if modelo["free"] else "#dc2626"
            st.markdown(
                f"<div style='background:{color}22; border-left:4px solid {color}; "
                f"padding:10px 14px; border-radius:6px; margin-bottom:10px;'>"
                f"<strong style='color:{color}'>{modelo['label']}</strong> "
                f"<span style='font-size:0.72em; font-weight:600; color:{badge_color}; "
                f"background:{badge_color}18; padding:2px 6px; border-radius:999px;'>"
                f"{badge_text}</span><br>"
                f"<small style='color:#888'>{modelo['provider']} · {mid}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if not res["ok"]:
                st.error(f"❌ Error: {res['error']}")
                st.caption(f"⏱ {res['tiempo_s']}s")
                continue

            st.caption(f"⏱ **{res['tiempo_s']}s**")

            # ── Modo Sentimiento ──────────────────────────────────────────
            if modo == "Sentimiento":
                p = res.get("parsed") or {}
                clima     = str(p.get("clima", "?")).lower()
                score     = float(p.get("score", 0.0))
                just      = str(p.get("justificacion", res["raw"]))
                scores_chart[modelo["label"]] = score

                emoji_map = {"positivo": "🟢", "negativo": "🔴", "neutral": "⚪"}
                color_map = {"positivo": "green", "negativo": "red", "neutral": "gray"}
                clima_color = color_map.get(clima, "black")
                st.markdown(
                    f"**Clima:** {emoji_map.get(clima, '?')} "
                    f"<span style='color:{clima_color}'>{clima.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(score, text=f"Score: {score:.3f}")
                st.markdown(f"**Justificación:**\n\n{just}")

                if not p:
                    with st.expander("Ver respuesta cruda"):
                        st.code(res["raw"], language="json")

            # ── Modo Resumen ──────────────────────────────────────────────
            elif modo == "Resumen":
                p = res.get("parsed") or {}
                resumen  = str(p.get("resumen",        res["raw"]))
                temas    = list(p.get("temas_principales", []))
                posturas = list(p.get("posturas_clave",    []))
                n_words  = len(resumen.split())
                scores_chart[modelo["label"]] = float(n_words)

                st.markdown(f"**Resumen** ({n_words} palabras):")
                st.markdown(f"> {resumen}")

                if temas:
                    st.markdown("**Temas:**")
                    st.markdown(" ".join(f"`{t}`" for t in temas))

                if posturas:
                    st.markdown("**Posturas:**")
                    for p_item in posturas:
                        st.markdown(f"• {p_item}")

                if not p:
                    with st.expander("Ver respuesta cruda"):
                        st.code(res["raw"])

            # ── Prompt libre ──────────────────────────────────────────────
            else:
                respuesta  = res["raw"]
                n_words    = len(respuesta.split())
                n_chars    = len(respuesta)
                scores_chart[modelo["label"]] = float(n_words)

                st.markdown(f"**Palabras:** {n_words} · **Chars:** {n_chars}")
                st.markdown(respuesta)

    # ── Gráficas comparativas ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Comparación Visual")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        if scores_chart:
            if modo == "Sentimiento":
                chart_title = "Score de Sentimiento por Modelo (0=neg, 1=pos)"
                y_label     = "Score"
            elif modo == "Resumen":
                chart_title = "Longitud del Resumen (palabras)"
                y_label     = "Palabras"
            else:
                chart_title = "Longitud de Respuesta (palabras)"
                y_label     = "Palabras"

            models_list  = list(scores_chart.keys())
            values_list  = list(scores_chart.values())
            colors_list  = [modelo["color"] for modelo in modelos_sel if modelo["label"] in models_list]

            fig_scores = px.bar(
                x=models_list,
                y=values_list,
                color=models_list,
                color_discrete_sequence=colors_list,
                title=chart_title,
                labels={"x": "Modelo", "y": y_label},
                text=[f"{v:.3f}" if modo == "Sentimiento" else str(int(v)) for v in values_list],
            )
            fig_scores.update_traces(textposition="outside")
            if modo == "Sentimiento":
                fig_scores.add_hline(y=0.5, line_dash="dot", line_color="gray",
                                     annotation_text="Neutro (0.5)")
                fig_scores.update_yaxes(range=[0, 1.15])
            fig_scores.update_layout(height=320, margin=dict(t=50, b=20, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_scores, use_container_width=True)

    with col_chart2:
        if tiempos_chart:
            models_t = list(tiempos_chart.keys())
            times_t  = list(tiempos_chart.values())
            colors_t = [modelo["color"] for modelo in modelos_sel if modelo["label"] in models_t]

            fig_time = px.bar(
                x=models_t,
                y=times_t,
                color=models_t,
                color_discrete_sequence=colors_t,
                title="Tiempo de Respuesta por Modelo (segundos)",
                labels={"x": "Modelo", "y": "Segundos"},
                text=[f"{t:.2f}s" for t in times_t],
            )
            fig_time.update_traces(textposition="outside")
            fig_time.update_layout(height=320, margin=dict(t=50, b=20, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tabla Resumen")

    tabla_rows = []
    for modelo in modelos_sel:
        mid = modelo["id"]
        res = resultados[mid]
        fila: Dict[str, Any] = {
            "Modelo":    f"{'🟢' if modelo['free'] else '🔴'} {modelo['label']}",
            "Proveedor": modelo["provider"].capitalize(),
            "Estado":    "✅ OK" if res["ok"] else "❌ Error",
            "Tiempo (s)": res["tiempo_s"],
        }
        if res["ok"]:
            if modo == "Sentimiento":
                p = res.get("parsed") or {}
                fila["Clima"]   = p.get("clima", "?")
                fila["Score"]   = f"{p.get('score', 0):.3f}"
                fila["Acuerdo"] = "—"
            elif modo == "Resumen":
                p = res.get("parsed") or {}
                fila["Palabras resumen"] = len(str(p.get("resumen", res["raw"])).split())
                fila["Temas detectados"] = len(p.get("temas_principales", []))
            else:
                fila["Palabras"] = len(res["raw"].split())
                fila["Chars"]    = len(res["raw"])
        tabla_rows.append(fila)

    import pandas as pd
    df_tabla = pd.DataFrame(tabla_rows)
    st.dataframe(df_tabla, use_container_width=True, hide_index=True)

    # ── Consenso (solo Sentimiento) ───────────────────────────────────────────
    if modo == "Sentimiento" and len(modelos_sel) >= 2:
        st.divider()
        st.subheader("🤝 Análisis de Consenso")

        climas_validos = []
        for modelo in modelos_sel:
            res = resultados[modelo["id"]]
            if res["ok"]:
                p = res.get("parsed") or {}
                c = str(p.get("clima", "")).lower()
                if c in ("positivo", "negativo", "neutral"):
                    climas_validos.append(c)

        if climas_validos:
            from collections import Counter
            conteo = Counter(climas_validos)
            clima_mayoría = conteo.most_common(1)[0][0]
            n_acuerdo     = conteo[clima_mayoría]
            pct_acuerdo   = round(n_acuerdo / len(climas_validos) * 100)

            col_cons1, col_cons2, col_cons3 = st.columns(3)
            col_cons1.metric("Clima mayoritario", clima_mayoría.upper())
            col_cons2.metric("Modelos en acuerdo", f"{n_acuerdo}/{len(climas_validos)}")
            col_cons3.metric("Nivel de consenso", f"{pct_acuerdo}%")

            if pct_acuerdo == 100:
                st.success(f"✅ **Consenso total**: todos los modelos coinciden en que el texto es **{clima_mayoría}**.")
            elif pct_acuerdo >= 60:
                st.warning(f"⚠️ **Consenso parcial** ({pct_acuerdo}%): la mayoría dice **{clima_mayoría}**, pero hay discrepancia.")
            else:
                st.error("❌ **Sin consenso**: los modelos tienen opiniones muy distintas sobre el sentimiento.")

            if len(scores_chart) >= 2:
                scores_vals = list(scores_chart.values())
                rango = max(scores_vals) - min(scores_vals)
                if rango > 0.3:
                    st.warning(f"📊 Rango de scores: {min(scores_vals):.3f} – {max(scores_vals):.3f} (diferencia de {rango:.3f}). Los modelos valoran la intensidad de forma diferente.")
                else:
                    st.success(f"📊 Rango de scores compacto: {min(scores_vals):.3f} – {max(scores_vals):.3f}. Los modelos son consistentes en la intensidad.")

    # ── Exportar JSON ─────────────────────────────────────────────────────────
    st.divider()
    export_data = {
        "modo":    modo,
        "prompt":  user_prompt_base,
        "modelos": [
            {
                "modelo":    m["label"],
                "proveedor": m["provider"],
                "tiempo_s":  resultados[m["id"]]["tiempo_s"],
                "ok":        resultados[m["id"]]["ok"],
                "respuesta": resultados[m["id"]].get("parsed") or resultados[m["id"]].get("raw", ""),
                "error":     resultados[m["id"]].get("error"),
            }
            for m in modelos_sel
        ],
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "📥 Descargar resultados JSON",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name=f"comparacion_{modo.lower()}_{ts}.json",
        mime="application/json",
    )
