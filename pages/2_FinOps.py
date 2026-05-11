"""
FinOps Command Center — Análisis de Costos y Eficiencia del Agente IA
Página 2 del sistema multi-página de Streamlit.

Secciones:
  1. KPIs globales
  2. Timeline de costos acumulados
  3. Inteligencia por servicio
  4. Análisis por modelo LLM
  5. Session Leaderboard
  6. Recomendaciones automáticas de optimización
  7. Proyector de presupuesto interactivo
  8. Exportar reporte
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from services.finops_service import get_cost_forecast, get_rich_report

load_dotenv()

SESSION_BUDGET = float(os.getenv("SESSION_BUDGET", "0.05"))

st.set_page_config(
    page_title="FinOps Command Center",
    page_icon="💰",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_SERVICE_COLORS = {
    "agente":      "#3498db",
    "sentimiento": "#e74c3c",
    "resumen":     "#2ecc71",
    "propagacion": "#f39c12",
    "metricas":    "#9b59b6",
    "scraping":    "#1abc9c",
    "unknown":     "#95a5a6",
}

def _svc_color(svc: str) -> str:
    for k, v in _SERVICE_COLORS.items():
        if k in svc.lower():
            return v
    return "#95a5a6"


def _generate_insights(
    service_stats: Dict[str, Any],
    model_stats: Dict[str, Any],
    session_eff: Dict[str, Any],
    all_calls: List[Dict],
    budget: float,
) -> List[Dict]:
    """Genera recomendaciones automáticas basadas en los datos reales del log."""
    insights: List[Dict] = []

    if not all_calls:
        return []

    total_cost_all = sum(s["cost_usd"] for s in service_stats.values()) if service_stats else 0.0

    # 1. Servicio más costoso
    if service_stats and total_cost_all > 0:
        most_exp_svc, most_exp_data = max(service_stats.items(), key=lambda x: x[1]["cost_usd"])
        pct = round(most_exp_data["cost_usd"] / total_cost_all * 100)
        insights.append({
            "type": "warning", "icon": "💸",
            "title": f"'{most_exp_svc}' acapara el {pct}% del gasto total",
            "detail": (
                f"${most_exp_data['cost_usd']:.5f} en {most_exp_data['calls']} llamadas "
                f"(${most_exp_data.get('avg_cost_usd', 0):.7f}/llamada). "
                "Evalúa si la frecuencia de uso es la adecuada."
            ),
        })

    # 2. Servicio más eficiente en tokens
    eff_candidates = [(s, d) for s, d in service_stats.items() if d.get("efficiency_ratio", 0) > 0]
    if len(eff_candidates) >= 1:
        best_eff_svc, best_eff_data = max(eff_candidates, key=lambda x: x[1]["efficiency_ratio"])
        insights.append({
            "type": "success", "icon": "✅",
            "title": f"'{best_eff_svc}' tiene la mejor eficiencia de tokens",
            "detail": (
                f"Por cada 100 tokens enviados genera "
                f"{best_eff_data['efficiency_ratio'] * 100:.0f} tokens de respuesta "
                f"(ratio {best_eff_data['efficiency_ratio']:.3f})."
            ),
        })

    # 3. Servicios con baja eficiencia (ratio < 0.4 y ≥3 llamadas)
    for svc, data in service_stats.items():
        if data.get("efficiency_ratio", 1.0) < 0.4 and data["calls"] >= 3:
            insights.append({
                "type": "warning", "icon": "⚠️",
                "title": f"Baja eficiencia en '{svc}'",
                "detail": (
                    f"Ratio output/input: {data['efficiency_ratio']:.2f}. "
                    f"Promedio de {data.get('avg_tokens_in', 0):.0f} tokens de entrada "
                    "generan poco contexto de salida. Considera acortar los prompts."
                ),
            })

    # 4. Salud del presupuesto global
    pct_budget = total_cost_all / budget * 100 if budget > 0 else 0
    if pct_budget >= 100:
        insights.append({
            "type": "error", "icon": "🔴",
            "title": "Presupuesto de sesión AGOTADO",
            "detail": f"Gasto acumulado ({pct_budget:.1f}%) supera el límite de ${budget:.3f}.",
        })
    elif pct_budget >= 80:
        insights.append({
            "type": "error", "icon": "🟠",
            "title": "Presupuesto en zona crítica",
            "detail": f"Has consumido el {pct_budget:.1f}% del presupuesto de sesión (${budget:.3f}).",
        })
    elif pct_budget >= 50:
        insights.append({
            "type": "warning", "icon": "🟡",
            "title": "Presupuesto en zona de alerta",
            "detail": f"{pct_budget:.1f}% consumido — quedan ${budget - total_cost_all:.5f} disponibles.",
        })
    else:
        insights.append({
            "type": "success", "icon": "💚",
            "title": "Presupuesto saludable",
            "detail": f"Solo {pct_budget:.1f}% consumido — amplio margen de ${budget - total_cost_all:.5f} restante.",
        })

    # 5. Llamada atípicamente costosa
    costs = [c.get("cost_usd", 0) for c in all_calls]
    if len(costs) > 2:
        avg_c = sum(costs) / len(costs)
        max_c = max(costs)
        if max_c > avg_c * 3:
            insights.append({
                "type": "warning", "icon": "📍",
                "title": "Llamada atípicamente costosa detectada",
                "detail": (
                    f"La llamada más cara (${max_c:.6f}) costó "
                    f"{max_c / avg_c:.1f}× el promedio (${avg_c:.6f}). "
                    "Puede indicar un prompt excepcionalmente largo."
                ),
            })

    # 6. Prompts muy largos
    if service_stats:
        max_ti_svc, max_ti_data = max(service_stats.items(), key=lambda x: x[1].get("avg_tokens_in", 0))
        if max_ti_data.get("avg_tokens_in", 0) > 200:
            insights.append({
                "type": "info", "icon": "💡",
                "title": f"'{max_ti_svc}' usa prompts de entrada largos",
                "detail": (
                    f"Promedio de {max_ti_data['avg_tokens_in']:.0f} tokens de entrada. "
                    "Considera resumir el contexto enviado para reducir costos."
                ),
            })

    # 7. Evaluación del modelo
    if model_stats:
        used = list(model_stats.keys())
        flash_models = [m for m in used if "flash" in m.lower() or "mini" in m.lower()]
        if flash_models:
            insights.append({
                "type": "success", "icon": "🤖",
                "title": "Modelos eficientes seleccionados",
                "detail": (
                    f"{', '.join(flash_models)} ofrecen excelente relación calidad-precio "
                    "con precios tier 'flash' / 'mini'."
                ),
            })
        else:
            insights.append({
                "type": "info", "icon": "🤖",
                "title": "Considera modelos de menor costo",
                "detail": (
                    f"Usas: {', '.join(used)}. Modelos como gemini-2.5-flash o gpt-4o-mini "
                    "ofrecen buena calidad a ~10× menor costo."
                ),
            })

    # 8. Sesión con mejor y peor eficiencia
    if len(session_eff) >= 2:
        best_sid = max(session_eff, key=lambda s: session_eff[s]["efficiency_score"])
        worst_sid = min(session_eff, key=lambda s: session_eff[s]["efficiency_score"])
        insights.append({
            "type": "info", "icon": "🏆",
            "title": "Disparidad de eficiencia entre sesiones",
            "detail": (
                f"Mejor sesión: `{best_sid[:10]}…` (score {session_eff[best_sid]['efficiency_score']}/100). "
                f"Peor sesión: `{worst_sid[:10]}…` (score {session_eff[worst_sid]['efficiency_score']}/100). "
                "Revisa qué herramientas usaste en cada sesión."
            ),
        })

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.title("💰 FinOps Command Center")
    st.caption("Monitoreo de costos, eficiencia y uso del Agente IA — datos en tiempo real")
with col_refresh:
    st.write("")
    st.write("")
    if st.button("🔄 Actualizar", use_container_width=True, type="primary"):
        st.rerun()

st.caption(f"Última actualización: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`")
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
report          = get_rich_report()
global_stats    = report["global"]
service_stats   = report["service_stats"]
model_stats     = report["model_stats"]
session_eff     = report["session_efficiency"]
all_calls       = report["all_calls"]
sessions_count  = report["sessions_count"]
has_data        = len(all_calls) > 0

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — KPIs Globales
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Resumen Global")

total_cost   = global_stats.get("total_cost_usd", 0.0)
total_calls_g = global_stats.get("total_calls", 0)
avg_sess_cost = round(total_cost / sessions_count, 6) if sessions_count > 0 else 0.0
top_model     = max(model_stats, key=lambda m: model_stats[m]["calls"]) if model_stats else "—"
top_svc       = max(service_stats, key=lambda s: service_stats[s]["calls"]) if service_stats else "—"
pct_budget_g  = round(total_cost / SESSION_BUDGET * 100, 1) if SESSION_BUDGET > 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sesiones registradas",     sessions_count)
k2.metric("Costo total acumulado",    f"${total_cost:.5f}")
k3.metric("Llamadas LLM totales",     f"{total_calls_g:,}")
k4.metric("Costo promedio / sesión",  f"${avg_sess_cost:.5f}")
k5.metric("% del presupuesto usado",  f"{pct_budget_g}%",
          delta=f"${SESSION_BUDGET - total_cost:.5f} restante",
          delta_color="inverse")

if not has_data:
    st.info("Aún no hay llamadas registradas. Usa el agente conversacional para generar datos de FinOps.")
    st.stop()

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — Timeline de Costos Acumulados
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Timeline de Costos Acumulados")

calls_con_ts = sorted(
    [c for c in all_calls if c.get("timestamp")],
    key=lambda x: x["timestamp"],
)

if calls_con_ts:
    cumulative = 0.0
    timeline_rows = []
    for c in calls_con_ts:
        cumulative += c.get("cost_usd", 0.0)
        try:
            ts = datetime.fromisoformat(c["timestamp"])
        except ValueError:
            continue
        timeline_rows.append({
            "Fecha/Hora":        ts,
            "Costo acumulado":   round(cumulative, 8),
            "Costo llamada":     c.get("cost_usd", 0.0),
            "Servicio":          c.get("service", "unknown"),
            "Modelo":            c.get("model", "unknown"),
            "Tokens entrada":    c.get("tokens_in", 0),
            "Tokens salida":     c.get("tokens_out", 0),
            "Session":           c.get("session_id", "?")[:10] + "…",
        })

    df_tl = pd.DataFrame(timeline_rows)

    col_tl1, col_tl2 = st.columns([3, 1])

    with col_tl1:
        fig_tl = px.line(
            df_tl,
            x="Fecha/Hora",
            y="Costo acumulado",
            color="Servicio",
            markers=True,
            title="Costo Acumulado en el Tiempo por Servicio",
            labels={"Costo acumulado": "USD (acumulado)"},
            hover_data=["Tokens entrada", "Tokens salida", "Costo llamada", "Session"],
            color_discrete_map={svc: _svc_color(svc) for svc in df_tl["Servicio"].unique()},
        )
        fig_tl.add_hline(
            y=SESSION_BUDGET,
            line_dash="dot",
            line_color="red",
            annotation_text=f"  Límite presupuesto ${SESSION_BUDGET}",
            annotation_position="bottom right",
        )
        fig_tl.update_layout(
            height=360,
            margin=dict(t=50, b=20, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    with col_tl2:
        ultimas = df_tl.tail(20).reset_index(drop=True)
        etiquetas = [f"#{i+1}" for i in range(len(ultimas))]
        fig_bar_calls = px.bar(
            ultimas,
            x="Costo llamada",
            y=etiquetas,
            orientation="h",
            color="Servicio",
            title="Últimas 20 llamadas",
            labels={"x": "Costo USD", "y": "Llamada"},
            color_discrete_map={svc: _svc_color(svc) for svc in ultimas["Servicio"].unique()},
        )
        fig_bar_calls.update_layout(
            height=360,
            margin=dict(t=50, b=20, l=0, r=0),
            showlegend=False,
        )
        st.plotly_chart(fig_bar_calls, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — Inteligencia por Servicio
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🔧 Inteligencia por Servicio")

if service_stats:
    svcs     = list(service_stats.keys())
    s_calls  = [service_stats[s]["calls"]                   for s in svcs]
    s_cost   = [service_stats[s]["cost_usd"]                for s in svcs]
    s_eff    = [service_stats[s].get("efficiency_ratio", 0) for s in svcs]
    s_avg_ti = [service_stats[s].get("avg_tokens_in", 0)    for s in svcs]
    s_avg_to = [service_stats[s].get("avg_tokens_out", 0)   for s in svcs]
    svc_colors = [_svc_color(s) for s in svcs]

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        fig_cost = px.bar(
            x=svcs, y=s_cost,
            color=svcs,
            color_discrete_sequence=svc_colors,
            title="Costo Total por Servicio (USD)",
            labels={"x": "Servicio", "y": "USD"},
            text=[f"${c:.5f}" for c in s_cost],
        )
        fig_cost.update_traces(textposition="outside")
        fig_cost.update_layout(
            height=320, margin=dict(t=50, b=20, l=0, r=0), showlegend=False
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_s2:
        fig_donut = px.pie(
            names=svcs, values=s_calls,
            hole=0.55,
            title="Distribución de Llamadas",
            color=svcs,
            color_discrete_sequence=svc_colors,
        )
        fig_donut.update_traces(textinfo="percent+label")
        fig_donut.update_layout(
            height=320, margin=dict(t=50, b=0, l=0, r=0), showlegend=False
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_s3:
        fig_eff = px.bar(
            x=svcs, y=s_eff,
            color=s_eff,
            color_continuous_scale="Greens",
            title="Eficiencia de Tokens (salida / entrada)",
            labels={"x": "Servicio", "y": "Ratio"},
            text=[f"{e:.3f}" for e in s_eff],
        )
        fig_eff.update_traces(textposition="outside")
        fig_eff.update_layout(
            height=320, margin=dict(t=50, b=20, l=0, r=0), showlegend=False
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    # Tokens agrupados por servicio
    fig_tokens = go.Figure()
    fig_tokens.add_trace(go.Bar(
        name="Avg tokens entrada", x=svcs, y=s_avg_ti,
        marker_color="#3498db", text=[f"{v:.0f}" for v in s_avg_ti], textposition="auto",
    ))
    fig_tokens.add_trace(go.Bar(
        name="Avg tokens salida", x=svcs, y=s_avg_to,
        marker_color="#2ecc71", text=[f"{v:.0f}" for v in s_avg_to], textposition="auto",
    ))
    fig_tokens.update_layout(
        barmode="group",
        title="Tokens Promedio por Servicio (entrada vs salida)",
        height=300,
        margin=dict(t=50, b=20, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tokens, use_container_width=True)

    # Tabla detallada
    st.markdown("**Tabla detallada por servicio**")
    df_svc = pd.DataFrame([
        {
            "Servicio":            svc,
            "Llamadas":            service_stats[svc]["calls"],
            "Costo total":         f"${service_stats[svc]['cost_usd']:.6f}",
            "Costo/llamada":       f"${service_stats[svc].get('avg_cost_usd', 0):.7f}",
            "Avg tokens entrada":  f"{service_stats[svc].get('avg_tokens_in', 0):.0f}",
            "Avg tokens salida":   f"{service_stats[svc].get('avg_tokens_out', 0):.0f}",
            "Eficiencia (out/in)": f"{service_stats[svc].get('efficiency_ratio', 0):.3f}",
        }
        for svc in svcs
    ])
    st.dataframe(df_svc, use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — Análisis por Modelo LLM
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🤖 Análisis por Modelo LLM")

if model_stats:
    models   = list(model_stats.keys())
    m_calls  = [model_stats[m]["calls"]                    for m in models]
    m_cost   = [model_stats[m]["cost_usd"]                 for m in models]
    m_cptk   = [model_stats[m].get("cost_per_1k_tokens", 0) for m in models]

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        fig_mdl = go.Figure()
        fig_mdl.add_trace(go.Bar(
            name="Llamadas", x=models, y=m_calls,
            marker_color="#3498db", yaxis="y",
        ))
        fig_mdl.add_trace(go.Bar(
            name="Costo USD ×10 000", x=models,
            y=[round(c * 10_000, 4) for c in m_cost],
            marker_color="#e74c3c", yaxis="y2",
        ))
        fig_mdl.update_layout(
            title="Llamadas y Costo por Modelo",
            yaxis=dict(title="Llamadas"),
            yaxis2=dict(title="Costo ×10 000 USD", overlaying="y", side="right"),
            barmode="group",
            height=320,
            margin=dict(t=50, b=20, l=0, r=0),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_mdl, use_container_width=True)

    with col_m2:
        df_mdl = pd.DataFrame([
            {
                "Modelo":              mdl,
                "Llamadas":            model_stats[mdl]["calls"],
                "Tokens entrada":      f"{model_stats[mdl]['tokens_in']:,}",
                "Tokens salida":       f"{model_stats[mdl]['tokens_out']:,}",
                "Costo total":         f"${model_stats[mdl]['cost_usd']:.6f}",
                "Costo / llamada":     f"${model_stats[mdl].get('avg_cost_per_call', 0):.7f}",
                "Costo / 1k tokens":   f"${model_stats[mdl].get('cost_per_1k_tokens', 0):.5f}",
            }
            for mdl in models
        ])
        st.dataframe(df_mdl, use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — Session Leaderboard
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🏆 Session Leaderboard")
st.caption("Ranking de sesiones por eficiencia compuesta (output/input ratio + costo/llamada)")

if session_eff:
    filas = []
    for sid, eff in session_eff.items():
        filas.append({
            "_score":          eff["efficiency_score"],
            "Session ID":      sid[:14] + "…" if len(sid) > 14 else sid,
            "Llamadas":        eff["calls_count"],
            "Costo total":     f"${eff['total_cost_usd']:.6f}",
            "Costo/llamada":   f"${eff['avg_cost_per_call']:.7f}",
            "Ratio out/in":    f"{eff['efficiency_ratio']:.3f}",
            "Duración (min)":  eff["duration_min"] or "—",
            "Burn rate ($/hr)": f"${eff['burn_rate_per_hour']:.5f}" if eff["burn_rate_per_hour"] else "—",
            "Eficiencia":      f"{eff['efficiency_score']}/100",
        })

    filas_sorted = sorted(filas, key=lambda r: r["_score"], reverse=True)
    medallas = ["🥇", "🥈", "🥉"]
    for i, fila in enumerate(filas_sorted):
        fila["#"] = medallas[i] if i < 3 else f"#{i + 1}"

    cols_show = ["#", "Session ID", "Llamadas", "Costo total", "Costo/llamada",
                 "Ratio out/in", "Duración (min)", "Burn rate ($/hr)", "Eficiencia"]
    df_lb = pd.DataFrame(filas_sorted)[cols_show]
    st.dataframe(df_lb, use_container_width=True, hide_index=True)
else:
    st.info("No hay datos de sesiones suficientes para el leaderboard.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — Recomendaciones Inteligentes de Optimización
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("💡 Recomendaciones Automáticas de Optimización")
st.caption("Generadas en tiempo real a partir de los patrones detectados en el log de uso")

insights = _generate_insights(service_stats, model_stats, session_eff, all_calls, SESSION_BUDGET)

if not insights:
    st.info("Acumula más llamadas para recibir recomendaciones personalizadas.")
else:
    for ins in insights:
        msg = f"**{ins['icon']} {ins['title']}** — {ins['detail']}"
        t = ins["type"]
        if t == "success":
            st.success(msg)
        elif t == "warning":
            st.warning(msg)
        elif t == "error":
            st.error(msg)
        else:
            st.info(msg)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — Proyector de Presupuesto Interactivo
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🔮 Proyector de Presupuesto Interactivo")
st.caption(
    "Usa el historial acumulado para estimar cuánto gastarás si sigues haciendo preguntas. "
    "Selecciona la sesión a proyectar y ajusta el slider."
)

if session_eff:
    session_ids = list(session_eff.keys())
    sid_labels  = {s: f"{s[:14]}… ({session_eff[s]['calls_count']} llamadas)" for s in session_ids}

    col_p1, col_p2 = st.columns([1, 2])

    with col_p1:
        sid_sel = st.selectbox(
            "Sesión a proyectar",
            options=session_ids,
            format_func=lambda s: sid_labels[s],
        )
        n_preguntas = st.slider(
            "Preguntas adicionales planeadas",
            min_value=1, max_value=50, value=10, step=1,
        )

    forecast = get_cost_forecast(sid_sel, n_preguntas)
    curr     = forecast["current_cost_usd"]
    extra    = forecast["projected_additional_usd"]
    total_p  = forecast["projected_total_usd"]
    remain   = forecast["budget_remaining_usd"]
    calls_lft = forecast["calls_remaining_in_budget"]
    exceed   = forecast["will_exceed_budget"]
    avg_c    = forecast["avg_cost_per_call_usd"]

    with col_p1:
        st.metric("Costo actual sesión",        f"${curr:.6f}")
        st.metric("Costo promedio por pregunta", f"${avg_c:.7f}")
        st.metric(f"Proyección +{n_preguntas} preguntas", f"${extra:.6f}",
                  delta=f"Total: ${total_p:.6f}", delta_color="inverse")
        st.metric("Preguntas restantes en presupuesto",
                  f"{calls_lft}" if calls_lft < 999 else "∞")

    with col_p2:
        pct_actual    = min(curr    / SESSION_BUDGET * 100, 150)
        pct_proyec    = min(total_p / SESSION_BUDGET * 100, 150)

        fig_gauge = go.Figure()
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=pct_actual,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Consumo ACTUAL"},
            domain={"x": [0, 0.45], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 150], "ticksuffix": "%"},
                "bar": {"color": "#e74c3c" if pct_actual >= 100 else "#3498db"},
                "steps": [
                    {"range": [0,   50],  "color": "#d5f5e3"},
                    {"range": [50,  80],  "color": "#fef9e7"},
                    {"range": [80,  100], "color": "#fde8d8"},
                    {"range": [100, 150], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.8, "value": 100},
            },
        ))
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=pct_proyec,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"Proyección +{n_preguntas} preg."},
            domain={"x": [0.55, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 150], "ticksuffix": "%"},
                "bar": {"color": "#e74c3c" if pct_proyec >= 100 else "#f39c12"},
                "steps": [
                    {"range": [0,   50],  "color": "#d5f5e3"},
                    {"range": [50,  80],  "color": "#fef9e7"},
                    {"range": [80,  100], "color": "#fde8d8"},
                    {"range": [100, 150], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.8, "value": 100},
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if exceed:
            st.error(
                f"⛔ Con {n_preguntas} preguntas más gastarías **${total_p:.5f}** "
                f"— superas el presupuesto de ${SESSION_BUDGET:.3f}. "
                f"Máximo recomendado: **{calls_lft} preguntas más**."
            )
        elif pct_proyec >= 80:
            st.warning(
                f"⚠️ Con {n_preguntas} preguntas consumirías el {pct_proyec:.1f}% del presupuesto."
            )
        else:
            st.success(
                f"✅ Con {n_preguntas} preguntas adicionales gastarías **${total_p:.5f}** "
                f"— dentro del presupuesto de ${SESSION_BUDGET:.3f}."
            )

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — Exportar Reporte
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📤 Exportar Reporte FinOps")

ts_export = datetime.now().strftime("%Y%m%d_%H%M%S")

export_payload = {
    "generated_at":    datetime.now(timezone.utc).isoformat(),
    "budget_usd":      SESSION_BUDGET,
    "global_stats":    global_stats,
    "service_stats":   service_stats,
    "model_stats":     model_stats,
    "session_efficiency": session_eff,
    "insights_count":  len(insights),
}

col_e1, col_e2 = st.columns(2)
with col_e1:
    st.download_button(
        label="📄 Descargar reporte JSON",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False),
        file_name=f"finops_report_{ts_export}.json",
        mime="application/json",
        use_container_width=True,
    )
with col_e2:
    if all_calls:
        df_calls_export = pd.DataFrame(all_calls)
        st.download_button(
            label="📊 Descargar log de llamadas CSV",
            data=df_calls_export.to_csv(index=False),
            file_name=f"finops_calls_{ts_export}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("📊 Descargar log CSV", disabled=True, use_container_width=True)
