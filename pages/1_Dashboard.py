"""
Dashboard Visual — Análisis de Twitter/X en Tiempo Real
Página 1 del sistema multi-página de Streamlit.

Funcionalidades:
  1. Buscar tweets por keyword vía scraping (POST al servicio FastAPI)
  2. Ejecutar análisis de sentimiento sobre cada tweet encontrado
  3. Visualizar distribución de sentimientos, engagement y top autores
  4. Seleccionar un tweet para analizar su propagación en el dataset base
"""

import json
import os
import time

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_HTTP_TIMEOUT = 60.0  # scraping puede tardar más que análisis normal

st.set_page_config(
    page_title="Dashboard — Análisis Twitter/X",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard de Análisis en Tiempo Real")
st.caption("Extrae tweets de Twitter/X, analiza su sentimiento y visualiza el impacto")

# =============================================================================
# SECCIÓN 1 — Búsqueda y Scraping
# =============================================================================

st.header("🔍 Búsqueda de Tweets")

col_q, col_n, col_lang, col_btn = st.columns([3, 1, 1, 1])
with col_q:
    query = st.text_input("Palabra clave o hashtag", placeholder="Ej: PETRO, #Colombia, reforma tributaria")
with col_n:
    n_tweets = st.number_input("Cantidad", min_value=5, max_value=100, value=20, step=5)
with col_lang:
    lang = st.selectbox("Idioma", ["es", "en", "todos"], index=0)
with col_btn:
    st.write("")
    st.write("")
    buscar = st.button("🚀 Scrapear", use_container_width=True, type="primary")

# Persistir tweets entre rerenders
if "tweets_scrapeados" not in st.session_state:
    st.session_state.tweets_scrapeados = []
if "analisis_sentimientos" not in st.session_state:
    st.session_state.analisis_sentimientos = {}

# --- Ejecutar scraping ---
if buscar and query.strip():
    params = {"query": query.strip(), "n": n_tweets}
    if lang != "todos":
        params["lang"] = lang

    with st.spinner(f"Extrayendo tweets sobre '{query}'... (puede tomar 15-30 segundos)"):
        try:
            resp = httpx.get(f"{_API}/scraping/tweets", params=params, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.tweets_scrapeados = data.get("tweets", [])
            st.session_state.analisis_sentimientos = {}  # resetear análisis anterior
            st.success(f"✅ {data['n_encontrados']} tweets extraídos para **'{query}'**")
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            st.error(f"❌ Error del servidor: {detail}")
            st.info(
                "Si ves 'No hay cuentas de Twitter disponibles', configura "
                "`TWITTER_ACCOUNTS=user:pass:email` en tu archivo `.env`"
            )
        except Exception as e:
            st.error(f"❌ No se pudo conectar con la API: {e}")
            st.info("Verifica que FastAPI esté corriendo en `http://127.0.0.1:8000`")

tweets = st.session_state.tweets_scrapeados

# =============================================================================
# SECCIÓN 2 — Análisis de Sentimientos
# =============================================================================

if tweets:
    st.divider()
    st.header("🧠 Análisis de Sentimientos")

    analizar_btn = st.button(
        f"Analizar sentimiento de {len(tweets)} tweets",
        type="primary",
        disabled=bool(st.session_state.analisis_sentimientos),
    )

    if analizar_btn:
        progress = st.progress(0, text="Analizando sentimientos...")
        for i, tw in enumerate(tweets):
            tweet_id = tw["id"]
            if tweet_id not in st.session_state.analisis_sentimientos:
                try:
                    resp = httpx.post(
                        f"{_API}/analisis/sentimientos",
                        json={"text": tw["texto"][:1000]},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.session_state.analisis_sentimientos[tweet_id] = resp.json()
                    else:
                        st.session_state.analisis_sentimientos[tweet_id] = {
                            "clima": "neutral", "score": 0.5, "justificacion": "Error en API"
                        }
                except Exception:
                    st.session_state.analisis_sentimientos[tweet_id] = {
                        "clima": "neutral", "score": 0.5, "justificacion": "Sin respuesta"
                    }
            progress.progress((i + 1) / len(tweets), text=f"Analizando {i+1}/{len(tweets)}...")
            time.sleep(0.1)
        progress.empty()
        st.rerun()

    # --- Preparar datos enriquecidos ---
    if st.session_state.analisis_sentimientos:
        enriched = []
        for tw in tweets:
            s = st.session_state.analisis_sentimientos.get(tw["id"], {})
            enriched.append({
                **tw,
                "clima":        s.get("clima", "neutral"),
                "score":        s.get("score", 0.5),
                "justificacion": s.get("justificacion", ""),
            })

        # =========================================================
        # GRÁFICAS — ROW 1
        # =========================================================
        col1, col2, col3 = st.columns(3)

        # Donut — distribución de climas
        climas = [e["clima"] for e in enriched]
        clima_counts = {c: climas.count(c) for c in set(climas)}
        color_map = {"positivo": "#2ecc71", "negativo": "#e74c3c", "neutral": "#95a5a6"}
        with col1:
            fig_donut = px.pie(
                names=list(clima_counts.keys()),
                values=list(clima_counts.values()),
                hole=0.5,
                color=list(clima_counts.keys()),
                color_discrete_map=color_map,
                title="Distribución de Sentimiento",
            )
            fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_donut, use_container_width=True)

        # Histograma — distribución de scores
        with col2:
            scores = [e["score"] for e in enriched]
            fig_hist = px.histogram(
                x=scores,
                nbins=10,
                title="Distribución de Scores (0=neg, 1=pos)",
                labels={"x": "Score", "count": "Tweets"},
                color_discrete_sequence=["#3498db"],
            )
            fig_hist.update_layout(margin=dict(t=40, b=20, l=0, r=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter — score vs likes coloreado por clima
        with col3:
            fig_scatter = px.scatter(
                x=[e["likes"] for e in enriched],
                y=[e["score"] for e in enriched],
                color=[e["clima"] for e in enriched],
                color_discrete_map=color_map,
                labels={"x": "Likes", "y": "Score sentimiento", "color": "Clima"},
                title="Sentimiento vs Engagement",
                hover_name=[f"@{e['autor']}" for e in enriched],
            )
            fig_scatter.update_layout(margin=dict(t=40, b=20, l=0, r=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # =========================================================
        # GRÁFICAS — ROW 2
        # =========================================================
        col4, col5 = st.columns([2, 1])

        # Bar chart — top autores por engagement
        with col4:
            autor_eng: dict = {}
            for e in enriched:
                a = f"@{e['autor']}"
                autor_eng[a] = autor_eng.get(a, 0) + e["likes"] + e["retweets"] * 2
            top_autores = sorted(autor_eng.items(), key=lambda x: x[1], reverse=True)[:10]
            fig_bar = px.bar(
                x=[t[1] for t in top_autores],
                y=[t[0] for t in top_autores],
                orientation="h",
                title="Top 10 Autores por Engagement",
                labels={"x": "Engagement (likes + retweets×2)", "y": "Autor"},
                color=[t[1] for t in top_autores],
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(margin=dict(t=40, b=20, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Gauge — score promedio de sentimiento
        with col5:
            avg_score = sum(scores) / len(scores) if scores else 0.5
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(avg_score, 3),
                delta={"reference": 0.5, "valueformat": ".3f"},
                title={"text": "Score Promedio de Sentimiento"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#3498db"},
                    "steps": [
                        {"range": [0, 0.35], "color": "#fadbd8"},
                        {"range": [0.35, 0.65], "color": "#fef9e7"},
                        {"range": [0.65, 1],   "color": "#d5f5e3"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.75,
                        "value": 0.5,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # =========================================================
        # TABLA DE TWEETS
        # =========================================================
        st.divider()
        st.subheader("📋 Tweets Analizados")

        clima_filtro = st.multiselect(
            "Filtrar por clima",
            options=["positivo", "negativo", "neutral"],
            default=["positivo", "negativo", "neutral"],
        )
        filtrados = [e for e in enriched if e["clima"] in clima_filtro]

        for e in filtrados:
            color = {"positivo": "🟢", "negativo": "🔴", "neutral": "⚪"}.get(e["clima"], "⚪")
            with st.expander(
                f"{color} **@{e['autor']}** — score {e['score']:.2f} | 👍 {e['likes']} 🔁 {e['retweets']}"
            ):
                st.markdown(f"> {e['texto']}")
                st.caption(f"📅 {e['fecha']} | 🌐 [Ver tweet]({e['url']})")
                st.info(f"**Justificación:** {e['justificacion']}")

# =============================================================================
# SECCIÓN 3 — Análisis de Propagación desde el Dataset Base
# =============================================================================

st.divider()
st.header("🌐 Análisis de Propagación del Dataset Base")
st.caption("Selecciona o escribe un post_id del dataset original para analizar su viralidad")

post_id_input = st.text_input(
    "Post ID a analizar",
    placeholder="Ej: 199219160505_1274366331365120",
    key="prop_post_id",
)

if st.button("📡 Analizar Propagación", disabled=not post_id_input.strip()):
    with st.spinner("Calculando propagación..."):
        try:
            resp = httpx.get(
                f"{_API}/analisis/propagacion",
                params={"post_id": post_id_input.strip()},
                timeout=30,
            )
            resp.raise_for_status()
            p = resp.json()
        except Exception as e:
            st.error(f"Error: {e}")
            p = None

    if p and p.get("encontrado"):
        st.success(f"**Nivel de impacto:** {p['nivel_impacto']} | **Arquetipo:** {p['arquetipo']}")
        st.info(
            f"**Mensaje original:**\n\n> {p.get('texto_original', 'N/A')}\n\n"
            f"**Autor:** `{p.get('autor_original', 'N/A')}` | **Fecha:** `{p.get('timestamp_original', 'N/A')}`"
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Score impacto", f"{p['score_impacto']}/100")
        m2.metric("Alcance total", p["alcance"])
        m3.metric("Replies directas", p["replies_directas"])
        m4.metric("Usuarios únicos", p.get("usuarios_unicos_en_cadena", "N/A"))
        m5.metric("Velocidad media", p.get("velocidad_media_label", "N/A"))

        # Gauge de score de impacto
        fig_imp = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p["score_impacto"],
            title={"text": "Score de Impacto (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e67e22"},
                "steps": [
                    {"range": [0, 25],  "color": "#d5f5e3"},
                    {"range": [25, 50], "color": "#fef9e7"},
                    {"range": [50, 75], "color": "#fde8d8"},
                    {"range": [75, 100], "color": "#fadbd8"},
                ],
            },
        ))
        fig_imp.update_layout(height=250, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("🔍 Ver JSON completo"):
            st.json(p)
    elif p:
        st.warning(f"Post ID `{post_id_input}` no encontrado en el dataset.")

# =============================================================================
# SECCIÓN 4 — Métricas Generales del Dataset
# =============================================================================

st.divider()
st.header("📈 Métricas Generales del Dataset")

if st.button("Cargar métricas del dataset"):
    with st.spinner("Calculando métricas..."):
        try:
            resp = httpx.post(f"{_API}/analisis/metricas", json={}, timeout=30)
            resp.raise_for_status()
            m = resp.json()
            st.session_state["metricas_dataset"] = m
        except Exception as e:
            st.error(f"Error: {e}")

m = st.session_state.get("metricas_dataset")
if m:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total mensajes", f"{m['total_mensajes']:,}")
    col_b.metric("Total likes acumulados", f"{m['total_likes']:,}")
    col_c.metric("Plataformas", len(m.get("plataformas", [])))

    col_inf, col_post = st.columns(2)
    with col_inf:
        st.subheader("🏆 Top 5 Influencers")
        for i, user in enumerate(m.get("top_influencers", []), 1):
            st.markdown(f"**{i}.** `{user}`")

    with col_post:
        st.subheader("🔥 Top 5 Posts por Likes")
        for post in m.get("top_posts_por_likes", []):
            with st.expander(f"👍 {post.get('likes', 0):,} likes — `{post.get('post_id', '')[:30]}...`"):
                st.write(post.get("text", "Sin texto"))
