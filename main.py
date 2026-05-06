from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from schemas import (
    TextAnalysisRequest,
    ResumenRequest,
    SentimientoResponse,
    ResumenResponse,
    MetricasResponse
)
from data_loader import dataframe_principal
from routers.propagacion_endpoint import router as propagacion_router
from services.nlp_service import analizar_sentimiento_llm, resumir_conversacion_llm

app = FastAPI(
    title="API Analítica - Reto MCP",
    description="Microservicios para el análisis de conversaciones digitales.",
    version="1.0.0"
)

# CORS: permite que Streamlit (puerto 8501) consuma la API (puerto 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Incluir Routers Modulares
app.include_router(propagacion_router)

# 1. Endpoint de Sentimientos
@app.post("/analisis/sentimientos", response_model=SentimientoResponse)
async def analizar_sentimiento(
    request: TextAnalysisRequest,
    session_id: str = Header(default="unknown", alias="X-Session-ID"),
):
    try:
        return analizar_sentimiento_llm(request.text, session_id=session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# 2. Endpoint de Resumen General
@app.post("/analisis/resumen", response_model=ResumenResponse)
async def analizar_resumen(
    request: ResumenRequest,
    session_id: str = Header(default="unknown", alias="X-Session-ID"),
):
    try:
        return resumir_conversacion_llm(
            request.textos,
            max_palabras=request.max_palabras,
            idioma=request.idioma,
            session_id=session_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# 3. Endpoint de Métricas (enriquecido: total mensajes, plataformas, top posts)
@app.post("/analisis/metricas", response_model=MetricasResponse)
async def analizar_metricas():
    if dataframe_principal is None:
        raise HTTPException(status_code=500, detail="Dataset no disponible")

    df = dataframe_principal

    total_likes_red = int(df["likes"].sum())
    total_mensajes = int(len(df))

    influencers = df.groupby("user_id")["likes"].sum().nlargest(5)
    top_influencers = influencers.index.tolist()

    top_posts = (
        df.nlargest(5, "likes")[["post_id", "likes", "text"]]
        .assign(text=lambda d: d["text"].str[:100])
        .to_dict(orient="records")
        if "post_id" in df.columns and "text" in df.columns
        else []
    )

    plataformas = (
        df["platform"].dropna().unique().tolist()
        if "platform" in df.columns
        else []
    )

    return {
        "total_likes": total_likes_red,
        "total_mensajes": total_mensajes,
        "top_influencers": top_influencers,
        "top_posts_por_likes": top_posts,
        "plataformas": plataformas,
    }
