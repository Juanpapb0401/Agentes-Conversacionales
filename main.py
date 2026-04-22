from fastapi import FastAPI, HTTPException
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

# Incluir Routers Modulares
app.include_router(propagacion_router)

# 1. Endpoint de Sentimientos (Rol 2 inyectará la lógica del LLM aquí)
@app.post("/analisis/sentimientos", response_model=SentimientoResponse)
async def analizar_sentimiento(request: TextAnalysisRequest):
    try:
        return analizar_sentimiento_llm(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# 2. Endpoint de Resumen General (Rol 2 inyectará la logica del LLM aqui)
@app.post("/analisis/resumen", response_model=ResumenResponse)
async def analizar_resumen(request: ResumenRequest):
    try:
        return resumir_conversacion_llm(
            request.textos,
            max_palabras=request.max_palabras,
            idioma=request.idioma
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# 3. Endpoint de Métricas (Lógica de Datos Tradicional)
@app.post("/analisis/metricas", response_model=MetricasResponse)
async def analizar_metricas():
    if dataframe_principal is None:
        raise HTTPException(status_code=500, detail="Dataset no disponible")

    # 1. Calcular total de likes de toda la red
    total_likes_red = int(dataframe_principal['likes'].sum())

    # 2. Encontrar top 5 influencers (usuarios con más likes sumados)
    influencers = dataframe_principal.groupby('user_id')['likes'].sum().nlargest(5)
    top_nombres = influencers.index.tolist()

    return {
        "total_likes": total_likes_red,
        "top_influencers": top_nombres
    }
