from fastapi import FastAPI, HTTPException
from schemas import (
    TextAnalysisRequest, PropagationRequest, 
    SentimientoResponse, MetricasResponse, PropagacionResponse
)
from data_loader import dataframe_principal

app = FastAPI(
    title="API Analítica - Reto MCP",
    description="Microservicios para el análisis de conversaciones digitales.",
    version="1.0.0"
)

# 1. Endpoint de Sentimientos (Rol 2 inyectará la lógica del LLM aquí)
@app.post("/analisis/sentimientos", response_model=SentimientoResponse)
async def analizar_sentimiento(request: TextAnalysisRequest):
    # DUMMY RESPONSE: Reemplazar con la llamada a OpenAI/Gemini
    return {
        "clima": "positivo",
        "score": 0.85,
        "justificacion": "Datos de prueba: el texto parece optimista."
    }

# 2. Endpoint de Métricas (Lógica de Datos Tradicional)
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

# 3. Endpoint de Propagación OBLIGATORIO (Rol 3 inyectará su algoritmo aquí)
@app.post("/analisis/propagacion", response_model=PropagacionResponse)
async def medir_propagacion(request: PropagationRequest):
    # DUMMY RESPONSE: Reemplazar con el algoritmo de grafos o recursión
    return {
        "id_original": request.post_id,
        "alcance": 120,
        "velocidad_media": "15 min"
    }