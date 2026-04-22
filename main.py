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
        raise HTTPException(status_code=500, detail="El dataset no está disponible.")
    
    # Ejemplo de lógica que puedes ir armando con tu equipo:
    # top_users = dataframe_principal.groupby('user_id')['likes'].sum().nlargest(5).index.tolist()
    
    return {
        "total_likes": 5000, # Reemplazar con lógica real
        "top_influencers": ["usuario_1", "usuario_2"] # Reemplazar con lógica real
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