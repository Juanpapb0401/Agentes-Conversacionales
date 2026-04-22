from pydantic import BaseModel
from typing import List, Optional

# --- Modelos de Entrada (Lo que envía el Agente) ---
class TextAnalysisRequest(BaseModel):
    text: str

class PropagationRequest(BaseModel):
    post_id: str

# --- Modelos de Salida (Lo que devuelve la API) ---
class SentimientoResponse(BaseModel):
    clima: str
    score: float
    justificacion: str

class MetricasResponse(BaseModel):
    total_likes: int
    top_influencers: List[str]

class PropagacionResponse(BaseModel):
    id_original: str
    alcance: int
    velocidad_media: str