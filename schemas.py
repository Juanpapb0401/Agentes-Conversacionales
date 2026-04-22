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
    encontrado: bool
    alcance: int
    replies_directas: int
    cadena_total_nodos: int
    velocidad_media_min: Optional[float]
    velocidad_media_label: str
    porcentaje_contenido_replicado: float
    score_impacto: float
    nivel_impacto: str