from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# --- Modelos de Entrada (Lo que envía el Agente) ---
class TextAnalysisRequest(BaseModel):
    text: str

class ResumenRequest(BaseModel):
    textos: List[str]
    max_palabras: int = 120
    idioma: str = "es"

class PropagationRequest(BaseModel):
    post_id: str

# --- Modelos de Salida (Lo que devuelve la API) ---
class SentimientoResponse(BaseModel):
    clima: str
    score: float
    justificacion: str

class ResumenResponse(BaseModel):
    resumen: str
    temas_principales: List[str]
    posturas_clave: List[str]
    alcance_textos: int

class MetricasResponse(BaseModel):
    total_likes: int
    total_mensajes: int
    top_influencers: List[str]
    top_posts_por_likes: List[Dict[str, Any]]
    plataformas: List[str]

class PropagacionResponse(BaseModel):
    id_original: str
    encontrado: bool
    texto_original: Optional[str] = None
    timestamp_original: Optional[str] = None
    autor_original: Optional[str] = None
    alcance: int
    replies_directas: int
    cadena_total_nodos: int
    profundidad_maxima: int
    usuarios_unicos_en_cadena: Optional[int] = None
    plataformas: Optional[List[str]] = None
    velocidad_media_min: Optional[float]
    velocidad_media_label: str
    porcentaje_contenido_replicado: float
    score_impacto: float
    nivel_impacto: str
    arquetipo: str
