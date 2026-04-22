from fastapi import APIRouter, HTTPException, Query
from schemas import PropagacionResponse
from data_loader import dataframe_principal
from services.propagacion_service import analizar_propagacion

router = APIRouter(
    prefix="/analisis",
    tags=["Análisis de Propagación"]
)

@router.get("/propagacion", response_model=PropagacionResponse)
async def get_propagacion(post_id: str = Query(..., description="ID del post a analizar")):
    """
    Endpoint para medir la propagación de un mensaje a través del dataset.
    Calcula alcance, velocidad, matching de contenido y score de impacto.
    """
    if dataframe_principal is None:
        raise HTTPException(status_code=500, detail="El dataset no está disponible.")

    # Ejecutar algoritmo del Rol 3
    resultado = analizar_propagacion(post_id, dataframe_principal)

    if not resultado.get("encontrado", False):
        raise HTTPException(status_code=404, detail=resultado.get("error", "Post no encontrado"))

    return resultado
