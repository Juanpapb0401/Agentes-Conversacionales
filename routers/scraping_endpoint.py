"""
Router — Scraping de Twitter/X
GET /scraping/tweets?query=PETRO&n=20&lang=es&excluir_replies=true
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from services.scraping_service import buscar_tweets

router = APIRouter(prefix="/scraping", tags=["Scraping"])


@router.get("/tweets")
def buscar_tweets_endpoint(
    query: str = Query(..., description="Palabra clave o hashtag a buscar"),
    n: int = Query(20, ge=1, le=200, description="Número de tweets a retornar"),
    lang: Optional[str] = Query("es", description="Filtro de idioma (ej: 'es', 'en'). Omitir para todos."),
    excluir_replies: bool = Query(True, description="Omitir tweets que son respuestas"),
):
    """
    Busca los N tweets más relevantes para una palabra clave usando web scraping.
    Los resultados se ordenan por relevancia (likes × 2 + retweets × 3 + replies).

    Requiere que la variable de entorno TWITTER_ACCOUNTS esté configurada:
        TWITTER_ACCOUNTS=usuario1:password1:email1@x.com,usuario2:password2:email2@x.com
    """
    resultado = buscar_tweets(
        query=query,
        n=n,
        lang=lang if lang else None,
        excluir_replies=excluir_replies,
    )

    if resultado.get("error") and resultado["n_encontrados"] == 0:
        raise HTTPException(status_code=503, detail=resultado["error"])

    return resultado
