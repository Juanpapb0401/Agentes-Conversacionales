"""
Servicio de Web Scraping — Twitter/X
Utiliza twscrape (autenticación por cuenta) para buscar los N tweets
más relevantes asociados a una palabra clave.

Gestión de cuentas:
  Las credenciales se leen de las variables de entorno TWITTER_ACCOUNTS,
  que debe tener el formato:  "usuario1:pass1,usuario2:pass2"
  opcionalmente con email:    "usuario1:pass1:email1@x.com,..."

El pool de cuentas se guarda en data/twscrape_pool.db (SQLite).
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from twscrape import API, gather
from twscrape.models import Tweet

# Ruta del pool de cuentas SQLite
_DB_PATH = os.path.join("data", "twscrape_pool.db")

# Límite de seguridad para no saturar la API
_MAX_TWEETS = 200


def _get_api() -> API:
    """Devuelve la instancia de API con la base de datos del pool."""
    return API(_DB_PATH)


async def _ensure_accounts(api: API) -> None:
    """
    Registra cuentas en el pool si aún no están y hace login de las pendientes.
    Lee la variable de entorno TWITTER_ACCOUNTS con formato:
        "user1:pass1:email1@x.com,user2:pass2:email2@x.com"
    """
    raw = os.getenv("TWITTER_ACCOUNTS", "").strip()
    if not raw:
        return

    # Obtener usernames ya registrados para no agregar duplicados
    existing = {acc.username.lower() for acc in await api.pool.get_all()}

    new_added = False
    for entry in raw.split(","):
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) < 2:
            continue
        username = parts[0]
        if username.lower() in existing:
            continue  # ya está en el pool, no agregar de nuevo
        password = parts[1]
        email    = parts[2] if len(parts) > 2 else f"{username}@example.com"
        await api.pool.add_account(
            username=username,
            password=password,
            email=email,
            email_password=password,
        )
        new_added = True

    # Solo hacer login si se agregaron cuentas nuevas o hay pendientes sin login
    all_accounts = await api.pool.get_all()
    needs_login = any(not acc.active for acc in all_accounts)
    if new_added or needs_login:
        await api.pool.login_all()


def _tweet_to_dict(tweet: Tweet) -> Dict[str, Any]:
    """Convierte un objeto Tweet de twscrape a dict serializable."""
    return {
        "id":         str(tweet.id),
        "url":        tweet.url,
        "texto":      tweet.rawContent,
        "autor":      tweet.user.username if tweet.user else "desconocido",
        "nombre":     tweet.user.displayname if tweet.user else "",
        "fecha":      tweet.date.isoformat() if tweet.date else None,
        "likes":      tweet.likeCount or 0,
        "retweets":   tweet.retweetCount or 0,
        "replies":    tweet.replyCount or 0,
        "views":      tweet.viewCount or 0,
        "idioma":     tweet.lang or "und",
        "es_reply":   tweet.inReplyToTweetId is not None,
    }


async def _buscar_tweets_async(
    query: str,
    n: int,
    lang: Optional[str],
    excluir_replies: bool,
) -> List[Dict[str, Any]]:
    """Lógica asíncrona de búsqueda. Llamada desde el wrapper síncrono."""
    api = _get_api()
    await _ensure_accounts(api)

    # Verificar que haya al menos una cuenta activa antes de buscar
    # (twscrape 0.17+ ya no lanza excepción, simplemente retorna vacío)
    active_accounts = [acc for acc in await api.pool.get_all() if acc.active]
    if not active_accounts:
        all_accounts = await api.pool.get_all()
        if not all_accounts:
            raise RuntimeError(
                "No hay cuentas configuradas. "
                "Agrega TWITTER_ACCOUNTS=usuario:contraseña:email en .env"
            )
        raise RuntimeError(
            f"Las {len(all_accounts)} cuenta(s) configurada(s) no pudieron autenticarse en Twitter. "
            "Verifica usuario/contraseña o si Twitter requiere verificación adicional."
        )

    # Construir query con filtros de calidad de Twitter
    q = query.strip()
    if lang:
        q += f" lang:{lang}"
    if excluir_replies:
        q += " -filter:replies"
    q += " -filter:retweets"  # solo tweets originales por defecto

    tweets: List[Dict[str, Any]] = []
    async for tw in api.search(q, limit=min(n, _MAX_TWEETS)):
        tweets.append(_tweet_to_dict(tw))
        if len(tweets) >= n:
            break

    # Ordenar por relevancia: score = likes*2 + retweets*3 + replies
    tweets.sort(
        key=lambda t: t["likes"] * 2 + t["retweets"] * 3 + t["replies"],
        reverse=True,
    )
    return tweets


def buscar_tweets(
    query: str,
    n: int = 20,
    lang: Optional[str] = "es",
    excluir_replies: bool = True,
) -> Dict[str, Any]:
    """
    Punto de entrada síncrono del servicio.
    Busca los N tweets más relevantes para la palabra clave dada.

    Args:
        query:           Término de búsqueda (ej: "PETRO", "#Colombia economía")
        n:               Cantidad máxima de tweets a retornar (máx 200)
        lang:            Filtrar por idioma (ej: "es", "en"). None = todos.
        excluir_replies: Si True, omite tweets que son respuestas.

    Returns:
        Dict con:
          - query:   término buscado
          - n_encontrados: cantidad real de tweets
          - tweets:  lista de dicts con los campos del tweet
          - error:   mensaje de error si algo falló (o None)
    """
    if not query or not query.strip():
        return {"query": query, "n_encontrados": 0, "tweets": [], "error": "Query vacía"}

    n = max(1, min(n, _MAX_TWEETS))

    try:
        # asyncio.run crea un nuevo event loop — compatible con FastAPI sync endpoints
        tweets = asyncio.run(_buscar_tweets_async(query, n, lang, excluir_replies))
        return {
            "query":        query,
            "n_encontrados": len(tweets),
            "tweets":       tweets,
            "error":        None,
        }
    except Exception as exc:
        return {
            "query":        query,
            "n_encontrados": 0,
            "tweets":       [],
            "error":        f"Error al scrapear: {str(exc)}",
        }
