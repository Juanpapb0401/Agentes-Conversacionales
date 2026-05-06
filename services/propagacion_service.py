import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, List, Dict, Any
import re

def analizar_propagacion(post_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función principal del Rol 3. Calcula el árbol de propagación, velocidad y score de impacto.
    
    Args:
        post_id: El ID del mensaje original a analizar.
        df: El DataFrame con las conversaciones.
        
    Returns:
        Un diccionario con métricas de alcance, velocidad, contenido y score.
    """
    resultado_vacio = {
        "id_original": post_id,
        "encontrado": False,
        "error": f"No se encontró el post_id '{post_id}' en el dataset.",
    }

    # --- Verificar que el post existe ---
    if "post_id" not in df.columns:
        return {**resultado_vacio, "error": "El dataset no tiene columna 'post_id'."}

    post_mask = df["post_id"] == str(post_id)
    if not post_mask.any():
        return resultado_vacio

    post_original = df[post_mask].iloc[0]

    # 1. BFS para construir árbol de propagación (Nodos que responden al original o a sus descendientes)
    nodos_propagacion, profundidad_max = _bfs_propagacion(post_id, df)

    # 2. Obtener respuestas directas (Hijos inmediatos en el árbol)
    replies_directas_df = df[
        (df.get("parent_id", pd.Series(dtype=str)) == str(post_id))
    ] if "parent_id" in df.columns else pd.DataFrame()
    num_replies_directas = len(replies_directas_df)

    # 3. Métricas de tiempo (Velocidad de propagación)
    metricas_tiempo = _calcular_velocidad(post_original, replies_directas_df)

    # 4. Content matching (¿Cuánto del mensaje original se mantiene en la cadena?)
    nodos_df = df[df["post_id"].isin(nodos_propagacion)] if nodos_propagacion else pd.DataFrame()
    matching = _content_matching(post_original, nodos_df)

    # 5. Métricas sociales y engagement
    usuarios_unicos = int(nodos_df["user_id"].nunique()) if "user_id" in nodos_df.columns else 0
    plataformas = nodos_df["platform"].dropna().unique().tolist() if "platform" in nodos_df.columns else []

    shares_originales = int(post_original.get("shares", 0) or 0)
    likes_originales  = int(post_original.get("likes", 0) or 0)
    
    # El alcance total se define como la suma de la cadena de respuestas + los shares del original
    alcance = len(nodos_propagacion) + shares_originales

    # 6. Cálculo del Score de Impacto (0-100)
    score = _calcular_score_impacto(
        alcance=alcance,
        usuarios=usuarios_unicos,
        matching=matching["porcentaje_matching"],
        velocidad=metricas_tiempo.get("velocidad_media_minutos"),
        likes=likes_originales,
        shares=shares_originales,
    )

    # 7. Nuevo Análisis de Arquetipo (Creatividad Rol 3)
    arquetipo = _determinar_arquetipo(alcance, profundidad_max, num_replies_directas)

    return {
        "id_original": post_id,
        "encontrado": True,
        "texto_original": str(post_original.get("text", ""))[:300],
        "timestamp_original": str(post_original.get("timestamp", "")),
        "autor_original": str(post_original.get("user_id", "desconocido")),

        # Métricas de Alcance
        "alcance": alcance,
        "replies_directas": num_replies_directas,
        "cadena_total_nodos": len(nodos_propagacion),
        "profundidad_maxima": profundidad_max,
        "usuarios_unicos_en_cadena": usuarios_unicos,
        "plataformas": plataformas,

        # Métricas de Velocidad
        "tiempo_primera_replica_min": metricas_tiempo.get("primera_replica_minutos"),
        "tiempo_ultima_replica_min":  metricas_tiempo.get("ultima_replica_minutos"),
        "velocidad_media_min":        metricas_tiempo.get("velocidad_media_minutos"),
        "velocidad_media_label":      metricas_tiempo.get("velocidad_label"),

        # Análisis de Contenido
        "porcentaje_contenido_replicado": matching["porcentaje_matching"],
        "nodos_con_match": matching["nodos_con_match"],

        # Engagement del original
        "likes_originales": likes_originales,
        "shares_originales": shares_originales,

        # Clasificación de Impacto (Creatividad Rol 3)
        "score_impacto": score,
        "nivel_impacto": _nivel_impacto(score),
        "arquetipo": arquetipo,
    }

def _bfs_propagacion(post_id: str, df: pd.DataFrame) -> tuple[List[str], int]:
    """
    Realiza una búsqueda en anchura (BFS) para encontrar todos los descendientes en el grafo de conversaciones.
    Detecta hilos de conversación anidados (comentarios de comentarios) y mide la profundidad máxima.
    """
    if "parent_id" not in df.columns:
        return [], 0

    # Construcción del mapa parent→hijos con groupby
    filtered = df[df["parent_id"].notna()][["parent_id", "post_id"]].copy()
    children_map: Dict[str, List[str]] = {}
    for parent, group in filtered.groupby("parent_id")["post_id"]:
        children_map[str(parent)] = group.astype(str).tolist()

    visitados = set()
    cola = deque([(str(post_id), 0)]) # (ID, nivel)
    max_depth = 0

    while cola:
        nodo_actual, nivel = cola.popleft()
        if nodo_actual in visitados and nodo_actual != str(post_id):
            continue
        
        if nodo_actual != str(post_id):
            visitados.add(nodo_actual)
            max_depth = max(max_depth, nivel)
            
        for hijo in children_map.get(nodo_actual, []):
            if hijo not in visitados:
                cola.append((hijo, nivel + 1))

    return list(visitados), max_depth

def _calcular_velocidad(post_original: pd.Series, replies_df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula cuánto tiempo tardan en aparecer las respuestas respecto al original."""
    if replies_df.empty or "timestamp" not in replies_df.columns:
        return {
            "primera_replica_minutos": None,
            "ultima_replica_minutos": None,
            "velocidad_media_minutos": None,
            "velocidad_label": "Sin datos de tiempo",
        }

    ts_original = post_original.get("timestamp")
    if pd.isna(ts_original):
        return {
            "primera_replica_minutos": None,
            "ultima_replica_minutos": None,
            "velocidad_media_minutos": None,
            "velocidad_label": "Timestamp no disponible",
        }

    # Calculamos diferencias en minutos
    deltas = []
    for ts in replies_df["timestamp"].dropna():
        delta_min = (ts - ts_original).total_seconds() / 60
        if delta_min >= 0:
            deltas.append(delta_min)

    if not deltas:
        return {
            "primera_replica_minutos": None,
            "ultima_replica_minutos": None,
            "velocidad_media_minutos": None,
            "velocidad_label": "Sin réplicas posteriores",
        }

    promedio = round(float(np.mean(deltas)), 2)
    return {
        "primera_replica_minutos": round(min(deltas), 2),
        "ultima_replica_minutos":  round(max(deltas), 2),
        "velocidad_media_minutos": promedio,
        "velocidad_label": _formato_tiempo(promedio),
    }

def _content_matching(post_original: pd.Series, nodos_df: pd.DataFrame) -> Dict[str, Any]:
    """Mide qué tanto se repite el mensaje original en las réplicas."""
    if nodos_df.empty or "text" not in nodos_df.columns:
        return {"porcentaje_matching": 0.0, "nodos_con_match": 0}

    texto_original = str(post_original.get("text", "")).lower()
    # Tomamos palabras de al menos 4 letras para evitar nexos comunes
    palabras_clave = set(re.findall(r'\b\w{4,}\b', texto_original))
    
    stop_words = {"para", "como", "este", "esta", "esto", "pero", "cuando", "donde", "with", "that"}
    palabras_clave -= stop_words

    if not palabras_clave:
        return {"porcentaje_matching": 0.0, "nodos_con_match": 0}

    con_match = 0
    for texto in nodos_df["text"].fillna(""):
        if any(p in str(texto).lower() for p in palabras_clave):
            con_match += 1

    total = len(nodos_df)
    porcentaje = round((con_match / total) * 100, 2) if total > 0 else 0.0
    return {"porcentaje_matching": porcentaje, "nodos_con_match": con_match}

def _calcular_score_impacto(alcance: int, usuarios: int, matching: float, velocidad: Optional[float], likes: int, shares: int) -> float:
    """Calcula un score compuesto de impacto basado en pesos de alcance, usuarios y velocidad."""
    s_alcance = min(alcance / 500, 1.0) * 100
    s_usuarios = min(usuarios / 200, 1.0) * 100
    s_matching = matching
    s_engagement = min((likes + shares) / 1000, 1.0) * 100
    
    # La velocidad premia los tiempos cortos (menor a 2 horas)
    if velocidad is not None and velocidad > 0:
        s_velocidad = max(0, (1 - velocidad / 120)) * 100
    else:
        s_velocidad = 0.0

    score = (s_alcance * 0.35 + s_usuarios * 0.20 + s_matching * 0.15 + s_velocidad * 0.15 + s_engagement * 0.15)
    return round(min(score, 100.0), 2)

def _nivel_impacto(score: float) -> str:
    """Clasifica el impacto mediante un label visual."""
    if score >= 75: return "🔴 Muy Alto"
    if score >= 50: return "🟠 Alto"
    if score >= 25: return "🟡 Medio"
    return "🟢 Bajo"

def _formato_tiempo(minutos: float) -> str:
    """Formatea minutos a un string legible."""
    if minutos < 1: return f"{round(minutos * 60)} seg"
    if minutos < 60: return f"{round(minutos, 1)} min"
    if minutos < 1440: return f"{round(minutos / 60, 1)} horas"
    return f"{round(minutos / 1440, 1)} días"

def _determinar_arquetipo(alcance: int, profundidad: int, directas: int) -> str:
    """
    Clasifica la topología de la propagación en arquetipos creativos.
    """
    if alcance == 0:
        return "Germinal (Sin eco)"
    
    # 1. Caso Estrella: Muchas respuestas pero nadie habla entre sí
    if directas > 5 and profundidad <= 2:
        return "Estrella (Ruido Efímero)"
    
    # 2. Caso Hilo: Poca gente pero mucha profundidad
    if profundidad > 4 and directas < 5:
        return "Hilo Crítico (Debate Profundo)"
    
    # 3. Caso Explosión: Mucho de todo
    if alcance > 20 and profundidad > 3:
        return "Explosión Viral (Ecosistema)"
    
    # 4. Por defecto
    if profundidad > 2:
        return "Ramificado (Conversación Estándar)"
        
    return "Reacción Simple"
