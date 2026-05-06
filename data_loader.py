import pandas as pd
import os
from typing import Optional

def load_dataset() -> Optional[pd.DataFrame]:
    """
    Carga el dataset .parquet desde la ruta configurada en la variable de entorno 
    DATASET_PATH o una ruta por defecto en 'data/'. 
    Normaliza los nombres de las columnas para asegurar compatibilidad.
    """
    # 1. Obtener la ruta del dataset (Prioridad: Variable de entorno > Default local)
    default_path = os.path.join("data", "Reto_data_20251023_122206.parquet")
    filepath = os.environ.get("DATASET_PATH", default_path)
    
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ Archivo no encontrado en: {filepath}. Intentando ruta relativa...")
            # Si la ruta de la env var falla, intentar con la default por si acaso
            filepath = default_path

        df = pd.read_parquet(filepath)
        
        # --- Normalizar nombres de columnas (lowercase, sin espacios) ---
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # --- Mapeo flexible de columnas comunes ---
        # Incluye aliases del dataset real (Brandwatch/Talkwalker style: sin underscores)
        column_aliases = {
            "post_id":     ["post_id", "id", "message_id", "msg_id", "tweet_id", "preciseid"],
            "parent_id":   ["parent_id", "parentid", "reply_to", "in_reply_to", "parent"],
            "user_id":     ["user_id", "author", "author_id", "authorid", "user"],
            "text":        ["text", "content", "message", "body", "tweet", "description"],
            "timestamp":   ["timestamp", "createdat", "created_at", "date", "datetime", "time"],
            "likes":       ["likes", "influencescore", "like_count", "favorites", "reactions", "rating"],
            "shares":      ["shares", "retweets", "reshares", "share_count"],
            "platform":    ["platform", "socialtype", "network", "source", "red", "sourcename"],
        }

        rename_map = {}
        for canonical, aliases in column_aliases.items():
            for alias in aliases:
                if alias in df.columns and canonical not in df.columns:
                    rename_map[alias] = canonical
                    break

        df = df.rename(columns=rename_map)

        # --- Limpieza y Conversión de tipos ---
        if "timestamp" in df.columns:
            # Soporta: ISO string, epoch segundos y epoch milisegundos
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            if ts.notna().any():
                # Epoch en milisegundos (> 1e10) → convertir a segundos
                ts = ts.where(ts < 1e10, ts / 1000)
                df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
                # Rellenar los que no son numéricos con parse de string
                mask = df["timestamp"].isna()
                if mask.any():
                    df.loc[mask, "timestamp"] = pd.to_datetime(
                        df.loc[mask, "timestamp"].astype(str), errors="coerce", utc=True
                    )
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        if "likes" in df.columns:
            df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
            
        if "post_id" in df.columns:
            df["post_id"] = df["post_id"].astype(str).str.strip()

        if "parent_id" in df.columns:
            df["parent_id"] = df["parent_id"].astype(str).str.strip()
            df["parent_id"] = df["parent_id"].replace({"nan": None, "None": None, "": None, "NaN": None})

        print(f"✅ Dataset cargado y normalizado. Filas: {len(df)}")
        return df

    except Exception as e:
        print(f"❌ Error crítico al cargar el dataset: {e}")
        return None

# Variable global (Singleton)
dataframe_principal = load_dataset()
