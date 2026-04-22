import pandas as pd
import os

def load_dataset():
    # Ruta relativa a la carpeta data
    filepath = os.path.join("data", "Reto_data_20251023_122206.parquet")
    
    try:
        df = pd.read_parquet(filepath)
        print(f"✅ Dataset cargado correctamente. Total filas: {len(df)}")
        
        # Limpieza preventiva básica
        if 'likes' in df.columns:
            df['likes'] = df['likes'].fillna(0)
            
        return df
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        print("Asegúrate de que el archivo parquet esté dentro de la carpeta 'data'.")
        return None

# Variable global que usará tu API
dataframe_principal = load_dataset()