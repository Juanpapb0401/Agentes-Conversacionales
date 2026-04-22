# Agentes-Conversacionales


Crear una carpeta data y dentro de ella colocar el archivo .parquet.

https://github.com/armandoordonez/AI-Engineering/blob/main/data/Reto_data_20251023_122206.parquet

## Variables de entorno para el Rol 2 (LLM)

Configura el proveedor y la API Key antes de ejecutar la API:

- `LLM_PROVIDER`: `openai` o `gemini`
- `OPENAI_API_KEY`: requerido si usas OpenAI
- `OPENAI_MODEL`: opcional (default: `gpt-4o-mini`)
- `GEMINI_API_KEY`: requerido si usas Gemini
- `GEMINI_MODEL`: opcional (default: `gemini-1.5-flash`)

Si usas un archivo `.env`, asegúrate de instalar `python-dotenv` y definir ahi las variables.

## Endpoints NLP

### Sentimientos

`POST /analisis/sentimientos`

Body ejemplo:

```json
{
	"text": "Me encanta el debate, pero hay opiniones muy fuertes."
}
```

### Resumen

`POST /analisis/resumen`

Body ejemplo:

```json
{
	"textos": [
		"Comentario 1...",
		"Comentario 2..."
	],
	"max_palabras": 120,
	"idioma": "es"
}
```
