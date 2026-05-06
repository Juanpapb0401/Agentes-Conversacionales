# Agente Conversacional — Arquitectura y Documentación Técnica

## Reto ICESI · Análisis de Conversaciones Digitales

---

## 1. Resumen del Sistema

Agente conversacional multi-turno que responde preguntas en lenguaje natural sobre un dataset de redes sociales (formato Brandwatch/Talkwalker, 4 795 mensajes). El usuario escribe una pregunta; el agente decide qué microservicio invocar, lo llama, y sintetiza la respuesta en lenguaje natural.

**Stack principal:**

| Capa | Tecnología | Versión |
|---|---|---|
| Interfaz de usuario | Streamlit | 1.x |
| Agente conversacional | LangGraph + LangChain | 1.1.9 |
| LLM | Google Gemini 3.1 Flash Lite Preview | API v1beta |
| Backend analítico | FastAPI | 0.11x |
| Análisis de datos | Pandas + NumPy | — |
| Dataset | Apache Parquet (Brandwatch) | 4 795 filas |
| Persistencia de memoria | MemorySaver (RAM) | LangGraph built-in |

---

## 2. Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIO (Navegador)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP / WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STREAMLIT UI  (app.py :8501)                    │
│                                                                  │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  Chat Input     │   │  Historial UI    │   │   Sidebar    │ │
│  │  (pregunta)     │   │  (renderizado)   │   │  FinOps +    │ │
│  └────────┬────────┘   └──────────────────┘   │  Ejemplos    │ │
│           │                                    └──────────────┘ │
└───────────┼─────────────────────────────────────────────────────┘
            │  grafo.invoke(HumanMessage, thread_id)
            ▼
┌─────────────────────────────────────────────────────────────────┐
│              LANGGRAPH AGENT  (agent/graph.py)                   │
│                                                                  │
│   [START]                                                        │
│      │                                                           │
│      ▼                                                           │
│  ┌───────────────┐    tool_calls?    ┌──────────────────────┐   │
│  │ llamar_modelo │ ──────────────►  │ ejecutar_herramienta │   │
│  │  (Gemini LLM) │                  │     (ToolNode)        │   │
│  └───────┬───────┘ ◄─── resultado ──└──────────┬───────────┘   │
│          │                                      │               │
│   sin tool_calls                          httpx POST/GET        │
│          │                                      │               │
│        [END]                                    │               │
│                                                 ▼               │
│   MemorySaver ──── thread_id ──── historial por sesión          │
└─────────────────────────────────────────────────────────────────┘
                                                  │
                               HTTP (httpx, :8000)│
                                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FASTAPI BACKEND  (main.py :8000)                 │
│                                                                  │
│  POST /analisis/sentimientos    ──► nlp_service.py              │
│  POST /analisis/resumen         ──► nlp_service.py              │
│  GET  /analisis/propagacion     ──► propagacion_service.py      │
│  POST /analisis/metricas        ──► Pandas (main.py inline)     │
│                                                                  │
│  Middleware: CORS (puerto 8501 permitido)                        │
│  Middleware: X-Session-ID header para FinOps tracking           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌─────────────────────────────────────────┐
│  Gemini API       │   │  DATA LAYER  (data_loader.py)           │
│  (REST v1beta)    │   │                                         │
│                   │   │  Singleton: dataframe_principal         │
│  sentimientos     │   │  Parquet → DataFrame normalizado        │
│  resumen          │   │  Aliases: id→post_id, parentid→         │
│                   │   │   parent_id, author→user_id,            │
└───────────────────┘   │   createdat→timestamp (epoch ms),       │
                        │   influencescore→likes,                 │
                        │   socialtype→platform                   │
                        └─────────────────────────────────────────┘
```

---

## 3. Diagrama de Flujo de una Consulta

```
Usuario escribe: "¿Qué tan viral fue el mensaje con ID 199219160505_...?"
        │
        ▼
Streamlit captura el texto → crea HumanMessage
        │
        ▼
grafo.invoke() con thread_id único por sesión
        │
        ▼
Nodo "llamar_modelo":
  - Construye: [SystemMessage + historial + HumanMessage]
  - Envía a Gemini 3.1 Flash Lite
  - Gemini responde con tool_call: tool_analizar_propagacion("199219160505_...")
        │
        ▼
Nodo "ejecutar_herramienta" (ToolNode):
  - Llama tool_analizar_propagacion via httpx
  - httpx GET http://127.0.0.1:8000/analisis/propagacion?post_id=...
        │
        ▼
FastAPI /analisis/propagacion:
  - propagacion_service.analizar_propagacion(post_id, df)
  - BFS sobre el grafo de conversaciones
  - Calcula alcance, velocidad, content matching, score
  - Retorna JSON con ~20 campos
        │
        ▼
ToolMessage con JSON de resultados → regresa al grafo
        │
        ▼
Nodo "llamar_modelo" (segunda vuelta):
  - Gemini recibe el JSON del tool
  - Sintetiza respuesta en lenguaje natural en español
  - Retorna AIMessage (sin tool_calls)
        │
        ▼
Streamlit:
  1. Muestra bloque "Mensaje original" con texto/autor/fecha
  2. Muestra respuesta final del agente
  3. (Opcional) Muestra paso intermedio con JSON raw
  4. Guarda todo en historial_ui → persiste entre rerenders
```

---

## 4. Los Cuatro Microservicios

### 4.1 Análisis de Sentimiento — `POST /analisis/sentimientos`

**¿Qué hace?** Clasifica el clima emocional de un texto.

**Input:**
```json
{ "text": "El gobierno decepcionó a todos con esta decisión" }
```

**Output:**
```json
{
  "clima": "negativo",
  "score": 0.1,
  "justificacion": "El verbo 'decepcionar' indica insatisfacción clara..."
}
```

**Cómo funciona:** Llama a Gemini con `responseMimeType: "application/json"` y un prompt que obliga la respuesta estructurada. El `score` va de 0 (muy negativo) a 1 (muy positivo).

---

### 4.2 Resumen de Conversación — `POST /analisis/resumen`

**¿Qué hace?** Resume una lista de textos identificando temas y posturas.

**Input:**
```json
{
  "textos": ["El gobierno anunció medidas", "La gente está molesta", "Habrá protestas"],
  "max_palabras": 120,
  "idioma": "es"
}
```

**Output:**
```json
{
  "resumen": "La comunidad reacciona negativamente ante medidas gubernamentales...",
  "temas_principales": ["protesta social", "decisión gubernamental"],
  "posturas_clave": ["desaprobación ciudadana", "llamado a la acción"],
  "alcance_textos": 3
}
```

**Cómo funciona:** Un solo prompt a Gemini con todos los textos concatenados. `thinkingBudget: 0` desactiva el razonamiento interno para ahorrar tokens.

---

### 4.3 Análisis de Propagación — `GET /analisis/propagacion?post_id=X`

**¿Qué hace?** Mide la viralidad de un mensaje trazando su árbol de respuestas.

**Output:**
```json
{
  "id_original": "199219160505_1274366331365120",
  "encontrado": true,
  "texto_original": "Texto del mensaje...",
  "alcance": 3,
  "replies_directas": 2,
  "cadena_total_nodos": 2,
  "profundidad_maxima": 1,
  "usuarios_unicos_en_cadena": 2,
  "velocidad_media_min": 19.7,
  "velocidad_media_label": "19.7 min",
  "porcentaje_contenido_replicado": 33.33,
  "score_impacto": 17.94,
  "nivel_impacto": "🟢 Bajo",
  "arquetipo": "Reacción Simple"
}
```

**Algoritmo BFS:**
```
1. Construir mapa { parent_id → [hijos] } via groupby (O(n))
2. BFS desde post_id → lista de todos los descendientes
3. Calcular métricas sobre el subgrafo encontrado
```

**Fórmula del Score de Impacto (0–100):**

```
Score = 0.35·S_alcance + 0.20·S_usuarios + 0.15·S_contenido + 0.15·S_velocidad + 0.15·S_engagement

Donde:
  S_alcance    = min(alcance / 500, 1) × 100
  S_usuarios   = min(usuarios_únicos / 200, 1) × 100
  S_contenido  = % palabras del original repetidas en réplicas
  S_velocidad  = max(0, 1 − velocidad_min / 120) × 100   ← premia respuestas < 2h
  S_engagement = min((likes + shares) / 1000, 1) × 100
```

**Arquetipos de propagación:**

| Arquetipo | Condición | Significado |
|---|---|---|
| Germinal (Sin eco) | alcance = 0 | Sin ninguna respuesta |
| Reacción Simple | profundidad ≤ 2, pocas directas | Eco mínimo |
| Estrella (Ruido Efímero) | directas > 5 y profundidad ≤ 2 | Muchos responden pero nadie continúa |
| Hilo Crítico (Debate Profundo) | profundidad > 4 y directas < 5 | Debate anidado entre pocos |
| Explosión Viral | alcance > 20 y profundidad > 3 | Alta difusión en múltiples capas |
| Ramificado | profundidad > 2 | Conversación estándar |

**Niveles de impacto:**

| Score | Nivel |
|---|---|
| ≥ 75 | 🔴 Muy Alto |
| 50–74 | 🟠 Alto |
| 25–49 | 🟡 Medio |
| < 25 | 🟢 Bajo |

---

### 4.4 Métricas Generales — `POST /analisis/metricas`

**¿Qué hace?** Estadísticas agregadas de toda la red.

**Output:**
```json
{
  "total_likes": 6197,
  "total_mensajes": 4795,
  "top_influencers": ["@grok", "@NoticiasCaracol", "@merryluz7771", "@lasillavacia", "@X"],
  "top_posts_por_likes": [
    { "post_id": "...", "likes": 450, "text": "Texto truncado a 100 chars..." }
  ],
  "plataformas": ["Twitter", "Facebook", "Instagram"]
}
```

**Cómo funciona:** Pandas puro — no usa LLM. `groupby(user_id)[likes].sum().nlargest(5)` y `nlargest(5, "likes")`.

---

## 5. El Agente LangGraph

### Grafo de estados

```
MessagesState = { messages: List[BaseMessage] }

Nodos:
  llamar_modelo     → invoca LLM con historial completo
  ejecutar_herramienta → ToolNode ejecuta la tool seleccionada

Aristas condicionales:
  tools_condition: si AIMessage tiene tool_calls → ejecutar_herramienta
                   si no                         → END
```

### Memoria multi-turno

`MemorySaver` persiste el historial de mensajes en RAM keyed por `thread_id`. Cada pestaña del navegador genera un UUID diferente, creando sesiones independientes. El historial se pasa completo al LLM en cada invocación: `[SystemMessage] + state["messages"]`.

### System Prompt

Define el rol del agente, las 4 tools disponibles, las instrucciones de comportamiento (idioma, precisión, nunca inventar datos) y el manejo de errores (qué hacer si `post_id` no existe en el dataset).

---

## 6. Capa de Datos

### Dataset (Brandwatch format)

| Columna original | Columna canónica | Tipo | Descripción |
|---|---|---|---|
| `id` | `post_id` | str | Identificador único del mensaje |
| `parentid` | `parent_id` | str / None | ID del mensaje al que responde |
| `author` | `user_id` | str | Nombre de usuario (@handle) |
| `createdat` | `timestamp` | datetime (UTC) | Época en milisegundos → convertida |
| `influencescore` | `likes` | int | Score de influencia del autor |
| `socialtype` | `platform` | str | Red social (Twitter, Facebook, etc.) |
| `text` | `text` | str | Contenido del mensaje |

### Normalización automática

`data_loader.py` aplica:
1. Lowercase + reemplazo de espacios en nombres de columnas
2. Mapeo de aliases por prioridad (lista ordenada: primer alias encontrado gana)
3. Conversión de epoch ms → `datetime` UTC (`ts / 1000` cuando `ts > 1e10`)
4. Limpieza de `parent_id`: `"nan"`, `"None"`, `""` → `None`

---

## 7. FinOps — Control de Costos

`services/finops_service.py` registra cada llamada LLM:

- **`SESSION_CTX`**: `contextvars.ContextVar` propagado vía header `X-Session-ID`
- **`log_call(thread_id, servicio, modelo, tokens_in, tokens_out)`**: acumula en dict en RAM
- **Presupuesto por sesión**: configurable con `SESSION_BUDGET=0.05` en `.env`
- **Dashboard**: barra de progreso en sidebar de Streamlit, tokens entrada/salida, costo total

Tarifas hardcodeadas en el servicio según el modelo activo.

---

## 8. Ejemplos para Testear

### Sentimiento

```
Analiza el sentimiento de: "El gobierno decepcionó a todos con esta decisión"
```
```
¿Este mensaje es positivo o negativo?: "Excelente trabajo del equipo, superaron las expectativas"
```
```
¿Cuál es el tono emocional de: "La situación es preocupante pero hay esperanza"?
```

### Resumen

```
Resume esta conversación: ["El gobierno anunció nuevas medidas", "La gente está molesta", "Habrá protestas mañana", "El presidente dará rueda de prensa"]
```
```
¿De qué tratan estos mensajes?: ["Gana Colombia!", "Gran partido de fútbol", "El árbitro fue injusto", "Merecíamos ganar"]
```

### Propagación (usar IDs reales del dataset)

```
¿Qué tan viral fue el mensaje con ID 199219160505_1274366331365120?
```
```
Analiza la propagación del post 199219160505_1274366331365120
```
```
¿Cuál fue el alcance del mensaje 199219160505_1274366331365120?
```

> **Nota:** Los `post_id` son strings alfanuméricos del dataset Brandwatch. Para encontrar más IDs con alta actividad, pregunta primero por las métricas generales.

### Métricas generales

```
¿Quiénes son los usuarios más influyentes?
```
```
Dame las métricas generales de la red
```
```
¿Cuántos mensajes tiene el dataset y cuáles son las plataformas?
```

### Consultas multi-herramienta (el agente encadena tools)

```
¿Quiénes son los influencers y qué tono tiene la conversación sobre el gobierno?
```
```
Resume la conversación y dime si el sentimiento es positivo o negativo: ["El congreso aprobó la ley", "Muchos ciudadanos en desacuerdo", "Las protestas continúan"]
```

---

## 9. Cómo Correr el Proyecto

```bash
# 1. Clonar y activar entorno
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu GEMINI_API_KEY

# 4. Colocar el dataset
# Copiar Reto_data_20251023_122206.parquet en data/

# Terminal 1 — Backend FastAPI
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend Streamlit
streamlit run app.py

# Tests
pytest tests/ -v
```

**Variables de entorno clave (`.env`):**

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-3.1-flash-lite-preview
API_BASE_URL=http://127.0.0.1:8000
SESSION_BUDGET=0.05
```

---

## 10. Estructura de Archivos

```
Agentes-Conversacionales/
│
├── app.py                          # Streamlit UI (Rol 4)
├── main.py                         # FastAPI app + endpoint métricas
├── schemas.py                      # Pydantic request/response models
├── data_loader.py                  # Carga y normalización del dataset
├── requirements.txt
├── .env / .env.example
│
├── agent/
│   ├── graph.py                    # LangGraph: nodos, grafo, MemorySaver
│   └── tools.py                    # 4 tools con @tool decorator (httpx → FastAPI)
│
├── routers/
│   └── propagacion_endpoint.py     # GET /analisis/propagacion
│
├── services/
│   ├── nlp_service.py              # Gemini API: sentimiento + resumen
│   ├── propagacion_service.py      # BFS + score de impacto + arquetipos
│   └── finops_service.py           # Tracking de tokens y costos
│
├── data/
│   └── *.parquet                   # Dataset (gitignored)
│
└── tests/
    ├── test_tools.py               # Tests de tools con respx mock
    └── test_graph.py               # Tests del grafo LangGraph
```

---

## 11. Decisiones de Diseño Relevantes

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| `MemorySaver` en RAM | Redis / base de datos | Simplicidad — no se requiere persistencia entre reinicios del servidor |
| `thinkingBudget: 0` en Gemini | Thinking habilitado | Los tokens de thinking consumen el presupuesto `maxOutputTokens` antes de responder |
| BFS con `groupby` | `iterrows()` fila por fila | `iterrows()` es ~100× más lento en DataFrames grandes |
| `contextvars.ContextVar` para FinOps | Variable global | Thread-safe: cada request FastAPI tiene su propio contexto |
| `author` antes que `authorid` en aliases | `authorid` primero | `authorid` en Brandwatch contiene IDs numéricos de Twitter; `author` contiene @handles |
| Singleton `dataframe_principal` | Cargar en cada request | El dataset es de solo lectura y carga en ~300ms — cachear elimina latencia |
