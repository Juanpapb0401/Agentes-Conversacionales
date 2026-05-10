# Reto ICESI — Agentes Conversacionales y Análisis de Conversaciones Digitales

**Universidad Icesi · Reto AI Engineering**  
Stack: Python · FastAPI · LangGraph · Gemini 2.5 Flash · Streamlit

---

## Arquitectura del Sistema

```mermaid
flowchart TD
    U([👤 Usuario]) -->|lenguaje natural| ST[Streamlit\napp.py]
    ST -->|HumanMessage + thread_id| AG[Agente LangGraph\nagent/graph.py]

    AG -->|system prompt +\nhistorial| LLM[Gemini 2.5 Flash\nLangChain]
    LLM -->|AIMessage con tool_calls| AG

    AG -->|ejecuta tool| TN[ToolNode\nagent/tools.py]

    TN -->|POST /analisis/sentimientos| S1[Servicio Sentimientos\nnlp_service.py]
    TN -->|POST /analisis/resumen| S2[Servicio Resumen\nnlp_service.py]
    TN -->|GET /analisis/propagacion| S3[Servicio Propagación\npropagacion_service.py]
    TN -->|POST /analisis/metricas| S4[Servicio Métricas\nmain.py]

    S1 & S2 -->|llamada REST| GAPI[Gemini API]
    S3 & S4 -->|Pandas queries| DS[(Dataset .parquet\ndata/)]

    TN -->|ToolMessage con JSON| AG
    AG -->|AIMessage final| ST
    ST -->|respuesta renderizada| U

    style AG fill:#4A90D9,color:#fff
    style LLM fill:#34A853,color:#fff
    style ST fill:#FF6B6B,color:#fff
    style DS fill:#F4B400,color:#000
```

---

## Flujo de una Consulta (Secuencia)

```mermaid
sequenceDiagram
    actor U as Usuario
    participant ST as Streamlit
    participant G as LangGraph Grafo
    participant LLM as Gemini 2.5 Flash
    participant T as FastAPI
    participant D as Dataset .parquet

    U->>ST: "¿Qué tan viral fue el post ID 1001?"
    ST->>G: invoke(HumanMessage, thread_id)
    G->>LLM: system_prompt + historial + tools
    LLM-->>G: tool_calls: [tool_analizar_propagacion(post_id=1001)]
    G->>T: GET /analisis/propagacion?post_id=1001
    T->>D: BFS + velocidad + score impacto
    D-->>T: métricas calculadas
    T-->>G: JSON {alcance, velocidad, score_impacto...}
    G->>LLM: historial + ToolMessage(resultado)
    LLM-->>G: "El post 1001 tuvo impacto Muy Alto..."
    G-->>ST: respuesta final
    ST-->>U: Respuesta en lenguaje natural
```

---

## Requisitos Previos

- Python 3.11 o superior
- API Key de Google Gemini → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (gratis)
- Dataset `.parquet` → [descargar aquí](https://github.com/armandoordonez/AI-Engineering/blob/main/data/Reto_data_20251023_122206.parquet)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Agentes-Conversacionales

# 2. Crear y activar el entorno virtual
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el dataset
mkdir data
# Copiar Reto_data_20251023_122206.parquet dentro de data/
```

---

## Configuración del Entorno

Crea el archivo `.env` en la raíz (nunca lo subas a git):

```env
# Proveedor LLM: "openai" o "gemini"
LLM_PROVIDER=gemini

# Google Gemini
GEMINI_API_KEY=tu-api-key-aqui
GEMINI_MODEL=gemini-2.5-flash

# OpenAI (alternativa)
OPENAI_API_KEY=sk-tu-key-aqui
OPENAI_MODEL=gpt-4o-mini

# URL base de la API (no cambiar si corres local)
API_BASE_URL=http://127.0.0.1:8000
```

---

## Cómo Correr la Aplicación

Se necesitan **dos terminales corriendo en paralelo**.

### Terminal 1 — API FastAPI (Backend MCP)

```bash
.\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
```

Swagger UI disponible en: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/analisis/sentimientos` | Clima emocional de un texto |
| `POST` | `/analisis/resumen` | Resumen y temas de una conversación |
| `GET` | `/analisis/propagacion?post_id=X` | Propagación e impacto de un mensaje |
| `POST` | `/analisis/metricas` | Estadísticas generales de la red |
| `GET` | `/scraping/tweets?query=X&n=20` | Tweets en tiempo real vía Twitter/X |

### Terminal 2 — Interfaz Streamlit (Frontend)

```bash
.\.venv\Scripts\streamlit.exe run app.py
```

Se abre en: [http://localhost:8501](http://localhost:8501)

---

## Despliegue con Docker

> Alternativa a la instalación manual. Requiere [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado.

### Requisitos previos

1. Tener el dataset en `data/Reto_data_20251023_122206.parquet`
2. Tener el archivo `.env` configurado en la raíz (igual que en el setup manual)
3. Para el scraping de Twitter, agregar en `.env`:
   ```env
   TWITTER_ACCOUNTS=usuario:contraseña:email@gmail.com
   ```

### Construir y levantar

```bash
# Desde la raíz del proyecto
docker compose -f docker/docker-compose.yml up --build
```

- API disponible en: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit disponible en: [http://localhost:8501](http://localhost:8501)

### Comandos útiles

```bash
# Detener todos los contenedores
docker compose -f docker/docker-compose.yml down

# Ver logs en tiempo real
docker compose -f docker/docker-compose.yml logs -f

# Reconstruir sin caché (después de cambiar código)
docker compose -f docker/docker-compose.yml up --build --force-recreate
```

### Cómo funciona internamente

```
[Navegador] ───► :8501 (agentes-app) ─► Streamlit + Agente LangGraph
                                         │
                                         │ HTTP interno
                                         ▼
                              :8000 (agentes-api) ─► FastAPI
                                         │
                                         ▼
                                   data/ (volumen compartido)
```

El contenedor `agentes-app` llama a `agentes-api` por su nombre de servicio interno (`http://api:8000`), no por `localhost`. Ambos comparten el volumen `data/` para acceder al dataset y a la base de datos de cuentas de twscrape.

---

## Estructura del Proyecto

```
Agentes-Conversacionales/
│
├── app.py                          # Interfaz Streamlit — entry point UI (Rol 4)
├── main.py                         # Servidor FastAPI — entry point API (Rol 1)
├── data_loader.py                  # Carga y normalización del .parquet (Rol 1)
├── schemas.py                      # Contratos Pydantic request/response (Rol 1)
├── requirements.txt                # Dependencias Python
├── .env                            # Variables de entorno (NO subir a git)
├── .env.example                    # Plantilla de variables de entorno
│
├── agent/                          # Capa de Orquestación — Rol 4
│   ├── tools.py                    # 5 tools @tool de LangChain (+ Twitter scraping)
│   └── graph.py                    # Grafo LangGraph + MemorySaver + FinOps
│
├── routers/                        # Endpoints adicionales de FastAPI
│   ├── propagacion_endpoint.py     # GET /analisis/propagacion (Rol 3)
│   └── scraping_endpoint.py        # GET /scraping/tweets
│
├── services/                       # Lógica de negocio
│   ├── nlp_service.py              # LLM: sentimientos + resumen (Rol 2)
│   ├── propagacion_service.py      # BFS + velocidad + score impacto (Rol 3)
│   └── scraping_service.py         # Twitter/X scraping con twscrape
│
├── pages/                          # Páginas adicionales de Streamlit
│   └── 1_Dashboard.py              # Dashboard visual (charts, scraping, métricas)
│
├── docker/                         # Configuración Docker
│   ├── Dockerfile.api              # Imagen del backend FastAPI
│   ├── Dockerfile.app              # Imagen del frontend Streamlit
│   └── docker-compose.yml          # Orquestación completa
│
├── docs/                           # Documentación complementaria
│   ├── ARQUITECTURA.md             # Diagramas y decisiones de diseño
│   ├── Explicacion.md              # Explicación narrativa del sistema
│   ├── contexto.md                 # Contexto del reto
│   └── Reto_ICESI_Agentes_Conversacionales.md
│
├── data/                           # Dataset (NO subir a git)
│   └── Reto_data_20251023_122206.parquet
│
└── tests/
    ├── test_tools.py               # 12 tests de tools
    └── test_graph.py               # 11 tests de grafo y memoria
```

---

## Correr los Tests

```bash
.\.venv\Scripts\pytest.exe tests/ -v
# Resultado esperado: 23 passed
```

---

## Ejemplos de Consultas al Agente

| Consulta | Tool invocada |
|---|---|
| "¿Cómo está el clima de esta conversación: '...'" | `tool_analizar_sentimiento` |
| "Resume estos comentarios: ['...', '...']" | `tool_resumir_conversacion` |
| "¿Qué tan viral fue el post con ID 1001?" | `tool_analizar_propagacion` |
| "¿Quiénes son los usuarios más influyentes?" | `tool_analizar_metricas` |
| "Busca 20 tweets recientes sobre #Colombia" | `tool_buscar_tweets` |

---

## Notas de Seguridad

- El `.env` está en `.gitignore` y **nunca debe subirse al repositorio**.
- Regenera tu API key si sospechas que fue expuesta: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
