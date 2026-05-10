# Explicación Completa del Proyecto: Agente Conversacional de Análisis Digital

---

## 1. ¿Qué problema resolvemos?

Las empresas y analistas tienen que procesar miles de publicaciones y comentarios de redes sociales para entender qué está pasando con una marca, un tema o un evento. Hacerlo manualmente es imposible.

**Nuestra solución:** un sistema donde el analista simplemente escribe una pregunta en lenguaje natural y el sistema automáticamente:
1. Entiende qué tipo de análisis necesita la pregunta
2. Ejecuta el análisis correcto sobre los datos reales
3. Devuelve una respuesta clara y estructurada

No requiere saber de programación, APIs ni modelos de lenguaje.

---

## 2. Vista General del Sistema

El sistema está compuesto por **tres procesos** que corren en paralelo y se comunican entre sí:

```
╔══════════════════════════════════════════════════════════════════╗
║                    PROCESO 1: Streamlit                          ║
║                    (puerto 8501 — lo que ve el usuario)          ║
║                                                                  ║
║   ┌─────────────────────────────────────────────────────────┐   ║
║   │  Interfaz de chat  │  Sidebar FinOps  │  Botones ejemplo │   ║
║   └──────────────────────────┬──────────────────────────────┘   ║
║                              │ llama a                           ║
║                    ┌─────────▼────────┐                         ║
║                    │  Agente LangGraph │  ← vive dentro de       ║
║                    │  (graph.py)       │    Streamlit            ║
║                    └─────────┬────────┘                         ║
╚══════════════════════════════╪═════════════════════════════════╝
                               │ HTTP (httpx)
                               │ con header X-Session-ID
╔══════════════════════════════╪═════════════════════════════════╗
║                    PROCESO 2: FastAPI                            ║
║                    (puerto 8000 — los microservicios)            ║
║                              │                                   ║
║            ┌─────────────────▼──────────────────┐               ║
║            │  /sentimientos  /resumen            │               ║
║            │  /propagacion   /metricas           │               ║
║            └─────────────────┬──────────────────┘               ║
║                              │ llama a                           ║
║              ┌───────────────┼────────────────┐                 ║
║              ▼               ▼                ▼                 ║
║         Gemini API       BFS Algorithm     Pandas               ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                    PROCESO 3: Gemini/OpenAI API                  ║
║                    (en la nube — el LLM externo)                 ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. El Flujo Completo: Paso a Paso

Cuando el usuario escribe *"¿Quiénes son los usuarios más influyentes?"*, esto es exactamente lo que pasa:

### Paso 1 — El usuario escribe en Streamlit

```
app.py recibe la pregunta desde st.chat_input()

Antes de invocar el agente, app.py hace dos cosas:
  1. SESSION_CTX.set(thread_id)  ← propaga el ID de sesión para FinOps
  2. Verifica que el presupuesto no esté agotado
```

### Paso 2 — Streamlit invoca el Agente LangGraph

```python
resultado = grafo.invoke(
    {"messages": [HumanMessage(content="¿Quiénes son los usuarios más influyentes?")]},
    config={"configurable": {"thread_id": "abc-123"}}
)
```

El grafo recupera el historial completo de esta sesión desde `MemorySaver` y lo combina con el nuevo mensaje.

### Paso 3 — El nodo `llamar_modelo` le pregunta al LLM

El LLM recibe:
```
[SystemMessage]  ← instrucciones del agente: qué herramientas tiene y cómo usarlas
[HumanMessage]   ← "¿Quiénes son los usuarios más influyentes?"
(+ historial de mensajes anteriores si los hay)
```

El LLM analiza la pregunta y decide: *"Esto requiere `tool_analizar_metricas`"*

Responde con un `AIMessage` que contiene `tool_calls`:
```json
{
  "tool_calls": [
    {"name": "tool_analizar_metricas", "args": {}}
  ]
}
```

También en este paso, el módulo FinOps registra cuántos tokens usó esta llamada al LLM.

### Paso 4 — El nodo `ejecutar_herramienta` llama a FastAPI

El `ToolNode` de LangGraph detecta que hay `tool_calls` y ejecuta la función:

```python
# En agent/tools.py
def tool_analizar_metricas():
    return _post("/analisis/metricas", {})
    # _post agrega automáticamente: headers={"X-Session-ID": "abc-123"}
```

Se hace una petición HTTP POST a `http://127.0.0.1:8000/analisis/metricas`.

### Paso 5 — FastAPI procesa la petición de métricas

```python
# En main.py
@app.post("/analisis/metricas")
async def analizar_metricas():
    df = dataframe_principal  # el dataset ya cargado en memoria

    influencers = df.groupby("user_id")["likes"].sum().nlargest(5)
    top_posts = df.nlargest(5, "likes")[["post_id", "likes", "text"]]
    plataformas = df["platform"].unique().tolist()

    return {
        "total_likes": int(df["likes"].sum()),
        "total_mensajes": len(df),
        "top_influencers": influencers.index.tolist(),
        "top_posts_por_likes": top_posts.to_dict(orient="records"),
        "plataformas": plataformas
    }
```

No llama al LLM. Es procesamiento de datos puro con Pandas. Rápido y económico.

### Paso 6 — El resultado vuelve al Agente como `ToolMessage`

LangGraph recibe el JSON de métricas y lo convierte en un `ToolMessage` que se agrega al historial de mensajes.

### Paso 7 — El LLM interpreta el resultado y responde en lenguaje natural

El LLM ahora recibe:
```
[SystemMessage]   ← instrucciones
[HumanMessage]    ← "¿Quiénes son los usuarios más influyentes?"
[AIMessage]       ← la decisión de llamar a tool_analizar_metricas
[ToolMessage]     ← {"top_influencers": ["user_7821", "user_3342", ...], "total_likes": 892341, ...}
```

El LLM genera la respuesta final en lenguaje natural:
> *"Los 5 usuarios más influyentes de la red son: user_7821 (líder con 45,230 likes acumulados), user_3342, user_9901, user_1104 y user_5567. La red tiene en total 892,341 likes distribuidos en 4,795 publicaciones, presentes en las plataformas Twitter, Facebook e Instagram."*

### Paso 8 — Streamlit muestra la respuesta

```python
for msg in resultado["messages"]:
    if isinstance(msg, ToolMessage):
        pasos_intermedios.append(...)  # para el toggle "mostrar pasos"
    elif isinstance(msg, AIMessage) and not msg.tool_calls:
        respuesta_final = msg.content   # ← esta es la que se muestra al usuario
```

---

## 4. Los 4 Servicios en Detalle

### Servicio 1: Análisis de Sentimientos

**Endpoint:** `POST /analisis/sentimientos`
**Cuándo lo usa el agente:** cuando el usuario pregunta por el "clima", "tono", "ambiente" o "sentimiento" de algo.

**Flujo interno:**
```
texto del usuario
    │
    ▼
nlp_service.analizar_sentimiento_llm()
    │
    ├── Construye un system_prompt:
    │   "Eres un analista de sentimientos. Devuelve SOLO JSON
    │    con claves: clima, score, justificacion..."
    │
    ├── Construye un user_prompt con el texto a analizar
    │
    ├── Llama a Gemini API (POST HTTP con urllib)
    │   maxOutputTokens: 250
    │   temperature: 0.2
    │   responseMimeType: "application/json"
    │
    ├── Gemini devuelve: {"clima": "positivo", "score": 0.85, ...}
    │
    ├── FinOps registra: X tokens entrada, Y tokens salida, costo $Z
    │
    └── Retorna el JSON validado
```

**Respuesta que genera:**
```json
{
  "clima": "positivo",
  "score": 0.85,
  "justificacion": "El texto expresa entusiasmo y satisfacción hacia el producto."
}
```

---

### Servicio 2: Resumen de Conversación

**Endpoint:** `POST /analisis/resumen`
**Cuándo lo usa el agente:** cuando el usuario quiere entender de qué trata una conversación sin leerla toda.

**Flujo interno:**
```
lista de textos
    │
    ├── Filtra textos vacíos
    ├── Une todos con separador "---"
    ├── Trunca a 12,000 caracteres (para no exceder el contexto del modelo)
    │
    ▼
nlp_service.resumir_conversacion_llm()
    │
    ├── Construye prompt pidiendo: resumen, temas_principales, posturas_clave
    ├── Llama a Gemini API
    │   maxOutputTokens: 1024
    │
    ├── Extrae y parsea el JSON de la respuesta
    │   (con _safe_json_parse que maneja markdown-wrapped JSON)
    │
    └── Retorna el resumen estructurado
```

**Respuesta que genera:**
```json
{
  "resumen": "La conversación gira en torno a las medidas económicas anunciadas...",
  "temas_principales": ["economía", "medidas gubernamentales", "protesta social"],
  "posturas_clave": ["Apoyo a las medidas", "Rechazo por impacto en clase media"],
  "alcance_textos": 42
}
```

---

### Servicio 3: Análisis de Propagación *(Obligatorio)*

**Endpoint:** `GET /analisis/propagacion?post_id=X`
**Cuándo lo usa el agente:** cuando el usuario pregunta por la viralidad o el impacto de un mensaje específico.

**Este servicio NO usa LLM. Es un algoritmo puro sobre el grafo de conversación.**

**¿Cómo funciona el algoritmo BFS?**

El dataset tiene publicaciones que responden a otras publicaciones (campo `parent_id`). Esto forma un árbol de conversación:

```
post_id: "c6adb46..."  ← mensaje original
    ├── reply_id: "a1b2c3..."  ← respuesta directa
    │       ├── reply_id: "d4e5f6..."  ← respuesta a la respuesta
    │       └── reply_id: "g7h8i9..."
    └── reply_id: "j0k1l2..."  ← otra respuesta directa
            └── reply_id: "m3n4o5..."
```

El algoritmo **BFS (Breadth-First Search)** recorre este árbol nivel por nivel:

```python
# Simplificación del algoritmo en propagacion_service.py
def calcular_propagacion(post_id):
    visitados = {post_id}
    cola = [post_id]
    todos_los_nodos = []

    while cola:
        nodo_actual = cola.pop(0)
        respuestas = df[df["parent_id"] == nodo_actual]  # ← busca replies

        for _, reply in respuestas.iterrows():
            if reply["post_id"] not in visitados:
                visitados.add(reply["post_id"])
                cola.append(reply["post_id"])
                todos_los_nodos.append(reply)

    return todos_los_nodos  # toda la cadena de propagación
```

**Métricas que calcula:**

| Métrica | Cálculo |
|---|---|
| **Alcance** | Cantidad total de nodos en toda la cadena |
| **Replies directas** | Solo el primer nivel de respuestas |
| **Usuarios únicos** | `len(set(nodo["user_id"] for nodo in cadena))` |
| **Velocidad media** | Promedio de minutos entre el post original y cada reply |
| **Contenido replicado** | % de palabras clave del post original que aparecen en los replies |

**Fórmula del Score de Impacto (0-100):**

```
score = (alcance_norm × 35)
      + (usuarios_norm × 20)
      + (contenido_norm × 15)
      + (velocidad_norm × 15)
      + (engagement_norm × 15)
```

Donde cada componente se normaliza entre 0 y 1 respecto al máximo del dataset.

**Niveles de impacto:**
- 🔴 **Muy Alto** (≥ 75): mensaje altamente viral con gran alcance y rápida propagación
- 🟠 **Alto** (50-74): propagación significativa
- 🟡 **Medio** (25-49): propagación moderada
- 🟢 **Bajo** (< 25): poca propagación

---

### Servicio 4: Métricas Generales

**Endpoint:** `POST /analisis/metricas`
**Cuándo lo usa el agente:** cuando el usuario quiere un panorama general de la red o preguntar por influencers.

**Flujo interno (100% Pandas, sin LLM):**

```python
# Total de likes en toda la red
total_likes = int(df["likes"].sum())

# Top 5 influencers: usuarios con más likes acumulados
influencers = df.groupby("user_id")["likes"].sum().nlargest(5)

# Top 5 posts más populares
top_posts = df.nlargest(5, "likes")[["post_id", "likes", "text"]]

# Plataformas presentes
plataformas = df["platform"].dropna().unique().tolist()
```

---

## 5. El Agente: Cerebro del Sistema

### ¿Qué es LangGraph?

LangGraph es un framework para construir agentes como **grafos de estados**. A diferencia de una simple cadena de pasos, un grafo puede:
- Tomar decisiones (edges condicionales)
- Repetir pasos (loops)
- Mantener memoria entre interacciones

### Estructura del Grafo

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    llamar_modelo    │  ← El LLM toma la decisión
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  ¿Hay tool_calls?   │
              └──────┬────────┬─────┘
                    SÍ        NO
                     │         │
                     ▼         ▼
        ┌──────────────────┐  ┌─────┐
        │ejecutar_herramienta│  │ END │
        └──────────┬─────────┘  └─────┘
                   │
                   │ resultado del tool
                   │
                   └──────────────────► (vuelve a llamar_modelo)
```

El loop puede repetirse múltiples veces. Por ejemplo:
1. Usuario pide análisis de sentimiento Y propagación en la misma pregunta
2. Primera vuelta: LLM decide llamar `tool_analizar_sentimiento`
3. Recibe resultado, segunda vuelta: LLM decide llamar `tool_analizar_propagacion`
4. Recibe resultado, tercera vuelta: LLM ya no necesita más herramientas → respuesta final

### Las 4 Herramientas (@tool)

Cada herramienta tiene una descripción en lenguaje natural que el LLM lee para decidir cuándo usarla:

```python
@tool
def tool_analizar_metricas() -> Dict:
    """
    Obtiene las métricas generales de toda la red: total de likes y top 5
    usuarios más influyentes. Úsala cuando el usuario quiera un panorama
    general de la actividad o pregunte por influencers o estadísticas globales.
    """
    return _post("/analisis/metricas", {})
```

La descripción del `@tool` es literalmente lo que el LLM lee para decidir. Si la pregunta menciona "influencer", "métricas", "estadísticas globales" → el LLM elige `tool_analizar_metricas`.

### Memoria Multi-turno con MemorySaver

```
Sesión del usuario (thread_id = "abc-123")
│
├── Turno 1: "¿Cuál es el clima de la red?"
│   → LLM llama tool_analizar_sentimiento
│   → Respuesta: "El clima es positivo con score 0.78"
│   → [guardado en MemorySaver]
│
├── Turno 2: "¿Y quiénes son los influencers?"
│   → LLM recupera el historial completo
│   → LLM llama tool_analizar_metricas
│   → Respuesta: "Los top influencers son..."
│   → [guardado en MemorySaver]
│
└── Turno 3: "¿El sentimiento que me diste antes era positivo?"
    → LLM lee el historial: recuerda turno 1
    → Responde SIN llamar herramienta: "Sí, el análisis de sentimiento
      del turno 1 mostró clima positivo con score 0.78"
```

Cada `thread_id` es un UUID generado por Streamlit al iniciar la sesión del navegador. Dos usuarios distintos nunca comparten memoria.

---

## 6. El Dataset y su Carga

### Formato y Estructura

El archivo `Reto_data_20251023_122206.parquet` es un formato columnar binario (más eficiente que CSV). Contiene **4,795 registros** de conversaciones digitales reales.

### Normalización Automática de Columnas

El dataset viene en formato Brandwatch/Talkwalker con nombres de columnas propios. `data_loader.py` los mapea automáticamente a nombres canónicos:

```python
column_aliases = {
    "post_id":   ["post_id", "id", "preciseid", "tweet_id", ...],
    "parent_id": ["parent_id", "parentid", "in_reply_to", ...],
    "user_id":   ["user_id", "authorid", "author", ...],
    "text":      ["text", "content", "message", "description", ...],
    "timestamp": ["timestamp", "createdat", "created_at", "date", ...],
    "likes":     ["likes", "influencescore", "like_count", "reactions", ...],
    "platform":  ["platform", "socialtype", "network", "source", ...],
}
```

Esto hace el sistema flexible: si en el futuro el dataset cambia de proveedor (de Brandwatch a Talkwalker), solo hay que agregar el alias nuevo en este diccionario.

### Conversión de Timestamps

El dataset puede traer fechas en múltiples formatos. El data_loader las convierte todas a UTC:

```python
# Soporta epoch en milisegundos (1698012345678)
# Soporta epoch en segundos    (1698012345)
# Soporta strings ISO          ("2023-10-22T14:32:00Z")
```

### Singleton Global

El dataset se carga **una sola vez** al iniciar FastAPI y queda en memoria como variable global `dataframe_principal`. Esto evita releer el archivo en cada request (sería muy lento). Los 4 servicios comparten este mismo DataFrame.

---

## 7. FinOps: Control de Costos de LLM

### ¿Por qué es necesario?

Los LLMs cobran por cada token procesado. Un token es aproximadamente 4 caracteres de texto. Sin control:
- Una sesión larga puede costar varios dólares sin que el usuario lo sepa
- En producción, múltiples usuarios pueden generar costos inesperados
- Es imposible auditar qué análisis fue el más caro

**FinOps** (Financial Operations) es la práctica de gestionar y optimizar los costos de servicios cloud/IA.

### Tabla de Precios Implementada

```python
PRICING = {
    "gemini-1.5-flash":  {"input": $0.075/1M,  "output": $0.30/1M},
    "gemini-2.5-flash":  {"input": $0.15/1M,   "output": $0.60/1M},
    "gpt-4o-mini":       {"input": $0.15/1M,   "output": $0.60/1M},
    ...
}
```

### ¿Dónde se capturan los tokens?

Las APIs de LLM siempre devuelven el conteo de tokens en su respuesta. Aprovechamos eso:

**En los servicios NLP (FastAPI) — llamadas HTTP directas:**
```python
# Gemini devuelve:
data["usageMetadata"]["promptTokenCount"]      # tokens de entrada
data["usageMetadata"]["candidatesTokenCount"]  # tokens de salida

# OpenAI devuelve:
data["usage"]["prompt_tokens"]
data["usage"]["completion_tokens"]
```

**En el Agente (LangGraph) — a través de LangChain:**
```python
# Después de cada llamada al LLM:
response.response_metadata["usage_metadata"]["prompt_token_count"]
response.response_metadata["usage_metadata"]["candidates_token_count"]
```

### ¿Cómo viaja el Session ID entre procesos?

El reto técnico fue asociar el costo de los servicios FastAPI a la sesión correcta de Streamlit, ya que son **procesos separados** que no comparten memoria.

**Solución: Header HTTP `X-Session-ID`**

```
app.py (Streamlit)
  │
  ├── SESSION_CTX.set("abc-123")   ← guarda el thread_id en un ContextVar
  │
  └── grafo.invoke(...)
          │
          └── tools.py
                  │
                  └── httpx.post("/analisis/sentimientos",
                                  headers={"X-Session-ID": "abc-123"})
                                                │
                                                ▼
                                          FastAPI
                                          lee el header
                                          pasa session_id="abc-123"
                                          a nlp_service.log_call("abc-123", ...)
```

### El Archivo de Log `data/finops_log.json`

Cada llamada LLM queda registrada:

```json
{
  "sessions": {
    "abc-123": {
      "total_cost_usd": 0.00089,
      "total_tokens_in": 1240,
      "total_tokens_out": 480,
      "calls": [
        {
          "timestamp": "2026-05-06T03:12:45Z",
          "service": "agente",
          "model": "gemini-1.5-flash",
          "tokens_in": 312,
          "tokens_out": 87,
          "cost_usd": 0.0000492
        },
        {
          "timestamp": "2026-05-06T03:12:47Z",
          "service": "sentimiento",
          "model": "gemini-1.5-flash",
          "tokens_in": 928,
          "tokens_out": 393,
          "cost_usd": 0.0000887
        }
      ]
    }
  },
  "global": {
    "total_cost_usd": 0.00412,
    "total_calls": 23
  }
}
```

### Dashboard en Streamlit

El sidebar muestra en tiempo real:

```
💰 FinOps Dashboard
━━━━━━━━━━━━━━━━━━━━━━
████████░░░░  $0.012 / $0.05

⚠️ 80% del presupuesto consumido

Tokens entrada   │ Tokens salida
     1,240       │      480

Llamadas LLM esta sesión: 6

▼ Estadísticas globales
  Costo total histórico: $0.041
  Total llamadas: 47
━━━━━━━━━━━━━━━━━━━━━━
```

### Budget Governor

Si el costo de la sesión supera `SESSION_BUDGET` (configurado en `.env`):

```python
if stats["total_cost_usd"] >= SESSION_BUDGET:
    # Muestra mensaje de error en el chat
    # Detiene la ejecución
    # El usuario debe limpiar la conversación para continuar
    st.stop()
```

Esto previene que una sesión sin control consuma recursos ilimitados.

---

## 8. Decisiones Técnicas Importantes

### ¿Por qué LangGraph en lugar de una cadena simple?

| Cadena simple (LangChain) | Grafo (LangGraph) |
|---|---|
| Pasos fijos y predefinidos | Puede decidir qué herramienta usar |
| No puede iterar | Puede hacer múltiples llamadas a tools |
| Sin memoria built-in | MemorySaver para multi-turno |
| No puede manejar estados complejos | Estado como grafo → flexible |

LangGraph nos da el **+10% del bono** en la evaluación por usar un framework avanzado con estado.

### ¿Por qué FastAPI separado de Streamlit?

Los servicios analíticos son **microservicios independientes**. Esto significa:
- Se pueden escalar por separado (si hay muchas peticiones de sentimiento, se puede replicar solo ese servicio)
- Se pueden probar individualmente desde Swagger (`/docs`)
- Se pueden reemplazar sin tocar la interfaz de usuario
- El agente puede estar en Streamlit y los servicios en un servidor separado simplemente cambiando `API_BASE_URL` en el `.env`

### ¿Por qué Pydantic en los schemas?

Pydantic valida automáticamente que los datos entrantes y salientes tengan el formato correcto. Si un servicio devuelve un campo inesperado o le falta uno, FastAPI lanza un error claro en lugar de pasar datos corruptos al agente.

### ¿Por qué urllib en lugar del SDK de Gemini?

Los servicios NLP (`nlp_service.py`) usan `urllib.request` (librería estándar de Python, sin dependencias externas) para llamar directamente a la API de Gemini. Esto:
- Evita instalar el SDK oficial de Gemini solo para los servicios
- Da control total sobre el payload (incluyendo `thinkingConfig`, `responseMimeType`, etc.)
- Es más liviano

El agente, en cambio, sí usa el SDK de LangChain porque necesita la integración con LangGraph.

---

## 9. Tecnologías y Por Qué Cada Una

| Tecnología | Versión | Para qué se usa | Por qué se eligió |
|---|---|---|---|
| **LangGraph** | ≥ 0.2.0 | Framework del agente con estado | Permite grafos con loops, memoria y tool-calling nativo |
| **LangChain** | ≥ 0.3.0 | Integración LLM en el agente | Abstrae Gemini/OpenAI con la misma interfaz |
| **FastAPI** | 0.136.0 | Servidor de microservicios | Rápido, async-first, genera docs automáticamente |
| **Streamlit** | ≥ 1.40.0 | Interfaz de chat | Permite hacer apps web de datos en Python puro |
| **Pandas** | 3.0.2 | Análisis del dataset | Estándar industria para datos tabulares en Python |
| **PyArrow** | 24.0.0 | Leer archivos .parquet | Formato columnar eficiente para datasets grandes |
| **Pydantic** | 2.13.3 | Validación de schemas | Validación automática de tipos en FastAPI |
| **httpx** | ≥ 0.27.0 | HTTP client en tools | Más moderno que requests, soporte async |
| **python-dotenv** | 1.2.2 | Variables de entorno | Separa configuración del código |
| **pytest** | ≥ 9.0.0 | Tests automatizados | 23 tests que verifican tools, grafo y memoria |

---

## 10. Cómo Correr el Proyecto

### Requisitos previos
- Python 3.11+
- Una API key de Google Gemini (gratis en [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Configuración del .env
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=tu-api-key-aqui
GEMINI_MODEL=gemini-1.5-flash   ← 1,500 req/día gratuitos
API_BASE_URL=http://127.0.0.1:8000
SESSION_BUDGET=0.05              ← presupuesto máximo por sesión en USD
```

### Levantar el sistema (dos terminales)

```bash
# Terminal 1 — Backend FastAPI
source venvIA2/bin/activate
uvicorn main:app --reload --port 8000
# Swagger: http://127.0.0.1:8000/docs

# Terminal 2 — Frontend Streamlit
source venvIA2/bin/activate
streamlit run app.py
# Chat: http://localhost:8501
```

### Verificar que todo funciona
```bash
pytest tests/ -v
# Resultado esperado: 23 passed
```

### Preguntas de prueba para la demo

| Pregunta | Servicio que activa |
|---|---|
| "¿Cuál es el clima general de la conversación?" | Sentimientos |
| "Dame un resumen de los primeros comentarios" | Resumen |
| "¿Quiénes son los 5 usuarios más influyentes?" | Métricas |
| "¿Qué impacto tuvo el mensaje con ID [post_id real]?" | Propagación |
| "¿El sentimiento que analizaste antes era positivo?" | Ninguno (usa memoria) |

---

## 11. Estructura de Carpetas

```
reto_icesi_prueba/
├── app.py                    ← Interfaz Streamlit + FinOps dashboard
├── main.py                   ← FastAPI: 3 endpoints + CORS
├── data_loader.py            ← Carga y normaliza el dataset .parquet
├── schemas.py                ← Modelos Pydantic (validación)
├── requirements.txt
├── .env                      ← Config local (no en git)
├── .env.example              ← Plantilla de configuración
│
├── agent/
│   ├── graph.py              ← Grafo LangGraph + MemorySaver + FinOps
│   └── tools.py              ← 4 @tool functions que llaman a FastAPI
│
├── services/
│   ├── nlp_service.py        ← Llama a Gemini/OpenAI para NLP + FinOps
│   ├── propagacion_service.py← Algoritmo BFS sobre el dataset
│   └── finops_service.py     ← Tracking de tokens, costos y log JSON
│
├── routers/
│   └── propagacion_endpoint.py ← GET /analisis/propagacion
│
├── data/
│   ├── Reto_data_20251023_122206.parquet  ← Dataset (4,795 registros)
│   └── finops_log.json       ← Log de costos (se crea automáticamente)
│
└── tests/
    ├── test_tools.py         ← 12 tests de las herramientas del agente
    └── test_graph.py         ← 11 tests del grafo y memoria multi-turno
```
