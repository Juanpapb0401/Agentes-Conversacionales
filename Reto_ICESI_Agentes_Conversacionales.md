# Reto ICESI: Agentes Conversacionales y Análisis de Conversaciones Digitales (LLM/NLP)

## Dataset
Puedes acceder al conjunto de datos en el siguiente enlace:
[Reto_data_20251023_122206.parquet](https://github.com/armandoordonez/AI-Engineering/blob/main/data/Reto_data_20251023_122206.parquet)

---

## 1. Introducción y Contexto
En la era digital, la capacidad de procesar y comprender grandes volúmenes de conversaciones en línea es crucial para la toma de decisiones estratégicas. Este reto busca simular un entorno de análisis de datos sociales reales, donde los participantes deberán construir un **Agente Conversacional inteligente** capaz de interactuar con microservicios especializados en el análisis de publicaciones y comentarios.

El objetivo final es crear un sistema donde un usuario pueda preguntar de forma natural (ej: “¿Cómo está el clima de la conversación?”) y el Agente Conversacional ejecute el pipeline de procesamiento analítico más adecuado para responder.

---

## 2. Los Análisis Disponibles y la Selección para el Reto
Atinna realiza una variedad de análisis avanzados sobre conversaciones digitales. El reto requiere que cada equipo desarrolle un total de **tres (3) servicios MCP**: dos seleccionados y uno obligatorio.

### 2.1. Análisis Disponibles para Selección (Escoger 2)
Los equipos deberán seleccionar dos (2) análisis de la siguiente lista para convertirlos en servicios MCP:

1.  **Topología de la red:** Analiza la interacción y conexión entre los comentarios y publicaciones (requiere análisis de grafos o jestructuras de reply/thread).
2.  **Resumen general de la conversación:** Generación de un resumen conciso sobre la temática principal y las posturas clave (requiere LLM para síntesis).
3.  **Análisis de sentimientos:** Determinación del clima semántico (positivo, negativo, neutral) de los comentarios (requiere LLM para inferencia).
4.  **Análisis de emociones:** Identificación de las emociones evocadas (Felicidad, Furia, Angustia, etc.) al leer el contenido (requiere LLM para clasificación detallada).
5.  **Análisis de métricas:** Identificación de actores influyentes, publicaciones con mayor repercusión, likes, y redes utilizadas (requiere procesamiento de datos tradicional).
6.  **Análisis geográfico:** Ubicación de dónde está ocurriendo la conversación (puntos cardinales) y su visualización potencial.

### 2.2. Análisis de Cumplimiento Obligatorio (El Servicio MCP #3)
Además de los dos análisis seleccionados, todos los equipos deberán desarrollar el siguiente servicio:

**Análisis de Propagación de un Mensaje (OBLIGATORIO)**
A partir de un ID de mensaje específico (ej: un post principal), el servicio debe analizar su propagación dentro de la red. Esto implica medir el alcance, la velocidad, y la presencia del contenido del mensaje original en las respuestas directas, comentarios o compartidos, con el fin de determinar el impacto mediático dentro de una conversación.

---

## 3. Estructura del Reto: Paso a Paso Detallado

### Fase 1: Ingeniería de Datos y Configuración del Entorno
1.  **Revisión del Dataset:** Analizar la estructura del dataset (campos clave: `post_id`, `user_id`, `text`, `likes`, `replies`, `timestamp`).
2.  **Preparación de Herramientas:** Configurar el entorno (se recomienda Python por librerías como Hugging Face, NLTK).
3.  **LLM Base:** Definir el modelo para inferencia (Google, OpenAI o modelos Open Source).

### Fase 2: Desarrollo y Exposición de Servicios Analíticos (MCP)
Por cada uno de los 3 análisis, se debe construir un servicio independiente (MCP):
1.  **Definición de la Función Analítica:** Implementar la lógica de procesamiento.
2.  **Exposición como Microservicio (MCP):** Utilizar frameworks como **FastMCP, FastAPI, o Flask**.

| Servicio MCP | Endpoint Sugerido | Respuesta JSON Típica |
| :--- | :--- | :--- |
| Análisis 1 (Seleccionado) | `/analisis/nombre_corto_1` | `{"resultado": "..."}` |
| Análisis 2 (Seleccionado) | `/analisis/nombre_corto_2` | `{"data": [ ... ]}` |
| Análisis 3 (Obligatorio) | `/analisis/propagacion` | `{"id_original": "...", "alcance": 500, "velocidad_media": "3 min"}` |

### Fase 3: Construcción del Agente Conversacional (Tool-Calling)
1.  **Selección del Framework:**
    * **Opción 1 (Avanzada):** LangGraph o LlamaIndex (Function/Tool Calling).
    * **Opción 2 (No-Code):** n8n o herramientas similares.
2.  **Definición de Herramientas (Tools):** Configurar el Agente para reconocer los servicios MCP de la Fase 2 mediante esquemas de función (Schema).
3.  **Lógica Conversacional:** El agente recibe la pregunta, decide la herramienta, ejecuta la llamada HTTP, procesa el JSON y genera una respuesta natural.

---

## 4. Criterios de Evaluación

| Criterio | Ponderación | Descripción |
| :--- | :---: | :--- |
| **Funcionalidad y Robustez** | 33% | Servicios MCP funcionales, rápidos y con formato JSON correcto. |
| **Calidad del Análisis** | 33% | Precisión, relevancia y sofisticación de los prompts o lógica de datos. |
| **Inteligencia del Agente** | 33% | Identificación de intención, selección de herramienta y respuesta natural. |
| **Uso de Frameworks Avanzados** | +10% | Uso efectivo de LangGraph, LlamaIndex o arquitecturas con estado. |

---

## Reto de Ingeniería: El Agente en Acción

### Ejemplos de Interacciones a Demostrar:
* **Usuario:** “Quiero saber cómo se ha propagado el mensaje con ID: 12345”
    * **Agente:** Llama al MCP de Propagación.
* **Usuario:** “¿Me puedes dar un resumen ejecutivo de la discusión sobre Temática XYZ?”
    * **Agente:** Llama al MCP de Resumen.
* **Usuario:** “Quiero saber la polaridad y el ambiente general en la red.”
    * **Agente:** Llama al MCP de Sentimientos.
* **Usuario:** “¿El sentimiento es positivo?”
    * **Agente:** Debe recordar el contexto o re-ejecutar el servicio para confirmar.
* **Usuario:** “¿Cuál es el post que más impacto ha tenido?”
    * **Agente:** Llama al MCP de Métricas de Influencia y extrae la información clave.
