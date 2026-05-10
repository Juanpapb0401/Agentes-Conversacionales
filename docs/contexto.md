
markdown_content = """# Documentación del Proyecto: Reto ICESI - Agentes Conversacionales y Análisis de Conversaciones Digitales (LLM/NLP)

## 1. Información General
* **Institución:** Universidad Icesi
* **Programa:** Ingeniería de Sistemas / Diseño de Medios Interactivos
* **Curso:** Reto AI Engineering

## 2. Especificaciones Oficiales del Reto

**Dataset:** [Reto_data_20251023_122206.parquet](https://github.com/armandoordonez/AI-Engineering/blob/main/data/Reto_data_20251023_122206.parquet)

### 2.1. Introducción y Contexto
En la era digital, la capacidad de procesar y comprender grandes volúmenes de conversaciones en línea es crucial para la toma de decisiones estratégicas. Este reto busca simular un entorno de análisis de datos sociales reales, donde los participantes deberán construir un Agente Conversacional inteligente capaz de interactuar con microservicios especializados en el análisis de publicaciones y comentarios.

El objetivo final es crear un sistema donde un usuario pueda preguntar de forma natural (ej: *"¿Cómo está el clima de la conversación?"*) y el Agente Conversacional ejecute el pipeline de procesamiento analítico más adecuado para responder.

### 2.2. Análisis Disponibles y Selección
El reto requiere desarrollar un total de **tres (3) servicios MCP**: dos seleccionados y uno obligatorio.

**Análisis Disponibles para Selección (Escoger 2):**
1. **Topología de la red:** Analiza la interacción y conexión entre los comentarios y publicaciones (requiere análisis de grafos o estructuras de reply/thread).
2. **Resumen general de la conversación:** Generación de un resumen conciso sobre la temática principal y las posturas clave (requiere LLM para síntesis).
3. **Análisis de sentimientos:** Determinación del clima semántico (positivo, negativo, neutral) de los comentarios (requiere LLM para inferencia).
4. **Análisis de emociones:** Identificación de las emociones evocadas (Felicidad, Furia, Angustia, etc.) al leer el contenido (requiere LLM para clasificación detallada).
5. **Análisis de métricas:** Identificación de actores influyentes, publicaciones con mayor repercusión, likes, y redes utilizadas (requiere procesamiento de datos tradicional).
6. **Análisis geográfico:** Ubicación de dónde está ocurriendo la conversación (puntos cardinales) y su visualización potencial.

**Análisis de Cumplimiento Obligatorio (El Servicio MCP #3):**
* **Análisis de Propagación de un Mensaje (OBLIGATORIO):** A partir de un ID de mensaje específico, el servicio debe analizar su propagación dentro de la red. Esto implica medir el alcance, la velocidad, y la presencia del contenido del mensaje original en las respuestas directas, los comentarios, o los mensajes compartidos, con el fin de determinar el impacto mediático de un comentario.

### 2.3. Estructura del Reto: Paso a Paso Detallado
* **Fase 1: Ingeniería de Datos y Configuración del Entorno:** Revisión del dataset, preparación de herramientas (Python), y definición del LLM base.
* **Fase 2: Desarrollo y Exposición de Servicios Analíticos (MCP):** Definición de las funciones analíticas y exposición como microservicios REST usando frameworks como FastAPI.
* **Fase 3: Construcción del Agente Conversacional (Tool-Calling):** Selección del framework (LangGraph/LlamaIndex para puntos extra), definición de herramientas (Tools), e implementación de la lógica conversacional.

### 2.4. Criterios de Evaluación
| Criterio | Ponderación | Descripción |
| :--- | :--- | :--- |
| **Funcionalidad y Robustez de los MCP** | 33% | Los 3 servicios funcionales, rápidos y en formato JSON. |
| **Calidad del Análisis** | 33% | Precisión, relevancia y sofisticación de los prompts/datos. |
| **Inteligencia del Agente** | 33% | Identificación de intención, tool-calling y conversación natural. |
| **Uso de Frameworks Avanzados** | 10% Extra | Uso efectivo de LangGraph, LlamaIndex o manejo de estado. |

---

## 3. Equipo de Trabajo y Asignación de Roles
Para cumplir con los requerimientos, el equipo se ha estructurado de la siguiente manera:

* **Rol 1: Data & API Engineer (Vanessa Sánchez Morales):** Responsable de descargar, limpiar y entender el `.parquet`. Creadora de la estructura base de los microservicios REST (FastAPI) asegurando los endpoints y el retorno de JSONs estructurados.
* **Rol 2: Especialista NLP / LLM Engineer:** Diseño de prompts avanzados y conexión con la API (OpenAI/Gemini) para los análisis de Sentimientos y Resumen General.
* **Rol 3: Algorithm Developer:** Encargado de la lógica de código puro para resolver el Análisis de Propagación (Obligatorio), calculando el alcance y la velocidad a partir de los timestamps y el `post_id`.
* **Rol 4: Agentic Framework Builder:** Construcción del "cerebro" final usando LangGraph (para asegurar el 10% extra). Configuración del Agente, creación de descripciones para el *Tool-Calling* y montaje de la interfaz (Streamlit/Terminal).

---

## 4. Arquitectura del Sistema
El proyecto sigue una arquitectura de microservicios desacoplados:
1. **Capa de Aplicación:** Interfaz para el usuario (Streamlit/CLI).
2. **Capa de Orquestación (Cerebro):** Agente basado en LangGraph que recibe lenguaje natural y decide qué herramienta invocar.
3. **Capa de Servicios (MCP - FastAPI):**
    * `/analisis/sentimientos`: (Seleccionado) Clasificación semántica del clima digital.
    * `/analisis/resumen` o `/analisis/metricas`: (Seleccionado) Segundo análisis del equipo.
    * `/analisis/propagacion`: (OBLIGATORIO) Medición de impacto mediático e ID de mensaje.
4. **Capa de Datos:** Dataset `Reto_data_20251023_122206.parquet` procesado con Pandas para acceso eficiente.

---

## 5. Estado Actual del Desarrollo (Data & API Engineering)
A la fecha, el Rol 1 ha completado los siguientes hitos:
- [x] Configuración del entorno virtual y resolución de dependencias.
- [x] Creación del módulo de carga de datos (`data_loader.py`) optimizado para `.parquet`.
- [x] Definición de contratos de datos (Schemas Pydantic) para las peticiones y respuestas.
- [x] Construcción del servidor base FastAPI (`main.py`) con los endpoints requeridos levantados localmente.

### Próximos Pasos Técnicos:
1. Reemplazar los datos "dummy" del endpoint de métricas/datos con la lógica real usando operaciones de Pandas.
2. Integrar el código de NLP (Rol 2) y Algoritmia (Rol 3) en las rutas correspondientes.
3. Dockerizar la API para unificar el entorno de despliegue entre los miembros del equipo.
"""

