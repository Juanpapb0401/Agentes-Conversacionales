# ============================================================
# Dockerfile — Streamlit Frontend (UI + Agente LangGraph)
# Build: docker build -f Dockerfile.app -t agentes-app .
# ============================================================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar UI y agente
COPY app.py ./
COPY pages/ pages/
COPY agent/ agent/
COPY services/ services/

EXPOSE 8501

# Desactivar verificación de uso y telemetría de Streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
