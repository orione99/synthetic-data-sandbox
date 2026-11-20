FROM python:3.11-slim

WORKDIR /app

# Dipendenze di sistema minime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installa PyTorch (versione leggera per container) e le altre dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copia il codice del progetto
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

EXPOSE 8000 8501