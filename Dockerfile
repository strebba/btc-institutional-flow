# ── Stage 1: builder — installa dipendenze con tutti i compilatori ──────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libxml2-dev libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime — immagine finale senza build tools ────────────────────────
FROM python:3.11-slim AS runtime

# Dipendenze runtime: libxml2/libxslt per lxml, curl per healthcheck,
# nginx per il reverse proxy (X-Frame-Options strip + WebSocket),
# supervisor per la gestione dei 3 processi (nginx/uvicorn/streamlit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 curl nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

# Utente non-root per i processi applicativi (nginx resta root per la porta)
RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copia i pacchetti installati dallo stage builder
COPY --from=builder /install /usr/local

# Copia solo il codice sorgente (vedi .dockerignore per esclusioni)
COPY src/ src/
COPY config/ config/
COPY run_api.py .
COPY .streamlit/ .streamlit/

# Reverse proxy + process manager
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf supervisord.conf

# DB SQLite versionato (fonte di verità — il filesystem DO è effimero,
# il refresh EDGAR committa il DB su main → redeploy automatico)
COPY data/structured_notes.db data/structured_notes.db

# Directory runtime con permessi corretti
RUN mkdir -p data logs && chown -R appuser:appuser data logs

EXPOSE 8080

# supervisord (root) gestisce nginx (root) + uvicorn/streamlit (appuser)
CMD ["supervisord", "-c", "/app/supervisord.conf"]
