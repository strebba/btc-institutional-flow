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

# Dipendenze runtime (solo libxml2/libxslt per lxml, curl per curl_cffi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 curl \
    && rm -rf /var/lib/apt/lists/*

# Utente non-root per ridurre la superficie d'attacco
RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copia i pacchetti installati dallo stage builder
COPY --from=builder /install /usr/local

# Copia solo il codice sorgente (vedi .dockerignore per esclusioni)
COPY src/ src/
COPY config/ config/
COPY run_api.py .

# Directory runtime con permessi corretti
RUN mkdir -p data logs && chown appuser:appuser data logs

USER appuser

EXPOSE 8000

CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"]
