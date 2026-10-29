#!/bin/bash
# Avvio locale dell'API standalone (senza nginx/supervisord) per sviluppo.
export CORS_ORIGINS="https://btc-institutional-flow-tpw9m.ondigitalocean.app,https://www.wagmi-lab.com,http://localhost:3000,http://localhost:8501"
# Solo loopback di default: in produzione l'API e' quella su DigitalOcean; la
# copia locale (Mac mini) serve ai test e non va esposta in rete.
# Override esplicito: API_HOST=0.0.0.0 ./start_api.sh
exec python3 run_api.py --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
