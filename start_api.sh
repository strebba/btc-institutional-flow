#!/bin/bash
# Avvio locale dell'API standalone (senza nginx/supervisord) per sviluppo.
export CORS_ORIGINS="https://btc-institutional-flow-tpw9m.ondigitalocean.app,https://www.wagmi-lab.com,http://localhost:3000,http://localhost:8501"
exec .venv/bin/python run_api.py --host 0.0.0.0 --port 8000
