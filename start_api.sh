#!/bin/bash
export CORS_ORIGINS="https://btc-institutional-flow-tpw9m.ondigitalocean.app,http://localhost:3000,http://localhost:8501"
exec .venv/bin/python run_api.py --host 0.0.0.0 --port 8000
