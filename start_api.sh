#!/bin/bash
export CORS_ORIGINS="https://seashell-app-h7hc4.ondigitalocean.app,http://localhost:3000,http://localhost:8501"
exec .venv/bin/python run_api.py --host 0.0.0.0 --port 8000
