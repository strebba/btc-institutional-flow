"""Entry point per avviare il backend FastAPI di btc-institutional-flow.

Uso:
  python run_api.py
  python run_api.py --host 0.0.0.0 --port 8000 --reload

Oppure direttamente con uvicorn:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Aggiunge la root del progetto al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Institutional Flow — API Server")
    parser.add_argument("--host",   default="0.0.0.0",   help="Indirizzo di ascolto (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Porta (default: 8000)")
    parser.add_argument("--reload", action="store_true",  help="Abilita hot-reload (sviluppo)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn non trovato. Installa con: pip install 'uvicorn[standard]'")
        sys.exit(1)

    print(f"Avvio server su http://{args.host}:{args.port}")
    print(f"Swagger UI: http://localhost:{args.port}/docs")
    print(f"Health:     http://localhost:{args.port}/api/health")

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
