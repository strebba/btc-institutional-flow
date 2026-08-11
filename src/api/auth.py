"""Autenticazione opzionale via API key con FastAPI Depends().

La protezione reale e' implementata a livello middleware in main.py.
Questa dependency e' disponibile per eventuali controlli per-route.
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Request


def require_api_key(request: Request) -> None:
    """Dependency FastAPI: verifica X-API-Key se API_KEY env var e' impostata.

    Se API_KEY non e' impostata, tutti gli accessi sono consentiti (sviluppo locale).
    L'endpoint /api/health e' escluso dal controllo (middleware-level).
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        return
    if request.headers.get("X-API-Key") != api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
