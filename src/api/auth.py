"""Autenticazione opzionale via API key con FastAPI Depends()."""
from __future__ import annotations

import os

from fastapi import HTTPException, Request

_API_KEY = os.getenv("API_KEY")


def require_api_key(request: Request) -> None:
    """Dependency FastAPI: verifica X-API-Key se API_KEY env var e' impostata.

    Se API_KEY non e' impostata, tutti gli accessi sono consentiti (sviluppo locale).
    L'endpoint /api/health e' escluso dal controllo (middleware-level).
    """
    if not _API_KEY:
        return
    if request.headers.get("X-API-Key") != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
