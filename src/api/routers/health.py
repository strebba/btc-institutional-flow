"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.helpers import http_error, ok

router = APIRouter(tags=["meta"])


@router.get("/api/health")
def health() -> JSONResponse:
    """Health check: verifica che il server sia attivo."""
    return ok({"service": "btc-institutional-flow", "healthy": True})


@router.get("/api/health/edgar")
def health_edgar() -> JSONResponse:
    """Health check EDGAR: freschezza dati e statistiche DB note strutturate."""
    from datetime import date

    from src.edgar.structured_notes_db import StructuredNotesDB

    try:
        db = StructuredNotesDB()
        stats = db.get_edgar_stats()
        last = stats["last_update"]
        total_notes = stats["total_notes"]
        total_barriers = stats["total_barriers"]
        active_barriers = stats["active_barriers"]

        stale_days = None
        if last:
            last_date = date.fromisoformat(last[:10])
            stale_days = (date.today() - last_date).days
            healthy = stale_days <= 14
        else:
            healthy = False

        return ok({
            "service": "btc-institutional-flow",
            "healthy": healthy,
            "edgar": {
                "last_update": last,
                "total_notes": total_notes,
                "total_barriers": total_barriers,
                "active_barriers": active_barriers,
                "stale_days": stale_days,
            },
        })
    except Exception as exc:
        raise http_error(f"EDGAR health check failed: {exc}", code=503)
