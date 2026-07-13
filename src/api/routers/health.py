"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.helpers import ok

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
        with db._conn() as conn:
            last = conn.execute("SELECT MAX(created_at) FROM notes").fetchone()[0]
            total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            total_barriers = conn.execute("SELECT COUNT(*) FROM barrier_levels").fetchone()[0]
            active_barriers = conn.execute(
                "SELECT COUNT(*) FROM barrier_levels WHERE status='active'"
            ).fetchone()[0]

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
        return ok({
            "service": "btc-institutional-flow",
            "healthy": False,
            "edgar": {"error": str(exc)},
        })
