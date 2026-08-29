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


#: Il workflow gira lunedi' e mercoledi': oltre 10 giorni senza un refresh
#: riuscito significa che sta davvero saltando, non che il mercato e' fermo.
_MAX_REFRESH_AGE_DAYS = 10

#: Usato solo per i DB anteriori alla tabella refresh_runs, dove l'unico segnale
#: disponibile resta l'eta' dell'ultima nota scritta.
_MAX_NOTES_AGE_FALLBACK_DAYS = 30


def _age_days(iso: str | None) -> float | None:
    """Eta' in giorni di un timestamp ISO, None se assente o illeggibile."""
    from datetime import date

    if not iso:
        return None
    try:
        return (date.today() - date.fromisoformat(iso[:10])).days
    except (ValueError, TypeError):
        return None


@router.get("/api/health/edgar")
def health_edgar() -> JSONResponse:
    """Health check EDGAR: stato della pipeline e statistiche del DB.

    La salute si misura sull'ultimo refresh **riuscito**, non sull'ultima nota
    scritta. Il workflow gira due volte a settimana e in una finestra tranquilla
    puo' legittimamente non trovare nessun filing nuovo: misurare l'eta' delle
    note farebbe sembrare rotta una pipeline sana (ed e' quello che succedeva).
    """
    from src.edgar.structured_notes_db import StructuredNotesDB

    try:
        stats = StructuredNotesDB().get_edgar_stats()

        refresh_at = stats.get("last_refresh_at")
        refresh_ok = stats.get("last_refresh_ok")
        refresh_age = _age_days(refresh_at)
        notes_age = _age_days(stats.get("last_note_written_at"))

        if refresh_at is None:
            # DB anteriore alla tabella refresh_runs: si ripiega sull'unico
            # segnale disponibile, con una soglia piu' larga perche' misura
            # l'attivita' del mercato e non quella della pipeline.
            healthy = notes_age is not None and notes_age <= _MAX_NOTES_AGE_FALLBACK_DAYS
            reason = (
                f"nessun refresh registrato; ultima nota scritta {notes_age} giorni fa"
                if notes_age is not None
                else "nessun refresh registrato e nessuna nota nel DB"
            )
        elif not refresh_ok:
            healthy = False
            reason = f"l'ultimo refresh ({refresh_at[:10]}) e' fallito"
        elif refresh_age is not None and refresh_age > _MAX_REFRESH_AGE_DAYS:
            healthy = False
            reason = (
                f"nessun refresh riuscito da {refresh_age} giorni "
                f"(soglia {_MAX_REFRESH_AGE_DAYS})"
            )
        else:
            healthy = True
            reason = (
                f"refresh riuscito {refresh_age} giorni fa; "
                f"nessuna nota nuova da {notes_age} giorni"
                if notes_age is not None
                else f"refresh riuscito {refresh_age} giorni fa"
            )

        return ok({
            "service": "btc-institutional-flow",
            "healthy": healthy,
            "edgar": {
                "reason": reason,
                "last_refresh_at": refresh_at,
                "last_refresh_ok": refresh_ok,
                "last_refresh_filings_seen": stats.get("last_refresh_filings_seen"),
                "refresh_age_days": refresh_age,
                "last_note_written_at": stats.get("last_note_written_at"),
                "notes_age_days": notes_age,
                # alias storico: stale_days ha sempre misurato l'eta' delle note
                "last_update": stats.get("last_update"),
                "stale_days": notes_age,
                "total_notes": stats["total_notes"],
                "total_barriers": stats["total_barriers"],
                "active_barriers": stats["active_barriers"],
            },
        })
    except Exception as exc:
        raise http_error(f"EDGAR health check failed: {exc}", code=503)
