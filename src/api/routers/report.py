"""Desk Note: JSON delle card, pagina web e stato degli eventi.

Riusa i payload che gli altri router gia' producono e li passa al motore
narrativo, cosi' il report non ha una sua pipeline dati da tenere allineata.
"""
from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.cache import cache_get, cache_set
from src.api.helpers import http_error, ok

router = APIRouter(tags=["report"])

_log = logging.getLogger("api.report")


def _payload(response: JSONResponse | dict | None) -> dict:
    """Estrae la chiave ``data`` da una risposta di un altro router.

    I router restituiscono JSONResponse; in test possono restituire dict. Un
    endpoint che fallisce vale come dato mancante, non come errore fatale: il
    Desk Note esce con le card che riesce a comporre.
    """
    if response is None:
        return {}
    if isinstance(response, JSONResponse):
        import json

        try:
            body = json.loads(bytes(response.body))
        except (ValueError, TypeError):
            return {}
        return body.get("data") or {}
    if isinstance(response, dict):
        return response.get("data") or response
    return {}


def _gather() -> dict[str, dict]:
    """Raccoglie i payload dei quattro endpoint, saltando quelli che falliscono."""
    from src.api.routers import barriers as r_barriers
    from src.api.routers import flows as r_flows
    from src.api.routers import forecast as r_forecast
    from src.api.routers import gex as r_gex
    from src.api.routers import signals as r_signals

    fonti = {
        "gex": r_gex.get_gex,
        "barriers": r_barriers.get_barriers,
        "flows": r_flows.get_flows,
        "signals": r_signals.get_signals,
        "forecast": r_forecast.forecast_status,
    }

    out: dict[str, dict] = {}
    for nome, fn in fonti.items():
        try:
            out[nome] = _payload(fn())
        except Exception as exc:  # noqa: BLE001 — una fonte giù costa card, non l'edizione
            _log.warning("Fonte %s non disponibile per il Desk Note: %s", nome, exc)
            out[nome] = {}
    return out


def _build_note():
    """Compone l'edizione, con cache breve perché a monte c'è il fetch Deribit."""
    from src.report.narrative import build_desk_note

    fonti = _gather()
    return build_desk_note(
        gex=fonti["gex"],
        barriers=fonti["barriers"],
        flows=fonti["flows"],
        signals=fonti["signals"],
        forecast=fonti["forecast"],
    ), fonti


@router.get("/api/report/cards")
def get_report_cards() -> JSONResponse:
    """Le card del Desk Note in JSON — la sorgente di tutti i renderer."""
    cached = cache_get("report_cards")
    if cached is not None:
        return cached

    try:
        note, _ = _build_note()
        response = ok(note.to_dict())
        cache_set("report_cards", response)
        return response
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Desk Note error: {exc}")


@router.get("/api/report/events")
def get_report_events() -> JSONResponse:
    """Cosa e' cambiato dall'ultima edizione, e se basta a farne uscire una nuova.

    Sola lettura: non sposta la linea di base, cosi' interrogarlo non consuma gli
    eventi per chi pubblica. Per fissarla c'e' POST /api/report/events/commit.
    """
    from src.report.events import ReportStateDB, detect_events, should_publish, snapshot_state

    try:
        fonti = _gather()
        corrente = snapshot_state(
            gex=fonti["gex"], barriers=fonti["barriers"], signals=fonti["signals"]
        )
        db = ReportStateDB()
        eventi = detect_events(corrente, db.load())
        return ok({
            "should_publish": should_publish(eventi),
            "events": [
                {
                    "key": e.key,
                    "severity": e.severity,
                    "title": e.title,
                    "detail": e.detail,
                    "meta": e.meta,
                }
                for e in eventi
            ],
            "state": corrente,
        })
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Desk Note events error: {exc}")


@router.post("/api/report/events/commit")
def commit_report_state() -> JSONResponse:
    """Fissa la fotografia corrente come nuova linea di base dopo la pubblicazione."""
    from src.report.events import ReportStateDB, snapshot_state

    try:
        fonti = _gather()
        corrente = snapshot_state(
            gex=fonti["gex"], barriers=fonti["barriers"], signals=fonti["signals"]
        )
        ReportStateDB().save(corrente)
        return ok({"committed": True, "state": corrente})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Desk Note commit error: {exc}")


@router.get("/report", response_class=HTMLResponse)
def get_report_page(
    export: bool = Query(
        False,
        description="Card a grandezza naturale 1080x1350 senza riduzione, per l'export PNG.",
    ),
) -> HTMLResponse:
    """La pagina del Desk Note, servita accanto alla dashboard."""
    from src.report.renderer import render_html

    try:
        note, _ = _build_note()
        return HTMLResponse(content=render_html(note, export=export))
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Desk Note page error: {exc}")
