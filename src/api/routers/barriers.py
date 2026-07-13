"""Barriers + Notes endpoint."""
from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.api.cache import cache_get, cache_set
from src.api.helpers import ok, http_error
from src.api.routers.gex import _get_gex_data

router = APIRouter(tags=["barriers"])


@router.get("/api/barriers")
def get_barriers() -> JSONResponse:
    import logging
    _log = logging.getLogger("api.barriers")

    cached = cache_get("barriers")
    if cached is not None:
        return cached

    try:
        from src.edgar.structured_notes_db import StructuredNotesDB, refresh_barrier_btc_prices
        from src.gex.deribit_client import DeribitClient

        db = StructuredNotesDB()

        try:
            refresh_barrier_btc_prices(db)
        except Exception as _e:
            _log.warning("IBIT/BTC ratio fetch fallito, uso valori esistenti nel DB: %s", _e)

        all_active = db.get_active_barriers()

        try:
            spot_price = DeribitClient().get_spot_price()
        except Exception:
            spot_price = None

        out = [
            {k: v for k, v in dict(b).items()}
            for b in all_active
            if b.get("level_price_btc") is not None
        ]
        pending_pricing = len(all_active) - len(out)
        last_filing_date = max(
            (b["issue_date"] for b in all_active if b.get("issue_date")), default=None
        )

        clusters_out: list[dict] = []
        confluence_out: list[dict] = []
        if out and spot_price and spot_price > 0:
            try:
                from src.edgar.barrier_utils import compute_confluence, detect_clusters

                gex_data = _get_gex_data()
                snap = gex_data["snapshot"]
                clusters = detect_clusters(out, spot_price)
                confluence_out = compute_confluence(
                    clusters,
                    put_wall=snap.put_wall,
                    call_wall=snap.call_wall,
                    gamma_flip=snap.gamma_flip_price,
                )
                clusters_out = [
                    {
                        "mean_price_btc": cl.mean_price_btc,
                        "total_notional_usd": cl.total_notional_usd,
                        "dominant_type": cl.dominant_type,
                        "sign": cl.sign,
                        "n_barriers": cl.n_barriers,
                        "distance_to_spot_pct": cl.distance_to_spot_pct,
                    }
                    for cl in clusters
                ]
            except Exception as _e:
                _log.warning("Confluenza barriere<->GEX non calcolata: %s", _e)

        response = ok({
            "count": len(out),
            "barriers": out,
            "spot_price": spot_price,
            "clusters": clusters_out,
            "confluence": confluence_out,
            "meta": {
                "total_active": len(all_active),
                "priced": len(out),
                "pending_pricing": pending_pricing,
                "last_filing_date": last_filing_date,
            },
        })
        cache_set("barriers", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Barriers error: {exc}")


# ─── Note helpers ──────────────────────────────────────────────────────────────


def _note_to_dict(note) -> dict:
    return {
        "filing_url": note.filing_url,
        "issuer": note.issuer,
        "issue_date": str(note.issue_date) if note.issue_date else None,
        "maturity_date": str(note.maturity_date) if note.maturity_date else None,
        "notional_usd": note.notional_usd,
        "product_type": note.product_type,
        "underlying": note.underlying,
        "initial_level": note.initial_level,
        "autocall_trigger_pct": note.autocall_trigger_pct,
        "knockin_barrier_pct": note.knockin_barrier_pct,
        "buffer_pct": note.buffer_pct,
        "coupon_rate": note.coupon_rate,
        "is_preliminary": bool(note.is_preliminary),
        "observation_dates": note.observation_dates,
        "barriers": [
            {
                "barrier_type": b.barrier_type,
                "level_pct": b.level_pct,
                "level_price_ibit": b.level_price_ibit,
                "level_price_btc": b.level_price_btc,
                "observation_date": str(b.observation_date) if b.observation_date else None,
                "status": b.status,
            }
            for b in note.barriers
        ],
    }


@router.get("/api/notes")
def get_notes(underlying: str = "IBIT", limit: int = 200) -> JSONResponse:
    try:
        from src.edgar.structured_notes_db import StructuredNotesDB

        db = StructuredNotesDB()
        notes = [n for n in db.get_all_notes()
                 if (n.underlying or "IBIT").upper() == underlying.upper()][:limit]
        return ok({"count": len(notes), "notes": [_note_to_dict(n) for n in notes]})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Notes error: {exc}")


@router.get("/api/notes/by-url")
def get_note_by_url(url: str) -> JSONResponse:
    try:
        from src.edgar.structured_notes_db import StructuredNotesDB

        db = StructuredNotesDB()
        note = db.get_note_by_url(url)
        if note is None:
            raise http_error("Nota non trovata", code=404)
        return ok({"note": _note_to_dict(note)})
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Note error: {exc}")
