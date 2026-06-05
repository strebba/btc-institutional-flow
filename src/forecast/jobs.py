"""Job riusabili del forecast loop: predict / verify / calibrate.

Single source of truth chiamata sia dagli script cron (`scripts/cron_*.py`) sia dallo
scheduler APScheduler dentro l'API. Funzioni sincrone (la rete è dentro); lo scheduler
async le esegue in un thread separato.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from src.config import setup_logging
from src.forecast.calibration import CalibrationReport, load_weights_config, run_calibration
from src.forecast.prediction_db import PredictionDB
from src.forecast.sources.dealer_flow import SOURCE as DEALER_FLOW
from src.forecast.sources.dealer_flow import build_dealer_flow_predictions

_log = setup_logging("forecast.jobs")

_TICKER = {"BTC": "BTC-USD"}


def _governance() -> dict:
    return load_weights_config().get("governance", {})


def run_daily_predict(*, horizon: int = 5, db: Optional[PredictionDB] = None) -> dict:
    """Genera le predizioni dealer-flow del giorno (status open). Rispetta il kill-switch."""
    from src.forecast.context import DataUnavailable, gather_dealer_flow_context

    if _governance().get("kill_switch", False):
        _log.warning("[predict] kill-switch attivo: generazione predizioni in pausa")
        return {"status": "skipped_kill_switch", "inserted": 0}

    db = db or PredictionDB()
    active = db.get_active_weights(DEALER_FLOW)
    weights = active[1] if active else None
    weights_version = active[0] if active else None

    try:
        ctx = gather_dealer_flow_context(weights=weights)
    except DataUnavailable as exc:
        _log.error("[predict] dati non disponibili: %s", exc)
        return {"status": "data_unavailable", "error": str(exc), "inserted": 0}

    snap = ctx.snapshot
    preds = build_dealer_flow_predictions(
        ctx.result, spot_price=ctx.spot,
        gamma_flip=snap.gamma_flip_price, max_pain=snap.max_pain,
        put_wall=snap.put_wall, call_wall=snap.call_wall,
        total_net_gex=snap.total_net_gex, horizon_days=horizon,
        weights_version=weights_version,
    )
    inserted = sum(1 for p in preds if db.insert_prediction(p) is not None)
    _log.info("[predict] %s score=%.1f → %d/%d salvate",
              ctx.result.signal, ctx.result.score, inserted, len(preds))
    return {"status": "ok", "signal": ctx.result.signal, "score": ctx.result.score,
            "inserted": inserted, "total": len(preds), "weights_version": weights_version}


def run_daily_verify(*, db: Optional[PredictionDB] = None) -> dict:
    """Verifica le predizioni mature usando i prezzi reali."""
    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.verifier import score_due_predictions

    db = db or PredictionDB()
    fetcher = PriceFetcher()

    def provider(asset, start, end):
        return fetcher.fetch(
            _TICKER.get(asset, asset),
            start_date=start.date(),
            end_date=(end + timedelta(days=1)).date(),
        )

    due = db.get_due()
    outcomes = score_due_predictions(db, provider, datetime.utcnow())
    hits = sum(1 for o in outcomes if o.hit)
    _log.info("[verify] mature=%d verificate=%d hit=%d", len(due), len(outcomes), hits)
    return {"status": "ok", "due": len(due), "verified": len(outcomes),
            "hit": hits, "miss": len(outcomes) - hits}


def run_weekly_calibrate(*, source: str = DEALER_FLOW, days: int = 180,
                         db: Optional[PredictionDB] = None) -> CalibrationReport:
    """Calcola metriche e propone (mai attiva) nuovi pesi."""
    db = db or PredictionDB()
    report = run_calibration(db, source=source, days=days)
    _log.info("[calibrate] %s gate=%s scored=%d", source, report.gate_ok,
              report.metrics["total_scored"])
    return report
