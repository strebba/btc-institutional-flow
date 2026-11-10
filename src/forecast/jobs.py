"""Job riusabili del forecast loop: predict / verify / calibrate.

Single source of truth chiamata sia dagli script cron (`scripts/cron_*.py`) sia dallo
scheduler APScheduler dentro l'API. Funzioni sincrone (la rete è dentro); lo scheduler
async le esegue in un thread separato.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.config import setup_logging
from src.forecast.calibration import CalibrationReport, load_weights_config, run_calibration
from src.forecast.prediction_db import PredictionDB
from src.forecast.sources.dealer_flow import SOURCE as DEALER_FLOW
from src.forecast.sources.dealer_flow import build_dealer_flow_predictions

_log = setup_logging("forecast.jobs")


def _cohort_timestamp(asof: datetime) -> str:
    """ISO UTC senza microsecondi, stesso formato di ``Prediction.created_at``."""
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    else:
        asof = asof.astimezone(timezone.utc)
    return asof.strftime("%Y-%m-%dT%H:%M:%S")


def _governance() -> dict:
    return load_weights_config().get("governance", {})


def run_daily_predict(*, horizon: int = 5, db: Optional[PredictionDB] = None,
                      asof: Optional[datetime] = None) -> dict:
    """Genera le predizioni dealer-flow del giorno (status open). Rispetta il kill-switch.

    ``asof`` fissa il timestamp UTC della cohort (la data è ``created_at[:10]``).
    Senza, ogni predizione usa l'istante di costruzione: stesso giorno, stesso effetto
    sull'idempotenza per-data. Il catch-up all'avvio lo passa per allineare la cohort
    alla data UTC già controllata.
    """
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

    # Uno stesso timestamp per v1 e per l'ombra: lo slot del giorno è unico,
    # anche quando il cron non passa ``asof`` (l'orologio di costruzione
    # altrimenti scatta di un secondo tra una riga e l'altra).
    created_at = _cohort_timestamp(asof or datetime.now(timezone.utc))
    snap = ctx.snapshot
    v1 = build_dealer_flow_predictions(
        ctx.result, spot_price=ctx.spot,
        gamma_flip=snap.gamma_flip_price, max_pain=snap.max_pain,
        total_net_gex=snap.total_net_gex, horizon_days=horizon,
        weights_version=weights_version,
        created_at=created_at,
    )
    v2 = _shadow_predictions(
        db, score=ctx.result.score, spot=ctx.spot, components=ctx.result.components,
        gamma_flip=snap.gamma_flip_price, max_pain=snap.max_pain,
        total_net_gex=snap.total_net_gex, horizon=horizon,
        weights_version=weights_version, created_at=created_at,
    )
    preds = list(v1) + list(v2)
    inserted = sum(1 for p in preds if db.insert_prediction(p) is not None)
    _log.info("[predict] %s score=%.1f → %d/%d salvate (v1=%d v2=%d)",
              ctx.result.signal, ctx.result.score, inserted, len(preds), len(v1), len(v2))
    return {"status": "ok", "signal": ctx.result.signal, "score": ctx.result.score,
            "inserted": inserted, "total": len(preds),
            "v1": len(v1), "v2": len(v2), "weights_version": weights_version}


def _shadow_predictions(db, *, score, spot, components, gamma_flip, max_pain,
                        total_net_gex, horizon, weights_version, created_at):
    """Cohort dealer_flow_v2 sugli stessi input. Un errore di prezzi/storia non ferma v1."""
    from src.forecast.sources.dealer_flow_v2 import (
        FLAT_BAND_FALLBACK_PCT,
        build_dealer_flow_v2_predictions,
        calibrate_prob,
        flat_band_pct_from_returns,
        matured_prob_history,
        trailing_log_returns,
    )

    when = datetime.fromisoformat(created_at).replace(tzinfo=timezone.utc)
    band = FLAT_BAND_FALLBACK_PCT
    try:
        from src.flows.price_fetcher import PriceFetcher
        rets = trailing_log_returns(PriceFetcher(db_path=getattr(db, "_path", None)), when.date())
        band = flat_band_pct_from_returns(rets, horizon)
    except Exception as exc:
        _log.warning("[predict] banda v2 di fallback %.1f%%: %s", FLAT_BAND_FALLBACK_PCT, exc)
    method, n = "shrink_base_rate", 0
    try:
        history = matured_prob_history(db, when)
        p_up, method = calibrate_prob(float(score), history)
        n = len(history)
    except Exception as exc:
        from src.forecast.sources.dealer_flow import _prob_up_from_score
        _log.warning("[predict] prob v2 non calibrata, uso la mappa grezza: %s", exc)
        p_up = _prob_up_from_score(float(score))
    return build_dealer_flow_v2_predictions(
        score=float(score), spot_price=float(spot), p_up=p_up, flat_band_pct=band,
        components=dict(components or {}),
        gamma_flip=gamma_flip, max_pain=max_pain, total_net_gex=total_net_gex,
        horizon_days=horizon, weights_version=weights_version, created_at=created_at,
        calibration=method, calibration_n=n,
    )


def run_shadow_catchup(db, *, day: str, horizon: int) -> dict:
    """Scrive la cohort v2 di ``day`` copiando gli input già salvati su dealer_flow.

    Non chiama Deribit e non riscrive v1. Se v2 c'è già, non fa nulla
    (l'indice per source resta quello di v1, intatto).
    """
    from src.forecast.sources.dealer_flow_v2 import (
        FLAT_BAND_FALLBACK_PCT,
        SOURCE as V2,
        calibrate_prob,
        flat_band_pct_from_returns,
        matured_prob_history,
        trailing_log_returns,
        v2_from_stored_v1,
    )

    if db.has_cohort(day, source=V2, horizon_days=horizon):
        return {"status": "skipped_exists", "inserted": 0, "total": 0}
    stored = [p for p in db.predictions_on_date(day, DEALER_FLOW) if int(p.horizon_days) == int(horizon)]
    if not stored:
        return {"status": "no_v1", "inserted": 0, "total": 0}
    when = datetime.fromisoformat(stored[0].created_at).replace(tzinfo=timezone.utc)
    band = FLAT_BAND_FALLBACK_PCT
    try:
        from src.flows.price_fetcher import PriceFetcher
        rets = trailing_log_returns(PriceFetcher(db_path=getattr(db, "_path", None)), when.date())
        band = flat_band_pct_from_returns(rets, horizon)
    except Exception as exc:
        _log.warning("[predict] catch-up v2 banda di fallback: %s", exc)
    score = next((p.score_ref for p in stored if p.score_ref is not None), 50.0)
    try:
        history = matured_prob_history(db, when)
        p_up, method = calibrate_prob(float(score), history)
        n = len(history)
    except Exception as exc:
        from src.forecast.sources.dealer_flow import _prob_up_from_score
        _log.warning("[predict] catch-up v2 prob non calibrata: %s", exc)
        p_up, method, n = _prob_up_from_score(float(score)), "shrink_base_rate", 0
    preds = v2_from_stored_v1(
        stored, p_up=p_up, flat_band_pct=band, calibration=method, calibration_n=n,
    )
    inserted = sum(1 for p in preds if db.insert_prediction(p) is not None)
    _log.info("[predict] shadow v2 %s: %d/%d salvate dagli input v1", day, inserted, len(preds))
    return {"status": "ok", "inserted": inserted, "total": len(preds)}


def run_daily_verify(*, db: Optional[PredictionDB] = None) -> dict:
    """Verifica le predizioni mature sul close e sul range delle barre UTC chiuse.

    Il giorno di fine orizzonte entra nel punteggio solo quando la sua barra
    giornaliera è completa (``partial=0``). Se la sessione è ancora aperta la
    predizione resta open fino al verify successivo.
    """
    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.verifier import score_due_predictions, verification_price_provider

    db = db or PredictionDB()
    fetcher = PriceFetcher()
    due = db.get_due()
    outcomes = score_due_predictions(
        db, verification_price_provider(fetcher), datetime.now(timezone.utc),
    )
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
