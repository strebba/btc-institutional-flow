"""Verifica d'esito: assegna un Outcome a ogni predizione matura.

`score_prediction` è puro (accetta un DataFrame di prezzi) → testabile offline.
`score_due_predictions` orchestra: legge le predizioni due dal DB, recupera i prezzi
reali e persiste gli esiti.

Convenzioni prezzi: DataFrame con DatetimeIndex e colonne `close`, `high`, `low`
(come restituito da src/flows/price_fetcher.PriceFetcher._load_from_db).
La finestra di valutazione va da created_at (escluso) a created_at + horizon_days (incluso).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from src.config import setup_logging
from src.forecast.models import (
    DIR_DOWN,
    DIR_FLAT,
    DIR_UP,
    Outcome,
    Prediction,
    TARGET_DIRECTION,
    TARGET_LEVEL,
    TARGET_PROB,
)

_log = setup_logging("forecast.verifier")

# Provider prezzi: (asset, start, end) -> DataFrame[close, high, low] con DatetimeIndex
PriceProvider = Callable[[str, datetime, datetime], pd.DataFrame]


def _window(prices: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(prices.index)
    df = prices.copy()
    df.index = idx
    mask = (idx > pd.Timestamp(start.date())) & (idx <= pd.Timestamp(end.date()))
    return df.loc[mask]


def score_prediction(pred: Prediction, prices: pd.DataFrame) -> Optional[Outcome]:
    """Valuta una singola predizione contro i prezzi della finestra. None se dati insufficienti."""
    created = datetime.fromisoformat(pred.created_at).replace(tzinfo=timezone.utc)
    win = _window(prices, created, created + timedelta(days=pred.horizon_days))
    if win.empty or "close" not in win.columns:
        return None

    ref = float(pred.target_spec.get("ref_price") or 0.0)
    realized_price = float(win["close"].iloc[-1])
    if ref <= 0:
        ref = float(win["close"].iloc[0])
    realized_return = realized_price / ref - 1.0 if ref else 0.0

    high = float(win["high"].max()) if "high" in win.columns else realized_price
    low = float(win["low"].min()) if "low" in win.columns else realized_price

    if pred.target_type == TARGET_DIRECTION:
        return _score_direction(pred, ref, realized_price, realized_return)
    if pred.target_type == TARGET_LEVEL:
        return _score_level(pred, ref, realized_price, realized_return, high, low)
    if pred.target_type == TARGET_PROB:
        return _score_prob(pred, ref, realized_price, realized_return)
    return None


def _score_direction(pred, ref, realized_price, ret) -> Outcome:
    band = float(pred.target_spec.get("flat_band_pct", 1.0)) / 100.0
    realized_dir = DIR_UP if ret > band else (DIR_DOWN if ret < -band else DIR_FLAT)
    predicted = pred.target_spec.get("direction")
    hit = realized_dir == predicted
    return Outcome(
        prediction_id=pred.id or 0,
        hit=hit,
        realized_return=ret,
        realized_price=realized_price,
        ref_price=ref,
        signed_error=ret,
        detail=f"predetto={predicted}, realizzato={realized_dir} (ret {ret:+.2%})",
    )


def _score_level(pred, ref, realized_price, ret, high, low) -> Outcome:
    level = float(pred.target_spec["level_price"])
    mode = pred.target_spec.get("mode", "reach")
    side = pred.target_spec.get("side", "above")
    touched = low <= level <= high

    if mode in ("reach", "break"):
        hit = touched
        verb = "raggiunto" if mode == "reach" else "rotto"
        detail = f"{pred.target_spec.get('level_name')} {level:,.0f} {verb}={hit} (range {low:,.0f}-{high:,.0f})"
    else:  # respect: il livello non deve essere violato dal lato sbagliato
        if side == "below":      # supporto (es. put wall): non scendere sotto
            hit = low >= level
        else:                     # resistenza (es. call wall): non salire sopra
            hit = high <= level
        detail = f"{pred.target_spec.get('level_name')} {level:,.0f} rispettato({side})={hit} (range {low:,.0f}-{high:,.0f})"

    signed_error = (realized_price - level) / level if level else None
    return Outcome(
        prediction_id=pred.id or 0,
        hit=hit,
        realized_return=ret,
        realized_price=realized_price,
        ref_price=ref,
        signed_error=signed_error,
        detail=detail,
    )


def _score_prob(pred, ref, realized_price, ret) -> Outcome:
    p = float(pred.target_spec.get("p", 0.5))
    event = ret > 0.0  # btc_return_positive
    brier = (p - (1.0 if event else 0.0)) ** 2
    hit = (p > 0.5) == event
    return Outcome(
        prediction_id=pred.id or 0,
        hit=hit,
        realized_return=ret,
        realized_price=realized_price,
        ref_price=ref,
        signed_error=p - (1.0 if event else 0.0),
        brier=brier,
        detail=f"p={p:.0%}, evento(ret>0)={event}, brier={brier:.3f}",
    )


def score_due_predictions(
    db, price_provider: PriceProvider, asof: Optional[datetime] = None,
) -> list[Outcome]:
    """Verifica tutte le predizioni mature nel DB e persiste gli esiti.

    Args:
        db: PredictionDB.
        price_provider: callable (asset, start, end) → DataFrame prezzi.
        asof: istante di riferimento (default now UTC).

    Returns:
        Lista degli Outcome creati.
    """
    due = db.get_due(asof)
    created: list[Outcome] = []
    for pred in due:
        start = datetime.fromisoformat(pred.created_at).replace(tzinfo=timezone.utc)
        end = pred.matures_at
        try:
            prices = price_provider(pred.asset, start, end)
        except Exception as exc:
            _log.warning("Prezzi non disponibili per pred %s: %s", pred.id, exc)
            continue
        outcome = score_prediction(pred, prices)
        if outcome is None:
            _log.info("Dati insufficienti per pred %s, resta open", pred.id)
            continue
        if db.insert_outcome(outcome):
            created.append(outcome)
            _log.info("Pred %s verificata: hit=%s — %s", pred.id, outcome.hit, outcome.detail)
    return created
