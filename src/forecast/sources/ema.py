"""Adapter EMA: crossover EMA 50/200 → predizione direction sullo stesso spine.

Backtestabile subito (storia prezzi ampia). Funzione pura su una serie di chiusure.
Orizzonte default ~15g (hold medio dal backtest del Bingx bot). La confidence deriva dallo
spread normalizzato tra le due EMA: più sono distanti, più il trend è consolidato.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.forecast.models import DIR_DOWN, DIR_UP, Prediction, TARGET_DIRECTION

SOURCE = "ema"
DEFAULT_FAST = 50
DEFAULT_SLOW = 200
DEFAULT_HORIZON_DAYS = 15
_SPREAD_SCALE = 20.0  # spread 5% → confidence ~1.0


def compute_ema_state(close: pd.Series, fast: int = DEFAULT_FAST, slow: int = DEFAULT_SLOW) -> dict:
    """Stato corrente del crossover: regime (bull/bear), spread normalizzato, prezzi EMA."""
    close = close.astype(float)
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    f, s = float(ema_f.iloc[-1]), float(ema_s.iloc[-1])
    spread = (f - s) / s if s else 0.0
    return {
        "regime": "bull" if f > s else "bear",
        "spread": spread,
        "ema_fast": f,
        "ema_slow": s,
        "spot": float(close.iloc[-1]),
    }


def build_ema_predictions(
    close: pd.Series,
    *,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    asset: str = "BTC",
    created_at: Optional[str] = None,
) -> list[Prediction]:
    """Costruisce una predizione direction dal regime EMA. Richiede ≥ `slow` chiusure."""
    if len(close) < slow:
        raise ValueError(f"servono almeno {slow} chiusure, ricevute {len(close)}")

    st = compute_ema_state(close, fast, slow)
    direction = DIR_UP if st["regime"] == "bull" else DIR_DOWN
    confidence = max(0.0, min(1.0, abs(st["spread"]) * _SPREAD_SCALE))

    common: dict = dict(source=SOURCE, asset=asset, horizon_days=horizon_days)
    if created_at is not None:
        common["created_at"] = created_at

    return [Prediction(
        target_type=TARGET_DIRECTION,
        target_spec={"direction": direction, "ref_price": st["spot"], "flat_band_pct": 2.0},
        confidence=confidence,
        rationale=(
            f"EMA{fast}/{slow} {st['regime']} (spread {st['spread']:+.1%}): direzione {direction} "
            f"attesa su {horizon_days}g. EMA{fast}={st['ema_fast']:,.0f} EMA{slow}={st['ema_slow']:,.0f}."
        ),
        score_ref=50.0 + st["spread"] * _SPREAD_SCALE * 50.0,
        components={"ema_spread": st["spread"]},
        **common,
    )]
