"""Harness di validazione: forward returns senza leakage, hit-rate, benchmark naive, walk-forward.

Best practice incorporate:
- **No lookahead:** `forward_returns` allinea il punto t al return realizzato t→t+H (il valore
  noto solo in futuro), così una predizione fatta in t è confrontata solo con prezzi successivi.
- **Benchmark obbligatori:** ogni strategia va confrontata con always-up / always-down / random /
  base-rate; se non li batte, non ha edge.
- **Walk-forward:** `walk_forward_windows` produce split train→test rolling; la calibrazione si fa
  solo sul train, la valutazione solo sul test (mai tuning sul test).

Riusa `src/analytics/backtest.py` per le metriche di equity (Sharpe/maxDD) quando serve un backtest
di strategia completo sui flussi/EMA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.config import setup_logging
from src.forecast.models import DIR_DOWN, DIR_FLAT, DIR_UP

_log = setup_logging("forecast.validation")


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Return semplice realizzato da t a t+horizon, indicizzato a t (NaN sugli ultimi h punti).

    Allineamento no-leakage: il valore in t usa solo prezzi ≥ t.
    """
    close = close.astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    return fwd


def direction_series(fwd: pd.Series, flat_band_pct: float = 1.0) -> pd.Series:
    """Converte i forward return in etichette up/down/flat."""
    band = flat_band_pct / 100.0
    out = pd.Series(DIR_FLAT, index=fwd.index, dtype=object)
    out[fwd > band] = DIR_UP
    out[fwd < -band] = DIR_DOWN
    out[fwd.isna()] = np.nan  # coda non maturata: resta NaN (no-leakage, non è "flat")
    return out


def directional_hit_rate(predicted: pd.Series, fwd: pd.Series, flat_band_pct: float = 1.0) -> dict:
    """Hit-rate di una serie di direzioni predette vs realizzate (allineate per indice)."""
    realized = direction_series(fwd, flat_band_pct)
    df = pd.DataFrame({"pred": predicted, "real": realized, "fwd": fwd}).dropna()
    if df.empty:
        return {"n": 0, "hit_rate": None}
    hits = int((df["pred"] == df["real"]).sum())
    return {
        "n": int(len(df)),
        "hits": hits,
        "hit_rate": round(hits / len(df), 3),
        "mean_fwd_when_up": round(float(df.loc[df["pred"] == DIR_UP, "fwd"].mean()), 4)
        if (df["pred"] == DIR_UP).any() else None,
    }


def benchmarks(fwd: pd.Series, flat_band_pct: float = 1.0) -> dict:
    """Hit-rate dei benchmark naive: always-up, always-down, random, base-rate(up)."""
    realized = direction_series(fwd, flat_band_pct).dropna()
    n = len(realized)
    if n == 0:
        return {"n": 0}
    up_rate = float((realized == DIR_UP).mean())
    return {
        "n": n,
        "always_up": round(up_rate, 3),
        "always_down": round(float((realized == DIR_DOWN).mean()), 3),
        "random": round(1.0 / 3.0, 3),  # 3 classi
        "base_rate_up": round(up_rate, 3),
    }


@dataclass
class WalkForwardWindow:
    train_idx: pd.Index
    test_idx: pd.Index


def walk_forward_windows(
    index: pd.Index, *, train_size: int, test_size: int, step: Optional[int] = None,
) -> Iterator[WalkForwardWindow]:
    """Genera split rolling train→test senza sovrapposizione del test (no tuning sul test)."""
    step = step or test_size
    n = len(index)
    start = 0
    while start + train_size + test_size <= n:
        tr = index[start: start + train_size]
        te = index[start + train_size: start + train_size + test_size]
        yield WalkForwardWindow(tr, te)
        start += step


def beats_benchmarks(hit_rate: Optional[float], bench: dict) -> bool:
    """True se l'hit-rate batte tutti i benchmark naive rilevanti."""
    if hit_rate is None or not bench or bench.get("n", 0) == 0:
        return False
    refs = [bench.get(k) for k in ("always_up", "always_down", "random") if bench.get(k) is not None]
    return all(hit_rate > r for r in refs)
