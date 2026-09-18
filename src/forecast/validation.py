"""Harness di validazione: forward returns senza leakage e walk-forward.

Best practice incorporate:
- **No lookahead:** `forward_returns` allinea il punto t al return realizzato t→t+H (il valore
  noto solo in futuro), così una predizione fatta in t è confrontata solo con prezzi successivi.
- **Walk-forward:** `walk_forward_windows` produce split train→test rolling; la calibrazione si fa
  solo sul train, la valutazione solo sul test (mai tuning sul test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Return semplice realizzato da t a t+horizon, indicizzato a t (NaN sugli ultimi h punti).

    Allineamento no-leakage: il valore in t usa solo prezzi ≥ t.
    """
    close = close.astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    return fwd


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

