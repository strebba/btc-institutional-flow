"""Test dell'harness di validazione: no-leakage, hit-rate, benchmark, walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast.models import DIR_DOWN, DIR_UP
from src.forecast.validation import (
    beats_benchmarks,
    benchmarks,
    direction_series,
    directional_hit_rate,
    forward_returns,
    walk_forward_windows,
)


def _idx(n):
    return pd.to_datetime([f"2026-01-{i+1:02d}" for i in range(n)])


def test_forward_returns_no_leakage():
    close = pd.Series([100.0, 110.0, 121.0], index=_idx(3))
    fwd = forward_returns(close, horizon=1)
    assert fwd.iloc[0] == pytest.approx(0.10)   # 100→110
    assert fwd.iloc[1] == pytest.approx(0.10)   # 110→121
    assert np.isnan(fwd.iloc[2])                 # nessun futuro → NaN (no leakage)


def test_directional_hit_rate():
    close = pd.Series([100, 105, 110, 100, 95], index=_idx(5), dtype=float)
    fwd = forward_returns(close, horizon=1)
    pred = pd.Series([DIR_UP, DIR_UP, DIR_DOWN, DIR_DOWN, DIR_UP], index=_idx(5))
    res = directional_hit_rate(pred, fwd, flat_band_pct=1.0)
    # fwd: +5%(up), +4.76%(up), -9%(down), -5%(down), NaN → 4 valutabili, tutte corrette
    assert res["n"] == 4
    assert res["hit_rate"] == 1.0


def test_benchmarks_and_beats():
    close = pd.Series([100, 102, 104, 106, 108], index=_idx(5), dtype=float)
    fwd = forward_returns(close, horizon=1)
    bench = benchmarks(fwd)
    assert bench["n"] == 4
    assert bench["always_up"] == 1.0  # sempre salito
    assert beats_benchmarks(0.5, bench) is False   # non batte always_up=1.0
    assert beats_benchmarks(None, bench) is False


def test_walk_forward_windows_no_test_overlap():
    idx = _idx(10)
    wins = list(walk_forward_windows(idx, train_size=4, test_size=2))
    assert len(wins) == 3
    # i test non si sovrappongono
    seen = set()
    for w in wins:
        assert len(w.train_idx) == 4 and len(w.test_idx) == 2
        for t in w.test_idx:
            assert t not in seen
            seen.add(t)
