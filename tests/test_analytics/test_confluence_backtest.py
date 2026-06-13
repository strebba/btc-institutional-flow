"""Test per src/analytics/confluence_backtest.py (scaffolding gated, Step 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.confluence_backtest import (
    MIN_CONFLUENCE_HISTORY_DAYS,
    confluence_backtest_ready,
    confluence_boost_series,
    run_confluence_backtest,
)


def _walls_df(n: int, put_wall=80_000.0, call_wall=90_000.0, flip=82_000.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"put_wall": put_wall, "call_wall": call_wall, "gamma_flip_price": flip},
        index=idx,
    )


# ─── gate ─────────────────────────────────────────────────────────────────────


class TestReadyGate:
    def test_none_not_ready(self):
        assert confluence_backtest_ready(None) is False

    def test_empty_not_ready(self):
        assert confluence_backtest_ready(pd.DataFrame()) is False

    def test_below_threshold_not_ready(self):
        assert confluence_backtest_ready(_walls_df(MIN_CONFLUENCE_HISTORY_DAYS - 1)) is False

    def test_at_threshold_ready(self):
        assert confluence_backtest_ready(_walls_df(MIN_CONFLUENCE_HISTORY_DAYS)) is True

    def test_all_nan_rows_not_counted(self):
        df = _walls_df(MIN_CONFLUENCE_HISTORY_DAYS)
        df[:] = np.nan
        assert confluence_backtest_ready(df) is False


# ─── boost series ─────────────────────────────────────────────────────────────


class TestBoostSeries:
    def test_no_barriers_returns_zeros(self):
        btc = pd.Series([85_000.0] * 5, index=pd.date_range("2026-01-01", periods=5))
        boost = confluence_boost_series(btc, None, _walls_df(5))
        assert (boost == 0.0).all()

    def test_bearish_confluence_is_negative(self):
        # knock_in a 80k (sotto spot) coincide col put_wall 80k → boost ribassista (<0)
        btc = pd.Series([85_000.0] * 3, index=pd.date_range("2026-01-01", periods=3))
        barriers = [
            {"level_price_btc": 80_000.0, "barrier_type": "knock_in", "notional_usd": 200e6},
        ]
        boost = confluence_boost_series(btc, barriers, _walls_df(3, put_wall=80_000.0))
        assert (boost < 0).all()

    def test_bullish_confluence_is_positive(self):
        # autocall a 90k (sopra spot) coincide col call_wall 90k → boost rialzista (>0)
        btc = pd.Series([85_000.0] * 3, index=pd.date_range("2026-01-01", periods=3))
        barriers = [
            {"level_price_btc": 90_000.0, "barrier_type": "autocall", "notional_usd": 200e6},
        ]
        boost = confluence_boost_series(btc, barriers, _walls_df(3, call_wall=90_000.0))
        assert (boost > 0).all()

    def test_dates_without_walls_are_zero(self):
        btc = pd.Series([85_000.0] * 5, index=pd.date_range("2026-01-01", periods=5))
        barriers = [{"level_price_btc": 80_000.0, "barrier_type": "knock_in", "notional_usd": 200e6}]
        walls = _walls_df(2, put_wall=80_000.0)  # solo i primi 2 giorni hanno wall
        boost = confluence_boost_series(btc, barriers, walls)
        assert (boost.iloc[2:] == 0.0).all()


# ─── run (gated) ──────────────────────────────────────────────────────────────


class TestRunGated:
    def test_insufficient_history_returns_not_ready(self):
        btc = pd.Series([85_000.0] * 3, index=pd.date_range("2026-01-01", periods=3))
        res = run_confluence_backtest(btc, None, _walls_df(3))
        assert res["ready"] is False
        assert res["n_history"] == 3
        assert res["required"] == MIN_CONFLUENCE_HISTORY_DAYS
        assert "insufficiente" in res["reason"].lower()

    def test_sufficient_history_runs_probe(self):
        n = MIN_CONFLUENCE_HISTORY_DAYS + 5
        idx = pd.date_range("2026-01-01", periods=n, freq="D")
        btc = pd.Series(np.linspace(80_000, 90_000, n), index=idx)
        barriers = [{"level_price_btc": 80_000.0, "barrier_type": "knock_in", "notional_usd": 200e6}]
        res = run_confluence_backtest(btc, barriers, _walls_df(n, put_wall=80_000.0))
        assert res["ready"] is True
        assert res["n_history"] == n
        assert "n_confluence_days" in res
        assert "ic_next_day" in res  # può essere None o float

    def test_empty_walls_not_ready(self):
        btc = pd.Series([85_000.0] * 3, index=pd.date_range("2026-01-01", periods=3))
        res = run_confluence_backtest(btc, None, pd.DataFrame())
        assert res["ready"] is False
        assert res["n_history"] == 0
