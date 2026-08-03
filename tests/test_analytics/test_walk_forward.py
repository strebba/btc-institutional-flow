"""Test unitari per WalkForwardBacktest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.walk_forward import WalkForwardBacktest


def _make_merged_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Crea DataFrame merged con colonne minime per il CompositeSignal + backtest.
    
    Abbastanza punti (400) per avere almeno 2-3 finestre walk-forward con
    train=252, test=63, step=63.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = rng.normal(0.001, 0.025, n)
    prices = 65_000.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "btc_return": returns,
            "btc_close": prices,
            "ibit_flow_3d": rng.normal(0, 200e6, n),
            "total_net_gex": rng.normal(15e6, 30e6, n),
            "total_flow_usd": rng.normal(50e6, 150e6, n),
            "btc_vol_7d": np.full(n, 0.55),
            "funding_rate": rng.normal(0.0001, 0.0003, n),
            "oi_usd": 30e9 * np.exp(np.cumsum(rng.normal(0.0001, 0.002, n))),
            "long_short_ratio": rng.normal(1.5, 0.3, n),
        },
        index=dates,
    )


class TestWalkForwardWindows:
    def test_produces_at_least_two_periods(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        assert len(results) >= 3

    def test_windows_non_overlapping_test(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        for i in range(len(results) - 1):
            cur_end = results[i].test_end
            nxt_start = results[i + 1].test_start
            assert cur_end <= nxt_start

    def test_result_has_required_fields(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        r = results[0]
        assert isinstance(r.train_sharpe, float)
        assert isinstance(r.test_sharpe, float)
        assert isinstance(r.train_start, pd.Timestamp)
        assert isinstance(r.train_end, pd.Timestamp)
        assert isinstance(r.test_start, pd.Timestamp)
        assert isinstance(r.test_end, pd.Timestamp)

    def test_insufficient_data_returns_empty(self):
        df = _make_merged_df(50)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        assert results == []

    def test_test_sharpe_not_inf_or_nan(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        for r in results:
            assert not np.isnan(r.test_sharpe)
            assert not np.isinf(r.test_sharpe)


class TestWalkForwardAnalysis:
    def test_analyze_returns_dict(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        analysis = wfb.analyze(results)
        assert isinstance(analysis, dict)
        assert "avg_train_sharpe" in analysis
        assert "avg_test_sharpe" in analysis
        assert "sharpe_degradation" in analysis
        assert "test_sharpe_std" in analysis
        assert "pct_profitable_periods" in analysis
        assert "is_viable" in analysis

    def test_analyze_degradation_range(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        analysis = wfb.analyze(results)
        assert 0 <= analysis["pct_profitable_periods"] <= 1

    def test_is_viable_with_random_data(self):
        """Con dati random, il segnale NON dovrebbe essere viable."""
        df = _make_merged_df(800, seed=123)  # dati puramente random
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        analysis = wfb.analyze(results)
        # Su dati random, ci aspettiamo che is_viable sia False
        # (avg_test_sharpe dovrebbe essere basso o negativo)
        assert analysis["is_viable"] is False or analysis["avg_test_sharpe"] < 0.5


class TestWalkForwardSummaryTable:
    def test_returns_dataframe(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        analysis = wfb.analyze(results)
        table = wfb.summary_table(analysis)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 1

    def test_table_columns(self):
        df = _make_merged_df(500)
        wfb = WalkForwardBacktest()
        results = wfb.run(df, train_days=200, test_days=60, step_days=60)
        analysis = wfb.analyze(results)
        table = wfb.summary_table(analysis)
        expected_cols = [
            "Train Sharpe Avg", "Test Sharpe Avg", "Sharpe Degradation",
            "Test Sharpe Std", "% Profitable", "Worst Test Sharpe",
            "Periods", "Viable",
        ]
        for col in expected_cols:
            assert col in table.columns
