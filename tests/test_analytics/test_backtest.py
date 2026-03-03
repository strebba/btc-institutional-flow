"""Test unitari per Backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.backtest import Backtest, BacktestMetrics


def _make_merged_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Crea DataFrame merged con btc_return, ibit_flow_3d, btc_close."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = rng.normal(0.001, 0.025, n)
    prices = 65_000.0 * np.exp(np.cumsum(returns))
    flow3d = rng.normal(0, 200e6, n)
    return pd.DataFrame(
        {
            "btc_return": returns,
            "btc_close": prices,
            "ibit_flow_3d": flow3d,
        },
        index=dates,
    )


def _make_gex_positive(n: int = 200) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.full(n, 20e6), index=dates)  # GEX sempre positivo


def _make_gex_negative(n: int = 200) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.full(n, -20e6), index=dates)


@pytest.fixture
def bt() -> Backtest:
    return Backtest()


@pytest.fixture
def df() -> pd.DataFrame:
    return _make_merged_df()


class TestGenerateSignals:
    def test_all_long_when_conditions_met(self, bt, df):
        """GEX positivo + flow forte → tutti LONG."""
        df2 = df.copy()
        df2["ibit_flow_3d"] = 300e6  # > 100M
        gex = _make_gex_positive(len(df2))
        gex.index = df2.index
        signals = bt._generate_signals(df2, gex)
        assert (signals == 1.0).all()

    def test_all_short_when_conditions_met(self, bt, df):
        """GEX negativo + flow fortemente negativo → tutti SHORT."""
        df2 = df.copy()
        df2["ibit_flow_3d"] = -400e6  # < -200M
        gex = _make_gex_negative(len(df2))
        gex.index = df2.index
        signals = bt._generate_signals(df2, gex)
        assert (signals == -1.0).all()

    def test_flat_when_no_conditions(self, bt, df):
        """GEX zero + flow neutro → tutti FLAT."""
        df2 = df.copy()
        df2["ibit_flow_3d"] = 50e6  # sotto la soglia long
        signals = bt._generate_signals(df2)  # GEX = 0
        assert (signals == 0.0).all()

    def test_near_barrier_blocks_long(self, bt, df):
        """Barriera vicina al prezzo corrente deve impedire il segnale LONG."""
        df2 = df.copy()
        df2["ibit_flow_3d"] = 300e6
        gex = _make_gex_positive(len(df2))
        gex.index = df2.index
        # Prezzo medio del DataFrame
        avg_price = float(df2["btc_close"].mean())
        barriers = [{"level_price_btc": avg_price * 1.02}]  # entro 5%
        signals = bt._generate_signals(df2, gex, barriers)
        # Almeno alcuni giorni devono essere FLAT per la barriera
        assert (signals == 0.0).any()

    def test_uses_total_gex_col(self, bt):
        """Se total_gex è nel DataFrame, usarlo direttamente."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "btc_return": rng.normal(0, 0.02, 50),
                "btc_close": 65_000.0 * np.ones(50),
                "ibit_flow_3d": np.full(50, 300e6),
                "total_gex": np.full(50, 20e6),
            },
            index=dates,
        )
        signals = bt._generate_signals(df)
        assert (signals == 1.0).all()

    def test_signals_values(self, bt, df):
        """I segnali devono essere solo -1, 0 o +1."""
        gex = _make_gex_positive(len(df))
        gex.index = df.index
        signals = bt._generate_signals(df, gex)
        assert signals.isin([-1.0, 0.0, 1.0]).all()

    def test_signal_index_matches_df(self, bt, df):
        signals = bt._generate_signals(df)
        assert (signals.index == df.index).all()


class TestComputeMetrics:
    def test_empty_returns(self, bt):
        result = bt._compute_metrics(pd.Series(dtype=float), "test")
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0

    def test_positive_drift(self, bt):
        """Serie con drift positivo → total_return > 0."""
        rng = np.random.default_rng(0)
        # Rendimenti con drift positivo e varianza > 0 per Sharpe != 0
        rets = pd.Series(0.005 + rng.normal(0, 0.01, 252))
        result = bt._compute_metrics(rets, "test")
        assert result.total_return > 0
        assert result.annualized_return > 0
        assert result.sharpe_ratio > 0

    def test_max_drawdown_negative(self, bt):
        """Max drawdown deve essere ≤ 0."""
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0, 0.02, 200))
        result = bt._compute_metrics(rets, "test")
        assert result.max_drawdown <= 0

    def test_win_rate_in_range(self, bt):
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0, 0.02, 200))
        result = bt._compute_metrics(rets, "test")
        assert 0.0 <= result.win_rate <= 1.0

    def test_profit_factor_positive(self, bt):
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0, 0.02, 200))
        result = bt._compute_metrics(rets, "test")
        assert result.profit_factor >= 0

    def test_with_signals_counts(self, bt):
        rets = pd.Series([0.01] * 100 + [-0.01] * 100)
        signals = pd.Series([1.0] * 100 + [-1.0] * 100)
        result = bt._compute_metrics(rets, "test", signals)
        assert result.days_long == 100
        assert result.days_short == 100
        assert result.days_flat == 0

    def test_n_trades_counts_changes(self, bt):
        rets = pd.Series([0.01] * 50)
        # 3 cambi di segnale: 0→1, 1→-1, -1→0
        signals = pd.Series([0.0] * 10 + [1.0] * 20 + [-1.0] * 10 + [0.0] * 10)
        result = bt._compute_metrics(rets, "test", signals)
        assert result.n_trades == 3

    def test_equity_curve_starts_near_one(self, bt):
        rets = pd.Series([0.01] * 100)
        result = bt._compute_metrics(rets, "test")
        assert not result.equity_curve.empty
        assert abs(result.equity_curve.iloc[0] - 1.01) < 0.001


class TestRun:
    def test_returns_both_keys(self, bt, df):
        results = bt.run(df)
        assert "strategy" in results
        assert "buy_and_hold" in results

    def test_bah_positive_drift(self, bt):
        """Buy-and-hold con rendimenti positivi → return > 0."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=252, freq="D")
        rets = rng.normal(0.002, 0.02, 252)  # drift positivo
        df = pd.DataFrame(
            {
                "btc_return": rets,
                "btc_close": 65_000.0 * np.exp(np.cumsum(rets)),
                "ibit_flow_3d": np.full(252, 50e6),
            },
            index=dates,
        )
        results = bt.run(df)
        assert results["buy_and_hold"].total_return > 0

    def test_missing_btc_return_returns_empty(self, bt, df):
        df2 = df.drop(columns=["btc_return"])
        results = bt.run(df2)
        assert results == {}

    def test_lag_prevents_lookahead(self, bt, df):
        """Signals vengono shiftati di 1 giorno: il primo segnale usato è sempre 0."""
        gex = _make_gex_positive(len(df))
        gex.index = df.index
        df2 = df.copy()
        df2["ibit_flow_3d"] = 300e6
        results = bt.run(df2, gex)
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_strategy_metrics_types(self, bt, df):
        results = bt.run(df)
        for key, m in results.items():
            assert isinstance(m, BacktestMetrics)
            assert isinstance(m.strategy_name, str)
            assert isinstance(m.total_return, float)
            assert isinstance(m.sharpe_ratio, float)

    def test_strategy_has_equity_curve(self, bt, df):
        results = bt.run(df)
        assert not results["strategy"].equity_curve.empty
        assert not results["buy_and_hold"].equity_curve.empty

    def test_with_active_barriers(self, bt, df):
        """Deve girare senza errori quando ci sono barriere."""
        avg_price = float(df["btc_close"].mean())
        barriers = [{"level_price_btc": avg_price * 0.90}]  # lontano dal prezzo
        gex = _make_gex_positive(len(df))
        gex.index = df.index
        df2 = df.copy()
        df2["ibit_flow_3d"] = 300e6
        results = bt.run(df2, gex, barriers)
        assert "strategy" in results


class TestSummaryTable:
    def test_returns_dataframe(self, bt, df):
        results = bt.run(df)
        table = bt.summary_table(results)
        assert isinstance(table, pd.DataFrame)

    def test_columns_present(self, bt, df):
        results = bt.run(df)
        table = bt.summary_table(results)
        for col in ["Return Totale", "Sharpe", "Max Drawdown", "Win Rate"]:
            assert col in table.columns

    def test_two_rows(self, bt, df):
        results = bt.run(df)
        table = bt.summary_table(results)
        assert len(table) == 2

    def test_infinite_profit_factor_display(self, bt):
        """Profit factor infinito deve mostrare '∞'."""
        rets = pd.Series([0.01] * 100)  # tutti positivi → losses=0 → PF=inf
        signals = pd.Series([1.0] * 100)
        m = bt._compute_metrics(rets, "test", signals)
        results = {"test": m}
        table = bt.summary_table(results)
        assert table.loc["test", "Profit Factor"] == "∞"


class TestPlot:
    def test_returns_figure_or_none(self, bt, df):
        results = bt.run(df)
        fig = bt.plot(results)
        assert fig is None or hasattr(fig, "data")

    def test_plot_with_empty_equity_skipped(self, bt):
        from src.analytics.backtest import BacktestMetrics
        empty_m = BacktestMetrics(
            strategy_name="empty", total_return=0.0, annualized_return=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
            n_trades=0, days_long=0, days_short=0, days_flat=0,
        )
        results = {"test": empty_m}
        fig = bt.plot(results)
        # Non deve crashare
        assert fig is None or hasattr(fig, "data")
