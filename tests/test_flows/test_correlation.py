"""Test unitari per FlowCorrelation."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from src.flows.correlation import FlowCorrelation
from src.flows.models import AggregateFlows


def _make_merged(n: int = 90) -> pd.DataFrame:
    """Genera un DataFrame merged sintetico per i test."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    df  = pd.DataFrame({
        "ibit_flow":  rng.normal(1e8, 3e8, n),
        "total_flow": rng.normal(3e8, 5e8, n),
        "btc_close":  np.cumsum(rng.normal(0, 1000, n)) + 50_000,
        "btc_return": rng.normal(0.001, 0.03, n),
        "ibit_close": np.cumsum(rng.normal(0, 0.5, n)) + 50,
        "btc_vol_7d": rng.uniform(0.3, 0.8, n),
    }, index=idx)
    df["ibit_btc_ratio"]   = df["ibit_close"] / df["btc_close"]
    df["btc_return_next1d"] = df["btc_return"].shift(-1)
    df["btc_return_next2d"] = df["btc_return"].shift(-2)
    df["ibit_flow_3d"]     = df["ibit_flow"].rolling(3).sum()
    df["total_flow_3d"]    = df["total_flow"].rolling(3).sum()
    return df


@pytest.fixture
def merged_df():
    return _make_merged()


@pytest.fixture
def agg_flows():
    base = date(2024, 1, 1)
    return [
        AggregateFlows(
            date=base + timedelta(days=i),
            total_flow_usd=float(i * 1e7),
            ibit_flow_usd=float(i * 5e6),
        )
        for i in range(30)
    ]


class TestMerge:
    def test_merge_columns(self, agg_flows):
        engine = FlowCorrelation()
        prices = _make_merged(40)[["btc_close", "btc_return", "ibit_close", "ibit_btc_ratio", "btc_vol_7d"]]
        merged = engine.merge(agg_flows, prices)
        assert "ibit_flow" in merged.columns
        assert "total_flow" in merged.columns

    def test_empty_flows(self):
        engine = FlowCorrelation()
        result = engine.merge([], pd.DataFrame())
        assert result.empty


class TestRollingCorrelations:
    def test_returns_dict(self, merged_df):
        engine = FlowCorrelation()
        result = engine.rolling_correlations(merged_df, windows=[30])
        assert "30d" in result

    def test_correlation_range(self, merged_df):
        engine = FlowCorrelation()
        result = engine.rolling_correlations(merged_df, windows=[30])
        corr   = result["30d"]["ibit_flow_vs_btc_return"].dropna()
        assert (corr >= -1.0).all() and (corr <= 1.0).all()

    def test_multiple_windows(self, merged_df):
        engine = FlowCorrelation()
        result = engine.rolling_correlations(merged_df, windows=[30, 60, 90])
        assert len(result) == 3


class TestSummaryStats:
    def test_keys_present(self, merged_df):
        engine = FlowCorrelation()
        stats  = engine.summary_stats(merged_df)
        assert "ibit" in stats
        assert "btc" in stats

    def test_net_flow_sign(self, merged_df):
        engine = FlowCorrelation()
        stats  = engine.summary_stats(merged_df)
        net = stats["ibit"]["net_flow_usd_b"]
        inflow  = stats["ibit"]["total_inflow_usd_b"]
        outflow = stats["ibit"]["total_outflow_usd_b"]
        assert abs(net - (inflow + outflow)) < 0.01  # outflow è già negativo



class TestMergeEth:
    @pytest.fixture
    def eth_flows(self):
        base = date(2024, 1, 1)
        return [
            AggregateFlows(
                date=base + timedelta(days=i),
                total_flow_usd=float(i * 2e6),
                ibit_flow_usd=0.0,
                flows_by_ticker={"ETHA": float(i * 1e6), "FETH": float(i * 1e6)},
                lead_flow_usd=float(i * 1e6),
            )
            for i in range(30)
        ]

    @pytest.fixture
    def eth_prices(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "eth_close": np.cumsum(rng.normal(0, 30, 30)) + 2_500,
            "eth_return": rng.normal(0, 0.03, 30),
            "eth_vol_7d": rng.uniform(0.4, 0.9, 30),
        }, index=idx)

    def test_colonne_eth_senza_residui_btc(self, eth_flows, eth_prices):
        merged = FlowCorrelation().merge(eth_flows, eth_prices, asset="ETH")
        assert {"etha_flow", "etha_flow_3d", "eth_return_next1d", "feth_flow"} <= set(merged.columns)
        assert "ibit_flow" not in merged.columns
        assert not any(c.startswith("btc") for c in merged.columns)

    def test_statistiche_e_correlazioni_eth(self, eth_flows, eth_prices):
        engine = FlowCorrelation()
        merged = engine.merge(eth_flows, eth_prices, asset="ETH")
        stats = engine.summary_stats(merged, asset="ETH")
        assert stats["etha"]["net_flow_usd_b"] == pytest.approx(sum(i * 1e6 for i in range(30)) / 1e9)
        assert "eth" in stats and "ETHA" not in stats.get("by_ticker", {})
        roll = engine.rolling_correlations(merged, windows=[10], asset="ETH")
        assert "etha_flow_vs_eth_return" in roll["10d"].columns
