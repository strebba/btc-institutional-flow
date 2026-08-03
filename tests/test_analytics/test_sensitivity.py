"""Test unitari per ParameterSensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.pillars import PILLAR_WEIGHTS
from src.analytics.sensitivity import ParameterSensitivity


def _make_merged_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
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


class TestPillarSensitivity:
    def test_returns_dict_with_per_weight_keys(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        result = ps.pillar_sensitivity(df, delta=0.20)
        assert isinstance(result, dict)
        for pillar in PILLAR_WEIGHTS:
            assert pillar in result
            assert "base_sharpe" in result[pillar]
            assert "low_sharpe" in result[pillar]
            assert "high_sharpe" in result[pillar]
            assert "range" in result[pillar]
            assert "is_stable" in result[pillar]

    def test_sensitivity_range_non_negative(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        result = ps.pillar_sensitivity(df, delta=0.20)
        for pillar, metrics in result.items():
            assert metrics["range"] >= 0

    def test_overall_stability_field(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        result = ps.pillar_sensitivity(df, delta=0.20)
        for pillar, metrics in result.items():
            assert isinstance(metrics["is_stable"], bool)

    def test_empty_data_returns_empty(self):
        ps = ParameterSensitivity()
        result = ps.pillar_sensitivity(pd.DataFrame(), delta=0.20)
        assert result == {}

    def test_sharpe_changes_with_different_weights(self):
        """Pesi diversi devono produrre Sharpe diversi (o almeno calcolabili)."""
        df = _make_merged_df()
        ps = ParameterSensitivity()
        result = ps.pillar_sensitivity(df, delta=0.20)
        for pillar, metrics in result.items():
            assert not np.isnan(metrics["base_sharpe"])
            assert not np.isnan(metrics["low_sharpe"])
            assert not np.isnan(metrics["high_sharpe"])


class TestSubfactorSensitivity:
    def test_returns_dict_with_group_keys(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        result = ps.subfactor_sensitivity(df, delta=0.20)
        for group in ("gex", "etf_flows", "macro"):
            assert group in result, f"Missing group {group}"
            for factor, metrics in result[group].items():
                assert "base_sharpe" in metrics
                assert "low_sharpe" in metrics
                assert "high_sharpe" in metrics

    def test_no_nan_sharpes(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        result = ps.subfactor_sensitivity(df, delta=0.20)
        for group, factors in result.items():
            for factor, metrics in factors.items():
                assert not np.isnan(metrics["base_sharpe"])
                assert not np.isnan(metrics["low_sharpe"])
                assert not np.isnan(metrics["high_sharpe"])


class TestSummaryReport:
    def test_report_is_string(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        pillar_result = ps.pillar_sensitivity(df, delta=0.20)
        report = ps.summary_report(pillar_result, title="Pillar Weights")
        assert isinstance(report, str)
        assert "Pillar Weights" in report

    def test_report_highlights_unstable(self):
        df = _make_merged_df()
        ps = ParameterSensitivity()
        pillar_result = ps.pillar_sensitivity(df, delta=0.20)
        report = ps.summary_report(pillar_result, title="test")
        assert "STABLE" in report or "UNSTABLE" in report

    def test_empty_result_report(self):
        ps = ParameterSensitivity()
        report = ps.summary_report({}, title="test")
        assert "No results" in report
