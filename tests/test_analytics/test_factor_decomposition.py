"""Test unitari per FactorDecomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.factor_decomposition import FactorDecomposition


def _make_returns(n: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.001, 0.02, n), index=dates, name="returns")


def _make_market_factor(n: int = 252, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.0005, 0.015, n), index=dates, name="market")


def _make_momentum_factor(n: int = 252, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.0002, 0.01, n), index=dates, name="momentum")


def _make_vol_factor(n: int = 252, seed: int = 13) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.0, 0.005, n), index=dates, name="volatility")


class TestCalculateFactorExposures:
    def test_returns_factor_exposures(self):
        returns = _make_returns()
        factors = pd.DataFrame({
            "market": _make_market_factor(),
            "momentum": _make_momentum_factor(),
        })
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(returns, factors)
        assert "factor_exposures" in result
        assert "market" in result["factor_exposures"]

    def test_exposure_has_required_fields(self):
        returns = _make_returns()
        factors = pd.DataFrame({"market": _make_market_factor()})
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(returns, factors)
        exp = result["factor_exposures"]["market"]
        assert "beta" in exp
        assert "t_stat" in exp
        assert "p_value" in exp
        assert "significant" in exp

    def test_alpha_returned(self):
        returns = _make_returns()
        factors = pd.DataFrame({"market": _make_market_factor()})
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(returns, factors)
        assert "alpha_annual" in result
        assert "alpha_t_stat" in result
        assert "alpha_significant" in result

    def test_r_squared_in_range(self):
        returns = _make_returns()
        factors = pd.DataFrame({"market": _make_market_factor()})
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(returns, factors)
        assert 0 <= result["r_squared"] <= 1

    def test_strategy_pure_beta_detected(self):
        """Una strategia che è market × 1.2 + piccolo noise → alpha ~0."""
        rng = np.random.default_rng(42)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        mkt = pd.Series(rng.normal(0.0005, 0.015, n), index=dates)
        strat = mkt * 1.2 + rng.normal(0, 0.001, n)  # quasi tutta beta
        factors = pd.DataFrame({"market": mkt})
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(strat, factors)
        assert not result["alpha_significant"]  # alpha non significativo
        assert result["r_squared"] > 0.8  # spiegato quasi tutto dal mercato

    def test_strategy_pure_alpha_detected(self):
        """Una strategia con puro alpha (costante + noise) → nessuna esposizione."""
        rng = np.random.default_rng(99)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        mkt = pd.Series(rng.normal(0.0005, 0.015, n), index=dates)
        alpha_daily = 0.002 / 252  # 0.2% alpha annualizzato
        strat = pd.Series(alpha_daily + rng.normal(0, 0.005, n), index=dates)
        factors = pd.DataFrame({"market": mkt})
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(strat, factors)
        # Il market beta dovrebbe essere non significativo
        assert not result["factor_exposures"]["market"]["significant"]

    def test_empty_returns(self):
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(pd.Series(dtype=float), pd.DataFrame())
        assert result["factor_exposures"] == {}
        assert result["alpha_annual"] == 0.0

    def test_misaligned_indexes(self):
        returns = _make_returns(200)
        factors = pd.DataFrame({"market": _make_market_factor(300)})
        fd = FactorDecomposition()
        result = fd.calculate_factor_exposures(returns, factors)
        assert result["r_squared"] >= 0


class TestDecomposeStrategyReturns:
    def test_returns_decomposition_structure(self):
        returns = _make_returns()
        factors = pd.DataFrame({"market": _make_market_factor()})
        fd = FactorDecomposition()
        result = fd.decompose_strategy_returns(returns, factors)
        assert "total_return" in result
        assert "factor_contributions" in result
        assert "true_alpha" in result
        assert "is_true_alpha" in result

    def test_true_alpha_approx_total_when_no_exposure(self):
        rng = np.random.default_rng(42)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        mkt = pd.Series(rng.normal(0.0005, 0.015, n), index=dates)
        strat = pd.Series(0.001 + rng.normal(0, 0.002, n), index=dates)
        factors = pd.DataFrame({"market": mkt})
        fd = FactorDecomposition()
        result = fd.decompose_strategy_returns(strat, factors)
        # Senza esposizione ai fattori, true_alpha ~ total_return
        assert abs(result["true_alpha"] - result["total_return"]) < 0.5

    def test_is_true_alpha_false_for_pure_beta(self):
        rng = np.random.default_rng(42)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        mkt = pd.Series(rng.normal(0.001, 0.015, n), index=dates)
        strat = mkt * 1.5  # 1.5x market beta
        factors = pd.DataFrame({"market": mkt})
        fd = FactorDecomposition()
        result = fd.decompose_strategy_returns(strat, factors)
        assert not result["is_true_alpha"]


class TestSummaryReport:
    def test_report_is_string(self):
        returns = _make_returns()
        factors = pd.DataFrame({"market": _make_market_factor()})
        fd = FactorDecomposition()
        decomp = fd.decompose_strategy_returns(returns, factors)
        exposures = fd.calculate_factor_exposures(returns, factors)
        report = fd.summary_report(decomp, exposures)
        assert isinstance(report, str)
        assert "FACTOR DECOMPOSITION" in report

    def test_report_handles_empty(self):
        fd = FactorDecomposition()
        decomp = fd.decompose_strategy_returns(pd.Series(dtype=float), pd.DataFrame())
        exposures = fd.calculate_factor_exposures(pd.Series(dtype=float), pd.DataFrame())
        report = fd.summary_report(decomp, exposures)
        assert "Dati insufficienti" in report
