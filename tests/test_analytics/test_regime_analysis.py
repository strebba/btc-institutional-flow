"""Test unitari per RegimeAnalysis."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.analytics.regime_analysis import RegimeAnalysis, RegimeComparisonResult, RegimeStats
from src.gex.models import GexSnapshot


def _make_merged_df(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Crea DataFrame merged con btc_return e btc_vol_7d."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = rng.normal(0, 0.02, n)
    vol7d = pd.Series(returns).rolling(7).std() * (252 ** 0.5)
    return pd.DataFrame(
        {"btc_return": returns, "btc_vol_7d": vol7d.values},
        index=dates,
    )


def _make_gex_series(n: int = 180, mostly_positive: bool = True) -> pd.Series:
    """Crea serie GEX con sign controllato."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Prima metà positiva, seconda metà negativa
    vals = np.concatenate([
        rng.uniform(2e6, 50e6, n // 2),   # positivo
        rng.uniform(-50e6, -2e6, n // 2), # negativo
    ])
    return pd.Series(vals, index=dates)


def _make_snapshot(gex: float, spot: float = 70_000, day_offset: int = 0) -> GexSnapshot:
    from datetime import timedelta
    ts = datetime(2024, 1, 1) + timedelta(days=day_offset)
    return GexSnapshot(
        timestamp=ts,
        spot_price=spot,
        total_net_gex=gex,
        gamma_flip_price=70_000,
        put_wall=65_000,
        call_wall=75_000,
        max_pain=70_000,
    )


@pytest.fixture
def analyzer() -> RegimeAnalysis:
    return RegimeAnalysis()


@pytest.fixture
def merged_df() -> pd.DataFrame:
    return _make_merged_df()


@pytest.fixture
def gex_series() -> pd.Series:
    return _make_gex_series()


class TestClassifyRegime:
    def test_positive_gamma(self, analyzer):
        assert analyzer.classify_regime(5_000_000) == "positive_gamma"

    def test_negative_gamma(self, analyzer):
        assert analyzer.classify_regime(-5_000_000) == "negative_gamma"

    def test_neutral_above_zero(self, analyzer):
        assert analyzer.classify_regime(500_000) == "neutral"

    def test_neutral_below_zero(self, analyzer):
        assert analyzer.classify_regime(-500_000) == "neutral"

    def test_exactly_threshold_positive(self, analyzer):
        # default threshold = 1_000_000
        assert analyzer.classify_regime(1_000_001) == "positive_gamma"

    def test_exactly_threshold_negative(self, analyzer):
        assert analyzer.classify_regime(-1_000_001) == "negative_gamma"


class TestBuildGexSeries:
    def test_empty_snapshots(self, analyzer):
        result = analyzer.build_gex_series([])
        assert result.empty

    def test_returns_series(self, analyzer):
        snaps = [_make_snapshot(10e6, day_offset=0), _make_snapshot(-5e6, day_offset=1)]
        result = analyzer.build_gex_series(snaps)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_sorted_index(self, analyzer):
        snaps = [_make_snapshot(10e6, day_offset=0), _make_snapshot(-5e6, day_offset=1)]
        result = analyzer.build_gex_series(snaps)
        assert result.index.is_monotonic_increasing


class TestAnalyze:
    def test_returns_regime_comparison_result(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        assert isinstance(result, RegimeComparisonResult)

    def test_has_both_stats(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        # Con GEX sia positivo che negativo deve avere entrambi
        assert result.positive_stats is not None
        assert result.negative_stats is not None

    def test_stats_types(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        if result.positive_stats:
            assert isinstance(result.positive_stats, RegimeStats)
            assert result.positive_stats.regime == "positive_gamma"
        if result.negative_stats:
            assert isinstance(result.negative_stats, RegimeStats)
            assert result.negative_stats.regime == "negative_gamma"

    def test_p_value_in_range(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        assert 0.0 <= result.p_value <= 1.0

    def test_significant_flag_consistency(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        assert result.significant == (result.p_value < 0.05)

    def test_uses_gex_col_when_no_series(self, analyzer, merged_df):
        df = merged_df.copy()
        # Aggiunge colonna GEX nel DataFrame
        rng = np.random.default_rng(1)
        df["total_gex"] = rng.uniform(-50e6, 50e6, len(df))
        result = analyzer.analyze(df)
        assert isinstance(result, RegimeComparisonResult)

    def test_zero_gex_fallback(self, analyzer, merged_df):
        """Senza GEX → tutto neutral → stats potrebbero essere None."""
        result = analyzer.analyze(merged_df)
        assert isinstance(result, RegimeComparisonResult)

    def test_gex_vol_correlation(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        if result.gex_vol_correlation is not None:
            assert isinstance(result.gex_vol_correlation, pd.Series)


class TestRegimeStats:
    def test_sharpe_sign(self, analyzer, merged_df, gex_series):
        """Sharpe deve avere lo stesso segno del mean_return."""
        result = analyzer.analyze(merged_df, gex_series)
        for stats in [result.positive_stats, result.negative_stats]:
            if stats and stats.std_return > 0:
                assert np.sign(stats.sharpe) == np.sign(stats.mean_return)

    def test_n_obs_positive(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        for stats in [result.positive_stats, result.negative_stats]:
            if stats:
                assert stats.n_obs > 0


class TestInterpret:
    def test_returns_string(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        assert isinstance(result.interpretation, str)
        assert "Regime" in result.interpretation

    def test_includes_statistics(self, analyzer, merged_df, gex_series):
        result = analyzer.analyze(merged_df, gex_series)
        assert "POSITIVE GAMMA" in result.interpretation or "NEGATIVE GAMMA" in result.interpretation

    def test_no_stats_empty_interpretation(self, analyzer):
        text = analyzer._interpret(None, None, 0.0, 1.0)
        assert isinstance(text, str)
