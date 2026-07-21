"""Test unitari per signal_validation.py (IC, alpha decay, null model)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.signal_validation import (
    alpha_decay,
    null_model_ic,
    rolling_information_coefficient,
    signal_validation_summary,
    spearman_ic,
)


def _make_scores_and_rets(
    n: int = 200, seed: int = 42, ic_strength: float = 0.3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Crea segnali e rendimenti con IC controllato.

    Il DGP: return_{t+1} = ic_strength * signal_z_t + noise.
    signal_z_t è il segnale standardizzato (z-score).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    raw_signals = rng.uniform(20, 80, n)
    signal_z = (raw_signals - raw_signals.mean()) / raw_signals.std()
    noise = rng.normal(0, 0.02, n)

    forward_rets = ic_strength * signal_z * 0.02 + noise
    forward_rets = pd.Series(forward_rets, index=dates)

    rets = pd.Series(rng.normal(0.001, 0.02, n), index=dates)
    rets.iloc[:-1] = forward_rets.iloc[:-1].values

    return pd.Series(raw_signals, index=dates), rets, forward_rets


class TestSpearmanIC:
    def test_ic_perfect_positive(self):
        scores = pd.Series(np.arange(100), name="score")
        fwd = pd.Series(np.arange(100), name="fwd")
        assert spearman_ic(scores, fwd) == pytest.approx(1.0, abs=1e-6)

    def test_ic_perfect_negative(self):
        scores = pd.Series(np.arange(100), name="score")
        fwd = pd.Series(np.arange(99, -1, -1), name="fwd")
        assert spearman_ic(scores, fwd) == pytest.approx(-1.0, abs=1e-6)

    def test_ic_random_noise(self):
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.uniform(0, 100, 200))
        fwd = pd.Series(rng.normal(0, 0.02, 200))
        ic = spearman_ic(scores, fwd)
        assert ic is not None
        assert -0.3 < ic < 0.3

    def test_ic_known_relationship(self):
        scores, _, fwd = _make_scores_and_rets(ic_strength=0.5)
        ic = spearman_ic(scores, fwd)
        assert ic is not None
        assert ic > 0.15  # con 200 punti e IC_strength=0.5, IC reale > 0.15

    def test_ic_insufficient_data(self):
        scores = pd.Series([1.0, 2.0])
        fwd = pd.Series([0.01, 0.02])
        assert spearman_ic(scores, fwd) is None

    def test_ic_constant_scores(self):
        scores = pd.Series(np.ones(100))
        fwd = pd.Series(np.random.default_rng(0).normal(0, 0.02, 100))
        assert spearman_ic(scores, fwd) is None


class TestRollingIC:
    def test_returns_expected_keys(self):
        scores, _, fwd = _make_scores_and_rets(n=200)
        result = rolling_information_coefficient(scores, fwd, window=60, min_periods=20)
        for key in ("ic_mean", "ic_std", "ir", "t_stat", "pct_positive",
                     "ic_series", "is_significant", "n_windows"):
            assert key in result

    def test_insufficient_data(self):
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        fwd = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        result = rolling_information_coefficient(scores, fwd, min_periods=20)
        assert result["ic_mean"] is None
        assert result["n_windows"] == 0

    def test_ic_series_non_empty(self):
        scores, _, fwd = _make_scores_and_rets(n=200)
        result = rolling_information_coefficient(scores, fwd, window=60, min_periods=20)
        assert len(result["ic_series"]) > 0

    def test_pct_positive_in_range(self):
        scores, _, fwd = _make_scores_and_rets(n=200)
        result = rolling_information_coefficient(scores, fwd)
        pct = result.get("pct_positive")
        assert pct is not None
        assert 0.0 <= pct <= 1.0


class TestAlphaDecay:
    def test_returns_dataframe(self):
        scores, rets, _ = _make_scores_and_rets(n=200)
        df = alpha_decay(scores, rets, max_horizon=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_columns(self):
        scores, rets, _ = _make_scores_and_rets(n=200)
        df = alpha_decay(scores, rets, max_horizon=5)
        for col in ("horizon", "ic", "t_stat", "pct_positive"):
            assert col in df.columns

    def test_horizon_sequence(self):
        scores, rets, _ = _make_scores_and_rets(n=200)
        df = alpha_decay(scores, rets, max_horizon=5)
        assert list(df["horizon"]) == [1, 2, 3, 4, 5]


class TestNullModelIC:
    def test_returns_expected_keys(self):
        scores, _, fwd = _make_scores_and_rets(n=200)
        result = null_model_ic(scores, fwd, n_permutations=20)
        for key in ("null_ic_mean", "null_ic_std", "null_ic_95pct",
                     "actual_ic", "p_value_empirico"):
            assert key in result

    def test_null_ic_near_zero(self):
        scores, _, fwd = _make_scores_and_rets(n=200)
        result = null_model_ic(scores, fwd, n_permutations=50)
        null_mean = result["null_ic_mean"]
        assert null_mean is not None
        assert abs(null_mean) < 0.05  # null model deve avere IC ~ 0

    def test_null_ic_with_permuted_signal(self):
        scores, _, fwd = _make_scores_and_rets(n=200, ic_strength=0.0)
        result = null_model_ic(scores, fwd, n_permutations=50)
        assert result["p_value_empirico"] is not None

    def test_insufficient_data(self):
        scores = pd.Series([1.0, 2.0])
        fwd = pd.Series([0.01, 0.02])
        result = null_model_ic(scores, fwd, n_permutations=20)
        assert result["null_ic_mean"] is None


class TestSignalValidationSummary:
    def test_returns_structure(self):
        scores, rets, _ = _make_scores_and_rets(n=200)
        df = pd.DataFrame({"composite_score": scores, "btc_return": rets})
        result = signal_validation_summary(df, min_periods=20)
        assert "rolling_ic" in result
        assert "alpha_decay_df" in result
        assert "null_model" in result
        assert "raw_ic" in result

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "composite_score": [50.0, 50.0],
            "btc_return": [0.01, 0.02],
        })
        result = signal_validation_summary(df, min_periods=100)
        assert result["rolling_ic"]["ic_mean"] is None
        assert result["raw_ic"] is None

    def test_known_positive_ic_detected(self):
        scores, rets, _ = _make_scores_and_rets(n=300, ic_strength=0.5)
        df = pd.DataFrame({"composite_score": scores, "btc_return": rets})
        result = signal_validation_summary(df, min_periods=20, window=60)
        raw_ic = result["raw_ic"]
        assert raw_ic is not None
        assert raw_ic > 0.0  # con IC_strength=0.5, raw_ic deve essere positivo
