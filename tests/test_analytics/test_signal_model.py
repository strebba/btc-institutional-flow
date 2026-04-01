"""Test per SignalModel — scoring multi-fattore 7 input."""
from __future__ import annotations

import pytest
import pandas as pd

from src.analytics.signal_model import (
    SignalModel,
    SignalInputs,
    SignalResult,
    LONG_THRESHOLD,
    RISK_OFF_THRESHOLD,
    SIGNAL_LONG,
    SIGNAL_CAUTION,
    SIGNAL_RISK_OFF,
    _score_gex,
    _score_etf_flow,
    _score_funding_rate,
    _score_oi_change,
    _score_long_short_ratio,
    _score_put_call_ratio,
    _score_liquidations,
)


# ─── Fattori individuali ──────────────────────────────────────────────────────

class TestScoreGex:
    def test_large_positive(self):
        assert _score_gex(1e9) == 1.0

    def test_small_positive(self):
        assert 0.5 < _score_gex(1.0) < 1.0

    def test_zero(self):
        # 0 non è > 0, cade nel ramo slightly-negative
        assert _score_gex(0) == 0.35

    def test_small_negative(self):
        assert 0.0 < _score_gex(-1e8) < 0.5

    def test_large_negative(self):
        assert _score_gex(-1e9) == 0.0


class TestScoreEtfFlow:
    def test_max_positive(self):
        assert _score_etf_flow(1e9) == 1.0

    def test_max_negative(self):
        assert _score_etf_flow(-1e9) == 0.0

    def test_zero(self):
        assert abs(_score_etf_flow(0) - 0.5) < 0.01

    def test_monotonic(self):
        """Più alto il flusso, più alto lo score."""
        assert _score_etf_flow(500e6) > _score_etf_flow(0) > _score_etf_flow(-500e6)


class TestScoreFundingRate:
    def test_negative_rate_bullish(self):
        """Funding negativo = paura = contrarian bullish."""
        assert _score_funding_rate(-5.0) > 0.7

    def test_neutral(self):
        assert 0.5 < _score_funding_rate(10.0) < 0.8

    def test_high_rate_bearish(self):
        assert _score_funding_rate(80.0) < 0.2

    def test_extreme_rate(self):
        assert _score_funding_rate(100.0) == 0.10

    def test_monotonic_decreasing(self):
        assert _score_funding_rate(-10) > _score_funding_rate(20) > _score_funding_rate(60)


class TestScoreOiChange:
    def test_large_increase(self):
        assert _score_oi_change(20.0) == 0.85

    def test_stable(self):
        assert _score_oi_change(0.0) == 0.50

    def test_large_decrease(self):
        assert _score_oi_change(-20.0) == 0.15

    def test_moderate_decrease(self):
        assert 0.3 < _score_oi_change(-10.0) < 0.5


class TestScoreLongShort:
    def test_crowded_long(self):
        assert _score_long_short_ratio(3.0) < 0.2

    def test_balanced(self):
        assert abs(_score_long_short_ratio(1.0) - 0.5) < 0.1

    def test_more_shorts(self):
        assert _score_long_short_ratio(0.5) > 0.7

    def test_monotonic_decreasing(self):
        assert _score_long_short_ratio(0.5) > _score_long_short_ratio(1.5) > _score_long_short_ratio(2.5)


class TestScorePutCall:
    def test_high_hedging(self):
        assert _score_put_call_ratio(2.0) >= 0.85

    def test_neutral(self):
        assert abs(_score_put_call_ratio(1.0) - 0.5) < 0.2

    def test_complacency(self):
        assert _score_put_call_ratio(0.4) < 0.2


class TestScoreLiquidations:
    def test_quiet_market(self):
        # < 100M totale → neutro
        assert _score_liquidations(40e6, 40e6) == 0.50

    def test_long_cascade_bottom_signal(self):
        # Molte liquidazioni long = capitulation = contrarian long
        score = _score_liquidations(800e6, 200e6)
        assert score > 0.55

    def test_short_squeeze(self):
        # Molte liquidazioni short = squeeze = contrarian short
        score = _score_liquidations(100e6, 800e6)
        assert score < 0.45


# ─── SignalModel.compute (live) ───────────────────────────────────────────────

class TestSignalModelCompute:
    def setup_method(self):
        self.model = SignalModel()

    def test_returns_signal_result(self):
        result = self.model.compute(SignalInputs(gex_usd=1e8, etf_flow_3d_usd=200e6))
        assert isinstance(result, SignalResult)

    def test_score_in_range(self):
        result = self.model.compute(SignalInputs(
            gex_usd=1e8, etf_flow_3d_usd=200e6,
            funding_rate_annualized_pct=15.0, oi_change_7d_pct=5.0,
        ))
        assert 0 <= result.score <= 100

    def test_strong_long_conditions(self):
        """Tutti i fattori bullish → LONG."""
        result = self.model.compute(SignalInputs(
            gex_usd=800e6,
            etf_flow_3d_usd=500e6,
            funding_rate_annualized_pct=-2.0,   # negativo = paura
            oi_change_7d_pct=12.0,
            long_short_ratio=0.6,               # più short = contrarian long
            put_call_ratio=1.8,                 # alto hedging
            liquidations_long_24h_usd=600e6,
            liquidations_short_24h_usd=100e6,
        ))
        assert result.signal == SIGNAL_LONG
        assert result.score >= LONG_THRESHOLD

    def test_strong_risk_off_conditions(self):
        """Tutti i fattori bearish → RISK_OFF."""
        result = self.model.compute(SignalInputs(
            gex_usd=-1e9,
            etf_flow_3d_usd=-600e6,
            funding_rate_annualized_pct=85.0,   # estremo surriscaldamento
            oi_change_7d_pct=-20.0,
            long_short_ratio=2.5,               # retail crowded long
            put_call_ratio=0.3,                 # complacency
            liquidations_long_24h_usd=50e6,
            liquidations_short_24h_usd=500e6,
        ))
        assert result.signal == SIGNAL_RISK_OFF
        assert result.score < RISK_OFF_THRESHOLD

    def test_barrier_override_blocks_long(self):
        """Barriera attiva vicina deve bloccare un segnale LONG."""
        result_no_barrier = self.model.compute(SignalInputs(
            gex_usd=800e6,
            etf_flow_3d_usd=500e6,
            funding_rate_annualized_pct=-2.0,
        ))
        result_with_barrier = self.model.compute(SignalInputs(
            gex_usd=800e6,
            etf_flow_3d_usd=500e6,
            funding_rate_annualized_pct=-2.0,
            near_active_barrier=True,
        ))
        # Con barriera non deve essere LONG (se era LONG senza)
        if result_no_barrier.signal == SIGNAL_LONG:
            assert result_with_barrier.signal != SIGNAL_LONG

    def test_none_inputs_fallback_to_50(self):
        """Nessun input disponibile → score neutro 50."""
        result = self.model.compute(SignalInputs())
        assert result.score == 50.0
        assert result.signal == SIGNAL_CAUTION

    def test_components_keys(self):
        """Il dict components ha le 7 chiavi attese."""
        result = self.model.compute(SignalInputs(gex_usd=1e8))
        assert set(result.components.keys()) == {"gex", "etf_flow", "funding_rate",
                                                  "oi_change", "long_short", "put_call",
                                                  "liquidations"}

    def test_partial_inputs_rescale(self):
        """Con solo 2 fattori il risultato è coerente (pesi riscalati)."""
        result = self.model.compute(SignalInputs(
            gex_usd=1e9,
            etf_flow_3d_usd=800e6,
        ))
        # Entrambi bullish → score > 50
        assert result.score > 50

    def test_reason_contains_signal(self):
        result = self.model.compute(SignalInputs(gex_usd=1e8))
        assert result.signal in result.reason

    def test_weights_sum_to_1(self):
        result = self.model.compute(SignalInputs(
            gex_usd=1e8, etf_flow_3d_usd=200e6,
            funding_rate_annualized_pct=20.0,
        ))
        total = sum(result.weights_used.values())
        assert abs(total - 1.0) < 1e-9


# ─── SignalModel.compute_series (backtest) ───────────────────────────────────

class TestSignalModelSeries:
    def setup_method(self):
        self.model = SignalModel()

    def _make_df(self, n: int = 10) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "_gex":         [1e8] * n,
            "ibit_flow_3d": [200e6] * n,
            "funding_rate": [0.01] * n,       # 1% 8h → ann ~10%
        }, index=idx)

    def test_returns_series(self):
        df = self._make_df()
        scores = self.model.compute_series(df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(df)

    def test_all_scores_in_range(self):
        df = self._make_df()
        scores = self.model.compute_series(df)
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_signals_from_scores(self):
        scores = pd.Series([70, 50, 30, 65, 39])
        signals = self.model.signals_from_scores(scores)
        assert signals.iloc[0] == 1.0   # 70 >= 65 → LONG
        assert signals.iloc[1] == 0.0   # 50 → FLAT
        assert signals.iloc[2] == -1.0  # 30 < 40 → RISK_OFF
        assert signals.iloc[3] == 1.0   # exactly 65 → LONG
        assert signals.iloc[4] == -1.0  # 39 < 40 → RISK_OFF

    def test_empty_df_returns_empty(self):
        scores = self.model.compute_series(pd.DataFrame())
        assert scores.empty or len(scores) == 0
