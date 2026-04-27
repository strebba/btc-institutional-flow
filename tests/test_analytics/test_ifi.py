"""Test per IFIModel e IFIDb."""
import math
import pytest
import numpy as np
import pandas as pd

from src.analytics.ifi import (
    IFIModel,
    IFIResult,
    WEIGHTS,
    regime_label,
    _sigmoid,
    _score_flow_momentum,
    _score_flow_trend,
    _score_price_momentum,
    _score_funding,
    _score_oi_momentum,
    _score_ls_squeeze,
    REGIME_ACCUMULATION,
    REGIME_MOMENTUM,
    REGIME_NEUTRAL,
    REGIME_DISTRIBUTION,
    REGIME_OUTFLOW,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, with_coinglass: bool = True) -> pd.DataFrame:
    """DataFrame sintetico per test."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng   = np.random.default_rng(42)

    # ETF flows: base positiva con rumore
    flows = rng.normal(loc=100e6, scale=200e6, size=n)
    flows[50:70] = -300e6   # periodo di outflow

    # BTC price: trend rialzista
    log_rets = rng.normal(0.001, 0.03, n)
    prices   = 40_000 * np.exp(np.cumsum(log_rets))
    vols     = pd.Series(log_rets).rolling(7).std().fillna(0.03).to_numpy() * (252**0.5)

    df = pd.DataFrame({
        "total_flow_usd": flows,
        "btc_close":      prices,
        "btc_vol_7d":     vols,
    }, index=dates)

    if with_coinglass:
        df["funding_rate"]      = rng.uniform(10, 60, n)
        df["oi_usd"]            = rng.uniform(20e9, 50e9, n)
        df["long_short_ratio"]  = rng.uniform(0.6, 1.8, n)

    return df


@pytest.fixture
def full_df():
    return _make_df(200, with_coinglass=True)


@pytest.fixture
def backbone_df():
    return _make_df(200, with_coinglass=False)


@pytest.fixture
def model():
    return IFIModel()


# ─── Weights ──────────────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# ─── Sigmoid ──────────────────────────────────────────────────────────────────

def test_sigmoid_zero():
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6

def test_sigmoid_positive_gt_half():
    assert _sigmoid(2.0) > 0.5

def test_sigmoid_negative_lt_half():
    assert _sigmoid(-2.0) < 0.5

def test_sigmoid_clamp():
    assert 0 < _sigmoid(100) < 1
    assert 0 < _sigmoid(-100) < 1


# ─── Regime labels ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (85,  REGIME_ACCUMULATION),
    (70,  REGIME_ACCUMULATION),
    (65,  REGIME_MOMENTUM),
    (55,  REGIME_MOMENTUM),
    (50,  REGIME_NEUTRAL),
    (45,  REGIME_NEUTRAL),
    (40,  REGIME_DISTRIBUTION),
    (30,  REGIME_DISTRIBUTION),
    (20,  REGIME_OUTFLOW),
    (0,   REGIME_OUTFLOW),
])
def test_regime_label(score, expected):
    assert regime_label(score) == expected


# ─── Factor scoring ───────────────────────────────────────────────────────────

class TestFlowMomentum:
    def test_returns_series_same_length(self, full_df):
        s = _score_flow_momentum(full_df["total_flow_usd"])
        assert len(s) == len(full_df)

    def test_values_in_unit_interval(self, full_df):
        s = _score_flow_momentum(full_df["total_flow_usd"])
        assert s.between(0, 1).all()

    def test_strong_inflow_scores_high(self):
        # Baseline neutro poi spike: z-score deve essere positivo
        flow = pd.Series([0.0] * 150 + [500e6] * 50)
        s    = _score_flow_momentum(flow)
        assert s.iloc[-1] > 0.6

    def test_strong_outflow_scores_low(self):
        # Baseline neutro poi outflow marcato
        flow = pd.Series([0.0] * 150 + [-500e6] * 50)
        s    = _score_flow_momentum(flow)
        assert s.iloc[-1] < 0.4


class TestFlowTrend:
    def test_returns_unit_interval(self, full_df):
        s = _score_flow_trend(full_df["total_flow_usd"])
        assert s.between(0, 1).all()


class TestPriceMomentum:
    def test_returns_unit_interval(self, full_df):
        s = _score_price_momentum(full_df["btc_close"], full_df["btc_vol_7d"])
        assert s.between(0, 1).all()

    def test_bull_trend_above_half(self):
        # Prezzo in forte salita
        prices = pd.Series(np.linspace(40000, 80000, 200))
        vols   = pd.Series([0.40] * 200)
        s = _score_price_momentum(prices, vols)
        assert s.iloc[-1] > 0.5


class TestFunding:
    def test_low_funding_scores_high(self):
        fr = pd.Series([5.0] * 100)
        assert _score_funding(fr).iloc[-1] > 0.55

    def test_high_funding_scores_low(self):
        fr = pd.Series([100.0] * 100)
        assert _score_funding(fr).iloc[-1] < 0.4

    def test_neutral_near_half(self):
        fr = pd.Series([20.0] * 100)
        s  = _score_funding(fr).iloc[-1]
        assert 0.40 <= s <= 0.60


class TestOIMomentum:
    def test_oi_growth_scores_above_half(self):
        oi = pd.Series(np.linspace(20e9, 30e9, 100))
        assert _score_oi_momentum(oi).iloc[-1] > 0.5

    def test_oi_decline_scores_below_half(self):
        oi = pd.Series(np.linspace(30e9, 20e9, 100))
        assert _score_oi_momentum(oi).iloc[-1] < 0.5


class TestLSSqueeze:
    def test_low_ls_high_squeeze(self):
        ls = pd.Series([0.6] * 100)
        assert _score_ls_squeeze(ls).iloc[-1] > 0.6

    def test_high_ls_low_squeeze(self):
        ls = pd.Series([2.0] * 100)
        assert _score_ls_squeeze(ls).iloc[-1] < 0.4


# ─── IFIModel.compute_series ─────────────────────────────────────────────────

class TestComputeSeries:
    def test_returns_series(self, model, full_df):
        s = model.compute_series(full_df)
        assert isinstance(s, pd.Series)

    def test_same_length_as_input(self, model, full_df):
        s = model.compute_series(full_df)
        assert len(s) == len(full_df)

    def test_values_in_0_100(self, model, full_df):
        s = model.compute_series(full_df)
        assert (s >= 0).all() and (s <= 100).all()

    def test_no_nans(self, model, full_df):
        s = model.compute_series(full_df)
        assert not s.isna().any()

    def test_backbone_only_still_works(self, model, backbone_df):
        s = model.compute_series(backbone_df)
        assert len(s) == len(backbone_df)
        assert (s >= 0).all() and (s <= 100).all()

    def test_empty_df_returns_empty(self, model):
        s = model.compute_series(pd.DataFrame())
        assert s.empty

    def test_outflow_period_scores_lower(self, model):
        """Il periodo di outflow (righe 50-70) deve avere IFI più basso della media."""
        df = _make_df(200, with_coinglass=False)
        s  = model.compute_series(df)
        # Usa righe 60-69 (outflow peak) vs 100-149 (recovery)
        outflow_mean = s.iloc[60:70].mean()
        recovery_mean = s.iloc[120:150].mean()
        assert outflow_mean < recovery_mean

    def test_weight_rescaling_for_partial_data(self, model):
        """Con solo flows + prezzo, i pesi vengono riscalati a sommare 1."""
        s = model.compute_series(_make_df(200, with_coinglass=False))
        # Se il rescaling funziona, i valori devono ancora essere in 0-100
        assert (s >= 0).all() and (s <= 100).all()


# ─── IFIModel.compute_latest ──────────────────────────────────────────────────

class TestComputeLatest:
    def test_returns_ifi_result(self, model, full_df):
        r = model.compute_latest(full_df)
        assert isinstance(r, IFIResult)

    def test_score_in_range(self, model, full_df):
        r = model.compute_latest(full_df)
        assert 0 <= r.score <= 100

    def test_regime_is_valid(self, model, full_df):
        r = model.compute_latest(full_df)
        assert r.regime in (
            REGIME_ACCUMULATION, REGIME_MOMENTUM, REGIME_NEUTRAL,
            REGIME_DISTRIBUTION, REGIME_OUTFLOW,
        )

    def test_weights_sum_to_one(self, model, full_df):
        r = model.compute_latest(full_df)
        assert abs(sum(r.weights_used.values()) - 1.0) < 1e-6

    def test_components_match_available_factors(self, model, backbone_df):
        r = model.compute_latest(backbone_df)
        # CoinGlass factors devono essere None per backbone_df
        assert r.components.get("funding")     is None
        assert r.components.get("oi_momentum") is None
        assert r.components.get("ls_squeeze")  is None
        # Flow e price devono essere presenti
        assert r.components.get("flow_momentum") is not None
        assert r.components.get("price_momentum") is not None
