"""Test per WagmiModel."""
import pytest
from src.analytics.wagmi import (
    WagmiInputs, WagmiModel, WagmiResult,
    LABEL_FULL_WAGMI, LABEL_WAGMI, LABEL_MEH, LABEL_DEGEN, LABEL_NGMI,
    _score_gamma_structure, _score_etf_flow, _score_squeeze_setup,
    _score_options_structure,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def full_inputs():
    return WagmiInputs(
        gex_usd=183e6,
        spot_price=77_756,
        gamma_flip_price=78_000,
        etf_flow_3d_usd=286e6,
        long_short_ratio=0.67,
        funding_rate_annualized_pct=68.1,
        oi_change_7d_pct=5.38,
        call_wall_price=80_000,
        put_wall_price=60_000,
        max_pain_price=74_000,
        liquidations_long_24h_usd=13_588_697,
        liquidations_short_24h_usd=6_189_467,
    )


@pytest.fixture
def model():
    return WagmiModel()


# ─── WagmiResult structure ────────────────────────────────────────────────────

def test_compute_returns_result(model, full_inputs):
    r = model.compute(full_inputs)
    assert isinstance(r, WagmiResult)
    assert 0 <= r.score <= 100
    assert r.label in {LABEL_FULL_WAGMI, LABEL_WAGMI, LABEL_MEH, LABEL_DEGEN, LABEL_NGMI}
    assert r.narrative
    assert r.components
    assert r.weights_used


def test_all_components_present(model, full_inputs):
    r = model.compute(full_inputs)
    expected = {"gamma_structure", "etf_flow", "squeeze_setup", "funding_rate",
                "oi_trend", "options_structure", "liquidation_bias"}
    assert set(r.components.keys()) == expected
    for v in r.components.values():
        assert v is not None


def test_weights_sum_to_one(model, full_inputs):
    r = model.compute(full_inputs)
    assert abs(sum(r.weights_used.values()) - 1.0) < 1e-6


def test_key_level_is_gamma_flip(model, full_inputs):
    r = model.compute(full_inputs)
    assert r.key_level == full_inputs.gamma_flip_price
    assert "gamma flip" in (r.key_level_label or "").lower()


# ─── Score thresholds ─────────────────────────────────────────────────────────

def test_live_snapshot_is_meh(model, full_inputs):
    """Snapshot 23 apr 2026 — below gamma flip, overheated funding → MEH."""
    r = model.compute(full_inputs)
    assert r.label == LABEL_MEH
    assert 51 <= r.score <= 64


def test_strong_bull_scenario_is_wagmi(model):
    inputs = WagmiInputs(
        gex_usd=800e6,
        spot_price=82_000,
        gamma_flip_price=78_000,
        etf_flow_3d_usd=600e6,
        long_short_ratio=0.75,
        funding_rate_annualized_pct=25.0,
        oi_change_7d_pct=12.0,
        call_wall_price=85_000,
        put_wall_price=65_000,
        max_pain_price=78_000,
        liquidations_long_24h_usd=5_000_000,
        liquidations_short_24h_usd=80_000_000,
    )
    r = WagmiModel().compute(inputs)
    assert r.score >= 65
    assert r.label in {LABEL_WAGMI, LABEL_FULL_WAGMI}


def test_bear_scenario_is_ngmi_or_degen(model):
    inputs = WagmiInputs(
        gex_usd=-500e6,
        spot_price=62_000,
        gamma_flip_price=78_000,
        etf_flow_3d_usd=-400e6,
        long_short_ratio=2.2,
        funding_rate_annualized_pct=5.0,
        oi_change_7d_pct=-18.0,
        call_wall_price=80_000,
        put_wall_price=55_000,
        max_pain_price=68_000,
        liquidations_long_24h_usd=600_000_000,
        liquidations_short_24h_usd=50_000_000,
    )
    r = WagmiModel().compute(inputs)
    assert r.score <= 50
    assert r.label in {LABEL_NGMI, LABEL_DEGEN}


# ─── Partial inputs (missing data graceful degradation) ───────────────────────

def test_partial_inputs_no_gamma(model):
    inputs = WagmiInputs(
        etf_flow_3d_usd=300e6,
        funding_rate_annualized_pct=20.0,
        long_short_ratio=0.9,
        oi_change_7d_pct=8.0,
    )
    r = model.compute(inputs)
    assert 0 <= r.score <= 100
    assert r.components["gamma_structure"] is None
    assert r.components["options_structure"] is None
    assert abs(sum(r.weights_used.values()) - 1.0) < 1e-6


def test_empty_inputs_returns_50(model):
    r = model.compute(WagmiInputs())
    assert r.score == 50.0
    assert r.label == LABEL_MEH


# ─── Scoring subfunctions ─────────────────────────────────────────────────────

class TestGammaStructure:
    def test_well_above_flip_positive_gex(self):
        s = _score_gamma_structure(500e6, 82_000, 78_000)
        assert s >= 0.80

    def test_just_below_flip_positive_gex(self):
        s = _score_gamma_structure(183e6, 77_756, 78_000)
        assert 0.40 <= s <= 0.60

    def test_negative_gex_below_flip(self):
        s = _score_gamma_structure(-300e6, 65_000, 78_000)
        assert s <= 0.25


class TestEtfFlow:
    def test_strong_inflows(self):
        assert _score_etf_flow(600e6) >= 0.85
    def test_strong_outflows(self):
        assert _score_etf_flow(-400e6) <= 0.20
    def test_neutral(self):
        s = _score_etf_flow(0)
        assert 0.45 <= s <= 0.55


class TestSqueezeSetup:
    def test_ideal_squeeze_setup(self):
        s = _score_squeeze_setup(0.67, 68.1)
        assert s >= 0.80

    def test_crowded_long(self):
        s = _score_squeeze_setup(2.5, 5.0)
        assert s <= 0.25


class TestOptionsStructure:
    def test_above_call_wall(self):
        s = _score_options_structure(81_000, 80_000, 60_000, 74_000)
        assert s >= 0.85

    def test_near_put_wall(self):
        s = _score_options_structure(62_000, 80_000, 60_000, 74_000)
        assert s <= 0.35
