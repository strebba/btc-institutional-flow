"""Test per l'architettura a 4 pilastri (src/analytics/pillars.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.pillars import (
    PILLAR_WEIGHTS,
    CompositeInputs,
    CompositeSignal,
    score_barrier_pillar,
    score_etf_flows_pillar,
    score_gex_pillar,
    score_macro_pillar,
    score_to_signal,
)


# ─── Pesi ─────────────────────────────────────────────────────────────────────

def test_pillar_weights_sum_to_one():
    assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9


def test_score_to_signal_thresholds():
    assert score_to_signal(70) == "LONG"
    assert score_to_signal(50) == "CAUTION"
    assert score_to_signal(30) == "RISK_OFF"


# ─── GEX pillar ───────────────────────────────────────────────────────────────

def test_gex_pillar_positive_higher_than_negative():
    pos = score_gex_pillar(gex_usd=8e8).score
    neg = score_gex_pillar(gex_usd=-8e8).score
    assert pos > neg
    assert 0 <= neg <= 100 and 0 <= pos <= 100


def test_gex_pillar_flip_context():
    # Spot sopra il flip → più supportivo che spot sotto il flip (a parità di GEX)
    above = score_gex_pillar(gex_usd=1e8, gamma_flip_price=90_000, spot_price=100_000).score
    below = score_gex_pillar(gex_usd=1e8, gamma_flip_price=110_000, spot_price=100_000).score
    assert above > below


def test_gex_pillar_no_data():
    assert score_gex_pillar(gex_usd=None).score is None


# ─── Barrier pillar ───────────────────────────────────────────────────────────

def test_barrier_knockin_below_is_bearish():
    """Knock-in appena sotto lo spot → score basso (accelerante ribasso)."""
    barriers = [{"barrier_type": "knock_in", "level_price_btc": 97_000,
                 "notional_usd": 100e6, "issuer": "JPM"}]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    assert ps.score is not None and ps.score < 40
    assert ps.components["dominant_direction"] == "accelerante_ribasso"


def test_barrier_buffer_below_is_bearish():
    """Buffer bucato al ribasso → dealer vende → score basso."""
    barriers = [{"barrier_type": "buffer", "level_price_btc": 95_000,
                 "notional_usd": 100e6, "issuer": "MS"}]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    assert ps.score is not None and ps.score < 40
    assert ps.components["dominant_direction"] == "accelerante_ribasso"


def test_barrier_autocall_above_is_supportive():
    """Autocall sopra lo spot → dealer compra su dip → score sopra neutro."""
    barriers = [{"barrier_type": "autocall", "level_price_btc": 103_000,
                 "notional_usd": 100e6, "issuer": "GS"}]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    assert ps.score is not None and ps.score > 50


def test_barrier_none_far_is_neutral():
    barriers = [{"barrier_type": "knock_in", "level_price_btc": 50_000,
                 "notional_usd": 100e6}]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    assert ps.score == 50.0


def test_barrier_notional_weighting():
    """Una barriera con notional enorme domina la direzione."""
    barriers = [
        {"barrier_type": "knock_in", "level_price_btc": 98_000, "notional_usd": 1_000e6},
        {"barrier_type": "autocall", "level_price_btc": 102_000, "notional_usd": 1e6},
    ]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    # Il knock-in domina → ribassista
    assert ps.score < 45


def test_barrier_spot_zero_no_crash():
    barriers = [{"barrier_type": "knock_in", "level_price_btc": 97_000, "notional_usd": 100e6}]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=0)
    assert ps.score is None  # nessun crash


def test_barrier_empty():
    assert score_barrier_pillar(active_barriers=[], spot_price=100_000).score is None


def test_barrier_knockout_above_is_supportive():
    """Knock-out sopra lo spot → dealer compra → score sopra neutro."""
    barriers = [{"barrier_type": "knock_out", "level_price_btc": 105_000,
                 "notional_usd": 100e6, "issuer": "JPM"}]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    assert ps.score is not None and ps.score > 50


def test_barrier_missing_notional_equal_weight():
    """Senza notional → equal-weight, nessun crash."""
    barriers = [
        {"barrier_type": "knock_in", "level_price_btc": 98_000},
        {"barrier_type": "knock_in", "level_price_btc": 99_000},
    ]
    ps = score_barrier_pillar(active_barriers=barriers, spot_price=100_000)
    assert ps.score is not None and ps.score < 40


# ─── ETF flows pillar ─────────────────────────────────────────────────────────

def test_etf_flows_inflow_higher_than_outflow():
    inflow = score_etf_flows_pillar(etf_flow_3d_usd=800e6).score
    outflow = score_etf_flows_pillar(etf_flow_3d_usd=-800e6).score
    assert inflow > outflow


def test_etf_flows_estimate_compresses_toward_neutral():
    full = score_etf_flows_pillar(etf_flow_3d_usd=900e6, is_estimate=False).score
    est  = score_etf_flows_pillar(etf_flow_3d_usd=900e6, is_estimate=True).score
    assert abs(est - 50) < abs(full - 50)  # stima più vicina al neutro


def test_etf_flows_no_data():
    assert score_etf_flows_pillar().score is None


# ─── Macro pillar ─────────────────────────────────────────────────────────────

def test_macro_pillar_contrarian_funding():
    cool = score_macro_pillar(funding_rate_annualized_pct=-5).score
    hot  = score_macro_pillar(funding_rate_annualized_pct=90).score
    assert cool > hot  # funding negativo (paura) > funding estremo (surriscaldato)


def test_macro_pillar_no_data():
    assert score_macro_pillar().score is None


# ─── CompositeSignal.compute (live) ───────────────────────────────────────────

def test_composite_compute_blends_and_labels():
    inputs = CompositeInputs(
        gex_usd=6e8, gamma_flip_price=95_000, spot_price=100_000,
        etf_flow_3d_usd=600e6,
        funding_rate_annualized_pct=5, long_short_ratio=0.8,
        active_barriers=[{"barrier_type": "knock_in", "level_price_btc": 60_000,
                          "notional_usd": 50e6}],
    )
    res = CompositeSignal().compute(inputs)
    assert 0 <= res.score <= 100
    assert res.signal in ("LONG", "CAUTION", "RISK_OFF")
    assert len(res.pillars) == 4
    # weights_used dei pilastri disponibili sommano a 1
    assert abs(sum(res.weights_used.values()) - 1.0) < 1e-6
    # legacy components a 7 chiavi presenti
    assert set(res.legacy_components) == {
        "gex", "etf_flow", "funding_rate", "oi_change",
        "long_short", "put_call", "liquidations",
    }


def test_composite_rescales_with_missing_pillars():
    inputs = CompositeInputs(gex_usd=5e8, spot_price=100_000)  # solo GEX
    res = CompositeSignal().compute(inputs)
    assert res.weights_used == {"gex": 1.0}
    assert res.score is not None


def test_composite_no_data_returns_caution():
    res = CompositeSignal().compute(CompositeInputs())
    assert res.signal == "CAUTION"
    assert res.score == 50.0


# ─── CompositeSignal.compute_series (backtest) ────────────────────────────────

def _synthetic_df(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "total_net_gex": rng.normal(2e8, 5e8, n),
            "total_flow_usd": rng.normal(50e6, 200e6, n),
            "btc_close": 100_000 + np.cumsum(rng.normal(0, 1000, n)),
            "btc_vol_7d": np.full(n, 0.5),
            "funding_rate": rng.normal(15, 10, n),
            "oi_usd": 30e9 + np.cumsum(rng.normal(0, 1e8, n)),
            "long_short_ratio": rng.normal(1.2, 0.3, n),
            "ibit_flow_3d": rng.normal(100e6, 300e6, n),
            "btc_return": rng.normal(0, 0.02, n),
        },
        index=idx,
    )


def test_compute_series_columns_and_range():
    df = _synthetic_df()
    barriers = [{"barrier_type": "knock_in", "level_price_btc": 95_000, "notional_usd": 100e6}]
    out = CompositeSignal().compute_series(df, active_barriers=barriers)
    for col in ("gex_score", "barrier_score", "etf_flows_score", "macro_score", "composite_score"):
        assert col in out.columns
    comp = out["composite_score"].dropna()
    assert not comp.empty
    assert comp.between(0, 100).all()
    assert out.index.equals(df.index)


def test_compute_series_signals_from_scores():
    df = _synthetic_df()
    cs = CompositeSignal()
    out = cs.compute_series(df)
    sig = cs.signals_from_scores(out["composite_score"])
    assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})
