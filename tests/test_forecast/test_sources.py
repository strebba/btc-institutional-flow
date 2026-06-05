"""Test degli adapter source EMA e portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast.models import DIR_DOWN, DIR_UP, TARGET_DIRECTION
from src.forecast.sources.ema import build_ema_predictions, compute_ema_state
from src.forecast.sources.portfolio import build_portfolio_predictions


def _trend(n, start, step):
    return pd.Series([start + i * step for i in range(n)], dtype=float)


# ─── EMA ─────────────────────────────────────────────────────────────────────

def test_ema_bull_uptrend():
    close = _trend(250, 100, 1)  # trend rialzista → EMA fast > slow
    preds = build_ema_predictions(close, slow=200)
    assert len(preds) == 1
    p = preds[0]
    assert p.target_type == TARGET_DIRECTION
    assert p.target_spec["direction"] == DIR_UP
    assert 0.0 <= p.confidence <= 1.0


def test_ema_bear_downtrend():
    close = _trend(250, 350, -1)  # trend ribassista
    preds = build_ema_predictions(close, slow=200)
    assert preds[0].target_spec["direction"] == DIR_DOWN


def test_ema_state_and_short_series():
    close = _trend(250, 100, 1)
    st = compute_ema_state(close, 50, 200)
    assert st["regime"] == "bull" and st["spread"] > 0
    with pytest.raises(ValueError):
        build_ema_predictions(_trend(50, 100, 1), slow=200)


# ─── Portfolio ───────────────────────────────────────────────────────────────

def test_portfolio_drift_directions():
    holdings = [
        {"asset": "BTC", "current_weight": 0.45, "target_weight": 0.30, "price": 100000},  # overweight
        {"asset": "ETH", "current_weight": 0.05, "target_weight": 0.15},                    # underweight
        {"asset": "SOL", "current_weight": 0.21, "target_weight": 0.20},                    # entro soglia
    ]
    preds = build_portfolio_predictions(holdings, drift_threshold=0.05)
    by_asset = {p.asset: p for p in preds}
    assert set(by_asset) == {"BTC", "ETH"}  # SOL sotto soglia → escluso
    assert by_asset["BTC"].target_spec["direction"] == DIR_DOWN  # overweight → trim
    assert by_asset["ETH"].target_spec["direction"] == DIR_UP    # underweight → add
    assert by_asset["BTC"].target_spec["ref_price"] == 100000
    assert "ref_price" not in by_asset["ETH"].target_spec        # niente prezzo fornito


def test_portfolio_confidence_scales_with_drift():
    small = build_portfolio_predictions(
        [{"asset": "BTC", "current_weight": 0.36, "target_weight": 0.30}], drift_threshold=0.05)
    big = build_portfolio_predictions(
        [{"asset": "BTC", "current_weight": 0.50, "target_weight": 0.30}], drift_threshold=0.05)
    assert big[0].confidence > small[0].confidence
