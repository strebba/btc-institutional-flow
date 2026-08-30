"""Payload minimi ma realistici, modellati sulle risposte vere degli endpoint."""
from __future__ import annotations

import pytest


@pytest.fixture
def gex_payload() -> dict:
    """GEX totale positivo ma negativo sotto lo spot — il caso contraddittorio."""
    return {
        "snapshot": {
            "spot_price": 77_703.9,
            "total_net_gex": 171_930_539.8,
            "gamma_flip_price": 79_783.46,
            "put_wall": 75_000.0,
            "call_wall": 82_000.0,
            "max_pain": 72_000.0,
        },
        "regime": {"label": "positive_gamma", "gex_percentile": 76.47},
        "options_metrics": {"put_call_ratio": 0.4535},
        "strike_profile": [
            {"strike": 75_000.0, "net_gex_m": -11.11, "call_oi": 7089.2, "put_oi": 9166.8},
            {"strike": 77_000.0, "net_gex_m": -10.57, "call_oi": 506.6, "put_oi": 1762.5},
            {"strike": 80_000.0, "net_gex_m": 28.32, "call_oi": 17128.0, "put_oi": 3475.0},
            {"strike": 82_000.0, "net_gex_m": 41.97, "call_oi": 14999.0, "put_oi": 499.0},
        ],
    }


@pytest.fixture
def barriers_payload() -> dict:
    return {
        "count": 3,
        "spot_price": 77_722.68,
        "meta": {"total_active": 293, "priced": 253},
        "barriers": [
            {"barrier_type": "knock_in", "level_price_btc": 76_897.43,
             "issuer": "JPMorgan", "notional_usd": None},
            {"barrier_type": "knock_in", "level_price_btc": 76_651.0,
             "issuer": "Morgan Stanley", "notional_usd": None},
            {"barrier_type": "autocall", "level_price_btc": 95_000.0,
             "issuer": "Goldman Sachs", "notional_usd": None},
        ],
    }


@pytest.fixture
def flows_payload() -> dict:
    return {
        "summary": {
            "ibit": {"net_flow_usd_b": 63.3647, "days_with_data": 660},
            "full_period_corr_ibit_btc_next1d": 0.1575,
            "by_ticker": {
                "GBTC": {"net_flow_usd_b": -27.605},
                "FBTC": {"net_flow_usd_b": 10.247},
            },
        }
    }


@pytest.fixture
def signals_payload() -> dict:
    """Riproduce il caso produzione: macro con 4 fattori su 5 mancanti."""
    return {
        "signal": "CAUTION",
        "score": 49.4,
        "inputs": {"ibit_flow_3d_usd_m": 445.0},
        "pillars": [
            {"name": "gex", "score": 56.6,
             "components": {"regime": 0.65, "flip": 0.37}},
            {"name": "barrier", "score": 45.8,
             "components": {"notional_weighted_distance": 0.34}},
            {"name": "etf_flows", "score": 69.4,
             "components": {"flow_momentum": 0.75, "flow_3d": 0.72}},
            {"name": "macro", "score": 15.0,
             "components": {"funding": None, "oi_change": None, "long_short": None,
                            "put_call": 0.15, "liquidations": None}},
        ],
    }
