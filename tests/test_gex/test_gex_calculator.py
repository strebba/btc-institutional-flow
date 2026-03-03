"""Test unitari per GexCalculator."""
from __future__ import annotations

from datetime import datetime

import pytest
from src.gex.gex_calculator import GexCalculator
from src.gex.models import GexSnapshot


def _make_option(
    strike: float,
    option_type: str,
    gamma: float = 0.0001,
    open_interest: float = 100.0,
) -> dict:
    return {
        "instrument_name": f"BTC-TEST-{strike:.0f}-{option_type[0].upper()}",
        "strike":          strike,
        "option_type":     option_type,
        "gamma":           gamma,
        "open_interest":   open_interest,
        "mark_price":      0.05,
        "mark_iv":         0.8,
        "delta":           0.5 if option_type == "call" else -0.5,
        "underlying_price": 70_000.0,
    }


@pytest.fixture
def calc() -> GexCalculator:
    return GexCalculator()


@pytest.fixture
def simple_chain() -> list[dict]:
    """Chain semplice: 3 strike, call e put."""
    return [
        _make_option(60_000, "call",  gamma=0.00005, open_interest=500),
        _make_option(60_000, "put",   gamma=0.00005, open_interest=300),
        _make_option(70_000, "call",  gamma=0.00010, open_interest=800),
        _make_option(70_000, "put",   gamma=0.00008, open_interest=600),
        _make_option(80_000, "call",  gamma=0.00003, open_interest=400),
        _make_option(80_000, "put",   gamma=0.00006, open_interest=700),
    ]


class TestOptionGex:
    def test_call_positive(self, calc):
        gex = calc._option_gex(gamma=0.0001, open_interest=100, option_type="call", spot_price=70_000)
        assert gex > 0

    def test_put_negative(self, calc):
        gex = calc._option_gex(gamma=0.0001, open_interest=100, option_type="put", spot_price=70_000)
        assert gex < 0

    def test_formula(self, calc):
        # GEX = gamma × OI × contract_size × spot² × 0.01 × sign
        expected = 0.0001 * 100 * 1.0 * (70_000 ** 2) * 0.01  # call → +
        gex = calc._option_gex(0.0001, 100, "call", 70_000)
        assert abs(gex - expected) < 1.0

    def test_magnitude_millions(self, calc):
        # Con valori realistici, GEX deve essere nell'ordine dei milioni
        gex = calc._option_gex(0.0001, 1000, "call", 70_000)
        assert abs(gex) > 1_000_000  # almeno 1M


class TestCalculateGex:
    def test_returns_snapshot(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        assert isinstance(snap, GexSnapshot)

    def test_empty_chain(self, calc):
        snap = calc.calculate_gex([], spot_price=70_000)
        assert snap.total_net_gex == 0.0
        assert snap.put_wall is None

    def test_strike_aggregation(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        # 3 strike distinti → 3 GexByStrike
        strikes = {g.strike for g in snap.gex_by_strike}
        assert len(strikes) == 3

    def test_total_gex_sign(self, calc):
        # Chain dominata da call → GEX totale positivo
        chain = [_make_option(70_000, "call", gamma=0.001, open_interest=10_000)]
        snap  = calc.calculate_gex(chain, spot_price=70_000)
        assert snap.total_net_gex > 0

    def test_put_wall_exists(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        # Almeno una barriera deve essere rilevata
        assert snap.put_wall is not None or snap.call_wall is not None

    def test_max_pain_in_strikes(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        if snap.max_pain:
            strikes = {o["strike"] for o in simple_chain}
            assert snap.max_pain in strikes

    def test_oi_totals(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        assert snap.total_call_oi > 0
        assert snap.total_put_oi > 0

    def test_put_call_ratio(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        assert snap.put_call_ratio is not None
        assert snap.put_call_ratio > 0


class TestGammaFlip:
    def test_flip_detected(self, calc):
        # Build chain dove il GEX cambia segno intorno a 70000
        chain = [
            _make_option(60_000, "put",  gamma=0.001, open_interest=5_000),  # −GEX dominante
            _make_option(70_000, "call", gamma=0.001, open_interest=3_000),  # +GEX
            _make_option(80_000, "call", gamma=0.001, open_interest=5_000),  # +GEX dominante
        ]
        snap = calc.calculate_gex(chain, spot_price=70_000)
        # Deve esistere un gamma flip
        assert snap.gamma_flip_price is not None

    def test_no_flip_all_same_sign(self, calc):
        # Tutte call → GEX sempre positivo → nessun flip
        chain = [_make_option(s, "call", gamma=0.001, open_interest=1_000)
                 for s in [60_000, 70_000, 80_000]]
        snap  = calc.calculate_gex(chain, spot_price=70_000)
        assert snap.gamma_flip_price is None


class TestGexToDict:
    def test_serializable(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        d    = calc.gex_to_dict(snap)
        import json
        # Deve essere serializzabile (nessun tipo numpy/datetime)
        json.dumps(d)

    def test_dict_keys(self, calc, simple_chain):
        snap = calc.calculate_gex(simple_chain, spot_price=70_000)
        d    = calc.gex_to_dict(snap)
        for key in ["spot_price", "total_net_gex_m", "put_wall", "call_wall", "max_pain"]:
            assert key in d
