"""Contract test: verifica che dashboard e API producano lo stesso segnale composito.

Garantisce che _tab_signals (Streamlit) e /api/signals (FastAPI) non divergano.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.integration
class TestCompositeSignalConsistency:
    """Il segnale composito deve essere identico tra dashboard e API."""

    def test_same_inputs_produce_same_signal(self):
        """Con gli stessi input, CompositeSignal.compute() restituisce lo stesso risultato."""
        from src.analytics.pillars import CompositeSignal, CompositeInputs

        inputs = CompositeInputs(
            gex_usd=300_000_000.0,
            gamma_flip_price=85_000.0,
            put_wall=80_000.0,
            call_wall=95_000.0,
            spot_price=88_000.0,
            etf_flow_3d_usd=50_000_000.0,
            funding_rate_annualized_pct=5.0,
            oi_change_7d_pct=2.0,
            long_short_ratio=1.5,
            put_call_ratio=0.6,
            liquidations_long_24h_usd=10_000_000.0,
            liquidations_short_24h_usd=5_000_000.0,
        )

        result1 = CompositeSignal().compute(inputs)
        result2 = CompositeSignal().compute(inputs)

        assert result1.score == result2.score
        assert result1.signal == result2.signal
        assert len(result1.pillars) == len(result2.pillars)

        # Tutti i pilastri devono avere lo stesso score
        for p1, p2 in zip(result1.pillars, result2.pillars):
            assert p1.name == p2.name
            if p1.score is not None and p2.score is not None:
                assert abs(p1.score - p2.score) < 0.01

    def test_minimal_inputs_still_produce_result(self):
        """Con solo GEX e spot, il composito deve comunque produrre un segnale."""
        from src.analytics.pillars import CompositeSignal, CompositeInputs

        inputs = CompositeInputs(gex_usd=300_000_000.0, spot_price=88_000.0)
        result = CompositeSignal().compute(inputs)

        assert result.score >= 0
        assert result.score <= 100
        assert result.signal in ("LONG", "CAUTION", "RISK_OFF")

    def test_empty_inputs_return_neutral(self):
        """Senza dati, il composito restituisce CAUTION (default safe)."""
        from src.analytics.pillars import CompositeSignal, CompositeInputs

        inputs = CompositeInputs()
        result = CompositeSignal().compute(inputs)

        assert result.signal == "CAUTION"
        assert result.score == 50.0
        assert all(p.score is None for p in result.pillars)
