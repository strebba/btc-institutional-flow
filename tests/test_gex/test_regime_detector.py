"""Test unitari per RegimeDetector."""
from __future__ import annotations

from datetime import datetime

import pytest
from src.gex.models import GexByStrike, GexSnapshot
from src.gex.regime_detector import RegimeDetector


def _make_snapshot(
    gex: float,
    spot: float = 70_000,
    put_wall: float = 65_000,
    call_wall: float = 75_000,
    flip: float = 70_000,
) -> GexSnapshot:
    return GexSnapshot(
        timestamp=datetime.utcnow(),
        spot_price=spot,
        total_net_gex=gex,
        gamma_flip_price=flip,
        put_wall=put_wall,
        call_wall=call_wall,
        max_pain=70_000.0,
    )


@pytest.fixture
def detector() -> RegimeDetector:
    return RegimeDetector()


class TestRegimeClassification:
    def test_positive_gamma(self, detector):
        snap  = _make_snapshot(gex=50_000_000)
        state = detector.detect(snap)
        assert state.regime == "positive_gamma"

    def test_negative_gamma(self, detector):
        snap  = _make_snapshot(gex=-50_000_000)
        state = detector.detect(snap)
        assert state.regime == "negative_gamma"

    def test_neutral(self, detector):
        snap  = _make_snapshot(gex=500_000)  # sotto la threshold di 1M
        state = detector.detect(snap)
        assert state.regime == "neutral"


class TestAlerts:
    def test_gamma_flip_alert(self, detector):
        # Prima snapshot positiva, poi negativa → flip alert
        snap1 = _make_snapshot(gex=10_000_000)
        detector.detect(snap1)
        snap2 = _make_snapshot(gex=-10_000_000)
        state = detector.detect(snap2)
        assert any("GAMMA FLIP" in a for a in state.alerts)

    def test_no_flip_same_sign(self, detector):
        snap1 = _make_snapshot(gex=10_000_000)
        detector.detect(snap1)
        snap2 = _make_snapshot(gex=5_000_000)
        state = detector.detect(snap2)
        assert not any("GAMMA FLIP" in a for a in state.alerts)

    def test_near_put_wall(self, detector):
        # Spot a 1% dal put_wall → alert
        snap  = _make_snapshot(gex=5_000_000, spot=65_500, put_wall=65_000)
        state = detector.detect(snap)
        assert any("PUT_WALL" in a for a in state.alerts)

    def test_not_near_put_wall(self, detector):
        # Spot a 20% dal put_wall → nessun alert
        snap  = _make_snapshot(gex=5_000_000, spot=80_000, put_wall=60_000)
        state = detector.detect(snap)
        assert not any("PUT_WALL" in a for a in state.alerts)

    def test_near_call_wall(self, detector):
        # Spot a 1% dal call_wall → alert
        snap  = _make_snapshot(gex=5_000_000, spot=74_500, call_wall=75_000)
        state = detector.detect(snap)
        assert any("CALL_WALL" in a for a in state.alerts)

    def test_extreme_gex_percentile(self, detector):
        # Aggiungo storico alto, poi un GEX bassissimo → 10° percentile
        for _ in range(20):
            detector.add_snapshot(_make_snapshot(gex=50_000_000))
        snap  = _make_snapshot(gex=-200_000_000)
        state = detector.detect(snap)
        assert any("ESTREMO" in a for a in state.alerts)


class TestSummary:
    def test_summary_string(self, detector):
        snap  = _make_snapshot(gex=30_000_000)
        state = detector.detect(snap)
        s     = detector.summary(state)
        assert "POSITIVE_GAMMA" in s
        assert "$" in s
