"""Test del forecast spine: models, prediction_db, dealer_flow source, verifier."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.analytics.signal_model import (
    SIGNAL_LONG,
    SIGNAL_RISK_OFF,
    SignalResult,
)
from src.forecast.models import (
    DIR_UP,
    Outcome,
    Prediction,
    STATUS_OPEN,
    STATUS_SCORED,
    TARGET_DIRECTION,
    TARGET_LEVEL,
    TARGET_PROB,
)
from src.forecast.prediction_db import PredictionDB
from src.forecast.sources.dealer_flow import build_dealer_flow_predictions
from src.forecast.verifier import score_due_predictions, score_prediction


def _signal(score: float, label: str) -> SignalResult:
    return SignalResult(
        score=score, signal=label,
        components={"gex": 0.8, "etf_flow": 0.6},
        weights_used={"gex": 0.5, "etf_flow": 0.5},
        reason="test",
    )


def _ts(days_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ─── models ──────────────────────────────────────────────────────────────────

def test_prediction_roundtrip_and_clamp():
    p = Prediction(
        source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 100000.0},
        horizon_days=5, confidence=1.7,  # verrà clampata a 1.0
    )
    assert p.confidence == 1.0
    row = p.to_row()
    p2 = Prediction.from_row({**row, "id": 1})
    assert p2.target_spec["direction"] == "up"
    assert p2.horizon_days == 5


def test_prediction_invalid_target_type():
    with pytest.raises(ValueError):
        Prediction(source="x", asset="BTC", target_type="bogus",
                   target_spec={}, horizon_days=1, confidence=0.5)


def test_is_due():
    mature = Prediction(source="s", asset="BTC", target_type=TARGET_DIRECTION,
                        target_spec={"direction": "up", "ref_price": 1.0},
                        horizon_days=5, confidence=0.5, created_at=_ts(10))
    fresh = Prediction(source="s", asset="BTC", target_type=TARGET_DIRECTION,
                       target_spec={"direction": "up", "ref_price": 1.0},
                       horizon_days=5, confidence=0.5, created_at=_ts(1))
    assert mature.is_due() is True
    assert fresh.is_due() is False


# ─── prediction_db ───────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    return PredictionDB(db_path=tmp_path / "forecast_test.db")


def test_insert_and_dedup(db):
    p = Prediction(source="dealer_flow", asset="BTC", target_type=TARGET_PROB,
                   target_spec={"event": "btc_return_positive", "p": 0.7},
                   horizon_days=5, confidence=0.4, created_at=_ts(0))
    pid = db.insert_prediction(p)
    assert pid is not None
    # stessa chiave naturale → dedup
    assert db.insert_prediction(p) is None
    assert db.count() == 1


def test_dedup_per_day_different_timestamp(db):
    """Stesso giorno + source/target/horizon ma timestamp diverso → deduplicato."""
    p1 = Prediction(source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
                    target_spec={"direction": "down", "ref_price": 100.0},
                    horizon_days=5, confidence=0.5, created_at="2026-06-06T07:30:00")
    p2 = Prediction(source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
                    target_spec={"direction": "down", "ref_price": 101.0},
                    horizon_days=5, confidence=0.5, created_at="2026-06-06T10:38:00")  # stesso giorno
    assert db.insert_prediction(p1) is not None
    assert db.insert_prediction(p2) is None       # bloccato dal dedup per-giorno
    assert db.count() == 1


def test_outcome_marks_scored(db):
    p = Prediction(source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
                   target_spec={"direction": "up", "ref_price": 100.0},
                   horizon_days=5, confidence=0.5, created_at=_ts(10))
    pid = db.insert_prediction(p)
    assert db.get_open() and db.get_open()[0].status == STATUS_OPEN
    db.insert_outcome(Outcome(prediction_id=pid, hit=True, realized_return=0.05))
    open_after = db.get_open()
    assert open_after == []
    recent = db.get_recent()
    assert recent[0].status == STATUS_SCORED


def test_weight_versions(db):
    v1 = db.insert_weight_version("dealer_flow", {"gex": 0.5}, rationale="base", activate=True)
    active = db.get_active_weights("dealer_flow")
    assert active is not None and active[0] == v1 and active[1]["gex"] == 0.5
    v2 = db.insert_weight_version("dealer_flow", {"gex": 0.6}, activate=True)
    assert db.get_active_weights("dealer_flow")[0] == v2  # v2 ora attiva


# ─── dealer_flow source ──────────────────────────────────────────────────────

def test_dealer_flow_positive_gamma_targets_max_pain():
    res = _signal(80.0, SIGNAL_LONG)
    preds = build_dealer_flow_predictions(
        res, spot_price=100000.0, max_pain=98000.0, gamma_flip=95000.0,
        total_net_gex=5e8, horizon_days=5,
    )
    types = {p.target_type for p in preds}
    assert types == {TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB}
    level = next(p for p in preds if p.target_type == TARGET_LEVEL)
    assert level.target_spec["level_name"] == "max_pain"
    assert level.target_spec["mode"] == "reach"
    direction = next(p for p in preds if p.target_type == TARGET_DIRECTION)
    assert direction.target_spec["direction"] == DIR_UP
    prob = next(p for p in preds if p.target_type == TARGET_PROB)
    assert prob.target_spec["p"] > 0.5  # score 80 → P(up) > 0.5


def test_dealer_flow_negative_gamma_targets_flip():
    res = _signal(25.0, SIGNAL_RISK_OFF)
    preds = build_dealer_flow_predictions(
        res, spot_price=100000.0, max_pain=98000.0, gamma_flip=99000.0,
        total_net_gex=-3e8, horizon_days=5,
    )
    level = next(p for p in preds if p.target_type == TARGET_LEVEL)
    assert level.target_spec["level_name"] == "gamma_flip"
    assert level.target_spec["mode"] == "break"
    prob = next(p for p in preds if p.target_type == TARGET_PROB)
    assert prob.target_spec["p"] < 0.5  # score 25 → P(up) < 0.5


# ─── verifier ────────────────────────────────────────────────────────────────

def _prices(start: datetime, days: int, closes: list[float], highs=None, lows=None):
    idx = pd.to_datetime([(start + timedelta(days=i + 1)).date() for i in range(days)])
    return pd.DataFrame({
        "close": closes,
        "high": highs or closes,
        "low": lows or closes,
    }, index=idx)


def test_verifier_direction_hit():
    created = datetime.now(timezone.utc) - timedelta(days=6)
    p = Prediction(id=1, source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
                   target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
                   horizon_days=5, confidence=0.6, created_at=created.strftime("%Y-%m-%dT%H:%M:%S"))
    prices = _prices(created, 5, [101, 102, 103, 104, 106])  # +6% → up
    out = score_prediction(p, prices)
    assert out is not None and out.hit is True
    assert out.realized_return == pytest.approx(0.06, abs=1e-6)


def test_verifier_level_reach():
    created = datetime.now(timezone.utc) - timedelta(days=6)
    p = Prediction(id=2, source="dealer_flow", asset="BTC", target_type=TARGET_LEVEL,
                   target_spec={"level_name": "max_pain", "level_price": 98.0,
                                "mode": "reach", "side": "below", "ref_price": 100.0},
                   horizon_days=5, confidence=0.6, created_at=created.strftime("%Y-%m-%dT%H:%M:%S"))
    # il low tocca 97 → livello 98 raggiunto
    prices = _prices(created, 5, [100, 99, 98.5, 99, 99],
                     highs=[100, 100, 99, 99, 99], lows=[99, 98, 97, 98, 98])
    out = score_prediction(p, prices)
    assert out is not None and out.hit is True


def test_verifier_prob_brier():
    created = datetime.now(timezone.utc) - timedelta(days=6)
    p = Prediction(id=3, source="dealer_flow", asset="BTC", target_type=TARGET_PROB,
                   target_spec={"event": "btc_return_positive", "p": 0.8},
                   horizon_days=5, confidence=0.6, created_at=created.strftime("%Y-%m-%dT%H:%M:%S"))
    prices = _prices(created, 5, [101, 102, 103, 104, 105])  # ret > 0 → evento True
    out = score_prediction(p, prices)
    assert out is not None and out.hit is True
    assert out.brier == pytest.approx((0.8 - 1.0) ** 2, abs=1e-9)


def test_verifier_insufficient_data_returns_none():
    created = datetime.now(timezone.utc) - timedelta(days=6)
    p = Prediction(id=4, source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
                   target_spec={"direction": "up", "ref_price": 100.0},
                   horizon_days=5, confidence=0.5, created_at=created.strftime("%Y-%m-%dT%H:%M:%S"))
    assert score_prediction(p, pd.DataFrame()) is None


def test_score_due_end_to_end(tmp_path):
    db = PredictionDB(db_path=tmp_path / "e2e.db")
    created = datetime.now(timezone.utc) - timedelta(days=6)
    cts = created.strftime("%Y-%m-%dT%H:%M:%S")
    pid = db.insert_prediction(Prediction(
        source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.6, created_at=cts,
    ))
    assert pid is not None

    def provider(asset, start, end):
        return _prices(created, 5, [101, 102, 103, 104, 106])

    outcomes = score_due_predictions(db, provider)
    assert len(outcomes) == 1 and outcomes[0].hit is True
    assert db.get_open() == []  # ora scored
