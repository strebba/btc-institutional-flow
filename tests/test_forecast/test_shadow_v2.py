"""Regole v2, pesi congelati, e il verdetto pre-registrato."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.analytics.factor_scorers import SIGNAL_CAUTION, SIGNAL_LONG
from src.forecast.calibration import run_calibration, weights_are_frozen
from src.forecast.models import TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB
from src.forecast.prediction_db import PredictionDB
from src.forecast.shadow_eval import (
    MIN_COHORTS,
    coverage,
    decide,
    evaluate,
    proportion,
    replay_indicative,
    wilson,
)
from src.forecast.sources.dealer_flow import build_dealer_flow_predictions
from src.forecast.sources.dealer_flow_v2 import (
    ABSTAIN_CONFIDENCE,
    MAX_PAIN_MAX_DISTANCE,
    SOURCE,
    build_dealer_flow_v2_predictions,
    calibrate_prob,
    filter_matured_prob_rows,
    flat_band_pct_from_returns,
)
from tests.test_forecast.test_calibration import _CFG, _seed_outcomes
from tests.test_forecast.test_forecast import _signal


def _v2(**overrides):
    args = dict(
        score=72.0, spot_price=100_000.0, p_up=0.6, flat_band_pct=4.0,
        gamma_flip=95_000.0, max_pain=102_000.0, total_net_gex=1e8,
        created_at="2026-11-10T07:30:00", weights_version=7,
        calibration="shrink_base_rate", calibration_n=0,
    )
    args.update(overrides)
    return build_dealer_flow_v2_predictions(**args)


def test_v1_caution_is_unchanged():
    preds = build_dealer_flow_predictions(
        _signal(50.0, SIGNAL_CAUTION), spot_price=100_000.0,
        max_pain=110_000.0, gamma_flip=90_000.0, total_net_gex=1e8,
    )
    direction = next(p for p in preds if p.target_type == TARGET_DIRECTION)
    level = next(p for p in preds if p.target_type == TARGET_LEVEL)
    assert direction.target_spec["direction"] == "flat"
    assert direction.target_spec["flat_band_pct"] == 1.0
    assert level.target_spec["level_name"] == "max_pain"
    assert all(p.source == "dealer_flow" for p in preds)


def test_v2_sign_abstain_and_drops_far_max_pain():
    abstain = _v2(score=52.0)
    assert TARGET_DIRECTION not in {p.target_type for p in abstain}
    assert _confidence_below(52.0)

    up = _v2(score=60.0)
    direction = next(p for p in up if p.target_type == TARGET_DIRECTION)
    assert direction.target_spec["direction"] == "up"
    assert direction.target_spec["flat_band_pct"] == 4.0
    assert direction.source == SOURCE
    assert direction.model_version == "v2"
    assert direction.weights_version == 7

    down = _v2(score=30.0, total_net_gex=-1e8, max_pain=None)
    assert next(p for p in down if p.target_type == TARGET_DIRECTION).target_spec["direction"] == "down"
    flip = next(p for p in down if p.target_type == TARGET_LEVEL)
    assert flip.target_spec["level_name"] == "gamma_flip"

    far = _v2(score=70.0, spot_price=100_000.0, max_pain=110_000.0, total_net_gex=1e8)
    assert abs(110_000 / 100_000 - 1) > MAX_PAIN_MAX_DISTANCE
    assert TARGET_LEVEL not in {p.target_type for p in far}

    near = _v2(score=70.0, spot_price=100_000.0, max_pain=103_000.0, total_net_gex=1e8)
    level = next(p for p in near if p.target_type == TARGET_LEVEL)
    assert level.target_spec["level_name"] == "max_pain"


def test_flat_band_uses_only_prior_returns_and_falls_back():
    assert flat_band_pct_from_returns([0.01] * 5, 5) == 4.0
    # 20 osservazioni all'1% di deviazione: σ_5d ≈ 0.01*sqrt(5), metà ≈ 1.12%.
    band = flat_band_pct_from_returns([0.01, -0.01] * 10, 5)
    assert band == pytest.approx(100 * 0.5 * (0.01) * 5 ** 0.5, rel=0.05)
    assert band != 1.0


def test_platt_can_flip_an_inversion_and_does_not_see_the_future():
    history = [(80.0, 0), (20.0, 1)] * 20
    high, method = calibrate_prob(80.0, history)
    low, _ = calibrate_prob(20.0, history)
    assert method == "platt"
    assert high < low
    assert high < 0.5

    shrunk, how = calibrate_prob(80.0, history[:5])
    raw = 0.5 + (80.0 - 50.0) / 100.0
    assert how == "shrink_base_rate"
    assert shrunk < raw  # tirata verso il base rate, il ranking non è rifittato

    before = datetime(2026, 11, 10, 7, 30, tzinfo=timezone.utc)
    rows = [
        _prob_row("2026-11-01T07:30:00", score=70.0, ret=0.02),  # matura il 6, prima
        _prob_row("2026-11-08T07:30:00", score=80.0, ret=-0.05),  # matura il 13, dopo
    ]
    used = filter_matured_prob_rows(rows, before)
    assert used == [(70.0, 1)]


def test_both_sources_share_a_day_without_colliding(tmp_path):
    db = PredictionDB(db_path=tmp_path / "preds.db")
    created = "2026-11-10T07:30:00"
    v1 = build_dealer_flow_predictions(
        _signal(72.0, SIGNAL_LONG), spot_price=100_000.0, max_pain=102_000.0,
        total_net_gex=1e8, created_at=created,
    )
    v2 = _v2(created_at=created)
    for pred in v1 + v2:
        assert db.insert_prediction(pred) is not None
    again = db.insert_prediction(v1[0])
    assert again is None
    assert db.has_cohort("2026-11-10", source="dealer_flow", horizon_days=5)
    assert db.has_cohort("2026-11-10", source=SOURCE, horizon_days=5)
    assert db.count() == len(v1) + len(v2)


def test_freeze_is_the_default_and_blocks_proposals(tmp_path):
    assert weights_are_frozen() is True
    assert weights_are_frozen({"governance": {}}) is True
    assert weights_are_frozen({"governance": {"freeze_weights": False}}) is False
    db = PredictionDB(db_path=tmp_path / "frozen.db")
    _seed_outcomes(db, 20)
    rep = run_calibration(db, days=365, cfg=_CFG)
    # _CFG ha freeze_weights false: la proposta c'è, ma non è attiva.
    assert rep.proposed_weights is not None
    assert db.get_active_weights("dealer_flow") is None


def test_decision_waits_for_minima_and_replay_is_labelled():
    assert decide(MIN_COHORTS - 1, {"beats": True}, {"beats": True}, {"beats": True}) == "not_yet"
    assert decide(MIN_COHORTS, {"beats": True}, {"beats": None}, {"beats": True}) == "not_yet"
    assert decide(MIN_COHORTS, {"beats": True}, {"beats": True}, {"beats": False}) == "v2_not_better"
    assert decide(MIN_COHORTS, {"beats": True}, {"beats": True}, {"beats": True}) == "v2_better"

    empty = evaluate([], [], start="2026-11-10")
    assert empty.decision == "not_yet"
    assert empty.n_cohorts == 0

    rate, low, high = wilson(30, 40)
    assert low < rate < high

    rows = [_direction_outcome("2026-10-01", "up", 0.05, hit=1)]
    text = "\n".join(replay_indicative(rows, start="2026-11-10"))
    assert "IN-SAMPLE / INDICATIVE ONLY" in text
    assert "not used for the decision" in text


def test_coverage_counts_abstentions_as_missing_direction():
    rows = [
        {"date": "2026-11-10", "target_type": TARGET_DIRECTION},
        {"date": "2026-11-10", "target_type": TARGET_PROB},
        {"date": "2026-11-11", "target_type": TARGET_PROB},
    ]
    assert coverage(rows) == pytest.approx(0.5)
    block = proportion("x", 8, 10)
    assert block.ci_low < block.rate < block.ci_high


def _confidence_below(score: float) -> bool:
    return abs(score - 50.0) / 50.0 < ABSTAIN_CONFIDENCE


def _prob_row(created: str, *, score: float, ret: float) -> dict:
    return {
        "source": "dealer_flow",
        "target_type": TARGET_PROB,
        "score_ref": score,
        "realized_return": ret,
        "created_at": created,
        "horizon_days": 5,
        "hit": 1,
    }


def _direction_outcome(day: str, direction: str, ret: float, *, hit: int) -> dict:
    created = f"{day}T07:30:00"
    return {
        "source": "dealer_flow",
        "asset": "BTC",
        "target_type": TARGET_DIRECTION,
        "target_spec": json.dumps(
            {"direction": direction, "ref_price": 100.0, "flat_band_pct": 1.0}
        ),
        "horizon_days": 5,
        "confidence": 0.4,
        "created_at": created,
        "date": day,
        "score_ref": 70.0,
        "hit": hit,
        "realized_return": ret,
        "brier": None,
    }
