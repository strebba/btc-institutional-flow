"""Test della calibrazione: statistiche, proposta pesi con guardrail, gate human-in-the-loop."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.analytics.signal_model import WEIGHTS as DEFAULT_WEIGHTS
from src.forecast.calibration import (
    binomial_p_value,
    compute_source_metrics,
    propose_weights,
    run_calibration,
    spearman_ic,
)
from src.forecast.models import Outcome, Prediction, TARGET_DIRECTION
from src.forecast.prediction_db import PredictionDB

_CFG = {
    "priors": dict(DEFAULT_WEIGHTS),
    "calibration": {"min_scored": 10, "min_scored_gex": 10, "max_step": 0.05,
                    "shrinkage": 0.3, "learning_rate": 0.5, "min_oos_improvement": 0.0},
    "governance": {"kill_switch": False, "freeze_weights": True},
}


def test_binomial_monotonic():
    assert binomial_p_value(0, 0) is None
    p_all = binomial_p_value(10, 10)
    p_half = binomial_p_value(5, 10)
    assert 0 < p_all < p_half <= 1.0  # più hit → p più piccolo (più significativo)
    assert binomial_p_value(10, 10) == pytest.approx(0.5 ** 10, abs=1e-9)


def test_spearman_ic_signs():
    assert spearman_ic([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0, abs=1e-6)
    assert spearman_ic([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0, abs=1e-6)
    assert spearman_ic([1, 1, 1], [1, 2, 3]) is None  # varianza nulla


def test_propose_weights_guardrails():
    active = dict(DEFAULT_WEIGHTS)
    # IC fortemente positivo su funding_rate, nullo altrove
    ic = {k: (0.9 if k == "funding_rate" else 0.0) for k in DEFAULT_WEIGHTS}
    n = {k: 50 for k in DEFAULT_WEIGHTS}
    proposed, rationale = propose_weights(active, ic, n, _CFG)
    assert proposed["funding_rate"] > active["funding_rate"]            # tilt nella giusta direzione
    assert proposed["funding_rate"] - active["funding_rate"] <= 0.05 + 1e-9  # cap del passo
    assert sum(proposed.values()) == pytest.approx(1.0, abs=1e-6)        # Σ=1
    assert all(v >= 0 for v in proposed.values())                       # niente pesi negativi


def test_propose_weights_gex_gate():
    active = dict(DEFAULT_WEIGHTS)
    ic = {k: 0.9 for k in DEFAULT_WEIGHTS}
    # gex con campione sotto soglia → non tiltato
    n = {k: (5 if k == "gex" else 50) for k in DEFAULT_WEIGHTS}
    cfg = {**_CFG, "calibration": {**_CFG["calibration"], "min_scored_gex": 40, "min_scored": 10}}
    proposed, _ = propose_weights(active, ic, n, cfg)
    assert sum(proposed.values()) == pytest.approx(1.0, abs=1e-6)


def _seed_outcomes(db: PredictionDB, n: int, *, hit_ratio: float = 0.7):
    """Inserisce n predizioni direction con componenti + esiti (per la calibrazione)."""
    base = datetime.now(timezone.utc) - timedelta(days=30)
    for i in range(n):
        ts = (base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
        # funding_rate alto → return alto (IC positivo costruito)
        fr = 0.9 if i % 2 == 0 else 0.1
        ret = 0.05 if i % 2 == 0 else -0.05
        pid = db.insert_prediction(Prediction(
            source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
            target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
            horizon_days=5, confidence=0.6, created_at=ts,
            components={"funding_rate": fr, "gex": 0.5},
            score_ref=70.0,
        ))
        hit = i < int(n * hit_ratio)
        db.insert_outcome(Outcome(prediction_id=pid, hit=hit, realized_return=ret,
                                  realized_price=100 * (1 + ret), ref_price=100.0,
                                  signed_error=ret))


def test_run_calibration_gate_rejects_low_n(tmp_path):
    db = PredictionDB(db_path=tmp_path / "cal_low.db")
    _seed_outcomes(db, 5)  # < min_scored (10)
    rep = run_calibration(db, days=365, cfg=_CFG)
    assert rep.gate_ok is False
    assert rep.proposed_weights is None
    assert db.get_proposed_weights("dealer_flow") == []  # nessuna proposta salvata


def test_run_calibration_proposes_but_not_active(tmp_path):
    db = PredictionDB(db_path=tmp_path / "cal_ok.db")
    _seed_outcomes(db, 20)
    rep = run_calibration(db, days=365, cfg=_CFG)
    assert rep.gate_ok is True
    assert rep.proposed_weights is not None
    # proposta salvata come 'proposed', NON attiva
    proposed = db.get_proposed_weights("dealer_flow")
    assert len(proposed) == 1
    assert db.get_active_weights("dealer_flow") is None  # nessuna attivazione automatica


def test_metrics_hit_rate(tmp_path):
    db = PredictionDB(db_path=tmp_path / "cal_m.db")
    _seed_outcomes(db, 20, hit_ratio=0.75)
    rows = db.get_with_outcomes(days=365, source="dealer_flow")
    m = compute_source_metrics(rows)
    assert m["by_target_type"][TARGET_DIRECTION]["scored"] == 20
    assert m["by_target_type"][TARGET_DIRECTION]["hit_rate"] == 0.75
    assert m["component_ic"]["funding_rate"] is not None
