"""Catch-up all'avvio del forecast scheduler.

Lo scheduler APScheduler è in-memory: se il processo parte dopo lo slot del
giorno, il cron non recupera lo slot già passato (misfire_grace_time vale solo
per un next_run_time già calcolato). Questi test fissano l'orologio via ``now``
e non toccano la rete.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.analytics.factor_scorers import SignalInputs, SignalResult
from src.api.scheduler import (
    forecast_catchup_enabled,
    run_forecast_startup_catchup,
    start_forecast_scheduler,
    stop_all_schedulers,
)
from src.forecast.context import DealerFlowContext
from src.forecast.models import Outcome, Prediction, TARGET_DIRECTION
from src.forecast.prediction_db import PredictionDB
from src.forecast.sources.dealer_flow import SOURCE as DEALER_FLOW
from src.gex.models import GexSnapshot

def _utc_on(day_offset: int, hour: int, minute: int) -> datetime:
    """Orario UTC relativo a oggi. Verify usa il clock reale, quindi 'oggi' deve coincidere."""
    base = datetime.now(timezone.utc).date() + timedelta(days=day_offset)
    return datetime(base.year, base.month, base.day, hour, minute, tzinfo=timezone.utc)


# Pomeriggio UTC, dopo predict 07:30 e verify 08:00.
LATE = _utc_on(0, 16, 10)
BEFORE_PREDICT = _utc_on(0, 7, 29)
AT_PREDICT = _utc_on(0, 7, 30)
TODAY = LATE.strftime("%Y-%m-%d")
YESTERDAY = _utc_on(-1, 7, 30).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _catchup_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORECAST_CATCHUP_ENABLED", raising=False)


@pytest.fixture
def db(tmp_path: Path) -> PredictionDB:
    return PredictionDB(db_path=tmp_path / "predictions.db")


def _context() -> DealerFlowContext:
    snapshot = GexSnapshot(
        timestamp=LATE,
        spot_price=90_000.0,
        total_net_gex=500e6,
        gamma_flip_price=85_000.0,
        put_wall=80_000.0,
        call_wall=95_000.0,
        max_pain=88_000.0,
    )
    inputs = SignalInputs(gex_usd=500e6, etf_flow_3d_usd=120e6, spot_price=90_000.0)
    result = SignalResult(
        score=72.0,
        signal="LONG",
        components={"gex": 1.0, "etf_flows": 0.8},
        weights_used={"gex": 0.4, "etf_flows": 0.3},
        reason="test",
    )
    return DealerFlowContext(
        result=result,
        snapshot=snapshot,
        spot=90_000.0,
        inputs=inputs,
        ibit_flow_3d=120e6,
        near_barrier=False,
    )


def _direction(created_at: str, *, horizon: int = 5) -> Prediction:
    return Prediction(
        source=DEALER_FLOW,
        asset="BTC",
        target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=horizon,
        confidence=0.4,
        created_at=created_at,
    )


@pytest.fixture
def live_predict():
    """Predict reale su DB di test; il confine di rete è mockato."""
    with patch("src.forecast.jobs._governance", return_value={"kill_switch": False}), \
         patch("src.forecast.context.gather_dealer_flow_context", return_value=_context()):
        yield


def _cohort(db: PredictionDB, date: str) -> list[Prediction]:
    return [p for p in db.get_recent(limit=50) if p.created_at[:10] == date]


def test_late_start_without_cohort_creates_one(db: PredictionDB, live_predict, caplog):
    caplog.set_level(logging.INFO, logger="api.scheduler")

    first = run_forecast_startup_catchup(now=LATE, db=db)
    second = run_forecast_startup_catchup(now=LATE, db=db)

    assert first["date"] == TODAY
    assert first["predict"] == "ran"
    assert first["predict_result"]["status"] == "ok"
    assert first["predict_result"]["inserted"] == first["predict_result"]["total"] == 3
    assert first["verify"] == "ran"
    assert first["verify_result"]["verified"] == 0

    cohort = _cohort(db, TODAY)
    assert len(cohort) == 3
    assert {p.target_type for p in cohort} == {"direction", "level", "prob"}
    assert {p.created_at for p in cohort} == {f"{TODAY}T16:10:00"}
    assert all(p.status == "open" for p in cohort)

    assert second["predict"] == "skipped_exists"
    assert second["predict_result"] is None
    assert len(_cohort(db, TODAY)) == 3
    assert f"nessuna cohort per {TODAY}" in caplog.text
    assert f"cohort dealer_flow già presente per {TODAY}" in caplog.text


def test_existing_cohort_skips_without_calling_predict(db: PredictionDB, caplog):
    """Una cohort già scritta (job delle 07:30, o un altro processo) non si rilancia."""
    caplog.set_level(logging.INFO, logger="api.scheduler")
    assert db.insert_prediction(_direction(f"{TODAY}T07:30:00")) is not None

    with patch("src.forecast.jobs.run_daily_predict") as predict:
        result = run_forecast_startup_catchup(now=LATE, db=db)

    predict.assert_not_called()
    assert result["predict"] == "skipped_exists"
    assert len(db.get_recent(limit=20)) == 1
    rows = db.get_recent(limit=20)
    assert rows[0].created_at == f"{TODAY}T07:30:00"
    assert rows[0].status == "open"
    assert "catch-up predict skip" in caplog.text


def test_concurrent_double_trigger_no_duplicate(db: PredictionDB, live_predict):
    """Due catch-up che passano entrambi il controllo non duplicano la cohort.

    È il caso di due processi all'avvio, o del catch-up che si sovrappone al
    cron delle 07:30: entrambi chiamano run_daily_predict, e l'indice
    idx_pred_day_unique ignora il secondo insert.
    """
    barrier = threading.Barrier(2)
    original = db.has_cohort

    def both_observe_empty(date: str, *, source: str, horizon_days: int | None = None) -> bool:
        exists = original(date, source=source, horizon_days=horizon_days)
        barrier.wait(timeout=5)
        return exists

    db.has_cohort = both_observe_empty  # type: ignore[method-assign]
    results: list[dict | None] = [None, None]
    errors: list[BaseException] = []

    def _run(index: int) -> None:
        try:
            results[index] = run_forecast_startup_catchup(now=LATE, db=db)
        except BaseException as exc:  # noqa: BLE001 — il test deve vedere ogni fallimento
            errors.append(exc)

    def _verify_noop(*, db: PredictionDB | None = None) -> dict:
        return {"status": "ok", "due": 0, "verified": 0, "hit": 0, "miss": 0}

    with patch("src.forecast.jobs.run_daily_verify", _verify_noop):
        threads = [threading.Thread(target=_run, args=(i,)) for i in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert errors == []
    assert all(thread.is_alive() is False for thread in threads)
    assert all(item is not None and item["predict"] == "ran" for item in results)
    inserted = sum(item["predict_result"]["inserted"] for item in results if item)
    cohort = _cohort(db, TODAY)
    assert inserted == 3
    assert len(cohort) == 3
    assert len({(p.source, p.target_type, p.horizon_days) for p in cohort}) == 3


def test_before_predict_slot_does_not_create_cohort(db: PredictionDB, caplog):
    caplog.set_level(logging.INFO, logger="api.scheduler")
    with patch("src.forecast.jobs.run_daily_predict") as predict, \
         patch("src.forecast.jobs.run_daily_verify") as verify:
        result = run_forecast_startup_catchup(now=BEFORE_PREDICT, db=db)

    predict.assert_not_called()
    verify.assert_not_called()
    assert result["predict"] == "skipped_before_slot"
    assert result["verify"] == "skipped_before_slot"
    assert db.count() == 0
    assert "prima dello slot 07:30 UTC" in caplog.text
    assert "prima dello slot 08:00 UTC" in caplog.text


def test_predict_slot_runs_verify_waits_for_its_own_slot(db: PredictionDB, live_predict):
    with patch("src.forecast.jobs.run_daily_verify") as verify:
        result = run_forecast_startup_catchup(now=AT_PREDICT, db=db)

    assert result["predict"] == "ran"
    assert result["verify"] == "skipped_before_slot"
    verify.assert_not_called()
    assert len(_cohort(db, TODAY)) == 3


def test_yesterday_cohort_does_not_satisfy_today(db: PredictionDB, live_predict):
    yesterday_id = db.insert_prediction(_direction(f"{YESTERDAY}T07:30:00"))
    assert yesterday_id is not None

    result = run_forecast_startup_catchup(now=LATE, db=db)

    assert result["predict"] == "ran"
    today = _cohort(db, TODAY)
    yesterday = _cohort(db, YESTERDAY)
    assert len(today) == 3
    assert len(yesterday) == 1
    assert yesterday[0].id == yesterday_id
    assert yesterday[0].status == "open"
    assert yesterday[0].created_at == f"{YESTERDAY}T07:30:00"


def test_verify_catchup_scores_only_due_and_leaves_existing_outcome(
    db: PredictionDB, caplog,
):
    """Verify in ritardo usa il verifier esistente: le mature aperte si chiudono,
    gli esiti già scritti e la cohort di oggi (non matura) restano com'erano.
    """
    caplog.set_level(logging.INFO, logger="api.scheduler")
    frozen_at = _utc_on(-40, 7, 30)
    due_at = _utc_on(-10, 7, 30)
    frozen_id = db.insert_prediction(_direction(frozen_at.strftime("%Y-%m-%dT%H:%M:%S")))
    assert frozen_id is not None
    assert db.insert_outcome(Outcome(
        prediction_id=frozen_id,
        hit=True,
        realized_return=0.02,
        realized_price=102.0,
        ref_price=100.0,
        signed_error=0.02,
        detail="frozen-outcome",
        scored_at=(frozen_at + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S"),
    ))
    due_id = db.insert_prediction(_direction(due_at.strftime("%Y-%m-%dT%H:%M:%S")))
    assert due_id is not None
    today_id = db.insert_prediction(_direction(f"{TODAY}T07:30:00"))
    assert today_id is not None
    before = _outcome_row(db, frozen_id)

    # Finestra del verifier: dal giorno dopo created_at fino a created_at + horizon.
    index = pd.to_datetime([
        (due_at.date() + timedelta(days=1)).isoformat(),
        (due_at.date() + timedelta(days=5)).isoformat(),
    ])
    prices = pd.DataFrame(
        {"close": [100.0, 110.0], "high": [101.0, 112.0], "low": [99.0, 109.0]},
        index=index,
    )
    with patch("src.flows.price_fetcher.PriceFetcher.fetch", return_value=prices):
        result = run_forecast_startup_catchup(now=LATE, db=db)

    assert result["predict"] == "skipped_exists"
    assert result["verify"] == "ran"
    assert result["verify_result"]["verified"] == 1
    assert result["verify_result"]["due"] == 1

    assert _outcome_row(db, frozen_id) == before
    due = next(p for p in db.get_recent(limit=20) if p.id == due_id)
    today = next(p for p in db.get_recent(limit=20) if p.id == today_id)
    assert due.status == "scored"
    assert today.status == "open"
    assert _outcome_row(db, today_id) is None
    assert "solo predizioni mature" in caplog.text


def test_custom_schedule_slots_are_honored(db: PredictionDB, live_predict):
    fc = {
        "horizon_days": 5,
        "schedule": {
            "predict_hour_utc": 9,
            "predict_minute_utc": 15,
            "verify_hour_utc": 9,
            "verify_minute_utc": 45,
        },
    }
    early = run_forecast_startup_catchup(
        fc, now=datetime(2026, 11, 5, 9, 14, tzinfo=timezone.utc), db=db,
    )
    assert early["predict"] == "skipped_before_slot"
    assert db.count() == 0

    at_predict = run_forecast_startup_catchup(
        fc, now=datetime(2026, 11, 5, 9, 15, tzinfo=timezone.utc), db=db,
    )
    assert at_predict["predict"] == "ran"
    assert at_predict["verify"] == "skipped_before_slot"
    assert len(_cohort(db, "2026-11-05")) == 3


@pytest.mark.parametrize("value,expected", [
    (None, True),
    ("", True),
    ("true", True),
    ("false", False),
    ("0", False),
    ("off", False),
    ("no", False),
    ("boh", True),
])
def test_catchup_env_toggle(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool):
    if value is None:
        monkeypatch.delenv("FORECAST_CATCHUP_ENABLED", raising=False)
    else:
        monkeypatch.setenv("FORECAST_CATCHUP_ENABLED", value)
    assert forecast_catchup_enabled() is expected


def test_disabled_env_skips_even_when_slot_passed(db: PredictionDB, monkeypatch, caplog):
    monkeypatch.setenv("FORECAST_CATCHUP_ENABLED", "false")
    caplog.set_level(logging.INFO, logger="api.scheduler")
    with patch("src.forecast.jobs.run_daily_predict") as predict, \
         patch("src.forecast.jobs.run_daily_verify") as verify:
        result = run_forecast_startup_catchup(now=LATE, db=db)

    predict.assert_not_called()
    verify.assert_not_called()
    assert result["enabled"] is False
    assert result["predict"] == "skipped_disabled"
    assert result["verify"] == "skipped_disabled"
    assert db.count() == 0
    assert "FORECAST_CATCHUP_ENABLED disabilitato" in caplog.text


def test_scheduler_start_passes_same_config_into_catchup(monkeypatch: pytest.MonkeyPatch):
    seen: dict = {}

    def _fake(fc, *, now=None, db=None):
        seen["fc"] = fc
        return {"predict": "skipped_before_slot"}

    monkeypatch.setattr("src.api.scheduler.run_forecast_startup_catchup", _fake)

    async def _run() -> None:
        try:
            await start_forecast_scheduler()
            for _ in range(40):
                if "fc" in seen:
                    break
                await asyncio.sleep(0.05)
        finally:
            stop_all_schedulers()

    asyncio.run(_run())
    from src.config import get_settings
    assert seen["fc"] == get_settings()["forecast"]


def test_scheduler_disabled_does_not_catch_up(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORECAST_SCHEDULER_ENABLED", "false")
    from src.config import get_settings
    get_settings.cache_clear()
    called: list[int] = []
    monkeypatch.setattr(
        "src.api.scheduler.run_forecast_startup_catchup",
        lambda *args, **kwargs: called.append(1),
    )

    async def _run() -> None:
        try:
            await start_forecast_scheduler()
            await asyncio.sleep(0.1)
        finally:
            stop_all_schedulers()

    asyncio.run(_run())
    assert called == []


def _outcome_row(db: PredictionDB, prediction_id: int) -> dict | None:
    with db._conn() as conn:
        row = conn.execute(
            "SELECT prediction_id, hit, realized_return, realized_price, ref_price, "
            "signed_error, detail, scored_at FROM outcomes WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
    return dict(row) if row else None
