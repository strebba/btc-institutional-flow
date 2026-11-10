"""Ricostruzione point-in-time: niente lookahead, niente altre coorti.

Le serie sintetiche contengono di proposito osservazioni successive allo slot
2026-11-03 07:30 UTC. Il valore usato deve essere quello già chiuso prima.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.analytics.factor_scorers import WEIGHTS
from src.forecast.models import Prediction, TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB
from src.forecast.prediction_db import CohortExistsError, PredictionDB
from src.forecast.reconstruct import (
    BarrierRow,
    FlowPrint,
    GexRow,
    LiqBar,
    LocalPaths,
    MarketHistory,
    TimedBar,
    WeightRow,
    assemble,
    cohort_exists,
    gex_rows_from_dicts,
    index_chart_range,
    load_local_snapshots,
    main,
    price_at_slot,
    read_sqlite,
    score_cohort,
    select_gex,
)
from src.forecast.sources.dealer_flow import SOURCE
from src.gex.gex_db import GexDB

SLOT = datetime(2026, 11, 3, 7, 30, tzinfo=timezone.utc)
DAY = 86_400


def _utc(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def _bar(day: date, value: float) -> TimedBar:
    return TimedBar(_utc(day.year, day.month, day.day), value, DAY)


def _history(*, gex_rows: list[GexRow] | None = None) -> MarketHistory:
    """Storico con un print lecito e uno successivo allo slot, per ogni input."""
    morning = GexRow(
        day=date(2026, 11, 3),
        timestamp=_utc(2026, 11, 3, 7, 15),
        total_net_gex=6e8,
        gamma_flip_price=97_000.0,
        call_wall=150_000.0,
        max_pain=99_000.0,
        put_call_ratio=0.8,
    )
    afternoon = GexRow(
        day=date(2026, 11, 3),
        timestamp=_utc(2026, 11, 3, 18, 0),
        total_net_gex=-9e9,
        gamma_flip_price=1.0,
        call_wall=100_000.0,
        max_pain=1.0,
        put_call_ratio=50.0,
    )
    later = GexRow(
        day=date(2026, 11, 4),
        timestamp=_utc(2026, 11, 4, 7, 15),
        total_net_gex=-9e9,
        gamma_flip_price=1.0,
        call_wall=100_000.0,
        max_pain=1.0,
        put_call_ratio=50.0,
    )
    flows = [
        FlowPrint(date(2026, 10, 24), "IBIT", 100.0, "coinglass"),
        FlowPrint(date(2026, 10, 25), "IBIT", 200.0, "coinglass"),
        FlowPrint(date(2026, 10, 26), "IBIT", 300.0, "coinglass"),
        FlowPrint(date(2026, 10, 27), "IBIT", 400.0, "coinglass"),
        FlowPrint(date(2026, 10, 28), "IBIT", 500.0, "coinglass"),
        FlowPrint(date(2026, 10, 29), "IBIT", 600.0, "coinglass"),
        FlowPrint(date(2026, 11, 2), "IBIT", 1e9, "coinglass"),
        FlowPrint(date(2026, 11, 2), "FBTC", 5e9, "coinglass"),
        FlowPrint(date(2026, 11, 3), "IBIT", 9999.0, "coinglass"),
        FlowPrint(date(2026, 11, 4), "IBIT", 8888.0, "coinglass"),
    ]
    oi_days = [date(2026, 10, d) for d in range(25, 32)] + [date(2026, 11, 1), date(2026, 11, 2)]
    return MarketHistory(
        spot_prints=[
            (_utc(2026, 11, 3, 7, 0), 100_000.0),
            (_utc(2026, 11, 3, 8, 0), 50_000.0),
            (_utc(2026, 11, 4, 7, 0), 1.0),
        ],
        flows=flows,
        funding_bars=[
            _bar(date(2026, 11, 2), -0.01),
            _bar(date(2026, 11, 3), 5.0),
        ],
        oi_bars=[_bar(day, 100.0) for day in oi_days] + [_bar(date(2026, 11, 3), 1000.0)],
        long_short_bars=[
            _bar(date(2026, 11, 2), 0.5),
            _bar(date(2026, 11, 3), 9.0),
        ],
        liquidation_bars=[
            LiqBar(_utc(2026, 11, 2), 1.0, 2.0, DAY),
            LiqBar(_utc(2026, 11, 3), 9e9, 9e9, DAY),
        ],
        gex_rows=gex_rows if gex_rows is not None else [morning, afternoon, later],
        weight_rows=[
            WeightRow(1, _utc(2026, 7, 22), dict(WEIGHTS), "retired"),
            WeightRow(3, _utc(2026, 11, 1), {k: 0.0 for k in WEIGHTS}, "proposed"),
            WeightRow(2, _utc(2026, 11, 3, 9, 0), {**WEIGHTS, "etf_flow": 1.0}, "active"),
        ],
        barrier_rows=[
            BarrierRow(date(2026, 11, 2), _utc(2026, 11, 2, 22, 30), 50_000.0),
            BarrierRow(date(2026, 11, 3), _utc(2026, 11, 3, 7, 0), 50_000.0),
            BarrierRow(date(2026, 11, 3), _utc(2026, 11, 3, 22, 30), 100_000.0),
        ],
        notes={"spot": "fixture", "etf": "fixture"},
    )


def _prices(start: datetime, days: int, closes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime([(start + timedelta(days=i + 1)).date() for i in range(days)])
    return pd.DataFrame({"close": closes, "high": closes, "low": closes}, index=idx)


def _dump(db: PredictionDB) -> tuple[list[dict], list[dict]]:
    with db._conn() as conn:
        preds = [dict(r) for r in conn.execute(
            "SELECT * FROM predictions ORDER BY id"
        ).fetchall()]
        outcomes = [dict(r) for r in conn.execute(
            "SELECT * FROM outcomes ORDER BY id"
        ).fetchall()]
    return preds, outcomes


def test_price_at_slot_keeps_the_print_at_the_slot_and_drops_the_next():
    slot = SLOT
    chosen = price_at_slot([
        (slot, 10.0),
        (slot + timedelta(seconds=1), 99.0),
    ], slot)
    assert chosen is not None
    assert chosen[1] == 10.0


def test_index_chart_range_covers_the_slot_without_asking_for_now_only():
    assert index_chart_range(SLOT, _utc(2026, 11, 10, 12)) == "1m"
    assert index_chart_range(SLOT, SLOT + timedelta(hours=2)) == "1d"
    assert index_chart_range(SLOT, SLOT + timedelta(days=40)) == "1y"


def test_assemble_ignores_everything_after_the_slot():
    result = assemble(SLOT, _history())
    assert result.refused is None
    assert result.degraded_inputs == []
    assert result.weights_version == 1
    assert result.signal_label == "LONG"
    assert result.signal_score == pytest.approx(72.0, abs=0.05)

    inputs = result.provenance["inputs"]
    assert inputs["spot"]["price"] == 100_000.0
    assert inputs["spot"]["print_at"] == "2026-11-03T07:00:00Z"
    assert inputs["spot"]["dropped_after_slot"] == 2
    # Calendario D-3..D-1 = 31/10, 1/11, 2/11. Il rolling sui giorni di borsa
    # avrebbe sommato anche 29 e 30 ottobre. Il print del 3/11 non entra.
    assert inputs["etf_window"] == {
        "2026-10-31": 0.0,
        "2026-11-01": 0.0,
        "2026-11-02": 1e9,
    }
    assert inputs["etf_flow_3d_usd"] == 1e9
    assert inputs["etf_dropped_on_or_after_slot"] == 2
    assert inputs["granger_lead_flow_usd"] == 200.0
    assert inputs["granger_lead_date"] == "2026-10-25"
    assert inputs["funding_rate_8h_pct"] == -0.01
    assert inputs["funding_bar_open"] == "2026-11-02T00:00:00Z"
    assert inputs["funding_dropped_unclosed"] == 1
    assert inputs["oi_change_7d_pct"] == 0.0
    assert inputs["oi_dropped_unclosed"] == 1
    assert inputs["long_short_ratio"] == 0.5
    assert inputs["liquidations_long_24h_usd"] == 1.0
    assert inputs["liquidations_short_24h_usd"] == 2.0
    assert inputs["liquidations_dropped_unclosed"] == 1
    assert inputs["gex"]["total_net_gex"] == 6e8
    assert inputs["gex"]["timestamp"] == "2026-11-03T07:15:00Z"
    assert inputs["gex"]["put_call_ratio"] == 0.8
    assert inputs["gex_same_day_after_slot"] == 1
    assert inputs["gex_later_days"] == 1
    assert inputs["near_barrier"] is False
    assert inputs["barrier_snapshot_date"] == "2026-11-03"

    assert {p.target_type for p in result.predictions} == {
        TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB,
    }
    for pred in result.predictions:
        assert pred.source == SOURCE
        assert pred.created_at == "2026-11-03T07:30:00"
        assert pred.weights_version == 1
        assert pred.metadata["reconstructed"] is True
        assert pred.metadata["degraded"] is False
        assert pred.components["gex"] == 1.0
        assert pred.components["put_call"] == pytest.approx(0.32, abs=1e-9)
    level = next(p for p in result.predictions if p.target_type == TARGET_LEVEL)
    assert level.target_spec["level_name"] == "max_pain"
    assert level.target_spec["level_price"] == 99_000.0
    assert level.target_spec["ref_price"] == 100_000.0


def test_missing_gex_snapshot_degrades_instead_of_using_a_later_one():
    afternoon = GexRow(
        day=date(2026, 11, 3),
        timestamp=_utc(2026, 11, 3, 18, 0),
        total_net_gex=-9e9,
        gamma_flip_price=1.0,
        call_wall=100_000.0,
        max_pain=1.0,
        put_call_ratio=50.0,
    )
    result = assemble(SLOT, _history(gex_rows=[afternoon]))
    assert result.refused is None
    assert result.degraded_inputs == ["gex", "put_call", "call_wall", "level"]
    assert result.provenance["inputs"]["gex"] is None
    assert TARGET_LEVEL not in {p.target_type for p in result.predictions}
    for pred in result.predictions:
        assert pred.metadata["reconstructed"] is True
        assert pred.metadata["degraded"] is True
        assert pred.components["gex"] is None
        assert pred.components["put_call"] is None


def test_refuses_when_the_only_spot_or_flows_are_after_the_slot():
    history = _history()
    history.spot_prints = [(_utc(2026, 11, 3, 8, 0), 50_000.0)]
    refused = assemble(SLOT, history)
    assert refused.refused == "spot_unavailable"
    assert refused.predictions == []
    assert refused.provenance["inputs"]["spot"]["dropped_after_slot"] == 1

    history = _history()
    history.flows = [FlowPrint(date(2026, 11, 3), "IBIT", 1e9, "coinglass")]
    refused = assemble(SLOT, history)
    assert refused.refused == "etf_flows_unavailable"
    assert refused.predictions == []


def test_weights_created_after_the_slot_fall_back_to_priors_when_nothing_is_eligible():
    history = _history()
    history.weight_rows = [
        WeightRow(9, _utc(2026, 11, 4), dict(WEIGHTS), "active"),
        WeightRow(8, _utc(2026, 11, 1), dict(WEIGHTS), "proposed"),
    ]
    result = assemble(SLOT, history)
    assert result.weights_version is None
    assert result.provenance["weights"]["source"] == "priors"
    assert result.provenance["weights"]["rejected_after_slot"] == 1
    assert all(p.weights_version is None for p in result.predictions)


def test_select_gex_same_day_before_slot_only():
    morning = GexRow(
        date(2026, 11, 3), _utc(2026, 11, 3, 7, 15),
        6e8, None, None, None, None,
    )
    chosen, counts = select_gex([morning], SLOT)
    assert chosen is morning
    assert counts["same_day_after_slot"] == 0
    chosen, counts = select_gex([], SLOT)
    assert chosen is None


def test_read_only_sqlite_does_not_create_a_file_and_drops_future_gex(tmp_path):
    missing = tmp_path / "absent.db"
    assert read_sqlite(missing, "SELECT 1") == []
    assert missing.exists() is False
    assert cohort_exists(missing, "2026-11-03") is False

    gex_path = tmp_path / "gex.db"
    gdb = GexDB(gex_path)
    with gdb._conn() as conn:
        conn.execute(
            """
            INSERT INTO gex_snapshots
                (date, asset, timestamp, spot_price, total_net_gex, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("2026-11-03", "BTC", "2026-11-03T18:00:00+00:00", 100_000.0, -9e9,
             "2026-11-03T18:00:00"),
        )
    paths = LocalPaths(tmp_path / "nope.db", gex_path, tmp_path / "nope.db")
    gex_rows, weights, barriers = load_local_snapshots(paths)
    assert weights == [] and barriers == []
    chosen, counts = select_gex(gex_rows, SLOT)
    assert chosen is None
    assert counts["same_day_after_slot"] == 1
    parsed = gex_rows_from_dicts([{"date": "2026-11-03", "timestamp": "not-a-timestamp"}])
    assert parsed == []


def test_metadata_column_is_added_to_an_existing_sqlite_and_old_rows_stay_null(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            asset TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_spec TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            confidence REAL NOT NULL,
            rationale TEXT,
            counter_analysis TEXT,
            human_overlay TEXT,
            score_ref REAL,
            components TEXT,
            weights_version INTEGER,
            model_version TEXT NOT NULL DEFAULT 'v1',
            status TEXT NOT NULL DEFAULT 'open'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO predictions (
            created_at, date, source, asset, target_type, target_spec,
            horizon_days, confidence, model_version, status
        ) VALUES ('2026-01-01T07:30:00', '2026-01-01', 'dealer_flow', 'BTC',
                  'direction', '{"direction":"up"}', 5, 0.5, 'v1', 'open')
        """
    )
    conn.commit()
    conn.close()

    db = PredictionDB(path)
    old = db.predictions_on_date("2026-01-01", SOURCE)
    assert len(old) == 1 and old[0].metadata is None
    rebuilt = Prediction(
        source=SOURCE, asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 1.0},
        horizon_days=5, confidence=0.4, created_at="2026-11-03T07:30:00",
        metadata={"reconstructed": True, "degraded_inputs": ["gex"]},
    )
    assert db.insert_prediction(rebuilt) is not None
    stored = db.predictions_on_date("2026-11-03", SOURCE)[0]
    assert stored.metadata["reconstructed"] is True
    assert db.predictions_on_date("2026-01-01", SOURCE)[0].metadata is None


def test_write_does_not_touch_other_cohorts_and_dry_run_writes_nothing(tmp_path, capsys):
    path = tmp_path / "preds.db"
    db = PredictionDB(path)
    other = Prediction(
        source=SOURCE, asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "down", "ref_price": 111.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.4, created_at="2026-11-02T07:30:00",
    )
    same_day_other_source = Prediction(
        source="ema", asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 222.0},
        horizon_days=5, confidence=0.2, created_at="2026-11-03T07:30:00",
    )
    assert db.insert_prediction(other) is not None
    assert db.insert_prediction(same_day_other_source) is not None
    before, before_outcomes = _dump(db)

    argv = [
        "--date", "2026-11-03", "--dry-run",
        "--predictions-db", str(path),
        "--gex-db", str(tmp_path / "no-gex.db"),
        "--notes-db", str(tmp_path / "no-notes.db"),
    ]
    rc = main(argv, loader=lambda slot, paths: _history())
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry-run"
    assert report["refused"] is None
    assert report["inserted_ids"] == []
    assert report["predictions"][0]["created_at"] == "2026-11-03T07:30:00"
    assert report["predictions"][0]["metadata"]["reconstructed"] is True
    assert report["provenance"]["inputs"]["spot"]["price"] == 100_000.0
    after, after_outcomes = _dump(db)
    assert after == before
    assert after_outcomes == before_outcomes

    rc = main(
        ["--date", "2026-11-03", "--write", "--predictions-db", str(path)],
        loader=lambda slot, paths: _history(),
    )
    assert rc == 0
    written = json.loads(capsys.readouterr().out)
    assert written["mode"] == "write"
    assert len(written["inserted_ids"]) == 3
    kept, _ = _dump(db)
    kept_other = [row for row in kept if row["date"] != "2026-11-03" or row["source"] != SOURCE]
    assert kept_other == before
    new_rows = [row for row in kept if row["date"] == "2026-11-03" and row["source"] == SOURCE]
    assert len(new_rows) == 3
    assert all(json.loads(row["metadata"])["reconstructed"] is True for row in new_rows)

    # Seconda scrittura: la coorte c'è già, le righe non cambiano.
    snapshot, _ = _dump(db)
    rc = main(
        ["--date", "2026-11-03", "--write", "--predictions-db", str(path)],
        loader=lambda slot, paths: _history(),
    )
    assert rc == 2
    again = json.loads(capsys.readouterr().out)
    assert again["refused"] == "cohort_exists"
    assert _dump(db)[0] == snapshot


def test_score_only_the_new_cohort(tmp_path):
    path = tmp_path / "preds.db"
    db = PredictionDB(path)
    scored = Prediction(
        source=SOURCE, asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.5, created_at="2026-09-01T07:30:00",
    )
    scored_id = db.insert_prediction(scored)
    from src.forecast.models import Outcome
    db.insert_outcome(Outcome(
        prediction_id=scored_id, hit=True, realized_return=0.02,
        detail="keep-me", scored_at="2026-09-10T08:00:00",
    ))
    other_due = Prediction(
        source=SOURCE, asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "down", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.5, created_at="2026-10-20T07:30:00",
    )
    other_id = db.insert_prediction(other_due)
    result = assemble(SLOT, _history())
    ids = db.insert_cohort(result.predictions)

    calls: list[str] = []

    def provider(asset, start, end):
        calls.append(start.date().isoformat())
        return _prices(start, 5, [101_000, 102_000, 103_000, 104_000, 110_000])

    outcomes = score_cohort(db, provider, ids, _utc(2026, 11, 10, 12))
    assert {o.prediction_id for o in outcomes} == set(ids)
    assert calls == ["2026-11-03"] * len(ids)
    still_open = db.predictions_on_date("2026-10-20", SOURCE)
    assert still_open[0].id == other_id
    assert still_open[0].status == "open"
    kept = db.predictions_on_date("2026-09-01", SOURCE)[0]
    assert kept.status == "scored"
    with db._conn() as conn:
        detail = conn.execute(
            "SELECT detail, scored_at FROM outcomes WHERE prediction_id = ?",
            (scored_id,),
        ).fetchone()
    assert detail["detail"] == "keep-me"
    assert detail["scored_at"] == "2026-09-10T08:00:00"


def test_insert_cohort_rolls_back_nothing_when_the_day_exists(tmp_path):
    db = PredictionDB(tmp_path / "p.db")
    existing = Prediction(
        source=SOURCE, asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "flat", "ref_price": 12_345.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.3, created_at="2026-11-03T07:30:00",
    )
    db.insert_prediction(existing)
    result = assemble(SLOT, _history())
    with pytest.raises(CohortExistsError):
        db.insert_cohort(result.predictions)
    rows = db.predictions_on_date("2026-11-03", SOURCE)
    assert len(rows) == 1
    assert rows[0].target_spec["ref_price"] == 12_345.0


def test_cli_refuses_postgres_and_does_not_open_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:secret-token@db.example/prod",
    )
    path = tmp_path / "preds.db"
    rc = main(
        ["--date", "2026-11-03", "--dry-run", "--predictions-db", str(path)],
        loader=lambda slot, paths: (_ for _ in ()).throw(AssertionError("loader called")),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "Postgres" in err
    assert "secret-token" not in err
    assert path.exists() is False


def test_score_flag_without_write_is_rejected(tmp_path, capsys):
    rc = main([
        "--date", "2026-11-03", "--dry-run", "--score",
        "--predictions-db", str(tmp_path / "p.db"),
    ])
    assert rc == 2
    assert "--score" in capsys.readouterr().err
