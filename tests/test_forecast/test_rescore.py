"""Rescore degli esiti già verificati sulle barre complete. Non tocca le predizioni."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.flows.price_backfill import main as backfill_main
from src.flows.price_fetcher import PriceFetcher
from src.forecast.models import Outcome, Prediction, TARGET_DIRECTION, TARGET_LEVEL
from src.forecast.prediction_db import PredictionDB
from src.forecast.rescore import main as rescore_main
from src.forecast.rescore import run_rescore


def _bars() -> pd.DataFrame:
    idx = pd.to_datetime([
        "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06",
    ])
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 104, 108],
            "High": [101, 102, 106, 105, 111],
            "Low": [99, 100, 101, 103, 107],
            "Close": [101, 102, 104, 108, 110],
            "Volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )


def _seed(db: PredictionDB, fetcher: PriceFetcher) -> tuple[int, int, int]:
    fetcher._store_df("BTC-USD", _bars())
    direction = Prediction(
        source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "up", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.4, created_at="2026-10-01T07:30:00",
        score_ref=70.0, weights_version=3,
    )
    level = Prediction(
        source="dealer_flow", asset="BTC", target_type=TARGET_LEVEL,
        target_spec={"level_name": "max_pain", "level_price": 105.0,
                     "mode": "reach", "side": "above", "ref_price": 100.0},
        horizon_days=5, confidence=0.4, created_at="2026-10-01T07:30:00",
    )
    # Senza la barra del 13 ottobre: l'esito vecchio deve restare.
    incomplete = Prediction(
        source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "down", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.2, created_at="2026-10-08T07:30:00",
    )
    ids = []
    for pred in (direction, level, incomplete):
        pid = db.insert_prediction(pred)
        assert pid is not None
        db.insert_outcome(Outcome(
            prediction_id=pid, hit=False, realized_return=0.0, realized_price=100.0,
            ref_price=100.0, signed_error=0.0, detail="stale-midday",
            scored_at="2026-10-20T08:00:00",
        ))
        ids.append(pid)
    # Predizione ancora aperta: il rescore non deve chiuderla né duplicarla.
    open_id = db.insert_prediction(Prediction(
        source="dealer_flow", asset="BTC", target_type=TARGET_DIRECTION,
        target_spec={"direction": "flat", "ref_price": 100.0, "flat_band_pct": 1.0},
        horizon_days=5, confidence=0.1, created_at="2026-10-20T07:30:00",
    ))
    assert open_id is not None
    return ids[0], ids[1], ids[2]


def _table(db: PredictionDB, sql: str) -> list[tuple]:
    with db._conn() as conn:
        return [tuple(row) for row in conn.execute(sql).fetchall()]


def test_dry_run_prints_diff_and_write_is_idempotent(tmp_path: Path, capsys):
    path = tmp_path / "runtime.db"
    db = PredictionDB(path)
    fetcher = PriceFetcher(db_path=path)
    direction_id, level_id, skipped_id = _seed(db, fetcher)
    before_preds = _table(db, "SELECT * FROM predictions ORDER BY id")
    before_weights = _table(db, "SELECT * FROM weight_versions ORDER BY id")

    preview = run_rescore(db, fetcher, write=False)
    assert preview.written is False
    by_id = {row.prediction_id: row for row in preview.rows}
    assert by_id[direction_id].changed
    assert by_id[direction_id].new["hit"] is True
    assert by_id[direction_id].new["realized_price"] == pytest.approx(110)
    assert by_id[level_id].changed
    assert by_id[level_id].new["hit"] is True  # high 106 tocca 105; il vecchio range no
    assert by_id[skipped_id].skip_reason
    assert _table(db, "SELECT realized_price, detail FROM outcomes ORDER BY prediction_id") == [
        (100.0, "stale-midday"),
        (100.0, "stale-midday"),
        (100.0, "stale-midday"),
    ]
    assert _table(db, "SELECT COUNT(*) FROM outcomes_archive") == [(0,)]

    # La suite Postgres ha DATABASE_URL sul database di test: il CLI si ferma
    # senza --allow-postgres. In SQLite la variabile è vuota e basta --db.
    argv = ["--dry-run", "--db", str(path)]
    if os.environ.get("DATABASE_URL", "").strip():
        argv.append("--allow-postgres")
    code = rescore_main(argv)
    printed = capsys.readouterr().out
    assert code == 0
    assert "100" in printed and "110" in printed
    assert "summary:" in printed
    assert "CHANGED" in printed

    written = run_rescore(db, fetcher, write=True, reason="partial-daily-bars")
    assert written.written is True
    assert len(written.changed) == 2
    assert _table(db, "SELECT * FROM predictions ORDER BY id") == before_preds
    assert _table(db, "SELECT * FROM weight_versions ORDER BY id") == before_weights
    assert db.count() == 4  # tre scored + una open, nessuna predizione nuova

    archived = _table(
        db,
        "SELECT prediction_id, realized_price, detail, reason FROM outcomes_archive ORDER BY id",
    )
    assert archived == [
        (direction_id, 100.0, "stale-midday", "partial-daily-bars"),
        (level_id, 100.0, "stale-midday", "partial-daily-bars"),
    ]
    refreshed = {
        row[0]: row
        for row in _table(db, "SELECT prediction_id, realized_price, hit FROM outcomes")
    }
    assert refreshed[direction_id][1] == pytest.approx(110)
    assert refreshed[direction_id][2] == 1
    assert refreshed[skipped_id][1] == pytest.approx(100)
    open_row = next(p for p in db.get_recent(limit=10) if p.id != direction_id
                    and p.created_at.startswith("2026-10-20"))
    assert open_row.status == "open"

    again = run_rescore(db, fetcher, write=True, reason="partial-daily-bars")
    assert again.changed == []
    assert _table(db, "SELECT COUNT(*) FROM outcomes_archive") == [(2,)]
    assert _table(db, "SELECT * FROM predictions ORDER BY id") == before_preds


def test_cli_refuses_postgres(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/runtime")
    db = str(tmp_path / "runtime.db")
    assert backfill_main(["--dry-run", "--db", db]) == 2
    assert rescore_main(["--dry-run", "--db", db]) == 2
    err = capsys.readouterr().err
    assert "Postgres" in err
    assert "--allow-postgres" in err


def test_backfill_cli_dry_run_does_not_write(monkeypatch, tmp_path: Path, capsys):
    path = tmp_path / "runtime.db"
    fetcher = PriceFetcher(db_path=path)
    fetcher._store_df("BTC-USD", _bars())
    monkeypatch.setattr(
        "src.flows.price_fetcher.yf.download",
        lambda *args, **kwargs: _bars().assign(Close=[201, 202, 204, 208, 210]),
    )
    argv = ["--dry-run", "--db", str(path), "--asof", "2026-10-07", "--ticker", "BTC-USD"]
    if os.environ.get("DATABASE_URL", "").strip():
        argv.append("--allow-postgres")
    code = backfill_main(argv)
    out = capsys.readouterr().out
    assert code == 0
    assert "2026-10-06" in out
    assert "110.00 -> 210.00" in out
    stored = fetcher._load_from_db("BTC-USD", date(2026, 10, 6), date(2026, 10, 6))
    assert stored["close"].iloc[0] == pytest.approx(110)
