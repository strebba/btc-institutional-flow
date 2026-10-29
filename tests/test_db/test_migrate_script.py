"""Test di scripts/migrate_sqlite_to_postgres.py (sorgenti SQLite temporanee).

I test che scrivono usano solo il Postgres locale di TEST_DATABASE_URL (skip se assente).
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_sqlite_to_postgres.py"
_spec = importlib.util.spec_from_file_location("migrate_sqlite_to_postgres", _SCRIPT)
mig = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mig  # richiesto da @dataclass
_spec.loader.exec_module(mig)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """runtime.db + structured_notes.db con lo schema reale delle classi DB."""
    monkeypatch.setenv("DATABASE_URL", "")  # crea gli schemi su SQLite
    from src.alerts.alert_db import AlertDB
    from src.edgar.structured_notes_db import StructuredNotesDB
    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.prediction_db import PredictionDB
    from src.gex.gex_db import GexDB

    runtime, notes = tmp_path / "runtime.db", tmp_path / "notes.db"
    PredictionDB(runtime)
    AlertDB(runtime)
    PriceFetcher(db_path=runtime)
    StructuredNotesDB(notes)
    GexDB(notes)

    with sqlite3.connect(runtime) as c:
        c.execute("INSERT INTO weight_versions (id, created_at, source, weights, active, status) "
                  "VALUES (3, '2026-10-01', 'dealer_flow', '{}', 1, 'active')")
        for i, d in ((5, "2026-10-20"), (9, "2026-10-21")):
            c.execute(
                "INSERT INTO predictions (id, created_at, date, source, asset, target_type, "
                "target_spec, horizon_days, confidence, weights_version) "
                "VALUES (?, ?, ?, 'dealer_flow', 'BTC', 'direction', '{}', 5, 0.6, 3)",
                (i, d + "T07:30:00", d),
            )
        c.execute("INSERT INTO outcomes (id, prediction_id, hit, scored_at) "
                  "VALUES (1, 5, 1, '2026-10-26')")
        c.execute("INSERT INTO outcomes (id, prediction_id, hit, scored_at) "
                  "VALUES (2, 9, 0, '2026-10-27')")
        c.execute("INSERT INTO alert_state VALUES ('gex_flip', '2026-10-28', 'h')")
        c.execute("INSERT INTO prices (ticker, date, close) VALUES ('BTC-USD', '2026-10-28', 70000.0)")
    with sqlite3.connect(notes) as c:
        c.execute("INSERT INTO notes (id, filing_url, created_at) VALUES (7, 'https://sec/a', '2026-10-01')")
        c.execute("INSERT INTO barrier_levels (id, note_id, barrier_type, level_pct, created_at) "
                  "VALUES (11, 7, 'knock_in', 0.7, '2026-10-01')")
        c.execute("INSERT INTO barrier_snapshots (barrier_id, snapshot_date, created_at) "
                  "VALUES (11, '2026-10-28', '2026-10-28 06:30:00')")
        c.execute("INSERT INTO refresh_runs (id, run_at, filings_seen, notes_written, ok) "
                  "VALUES (4, '2026-10-27', 3, 1, 1)")
        c.execute("INSERT INTO macro_snapshots (snapshot_date, asset, captured_at, oi_usd) "
                  "VALUES ('2026-10-28', 'BTC', '2026-10-28T06:10:00', 1.5e10)")
        c.execute("INSERT INTO gex_snapshots (date, asset, timestamp, spot_price, total_net_gex, "
                  "created_at) VALUES ('2026-10-28', 'BTC', 't', 70000, 1.2e9, 'c')")
    return {"runtime": runtime, "notes": notes, "gex": notes}


def _by_table(results):
    return {r.table: r for r in results}


def test_dry_run_offline_counts_without_writing(sources):
    before = {p: p.read_bytes() for p in set(sources.values())}
    res = _by_table(mig.migrate(sources, None, dry_run=True))
    assert res["predictions"].source_rows == 2
    assert res["outcomes"].source_rows == 2
    assert res["gex_snapshots"].source_rows == 1
    assert res["predictions"].inserted == 0
    assert {p: p.read_bytes() for p in before} == before  # sorgenti intatte


def test_missing_source_table_is_reported(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    res = _by_table(mig.migrate({"runtime": empty}, None, dry_run=True))
    assert res["predictions"].note == "tabella assente nella sorgente"
    assert res["notes"].note.startswith("sorgente")


def test_cli_requires_yes_for_real_run(sources):
    with pytest.raises(SystemExit) as exc:
        mig.main(["--runtime-db", str(sources["runtime"]),
                  "--database-url", "postgresql://u@127.0.0.1:1/x"])
    assert exc.value.code == 2


def test_cli_requires_a_source():
    with pytest.raises(SystemExit) as exc:
        mig.main(["--dry-run"])
    assert exc.value.code == 2


def test_cli_dry_run_offline(sources, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "")
    assert mig.main(["--runtime-db", str(sources["runtime"]), "--dry-run"]) == 0
    assert "DRY-RUN" in capsys.readouterr().out


# ─── Postgres locale ─────────────────────────────────────────────────────────


@pytest.fixture
def clean_pg(pg_url):
    from tests.conftest import truncate_pg_tables

    truncate_pg_tables(pg_url)
    return pg_url


def test_migration_copies_everything_and_is_idempotent(sources, clean_pg, monkeypatch):
    first = _by_table(mig.migrate(sources, clean_pg, dry_run=False))
    expected = {
        "weight_versions": 1, "predictions": 2, "outcomes": 2, "alert_state": 1,
        "prices": 1, "gex_snapshots": 1, "notes": 1, "barrier_levels": 1,
        "barrier_snapshots": 1, "refresh_runs": 1, "macro_snapshots": 1,
    }
    assert {t: first[t].inserted for t in expected} == expected
    assert all(r.id_conflicts == 0 and r.skipped_orphans == 0 for r in first.values())

    second = _by_table(mig.migrate(sources, clean_pg, dry_run=False))
    assert all(r.inserted == 0 for r in second.values())
    assert second["predictions"].target_before == 2

    # id preservati e sequenze riallineate: l'app continua a scrivere senza collisioni
    monkeypatch.setenv("DATABASE_URL", clean_pg)
    from src.db import pg_connect

    conn = pg_connect(clean_pg)
    try:
        ids = sorted(r[0] for r in conn.execute("SELECT id FROM predictions"))
        assert ids == [5, 9]
        assert conn.execute("SELECT prediction_id FROM outcomes WHERE id = 2").fetchone()[0] == 9
        assert conn.execute("SELECT note_id FROM barrier_levels").fetchone()[0] == 7
        new_id = conn.execute(
            "INSERT INTO predictions (created_at, date, source, asset, target_type, "
            "target_spec, horizon_days, confidence) VALUES "
            "('2026-10-29T07:30:00', '2026-10-29', 'dealer_flow', 'BTC', 'direction', '{}', 5, 0.5) "
            "RETURNING id"
        ).fetchone()[0]
        assert new_id == 10
        conn.rollback()
    finally:
        conn.close()


def test_dry_run_against_postgres_writes_nothing(sources, clean_pg):
    res = _by_table(mig.migrate(sources, clean_pg, dry_run=True))
    assert res["predictions"].target_before == 0
    from src.db import pg_connect

    conn = pg_connect(clean_pg)
    try:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    finally:
        conn.close()


def test_conflicting_parent_skips_children(sources, clean_pg):
    """Se in Postgres l'id 5 e' gia' una predizione diversa, l'outcome non si aggancia."""
    from src.db import pg_connect

    conn = pg_connect(clean_pg)
    conn.execute(
        "INSERT INTO predictions (id, created_at, date, source, asset, target_type, "
        "target_spec, horizon_days, confidence) VALUES "
        "(5, 'x', '2026-01-01', 'dealer_flow', 'BTC', 'direction', '{}', 5, 0.5)"
    )
    conn.commit()
    conn.close()

    res = _by_table(mig.migrate(sources, clean_pg, dry_run=False))
    assert res["predictions"].inserted == 1          # solo la 9
    assert res["predictions"].id_conflicts == 1      # la 5 e' in conflitto
    assert res["outcomes"].skipped_orphans == 1      # l'outcome della 5 non viene copiato
    assert res["outcomes"].inserted == 1
