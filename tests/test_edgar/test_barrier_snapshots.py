"""barrier_snapshots: conteggio reale, data UTC, una riga rotta non annulla le altre.

Gira su SQLite e, con BIF_TEST_BACKEND=postgres, sullo stesso codice contro Postgres.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from src.edgar.models import BarrierLevel, StructuredNote
from src.edgar.structured_notes_db import StructuredNotesDB


def _note(url: str = "https://example.com/snap.htm") -> StructuredNote:
    return StructuredNote(
        filing_url=url,
        issuer="JPMorgan",
        issue_date=date(2024, 6, 1),
        maturity_date=date(2025, 6, 1),
        notional_usd=5_000_000.0,
        product_type="autocallable",
        underlying="IBIT",
        barriers=[
            BarrierLevel(
                barrier_type="knock_in", level_pct=70.0,
                level_price_ibit=31.5, level_price_btc=80_000.0,
            ),
            BarrierLevel(
                barrier_type="autocall", level_pct=100.0,
                level_price_ibit=45.0, level_price_btc=95_000.0,
            ),
        ],
    )


def _count(db: StructuredNotesDB, day: str | None = None) -> int:
    with db._conn() as conn:
        if day is None:
            return int(conn.execute("SELECT COUNT(*) FROM barrier_snapshots").fetchone()[0])
        return int(conn.execute(
            "SELECT COUNT(*) FROM barrier_snapshots WHERE snapshot_date = ?",
            (day,),
        ).fetchone()[0])


def test_snapshot_is_idempotent_and_counts_inserted_rows(tmp_path):
    db = StructuredNotesDB(db_path=tmp_path / "notes.db")
    db.upsert_note(_note())
    day = date(2026, 11, 10)
    assert db.snapshot_active_barriers(on_date=day) == 2
    assert _count(db, "2026-11-10") == 2
    # Il bug precedente incrementava anche gli INSERT ignorati e restituiva 2.
    assert db.snapshot_active_barriers(on_date=day) == 0
    assert _count(db, "2026-11-10") == 2


def test_default_snapshot_date_is_utc(tmp_path, monkeypatch):
    db = StructuredNotesDB(db_path=tmp_path / "notes.db")
    db.upsert_note(_note("https://example.com/utc.htm"))

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 10, 23, 40, tzinfo=timezone.utc)

    monkeypatch.setattr("src.edgar.structured_notes_db.datetime", Frozen)
    assert db.snapshot_active_barriers() == 2
    assert _count(db, "2026-11-10") == 2


def test_a_bad_row_does_not_drop_the_good_ones(tmp_path, monkeypatch):
    db = StructuredNotesDB(db_path=tmp_path / "notes.db")
    db.upsert_note(_note("https://example.com/partial.htm"))
    good = db.get_active_barriers()
    assert len(good) == 2
    bad = {
        "id": None,
        "level_price_btc": 1.0,
        "level_price_ibit": 1.0,
        "status": "active",
        "notional_usd": None,
        "issuer": "x",
        "barrier_type": "knock_in",
    }
    monkeypatch.setattr(db, "get_active_barriers", lambda: [*good, bad])
    written = db.snapshot_active_barriers(on_date=date(2026, 3, 1))
    assert written == 2
    assert _count(db, "2026-03-01") == 2
