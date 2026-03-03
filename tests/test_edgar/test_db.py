"""Test unitari per StructuredNotesDB."""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest
from src.edgar.models import BarrierLevel, StructuredNote
from src.edgar.structured_notes_db import StructuredNotesDB


@pytest.fixture
def db(tmp_path: Path) -> StructuredNotesDB:
    """DB temporaneo per i test."""
    return StructuredNotesDB(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_note() -> StructuredNote:
    return StructuredNote(
        filing_url="https://example.com/test.htm",
        issuer="JPMorgan",
        issue_date=date(2024, 6, 1),
        maturity_date=date(2025, 6, 1),
        notional_usd=10_000_000.0,
        product_type="autocallable",
        initial_level=45.0,
        autocall_trigger_pct=100.0,
        knockin_barrier_pct=70.0,
        barriers=[
            BarrierLevel(barrier_type="knock_in", level_pct=70.0, level_price_ibit=31.5),
            BarrierLevel(barrier_type="autocall",  level_pct=100.0, level_price_ibit=45.0),
        ],
    )


class TestUpsertNote:
    def test_insert_new(self, db, sample_note):
        note_id = db.upsert_note(sample_note)
        assert note_id > 0

    def test_idempotent(self, db, sample_note):
        id1 = db.upsert_note(sample_note)
        id2 = db.upsert_note(sample_note)  # stesso url → update
        assert id1 == id2

    def test_barriers_saved(self, db, sample_note):
        note_id = db.upsert_note(sample_note)
        retrieved = db.get_all_notes()
        assert len(retrieved) == 1
        assert len(retrieved[0].barriers) == 2


class TestGetAllNotes:
    def test_empty_db(self, db):
        assert db.get_all_notes() == []

    def test_returns_notes(self, db, sample_note):
        db.upsert_note(sample_note)
        notes = db.get_all_notes()
        assert len(notes) == 1
        n = notes[0]
        assert n.issuer == "JPMorgan"
        assert n.notional_usd == 10_000_000.0
        assert n.initial_level == 45.0


class TestActiveBarriers:
    def test_active(self, db, sample_note):
        db.upsert_note(sample_note)
        active = db.get_active_barriers()
        assert len(active) == 2
        assert all(b["status"] == "active" for b in active)


class TestComputeBtcPrices:
    def test_conversion(self, db, sample_note):
        db.upsert_note(sample_note)
        ratio = 0.001  # 1 IBIT ≈ 0.001 BTC
        db.compute_btc_prices(ibit_btc_ratio=ratio)
        notes = db.get_all_notes()
        btc_prices = [b.level_price_btc for b in notes[0].barriers if b.level_price_ibit]
        assert all(p is not None for p in btc_prices)
        # level_price_ibit = 31.5 → btc_price = 31.5 / 0.001 = 31500
        assert any(abs(p - 31_500) < 1 for p in btc_prices)


class TestUpdateBarrierStatus:
    def test_knockin_triggered(self, db, sample_note):
        db.upsert_note(sample_note)
        # IBIT a 30 < 31.5 → knock_in triggered
        counts = db.update_barrier_statuses(current_ibit_price=30.0)
        assert counts["triggered"] >= 1

    def test_no_trigger(self, db, sample_note):
        db.upsert_note(sample_note)
        # IBIT a 40: knock_in level=31.5 (40>31.5 → no), autocall level=45 (40<45 → no)
        counts = db.update_barrier_statuses(current_ibit_price=40.0)
        assert counts["triggered"] == 0


class TestSummary:
    def test_summary(self, db, sample_note):
        db.upsert_note(sample_note)
        s = db.summary()
        assert s["total_notes"] == 1
        assert s["total_barriers"] == 2
        assert s["active_barriers"] == 2
        assert s["total_notional_usd"] == 10_000_000.0
        assert "autocallable" in s["by_product_type"]
        assert "JPMorgan" in s["by_issuer"]
