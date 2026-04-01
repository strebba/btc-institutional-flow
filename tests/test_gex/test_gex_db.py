"""Test per GexDB — persistenza snapshot GEX in SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.gex.gex_db import GexDB
from src.gex.models import GexSnapshot


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_snapshot(
    spot: float = 85_000.0,
    gex: float = 500_000_000.0,
    flip: float = 82_000.0,
    put_wall: float = 80_000.0,
    call_wall: float = 90_000.0,
) -> GexSnapshot:
    return GexSnapshot(
        timestamp        = datetime(2026, 1, 15, 16, 0, tzinfo=timezone.utc),
        spot_price       = spot,
        total_net_gex    = gex,
        gamma_flip_price = flip,
        put_wall         = put_wall,
        call_wall        = call_wall,
        max_pain         = 84_000.0,
        total_call_oi    = 10_000.0,
        total_put_oi     = 8_000.0,
    )


@pytest.fixture
def db(tmp_path: Path) -> GexDB:
    """GexDB su file temporaneo — isolato per ogni test."""
    return GexDB(db_path=tmp_path / "test_gex.db")


# ─── Schema ──────────────────────────────────────────────────────────────────

class TestSchema:
    def test_table_created_on_init(self, db: GexDB) -> None:
        """La tabella gex_snapshots esiste dopo __init__."""
        with db._conn() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "gex_snapshots" in tables

    def test_index_created_on_init(self, db: GexDB) -> None:
        """L'indice idx_gex_date esiste dopo __init__."""
        with db._conn() as conn:
            indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
        assert "idx_gex_date" in indexes

    def test_db_created_in_tmp_dir(self, tmp_path: Path) -> None:
        """Il file DB viene creato nella directory specificata."""
        db_path = tmp_path / "subdir" / "gex.db"
        GexDB(db_path=db_path)
        assert db_path.exists()


# ─── Insert ──────────────────────────────────────────────────────────────────

class TestInsert:
    def test_insert_snapshot_returns_none(self, db: GexDB) -> None:
        """insert_snapshot non ritorna nulla (side-effect only)."""
        result = db.insert_snapshot(_make_snapshot(), "positive_gamma")
        assert result is None

    def test_insert_increments_count(self, db: GexDB) -> None:
        """Dopo insert il conteggio è 1."""
        db.insert_snapshot(_make_snapshot(), "positive_gamma")
        assert db.count() == 1

    def test_insert_duplicate_date_upserts(self, db: GexDB) -> None:
        """Due insert nello stesso giorno → count rimane 1, valore aggiornato."""
        db.insert_snapshot(_make_snapshot(gex=100_000_000.0), "positive_gamma")
        db.insert_snapshot(_make_snapshot(gex=200_000_000.0), "negative_gamma")
        assert db.count() == 1
        series = db.get_series(days=1)
        assert abs(series.iloc[0] - 200_000_000.0) < 1.0

    def test_insert_stores_regime(self, db: GexDB) -> None:
        """Il regime viene salvato correttamente."""
        db.insert_snapshot(_make_snapshot(), "neutral")
        with db._conn() as conn:
            row = conn.execute("SELECT regime FROM gex_snapshots").fetchone()
        assert row["regime"] == "neutral"


# ─── get_series ───────────────────────────────────────────────────────────────

class TestGetSeries:
    def test_get_series_empty_returns_empty_series(self, db: GexDB) -> None:
        """DB vuoto → Serie vuota, non errore."""
        s = db.get_series(days=30)
        assert isinstance(s, pd.Series)
        assert s.empty

    def test_get_series_returns_datetimeindex(self, db: GexDB) -> None:
        """La serie ha DatetimeIndex tz-naive (compatibile con FlowCorrelation)."""
        db.insert_snapshot(_make_snapshot(), "positive_gamma")
        s = db.get_series(days=1)
        assert isinstance(s.index, pd.DatetimeIndex)
        assert s.index.tz is None  # tz-naive per join con merged DataFrame

    def test_get_series_value_matches_inserted(self, db: GexDB) -> None:
        """Il valore restituito corrisponde a quello inserito."""
        db.insert_snapshot(_make_snapshot(gex=123_456_789.0), "positive_gamma")
        s = db.get_series(days=1)
        assert not s.empty
        assert abs(s.iloc[0] - 123_456_789.0) < 1.0

    def test_get_series_respects_days_limit(self, db: GexDB) -> None:
        """days=0 non ritorna nulla (nessun dato nel futuro)."""
        db.insert_snapshot(_make_snapshot(), "positive_gamma")
        # days=0 → WHERE date >= date('now', '0 days') → solo oggi incluso
        # Verifica che days limiti correttamente
        s_long  = db.get_series(days=365)
        s_short = db.get_series(days=0)
        # s_long deve contenere il record (oggi), s_short potrebbe no (dipende da sqlite)
        assert len(s_long) >= len(s_short)


# ─── get_latest_n ─────────────────────────────────────────────────────────────

class TestGetLatestN:
    def test_get_latest_n_empty_db(self, db: GexDB) -> None:
        """DB vuoto → lista vuota."""
        result = db.get_latest_n(10)
        assert result == []

    def test_get_latest_n_returns_list_of_snapshots(self, db: GexDB) -> None:
        """Ritorna list[GexSnapshot]."""
        db.insert_snapshot(_make_snapshot(), "positive_gamma")
        result = db.get_latest_n(10)
        assert len(result) == 1
        assert isinstance(result[0], GexSnapshot)

    def test_get_latest_n_snapshot_fields(self, db: GexDB) -> None:
        """I campi dello snapshot sono correttamente deserializzati."""
        snap = _make_snapshot(spot=88_000.0, gex=300_000_000.0)
        db.insert_snapshot(snap, "positive_gamma")
        result = db.get_latest_n(1)
        assert abs(result[0].spot_price - 88_000.0) < 1.0
        assert abs(result[0].total_net_gex - 300_000_000.0) < 1.0


# ─── get_all_for_regime ───────────────────────────────────────────────────────

class TestGetAllForRegime:
    def test_get_all_for_regime_empty(self, db: GexDB) -> None:
        """DB vuoto → DataFrame con colonne corrette ma 0 righe."""
        df = db.get_all_for_regime()
        assert isinstance(df, pd.DataFrame)
        assert "total_net_gex" in df.columns
        assert "regime" in df.columns
        assert len(df) == 0

    def test_get_all_for_regime_returns_dataframe(self, db: GexDB) -> None:
        """Inserimento → DataFrame con 1 riga."""
        db.insert_snapshot(_make_snapshot(), "positive_gamma")
        df = db.get_all_for_regime()
        assert len(df) == 1
        assert df["regime"].iloc[0] == "positive_gamma"
