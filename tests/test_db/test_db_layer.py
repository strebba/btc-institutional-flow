"""Test del layer DB unificato (src.db): SQLite sempre, Postgres se disponibile."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from src import db as dbmod
from src.db import (
    PgRow,
    connect,
    days_ago_iso,
    get_database_url,
    redact_url,
    require_postgres,
    translate_sql,
)

# ─── translate_sql ───────────────────────────────────────────────────────────


class TestTranslateSql:
    def test_placeholders(self):
        assert translate_sql("SELECT * FROM t WHERE a = ? AND b = ?") == (
            "SELECT * FROM t WHERE a = %s AND b = %s"
        )

    def test_literal_percent_escaped(self):
        assert translate_sql("SELECT * FROM t WHERE a LIKE 'x%' AND b = ?") == (
            "SELECT * FROM t WHERE a LIKE 'x%%' AND b = %s"
        )
        assert translate_sql("SELECT a % 2 FROM t") == "SELECT a %% 2 FROM t"

    def test_question_mark_inside_string_untouched(self):
        assert translate_sql("SELECT 'perche?' , ?") == "SELECT 'perche?' , %s"

    def test_comment_with_apostrophe(self):
        sql = "-- e' un commento con ?\nSELECT ?"
        assert translate_sql(sql) == "-- e' un commento con ?\nSELECT %s"

    def test_insert_or_ignore(self):
        assert translate_sql("INSERT OR IGNORE INTO t (a) VALUES (?)") == (
            "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING"
        )

    def test_insert_or_ignore_with_returning(self):
        assert translate_sql("insert or ignore into t (a) values (?) RETURNING id;") == (
            "INSERT INTO t (a) values (%s) ON CONFLICT DO NOTHING RETURNING id"
        )

    def test_insert_or_replace_rejected(self):
        with pytest.raises(ValueError):
            translate_sql("INSERT OR REPLACE INTO t VALUES (?)")


# ─── Selezione backend ───────────────────────────────────────────────────────


class TestBackendSelection:
    def test_empty_means_sqlite(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "")
        assert get_database_url() is None
        assert dbmod.backend_name() == "sqlite"

    @pytest.mark.parametrize("url", [
        "postgres://u:p@h:5432/db", "postgresql://u:p@h:6543/postgres?sslmode=require",
    ])
    def test_postgres_schemes(self, monkeypatch, url):
        monkeypatch.setenv("DATABASE_URL", url)
        assert get_database_url() == url
        assert dbmod.is_postgres()

    def test_other_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "mysql://u@h/db")
        with pytest.raises(ValueError):
            get_database_url()

    def test_redact_url_hides_password(self):
        out = redact_url("postgresql://user:s3cret@db.example.com:6543/postgres")
        assert "s3cret" not in out
        assert out == "postgresql://user@db.example.com:6543/postgres"

    def test_require_postgres_exits_without_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "")
        with pytest.raises(SystemExit):
            require_postgres("test")

    def test_days_ago_iso(self):
        expected = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        assert days_ago_iso(7) == expected


# ─── SQLite ──────────────────────────────────────────────────────────────────


class TestSqliteConnect:
    @pytest.fixture(autouse=True)
    def _force_sqlite(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "")

    def test_commit_on_success(self, tmp_path):
        p = tmp_path / "x.db"
        with connect(p) as conn:
            conn.execute("CREATE TABLE t (a INTEGER)")
            conn.execute("INSERT INTO t VALUES (?)", (1,))
        with sqlite3.connect(p) as raw:
            assert raw.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1

    def test_rollback_on_error(self, tmp_path):
        p = tmp_path / "x.db"
        with connect(p) as conn:
            conn.execute("CREATE TABLE t (a INTEGER)")
        with pytest.raises(RuntimeError):
            with connect(p) as conn:
                conn.execute("INSERT INTO t VALUES (?)", (1,))
                raise RuntimeError("boom")
        with sqlite3.connect(p) as raw:
            assert raw.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0

    def test_rows_by_name_and_wal(self, tmp_path):
        with connect(tmp_path / "x.db") as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
            conn.execute("INSERT INTO t VALUES (1, 'x')")
            row = conn.execute("SELECT a, b FROM t").fetchone()
            assert row["a"] == 1 and row["b"] == "x"


def test_pgrow_access_by_name_and_index():
    row = PgRow((1, "x"), {"a": 0, "b": 1})
    assert row[0] == 1 and row["b"] == "x"
    assert row.keys() == ["a", "b"]
    assert tuple(row) == (1, "x")
    with pytest.raises(IndexError):
        row["missing"]


# ─── Postgres (solo con TEST_DATABASE_URL locale) ────────────────────────────


@pytest.fixture
def pg(pg_url, monkeypatch):
    from tests.conftest import truncate_pg_tables

    truncate_pg_tables(pg_url)
    monkeypatch.setenv("DATABASE_URL", pg_url)
    conn = dbmod.pg_connect(pg_url)
    yield conn
    conn.close()


class TestPostgres:
    def test_schema_applied_and_tracked(self, pg, pg_url):
        dbmod.reset_schema_cache()
        assert dbmod.ensure_postgres_schema(pg_url) == []  # gia' applicato: idempotente
        versions = [r[0] for r in pg.execute("SELECT version FROM schema_migrations")]
        assert versions == [p.stem for p in dbmod.migration_files()]

    def test_round_trip_with_sqlite_dialect(self, pg):
        cur = pg.execute(
            "INSERT OR IGNORE INTO alert_state (alert_type, last_sent_at, last_payload_hash) "
            "VALUES (?, ?, ?) RETURNING alert_type",
            ("gex_flip", "2026-10-29T08:00:00", "abc"),
        )
        assert cur.fetchall()[0]["alert_type"] == "gex_flip"
        cur = pg.execute(
            "INSERT OR IGNORE INTO alert_state (alert_type, last_sent_at, last_payload_hash) "
            "VALUES (?, ?, ?) RETURNING alert_type",
            ("gex_flip", "x", "y"),
        )
        assert cur.fetchall() == []  # duplicato ignorato, nessun errore
        row = pg.execute("SELECT * FROM alert_state WHERE alert_type = ?", ("gex_flip",)).fetchone()
        assert row["last_payload_hash"] == "abc"

    def test_pragma_table_info(self, pg):
        names = [r["name"] for r in pg.execute("PRAGMA table_info(macro_snapshots)")]
        assert names[:2] == ["snapshot_date", "asset"]
        assert pg.execute("PRAGMA journal_mode=WAL").fetchone() is None

    def test_adapters(self, pg):
        import numpy as np

        pg.execute(
            "INSERT INTO prices (ticker, date, close, volume, created_at) VALUES (?, ?, ?, ?, ?)",
            ("BTC-USD", date(2026, 10, 29), np.float64(70000.5), np.int64(12),
             datetime(2026, 10, 29, 8, 0, 0)),
        )
        row = pg.execute("SELECT date, close, volume, created_at FROM prices").fetchone()
        assert row["date"] == "2026-10-29"
        assert row["close"] == 70000.5
        assert row["volume"] == 12
        assert row["created_at"] == "2026-10-29 08:00:00"
        # SUM/AVG su interi danno numeric: letto come float (come SQLite), non Decimal
        agg = pg.execute("SELECT AVG(volume)::numeric AS a, SUM(1) AS s FROM prices").fetchone()
        assert isinstance(agg["a"], float) and agg["a"] == 12.0

    def test_bool_stored_as_int(self, pg):
        pg.execute(
            "INSERT INTO refresh_runs (run_at, filings_seen, notes_written, ok) VALUES (?, ?, ?, ?)",
            ("2026-10-29", 1, 1, True),
        )
        assert pg.execute("SELECT ok FROM refresh_runs").fetchone()[0] == 1

    def test_lastrowid_not_supported(self, pg):
        cur = pg.execute("SELECT 1")
        with pytest.raises(AttributeError):
            cur.lastrowid
