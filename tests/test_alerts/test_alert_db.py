"""Test per AlertDB — cooldown e dedup persistenti su SQLite."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.alerts.alert_db import AlertDB, payload_hash


@pytest.fixture
def db(tmp_path: Path) -> AlertDB:
    return AlertDB(db_path=tmp_path / "alerts.db")


class TestSchema:
    def test_table_created_on_init(self, db: AlertDB) -> None:
        with db._conn() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "alert_state" in tables


class TestCooldown:
    def test_no_cooldown_if_never_sent(self, db: AlertDB) -> None:
        assert db.within_cooldown("daily_recap", hours=24) is False

    def test_cooldown_active_right_after_send(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "hello")
        assert db.within_cooldown("daily_recap", hours=24) is True

    def test_cooldown_expired_after_manual_backdate(self, db: AlertDB) -> None:
        """Simula alert inviato 25 ore fa → cooldown 24h è scaduto."""
        past = (datetime.now(tz=timezone.utc) - timedelta(hours=25)).isoformat()
        with db._conn() as conn:
            conn.execute(
                "INSERT INTO alert_state VALUES (?, ?, ?)",
                ("daily_recap", past, payload_hash("old")),
            )
        assert db.within_cooldown("daily_recap", hours=24) is False

    def test_cooldown_independent_per_alert_type(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "x")
        assert db.within_cooldown("daily_recap", hours=24) is True
        assert db.within_cooldown("etf_flow_event", hours=24) is False


class TestDedup:
    def test_duplicate_returns_false_if_never_sent(self, db: AlertDB) -> None:
        assert db.is_duplicate("daily_recap", "x") is False

    def test_duplicate_returns_true_for_same_payload(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "identical content")
        assert db.is_duplicate("daily_recap", "identical content") is True

    def test_duplicate_returns_false_for_different_payload(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "first")
        assert db.is_duplicate("daily_recap", "second") is False

    def test_record_sent_overwrites_previous(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "v1")
        db.record_sent("daily_recap", "v2")
        assert db.is_duplicate("daily_recap", "v2") is True
        assert db.is_duplicate("daily_recap", "v1") is False


class TestPayloadHash:
    def test_hash_is_deterministic(self) -> None:
        assert payload_hash("abc") == payload_hash("abc")

    def test_hash_differs_for_different_payloads(self) -> None:
        assert payload_hash("abc") != payload_hash("abd")

    def test_hash_is_hex_sha256(self) -> None:
        h = payload_hash("any")
        assert len(h) == 64
        int(h, 16)  # non solleva se hex valido


class TestSentToday:
    def test_false_if_never_sent(self, db: AlertDB) -> None:
        assert db.sent_today("daily_recap") is False

    def test_true_right_after_send(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "hello")
        assert db.sent_today("daily_recap") is True

    def test_false_if_sent_yesterday(self, db: AlertDB) -> None:
        """Messaggio inviato ieri pomeriggio non deve bloccare quello di oggi."""
        yesterday_afternoon = (
            datetime.now(tz=timezone.utc) - timedelta(hours=18)
        ).isoformat()
        with db._conn() as conn:
            conn.execute(
                "INSERT INTO alert_state VALUES (?, ?, ?)",
                ("daily_recap", yesterday_afternoon, payload_hash("old")),
            )
        # Se sono passate 18h ma siamo in un nuovo giorno UTC, deve essere False
        # (il test potrebbe essere tautologico se gira poco dopo mezzanotte UTC,
        #  ma copre il caso comune)
        yesterday_dt = datetime.now(tz=timezone.utc) - timedelta(hours=18)
        today_dt = datetime.now(tz=timezone.utc)
        if yesterday_dt.date() < today_dt.date():
            assert db.sent_today("daily_recap") is False

    def test_independent_per_type(self, db: AlertDB) -> None:
        db.record_sent("daily_recap", "x")
        assert db.sent_today("daily_recap") is True
        assert db.sent_today("etf_flow_event") is False


class TestPersistence:
    def test_state_survives_new_instance(self, tmp_path: Path) -> None:
        """Simula restart: istanza A scrive, istanza B legge."""
        path = tmp_path / "alerts.db"
        a = AlertDB(db_path=path)
        a.record_sent("daily_recap", "persisted")

        b = AlertDB(db_path=path)
        assert b.within_cooldown("daily_recap", hours=24) is True
        assert b.is_duplicate("daily_recap", "persisted") is True
