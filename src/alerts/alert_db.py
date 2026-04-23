"""Persistenza stato alert su SQLite per dedup + cooldown cross-restart.

Tabella alert_state: 1 riga per tipo di alert con timestamp dell'ultimo invio
e hash del payload. Evita duplicati quando il backend riparte (l'in-memory
set di PTF-Dashboard perde lo stato). Condivide il file SQLite del progetto
(data/structured_notes.db) — stessa pattern di GexDB.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

from src.config import get_settings, setup_logging

_log = setup_logging("alerts.db")

_DDL = """
CREATE TABLE IF NOT EXISTS alert_state (
    alert_type        TEXT    PRIMARY KEY,
    last_sent_at      TEXT    NOT NULL,          -- ISO UTC datetime
    last_payload_hash TEXT    NOT NULL           -- sha256 hex del payload
);
"""


def payload_hash(payload: str) -> str:
    """sha256 hex del payload — usato per dedup identico."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AlertDB:
    """Registro degli alert inviati per cooldown e dedup.

    Args:
        db_path: percorso al file SQLite (default da settings.yaml).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings()
        self._path = Path(db_path or cfg["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ─── Read ────────────────────────────────────────────────────────────────

    def get_last_sent(self, alert_type: str) -> Optional[tuple[datetime, str]]:
        """Ritorna (timestamp UTC, hash) dell'ultimo invio, None se mai inviato."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_sent_at, last_payload_hash FROM alert_state WHERE alert_type = ?",
                (alert_type,),
            ).fetchone()
        if not row:
            return None
        try:
            ts = datetime.fromisoformat(row["last_sent_at"])
        except ValueError:
            return None
        return ts, row["last_payload_hash"]

    def within_cooldown(self, alert_type: str, hours: float) -> bool:
        """True se l'ultimo alert di questo tipo è stato inviato entro `hours` ore."""
        last = self.get_last_sent(alert_type)
        if last is None:
            return False
        ts, _ = last
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return datetime.now(tz=timezone.utc) - ts < timedelta(hours=hours)

    def is_duplicate(self, alert_type: str, payload: str) -> bool:
        """True se l'hash del payload coincide con l'ultimo inviato."""
        last = self.get_last_sent(alert_type)
        if last is None:
            return False
        _, last_hash = last
        return last_hash == payload_hash(payload)

    # ─── Write ───────────────────────────────────────────────────────────────

    def record_sent(self, alert_type: str, payload: str) -> None:
        """Registra l'invio: upsert con timestamp UTC corrente e hash del payload."""
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        phash = payload_hash(payload)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO alert_state (alert_type, last_sent_at, last_payload_hash)
                VALUES (?, ?, ?)
                ON CONFLICT(alert_type) DO UPDATE SET
                    last_sent_at      = excluded.last_sent_at,
                    last_payload_hash = excluded.last_payload_hash
                """,
                (alert_type, now_iso, phash),
            )
        _log.info("Alert registrato: type=%s hash=%s", alert_type, phash[:8])
