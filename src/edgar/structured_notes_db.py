"""Database SQLite per le note strutturate e i barrier levels.

Schema:
  - notes: una riga per ogni nota strutturata estratta da EDGAR.
  - barrier_levels: una o più righe per ogni nota, una per barrier.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Generator, Optional

from src.config import get_settings, setup_logging
from src.edgar.models import BarrierLevel, StructuredNote

_log = setup_logging("edgar.db")

# ─── DDL ─────────────────────────────────────────────────────────────────────

_DDL_NOTES = """
CREATE TABLE IF NOT EXISTS notes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_url          TEXT    NOT NULL UNIQUE,
    issuer              TEXT,
    issue_date          TEXT,
    maturity_date       TEXT,
    notional_usd        REAL,
    product_type        TEXT,
    underlying          TEXT    DEFAULT 'IBIT',
    initial_level       REAL,
    autocall_trigger_pct REAL,
    knockin_barrier_pct  REAL,
    buffer_pct          REAL,
    participation_rate  REAL,
    coupon_rate         REAL,
    observation_dates   TEXT,   -- JSON list of date strings
    raw_text            TEXT,
    created_at          TEXT    NOT NULL
);
"""

_DDL_BARRIERS = """
CREATE TABLE IF NOT EXISTS barrier_levels (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id             INTEGER NOT NULL REFERENCES notes(id),
    barrier_type        TEXT    NOT NULL,  -- knock_in | autocall | buffer | knock_out
    level_pct           REAL    NOT NULL,
    level_price_ibit    REAL,
    level_price_btc     REAL,
    observation_date    TEXT,
    status              TEXT    DEFAULT 'active',  -- active | triggered | expired
    created_at          TEXT    NOT NULL
);
"""

_DDL_IDX = """
CREATE INDEX IF NOT EXISTS idx_notes_issuer       ON notes(issuer);
CREATE INDEX IF NOT EXISTS idx_notes_product      ON notes(product_type);
CREATE INDEX IF NOT EXISTS idx_notes_issue_date   ON notes(issue_date);
CREATE INDEX IF NOT EXISTS idx_barriers_note      ON barrier_levels(note_id);
CREATE INDEX IF NOT EXISTS idx_barriers_type      ON barrier_levels(barrier_type);
CREATE INDEX IF NOT EXISTS idx_barriers_status    ON barrier_levels(status);
"""


# ─── DB Manager ──────────────────────────────────────────────────────────────

class StructuredNotesDB:
    """Gestisce il database SQLite delle note strutturate.

    Args:
        db_path: percorso al file SQLite (default da settings.yaml).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings()
        self._path = Path(db_path or cfg["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager per la connessione SQLite con WAL mode.

        Yields:
            sqlite3.Connection: connessione attiva.
        """
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Crea le tabelle, gli indici e applica le migrazioni incrementali."""
        with self._conn() as conn:
            conn.executescript(_DDL_NOTES + _DDL_BARRIERS + _DDL_IDX)
            self._migrate(conn)
        _log.info("Schema inizializzato: %s", self._path)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Applica migrazioni incrementali basate su PRAGMA user_version.

        Versioni:
          0 → 1: schema iniziale (nessuna modifica necessaria, solo bump version)
        """
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            # v1: schema attuale — nessuna ALTER TABLE necessaria
            conn.execute("PRAGMA user_version = 1")
            _log.debug("DB migrato a versione 1")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _d(d: Optional[date]) -> Optional[str]:
        """Converte date → stringa ISO."""
        return d.isoformat() if d else None

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    # ─── Note CRUD ───────────────────────────────────────────────────────────

    def upsert_note(self, note: StructuredNote) -> int:
        """Inserisce o aggiorna una nota strutturata nel DB.

        Se esiste già un record con lo stesso filing_url viene aggiornato.

        Args:
            note: dataclass StructuredNote da salvare.

        Returns:
            int: id del record inserito/aggiornato.
        """
        obs_json = json.dumps(note.observation_dates) if note.observation_dates else "[]"
        now      = self._now()

        with self._conn() as conn:
            # Controlla se esiste già
            existing = conn.execute(
                "SELECT id FROM notes WHERE filing_url = ?", (note.filing_url,)
            ).fetchone()

            if existing:
                note_id = existing["id"]
                conn.execute(
                    """
                    UPDATE notes SET
                        issuer=?, issue_date=?, maturity_date=?, notional_usd=?,
                        product_type=?, underlying=?, initial_level=?,
                        autocall_trigger_pct=?, knockin_barrier_pct=?,
                        buffer_pct=?, participation_rate=?, coupon_rate=?,
                        observation_dates=?, raw_text=?
                    WHERE id=?
                    """,
                    (
                        note.issuer, self._d(note.issue_date), self._d(note.maturity_date),
                        note.notional_usd, note.product_type, note.underlying,
                        note.initial_level, note.autocall_trigger_pct, note.knockin_barrier_pct,
                        note.buffer_pct, note.participation_rate, note.coupon_rate,
                        obs_json, note.raw_text, note_id,
                    ),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO notes
                        (filing_url, issuer, issue_date, maturity_date, notional_usd,
                         product_type, underlying, initial_level, autocall_trigger_pct,
                         knockin_barrier_pct, buffer_pct, participation_rate, coupon_rate,
                         observation_dates, raw_text, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        note.filing_url, note.issuer, self._d(note.issue_date),
                        self._d(note.maturity_date), note.notional_usd, note.product_type,
                        note.underlying, note.initial_level, note.autocall_trigger_pct,
                        note.knockin_barrier_pct, note.buffer_pct, note.participation_rate,
                        note.coupon_rate, obs_json, note.raw_text, now,
                    ),
                )
                note_id = cur.lastrowid

            # Salva i barrier levels (rimuovi i vecchi e reinserisci)
            conn.execute("DELETE FROM barrier_levels WHERE note_id=?", (note_id,))
            for b in note.barriers:
                conn.execute(
                    """
                    INSERT INTO barrier_levels
                        (note_id, barrier_type, level_pct, level_price_ibit,
                         level_price_btc, observation_date, status, created_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        note_id, b.barrier_type, b.level_pct, b.level_price_ibit,
                        b.level_price_btc, self._d(b.observation_date),
                        b.status, now,
                    ),
                )

        _log.debug("Nota salvata: id=%d url=%s barriers=%d", note_id, note.filing_url, len(note.barriers))
        return note_id

    def upsert_notes(self, notes: list[StructuredNote]) -> list[int]:
        """Salva una lista di note nel DB.

        Args:
            notes: lista di StructuredNote.

        Returns:
            list[int]: id dei record salvati.
        """
        ids = [self.upsert_note(n) for n in notes]
        _log.info("Salvate %d note nel DB", len(ids))
        return ids

    def get_note_by_url(self, url: str) -> Optional[StructuredNote]:
        """Recupera una nota per filing URL.

        Args:
            url: URL del filing.

        Returns:
            StructuredNote | None.
        """
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM notes WHERE filing_url=?", (url,)).fetchone()
            if not row:
                return None
            return self._row_to_note(conn, dict(row))

    def get_all_notes(self) -> list[StructuredNote]:
        """Recupera tutte le note dal DB.

        Returns:
            list[StructuredNote].
        """
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM notes ORDER BY issue_date DESC").fetchall()
            return [self._row_to_note(conn, dict(r)) for r in rows]

    def get_active_barriers(self) -> list[dict]:
        """Restituisce tutti i barrier levels con status='active'.

        Returns:
            list[dict]: barrier levels con dati dalla nota associata.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT b.*, n.issuer, n.product_type, n.maturity_date,
                       n.initial_level, n.filing_url
                FROM barrier_levels b
                JOIN notes n ON b.note_id = n.id
                WHERE b.status = 'active'
                ORDER BY b.level_price_ibit ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Barrier price conversion ─────────────────────────────────────────────

    def compute_btc_prices(self, ibit_btc_ratio: float) -> int:
        """Calcola i prezzi BTC per tutte le barriere che non ce l'hanno.

        Args:
            ibit_btc_ratio: rapporto IBIT/BTC (es. 0.001 se 1 IBIT ≈ 0.001 BTC).

        Returns:
            int: numero di barriere aggiornate.
        """
        updated = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, level_price_ibit FROM barrier_levels WHERE level_price_btc IS NULL"
            ).fetchall()
            for row in rows:
                if row["level_price_ibit"]:
                    btc_price = row["level_price_ibit"] / ibit_btc_ratio
                    conn.execute(
                        "UPDATE barrier_levels SET level_price_btc=? WHERE id=?",
                        (btc_price, row["id"]),
                    )
                    updated += 1
        _log.info("Aggiornati prezzi BTC per %d barriere", updated)
        return updated

    def update_barrier_statuses(self, current_ibit_price: float) -> dict[str, int]:
        """Aggiorna lo status delle barriere in base al prezzo corrente.

        Una barriera è 'triggered' se il prezzo IBIT corrente ha attraversato
        il livello della barriera. La logica dipende dal tipo:
          - knock_in: triggered se current ≤ level (prezzo sceso sotto)
          - autocall:  triggered se current ≥ level (prezzo salito sopra)
          - buffer:    triggered se current ≤ level
          - knock_out: triggered se current ≥ level

        Args:
            current_ibit_price: prezzo corrente di IBIT.

        Returns:
            dict con conteggi: {"triggered": N, "reactivated": M}.
        """
        counts = {"triggered": 0, "reactivated": 0}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, barrier_type, level_price_ibit, status FROM barrier_levels "
                "WHERE level_price_ibit IS NOT NULL"
            ).fetchall()

            for row in rows:
                btype = row["barrier_type"]
                level = row["level_price_ibit"]
                old_status = row["status"]

                if btype in ("knock_in", "buffer"):
                    new_triggered = current_ibit_price <= level
                else:  # autocall, knock_out
                    new_triggered = current_ibit_price >= level

                if new_triggered and old_status == "active":
                    conn.execute(
                        "UPDATE barrier_levels SET status='triggered' WHERE id=?", (row["id"],)
                    )
                    counts["triggered"] += 1
                elif not new_triggered and old_status == "triggered":
                    conn.execute(
                        "UPDATE barrier_levels SET status='active' WHERE id=?", (row["id"],)
                    )
                    counts["reactivated"] += 1

        _log.info("Barrier status aggiornati: %s (prezzo IBIT=%.2f)", counts, current_ibit_price)
        return counts

    # ─── Helpers privati ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_note(conn: sqlite3.Connection, row: dict) -> StructuredNote:
        """Converte una riga del DB in StructuredNote con barriers.

        Args:
            conn: connessione SQLite attiva.
            row: dict dalla tabella notes.

        Returns:
            StructuredNote: con barriers popolati.
        """
        def _date(s: Optional[str]) -> Optional[date]:
            return date.fromisoformat(s) if s else None

        barrier_rows = conn.execute(
            "SELECT * FROM barrier_levels WHERE note_id=?", (row["id"],)
        ).fetchall()

        barriers = [
            BarrierLevel(
                barrier_type=b["barrier_type"],
                level_pct=b["level_pct"],
                level_price_ibit=b["level_price_ibit"],
                level_price_btc=b["level_price_btc"],
                observation_date=_date(b["observation_date"]),
                status=b["status"],
                note_id=b["note_id"],
                id=b["id"],
            )
            for b in barrier_rows
        ]

        obs = json.loads(row.get("observation_dates") or "[]")

        return StructuredNote(
            id=row["id"],
            filing_url=row["filing_url"],
            issuer=row["issuer"],
            issue_date=_date(row["issue_date"]),
            maturity_date=_date(row["maturity_date"]),
            notional_usd=row["notional_usd"],
            product_type=row["product_type"],
            underlying=row.get("underlying", "IBIT"),
            initial_level=row["initial_level"],
            autocall_trigger_pct=row["autocall_trigger_pct"],
            knockin_barrier_pct=row["knockin_barrier_pct"],
            buffer_pct=row["buffer_pct"],
            participation_rate=row["participation_rate"],
            coupon_rate=row["coupon_rate"],
            observation_dates=obs,
            barriers=barriers,
        )

    # ─── Stats ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Restituisce statistiche aggregate sul DB.

        Returns:
            dict: conteggi e aggregati principali.
        """
        with self._conn() as conn:
            total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            total_barriers = conn.execute("SELECT COUNT(*) FROM barrier_levels").fetchone()[0]
            active_barriers = conn.execute(
                "SELECT COUNT(*) FROM barrier_levels WHERE status='active'"
            ).fetchone()[0]
            total_notional = conn.execute(
                "SELECT SUM(notional_usd) FROM notes WHERE notional_usd IS NOT NULL"
            ).fetchone()[0] or 0
            by_type = dict(conn.execute(
                "SELECT product_type, COUNT(*) FROM notes GROUP BY product_type"
            ).fetchall())
            by_issuer = dict(conn.execute(
                "SELECT issuer, COUNT(*) FROM notes GROUP BY issuer"
            ).fetchall())

        return {
            "total_notes":       total_notes,
            "total_barriers":    total_barriers,
            "active_barriers":   active_barriers,
            "total_notional_usd": total_notional,
            "by_product_type":   by_type,
            "by_issuer":         by_issuer,
        }
