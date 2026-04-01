"""Database SQLite per gli snapshot GEX storici.

Persiste uno snapshot GEX per giorno (UPSERT su date), consentendo
al backtest e al regime analysis di operare su serie storiche reali
invece di un singolo punto live.

Schema:
  - gex_snapshots: una riga per giorno di trading.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.config import get_settings, setup_logging
from src.gex.models import GexByStrike, GexSnapshot

_log = setup_logging("gex.db")

_DDL = """
CREATE TABLE IF NOT EXISTS gex_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL UNIQUE,   -- YYYY-MM-DD, chiave primaria logica
    timestamp        TEXT    NOT NULL,          -- ISO datetime del calcolo
    spot_price       REAL    NOT NULL,
    total_net_gex    REAL    NOT NULL,          -- USD raw (es. 450_000_000)
    gamma_flip_price REAL,
    put_wall         REAL,
    call_wall        REAL,
    max_pain         REAL,
    regime           TEXT,                      -- positive_gamma|negative_gamma|neutral
    total_call_oi    REAL,
    total_put_oi     REAL,
    put_call_ratio   REAL,
    n_instruments    INTEGER,
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gex_date ON gex_snapshots(date);
"""


class GexDB:
    """Gestisce il database SQLite degli snapshot GEX.

    Segue il pattern di PriceFetcher / StructuredNotesDB:
      - connessione WAL per concorrenza API + script
      - UPSERT su (date) per idempotenza dei cron job

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
        """Context manager WAL con commit/rollback automatico."""
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
        """Crea tabella e indice se non esistono (idempotente)."""
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ─── Write ───────────────────────────────────────────────────────────────

    def insert_snapshot(self, snapshot: GexSnapshot, regime: str) -> None:
        """Salva o aggiorna lo snapshot GEX del giorno corrente.

        Usa UPSERT su (date): se oggi è già presente, aggiorna tutti i campi.
        Sicuro da chiamare più volte nello stesso giorno (cron ogni 4h).

        Args:
            snapshot: GexSnapshot appena calcolato da Deribit.
            regime: stringa regime da RegimeDetector ('positive_gamma' ecc.).
        """
        today   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO gex_snapshots
                    (date, timestamp, spot_price, total_net_gex, gamma_flip_price,
                     put_wall, call_wall, max_pain, regime, total_call_oi,
                     total_put_oi, put_call_ratio, n_instruments, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(date) DO UPDATE SET
                    timestamp        = excluded.timestamp,
                    spot_price       = excluded.spot_price,
                    total_net_gex    = excluded.total_net_gex,
                    gamma_flip_price = excluded.gamma_flip_price,
                    put_wall         = excluded.put_wall,
                    call_wall        = excluded.call_wall,
                    max_pain         = excluded.max_pain,
                    regime           = excluded.regime,
                    total_call_oi    = excluded.total_call_oi,
                    total_put_oi     = excluded.total_put_oi,
                    put_call_ratio   = excluded.put_call_ratio,
                    n_instruments    = excluded.n_instruments
                """,
                (
                    today,
                    snapshot.timestamp.isoformat(),
                    snapshot.spot_price,
                    snapshot.total_net_gex,
                    snapshot.gamma_flip_price,
                    snapshot.put_wall,
                    snapshot.call_wall,
                    snapshot.max_pain,
                    regime,
                    snapshot.total_call_oi,
                    snapshot.total_put_oi,
                    snapshot.put_call_ratio,
                    len(snapshot.gex_by_strike),
                    now_iso,
                ),
            )

        _log.info(
            "GEX snapshot salvato: date=%s spot=%.0f gex=%.1fM regime=%s",
            today, snapshot.spot_price, snapshot.total_net_gex / 1e6, regime,
        )

    # ─── Read ────────────────────────────────────────────────────────────────

    def get_series(self, days: int = 365) -> pd.Series:
        """Restituisce la serie storica del GEX totale.

        Args:
            days: numero di giorni passati da includere.

        Returns:
            pd.Series con DatetimeIndex (UTC, normalizzato a mezzanotte)
            e valori float (total_net_gex in USD raw).
            Serie vuota se non ci sono dati nel DB.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT date, total_net_gex
                FROM gex_snapshots
                WHERE date >= date('now', ? || ' days')
                ORDER BY date ASC
                """,
                (f"-{days}",),
            ).fetchall()

        if not rows:
            return pd.Series(dtype=float, name="total_net_gex")

        # tz-naive per compatibilità con i DataFrame di FlowCorrelation (tz-naive)
        index  = pd.to_datetime([r["date"] for r in rows])
        values = [r["total_net_gex"] for r in rows]
        return pd.Series(values, index=index, name="total_net_gex")

    def get_latest_n(self, n: int = 90) -> list[GexSnapshot]:
        """Restituisce gli ultimi N snapshot come oggetti GexSnapshot.

        Usato per pre-popolare RegimeDetector._history al boot.

        Args:
            n: numero massimo di snapshot da restituire.

        Returns:
            list[GexSnapshot] ordinata per data crescente.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, spot_price, total_net_gex, gamma_flip_price,
                       put_wall, call_wall, max_pain, total_call_oi, total_put_oi,
                       put_call_ratio
                FROM gex_snapshots
                ORDER BY date DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()

        snapshots = []
        for r in reversed(rows):  # riordina crescente
            try:
                ts = datetime.fromisoformat(r["timestamp"])
            except Exception:
                ts = datetime.now(tz=timezone.utc)
            snapshots.append(
                GexSnapshot(
                    timestamp        = ts,
                    spot_price       = r["spot_price"],
                    total_net_gex    = r["total_net_gex"],
                    gamma_flip_price = r["gamma_flip_price"],
                    put_wall         = r["put_wall"],
                    call_wall        = r["call_wall"],
                    max_pain         = r["max_pain"],
                    total_call_oi    = r["total_call_oi"] or 0.0,
                    total_put_oi     = r["total_put_oi"] or 0.0,
                    put_call_ratio   = r["put_call_ratio"],
                )
            )
        return snapshots

    def get_all_for_regime(self) -> pd.DataFrame:
        """Restituisce tutti gli snapshot come DataFrame per RegimeAnalysis.

        Returns:
            pd.DataFrame con colonne: date (DatetimeIndex), total_net_gex, regime.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, total_net_gex, regime FROM gex_snapshots ORDER BY date ASC"
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["total_net_gex", "regime"])

        index  = pd.to_datetime([r["date"] for r in rows])  # tz-naive
        df = pd.DataFrame(
            {
                "total_net_gex": [r["total_net_gex"] for r in rows],
                "regime":        [r["regime"] for r in rows],
            },
            index=index,
        )
        return df

    def count(self) -> int:
        """Conta il numero totale di snapshot nel DB."""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM gex_snapshots").fetchone()[0]
