"""Database SQLite per gli snapshot GEX storici.

Persiste uno snapshot GEX per giorno e per asset (UPSERT su date+asset), consentendo
al backtest e al regime analysis di operare su serie storiche reali
invece di un singolo punto live.

Schema:
  - gex_snapshots: una riga per giorno di trading e per asset (BTC, ETH).

Lo schema pre-ETH aveva ``date UNIQUE`` e nessuna colonna asset: SQLite non
permette di togliere un vincolo UNIQUE, quindi ``_ensure_table`` ricostruisce la
tabella una volta sola, marcando le righe esistenti come BTC.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.config import setup_logging
from src.gex.models import GexSnapshot

_log = setup_logging("gex.db")

_VERSIONED_DB = Path(__file__).resolve().parent.parent.parent / "data" / "structured_notes.db"

_DDL = """
CREATE TABLE IF NOT EXISTS gex_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL,          -- YYYY-MM-DD
    asset            TEXT    NOT NULL DEFAULT 'BTC',  -- BTC | ETH
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
    created_at       TEXT    NOT NULL,
    UNIQUE (date, asset)
);
CREATE INDEX IF NOT EXISTS idx_gex_date ON gex_snapshots(date);
"""

_COLUMNS = (
    "date, timestamp, spot_price, total_net_gex, gamma_flip_price, put_wall, "
    "call_wall, max_pain, regime, total_call_oi, total_put_oi, put_call_ratio, "
    "n_instruments, created_at"
)


class GexDB:
    """Gestisce il database SQLite degli snapshot GEX.

    Segue il pattern di PriceFetcher / StructuredNotesDB:
      - connessione WAL per concorrenza API + script
      - UPSERT su (date) per idempotenza dei cron job

    Args:
        db_path: percorso al file SQLite (default da settings.yaml).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path or _VERSIONED_DB)
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
        """Crea tabella e indice se non esistono, migrando lo schema pre-ETH (idempotente)."""
        with self._conn() as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(gex_snapshots)")}
            if cols and "asset" not in cols:
                self._migrate_add_asset(conn)
            conn.executescript(_DDL)

    @staticmethod
    def _migrate_add_asset(conn: sqlite3.Connection) -> None:
        """Ricostruisce la tabella con UNIQUE(date, asset); le righe esistenti sono BTC.

        Un solo script in BEGIN/COMMIT: se la copia fallisce non resta una tabella
        a metà (``executescript`` farebbe commit implicito tra uno statement e l'altro).
        """
        n = conn.execute("SELECT COUNT(*) FROM gex_snapshots").fetchone()[0]
        conn.commit()
        conn.executescript(
            "BEGIN;"
            "DROP INDEX IF EXISTS idx_gex_date;"
            "ALTER TABLE gex_snapshots RENAME TO gex_snapshots_legacy;"
            + _DDL
            + f"INSERT INTO gex_snapshots (id, asset, {_COLUMNS}) "
            f"SELECT id, 'BTC', {_COLUMNS} FROM gex_snapshots_legacy;"
            "DROP TABLE gex_snapshots_legacy;"
            "COMMIT;"
        )
        _log.info("gex_snapshots migrata a schema multi-asset: %d righe marcate BTC", n)

    # ─── Write ───────────────────────────────────────────────────────────────

    def insert_snapshot(self, snapshot: GexSnapshot, regime: str, asset: str = "BTC") -> None:
        """Salva o aggiorna lo snapshot GEX del giorno corrente per l'asset.

        Usa UPSERT su (date, asset): se oggi è già presente, aggiorna tutti i campi.
        Sicuro da chiamare più volte nello stesso giorno (cron ogni 4h).

        Args:
            snapshot: GexSnapshot appena calcolato da Deribit.
            regime: stringa regime da RegimeDetector ('positive_gamma' ecc.).
            asset: "BTC" o "ETH".
        """
        today   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO gex_snapshots
                    (date, asset, timestamp, spot_price, total_net_gex, gamma_flip_price,
                     put_wall, call_wall, max_pain, regime, total_call_oi,
                     total_put_oi, put_call_ratio, n_instruments, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(date, asset) DO UPDATE SET
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
                    asset,
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
            "GEX snapshot salvato: asset=%s date=%s spot=%.0f gex=%.1fM regime=%s",
            asset, today, snapshot.spot_price, snapshot.total_net_gex / 1e6, regime,
        )

    # ─── Read ────────────────────────────────────────────────────────────────

    def get_series(self, days: int = 365, asset: str = "BTC") -> pd.Series:
        """Restituisce la serie storica del GEX totale dell'asset.

        Args:
            days: numero di giorni passati da includere.
            asset: "BTC" o "ETH".

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
                WHERE date >= date('now', ? || ' days') AND asset = ?
                ORDER BY date ASC
                """,
                (f"-{days}", asset),
            ).fetchall()

        if not rows:
            return pd.Series(dtype=float, name="total_net_gex")

        # tz-naive per compatibilità con i DataFrame di FlowCorrelation (tz-naive)
        index  = pd.to_datetime([r["date"] for r in rows])
        values = [r["total_net_gex"] for r in rows]
        return pd.Series(values, index=index, name="total_net_gex")

    def get_latest_n(self, n: int = 90, asset: str = "BTC") -> list[GexSnapshot]:
        """Restituisce gli ultimi N snapshot come oggetti GexSnapshot.

        Usato per pre-popolare RegimeDetector._history al boot.

        Args:
            n: numero massimo di snapshot da restituire.
            asset: "BTC" o "ETH".

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
                WHERE asset = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (asset, n),
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

    def count(self, asset: str = "BTC") -> int:
        """Conta gli snapshot dell'asset nel DB."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM gex_snapshots WHERE asset = ?", (asset,)
            ).fetchone()[0]

    def get_last_regime_label(self, asset: str = "BTC") -> Optional[str]:
        """Restituisce il regime dell'ultimo snapshot dell'asset, o None se assente."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT regime FROM gex_snapshots WHERE asset = ? ORDER BY date DESC LIMIT 1",
                (asset,),
            ).fetchone()
        return row["regime"] if row else None
