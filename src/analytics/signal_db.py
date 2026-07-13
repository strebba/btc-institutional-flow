"""Persistenza SQLite per i segnali generati dalla strategia multi-fattore.

Tabella signal_history: una riga per calcolo, con UNIQUE su timestamp.
INSERT OR IGNORE garantisce idempotenza se lo stesso timestamp viene
inserito più volte (es. API cache miss ravvicinati).
Stessa strategia di ifi_db.py: WAL mode, path da settings.yaml.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.analytics.factor_scorers import SignalResult
from src.config import get_settings, setup_logging

_log = setup_logging("analytics.signal_db")

_DDL = """
CREATE TABLE IF NOT EXISTS signal_history (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT NOT NULL UNIQUE,  -- ISO UTC secondi, chiave di dedup
    date                 TEXT NOT NULL,          -- YYYY-MM-DD
    score                REAL NOT NULL,          -- 0-100
    signal               TEXT NOT NULL,          -- LONG|CAUTION|RISK_OFF
    signal_reason        TEXT,
    comp_gex             REAL,                   -- componenti normalizzate 0-1
    comp_etf_flow        REAL,
    comp_funding_rate    REAL,
    comp_oi_change       REAL,
    comp_long_short      REAL,
    comp_put_call        REAL,
    comp_liquidations    REAL,
    spot_price_usd       REAL,                   -- input grezzi
    total_gex_usd        REAL,
    ibit_flow_3d_usd     REAL,
    funding_rate_pct     REAL,
    oi_change_7d_pct     REAL,
    long_short_ratio     REAL,
    put_call_ratio       REAL,
    liq_long_usd         REAL,
    liq_short_usd        REAL,
    near_active_barrier  INTEGER,                -- 0/1
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signal_ts   ON signal_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_date ON signal_history(date);
"""


class SignalDB:
    """Gestisce la persistenza della serie storica dei segnali su SQLite."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings()
        self._path = Path(db_path or cfg["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ─── Write ────────────────────────────────────────────────────────────────

    def insert(
        self,
        result: SignalResult,
        *,
        timestamp: str | None = None,
        spot_price_usd: Optional[float] = None,
        total_gex_usd: Optional[float] = None,
        ibit_flow_3d_usd: Optional[float] = None,
        funding_rate_pct: Optional[float] = None,
        oi_change_7d_pct: Optional[float] = None,
        long_short_ratio: Optional[float] = None,
        put_call_ratio: Optional[float] = None,
        liq_long_usd: Optional[float] = None,
        liq_short_usd: Optional[float] = None,
        near_active_barrier: bool = False,
    ) -> bool:
        """Inserisce un segnale. Restituisce True se inserito, False se già presente."""
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        date = ts[:10]
        c = result.components

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO signal_history (
                    timestamp, date, score, signal, signal_reason,
                    comp_gex, comp_etf_flow, comp_funding_rate, comp_oi_change,
                    comp_long_short, comp_put_call, comp_liquidations,
                    spot_price_usd, total_gex_usd, ibit_flow_3d_usd,
                    funding_rate_pct, oi_change_7d_pct, long_short_ratio,
                    put_call_ratio, liq_long_usd, liq_short_usd, near_active_barrier
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts, date, round(result.score, 2), result.signal, result.reason,
                    _safe(c.get("gex")), _safe(c.get("etf_flow")),
                    _safe(c.get("funding_rate")), _safe(c.get("oi_change")),
                    _safe(c.get("long_short")), _safe(c.get("put_call")),
                    _safe(c.get("liquidations")),
                    _safe(spot_price_usd), _safe(total_gex_usd), _safe(ibit_flow_3d_usd),
                    _safe(funding_rate_pct), _safe(oi_change_7d_pct),
                    _safe(long_short_ratio), _safe(put_call_ratio),
                    _safe(liq_long_usd), _safe(liq_short_usd),
                    1 if near_active_barrier else 0,
                ),
            )
            return cur.rowcount == 1

    # ─── Read ─────────────────────────────────────────────────────────────────

    def get_latest(self, n: int = 10) -> list[dict]:
        """Restituisce gli ultimi n segnali, dal più recente."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, date, score, signal, signal_reason,
                       comp_gex, comp_etf_flow, comp_funding_rate, comp_oi_change,
                       comp_long_short, comp_put_call, comp_liquidations,
                       spot_price_usd, total_gex_usd, ibit_flow_3d_usd,
                       funding_rate_pct, oi_change_7d_pct, long_short_ratio,
                       put_call_ratio, liq_long_usd, liq_short_usd, near_active_barrier
                FROM signal_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_series(self, days: int = 90) -> pd.DataFrame:
        """Restituisce la serie storica come DataFrame (index = timestamp).

        Colonne: date, score, signal, spot_price_usd, total_gex_usd, …
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, date, score, signal, signal_reason,
                       spot_price_usd, total_gex_usd, ibit_flow_3d_usd,
                       funding_rate_pct, oi_change_7d_pct, long_short_ratio,
                       put_call_ratio, liq_long_usd, liq_short_usd, near_active_barrier,
                       comp_gex, comp_etf_flow, comp_funding_rate, comp_oi_change,
                       comp_long_short, comp_put_call, comp_liquidations
                FROM signal_history
                WHERE date >= date('now', ? || ' days')
                ORDER BY timestamp ASC
                """,
                (f"-{days}",),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe(v: Optional[float]) -> Optional[float]:
    import math
    if v is None:
        return None
    try:
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    except (TypeError, ValueError):
        return None
