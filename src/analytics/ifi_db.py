"""Persistenza SQLite per l'Institutional Flow Index.

Tabella ifi_history: una riga per giorno, UPSERT-safe (idempotente per il cron).
Stessa strategia di gex_db.py: WAL mode, path da settings.yaml.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.config import get_settings, setup_logging

_log = setup_logging("analytics.ifi_db")

_DDL = """
CREATE TABLE IF NOT EXISTS ifi_history (
    date              TEXT PRIMARY KEY,   -- YYYY-MM-DD
    score             REAL NOT NULL,      -- IFI 0-100
    regime            TEXT NOT NULL,      -- Accumulation|Momentum|Neutral|Distribution|Outflow
    flow_score        REAL,              -- componente flow_momentum (0-1)
    trend_score       REAL,              -- componente flow_trend (0-1)
    price_score       REAL,              -- componente price_momentum (0-1)
    funding_score     REAL,              -- componente funding (0-1), null se non disponibile
    oi_score          REAL,              -- componente oi_momentum (0-1), null se non disponibile
    btc_price         REAL,              -- prezzo BTC USD del giorno
    total_flow_usd    REAL,              -- flusso ETF totale USD del giorno
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ifi_date ON ifi_history(date);
"""


class IFIDb:
    """Gestisce la persistenza della serie storica IFI su SQLite."""

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

    def upsert(
        self,
        date: str,
        score: float,
        regime: str,
        flow_score:    Optional[float] = None,
        trend_score:   Optional[float] = None,
        price_score:   Optional[float] = None,
        funding_score: Optional[float] = None,
        oi_score:      Optional[float] = None,
        btc_price:     Optional[float] = None,
        total_flow_usd: Optional[float] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ifi_history
                    (date, score, regime, flow_score, trend_score, price_score,
                     funding_score, oi_score, btc_price, total_flow_usd, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(date) DO UPDATE SET
                    score=excluded.score,
                    regime=excluded.regime,
                    flow_score=coalesce(excluded.flow_score, flow_score),
                    trend_score=coalesce(excluded.trend_score, trend_score),
                    price_score=coalesce(excluded.price_score, price_score),
                    funding_score=coalesce(excluded.funding_score, funding_score),
                    oi_score=coalesce(excluded.oi_score, oi_score),
                    btc_price=coalesce(excluded.btc_price, btc_price),
                    total_flow_usd=coalesce(excluded.total_flow_usd, total_flow_usd),
                    created_at=excluded.created_at
                """,
                (date, round(score, 2), regime,
                 _safe(flow_score), _safe(trend_score), _safe(price_score),
                 _safe(funding_score), _safe(oi_score),
                 _safe(btc_price), _safe(total_flow_usd), now),
            )

    def upsert_series(
        self,
        scores: pd.Series,
        factor_df: Optional[pd.DataFrame],
        btc_prices: Optional[pd.Series],
        flows: Optional[pd.Series],
    ) -> int:
        """Upsert di una serie di score (DatetimeIndex). Restituisce il numero di righe salvate."""
        from src.analytics.ifi import regime_label

        count = 0
        for ts, score in scores.items():
            date = str(ts.date()) if hasattr(ts, "date") else str(ts)
            regime = regime_label(float(score))

            btc_price    = _lookup(btc_prices, ts)
            total_flow   = _lookup(flows, ts)
            flow_score   = _factor(factor_df, ts, "flow_momentum")
            trend_score  = _factor(factor_df, ts, "flow_trend")
            price_score  = _factor(factor_df, ts, "price_momentum")
            funding_score= _factor(factor_df, ts, "funding")
            oi_score     = _factor(factor_df, ts, "oi_momentum")

            self.upsert(
                date=date, score=float(score), regime=regime,
                flow_score=flow_score, trend_score=trend_score,
                price_score=price_score, funding_score=funding_score,
                oi_score=oi_score, btc_price=btc_price,
                total_flow_usd=total_flow,
            )
            count += 1

        return count

    # ─── Read ─────────────────────────────────────────────────────────────────

    def get_series(self, days: int = 500) -> pd.DataFrame:
        """Restituisce la serie storica IFI come DataFrame (DatetimeIndex).

        Colonne: score, regime, btc_price, total_flow_usd
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT date, score, regime, btc_price, total_flow_usd
                FROM ifi_history
                ORDER BY date DESC
                LIMIT ?
                """,
                (days,),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df

    def get_latest(self) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT date, score, regime, btc_price, total_flow_usd "
                "FROM ifi_history ORDER BY date DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM ifi_history").fetchone()[0]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe(v: Optional[float]) -> Optional[float]:
    """Converte NaN/inf in None per SQLite."""
    import math
    if v is None:
        return None
    try:
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    except (TypeError, ValueError):
        return None


def _lookup(series: Optional[pd.Series], ts) -> Optional[float]:
    if series is None:
        return None
    try:
        v = series.get(ts) if hasattr(series, "get") else None
        if v is None and ts in series.index:
            v = series[ts]
        return _safe(float(v)) if v is not None else None
    except Exception:
        return None


def _factor(factor_df: Optional[pd.DataFrame], ts, col: str) -> Optional[float]:
    if factor_df is None or col not in factor_df.columns:
        return None
    try:
        if ts in factor_df.index:
            return _safe(float(factor_df.loc[ts, col]))
    except Exception:
        pass
    return None
