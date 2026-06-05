"""Persistenza SQLite del forecast spine.

Tre tabelle:
- `predictions`     — una riga per previsione (UNIQUE su created_at+source+target_type+horizon_days
                      per idempotenza: rilanciare cron_predict nello stesso istante non duplica).
- `outcomes`        — una riga per esito (UNIQUE su prediction_id).
- `weight_versions` — versioni dei pesi del SignalModel (audit trail del self-learning).

Stessa strategia di src/analytics/signal_db.py: WAL mode, path da settings.yaml,
INSERT OR IGNORE per idempotenza.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from src.config import get_settings, setup_logging
from src.forecast.models import Outcome, Prediction, STATUS_OPEN, STATUS_SCORED

_log = setup_logging("forecast.prediction_db")

_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,          -- ISO UTC secondi
    date             TEXT NOT NULL,          -- YYYY-MM-DD
    source           TEXT NOT NULL,
    asset            TEXT NOT NULL,
    target_type      TEXT NOT NULL,          -- direction | level | prob
    target_spec      TEXT NOT NULL,          -- JSON
    horizon_days     INTEGER NOT NULL,
    confidence       REAL NOT NULL,          -- 0-1
    rationale        TEXT,
    counter_analysis TEXT,
    human_overlay    TEXT,
    score_ref        REAL,
    components       TEXT,                   -- JSON componenti normalizzate
    weights_version  INTEGER,
    status           TEXT NOT NULL DEFAULT 'open',
    UNIQUE(created_at, source, target_type, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_pred_date   ON predictions(date);
CREATE INDEX IF NOT EXISTS idx_pred_source ON predictions(source);

CREATE TABLE IF NOT EXISTS outcomes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id    INTEGER NOT NULL UNIQUE,
    hit              INTEGER NOT NULL,       -- 0/1
    realized_return  REAL,
    realized_price   REAL,
    ref_price        REAL,
    signed_error     REAL,
    brier            REAL,
    detail           TEXT,
    scored_at        TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);
CREATE INDEX IF NOT EXISTS idx_outcome_pred ON outcomes(prediction_id);

CREATE TABLE IF NOT EXISTS weight_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    source      TEXT NOT NULL,
    weights     TEXT NOT NULL,              -- JSON {component: weight}
    active      INTEGER NOT NULL DEFAULT 0, -- 1 = versione attiva per la source
    rationale   TEXT
);
CREATE INDEX IF NOT EXISTS idx_wv_source_active ON weight_versions(source, active);
"""

_PRED_COLS = (
    "created_at, date, source, asset, target_type, target_spec, horizon_days, "
    "confidence, rationale, counter_analysis, human_overlay, score_ref, "
    "components, weights_version, status"
)


class PredictionDB:
    """Persistenza di predizioni, esiti e versioni dei pesi."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings()
        self._path = Path(db_path or cfg["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ─── Predictions ────────────────────────────────────────────────────────

    def insert_prediction(self, pred: Prediction) -> Optional[int]:
        """Inserisce una previsione. Restituisce l'id, o None se duplicata."""
        row = pred.to_row()
        placeholders = ",".join("?" for _ in _PRED_COLS.split(","))
        with self._conn() as conn:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO predictions ({_PRED_COLS}) VALUES ({placeholders})",
                tuple(row[c.strip()] for c in _PRED_COLS.split(",")),
            )
            if cur.rowcount == 1:
                return int(cur.lastrowid)
        return None

    def update_human_fields(
        self,
        prediction_id: int,
        *,
        counter_analysis: Optional[str] = None,
        human_overlay: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Aggiorna i campi popolati nel daily-review (overlay umano e contro-analisi)."""
        sets, vals = [], []
        if counter_analysis is not None:
            sets.append("counter_analysis = ?"); vals.append(counter_analysis)
        if human_overlay is not None:
            sets.append("human_overlay = ?"); vals.append(human_overlay)
        if confidence is not None:
            sets.append("confidence = ?"); vals.append(float(max(0.0, min(1.0, confidence))))
        if not sets:
            return
        vals.append(prediction_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE predictions SET {', '.join(sets)} WHERE id = ?", vals)

    def get_open(self) -> list[Prediction]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE status = ? ORDER BY created_at ASC",
                (STATUS_OPEN,),
            ).fetchall()
        return [Prediction.from_row(dict(r)) for r in rows]

    def get_due(self, asof: Optional[datetime] = None) -> list[Prediction]:
        """Predizioni open il cui orizzonte è maturo (verificabili adesso)."""
        return [p for p in self.get_open() if p.is_due(asof)]

    def get_recent(self, limit: int = 50) -> list[Prediction]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Prediction.from_row(dict(r)) for r in rows]

    def get_with_outcomes(self, days: int = 180, source: Optional[str] = None) -> list[dict]:
        """Join predictions+outcomes (left) per la calibrazione e i report."""
        q = (
            "SELECT p.*, o.hit, o.realized_return, o.realized_price, o.signed_error, "
            "       o.brier, o.detail AS outcome_detail, o.scored_at "
            "FROM predictions p LEFT JOIN outcomes o ON o.prediction_id = p.id "
            "WHERE p.date >= date('now', ? || ' days') "
        )
        params: list[Any] = [f"-{days}"]
        if source:
            q += "AND p.source = ? "
            params.append(source)
        q += "ORDER BY p.created_at ASC"
        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    # ─── Outcomes ─────────────────────────────────────────────────────────────

    def insert_outcome(self, outcome: Outcome) -> bool:
        """Inserisce un esito e marca la predizione come scored. True se inserito."""
        row = outcome.to_row()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO outcomes (
                    prediction_id, hit, realized_return, realized_price, ref_price,
                    signed_error, brier, detail, scored_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["prediction_id"], row["hit"], row["realized_return"],
                    row["realized_price"], row["ref_price"], row["signed_error"],
                    row["brier"], row["detail"], row["scored_at"],
                ),
            )
            inserted = cur.rowcount == 1
            if inserted:
                conn.execute(
                    "UPDATE predictions SET status = ? WHERE id = ?",
                    (STATUS_SCORED, row["prediction_id"]),
                )
        return inserted

    # ─── Weight versions (self-learning) ──────────────────────────────────────

    def insert_weight_version(
        self, source: str, weights: dict[str, float], *,
        rationale: str = "", activate: bool = False,
    ) -> int:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO weight_versions (created_at, source, weights, active, rationale) "
                "VALUES (?,?,?,?,?)",
                (ts, source, json.dumps(weights, ensure_ascii=False), 0, rationale),
            )
            vid = int(cur.lastrowid)
            if activate:
                conn.execute("UPDATE weight_versions SET active = 0 WHERE source = ?", (source,))
                conn.execute("UPDATE weight_versions SET active = 1 WHERE id = ?", (vid,))
        return vid

    def activate_weight_version(self, version_id: int, source: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE weight_versions SET active = 0 WHERE source = ?", (source,))
            conn.execute("UPDATE weight_versions SET active = 1 WHERE id = ?", (version_id,))

    def get_active_weights(self, source: str) -> Optional[tuple[int, dict[str, float]]]:
        """Restituisce (version_id, weights) della versione attiva, o None se non esiste."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, weights FROM weight_versions WHERE source = ? AND active = 1 "
                "ORDER BY id DESC LIMIT 1",
                (source,),
            ).fetchone()
        if not row:
            return None
        return int(row["id"]), json.loads(row["weights"])

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
