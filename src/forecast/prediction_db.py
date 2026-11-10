"""Persistenza SQLite del forecast spine.

Tre tabelle:
- `predictions`     — una riga per previsione (UNIQUE su created_at+source+target_type+horizon_days
                      per idempotenza: rilanciare il job predict nello stesso istante non duplica).
- `outcomes`        — una riga per esito (UNIQUE su prediction_id).
- `weight_versions` — versioni dei pesi del SignalModel (audit trail del self-learning).

WAL mode, path da settings.yaml, INSERT OR IGNORE per idempotenza.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from src.config import get_settings, setup_logging
from src.db import connect, days_ago_iso, ensure_postgres_schema, is_postgres
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
    model_version    TEXT NOT NULL DEFAULT 'v1',
    status           TEXT NOT NULL DEFAULT 'open',
    metadata         TEXT,                   -- JSON; NULL sulle righe live
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
    status      TEXT NOT NULL DEFAULT 'proposed',  -- proposed | active | retired
    rationale   TEXT
);
CREATE INDEX IF NOT EXISTS idx_wv_source_active ON weight_versions(source, active);
"""

_PRED_COLS = (
    "created_at, date, source, asset, target_type, target_spec, horizon_days, "
    "confidence, rationale, counter_analysis, human_overlay, score_ref, "
    "components, weights_version, model_version, status, metadata"
)


class CohortExistsError(RuntimeError):
    """La coorte (date, source) è già nel DB: non viene sovrascritta."""

    def __init__(self, day: str, source: str) -> None:
        super().__init__(f"coorte già presente per {day} source={source}")
        self.day = day
        self.source = source


def _insert_sql() -> str:
    placeholders = ",".join("?" for _ in _PRED_COLS.split(","))
    return (
        f"INSERT OR IGNORE INTO predictions ({_PRED_COLS}) VALUES ({placeholders}) "
        "RETURNING id"
    )


def _pred_values(pred: Prediction) -> tuple:
    row = pred.to_row()
    return tuple(row[c.strip()] for c in _PRED_COLS.split(","))


class PredictionDB:
    """Persistenza di predizioni, esiti e versioni dei pesi."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings()
        self._path = Path(db_path or cfg["database"]["path"])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        """Connessione al backend attivo (SQLite su ``self._path`` o Postgres)."""
        with connect(
            self._path,
            sqlite_pragmas=("journal_mode=WAL", "foreign_keys=ON"),
            sqlite_timeout=30,
        ) as conn:
            yield conn

    def _init_db(self) -> None:
        if is_postgres():
            # Su Postgres lo schema e' gestito da src/db/migrations (gia' migrato).
            ensure_postgres_schema()
            return
        with self._conn() as conn:
            conn.executescript(_DDL)
            # Migrazione: aggiunge la colonna status se un DB pre-esistente non l'ha.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(weight_versions)")}
            if "status" not in cols:
                conn.execute(
                    "ALTER TABLE weight_versions ADD COLUMN status TEXT NOT NULL DEFAULT 'proposed'"
                )
                conn.execute("UPDATE weight_versions SET status = 'active' WHERE active = 1")
            # Indice su status creato dopo la migrazione (la colonna ora esiste sicuramente).
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wv_source_status ON weight_versions(source, status)"
            )
            # Migrazione: aggiunge model_version e classifica retroattivamente le predizioni
            # prodotte con modelli precedenti (v0=7 componenti, v0b=barrier_confluence transitorio).
            pred_cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
            if "model_version" not in pred_cols:
                conn.execute(
                    "ALTER TABLE predictions ADD COLUMN model_version TEXT NOT NULL DEFAULT 'v1'"
                )
                conn.execute("UPDATE predictions SET model_version = 'v0' WHERE id <= 8")
                conn.execute(
                    "UPDATE predictions SET model_version = 'v0b' WHERE id BETWEEN 9 AND 11"
                )
            # Dedup per-GIORNO: un solo set (source, target_type, horizon) per data.
            # Evita doppioni quando run manuale e schedulato girano lo stesso giorno o al riavvio.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_day_unique "
                "ON predictions(date, source, target_type, horizon_days)"
            )
            # Flag di ricostruzione: colonna nullable, le righe esistenti restano NULL.
            pred_cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
            if "metadata" not in pred_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN metadata TEXT")

    # ─── Predictions ────────────────────────────────────────────────────────

    def insert_prediction(self, pred: Prediction) -> Optional[int]:
        """Inserisce una previsione. Restituisce l'id, o None se duplicata."""
        with self._conn() as conn:
            cur = conn.execute(_insert_sql(), _pred_values(pred))
            # RETURNING (non lastrowid): funziona su SQLite e Postgres; nessuna riga se ignorata.
            # fetchall (non fetchone): esaurisce lo statement prima del commit su SQLite.
            inserted = cur.fetchall()
        return int(inserted[0][0]) if inserted else None

    def insert_cohort(self, preds: list[Prediction]) -> list[int]:
        """Inserisce un'intera coorte in una transazione, senza toccare le altre.

        Se esiste già una riga per la stessa data e source, non scrive nulla
        e solleva ``CohortExistsError`` (rollback).
        """
        if not preds:
            return []
        day = preds[0].to_row()["date"]
        source = preds[0].source
        if any(p.to_row()["date"] != day or p.source != source for p in preds):
            raise ValueError("insert_cohort richiede una sola data e una sola source")
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE date = ? AND source = ?",
                (day, source),
            ).fetchone()[0]
            if existing:
                raise CohortExistsError(day, source)
            ids: list[int] = []
            for pred in preds:
                cur = conn.execute(_insert_sql(), _pred_values(pred))
                inserted = cur.fetchall()
                if not inserted:
                    raise CohortExistsError(day, source)
                ids.append(int(inserted[0][0]))
            return ids

    def predictions_on_date(self, day: str, source: str) -> list[Prediction]:
        """Righe di una coorte (data + source), senza filtrare per status."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE date = ? AND source = ? ORDER BY id ASC",
                (day, source),
            ).fetchall()
        return [Prediction.from_row(dict(r)) for r in rows]

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

    def get_with_outcomes(
        self,
        days: int = 180,
        source: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> list[dict]:
        """Join predictions+outcomes (left) per la calibrazione e i report.

        model_version: se specificato, filtra per escludere predizioni di modelli eterogenei
        (es. 'v1' per usare solo dati prodotti con il SignalModel a 8 componenti).
        """
        q = (
            "SELECT p.*, o.hit, o.realized_return, o.realized_price, o.signed_error, "
            "       o.brier, o.detail AS outcome_detail, o.scored_at "
            "FROM predictions p LEFT JOIN outcomes o ON o.prediction_id = p.id "
            "WHERE p.date >= ? "
        )
        params: list[Any] = [days_ago_iso(days)]
        if source:
            q += "AND p.source = ? "
            params.append(source)
        if model_version:
            q += "AND p.model_version = ? "
            params.append(model_version)
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
        """Inserisce una versione pesi (default status 'proposed'). Se activate, la rende attiva."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO weight_versions (created_at, source, weights, active, status, rationale) "
                "VALUES (?,?,?,?,?,?) RETURNING id",
                (ts, source, json.dumps(weights, ensure_ascii=False), 0, "proposed", rationale),
            )
            vid = int(cur.fetchall()[0][0])
            if activate:
                self._activate(conn, vid, source)
        return vid

    @staticmethod
    def _activate(conn: Any, version_id: int, source: str) -> None:
        # Retira l'attiva corrente, attiva la nuova (audit trail: niente cancellazioni).
        conn.execute(
            "UPDATE weight_versions SET active = 0, status = 'retired' "
            "WHERE source = ? AND active = 1",
            (source,),
        )
        conn.execute(
            "UPDATE weight_versions SET active = 1, status = 'active' WHERE id = ?",
            (version_id,),
        )

    def activate_weight_version(self, version_id: int, source: str) -> None:
        """Attivazione human-gated di una versione (il workflow Tuning chiama qui)."""
        with self._conn() as conn:
            self._activate(conn, version_id, source)

    def get_proposed_weights(self, source: str) -> list[dict]:
        """Versioni in stato 'proposed' in attesa di approvazione, dalla più recente."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, weights, rationale FROM weight_versions "
                "WHERE source = ? AND status = 'proposed' ORDER BY id DESC",
                (source,),
            ).fetchall()
        return [{**dict(r), "weights": json.loads(r["weights"])} for r in rows]

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

    def has_cohort(
        self,
        date: str,
        *,
        source: str,
        horizon_days: Optional[int] = None,
    ) -> bool:
        """True se esiste già almeno una predizione della cohort di quel giorno.

        La data è la colonna ``date`` (``created_at`` UTC, ``YYYY-MM-DD``), la stessa
        che lo unique index ``idx_pred_day_unique`` usa per impedire un secondo set
        nello stesso giorno. ``horizon_days`` restringe al job giornaliero (default 5).
        """
        q = "SELECT 1 FROM predictions WHERE date = ? AND source = ?"
        params: list[Any] = [date, source]
        if horizon_days is not None:
            q += " AND horizon_days = ?"
            params.append(int(horizon_days))
        q += " LIMIT 1"
        with self._conn() as conn:
            return conn.execute(q, params).fetchone() is not None
