"""Ricalcola gli esiti già verificati sulle barre giornaliere complete.

Non crea predizioni e non modifica ``predictions`` né ``weight_versions``.
Legge i prezzi già nel DB (niente download): le barre parziali vanno
sostituite prima con ``scripts/backfill_prices.py --write``.

``--dry-run`` stampa vecchio contro nuovo per ogni esito e un riepilogo.
``--write`` archivia la riga corrente in ``outcomes_archive`` (valori, timestamp,
motivo) e poi aggiorna ``outcomes``. Un secondo ``--write`` con gli stessi
prezzi non ri-archivia: gli esiti coincidono già.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from src.config import TICKER_MAP
from src.forecast.models import TARGET_DIRECTION, Outcome, Prediction
from src.forecast.verifier import (
    _cohort_key,
    prediction_for_scoring,
    score_prediction,
)

_COMPARE = (
    "hit", "realized_return", "realized_price", "ref_price",
    "signed_error", "brier", "detail",
)
_DEFAULT_REASON = "rescore_complete_daily_bars"


@dataclass
class RescoreRow:
    prediction_id: int
    target_type: str
    day: str
    old: dict[str, Any]
    new: Optional[dict[str, Any]]
    skip_reason: Optional[str]

    @property
    def changed_fields(self) -> list[str]:
        if self.new is None or self.skip_reason:
            return []
        return [name for name in _COMPARE if not _same(self.old.get(name), self.new.get(name))]

    @property
    def changed(self) -> bool:
        return bool(self.changed_fields)


@dataclass
class RescoreReport:
    rows: tuple[RescoreRow, ...]
    written: bool
    reason: str

    @property
    def changed(self) -> list[RescoreRow]:
        return [row for row in self.rows if row.changed]

    @property
    def unchanged(self) -> int:
        return sum(1 for row in self.rows if row.skip_reason is None and not row.changed)

    @property
    def skipped(self) -> list[RescoreRow]:
        return [row for row in self.rows if row.skip_reason]

    def format_lines(self) -> list[str]:
        lines: list[str] = []
        for row in self.rows:
            if row.skip_reason:
                lines.append(
                    f"pred={row.prediction_id} {row.target_type} {row.day} "
                    f"SKIP {row.skip_reason}"
                )
                continue
            status = "CHANGED" if row.changed else "unchanged"
            bits = []
            assert row.new is not None
            fields = row.changed_fields or list(_COMPARE)
            for name in fields:
                bits.append(f"{name} {_cell(row.old.get(name))} -> {_cell(row.new.get(name))}")
            lines.append(
                f"pred={row.prediction_id} {row.target_type} {row.day} {status}: "
                + "; ".join(bits)
            )
        lines.append(
            f"summary: changed={len(self.changed)} unchanged={self.unchanged} "
            f"skipped={len(self.skipped)} written={self.written}"
        )
        if not self.written:
            lines.append(
                "dry-run: nessun esito modificato. I prezzi sono quelli già nel DB; "
                "se le barre parziali non sono state sostituite, lanciare prima "
                "backfill_prices.py --write."
            )
        return lines


def run_rescore(db, fetcher, *, write: bool, reason: str = _DEFAULT_REASON) -> RescoreReport:
    """Confronta ogni esito con ``score_prediction`` sulle barre complete del DB."""
    stored = db.scored_rows()
    anchors: dict[tuple, Prediction] = {}
    for row in stored:
        if row.get("target_type") != TARGET_DIRECTION:
            continue
        pred = Prediction.from_row(row)
        anchors[_cohort_key(pred)] = pred

    frames: dict[str, Any] = {}
    rows: list[RescoreRow] = []
    pending: list[tuple[dict, Outcome]] = []
    for row in stored:
        pred = Prediction.from_row(row)
        anchor = anchors.get(_cohort_key(pred))
        scoring = prediction_for_scoring(pred, anchor)
        ticker = TICKER_MAP.get(pred.asset, pred.asset)
        if ticker not in frames:
            frames[ticker] = fetcher._load_from_db(ticker, date(2010, 1, 1), date(2100, 1, 1))
        outcome = score_prediction(scoring, frames[ticker])
        old = _old_view(row)
        base = RescoreRow(
            prediction_id=int(pred.id or row["id"]),
            target_type=pred.target_type,
            day=pred.created_at[:10],
            old=old,
            new=None,
            skip_reason=None,
        )
        if outcome is None:
            rows.append(RescoreRow(
                base.prediction_id, base.target_type, base.day, old, None,
                "barra di fine orizzonte assente o ancora parziale",
            ))
            continue
        outcome = Outcome(
            prediction_id=base.prediction_id,
            hit=outcome.hit,
            realized_return=outcome.realized_return,
            realized_price=outcome.realized_price,
            ref_price=outcome.ref_price,
            signed_error=outcome.signed_error,
            brier=outcome.brier,
            detail=outcome.detail,
        )
        new = _new_view(outcome)
        built = RescoreRow(base.prediction_id, base.target_type, base.day, old, new, None)
        rows.append(built)
        if built.changed:
            pending.append((row, outcome))

    if write and pending:
        db.apply_outcome_rescore(pending, reason=reason)
    return RescoreReport(rows=tuple(rows), written=bool(write), reason=reason)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ricalcola gli esiti già verificati usando le barre giornaliere complete "
            "già presenti nel DB. Non modifica le predizioni né i pesi."
        ),
    )
    parser.add_argument("--db", required=True, help="SQLite con predictions, outcomes e prices (data/runtime.db)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Stampa vecchio/nuovo, non scrive")
    mode.add_argument("--write", action="store_true", help="Archivia gli esiti e aggiorna outcomes")
    parser.add_argument("--reason", default=_DEFAULT_REASON, help="Motivo scritto in outcomes_archive")
    parser.add_argument(
        "--allow-postgres", action="store_true",
        help="Consente DATABASE_URL Postgres. Senza questo flag il tool si ferma.",
    )
    args = parser.parse_args(argv)

    from src.db import refuse_postgres_unless

    try:
        refuse_postgres_unless(args.allow_postgres, "rescore_outcomes")
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    from pathlib import Path

    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.prediction_db import PredictionDB

    db_path = Path(args.db)
    report = run_rescore(
        PredictionDB(db_path),
        PriceFetcher(db_path=db_path),
        write=args.write,
        reason=args.reason,
    )
    print("\n".join(report.format_lines()))
    return 0


def _old_view(row: dict) -> dict[str, Any]:
    return {
        "hit": bool(row.get("hit")),
        "realized_return": row.get("realized_return"),
        "realized_price": row.get("realized_price"),
        "ref_price": row.get("ref_price"),
        "signed_error": row.get("signed_error"),
        "brier": row.get("brier"),
        "detail": row.get("outcome_detail") or "",
    }


def _new_view(outcome: Outcome) -> dict[str, Any]:
    return {
        "hit": bool(outcome.hit),
        "realized_return": outcome.realized_return,
        "realized_price": outcome.realized_price,
        "ref_price": outcome.ref_price,
        "signed_error": outcome.signed_error,
        "brier": outcome.brier,
        "detail": outcome.detail,
    }


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return (left or "") == (right or "")
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-6)
    except (TypeError, ValueError):
        return left == right


def _cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
