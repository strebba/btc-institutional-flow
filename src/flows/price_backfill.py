"""CLI una tantum: riscarica l'OHLC giornaliero e sostituisce le barre parziali.

Il fetch quotidiano rinfresca solo le ultime sessioni salvate. Le barre
scritte a metà giornata nei mesi precedenti restano sbagliate finché non si
lancia questo comando. ``--dry-run`` logga le diff e non scrive.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Riscarica le barre giornaliere di tutto lo storico salvato e "
            "sostituisce le righe il cui OHLC non è il close/high/low della "
            "sessione UTC chiusa. Il giorno UTC corrente resta partial."
        ),
    )
    parser.add_argument("--db", required=True, help="SQLite dei prezzi (data/runtime.db)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Stampa le diff, non scrive")
    mode.add_argument("--write", action="store_true", help="Sovrascrive le barre diverse")
    parser.add_argument("--ticker", action="append", default=None, help="Limita a un ticker (ripetibile)")
    parser.add_argument("--asof", default=None, help="Giorno UTC corrente YYYY-MM-DD (default: oggi)")
    parser.add_argument(
        "--allow-postgres", action="store_true",
        help="Consente DATABASE_URL Postgres. Senza questo flag il tool si ferma.",
    )
    args = parser.parse_args(argv)

    from src.db import refuse_postgres_unless

    try:
        refuse_postgres_unless(args.allow_postgres, "backfill_prices")
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    asof = None
    if args.asof:
        try:
            asof = date.fromisoformat(args.asof)
        except ValueError:
            print(f"error: --asof non valida {args.asof}", file=sys.stderr)
            return 2

    from src.flows.price_fetcher import PriceFetcher

    try:
        report = PriceFetcher(db_path=Path(args.db)).backfill_history(
            tickers=args.ticker,
            dry_run=not args.write,
            asof=asof,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(report.format_lines()))
    return 0
