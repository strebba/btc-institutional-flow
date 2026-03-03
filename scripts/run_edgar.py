"""Script CLI per eseguire l'intero pipeline EDGAR e stampare un summary.

Uso:
  python scripts/run_edgar.py [--max N] [--query TERM]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Aggiungi la root del progetto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.edgar.search import EdgarEftsSearcher
from src.edgar.parser import ProspectusParser
from src.edgar.structured_notes_db import StructuredNotesDB
from src.config import setup_logging

console = Console()
_log    = setup_logging("run_edgar")


def print_note_summary(notes, db: StructuredNotesDB) -> None:
    """Stampa tabella riepilogativa delle note trovate."""
    table = Table(title="Note Strutturate IBIT trovate su SEC EDGAR", show_lines=True)
    table.add_column("Emittente",    style="cyan",    no_wrap=True)
    table.add_column("Tipo",         style="magenta", no_wrap=True)
    table.add_column("Data",         style="green")
    table.add_column("Nozionale $M", style="yellow",  justify="right")
    table.add_column("Initial IBIT", style="blue",    justify="right")
    table.add_column("Barriers",     style="red")
    table.add_column("URL",          style="dim")

    for note in notes:
        barrier_summary = "; ".join(
            f"{b.barrier_type}@{b.level_pct:.0f}%"
            for b in note.barriers[:3]
        )
        notional_m = f"{note.notional_usd/1e6:.1f}" if note.notional_usd else "?"
        init       = f"{note.initial_level:.2f}" if note.initial_level else "?"
        date_str   = str(note.issue_date) if note.issue_date else "?"

        table.add_row(
            note.issuer or "?",
            note.product_type or "?",
            date_str,
            notional_m,
            init,
            barrier_summary or "—",
            note.filing_url[-60:],
        )

    console.print(table)

    stats = db.summary()
    console.print("\n[bold]─── DB Summary ───[/bold]")
    console.print(f"  Note totali nel DB : [cyan]{stats['total_notes']}[/cyan]")
    console.print(f"  Barrier levels     : [cyan]{stats['total_barriers']}[/cyan]")
    console.print(f"  Barrier attive     : [green]{stats['active_barriers']}[/green]")
    noz_b = stats["total_notional_usd"] / 1e9
    console.print(f"  Nozionale totale   : [yellow]${noz_b:.2f}B[/yellow]")
    console.print(f"  Per tipo           : {json.dumps(stats['by_product_type'], indent=2)}")
    console.print(f"  Per emittente      : {json.dumps(stats['by_issuer'], indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDGAR IBIT structured notes scraper")
    parser.add_argument("--max",   type=int, default=10, help="Max filing da parsare (default 10)")
    parser.add_argument("--query", type=str, default=None, help="Query override (es. 'IBIT')")
    parser.add_argument("--no-parse", action="store_true", help="Solo ricerca, no parsing")
    args = parser.parse_args()

    console.rule("[bold blue]EDGAR IBIT Scraper[/bold blue]")

    # ── 1. Ricerca EDGAR ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 1: Ricerca filing su SEC EDGAR...[/bold]")
    searcher = EdgarEftsSearcher()

    if args.query:
        filings = searcher.search(query=args.query)
    else:
        filings = searcher.collect_all_filings()

    console.print(f"  Trovati [cyan]{len(filings)}[/cyan] filing unici")

    if not filings:
        console.print("[red]Nessun filing trovato. Controlla la connessione e l'user-agent.[/red]")
        sys.exit(1)

    # Stampa preview filing trovati
    console.print("\n[bold]Preview primi 5 filing:[/bold]")
    for f in filings[:5]:
        console.print(
            f"  [{f.get('form_type','?')}] {f.get('entity_name','?')[:40]} "
            f"  {f.get('filing_date','?')}  {f['url'][-70:]}"
        )

    if args.no_parse:
        console.print("\n[yellow]--no-parse: skip parsing.[/yellow]")
        return

    # ── 2. Parsing ────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 2: Parsing di max {args.max} filing...[/bold]")
    parser_obj = ProspectusParser()
    notes      = parser_obj.parse_batch(filings, max_items=args.max)

    console.print(f"  Note estratte con dati: [green]{len(notes)}[/green]/{args.max}")

    # ── 3. Salvataggio DB ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 3: Salvataggio nel DB...[/bold]")
    db  = StructuredNotesDB()
    ids = db.upsert_notes(notes)
    console.print(f"  Salvate {len(ids)} note (id: {ids[:5]}...)")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 4: Summary risultati[/bold]")
    print_note_summary(notes, db)

    console.rule("[bold green]Completato[/bold green]")


if __name__ == "__main__":
    main()
