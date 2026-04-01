"""Script CLI: verifica e scarica dati da CoinGlass API v4.

Uso:
    COINGLASS_API_KEY=<key> python3 scripts/run_coinglass.py

Stampa:
  - Ultimi 5 flussi IBIT giornalieri
  - Ultimo funding rate OI-weighted
  - Ultimo futures OI aggregato
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.flows.coinglass_client import CoinGlassClient, CoinGlassError

console = Console()


def main() -> None:
    console.rule("[bold blue]CoinGlass API — Test & Fetch[/bold blue]")
    client = CoinGlassClient()

    # ── ETF Flows ─────────────────────────────────────────────────────────────
    console.print("\n[bold]ETF Flows IBIT (ultimi 5 giorni):[/bold]")
    try:
        flows = client.fetch_etf_flows(days=30)
        ibit  = [f for f in flows if f.ticker == "IBIT"]
        if not ibit:
            console.print("  [yellow]Nessun flusso IBIT ricevuto[/yellow]")
        else:
            table = Table()
            table.add_column("Date",       style="cyan")
            table.add_column("Flow USD M", justify="right")
            table.add_column("Source")
            for f in ibit[-5:]:
                color = "green" if f.flow_usd >= 0 else "red"
                table.add_row(
                    str(f.date),
                    f"[{color}]{f.flow_usd/1e6:+.1f}[/{color}]",
                    f.source,
                )
            console.print(table)
            console.print(f"  Totale record flussi: {len(flows)}")
    except CoinGlassError as e:
        console.print(f"  [red]{e}[/red]")
        sys.exit(1)

    # ── Funding Rate ──────────────────────────────────────────────────────────
    console.print("\n[bold]Funding Rate OI-weighted (ultimo valore):[/bold]")
    try:
        fr = client.fetch_funding_rate_history(days=7)
        if fr.empty:
            console.print("  [yellow]Nessun dato funding rate[/yellow]")
        else:
            last_date = fr.index[-1].strftime("%Y-%m-%d")
            last_val  = fr.iloc[-1]
            color = "red" if last_val > 0.05 else "green"
            console.print(f"  [{color}]{last_date}: {last_val:.4f}% (8h)[/{color}]")
    except Exception as e:
        console.print(f"  [yellow]Funding rate non disponibile: {e}[/yellow]")

    # ── Futures OI ────────────────────────────────────────────────────────────
    console.print("\n[bold]Futures OI Aggregato (ultimo valore):[/bold]")
    try:
        oi = client.fetch_aggregated_oi_history(days=7)
        if oi.empty:
            console.print("  [yellow]Nessun dato OI[/yellow]")
        else:
            last_date = oi.index[-1].strftime("%Y-%m-%d")
            last_val  = oi.iloc[-1]
            console.print(f"  {last_date}: ${last_val/1e9:.2f}B")
    except Exception as e:
        console.print(f"  [yellow]OI non disponibile: {e}[/yellow]")

    console.rule("[bold green]Completato[/bold green]")


if __name__ == "__main__":
    main()
