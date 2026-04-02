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
    console.print("\n[bold]ETF Flows (ultimi 5 giorni):[/bold]")
    try:
        flows = client.fetch_etf_flows(days=30)
        if not flows:
            console.print("  [yellow]Nessun flusso ETF ricevuto[/yellow]")
        else:
            # Group by date and show all tickers
            from collections import defaultdict

            by_date: dict = defaultdict(dict)
            for f in flows:
                by_date[f.date][f.ticker] = f.flow_usd

            sorted_dates = sorted(by_date.keys())[-5:]
            all_tickers = sorted({f.ticker for f in flows})

            table = Table()
            table.add_column("Date", style="cyan")
            for tk in all_tickers:
                table.add_column(tk, justify="right")
            table.add_column("Total", justify="right")
            table.add_column("Source")

            for d in sorted_dates:
                row_data = [str(d)]
                total = 0.0
                for tk in all_tickers:
                    val = by_date[d].get(tk, 0)
                    total += val
                    color = "green" if val >= 0 else "red"
                    row_data.append(f"[{color}]{val / 1e6:+.1f}[/{color}]")
                total_color = "green" if total >= 0 else "red"
                row_data.append(f"[{total_color}]{total / 1e6:+.1f}[/{total_color}]")
                row_data.append("coinglass")
                table.add_row(*row_data)
            console.print(table)
            console.print(f"  ETF disponibili: {', '.join(all_tickers)}")
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
            last_val = fr.iloc[-1]
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
            last_val = oi.iloc[-1]
            console.print(f"  {last_date}: ${last_val / 1e9:.2f}B")
    except Exception as e:
        console.print(f"  [yellow]OI non disponibile: {e}[/yellow]")

    console.rule("[bold green]Completato[/bold green]")


if __name__ == "__main__":
    main()
