"""Script CLI per il Modulo 3: ETF Flow Tracker.

Uso:
  python scripts/run_flows.py [--no-fetch] [--plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.flows.scraper import FarsideScraper
from src.flows.price_fetcher import PriceFetcher
from src.flows.correlation import FlowCorrelation
from src.config import setup_logging

console = Console()
_log    = setup_logging("run_flows")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Flow Tracker")
    parser.add_argument("--no-fetch",  action="store_true", help="Usa solo cache locale")
    parser.add_argument("--plot",      action="store_true", help="Mostra grafico Plotly")
    parser.add_argument("--window",    type=int, default=30, help="Finestra rolling (default 30)")
    args = parser.parse_args()

    console.rule("[bold blue]ETF Flow Tracker[/bold blue]")

    # ── 1. Flussi Farside ────────────────────────────────────────────────────
    console.print("\n[bold]Step 1: Scraping Farside Investors...[/bold]")
    scraper = FarsideScraper()
    flows   = scraper.fetch()

    if not flows:
        console.print("[red]Nessun dato di flusso recuperato.[/red]")
        sys.exit(1)

    agg_flows = scraper.aggregate(flows)
    df_pivot  = scraper.to_dataframe(flows)

    console.print(f"  Giorni di dati: [cyan]{len(df_pivot)}[/cyan]")
    console.print(f"  Range: {df_pivot.index.min().date()} → {df_pivot.index.max().date()}")

    # Tabella top 10 giorni per flusso IBIT
    if "IBIT" in df_pivot.columns:
        top10 = df_pivot["IBIT"].dropna().nlargest(5)
        table = Table(title="Top 5 giorni IBIT inflow")
        table.add_column("Data",     style="cyan")
        table.add_column("IBIT $M",  style="green", justify="right")
        table.add_column("Total $M", style="yellow", justify="right")
        for d, val in top10.items():
            total = df_pivot.loc[d, "total"] / 1e6 if "total" in df_pivot.columns else 0
            table.add_row(str(d.date()), f"{val/1e6:.0f}", f"{total:.0f}")
        console.print(table)

        ibit_net = df_pivot["IBIT"].sum() / 1e9
        console.print(f"\n  IBIT net inflow totale: [bold green]${ibit_net:.2f}B[/bold green]")

    # ── 2. Prezzi ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 2: Download prezzi BTC-USD e IBIT...[/bold]")
    fetcher = PriceFetcher()

    if not args.no_fetch:
        prices = fetcher.get_all_prices()
    else:
        from datetime import date, timedelta
        prices = fetcher.get_all_prices(
            start_date=date.today() - timedelta(days=400)
        )

    if prices.empty:
        console.print("[red]Nessun prezzo disponibile.[/red]")
        sys.exit(1)

    console.print(f"  Prezzi: {len(prices)} giorni")
    console.print(f"  BTC ultimo close: [yellow]${prices['btc_close'].dropna().iloc[-1]:,.0f}[/yellow]")
    console.print(f"  IBIT ultimo close: [yellow]${prices['ibit_close'].dropna().iloc[-1]:.2f}[/yellow]")

    ratio = fetcher.get_ibit_btc_ratio()
    if ratio:
        console.print(f"  IBIT/BTC ratio corrente: [cyan]{ratio:.6f}[/cyan]")

    # ── 3. Merge + Correlazione ───────────────────────────────────────────────
    console.print("\n[bold]Step 3: Merge e analisi correlazione...[/bold]")
    corr_engine = FlowCorrelation()
    merged      = corr_engine.merge(agg_flows, prices)

    if merged.empty:
        console.print("[red]Merge vuoto — controlla le date dei dati.[/red]")
        sys.exit(1)

    console.print(f"  Righe nel dataset merged: [cyan]{len(merged)}[/cyan]")
    console.print(f"  Righe con flussi IBIT:    [cyan]{merged['ibit_flow'].notna().sum()}[/cyan]")
    console.print(f"  Righe con prezzi BTC:     [cyan]{merged['btc_close'].notna().sum()}[/cyan]")

    # Statistiche
    stats = corr_engine.summary_stats(merged)
    if "ibit" in stats:
        s = stats["ibit"]
        console.print(f"\n  IBIT inflows totali:  [green]${s['total_inflow_usd_b']:.2f}B[/green]")
        console.print(f"  IBIT outflows totali: [red]${s['total_outflow_usd_b']:.2f}B[/red]")
        console.print(f"  IBIT net flow:        [yellow]${s['net_flow_usd_b']:.2f}B[/yellow]")

    if "full_period_corr_ibit_btc_next1d" in stats:
        c = stats["full_period_corr_ibit_btc_next1d"]
        color = "green" if c > 0 else "red"
        console.print(
            f"\n  Correlazione IBIT flows → BTC next-day return: "
            f"[{color}]{c:.4f}[/{color}]"
        )

    # Rolling correlations
    console.print(f"\n[bold]Rolling correlations (ultima settimana):[/bold]")
    roll_corrs = corr_engine.rolling_correlations(merged, windows=[30, 60, 90])
    corr_table = Table()
    corr_table.add_column("Finestra", style="cyan")
    corr_table.add_column("IBIT→BTC_ret",  justify="right")
    corr_table.add_column("Total→BTC_ret", justify="right")
    corr_table.add_column("IBIT→BTC_vol",  justify="right")

    for window_key, corr_df in roll_corrs.items():
        if corr_df.empty:
            continue
        last = corr_df.dropna(how="all").iloc[-1] if not corr_df.dropna(how="all").empty else None
        if last is None:
            continue
        def fmt(v):
            if hasattr(v, '__float__') and not (v != v):  # not NaN
                c = "green" if float(v) > 0 else "red"
                return f"[{c}]{float(v):.3f}[/{c}]"
            return "[dim]n/a[/dim]"
        corr_table.add_row(
            window_key,
            fmt(last.get("ibit_flow_vs_btc_return", float("nan"))),
            fmt(last.get("total_flow_vs_btc_return", float("nan"))),
            fmt(last.get("ibit_flow_vs_btc_vol", float("nan"))),
        )
    console.print(corr_table)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    if args.plot:
        console.print("\n[bold]Step 4: Generazione grafico Plotly...[/bold]")
        fig = corr_engine.plot_flows(merged, window=args.window)
        if fig:
            fig.show()

    console.rule("[bold green]Completato[/bold green]")


if __name__ == "__main__":
    main()
