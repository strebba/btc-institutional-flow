"""Script CLI per il Modulo 2: GEX Calculator.

Uso:
  python scripts/run_gex.py [--plot] [--top N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.gex.deribit_client import DeribitClient
from src.gex.gex_calculator import GexCalculator
from src.gex.regime_detector import RegimeDetector
from src.config import setup_logging

console = Console()
_log    = setup_logging("run_gex")


def main() -> None:
    parser = argparse.ArgumentParser(description="GEX Calculator — Deribit BTC options")
    parser.add_argument("--plot",  action="store_true", help="Mostra grafico Plotly")
    parser.add_argument("--top",   type=int, default=20, help="Top N strike per tabella")
    args = parser.parse_args()

    console.rule("[bold blue]GEX Calculator — BTC Deribit[/bold blue]")

    # ── 1. Spot price ─────────────────────────────────────────────────────────
    client = DeribitClient()
    console.print("\n[bold]Step 1: Prezzo spot BTC...[/bold]")
    try:
        spot = client.get_spot_price()
        console.print(f"  BTC spot: [yellow]${spot:,.0f}[/yellow]")
    except Exception as e:
        console.print(f"[red]Errore get_spot_price: {e}[/red]")
        sys.exit(1)

    # ── 2. Download chain ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 2: Download chain opzioni BTC...[/bold]")
    options = client.fetch_all_options("BTC")
    console.print(f"  Opzioni scaricate: [cyan]{len(options)}[/cyan]")

    if not options:
        console.print("[red]Nessuna opzione disponibile.[/red]")
        sys.exit(1)

    # Statistiche chain
    calls = [o for o in options if o["option_type"] == "call"]
    puts  = [o for o in options if o["option_type"] == "put"]
    console.print(f"  Call: [green]{len(calls)}[/green]   Put: [red]{len(puts)}[/red]")
    total_oi = sum(o["open_interest"] for o in options)
    console.print(f"  OI totale: [cyan]{total_oi:,.0f}[/cyan] contratti BTC")

    # ── 3. Calcolo GEX ────────────────────────────────────────────────────────
    console.print("\n[bold]Step 3: Calcolo GEX...[/bold]")
    calculator = GexCalculator()
    snapshot   = calculator.calculate_gex(options, spot)
    gex_dict   = calculator.gex_to_dict(snapshot)

    # Summary metriche chiave
    console.print(f"\n  Total Net GEX:    [{'green' if snapshot.total_net_gex >= 0 else 'red'}]"
                  f"${gex_dict['total_net_gex_m']:+.1f}M[/]")
    console.print(f"  Gamma Flip:       [yellow]${snapshot.gamma_flip_price or 0:,.0f}[/yellow]")
    console.print(f"  Put Wall:         [red]${snapshot.put_wall or 0:,.0f}[/red]")
    console.print(f"  Call Wall:        [green]${snapshot.call_wall or 0:,.0f}[/green]")
    console.print(f"  Max Pain:         [blue]${snapshot.max_pain or 0:,.0f}[/blue]")
    console.print(f"  Put/Call ratio:   [cyan]{snapshot.put_call_ratio or 0:.3f}[/cyan]")

    if snapshot.distance_to_put_wall_pct is not None:
        d = snapshot.distance_to_put_wall_pct
        color = "red" if abs(d) < 5 else "yellow"
        console.print(f"  Dist → Put Wall:  [{color}]{d:+.1f}%[/{color}]")
    if snapshot.distance_to_call_wall_pct is not None:
        d = snapshot.distance_to_call_wall_pct
        color = "green" if 0 < d < 10 else "yellow"
        console.print(f"  Dist → Call Wall: [{color}]{d:+.1f}%[/{color}]")

    # ── 4. Regime ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 4: Classificazione regime...[/bold]")
    detector = RegimeDetector()
    state    = detector.detect(snapshot)
    console.print(detector.summary(state))

    # ── 5. Top strike table ───────────────────────────────────────────────────
    console.print(f"\n[bold]Top {args.top} strike per |GEX| (intorno allo spot):[/bold]")

    # Filtra strike entro ±30% dallo spot
    relevant = [
        gs for gs in snapshot.gex_by_strike
        if abs(gs.strike - spot) / spot < 0.30
    ]
    relevant_sorted = sorted(relevant, key=lambda g: abs(g.net_gex), reverse=True)

    table = Table()
    table.add_column("Strike",    style="cyan",  justify="right")
    table.add_column("Net GEX $M", justify="right")
    table.add_column("Call GEX",  style="green", justify="right")
    table.add_column("Put GEX",   style="red",   justify="right")
    table.add_column("Call OI",   justify="right")
    table.add_column("Put OI",    justify="right")

    for gs in relevant_sorted[:args.top]:
        net_color = "green" if gs.net_gex >= 0 else "red"
        table.add_row(
            f"${gs.strike:,.0f}",
            f"[{net_color}]{gs.net_gex/1e6:+.2f}[/{net_color}]",
            f"{gs.call_gex/1e6:.2f}",
            f"{gs.put_gex/1e6:.2f}",
            f"{gs.call_oi:,.0f}",
            f"{gs.put_oi:,.0f}",
        )
    console.print(table)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    if args.plot:
        _plot_gex(snapshot, spot)

    console.rule("[bold green]Completato[/bold green]")


def _plot_gex(snapshot, spot: float) -> None:
    """Genera grafico Plotly GEX per strike."""
    try:
        import plotly.graph_objects as go
        from src.config import get_settings
        theme = get_settings()["dashboard"]["theme"]

        # Filtra strike entro ±40% dallo spot
        strikes = [g.strike for g in snapshot.gex_by_strike if abs(g.strike - spot) / spot < 0.40]
        net_gex = [g.net_gex / 1e6 for g in snapshot.gex_by_strike if abs(g.strike - spot) / spot < 0.40]
        colors  = [theme["positive"] if v >= 0 else theme["negative"] for v in net_gex]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=strikes, y=net_gex,
            marker_color=colors,
            name="Net GEX ($M)",
        ))
        # Linee verticali per spot, put wall, call wall
        for price, label, color in [
            (spot,                      "Spot",       "white"),
            (snapshot.put_wall or 0,    "Put Wall",   theme["negative"]),
            (snapshot.call_wall or 0,   "Call Wall",  theme["positive"]),
            (snapshot.gamma_flip_price or 0, "Flip",  "orange"),
        ]:
            if price:
                fig.add_vline(x=price, line_dash="dash", line_color=color,
                              annotation_text=label, annotation_position="top")
        fig.update_layout(
            title=f"BTC GEX by Strike | Total Net GEX: ${snapshot.total_net_gex/1e6:.1f}M",
            xaxis_title="Strike ($)",
            yaxis_title="Net GEX ($M)",
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text"]),
        )
        fig.show()
    except ImportError:
        console.print("[yellow]plotly non installato — skip grafico[/yellow]")


if __name__ == "__main__":
    main()
