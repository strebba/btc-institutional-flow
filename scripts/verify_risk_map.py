"""Walk-forward della risk map su barre giornaliere Binance. Non scrive sul DB.

Ricalcola, per ogni giorno, la mappa usando solo le barre chiuse prima di
quel giorno e la confronta con il range e i tocchi dei 5 giorni successivi.

Non riproduce alla lettera il backtest del 10/11/2026 (quello usava barre
orarie 08:00–08:00 e i max pain salvati nelle coorti). Qui le barre sono
giornaliere UTC e i tocchi sono una griglia di distanze in unità di σ, perché
lo storico dei max pain non sta in questo repo. I numeri vanno letti come
controllo di calibrazione, non come copia delle cifre del JSON del backtest.

Uso:
    python3 scripts/verify_risk_map.py
    python3 scripts/verify_risk_map.py --csv data/btcusdt_1d.csv --min-returns 180
    python3 scripts/verify_risk_map.py --sample-date 2026-11-10
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.risk_map import (  # noqa: E402
    TOUCH_Z_GRID,
    build_risk_map,
    decision_asof_for,
    evaluate_walk_forward,
    fetch_binance_daily,
    prepare_bars,
    walk_forward,
)


def _load(args: argparse.Namespace):
    import pandas as pd

    if args.csv:
        df = pd.read_csv(args.csv)
        return df
    return fetch_binance_daily(symbol=args.symbol, start=args.start)


def _filter(wf: dict, min_returns: int) -> dict:
    return {
        "range_rows": [r for r in wf["range_rows"] if r["n_returns"] >= min_returns],
        "touch_rows": [t for t in wf["touch_rows"] if t["n_returns"] >= min_returns],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifica walk-forward della risk map")
    parser.add_argument("--csv", help="OHLC giornaliero (date,open,high,low,close) invece di Binance")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-returns", type=int, default=180,
                        help="scarta le origini con meno rendimenti (burn-in del GARCH)")
    parser.add_argument("--sample-date", help="YYYY-MM-DD: stampa solo la mappa di quel giorno e esce")
    parser.add_argument("--ref-price", type=float, default=None,
                        help="spot per il sample; default = ultimo close chiuso")
    args = parser.parse_args(argv)

    bars = _load(args)
    if args.sample_date:
        day = date.fromisoformat(args.sample_date)
        asof = decision_asof_for(day)
        closed = prepare_bars(bars, asof=asof)
        if closed.empty:
            print(json.dumps({"error": "nessuna barra chiusa", "asof": asof.isoformat()}))
            return 1
        ref = args.ref_price if args.ref_price else float(closed["close"].iloc[-1])
        payload = build_risk_map(bars, ref_price=ref, levels=[], asof=asof, horizon=args.horizon)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    asof = datetime.now(timezone.utc)
    wf = walk_forward(bars, asof=asof, horizon=args.horizon, z_grid=TOUCH_Z_GRID)
    scored = evaluate_walk_forward(_filter(wf, args.min_returns))
    latest = wf.get("latest") or {}
    ref = args.ref_price or latest.get("ref_close") or 0.0
    sample = None
    if ref and float(ref) > 0:
        sample = build_risk_map(bars, ref_price=float(ref), levels=[], asof=asof, horizon=args.horizon)
    report = {
        "horizon_days": args.horizon,
        "min_returns": args.min_returns,
        "note": (
            "Barre giornaliere Binance, decisione sulle barre chiuse prima del giorno. "
            "I tocchi sono una griglia di z, non i max pain delle coorti. "
            "range_model è la CDF del range browniano; range_empirical è la frequenza "
            "dei range/σ passati (è il numero pubblicato quando la storia basta). "
            "touch_used è la frequenza delle escursioni, altrimenti 2(1-Φ(z))."
        ),
        "metrics": scored,
        "sample": sample,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
