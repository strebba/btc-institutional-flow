"""Job giornaliero di produzione: raccolta dati + forecast, tutto su Postgres.

Sostituisce il job del Mac mini (LaunchAgent ``com.wagmi.forecast-daily``) e lo
scheduler forecast in-process dell'API. Lanciato dal workflow
``.github/workflows/daily-pipeline.yml``; scrive SOLO sul Postgres indicato da
``DATABASE_URL`` (rifiuta di partire senza: su un runner effimero un fallback
SQLite perderebbe i dati senza errori visibili).

Passi, nell'ordine (ognuno isolato: un errore non ferma i successivi):
  1. gex        snapshot GEX Deribit BTC ed ETH          → gex_snapshots
  2. prices     prezzi spot + ETF (yfinance) per asset   → prices
  3. coinglass  verifica raggiungibilita' CoinGlass (sola lettura, serve la chiave)
  4. flows      verifica scraping Farside (sola lettura: i flussi ETF non sono
                persistiti, l'API li legge live)
  5. predict    predizioni dealer-flow del giorno       → predictions
  6. verify     esiti delle predizioni mature           → outcomes
  7. calibrate  solo il lunedi' (o --calibrate): propone pesi, mai li attiva
                                                        → weight_versions

Gli snapshot macro (funding/OI CoinGecko) e il refresh EDGAR restano nei loro
workflow (macro-snapshot.yml, edgar-refresh.yml), che ora scrivono anch'essi su
Postgres invece di committare il DB SQLite.

Uso:
    DATABASE_URL=postgresql://... python3 scripts/cron_daily.py
    python3 scripts/cron_daily.py --steps gex,predict        # solo alcuni passi
    python3 scripts/cron_daily.py --calibrate                # forza la calibrazione

Exit code:
    0 — tutti i passi critici (gex, prices, predict, verify, calibrate) riusciti
    1 — almeno un passo critico fallito (i passi di sola verifica non contano)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assets import ASSETS
from src.config import setup_logging
from src.db import ensure_postgres_schema, redact_url, require_postgres

_log = setup_logging("cron_daily")

#: Passi che scrivono su DB: se falliscono il job esce con 1.
CRITICAL = {"gex", "prices", "predict", "verify", "calibrate"}
ALL_STEPS = ["gex", "prices", "coinglass", "flows", "predict", "verify", "calibrate"]


def step_gex() -> str:
    from scripts.cron_gex import snapshot_asset
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_db import GexDB

    db, client = GexDB(), DeribitClient()
    esiti = {a: snapshot_asset(a, client, db) for a in ASSETS}
    falliti = [a for a, ok in esiti.items() if not ok]
    if falliti:
        raise RuntimeError(f"snapshot GEX fallito per {falliti}")
    return f"snapshot salvati per {list(esiti)}"


def step_prices() -> str:
    from src.flows.price_fetcher import PriceFetcher

    fetcher = PriceFetcher()
    righe = {}
    for asset in ASSETS:
        df = fetcher.get_all_prices(asset=asset)
        if df.empty:
            raise RuntimeError(f"nessun prezzo per {asset}")
        righe[asset] = len(df)
    return f"prezzi aggiornati {righe}"


def step_coinglass() -> str:
    from src.flows.coinglass_client import CoinGlassClient

    flows = CoinGlassClient().fetch_etf_flows(days=7)
    return f"CoinGlass raggiungibile: {len(flows)} record flussi ETF (7g)"


def step_flows() -> str:
    from src.flows.scraper import FarsideScraper

    flows = FarsideScraper().fetch()
    if not flows:
        raise RuntimeError("Farside non ha restituito flussi")
    return f"Farside raggiungibile: {len(flows)} record"


def step_predict() -> str:
    from src.config import get_settings
    from src.forecast.jobs import run_daily_predict

    horizon = int(get_settings().get("forecast", {}).get("horizon_days", 5))
    res = run_daily_predict(horizon=horizon)
    if res.get("status") not in ("ok", "skipped_kill_switch"):
        raise RuntimeError(f"predict: {res}")
    return f"predict {res}"


def step_verify() -> str:
    from src.forecast.jobs import run_daily_verify

    return f"verify {run_daily_verify()}"


def step_calibrate() -> str:
    from src.forecast.jobs import run_weekly_calibrate

    rep = run_weekly_calibrate()
    return f"calibrate gate={rep.gate_ok} scored={rep.metrics.get('total_scored')}"


STEP_FUNCS: dict[str, Callable[[], str]] = {
    "gex": step_gex,
    "prices": step_prices,
    "coinglass": step_coinglass,
    "flows": step_flows,
    "predict": step_predict,
    "verify": step_verify,
    "calibrate": step_calibrate,
}


def select_steps(requested: str | None, force_calibrate: bool, now: datetime) -> list[str]:
    """Passi da eseguire: calibrate solo il lunedi' (UTC) salvo richiesta esplicita."""
    if requested:
        steps = [s.strip() for s in requested.split(",") if s.strip()]
        unknown = sorted(set(steps) - set(ALL_STEPS))
        if unknown:
            raise SystemExit(f"passi sconosciuti: {unknown} (validi: {ALL_STEPS})")
        return [s for s in ALL_STEPS if s in steps]
    steps = list(ALL_STEPS)
    if not (force_calibrate or now.weekday() == 0):
        steps.remove("calibrate")
    return steps


def run(steps: list[str]) -> dict[str, tuple[bool, str]]:
    """Esegue i passi in ordine; ogni errore viene loggato e non blocca i successivi."""
    results: dict[str, tuple[bool, str]] = {}
    for name in steps:
        t0 = time.monotonic()
        try:
            msg = STEP_FUNCS[name]()
            results[name] = (True, msg)
            _log.info("[%s] OK in %.1fs — %s", name, time.monotonic() - t0, msg)
        except Exception as exc:  # noqa: BLE001 - il job deve proseguire
            results[name] = (False, f"{type(exc).__name__}: {exc}")
            _log.exception("[%s] FALLITO in %.1fs", name, time.monotonic() - t0)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Job giornaliero raccolta + forecast (Postgres)")
    parser.add_argument("--steps", help=f"sottoinsieme separato da virgole di {ALL_STEPS}")
    parser.add_argument("--calibrate", action="store_true", help="forza la calibrazione")
    args = parser.parse_args(argv)

    url = require_postgres("cron_daily")
    applied = ensure_postgres_schema(url)
    _log.info("DB: %s (migrazioni applicate ora: %s)", redact_url(url), applied or "nessuna")

    steps = select_steps(args.steps, args.calibrate, datetime.now(timezone.utc))
    results = run(steps)

    print("\n== riepilogo cron_daily ==")
    for name, (ok, msg) in results.items():
        print(f"{'OK  ' if ok else 'FAIL'} {name:<10} {msg}")
    failed_critical = [n for n, (ok, _) in results.items() if not ok and n in CRITICAL]
    return 1 if failed_critical else 0


if __name__ == "__main__":
    sys.exit(main())
