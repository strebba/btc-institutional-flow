"""Cron script: aggiorna l'Institutional Flow Index nel DB.

Uso:
    python3 scripts/cron_ifi.py              # aggiorna il giorno corrente
    python3 scripts/cron_ifi.py --backfill   # backfill tutta la storia disponibile
    python3 scripts/cron_ifi.py --days 7     # aggiorna ultimi N giorni

Scheduling suggerito (crontab, ogni giorno alle 22:00 UTC):
    0 22 * * *  /path/venv/bin/python3 /path/scripts/cron_ifi.py

Exit code:
    0 — IFI aggiornato con successo
    1 — fetch dati fallito
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.ifi import (
    IFIModel,
    _score_flow_momentum,
    _score_flow_trend,
    _score_price_momentum,
    _score_funding,
    _score_oi_momentum,
    _score_ls_squeeze,
)
from src.analytics.ifi_db import IFIDb
from src.config import setup_logging
from src.flows.correlation import FlowCorrelation
from src.flows.price_fetcher import PriceFetcher
from src.flows.scraper import FarsideScraper

_log = setup_logging("cron_ifi")

# Finestra massima per backfill (Farside ha storia illimitata, CoinGlass fino a 500gg)
_BACKFILL_DAYS = 520


def _build_df(fetch_days: int) -> pd.DataFrame:
    """Costruisce il DataFrame base (flows + prezzi + CoinGlass opzionale)."""
    scraper = FarsideScraper()
    raw     = scraper.fetch()
    agg     = scraper.aggregate(raw)

    fetcher = PriceFetcher()
    prices  = fetcher.get_all_prices()

    corr   = FlowCorrelation()
    merged = corr.merge(agg, prices)

    if merged.empty:
        raise ValueError("merged DataFrame vuoto — verifica scraper e price fetcher")

    # Rinomina total_flow → total_flow_usd per coerenza con IFIModel
    if "total_flow" in merged.columns and "total_flow_usd" not in merged.columns:
        merged = merged.rename(columns={"total_flow": "total_flow_usd"})

    # ── CoinGlass (opzionale) ─────────────────────────────────────────────────
    cg_days = min(fetch_days, 500)
    try:
        from src.flows.coinglass_client import CoinGlassClient
        cg = CoinGlassClient()

        try:
            fr = cg.fetch_funding_rate_history(days=min(cg_days, 333))
            if not fr.empty:
                fr_ann = (fr * 3 * 365 * 100).rename("funding_rate")
                fr_daily = fr_ann.resample("D").last()
                fr_daily.index = fr_daily.index.tz_localize(None).normalize()
                merged = merged.join(fr_daily, how="left")
        except Exception as e:
            _log.warning("Funding rate fetch fallito: %s", e)

        try:
            oi = cg.fetch_aggregated_oi_history(days=cg_days)
            if not oi.empty:
                oi_daily = oi.resample("D").last().rename("oi_usd")
                oi_daily.index = oi_daily.index.tz_localize(None).normalize()
                merged = merged.join(oi_daily, how="left")
        except Exception as e:
            _log.warning("OI fetch fallito: %s", e)

        try:
            ls = cg.fetch_long_short_ratio(days=cg_days)
            if not ls.empty:
                ls_daily = ls.resample("D").last().rename("long_short_ratio")
                ls_daily.index = ls_daily.index.tz_localize(None).normalize()
                merged = merged.join(ls_daily, how="left")
        except Exception as e:
            _log.warning("L/S ratio fetch fallito: %s", e)

    except Exception as e:
        _log.warning("CoinGlass non disponibile: %s", e)

    return merged


def _build_factor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ricostruisce il factor DataFrame per il storage dei componenti."""
    factors: dict[str, pd.Series] = {}

    flow = df.get("total_flow_usd") or df.get("total_flow")
    if flow is not None and flow.notna().sum() >= 30:
        factors["flow_momentum"] = _score_flow_momentum(flow.fillna(0.0))
        factors["flow_trend"]    = _score_flow_trend(flow.fillna(0.0))

    if "btc_close" in df.columns and "btc_vol_7d" in df.columns:
        factors["price_momentum"] = _score_price_momentum(df["btc_close"], df["btc_vol_7d"])

    if "funding_rate" in df.columns:
        factors["funding"] = _score_funding(df["funding_rate"])
    if "oi_usd" in df.columns:
        factors["oi_momentum"] = _score_oi_momentum(df["oi_usd"])
    if "long_short_ratio" in df.columns:
        factors["ls_squeeze"] = _score_ls_squeeze(df["long_short_ratio"])

    return pd.DataFrame(factors, index=df.index) if factors else pd.DataFrame(index=df.index)


def run(backfill: bool = False, days: int = 1) -> int:
    """Aggiorna IFI DB. Restituisce 0 (ok) o 1 (errore)."""
    db = IFIDb()

    # Buffer extra per le rolling windows (90gg per flow_momentum)
    fetch_days = _BACKFILL_DAYS if backfill else max(days + 100, 150)

    _log.info("Fetch dati (fetch_days=%d, backfill=%s)...", fetch_days, backfill)
    try:
        df = _build_df(fetch_days)
    except Exception as exc:
        _log.error("Build DataFrame fallito: %s", exc)
        return 1

    _log.info(
        "DataFrame: %d righe da %s a %s",
        len(df),
        df.index.min().date() if not df.empty else "?",
        df.index.max().date() if not df.empty else "?",
    )

    model  = IFIModel()
    scores = model.compute_series(df)

    # Seleziona il sottoinsieme da persistere
    if backfill:
        # Tutti i giorni con rolling window sufficiente (skip prime 90 righe)
        subset = scores.iloc[90:]
    else:
        subset = scores.iloc[-max(days, 1):]

    factor_df  = _build_factor_df(df)
    btc_prices = df.get("btc_close")
    flows      = df.get("total_flow_usd") or df.get("total_flow")

    n = db.upsert_series(subset, factor_df if not factor_df.empty else None, btc_prices, flows)
    total = db.count()
    _log.info("IFI: %d righe salvate. Totale nel DB: %d", n, total)
    print(f"[OK] IFI aggiornato: +{n} righe | totale={total} | score_oggi={scores.iloc[-1]:.1f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggiorna Institutional Flow Index DB")
    parser.add_argument("--backfill", action="store_true", help="Ricalcola tutta la storia disponibile")
    parser.add_argument("--days",     type=int, default=1, help="Giorni da aggiornare (default 1)")
    args = parser.parse_args()

    sys.exit(run(backfill=args.backfill, days=args.days))
