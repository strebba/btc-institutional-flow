"""Logica di aggiornamento IFI DB — condivisa tra cron_ifi.py e il startup di main.py.

Esportazioni pubbliche:
    run(backfill, days) → int   0=ok, 1=errore
    BACKFILL_DAYS               finestra massima per backfill
"""
from __future__ import annotations

import pandas as pd

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

_log = setup_logging("analytics.ifi_updater")

BACKFILL_DAYS = 520


def _build_df(fetch_days: int) -> pd.DataFrame:
    scraper = FarsideScraper()
    raw = scraper.fetch()
    agg = scraper.aggregate(raw)

    fetcher = PriceFetcher()
    prices = fetcher.get_all_prices()

    corr = FlowCorrelation()
    merged = corr.merge(agg, prices)

    if merged.empty:
        raise ValueError("merged DataFrame vuoto — verifica scraper e price fetcher")

    if "total_flow" in merged.columns and "total_flow_usd" not in merged.columns:
        merged = merged.rename(columns={"total_flow": "total_flow_usd"})

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
    factors: dict[str, pd.Series] = {}

    flow = df.get("total_flow_usd") or df.get("total_flow")
    if flow is not None and flow.notna().sum() >= 30:
        factors["flow_momentum"] = _score_flow_momentum(flow.fillna(0.0))
        factors["flow_trend"] = _score_flow_trend(flow.fillna(0.0))

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
    fetch_days = BACKFILL_DAYS if backfill else max(days + 100, 150)

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

    model = IFIModel()
    scores = model.compute_series(df)

    if backfill:
        subset = scores.iloc[90:]
    else:
        subset = scores.iloc[-max(days, 1):]

    factor_df = _build_factor_df(df)
    btc_prices = df.get("btc_close")
    flows = df.get("total_flow_usd") or df.get("total_flow")

    n = db.upsert_series(subset, factor_df if not factor_df.empty else None, btc_prices, flows)
    total = db.count()
    _log.info("IFI: %d righe salvate. Totale nel DB: %d", n, total)
    return 0
