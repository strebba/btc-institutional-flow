"""Macro endpoint — funding, open interest, long/short, liquidazioni, taker (CoinGlass + ripiego CoinGecko)."""
from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.cache import asset_key, cache_get, cache_set
from src.api.helpers import AssetParam, ok, http_error

router = APIRouter(tags=["macro"])


@router.get("/api/macro")
def get_macro(asset: AssetParam = "btc") -> JSONResponse:
    import logging
    _log = logging.getLogger("api.macro")

    asset_u = asset.upper()
    cached = cache_get(asset_key("macro", asset_u))
    if cached is not None:
        return cached

    try:
        from src.flows.coinglass_client import CoinGlassClient
        from src.flows.funding import annualize_funding_pct

        cg = CoinGlassClient()

        funding_rate_ann_pct: float | None = None
        funding_rate_8h_pct: float | None = None
        funding_history: list[dict] = []
        try:
            fr_series = cg.fetch_funding_rate_history(days=90, asset=asset_u)
            if not fr_series.empty:
                # CoinGlass restituisce gia' punti percentuali per 8 ore:
                # nessun x100, cfr. src/flows/funding.py
                funding_rate_8h_pct = round(float(fr_series.iloc[-1]), 4)
                funding_rate_ann_pct = round(annualize_funding_pct(float(fr_series.iloc[-1])), 2)
                funding_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "rate_8h_pct": round(float(v), 4)}
                    for ts, v in fr_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Funding rate fetch fallito in /macro: %s", _e)

        oi_latest_usd: float | None = None
        oi_change_7d_pct: float | None = None
        oi_history: list[dict] = []
        try:
            oi_series = cg.fetch_aggregated_oi_history(days=90, asset=asset_u)
            if not oi_series.empty:
                oi_latest_usd = round(float(oi_series.iloc[-1]), 0)
                if len(oi_series) >= 8:
                    oi_7d_ago = float(oi_series.iloc[-8])
                    if oi_7d_ago > 0:
                        oi_change_7d_pct = round(
                            (float(oi_series.iloc[-1]) - oi_7d_ago) / oi_7d_ago * 100, 2)
                oi_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "oi_usd": round(float(v), 0)}
                    for ts, v in oi_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("OI fetch fallito in /macro: %s", _e)

        long_short_ratio_latest: float | None = None
        ls_history: list[dict] = []
        try:
            ls_series = cg.fetch_long_short_ratio(days=90, asset=asset_u)
            if not ls_series.empty:
                long_short_ratio_latest = round(float(ls_series.iloc[-1]), 4)
                ls_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "ratio": round(float(v), 4)}
                    for ts, v in ls_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Long/short ratio fetch fallito in /macro: %s", _e)

        liquidations_long_24h_usd: float | None = None
        liquidations_short_24h_usd: float | None = None
        liquidations_total_24h_usd: float | None = None
        liq_history: list[dict] = []
        try:
            liq_df = cg.fetch_liquidations(days=90, asset=asset_u)
            if not liq_df.empty:
                liquidations_long_24h_usd = round(float(liq_df["long_usd"].iloc[-1]), 0)
                liquidations_short_24h_usd = round(float(liq_df["short_usd"].iloc[-1]), 0)
                liquidations_total_24h_usd = round(float(liq_df["total_usd"].iloc[-1]), 0)
                liq_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "long_usd": round(float(row["long_usd"]), 0),
                     "short_usd": round(float(row["short_usd"]), 0),
                     "total_usd": round(float(row["total_usd"]), 0)}
                    for ts, row in liq_df.tail(90).iterrows()
                ]
        except Exception as _e:
            _log.warning("Liquidations fetch fallito in /macro: %s", _e)

        taker_buy_ratio_latest: float | None = None
        taker_history: list[dict] = []
        try:
            tk_series = cg.fetch_taker_volume(days=90, asset=asset_u)
            if not tk_series.empty:
                taker_buy_ratio_latest = round(float(tk_series.iloc[-1]), 4)
                taker_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "buy_ratio": round(float(v), 4)}
                    for ts, v in tk_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Taker volume fetch fallito in /macro: %s", _e)

        # Perche' i campi sono vuoti: senza questo un consumatore non distingue
        # "nessun dato" da "mercato piatto".
        from src.flows.macro_fetcher import (
            SOURCE_COINGECKO,
            SOURCE_COINGLASS,
            STATUS_NO_API_KEY,
            STATUS_OK,
            STATUS_PARTIAL_COINGECKO,
            STATUS_UNAVAILABLE,
        )

        da_coinglass = any(
            v is not None
            for v in (
                funding_rate_ann_pct, oi_change_7d_pct, long_short_ratio_latest,
                liquidations_long_24h_usd, taker_buy_ratio_latest,
            )
        )
        funding_source = SOURCE_COINGLASS if funding_rate_ann_pct is not None else None

        # Ripiego: CoinGecko copre funding e open interest senza chiave. Non ha
        # storico, quindi le serie restano vuote e la variazione a 7 giorni la
        # ricostruiamo dagli snapshot che accumuliamo noi.
        da_coingecko = False
        if funding_rate_ann_pct is None:
            try:
                from src.flows.coingecko_client import CoinGeckoClient
                from src.flows.funding import funding_pct_8h_from_annual
                from src.flows.macro_fetcher import _oi_change_dallo_storico

                _f, _oi, _n = CoinGeckoClient().fetch_funding_and_oi(asset_u)
                if _f is not None:
                    funding_rate_ann_pct = round(_f, 2)
                    funding_rate_8h_pct = round(funding_pct_8h_from_annual(_f), 4)
                    funding_source = SOURCE_COINGECKO
                    da_coingecko = True
                    if _oi is not None and oi_latest_usd is None:
                        oi_latest_usd = round(_oi, 0)
                    if oi_change_7d_pct is None:
                        oi_change_7d_pct = _oi_change_dallo_storico(None, asset=asset_u)
            except Exception as _e:
                _log.warning("Ripiego CoinGecko fallito in /macro: %s", _e)

        if da_coinglass:
            source_status = STATUS_OK
        elif da_coingecko:
            source_status = STATUS_PARTIAL_COINGECKO
        elif not cg.has_api_key:
            source_status = STATUS_NO_API_KEY
        else:
            source_status = STATUS_UNAVAILABLE

        macro_data = {
            "asset": asset_u,
            "source_status": source_status,
            "funding_source": funding_source,
            "funding_rate_8h_pct": funding_rate_8h_pct,
            "funding_rate_annualized_pct": funding_rate_ann_pct,
            "futures_oi_usd": oi_latest_usd,
            "oi_change_7d_pct": oi_change_7d_pct,
            "long_short_ratio_latest": long_short_ratio_latest,
            "liquidations_long_24h_usd": liquidations_long_24h_usd,
            "liquidations_short_24h_usd": liquidations_short_24h_usd,
            "liquidations_total_24h_usd": liquidations_total_24h_usd,
            "taker_buy_ratio_latest": taker_buy_ratio_latest,
            "history": {
                "funding_rate": funding_history,
                "oi": oi_history,
                "long_short": ls_history,
                "liquidations": liq_history,
                "taker": taker_history,
            },
        }
        response = ok(macro_data)
        cache_set(asset_key("macro", asset_u), response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Macro error: {exc}")
