"""Signals, Pillars Series, Macro, IFI endpoints."""
from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.cache import cache_get, cache_set
from src.api.helpers import ok, http_error
from src.api.routers.gex import _get_gex_data

router = APIRouter(tags=["signals"])


@router.get("/api/pillars/series")
def get_pillars_series(pillar: str = "composite", days: int = 180) -> JSONResponse:
    import logging
    _log = logging.getLogger("api.signals")

    valid = {"composite", "gex", "barrier", "etf_flows", "macro"}
    if pillar not in valid:
        raise http_error(f"pillar non valido: {pillar} (validi: {sorted(valid)})", code=400)

    cache_key = f"pillars_series_{pillar}_{days}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation
        from src.edgar.structured_notes_db import StructuredNotesDB
        from src.analytics.pillars import CompositeSignal

        scraper = FarsideScraper()
        merged = FlowCorrelation().merge(
            scraper.aggregate(scraper.fetch()), PriceFetcher().get_all_prices()
        )
        if "total_flow" in merged.columns and "total_flow_usd" not in merged.columns:
            merged = merged.rename(columns={"total_flow": "total_flow_usd"})

        try:
            from src.gex.gex_db import GexDB
            gex_series = GexDB().get_series(days=max(days, 365))
            if not gex_series.empty:
                merged = merged.join(gex_series.rename("total_net_gex"), how="left")
                merged["total_net_gex"] = merged["total_net_gex"].ffill()
        except Exception as _e:
            _log.warning("GEX series non disponibile per /pillars/series: %s", _e)

        active_barriers = StructuredNotesDB().get_active_barriers()
        scores_df = CompositeSignal().compute_series(merged, active_barriers=active_barriers)

        col = f"{pillar}_score" if pillar != "composite" else "composite_score"
        if col not in scores_df.columns:
            raise http_error(f"colonna {col} non disponibile", code=500)

        series = scores_df[col].dropna().tail(days)
        btc = merged.get("btc_close")
        rows = [
            {
                "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                "score": round(float(v), 2),
                "btc_price": float(btc[ts]) if btc is not None and ts in btc.index else None,
            }
            for ts, v in series.items()
        ]
        response = ok({
            "pillar": pillar,
            "series": rows,
            "current": rows[-1] if rows else None,
            "stats": {"days_available": len(rows)},
        })
        cache_set(cache_key, response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Pillars series error: {exc}")


@router.get("/api/signals")
def get_signals() -> JSONResponse:
    import logging
    _log = logging.getLogger("api.signals")

    cached = cache_get("signals")
    if cached is not None:
        return cached

    try:
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation
        from src.edgar.structured_notes_db import StructuredNotesDB
        from src.analytics.backtest import Backtest
        from src.analytics.pillars import CompositeSignal, CompositeInputs
        from src.flows.coinglass_client import CoinGlassClient

        gex_data = _get_gex_data()
        snapshot = gex_data["snapshot"]
        spot = gex_data["spot"]
        _gex_db = gex_data["gex_db"]
        total_gex = snapshot.total_net_gex

        scraper = FarsideScraper()
        raw_flows = scraper.fetch()
        agg_flows = scraper.aggregate(raw_flows)
        fetcher = PriceFetcher()
        prices = fetcher.get_all_prices()
        corr_eng = FlowCorrelation()
        merged = corr_eng.merge(agg_flows, prices)

        ibit_flow_3d = 0.0
        if not merged.empty and "ibit_flow_3d" in merged.columns:
            last_val = merged["ibit_flow_3d"].dropna()
            if not last_val.empty:
                ibit_flow_3d = float(last_val.iloc[-1])

        db = StructuredNotesDB()
        active_barriers = db.get_active_barriers()

        barrier_exclusion_pct = 0.05
        near_barrier = False
        if spot > 0:
            for b in active_barriers:
                bp = b.get("level_price_btc") or 0.0
                if bp > 0 and abs(spot - bp) / spot < barrier_exclusion_pct:
                    near_barrier = True
                    break

        funding_rate_ann: float | None = None
        oi_change_7d_pct: float | None = None
        long_short_ratio: float | None = None
        liquidations_long: float | None = None
        liquidations_short: float | None = None

        macro_data_cached = cache_get("macro_data")
        if macro_data_cached is not None:
            funding_rate_ann = macro_data_cached.get("funding_rate_annualized_pct")
            oi_change_7d_pct = macro_data_cached.get("oi_change_7d_pct")
            long_short_ratio = macro_data_cached.get("long_short_ratio_latest")
            liquidations_long = macro_data_cached.get("liquidations_long_24h_usd")
            liquidations_short = macro_data_cached.get("liquidations_short_24h_usd")

        _cg: CoinGlassClient | None = None

        if funding_rate_ann is None:
            try:
                _cg = _cg or CoinGlassClient()
                fr_series = _cg.fetch_funding_rate_history(days=14)
                if not fr_series.empty:
                    funding_rate_ann = float(fr_series.iloc[-1]) * 3 * 365 * 100
            except Exception as _e:
                _log.warning("Funding rate fetch fallito in /signals: %s", _e)

        if oi_change_7d_pct is None:
            try:
                _cg = _cg or CoinGlassClient()
                oi_series = _cg.fetch_aggregated_oi_history(days=14)
                if len(oi_series) >= 7:
                    oi_7d_ago = float(oi_series.iloc[-8])
                    oi_now = float(oi_series.iloc[-1])
                    if oi_7d_ago > 0:
                        oi_change_7d_pct = (oi_now - oi_7d_ago) / oi_7d_ago * 100
            except Exception as _e:
                _log.warning("OI change fetch fallito in /signals: %s", _e)

        if long_short_ratio is None:
            try:
                _cg = _cg or CoinGlassClient()
                ls_series = _cg.fetch_long_short_ratio(days=3)
                if not ls_series.empty:
                    long_short_ratio = float(ls_series.iloc[-1])
            except Exception as _e:
                _log.warning("Long/short ratio fetch fallito in /signals: %s", _e)

        if liquidations_long is None:
            try:
                _cg = _cg or CoinGlassClient()
                liq_df = _cg.fetch_liquidations(days=2)
                if not liq_df.empty:
                    liquidations_long = float(liq_df["long_usd"].iloc[-1])
                    liquidations_short = float(liq_df["short_usd"].iloc[-1])
            except Exception as _e:
                _log.warning("Liquidations fetch fallito in /signals: %s", _e)

        flow_is_estimate = bool(raw_flows) and all(
            f.source.startswith("yfinance") for f in raw_flows
        )
        composite_inputs = CompositeInputs(
            gex_usd=total_gex,
            gamma_flip_price=snapshot.gamma_flip_price,
            put_wall=snapshot.put_wall,
            call_wall=snapshot.call_wall,
            active_barriers=active_barriers,
            etf_flow_3d_usd=ibit_flow_3d,
            flow_history_df=merged if not merged.empty else None,
            flow_is_estimate=flow_is_estimate,
            funding_rate_annualized_pct=funding_rate_ann,
            oi_change_7d_pct=oi_change_7d_pct,
            long_short_ratio=long_short_ratio,
            put_call_ratio=snapshot.put_call_ratio,
            liquidations_long_24h_usd=liquidations_long,
            liquidations_short_24h_usd=liquidations_short,
            spot_price=spot,
        )
        composite = CompositeSignal()
        signal_result = composite.compute(composite_inputs)

        try:
            from src.analytics.signal_db import SignalDB
            SignalDB().insert(
                signal_result,
                spot_price_usd=spot,
                total_gex_usd=total_gex,
                ibit_flow_3d_usd=ibit_flow_3d,
                funding_rate_pct=funding_rate_ann,
                oi_change_7d_pct=oi_change_7d_pct,
                long_short_ratio=long_short_ratio,
                put_call_ratio=snapshot.put_call_ratio,
                liq_long_usd=liquidations_long,
                liq_short_usd=liquidations_short,
                near_active_barrier=near_barrier,
            )
        except Exception:
            _log.warning("signal_db insert fallito", exc_info=True)

        backtest_metrics: dict = {}
        equity_curve: list[dict] = []
        if not merged.empty and "btc_return" in merged.columns:
            _gex_series = _gex_db.get_series(days=365)
            bt = Backtest()
            results = bt.run(
                merged,
                gex_series=_gex_series if not _gex_series.empty else None,
                active_barriers=active_barriers,
                composite=composite,
            )
            for key, m in results.items():
                backtest_metrics[key] = {
                    "strategy_name": m.strategy_name,
                    "total_return_pct": round(m.total_return * 100, 2),
                    "annualized_return_pct": round(m.annualized_return * 100, 2),
                    "sharpe_ratio": round(m.sharpe_ratio, 3),
                    "max_drawdown_pct": round(m.max_drawdown * 100, 2),
                    "win_rate_pct": round(m.win_rate * 100, 2),
                    "profit_factor": round(m.profit_factor, 3) if m.profit_factor < 1000 else None,
                    "n_trades": m.n_trades,
                    "days_long": m.days_long,
                    "days_short": m.days_short,
                    "days_flat": m.days_flat,
                }
            if "strategy" in results and not results["strategy"].equity_curve.empty:
                ec = results["strategy"].equity_curve
                bah_ec = results.get("buy_and_hold")
                bah_vals = bah_ec.equity_curve if bah_ec else None
                for ts, val in ec.items():
                    row: dict = {
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "strategy": round(float(val), 4),
                    }
                    if bah_vals is not None and ts in bah_vals.index:
                        row["buy_and_hold"] = round(float(bah_vals[ts]), 4)
                    equity_curve.append(row)

        pillars_out = [
            {"name": p.name, "score": p.score, "weight": round(p.weight, 4),
             "components": p.components, "reason": p.reason}
            for p in signal_result.pillars
        ]

        response = ok({
            "signal": signal_result.signal,
            "signal_reason": signal_result.reason,
            "score": signal_result.score,
            "components": signal_result.legacy_components,
            "weights": signal_result.weights_used,
            "pillars": pillars_out,
            "inputs": {
                "spot_price_usd": spot,
                "total_gex_usd_m": round(total_gex / 1e6, 2),
                "ibit_flow_3d_usd_m": round(ibit_flow_3d / 1e6, 2),
                "funding_rate_annualized_pct": funding_rate_ann,
                "oi_change_7d_pct": oi_change_7d_pct,
                "long_short_ratio": long_short_ratio,
                "put_call_ratio": snapshot.put_call_ratio,
                "liquidations_long_24h_usd": liquidations_long,
                "liquidations_short_24h_usd": liquidations_short,
                "near_active_barrier": near_barrier,
                "active_barriers_count": len(active_barriers),
            },
            "backtest": backtest_metrics,
            "equity_curve": equity_curve,
        })
        cache_set("signals", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Signals error: {exc}")


@router.get("/api/macro", tags=["macro"])
def get_macro() -> JSONResponse:
    import logging
    _log = logging.getLogger("api.signals")

    cached = cache_get("macro")
    if cached is not None:
        return cached

    try:
        from src.flows.coinglass_client import CoinGlassClient

        cg = CoinGlassClient()

        funding_rate_ann_pct: float | None = None
        funding_rate_8h_pct: float | None = None
        funding_history: list[dict] = []
        try:
            fr_series = cg.fetch_funding_rate_history(days=90)
            if not fr_series.empty:
                funding_rate_8h_pct = round(float(fr_series.iloc[-1]) * 100, 4)
                funding_rate_ann_pct = round(float(fr_series.iloc[-1]) * 3 * 365 * 100, 2)
                funding_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "rate_8h_pct": round(float(v) * 100, 4)}
                    for ts, v in fr_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Funding rate fetch fallito in /macro: %s", _e)

        oi_latest_usd: float | None = None
        oi_change_7d_pct: float | None = None
        oi_history: list[dict] = []
        try:
            oi_series = cg.fetch_aggregated_oi_history(days=90)
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
            ls_series = cg.fetch_long_short_ratio(days=90)
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
            liq_df = cg.fetch_liquidations(days=90)
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
            tk_series = cg.fetch_taker_volume(days=90)
            if not tk_series.empty:
                taker_buy_ratio_latest = round(float(tk_series.iloc[-1]), 4)
                taker_history = [
                    {"date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                     "buy_ratio": round(float(v), 4)}
                    for ts, v in tk_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Taker volume fetch fallito in /macro: %s", _e)

        macro_data = {
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
        cache_set("macro_data", macro_data)
        response = ok(macro_data)
        cache_set("macro", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Macro error: {exc}")
