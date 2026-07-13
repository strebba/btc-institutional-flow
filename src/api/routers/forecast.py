"""Forecast spine endpoints: predictions, verification, calibration, governance."""
from __future__ import annotations

import traceback

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.helpers import ok, http_error

router = APIRouter(tags=["forecast"])


@router.get("/api/predictions")
def get_predictions(limit: int = 50, days: int = 180) -> JSONResponse:
    try:
        from src.forecast.prediction_db import PredictionDB
        rows = PredictionDB().get_with_outcomes(days=days)
        rows = rows[-limit:]
        return ok({"count": len(rows), "predictions": rows})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"predictions error: {exc}")


@router.post("/api/predictions/{prediction_id}/review")
async def review_prediction(prediction_id: int, request: Request) -> JSONResponse:
    try:
        from src.forecast.prediction_db import PredictionDB
        body = await request.json()
        PredictionDB().update_human_fields(
            prediction_id,
            counter_analysis=body.get("counter_analysis"),
            human_overlay=body.get("human_overlay"),
            confidence=body.get("confidence"),
        )
        return ok({"updated": prediction_id})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"review error: {exc}")


@router.post("/api/predictions/verify")
def verify_predictions() -> JSONResponse:
    try:
        from datetime import datetime as _dt, timedelta as _td
        from src.flows.price_fetcher import PriceFetcher
        from src.forecast.prediction_db import PredictionDB
        from src.forecast.verifier import score_due_predictions

        db = PredictionDB()
        fetcher = PriceFetcher()
        _ticker = {"BTC": "BTC-USD"}

        def provider(asset, start, end):
            return fetcher.fetch(
                _ticker.get(asset, asset),
                start_date=start.date(),
                end_date=(end + _td(days=1)).date(),
            )

        outcomes = score_due_predictions(db, provider, _dt.utcnow())
        hits = sum(1 for o in outcomes if o.hit)
        return ok({"verified": len(outcomes), "hit": hits, "miss": len(outcomes) - hits})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"verify error: {exc}")


@router.get("/api/calibration")
def get_calibration(days: int = 180) -> JSONResponse:
    try:
        from collections import defaultdict
        from src.forecast.prediction_db import PredictionDB

        db = PredictionDB()
        rows = db.get_with_outcomes(days=days)
        agg: dict = defaultdict(lambda: {"n": 0, "scored": 0, "hits": 0, "brier_sum": 0.0, "brier_n": 0})
        for r in rows:
            key = f"{r['source']}/{r['target_type']}"
            a = agg[key]
            a["n"] += 1
            if r.get("hit") is not None:
                a["scored"] += 1
                a["hits"] += int(r["hit"])
                if r.get("brier") is not None:
                    a["brier_sum"] += float(r["brier"])
                    a["brier_n"] += 1

        summary = {}
        for key, a in agg.items():
            summary[key] = {
                "predictions": a["n"],
                "scored": a["scored"],
                "open": a["n"] - a["scored"],
                "hit_rate": round(a["hits"] / a["scored"], 3) if a["scored"] else None,
                "mean_brier": round(a["brier_sum"] / a["brier_n"], 4) if a["brier_n"] else None,
            }

        from src.forecast.sources.dealer_flow import SOURCE as _DF
        active = db.get_active_weights(_DF)
        weights = {
            "active": {"version": active[0], "weights": active[1]} if active else None,
            "proposed": db.get_proposed_weights(_DF),
        }
        return ok({"window_days": days, "by_source_target": summary, "dealer_flow_weights": weights})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"calibration error: {exc}")


@router.get("/api/forecast/status")
def forecast_status() -> JSONResponse:
    try:
        from datetime import datetime, timezone
        from src.forecast.calibration import load_weights_config
        from src.forecast.prediction_db import PredictionDB

        db = PredictionDB()
        recent = db.get_recent(limit=1)
        last = recent[0].created_at if recent else None

        fresh = None
        if last:
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(last).replace(tzinfo=timezone.utc)).total_seconds() / 3600
            from src.config import get_settings
            max_h = float(get_settings().get("forecast", {}).get("freshness_max_hours", 30))
            fresh = age_h <= max_h

        gov = load_weights_config().get("governance", {})
        from src.api.scheduler import _forecast_scheduler
        return ok({
            "last_prediction": last,
            "fresh": fresh,
            "open": len(db.get_open()),
            "total": db.count(),
            "kill_switch": bool(gov.get("kill_switch", False)),
            "freeze_weights": bool(gov.get("freeze_weights", True)),
            "scheduler_running": _forecast_scheduler is not None,
        })
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"forecast status error: {exc}")


@router.post("/api/weights/{version_id}/activate")
async def activate_weights(version_id: int, request: Request) -> JSONResponse:
    try:
        from src.forecast.prediction_db import PredictionDB
        body = await request.json() if await request.body() else {}
        source = body.get("source", "dealer_flow")
        db = PredictionDB()
        db.activate_weight_version(version_id, source)
        active = db.get_active_weights(source)
        return ok({"activated": version_id, "source": source,
                   "active_weights": active[1] if active else None})
    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"activate weights error: {exc}")


# ─── Deprecated IFI endpoint ──────────────────────────────────────────────────


@router.get("/api/ifi", tags=["deprecated"])
def get_ifi() -> JSONResponse:
    import logging
    _log = logging.getLogger("api.signals")

    from src.api.cache import cache_get, cache_set
    cached = cache_get("ifi")
    if cached is not None:
        return cached

    try:
        from src.analytics.ifi import IFIModel, regime_label
        from src.analytics.ifi_db import IFIDb
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation

        db = IFIDb()
        series_df = db.get_series(days=520)
        current = db.get_latest()

        if series_df.empty:
            _log.info("IFI DB vuoto — calcolo on-the-fly (solo flows+prezzo)")
            scraper = FarsideScraper()
            raw = scraper.fetch()
            agg = scraper.aggregate(raw)
            fetcher = PriceFetcher()
            prices = fetcher.get_all_prices()
            corr = FlowCorrelation()
            merged = corr.merge(agg, prices)

            if "total_flow" in merged.columns:
                merged = merged.rename(columns={"total_flow": "total_flow_usd"})

            model = IFIModel()
            scores = model.compute_series(merged)
            scores = scores.iloc[90:]

            btc_prices = merged.get("btc_close")
            flows = merged.get("total_flow_usd")

            series_rows = []
            for ts, score in scores.items():
                date = str(ts.date()) if hasattr(ts, "date") else str(ts)
                btc = float(btc_prices[ts]) if btc_prices is not None and ts in btc_prices.index else None
                flow = float(flows[ts]) if flows is not None and ts in flows.index else None
                series_rows.append({
                    "date": date,
                    "score": round(float(score), 2),
                    "regime": regime_label(float(score)),
                    "btc_price": btc,
                    "total_flow_usd_m": round(flow / 1e6, 2) if flow is not None and flow == flow else None,
                })

            current_score = float(scores.iloc[-1]) if not scores.empty else 50.0
            current_regime = regime_label(current_score)
            current_date = str(scores.index[-1].date()) if not scores.empty else ""

            response = ok({
                "current": {"score": current_score, "regime": current_regime, "date": current_date},
                "series": series_rows,
                "stats": {"days_available": len(series_rows), "data_start": series_rows[0]["date"] if series_rows else None,
                          "source": "computed_on_the_fly"},
            })
            cache_set("ifi", response)
            return response

        series_rows = []
        for ts, row in series_df.iterrows():
            date = str(ts.date()) if hasattr(ts, "date") else str(ts)
            flow = row.get("total_flow_usd")
            series_rows.append({
                "date": date,
                "score": round(float(row["score"]), 2),
                "regime": row["regime"],
                "btc_price": row.get("btc_price"),
                "total_flow_usd_m": round(float(flow) / 1e6, 2) if flow is not None and flow == flow else None,
            })

        c_score = float(current["score"]) if current else 50.0
        c_regime = current["regime"] if current else "Neutral"
        c_date = current["date"] if current else ""

        response = ok({
            "current": {"score": c_score, "regime": c_regime, "date": c_date},
            "series": series_rows,
            "stats": {"days_available": len(series_rows),
                      "data_start": series_rows[0]["date"] if series_rows else None, "source": "db"},
        })
        cache_set("ifi", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"IFI error: {exc}")
