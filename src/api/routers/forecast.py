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
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        from src.flows.price_fetcher import PriceFetcher
        from src.forecast.prediction_db import PredictionDB
        from src.forecast.verifier import score_due_predictions

        db = PredictionDB()
        fetcher = PriceFetcher()
        from src.config import TICKER_MAP

        def provider(asset, start, end):
            return fetcher.fetch(
                TICKER_MAP.get(asset, asset),
                start_date=start.date(),
                end_date=(end + _td(days=1)).date(),
            )

        outcomes = score_due_predictions(db, provider, _dt.now(_tz.utc))
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
