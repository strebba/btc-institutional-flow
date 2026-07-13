"""GEX endpoint — Gamma Exposure snapshot da Deribit."""
from __future__ import annotations

import traceback

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.cache import cache_get, cache_set, _gex_fetch_lock, _TTL
from src.api.helpers import ok, http_error

router = APIRouter(prefix="/api/gex", tags=["gex"])

# ─── CoinGlass GEX enrichment — cache 1 ora ────────────────────────────────────


def _enrich_gex_with_coinglass(our_call_oi: float, our_put_oi: float) -> dict:
    cached = cache_get("gex_enrichment")
    if cached is not None:
        deribit_oi_contracts = cached.get("_deribit_oi_contracts", 0)
        if deribit_oi_contracts > 0:
            our_total = our_call_oi + our_put_oi
            cov = round(min(our_total / deribit_oi_contracts * 100, 100.0), 1)
            label = "good" if cov >= 80 else "degraded" if cov >= 50 else "poor"
            cached["data_quality"]["coverage_pct"] = cov
            cached["data_quality"]["quality_label"] = label
            cached["data_quality"]["fetched_oi_contracts"] = round(our_total, 1)
        return cached

    from src.flows.coinglass_client import CoinGlassClient

    import logging
    _log = logging.getLogger("api.gex")

    result: dict = {
        "data_quality": {
            "coverage_pct": None,
            "quality_label": "unknown",
            "deribit_oi_usd": None,
            "fetched_oi_contracts": round(our_call_oi + our_put_oi, 1),
        },
        "market_context": {
            "market_pcr": None,
            "market_max_pain": None,
            "exchanges_included": [],
        },
        "_deribit_oi_contracts": 0,
    }

    try:
        cg = CoinGlassClient()
        options_info = cg.fetch_options_info("BTC")
        deribit_info = next(
            (x for x in options_info
             if isinstance(x, dict) and "deribit" in str(x.get("exchange_name", "")).lower()),
            None,
        )
        if deribit_info:
            cg_oi_contracts = float(deribit_info.get("open_interest") or 0)
            cg_oi_usd = float(deribit_info.get("open_interest_usd") or 0)
            result["_deribit_oi_contracts"] = cg_oi_contracts
            result["data_quality"]["deribit_oi_usd"] = round(cg_oi_usd, 0) if cg_oi_usd else None
            if cg_oi_contracts > 0:
                our_total = our_call_oi + our_put_oi
                cov = round(min(our_total / cg_oi_contracts * 100, 100.0), 1)
                label = "good" if cov >= 80 else "degraded" if cov >= 50 else "poor"
                result["data_quality"]["coverage_pct"] = cov
                result["data_quality"]["quality_label"] = label
                result["data_quality"]["fetched_oi_contracts"] = round(our_total, 1)

        _EXCHANGES = ["Deribit", "Bybit", "Binance", "OKX"]
        total_call_notional = 0.0
        total_put_notional = 0.0
        weighted_pain_sum = 0.0
        total_pain_weight = 0.0
        exchanges_with_data: list[str] = []

        for exch in _EXCHANGES:
            mp_data = cg.fetch_options_max_pain("BTC", exch)
            if not mp_data:
                continue
            exchanges_with_data.append(exch)
            for expiry in mp_data:
                if not isinstance(expiry, dict):
                    continue
                call_n = float(expiry.get("call_open_interest_notional") or 0)
                put_n = float(expiry.get("put_open_interest_notional") or 0)
                max_pain = float(expiry.get("max_pain_price") or 0)
                weight = call_n + put_n
                total_call_notional += call_n
                total_put_notional += put_n
                if max_pain > 0 and weight > 0:
                    weighted_pain_sum += max_pain * weight
                    total_pain_weight += weight

        if total_call_notional > 0:
            result["market_context"]["market_pcr"] = round(total_put_notional / total_call_notional, 3)
        if total_pain_weight > 0:
            result["market_context"]["market_max_pain"] = round(weighted_pain_sum / total_pain_weight, 0)
        result["market_context"]["exchanges_included"] = exchanges_with_data

    except Exception as _e:
        import logging
        logging.getLogger("api.gex").warning("CoinGlass GEX enrichment failed: %s", _e)

    cache_set("gex_enrichment", result)
    return result


# ─── GEX shared fetch (dedup lock) ────────────────────────────────────────────


def _get_gex_data() -> dict:
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.gex.gex_db import GexDB
    from src.gex.regime_detector import RegimeDetector

    import logging
    _log = logging.getLogger("api.gex")

    cached = cache_get("_gex_data")
    if cached is not None:
        return cached

    with _gex_fetch_lock:
        cached = cache_get("_gex_data")
        if cached is not None:
            return cached

        client = DeribitClient()
        spot = client.get_spot_price()
        options = client.fetch_all_options("BTC")
        calculator = GexCalculator()
        snapshot = calculator.calculate_gex(options, spot)

        gex_db = GexDB()
        detector = RegimeDetector()
        detector.load_history_from_db(gex_db.get_latest_n(90))
        state = detector.detect(snapshot)
        try:
            gex_db.insert_snapshot(snapshot, state.regime)
        except Exception as _e:
            _log.warning("GEX DB persist failed: %s", _e)

        data = {"snapshot": snapshot, "spot": spot, "state": state, "gex_db": gex_db}
        cache_set("_gex_data", data)
        return data


# ─── GET /api/gex ──────────────────────────────────────────────────────────────


@router.get("")
def get_gex() -> JSONResponse:
    import logging
    _log = logging.getLogger("api.gex")

    cached = cache_get("gex")
    if cached is not None:
        return cached

    try:
        from src.gex.gex_calculator import GexCalculator

        gex_data = _get_gex_data()
        snapshot = gex_data["snapshot"]
        spot = gex_data["spot"]
        state = gex_data["state"]

        calculator = GexCalculator()
        gex_dict = calculator.gex_to_dict(snapshot)

        strike_profile = [
            {
                "strike": gs.strike,
                "net_gex_m": round(gs.net_gex / 1e6, 4),
                "call_gex_m": round(gs.call_gex / 1e6, 4),
                "put_gex_m": round(gs.put_gex / 1e6, 4),
                "call_oi": gs.call_oi,
                "put_oi": gs.put_oi,
            }
            for gs in snapshot.gex_by_strike
            if spot > 0 and abs(gs.strike - spot) / spot < 0.40
        ]

        enrichment = _enrich_gex_with_coinglass(snapshot.total_call_oi, snapshot.total_put_oi)

        response = ok({
            "snapshot": gex_dict,
            "regime": {
                "label": state.regime,
                "alerts": state.alerts,
                "gex_percentile": state.gex_percentile,
            },
            "strike_profile": strike_profile,
            "options_metrics": {
                "put_call_ratio": snapshot.put_call_ratio,
                "max_pain": snapshot.max_pain,
                "distance_to_put_wall_pct": snapshot.distance_to_put_wall_pct,
                "distance_to_call_wall_pct": snapshot.distance_to_call_wall_pct,
                "total_call_oi": snapshot.total_call_oi,
                "total_put_oi": snapshot.total_put_oi,
            },
            "data_quality": enrichment.get("data_quality", {}),
            "market_context": enrichment.get("market_context", {}),
        })
        cache_set("gex", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"GEX error: {exc}")
