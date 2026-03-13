"""FastAPI backend per btc-institutional-flow.

Endpoints:
  GET /api/health    - Health check
  GET /api/gex       - GEX snapshot: regime, gamma flip, walls, profilo per strike
  GET /api/flows     - ETF flows IBIT: serie storica, correlazione, Granger
  GET /api/barriers  - Barriere attive da SEC EDGAR
  GET /api/signals   - Segnale composito LONG/CAUTION/RISK_OFF + Sharpe backtest
  GET /docs          - Swagger UI (FastAPI built-in)
"""
from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any

import json
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─── In-memory TTL cache ───────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}  # key → (timestamp, payload)

_TTL = {
    "gex":      300,   # 5 min  — opzioni Deribit, ~90s fetch
    "flows":    900,   # 15 min — Farside scrape
    "barriers": 3600,  # 1 ora  — dati SEC EDGAR statici
    "signals":  300,   # 5 min  — dipende da gex + flows
}


def _cache_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _TTL.get(key, 300):
        return entry[1]
    return None


def _cache_set(key: str, payload: Any) -> None:
    _cache[key] = (time.time(), payload)

app = FastAPI(
    title="BTC Institutional Flow API",
    description="GEX, ETF flows, SEC barriers, composite signals for BTC institutional analysis.",
    version="1.0.0",
)

# Origini consentite: in produzione imposta CORS_ORIGINS come variabile d'ambiente
# Es: CORS_ORIGINS="https://tua-ptf-dashboard.com,https://www.tua-ptf-dashboard.com"
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8501,https://seashell-app-h7hc4.ondigitalocean.app",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Serializza tipi numpy (int64, float32, bool_, ecc.) in tipi Python nativi."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _ok(data: Any) -> JSONResponse:
    payload = {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "data": data}
    return JSONResponse(content=json.loads(json.dumps(payload, cls=_NumpyEncoder)))


def _error(msg: str, code: int = 500) -> HTTPException:
    return HTTPException(status_code=code, detail=msg)


# ─── /api/health ──────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Health check: verifica che il server sia attivo."""
    return _ok({"service": "btc-institutional-flow", "healthy": True})


# ─── /api/gex ─────────────────────────────────────────────────────────────────

@app.get("/api/gex", tags=["gex"])
def get_gex() -> dict:
    """GEX snapshot: regime, gamma flip, walls, profilo netto per strike.

    Scarica la chain di opzioni BTC da Deribit in tempo reale,
    calcola il Gamma Exposure e classifica il regime corrente.
    Risposta cachata per 5 minuti (fetch Deribit ~90s).
    """
    cached = _cache_get("gex")
    if cached is not None:
        return cached

    try:
        from src.gex.deribit_client import DeribitClient
        from src.gex.gex_calculator import GexCalculator
        from src.gex.regime_detector import RegimeDetector

        client     = DeribitClient()
        spot       = client.get_spot_price()
        options    = client.fetch_all_options("BTC")

        calculator = GexCalculator()
        snapshot   = calculator.calculate_gex(options, spot)
        gex_dict   = calculator.gex_to_dict(snapshot)

        detector   = RegimeDetector()
        state      = detector.detect(snapshot)

        # Profilo per strike (intorno a ±40% dallo spot)
        strike_profile = [
            {
                "strike":   gs.strike,
                "net_gex_m": round(gs.net_gex / 1e6, 4),
                "call_gex_m": round(gs.call_gex / 1e6, 4),
                "put_gex_m":  round(gs.put_gex / 1e6, 4),
                "call_oi":  gs.call_oi,
                "put_oi":   gs.put_oi,
            }
            for gs in snapshot.gex_by_strike
            if abs(gs.strike - spot) / spot < 0.40
        ]

        response = _ok({
            "snapshot": gex_dict,
            "regime": {
                "label":         state.regime,
                "alerts":        state.alerts,
                "gex_percentile": state.gex_percentile,
            },
            "strike_profile": strike_profile,
        })
        _cache_set("gex", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"GEX error: {exc}")


# ─── /api/flows ───────────────────────────────────────────────────────────────

@app.get("/api/flows", tags=["flows"])
def get_flows() -> dict:
    """ETF flows IBIT: serie storica, correlazioni rolling, Granger causality.

    Scarica i flussi da Farside Investors e i prezzi da yfinance,
    calcola correlazioni rolling (30/60/90d) e test di Granger.
    Risposta cachata per 15 minuti.
    """
    cached = _cache_get("flows")
    if cached is not None:
        return cached

    try:
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation
        from src.analytics.granger import GrangerAnalysis

        scraper    = FarsideScraper()
        raw_flows  = scraper.fetch()
        agg_flows  = scraper.aggregate(raw_flows)
        df_pivot   = scraper.to_dataframe(raw_flows)

        fetcher    = PriceFetcher()
        prices     = fetcher.get_all_prices()

        corr_eng   = FlowCorrelation()
        merged     = corr_eng.merge(agg_flows, prices)

        if merged.empty:
            raise ValueError("Merge flussi/prezzi vuoto")

        stats       = corr_eng.summary_stats(merged)
        roll_corrs  = corr_eng.rolling_correlations(merged, windows=[30, 60, 90])

        # Granger causality
        granger_eng = GrangerAnalysis()
        granger_raw = granger_eng.run(merged)
        granger_out: dict[str, list] = {}
        for direction, results in granger_raw.items():
            granger_out[direction] = [
                {
                    "lag":         r.lag,
                    "f_stat":      round(r.f_stat, 4),
                    "p_value":     round(r.p_value, 6),
                    "significant": r.significant,
                }
                for r in results
            ]

        # Serie storica IBIT (ultimi 365 gg) con prezzo BTC
        history: list[dict] = []
        ibit_col = "IBIT" if "IBIT" in df_pivot.columns else None
        if ibit_col:
            recent = df_pivot[ibit_col].dropna().tail(365)
            # Aggiungi btc_close dal merged dataframe (se disponibile)
            btc_prices: dict = {}
            if not merged.empty and "btc_close" in merged.columns:
                for idx, val in merged["btc_close"].dropna().items():
                    btc_prices[str(idx.date())] = float(val)
            history = [
                {
                    "date": str(d.date()),
                    "ibit_flow_usd": float(v),
                    "btc_close": btc_prices.get(str(d.date())),
                }
                for d, v in recent.items()
            ]

        # Rolling correlations: ultima osservazione disponibile per ogni finestra
        corr_latest: dict[str, dict] = {}
        for window_key, corr_df in roll_corrs.items():
            last = corr_df.dropna(how="all")
            if not last.empty:
                row = last.iloc[-1].to_dict()
                corr_latest[window_key] = {
                    k: round(float(v), 4) if v == v else None
                    for k, v in row.items()
                }

        response = _ok({
            "summary": stats,
            "history": history,
            "rolling_correlations_latest": corr_latest,
            "granger": granger_out,
        })
        _cache_set("flows", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Flows error: {exc}")


# ─── /api/barriers ────────────────────────────────────────────────────────────

@app.get("/api/barriers", tags=["barriers"])
def get_barriers() -> dict:
    """Barriere attive da SEC EDGAR (structured notes su IBIT/BTC).

    Restituisce i barrier levels con status='active' dal DB locale.
    Risposta cachata per 1 ora (dati statici).
    """
    cached = _cache_get("barriers")
    if cached is not None:
        return cached

    try:
        from src.edgar.structured_notes_db import StructuredNotesDB
        from src.gex.deribit_client import DeribitClient
        from src.flows.price_fetcher import PriceFetcher

        db = StructuredNotesDB()

        # Calcola dinamicamente level_price_btc per le barriere che non ce l'hanno
        try:
            prices = PriceFetcher().get_all_prices()
            latest = prices[["btc_close", "ibit_close"]].dropna().iloc[-1]
            if latest["ibit_close"] > 0 and latest["btc_close"] > 0:
                ratio = latest["ibit_close"] / latest["btc_close"]
                db.compute_btc_prices(ibit_btc_ratio=ratio)
        except Exception:
            pass  # se il fetch fallisce, usa i valori già nel DB

        barriers = db.get_active_barriers()

        # Spot price per calcolo prossimità
        try:
            spot_price = DeribitClient().get_spot_price()
        except Exception:
            spot_price = None

        # Serializza (sqlite3.Row → dict già fatto da get_active_barriers)
        out = []
        for b in barriers:
            row = dict(b)
            # Converti campi non-serializzabili
            out.append({k: v for k, v in row.items()})

        response = _ok({
            "count":      len(out),
            "barriers":   out,
            "spot_price": spot_price,
        })
        _cache_set("barriers", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Barriers error: {exc}")


# ─── /api/signals ─────────────────────────────────────────────────────────────

@app.get("/api/signals", tags=["signals"])
def get_signals() -> dict:
    """Segnale composito LONG/CAUTION/RISK_OFF + backtest Sharpe.

    Combina GEX regime, IBIT 3-day flow e prossimità barriere
    per generare un segnale operativo e metrics di backtest.
    Risposta cachata per 5 minuti.
    """
    cached = _cache_get("signals")
    if cached is not None:
        return cached

    try:
        from src.gex.deribit_client import DeribitClient
        from src.gex.gex_calculator import GexCalculator
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation
        from src.edgar.structured_notes_db import StructuredNotesDB
        from src.analytics.backtest import Backtest

        # ── GEX ────────────────────────────────────────────────────────────────
        client     = DeribitClient()
        spot       = client.get_spot_price()
        options    = client.fetch_all_options("BTC")
        calculator = GexCalculator()
        snapshot   = calculator.calculate_gex(options, spot)
        total_gex  = snapshot.total_net_gex

        # ── Flussi ─────────────────────────────────────────────────────────────
        scraper   = FarsideScraper()
        raw_flows = scraper.fetch()
        agg_flows = scraper.aggregate(raw_flows)
        fetcher   = PriceFetcher()
        prices    = fetcher.get_all_prices()
        corr_eng  = FlowCorrelation()
        merged    = corr_eng.merge(agg_flows, prices)

        ibit_flow_3d = 0.0
        if not merged.empty and "ibit_flow_3d" in merged.columns:
            last_val = merged["ibit_flow_3d"].dropna()
            if not last_val.empty:
                ibit_flow_3d = float(last_val.iloc[-1])

        # ── Barriere ───────────────────────────────────────────────────────────
        db              = StructuredNotesDB()
        active_barriers = db.get_active_barriers()

        barrier_exclusion_pct = 0.05
        near_barrier = False
        if spot > 0:
            for b in active_barriers:
                bp = b.get("level_price_btc") or 0.0
                if bp > 0 and abs(spot - bp) / spot < barrier_exclusion_pct:
                    near_barrier = True
                    break

        # ── Segnale composito ──────────────────────────────────────────────────
        long_flow_threshold  =  100e6   # +100M USD flow 3d
        short_flow_threshold = -200e6   # -200M USD flow 3d

        if total_gex > 0 and ibit_flow_3d > long_flow_threshold and not near_barrier:
            signal = "LONG"
            signal_reason = (
                f"GEX positivo ({total_gex/1e6:+.1f}M), "
                f"IBIT flow 3d = {ibit_flow_3d/1e6:+.0f}M, "
                f"nessuna barriera entro 5%"
            )
        elif total_gex < 0 and ibit_flow_3d < short_flow_threshold:
            signal = "RISK_OFF"
            signal_reason = (
                f"GEX negativo ({total_gex/1e6:+.1f}M), "
                f"IBIT flow 3d = {ibit_flow_3d/1e6:+.0f}M"
            )
        else:
            signal = "CAUTION"
            reasons = []
            if total_gex <= 0:
                reasons.append(f"GEX={total_gex/1e6:+.1f}M")
            if ibit_flow_3d <= long_flow_threshold:
                reasons.append(f"flow 3d={ibit_flow_3d/1e6:+.0f}M")
            if near_barrier:
                reasons.append("vicino a barriera attiva")
            signal_reason = "Condizioni miste: " + "; ".join(reasons) if reasons else "Condizioni neutrali"

        # ── Backtest ───────────────────────────────────────────────────────────
        backtest_metrics: dict = {}
        if not merged.empty and "btc_return" in merged.columns:
            bt      = Backtest()
            results = bt.run(merged, active_barriers=active_barriers)
            for key, m in results.items():
                backtest_metrics[key] = {
                    "strategy_name":     m.strategy_name,
                    "total_return_pct":  round(m.total_return * 100, 2),
                    "annualized_return_pct": round(m.annualized_return * 100, 2),
                    "sharpe_ratio":      round(m.sharpe_ratio, 3),
                    "max_drawdown_pct":  round(m.max_drawdown * 100, 2),
                    "win_rate_pct":      round(m.win_rate * 100, 2),
                    "profit_factor":     round(m.profit_factor, 3) if m.profit_factor < 1000 else None,
                    "n_trades":          m.n_trades,
                    "days_long":         m.days_long,
                    "days_short":        m.days_short,
                    "days_flat":         m.days_flat,
                }

        response = _ok({
            "signal":         signal,
            "signal_reason":  signal_reason,
            "inputs": {
                "spot_price_usd":   spot,
                "total_gex_usd_m":  round(total_gex / 1e6, 2),
                "ibit_flow_3d_usd_m": round(ibit_flow_3d / 1e6, 2),
                "near_active_barrier": near_barrier,
                "active_barriers_count": len(active_barriers),
            },
            "backtest": backtest_metrics,
        })
        _cache_set("signals", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Signals error: {exc}")
