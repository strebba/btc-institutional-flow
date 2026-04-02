"""FastAPI backend per btc-institutional-flow.

Endpoints:
  GET /api/health    - Health check
  GET /api/gex       - GEX snapshot: regime, gamma flip, walls, profilo per strike
  GET /api/flows     - ETF flows IBIT: serie storica, correlazione, Granger
  GET /api/barriers  - Barriere attive da SEC EDGAR
  GET /api/signals   - Segnale composito LONG/CAUTION/RISK_OFF + Sharpe backtest
  GET /api/macro     - Indicatori macro: funding rate, OI, long/short ratio, liquidazioni
  GET /docs          - Swagger UI (FastAPI built-in)
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from typing import Any

import json
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.config import setup_logging

_log = setup_logging("api.main")

# ─── In-memory TTL cache ───────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}  # key → (timestamp, payload)

_TTL = {
    "gex": 300,  # 5 min  — opzioni Deribit, ~90s fetch
    "_gex_data": 300,  # 5 min  — raw GexSnapshot objects (condivisi tra /gex e /signals)
    "flows": 900,  # 15 min — Farside scrape
    "barriers": 3600,  # 1 ora  — dati SEC EDGAR statici
    "signals": 300,  # 5 min  — dipende da gex + flows
    "macro": 3600,  # 1 ora  — dati CoinGlass giornalieri
}

# Lock che impedisce fetch Deribit concorrenti: il secondo richiedente attende
# il primo e poi legge dalla cache invece di lanciare un nuovo fetch da 888 opzioni.
_gex_fetch_lock = threading.Lock()


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

# Autenticazione opzionale via API key — impostare API_KEY env var per abilitarla.
# Se non impostata, tutti gli accessi sono consentiti (utile per sviluppo locale).
_API_KEY = os.getenv("API_KEY")


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    if _API_KEY and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        if request.headers.get("X-API-Key") != _API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ─── Helpers ──────────────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """Serializza tipi numpy e float NaN/Inf in tipi Python JSON-compatibili."""

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

    def iterencode(self, obj: Any, _one_shot: bool = False):
        # Converte float NaN/Inf nativi Python in None prima della serializzazione
        obj = _sanitize(obj)
        return super().iterencode(obj, _one_shot)


def _sanitize(obj: Any) -> Any:
    """Converte ricorsivamente float NaN/Inf in None per JSON compliance."""
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _ok(data: Any) -> JSONResponse:
    payload = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "data": data}
    return JSONResponse(content=json.loads(json.dumps(payload, cls=_NumpyEncoder)))


def _error(msg: str, code: int = 500) -> HTTPException:
    return HTTPException(status_code=code, detail=msg)


# ─── /api/health ──────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Health check: verifica che il server sia attivo."""
    return _ok({"service": "btc-institutional-flow", "healthy": True})


# ─── GEX shared fetch (dedup lock) ────────────────────────────────────────────


def _get_gex_data() -> dict:
    """Fetch GEX objects con lock anti-concorrenza.

    Garantisce che un solo fetch Deribit sia in corso alla volta: il secondo
    chiamante (es. /signals che arriva mentre /gex sta già fetchando) attende
    il lock e poi trova i dati già in cache.

    Returns:
        dict con chiavi: snapshot, spot, state, gex_db
    """
    from src.gex.deribit_client import DeribitClient
    from src.gex.gex_calculator import GexCalculator
    from src.gex.gex_db import GexDB
    from src.gex.regime_detector import RegimeDetector

    # Fast path — nessun lock necessario
    cached = _cache_get("_gex_data")
    if cached is not None:
        return cached

    with _gex_fetch_lock:
        # Double-check: un altro thread potrebbe aver popolato la cache mentre aspettavamo
        cached = _cache_get("_gex_data")
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
        _cache_set("_gex_data", data)
        return data


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
        from src.gex.gex_calculator import GexCalculator

        gex_data = _get_gex_data()
        snapshot = gex_data["snapshot"]
        spot = gex_data["spot"]
        state = gex_data["state"]

        calculator = GexCalculator()
        gex_dict = calculator.gex_to_dict(snapshot)

        # Profilo per strike (intorno a ±40% dallo spot)
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
            if abs(gs.strike - spot) / spot < 0.40
        ]

        response = _ok(
            {
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
            }
        )
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

        scraper = FarsideScraper()
        raw_flows = scraper.fetch()
        agg_flows = scraper.aggregate(raw_flows)
        df_pivot = scraper.to_dataframe(raw_flows)

        fetcher = PriceFetcher()
        prices = fetcher.get_all_prices()

        corr_eng = FlowCorrelation()
        merged = corr_eng.merge(agg_flows, prices)

        if merged.empty:
            raise ValueError("Merge flussi/prezzi vuoto")

        stats = corr_eng.summary_stats(merged)
        roll_corrs = corr_eng.rolling_correlations(merged, windows=[30, 60, 90])

        # Granger causality
        granger_eng = GrangerAnalysis()
        granger_raw = granger_eng.run(merged)
        granger_out: dict[str, list] = {}
        for direction, results in granger_raw.items():
            granger_out[direction] = [
                {
                    "lag": r.lag,
                    "f_stat": round(r.f_stat, 4),
                    "p_value": round(r.p_value, 6),
                    "significant": r.significant,
                }
                for r in results
            ]

        # ── Serie storica per-ticker + totale (ultimi 365gg) ──────────────
        # Costruisci dizionari lookup da merged (btc_close, btc_vol_7d, ibit_btc_ratio)
        btc_prices: dict[str, float] = {}
        btc_vols: dict[str, float] = {}
        ibit_btc_vals: dict[str, float] = {}
        total_flow_vals: dict[str, float] = {}
        if not merged.empty:
            for col, target in [
                ("btc_close", btc_prices),
                ("btc_vol_7d", btc_vols),
                ("ibit_btc_ratio", ibit_btc_vals),
                ("total_flow", total_flow_vals),
            ]:
                if col in merged.columns:
                    for idx, val in merged[col].dropna().items():
                        target[str(idx.date())] = float(val)

        # Detect all ETF tickers from df_pivot (exclude 'total' and 'date')
        all_etf_tickers = [tk for tk in df_pivot.columns if tk.lower() not in ("total", "date")]
        ticker_series: dict[str, dict[str, float]] = {}
        for tk in all_etf_tickers:
            ticker_series[tk] = {
                str(d.date()): float(v) for d, v in df_pivot[tk].dropna().tail(365).items()
            }

        # Unifica tutto in history (365gg, tutti i ticker disponibili)
        history: list[dict] = []
        # Use IBIT as primary date anchor if available, otherwise use total_flow dates
        primary_series = ticker_series.get("IBIT", {})
        if not primary_series:
            # Fallback: use the first available ticker with data
            for tk in all_etf_tickers:
                if ticker_series.get(tk):
                    primary_series = ticker_series[tk]
                    break

        all_dates = sorted(set(primary_series) | set(total_flow_vals), reverse=False)[-365:]
        for d in all_dates:
            row: dict = {"date": d}
            # Primary ETF (IBIT if available)
            primary_ticker = (
                "IBIT"
                if "IBIT" in ticker_series
                else (all_etf_tickers[0] if all_etf_tickers else None)
            )
            if primary_ticker:
                row[f"{primary_ticker.lower()}_flow_usd"] = ticker_series.get(
                    primary_ticker, {}
                ).get(d)
            row["total_flow_usd"] = total_flow_vals.get(d)
            row["btc_close"] = btc_prices.get(d)
            row["btc_vol_7d"] = btc_vols.get(d)
            row["ibit_btc_ratio"] = ibit_btc_vals.get(d)
            # Flussi per-ticker extra
            for tk in all_etf_tickers:
                if tk != primary_ticker:
                    row[f"{tk.lower()}_flow_usd"] = ticker_series.get(tk, {}).get(d)
            history.append(row)

        # Rolling correlations: ultima osservazione disponibile per ogni finestra
        corr_latest: dict[str, dict] = {}
        for window_key, corr_df in roll_corrs.items():
            last = corr_df.dropna(how="all")
            if not last.empty:
                row = last.iloc[-1].to_dict()
                corr_latest[window_key] = {
                    k: round(float(v), 4) if v == v else None for k, v in row.items()
                }

        response = _ok(
            {
                "summary": stats,
                "history": history,
                "rolling_correlations_latest": corr_latest,
                "granger": granger_out,
            }
        )
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
        except Exception as _e:
            _log.warning("IBIT/BTC ratio fetch fallito, uso valori esistenti nel DB: %s", _e)

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

        response = _ok(
            {
                "count": len(out),
                "barriers": out,
                "spot_price": spot_price,
            }
        )
        _cache_set("barriers", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Barriers error: {exc}")


# ─── /api/signals ─────────────────────────────────────────────────────────────


@app.get("/api/signals", tags=["signals"])
def get_signals() -> dict:
    """Segnale composito LONG/CAUTION/RISK_OFF + backtest Sharpe.

    Usa il modello multi-fattore (7 input, scoring 0-100):
    GEX regime, ETF flow 3d, funding rate, OI change, long/short ratio,
    put/call ratio, liquidazioni. Punteggio ≥65 = LONG, <40 = RISK_OFF.
    Risposta cachata per 5 minuti.
    """
    cached = _cache_get("signals")
    if cached is not None:
        return cached

    try:
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation
        from src.edgar.structured_notes_db import StructuredNotesDB
        from src.analytics.backtest import Backtest
        from src.analytics.signal_model import SignalModel, SignalInputs
        from src.flows.coinglass_client import CoinGlassClient

        # ── GEX — riusa il fetch già in cache (o attende il lock) ──────────────
        gex_data = _get_gex_data()
        snapshot = gex_data["snapshot"]
        spot = gex_data["spot"]
        _gex_db = gex_data["gex_db"]
        total_gex = snapshot.total_net_gex

        # ── Flussi ─────────────────────────────────────────────────────────────
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

        # ── Barriere ───────────────────────────────────────────────────────────
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

        # ── Macro CoinGlass (dalla cache macro se disponibile) ─────────────────
        funding_rate_ann: float | None = None
        oi_change_7d_pct: float | None = None
        long_short_ratio: float | None = None
        liquidations_long: float | None = None
        liquidations_short: float | None = None

        # Legge dalla cache del dict macro (non dalla JSONResponse)
        macro_data_cached = _cache_get("macro_data")
        if macro_data_cached is not None:
            funding_rate_ann = macro_data_cached.get("funding_rate_annualized_pct")
            oi_change_7d_pct = macro_data_cached.get("oi_change_7d_pct")
            long_short_ratio = macro_data_cached.get("long_short_ratio_latest")
            liquidations_long = macro_data_cached.get("liquidations_long_24h_usd")
            liquidations_short = macro_data_cached.get("liquidations_short_24h_usd")

        # Istanza singola condivisa tra tutti i fetch CoinGlass di questo endpoint
        _cg: CoinGlassClient | None = None

        if funding_rate_ann is None:
            # Fetch diretto senza cache (per primo avvio)
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

        # ── SignalModel multi-fattore ──────────────────────────────────────────
        signal_inputs = SignalInputs(
            gex_usd=total_gex,
            etf_flow_3d_usd=ibit_flow_3d,
            funding_rate_annualized_pct=funding_rate_ann,
            oi_change_7d_pct=oi_change_7d_pct,
            long_short_ratio=long_short_ratio,
            put_call_ratio=snapshot.put_call_ratio,
            liquidations_long_24h_usd=liquidations_long,
            liquidations_short_24h_usd=liquidations_short,
            near_active_barrier=near_barrier,
        )
        model = SignalModel()
        signal_result = model.compute(signal_inputs)

        # ── Backtest ───────────────────────────────────────────────────────────
        backtest_metrics: dict = {}
        equity_curve: list[dict] = []
        if not merged.empty and "btc_return" in merged.columns:
            _gex_series = _gex_db.get_series(days=365)
            bt = Backtest()
            results = bt.run(
                merged,
                gex_series=_gex_series if not _gex_series.empty else None,
                active_barriers=active_barriers,
                signal_model=model,
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
            # Equity curve per il frontend (strategia principale)
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

        response = _ok(
            {
                "signal": signal_result.signal,
                "signal_reason": signal_result.reason,
                "score": signal_result.score,
                "components": signal_result.components,
                "weights": signal_result.weights_used,
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
            }
        )
        _cache_set("signals", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Signals error: {exc}")


# ─── /api/macro ───────────────────────────────────────────────────────────────


@app.get("/api/macro", tags=["macro"])
def get_macro() -> dict:
    """Indicatori macro da CoinGlass: funding rate, OI, long/short, liquidazioni.

    Restituisce i principali indicatori di sentiment e posizionamento
    del mercato futures BTC. Risposta cachata per 1 ora (dati giornalieri).
    """
    cached = _cache_get("macro")
    if cached is not None:
        return cached

    try:
        from src.flows.coinglass_client import CoinGlassClient

        cg = CoinGlassClient()

        # ── Funding rate ───────────────────────────────────────────────────────
        funding_rate_ann_pct: float | None = None
        funding_rate_8h_pct: float | None = None
        funding_history: list[dict] = []
        try:
            fr_series = cg.fetch_funding_rate_history(days=90)
            if not fr_series.empty:
                funding_rate_8h_pct = round(float(fr_series.iloc[-1]) * 100, 4)
                funding_rate_ann_pct = round(float(fr_series.iloc[-1]) * 3 * 365 * 100, 2)
                funding_history = [
                    {
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "rate_8h_pct": round(float(v) * 100, 4),
                    }
                    for ts, v in fr_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Funding rate fetch fallito in /macro: %s", _e)

        # ── Aggregated OI ──────────────────────────────────────────────────────
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
                            (float(oi_series.iloc[-1]) - oi_7d_ago) / oi_7d_ago * 100, 2
                        )
                oi_history = [
                    {
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "oi_usd": round(float(v), 0),
                    }
                    for ts, v in oi_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("OI fetch fallito in /macro: %s", _e)

        # ── Long/Short ratio ───────────────────────────────────────────────────
        long_short_ratio_latest: float | None = None
        ls_history: list[dict] = []
        try:
            ls_series = cg.fetch_long_short_ratio(days=90)
            if not ls_series.empty:
                long_short_ratio_latest = round(float(ls_series.iloc[-1]), 4)
                ls_history = [
                    {
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "ratio": round(float(v), 4),
                    }
                    for ts, v in ls_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Long/short ratio fetch fallito in /macro: %s", _e)

        # ── Liquidations ───────────────────────────────────────────────────────
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
                    {
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "long_usd": round(float(row["long_usd"]), 0),
                        "short_usd": round(float(row["short_usd"]), 0),
                        "total_usd": round(float(row["total_usd"]), 0),
                    }
                    for ts, row in liq_df.tail(90).iterrows()
                ]
        except Exception as _e:
            _log.warning("Liquidations fetch fallito in /macro: %s", _e)

        # ── Taker volume ───────────────────────────────────────────────────────
        taker_buy_ratio_latest: float | None = None
        taker_history: list[dict] = []
        try:
            tk_series = cg.fetch_taker_volume(days=90)
            if not tk_series.empty:
                taker_buy_ratio_latest = round(float(tk_series.iloc[-1]), 4)
                taker_history = [
                    {
                        "date": str(ts.date()) if hasattr(ts, "date") else str(ts),
                        "buy_ratio": round(float(v), 4),
                    }
                    for ts, v in tk_series.tail(90).items()
                ]
        except Exception as _e:
            _log.warning("Taker volume fetch fallito in /macro: %s", _e)

        macro_data = {
            # Valori puntuali (latest)
            "funding_rate_8h_pct": funding_rate_8h_pct,
            "funding_rate_annualized_pct": funding_rate_ann_pct,
            "futures_oi_usd": oi_latest_usd,
            "oi_change_7d_pct": oi_change_7d_pct,
            "long_short_ratio_latest": long_short_ratio_latest,
            "liquidations_long_24h_usd": liquidations_long_24h_usd,
            "liquidations_short_24h_usd": liquidations_short_24h_usd,
            "liquidations_total_24h_usd": liquidations_total_24h_usd,
            "taker_buy_ratio_latest": taker_buy_ratio_latest,
            # Serie storiche per grafici (90gg)
            "history": {
                "funding_rate": funding_history,
                "oi": oi_history,
                "long_short": ls_history,
                "liquidations": liq_history,
                "taker": taker_history,
            },
        }
        # Caching separato del dict raw: usato da /signals senza accedere a JSONResponse.body
        _cache_set("macro_data", macro_data)
        response = _ok(macro_data)
        _cache_set("macro", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Macro error: {exc}")
