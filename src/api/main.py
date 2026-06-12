"""FastAPI backend per btc-institutional-flow.

Endpoints:
  GET /api/health    - Health check
  GET /api/gex       - GEX snapshot: regime, gamma flip, walls, profilo per strike
  GET /api/flows     - ETF flows IBIT: serie storica, correlazione, Granger
  GET /api/barriers  - Barriere attive da SEC EDGAR
  GET /api/signals   - Segnale composito LONG/CAUTION/RISK_OFF + Sharpe backtest
  GET /api/macro     - Indicatori macro: funding rate, OI, long/short ratio, liquidazioni
  GET /api/ifi       - Institutional Flow Index: serie storica giornaliera 0-100 (D/W chart)
  POST /api/telegram/webhook - Webhook Telegram: gestisce /recap da gruppo
  GET /docs          - Swagger UI (FastAPI built-in)
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import json
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.config import get_settings, setup_logging

_log = setup_logging("api.main")

# ─── Alert scheduler (lazy init, graceful degrade) ─────────────────────────────

_alert_scheduler = None  # type: ignore[var-annotated]
_ifi_scheduler = None  # type: ignore[var-annotated]

# ─── In-memory TTL cache ───────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}  # key → (timestamp, payload)

_TTL = {
    "gex":            300,   # 5 min  — opzioni Deribit, ~90s fetch
    "_gex_data":      300,   # 5 min  — raw GexSnapshot objects (condivisi tra /gex e /signals)
    "gex_enrichment": 3600,  # 1 ora  — CoinGlass coverage score + multi-exchange PCR
    "flows":          900,   # 15 min — Farside scrape
    "barriers":       3600,  # 1 ora  — dati SEC EDGAR statici
    "signals":        300,   # 5 min  — dipende da gex + flows
    "macro":          3600,  # 1 ora  — dati CoinGlass giornalieri
    "ifi":            900,   # 15 min — serie giornaliera, cambia lentamente
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


# ─── Alert scheduler startup/shutdown ──────────────────────────────────────────


async def _catch_up_daily_recap(monitor: Any, recap_cfg: dict) -> None:
    """Invia il daily_recap se il processo è ripartito dopo l'orario pianificato e il recap non è stato ancora inviato oggi."""
    from src.alerts.alert_db import AlertDB
    from src.alerts.gex_alert_monitor import ALERT_DAILY_RECAP

    now_utc = datetime.now(timezone.utc)
    scheduled_hour = int(recap_cfg.get("hour_utc", 7))
    scheduled_minute = int(recap_cfg.get("minute_utc", 0))

    past_scheduled = now_utc.hour > scheduled_hour or (
        now_utc.hour == scheduled_hour and now_utc.minute >= scheduled_minute
    )
    if not past_scheduled:
        return

    alert_db = AlertDB()
    if alert_db.sent_today(ALERT_DAILY_RECAP):
        _log.info("[alerts] catch-up skip: recap già inviato oggi (UTC)")
        return

    _log.info("[alerts] catch-up: daily_recap non inviato oggi — lancio ora")
    await monitor.send_daily_recap()


@app.on_event("startup")
async def _start_alert_scheduler() -> None:
    """Avvia APScheduler con 2 job: daily recap + ETF flow check.

    Graceful degrade: se TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID mancano, l'app
    parte lo stesso e i job non vengono registrati.
    """
    global _alert_scheduler

    cfg = get_settings().get("alerts", {})
    if not cfg.get("telegram_enabled", True):
        _log.info("[alerts] disabilitati via settings.alerts.telegram_enabled=false")
        return

    missing = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.getenv(v)]
    if missing:
        _log.warning("[alerts] env var mancanti %s — scheduler non avviato", missing)
        return

    try:
        from apscheduler.events import (
            EVENT_JOB_ERROR,
            EVENT_JOB_EXECUTED,
            EVENT_JOB_MISSED,
        )
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        from src.alerts.gex_alert_monitor import GexAlertMonitor

        monitor = GexAlertMonitor()
        scheduler = AsyncIOScheduler(timezone="UTC")

        recap_cfg = cfg.get("daily_recap", {})
        scheduler.add_job(
            monitor.send_daily_recap,
            CronTrigger(
                hour=int(recap_cfg.get("hour_utc", 7)),
                minute=int(recap_cfg.get("minute_utc", 0)),
                timezone="UTC",
            ),
            id="daily_recap",
            replace_existing=True,
            misfire_grace_time=int(timedelta(hours=2).total_seconds()),
        )

        flow_cfg = cfg.get("etf_flow_check", {})
        scheduler.add_job(
            monitor.check_etf_flows,
            IntervalTrigger(hours=int(flow_cfg.get("interval_hours", 4))),
            id="etf_flow_check",
            replace_existing=True,
        )

        def _job_listener(event: Any) -> None:
            if event.code == EVENT_JOB_MISSED:
                _log.warning(
                    "[alerts] job %s MISSED scheduled run at %s",
                    event.job_id, event.scheduled_run_time,
                )
            elif getattr(event, "exception", None):
                _log.error("[alerts] job %s failed: %s", event.job_id, event.exception)
            else:
                _log.info("[alerts] job %s executed", event.job_id)

        scheduler.add_listener(
            _job_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
        )

        scheduler.start()
        _alert_scheduler = scheduler
        _log.info(
            "[alerts] scheduler started — daily_recap %02d:%02d UTC, etf_flow_check every %dh",
            int(recap_cfg.get("hour_utc", 7)),
            int(recap_cfg.get("minute_utc", 0)),
            int(flow_cfg.get("interval_hours", 4)),
        )

        asyncio.create_task(_catch_up_daily_recap(monitor, recap_cfg))
        asyncio.create_task(_register_telegram_webhook(monitor._telegram))
    except Exception:
        _log.exception("[alerts] scheduler startup failed")


async def _register_telegram_webhook(telegram: Any) -> None:
    """Registra il webhook Telegram e i comandi del bot se TELEGRAM_WEBHOOK_URL è impostato."""
    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").rstrip("/")
    if not webhook_url or telegram is None:
        return
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    await telegram.set_webhook(f"{webhook_url}/api/telegram/webhook", secret_token=secret)
    await telegram.set_commands([
        {"command": "recap", "description": "Invia il recap GEX + IFI aggiornato"},
    ])


async def _auto_ifi_update(*, backfill: bool = False, days: int = 1) -> None:
    """Aggiorna IFI DB in un thread executor per non bloccare l'event loop."""
    try:
        from src.analytics.ifi_updater import run as _ifi_run
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _ifi_run(backfill=backfill, days=days)
        )
        _log.info(
            "[ifi] aggiornamento completato (backfill=%s, days=%d, result=%d)",
            backfill, days, result,
        )
    except Exception:
        _log.exception("[ifi] aggiornamento fallito")


@app.on_event("startup")
async def _start_ifi_scheduler() -> None:
    """Avvia scheduler IFI: backfill automatico se DB vuoto + cron giornaliero 22:00 UTC."""
    global _ifi_scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from src.analytics.ifi_db import IFIDb

        db_count = IFIDb().count()
        if db_count < 30:
            _log.info("[ifi] DB ha %d righe — lancio backfill in background", db_count)
            asyncio.create_task(_auto_ifi_update(backfill=True))
        else:
            _log.info("[ifi] DB ha %d righe — aggiornamento giornaliero in background", db_count)
            asyncio.create_task(_auto_ifi_update(days=1))

        scheduler = AsyncIOScheduler(timezone="UTC")
        scheduler.add_job(
            _auto_ifi_update,
            CronTrigger(hour=22, minute=0, timezone="UTC"),
            id="ifi_daily_update",
            kwargs={"days": 1},
            replace_existing=True,
            misfire_grace_time=int(timedelta(hours=3).total_seconds()),
        )
        scheduler.start()
        _ifi_scheduler = scheduler
        _log.info("[ifi] scheduler started — daily update at 22:00 UTC")
    except Exception:
        _log.exception("[ifi] scheduler startup failed")


@app.on_event("shutdown")
async def _stop_alert_scheduler() -> None:
    global _alert_scheduler, _ifi_scheduler
    if _alert_scheduler is not None:
        _alert_scheduler.shutdown(wait=False)
        _alert_scheduler = None
        _log.info("[alerts] scheduler stopped")
    if _ifi_scheduler is not None:
        _ifi_scheduler.shutdown(wait=False)
        _ifi_scheduler = None
        _log.info("[ifi] scheduler stopped")


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


# ─── /api/telegram/webhook ────────────────────────────────────────────────────


@app.post("/api/telegram/webhook", include_in_schema=False)
async def telegram_webhook(request: Request) -> JSONResponse:
    """Riceve aggiornamenti da Telegram e gestisce i comandi del bot.

    Comandi supportati:
      /recap — invia il recap GEX + IFI aggiornato nella chat corrente
    """
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    if secret:
        received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if received != secret:
            return JSONResponse({"ok": False}, status_code=403)

    try:
        update = await request.json()
    except Exception:
        return JSONResponse({"ok": True})

    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return JSONResponse({"ok": True})

    text: str = msg.get("text", "")
    chat_id = str(msg.get("chat", {}).get("id", ""))

    if not text.startswith("/recap") or not chat_id:
        return JSONResponse({"ok": True})

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return JSONResponse({"ok": True})

    try:
        from src.alerts.gex_alert_monitor import GexAlertMonitor
        from src.alerts.telegram_client import TelegramClient

        monitor = GexAlertMonitor()
        message = await monitor.build_recap_message()
        if message:
            client = TelegramClient(bot_token=token, chat_id=chat_id)
            await client.send_to(chat_id, message)
            _log.info("[webhook] /recap inviato a chat %s", chat_id)
        else:
            await TelegramClient(bot_token=token, chat_id=chat_id).send_to(
                chat_id, "<i>Nessun dato GEX disponibile al momento.</i>"
            )
    except Exception:
        _log.exception("[webhook] errore gestione /recap")

    return JSONResponse({"ok": True})


# ─── CoinGlass GEX enrichment ─────────────────────────────────────────────────

def _enrich_gex_with_coinglass(our_call_oi: float, our_put_oi: float) -> dict:
    """Arricchisce il GEX con coverage score e metriche multi-exchange da CoinGlass.

    Cachato 1 ora perché i dati CoinGlass cambiano lentamente (aggregati giornalieri).

    Args:
        our_call_oi: call OI totale fetchato da Deribit (contratti BTC).
        our_put_oi: put OI totale fetchato da Deribit (contratti BTC).

    Returns:
        dict con:
          data_quality: {coverage_pct, quality_label, deribit_oi_usd, fetched_oi_contracts}
          market_context: {market_pcr, market_max_pain, exchanges_included}
    """
    cached = _cache_get("gex_enrichment")
    if cached is not None:
        # Aggiorna coverage_pct con OI attuale (Deribit fluttua ogni fetch)
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

    result: dict = {
        "data_quality": {
            "coverage_pct":        None,
            "quality_label":       "unknown",
            "deribit_oi_usd":      None,
            "fetched_oi_contracts": round(our_call_oi + our_put_oi, 1),
        },
        "market_context": {
            "market_pcr":          None,
            "market_max_pain":     None,
            "exchanges_included":  [],
        },
        "_deribit_oi_contracts": 0,  # campo interno per aggiornamento coverage
    }

    try:
        cg = CoinGlassClient()

        # ── 1. Coverage score ────────────────────────────────────────────────
        options_info = cg.fetch_options_info("BTC")
        deribit_info = next(
            (x for x in options_info
             if isinstance(x, dict) and "deribit" in str(x.get("exchange_name", "")).lower()),
            None,
        )
        if deribit_info:
            cg_oi_contracts = float(deribit_info.get("open_interest") or 0)
            cg_oi_usd       = float(deribit_info.get("open_interest_usd") or 0)
            result["_deribit_oi_contracts"] = cg_oi_contracts
            result["data_quality"]["deribit_oi_usd"] = round(cg_oi_usd, 0) if cg_oi_usd else None

            if cg_oi_contracts > 0:
                our_total = our_call_oi + our_put_oi
                cov   = round(min(our_total / cg_oi_contracts * 100, 100.0), 1)
                label = "good" if cov >= 80 else "degraded" if cov >= 50 else "poor"
                result["data_quality"]["coverage_pct"]  = cov
                result["data_quality"]["quality_label"] = label
                result["data_quality"]["fetched_oi_contracts"] = round(our_total, 1)

        # ── 2. Multi-exchange PCR e max pain ─────────────────────────────────
        _EXCHANGES = ["Deribit", "Bybit", "Binance", "OKX"]
        total_call_notional = 0.0
        total_put_notional  = 0.0
        weighted_pain_sum   = 0.0
        total_pain_weight   = 0.0
        exchanges_with_data: list[str] = []

        for exch in _EXCHANGES:
            mp_data = cg.fetch_options_max_pain("BTC", exch)
            if not mp_data:
                continue
            exchanges_with_data.append(exch)
            for expiry in mp_data:
                if not isinstance(expiry, dict):
                    continue
                call_n   = float(expiry.get("call_open_interest_notional") or 0)
                put_n    = float(expiry.get("put_open_interest_notional") or 0)
                max_pain = float(expiry.get("max_pain_price") or 0)
                weight   = call_n + put_n
                total_call_notional += call_n
                total_put_notional  += put_n
                if max_pain > 0 and weight > 0:
                    weighted_pain_sum  += max_pain * weight
                    total_pain_weight  += weight

        if total_call_notional > 0:
            result["market_context"]["market_pcr"] = round(
                total_put_notional / total_call_notional, 3
            )
        if total_pain_weight > 0:
            result["market_context"]["market_max_pain"] = round(
                weighted_pain_sum / total_pain_weight, 0
            )
        result["market_context"]["exchanges_included"] = exchanges_with_data

    except Exception as _e:
        _log.warning("CoinGlass GEX enrichment failed: %s", _e)

    _cache_set("gex_enrichment", result)
    return result


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

        # Profilo per strike (intorno a ±40% dallo spot).
        # Guardia spot>0: evita ZeroDivisionError se lo spot non è disponibile.
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

        # Enrichment CoinGlass: coverage score + multi-exchange PCR/max pain.
        # TTL separato (1h) — indipendente dal fetch Deribit (5min).
        enrichment = _enrich_gex_with_coinglass(snapshot.total_call_oi, snapshot.total_put_oi)

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
                # Nuovi campi: qualità del dato e contesto di mercato multi-exchange
                "data_quality": enrichment.get("data_quality", {}),
                "market_context": enrichment.get("market_context", {}),
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

        # ── Qualità dati: fonte dominante dei flussi ──────────────────────────
        # Il fallback yfinance ("yfinance_estimate*") è una stima ad-hoc da tracking
        # error: segnaliamo la qualità così i consumatori possono avvisare l'utente.
        source_counts: dict[str, int] = {}
        for f in raw_flows:
            source_counts[f.source] = source_counts.get(f.source, 0) + 1
        dominant_source = (
            max(source_counts, key=source_counts.get) if source_counts else "unknown"
        )
        is_estimate = dominant_source.startswith("yfinance")
        flow_quality = {
            "dominant_source": dominant_source,
            "source_breakdown": source_counts,
            "quality_label": "low_estimate" if is_estimate else "ok",
            "is_estimate": is_estimate,
        }

        response = _ok(
            {
                "summary": stats,
                "history": history,
                "rolling_correlations_latest": corr_latest,
                "granger": granger_out,
                "data_quality": flow_quality,
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


# ─── /api/notes ───────────────────────────────────────────────────────────────


def _note_to_dict(note) -> dict:
    """Serializza una StructuredNote (con le sue barriere) in dict JSON-safe."""
    return {
        "filing_url":      note.filing_url,
        "issuer":          note.issuer,
        "issue_date":      str(note.issue_date) if note.issue_date else None,
        "maturity_date":   str(note.maturity_date) if note.maturity_date else None,
        "notional_usd":    note.notional_usd,
        "product_type":    note.product_type,
        "underlying":      note.underlying,
        "initial_level":   note.initial_level,
        "autocall_trigger_pct": note.autocall_trigger_pct,
        "knockin_barrier_pct":  note.knockin_barrier_pct,
        "buffer_pct":      note.buffer_pct,
        "coupon_rate":     note.coupon_rate,
        "is_preliminary":  bool(note.is_preliminary),
        "observation_dates": note.observation_dates,
        "barriers": [
            {
                "barrier_type":     b.barrier_type,
                "level_pct":        b.level_pct,
                "level_price_ibit": b.level_price_ibit,
                "level_price_btc":  b.level_price_btc,
                "observation_date": str(b.observation_date) if b.observation_date else None,
                "status":           b.status,
            }
            for b in note.barriers
        ],
    }


@app.get("/api/notes", tags=["barriers"])
def get_notes(underlying: str = "IBIT", limit: int = 200) -> dict:
    """Elenco note strutturate (lettura aggregata, per drill-down EDGAR).

    Restituisce metadati sintetici (no raw_text) per ogni nota del sottostante.
    """
    try:
        from src.edgar.structured_notes_db import StructuredNotesDB

        db = StructuredNotesDB()
        notes = [n for n in db.get_all_notes()
                 if (n.underlying or "IBIT") == underlying][:limit]
        return _ok({"count": len(notes), "notes": [_note_to_dict(n) for n in notes]})
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Notes error: {exc}")


@app.get("/api/notes/by-url", tags=["barriers"])
def get_note_by_url(url: str) -> dict:
    """Dettaglio di una singola nota strutturata dal suo filing_url (drill-down)."""
    try:
        from src.edgar.structured_notes_db import StructuredNotesDB

        db = StructuredNotesDB()
        note = db.get_note_by_url(url)
        if note is None:
            raise _error("Nota non trovata", code=404)
        return _ok({"note": _note_to_dict(note)})
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Note error: {exc}")


# ─── /api/pillars/series ──────────────────────────────────────────────────────


@app.get("/api/pillars/series", tags=["signals"])
def get_pillars_series(pillar: str = "composite", days: int = 180) -> dict:
    """Serie storica di un pilastro (o del composito) come indice 0-100 chartabile.

    Sostituto generalizzato di /api/ifi: espone qualunque pilastro come serie
    temporale (gex | barrier | etf_flows | macro | composite).
    """
    valid = {"composite", "gex", "barrier", "etf_flows", "macro"}
    if pillar not in valid:
        raise _error(f"pillar non valido: {pillar} (validi: {sorted(valid)})", code=400)

    cache_key = f"pillars_series_{pillar}_{days}"
    cached = _cache_get(cache_key)
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

        # Arricchisci con la serie GEX storica (DB locale) se disponibile
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
            raise _error(f"colonna {col} non disponibile", code=500)

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
        response = _ok({
            "pillar": pillar,
            "series": rows,
            "current": rows[-1] if rows else None,
            "stats": {"days_available": len(rows)},
        })
        _cache_set(cache_key, response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"Pillars series error: {exc}")


# ─── /api/signals ─────────────────────────────────────────────────────────────


@app.get("/api/signals", tags=["signals"])
def get_signals() -> dict:
    """Segnale composito LONG/CAUTION/RISK_OFF + backtest Sharpe.

    Architettura a 4 PILASTRI (ognuno con sotto-score 0-100):
      - gex:       regime dealer gamma + contesto gamma-flip (Deribit)
      - barrier:   livelli note strutturate IBIT, direzionale + notional-weighted (EDGAR)
      - etf_flows: domanda spot istituzionale (Farside/CoinGlass)
      - macro:     funding/OI/long-short/put-call/liquidazioni (CoinGlass)
    Score finale = blend pesato dei pilastri. ≥65 = LONG, <40 = RISK_OFF.

    Campo `pillars`: dettaglio dei 4 sotto-score. I campi `components`/`weights`
    espongono i 7 fattori storici per retro-compatibilità. Cachato 5 minuti.
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
        from src.analytics.pillars import CompositeSignal, CompositeInputs
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

        # ── Segnale composito a 4 PILASTRI (GEX, Barrier, ETF Flows, Macro) ────
        # Le barriere sono ora un pilastro pesato di prima classe (non più solo veto).
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

        # ── Salva segnale nel DB (silent failure, non blocca la risposta) ──────
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

        # Pilastri serializzati per il frontend (sotto-score + componenti leggibili)
        pillars_out = [
            {
                "name": p.name,
                "score": p.score,
                "weight": round(p.weight, 4),
                "components": p.components,
                "reason": p.reason,
            }
            for p in signal_result.pillars
        ]

        response = _ok(
            {
                "signal": signal_result.signal,
                "signal_reason": signal_result.reason,
                "score": signal_result.score,
                # components/weights: 7 fattori storici (retro-compat) derivati dai pilastri
                "components": signal_result.legacy_components,
                "weights": signal_result.weights_used,
                # Nuova vista a 4 pilastri (additiva)
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


# ─── /api/ifi ─────────────────────────────────────────────────────────────────


@app.get("/api/ifi", tags=["signals", "deprecated"])
def get_ifi() -> dict:
    """[DEPRECATO] Institutional Flow Index (IFI): serie storica giornaliera 0-100.

    ⚠️ DEPRECATO: l'IFI è stato assorbito nell'architettura a 4 pilastri. Le sue
    componenti flow vivono ora nel pilastro ETF Flows; funding/OI/L-S nel pilastro
    Macro. Usare `/api/pillars/series?pillar=etf_flows` (o `composite`) come sostituto
    generalizzato. Questo endpoint resta attivo per retro-compatibilità e continua
    a servire lo storico già accumulato in ifi_history.

    Indicatore chartabile su timeframe D/W basato su ETF flows (da gen 2024),
    prezzo BTC e dati CoinGlass (funding/OI/L/S per ultimi 333-500gg).

    Fast path: legge da DB ifi_history (popolato dal cron_ifi.py).
    Slow path: calcola on-the-fly se DB vuoto (solo flows + prezzo, no CoinGlass).
    Risposta cachata 15 minuti.
    """
    cached = _cache_get("ifi")
    if cached is not None:
        return cached

    try:
        from src.analytics.ifi import IFIModel, regime_label
        from src.analytics.ifi_db import IFIDb
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation

        db = IFIDb()

        # ── Fast path: leggi dal DB ────────────────────────────────────────────
        series_df = db.get_series(days=520)
        current = db.get_latest()

        # ── Slow path: se DB vuoto, calcola on-the-fly ────────────────────────
        if series_df.empty:
            _log.info("IFI DB vuoto — calcolo on-the-fly (solo flows+prezzo)")
            scraper = FarsideScraper()
            raw     = scraper.fetch()
            agg     = scraper.aggregate(raw)
            fetcher = PriceFetcher()
            prices  = fetcher.get_all_prices()
            corr    = FlowCorrelation()
            merged  = corr.merge(agg, prices)

            if "total_flow" in merged.columns:
                merged = merged.rename(columns={"total_flow": "total_flow_usd"})

            model  = IFIModel()
            scores = model.compute_series(merged)
            scores = scores.iloc[90:]  # skip warm-up

            btc_prices = merged.get("btc_close")
            flows      = merged.get("total_flow_usd")

            series_rows = []
            for ts, score in scores.items():
                date  = str(ts.date()) if hasattr(ts, "date") else str(ts)
                btc   = float(btc_prices[ts]) if btc_prices is not None and ts in btc_prices.index else None
                flow  = float(flows[ts]) if flows is not None and ts in flows.index else None
                series_rows.append({
                    "date":              date,
                    "score":             round(float(score), 2),
                    "regime":            regime_label(float(score)),
                    "btc_price":         btc,
                    "total_flow_usd_m":  round(flow / 1e6, 2) if flow is not None and flow == flow else None,
                })

            current_score  = float(scores.iloc[-1]) if not scores.empty else 50.0
            current_regime = regime_label(current_score)
            current_date   = str(scores.index[-1].date()) if not scores.empty else ""

            response = _ok({
                "current": {"score": current_score, "regime": current_regime, "date": current_date},
                "series":  series_rows,
                "stats":   {
                    "days_available": len(series_rows),
                    "data_start":     series_rows[0]["date"] if series_rows else None,
                    "source":         "computed_on_the_fly",
                },
            })
            _cache_set("ifi", response)
            return response

        # ── Serializza serie dal DB ────────────────────────────────────────────
        series_rows = []
        for ts, row in series_df.iterrows():
            date  = str(ts.date()) if hasattr(ts, "date") else str(ts)
            flow  = row.get("total_flow_usd")
            series_rows.append({
                "date":             date,
                "score":            round(float(row["score"]), 2),
                "regime":           row["regime"],
                "btc_price":        row.get("btc_price"),
                "total_flow_usd_m": round(float(flow) / 1e6, 2) if flow is not None and flow == flow else None,
            })

        c_score  = float(current["score"]) if current else 50.0
        c_regime = current["regime"] if current else "Neutral"
        c_date   = current["date"]  if current else ""

        response = _ok({
            "current": {"score": c_score, "regime": c_regime, "date": c_date},
            "series":  series_rows,
            "stats":   {
                "days_available": len(series_rows),
                "data_start":     series_rows[0]["date"] if series_rows else None,
                "source":         "db",
            },
        })
        _cache_set("ifi", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise _error(f"IFI error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# FORECAST SPINE — predizioni, verifica, calibrazione
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/predictions", tags=["forecast"])
def get_predictions(limit: int = 50, days: int = 180) -> JSONResponse:
    """Previsioni recenti con i relativi esiti (join predictions+outcomes)."""
    try:
        from src.forecast.prediction_db import PredictionDB
        rows = PredictionDB().get_with_outcomes(days=days)
        rows = rows[-limit:]
        return _ok({"count": len(rows), "predictions": rows})
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"predictions error: {exc}")


@app.post("/api/predictions/{prediction_id}/review", tags=["forecast"])
async def review_prediction(prediction_id: int, request: Request) -> JSONResponse:
    """Aggiorna l'overlay umano / contro-analisi di una previsione (dal daily-review).

    Body JSON: {"counter_analysis": str?, "human_overlay": str?, "confidence": float?}
    """
    try:
        from src.forecast.prediction_db import PredictionDB
        body = await request.json()
        PredictionDB().update_human_fields(
            prediction_id,
            counter_analysis=body.get("counter_analysis"),
            human_overlay=body.get("human_overlay"),
            confidence=body.get("confidence"),
        )
        return _ok({"updated": prediction_id})
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"review error: {exc}")


@app.post("/api/predictions/verify", tags=["forecast"])
def verify_predictions() -> JSONResponse:
    """Verifica le previsioni mature usando i prezzi reali e salva gli esiti."""
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
        return _ok({"verified": len(outcomes), "hit": hits, "miss": len(outcomes) - hits})
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"verify error: {exc}")


@app.get("/api/calibration", tags=["forecast"])
def get_calibration(days: int = 180) -> JSONResponse:
    """Riepilogo performance: hit-rate per source/target_type + Brier medio (target prob).

    Fase 1: solo report descrittivo. La proposta di nuovi pesi arriva con la fase 2
    (src/forecast/calibration.py).
    """
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
                    a["brier_sum"] += float(r["brier"]); a["brier_n"] += 1

        summary = {}
        for key, a in agg.items():
            summary[key] = {
                "predictions": a["n"],
                "scored": a["scored"],
                "open": a["n"] - a["scored"],
                "hit_rate": round(a["hits"] / a["scored"], 3) if a["scored"] else None,
                "mean_brier": round(a["brier_sum"] / a["brier_n"], 4) if a["brier_n"] else None,
            }

        # Pesi attivi + eventuali proposte in attesa di approvazione (dealer_flow)
        from src.forecast.sources.dealer_flow import SOURCE as _DF
        active = db.get_active_weights(_DF)
        weights = {
            "active": {"version": active[0], "weights": active[1]} if active else None,
            "proposed": db.get_proposed_weights(_DF),
        }
        return _ok({"window_days": days, "by_source_target": summary, "dealer_flow_weights": weights})
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"calibration error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# FORECAST SCHEDULER (APScheduler in-API) + status/governance
# ═══════════════════════════════════════════════════════════════════════════════

_forecast_scheduler = None  # type: ignore[var-annotated]


async def _job_predict() -> None:
    from src.forecast.jobs import run_daily_predict
    fc = get_settings().get("forecast", {})
    res = await asyncio.to_thread(run_daily_predict, horizon=int(fc.get("horizon_days", 5)))
    _log.info("[forecast] predict: %s", res)


async def _job_verify() -> None:
    from src.forecast.jobs import run_daily_verify
    res = await asyncio.to_thread(run_daily_verify)
    _log.info("[forecast] verify: %s", res)


async def _job_calibrate() -> None:
    from src.forecast.jobs import run_weekly_calibrate
    rep = await asyncio.to_thread(run_weekly_calibrate)
    _log.info("[forecast] calibrate: gate=%s scored=%s", rep.gate_ok, rep.metrics["total_scored"])


@app.on_event("startup")
async def _start_forecast_scheduler() -> None:
    """Avvia lo scheduler forecast (predict/verify/calibrate). Indipendente da Telegram.

    misfire_grace_time + coalesce: tollera run saltati (macchina locale spenta) eseguendo
    una sola volta al ritorno online. Disabilitabile via settings.forecast.enabled=false.
    """
    global _forecast_scheduler
    fc = get_settings().get("forecast", {})
    if not fc.get("enabled", True):
        _log.info("[forecast] scheduler disabilitato (settings.forecast.enabled=false)")
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        s = fc.get("schedule", {})
        grace = int(timedelta(hours=6).total_seconds())
        sch = AsyncIOScheduler(timezone="UTC")
        sch.add_job(
            _job_predict,
            CronTrigger(hour=int(s.get("predict_hour_utc", 7)),
                        minute=int(s.get("predict_minute_utc", 30)), timezone="UTC"),
            id="forecast_predict", replace_existing=True,
            misfire_grace_time=grace, coalesce=True,
        )
        sch.add_job(
            _job_verify,
            CronTrigger(hour=int(s.get("verify_hour_utc", 8)),
                        minute=int(s.get("verify_minute_utc", 0)), timezone="UTC"),
            id="forecast_verify", replace_existing=True,
            misfire_grace_time=grace, coalesce=True,
        )
        sch.add_job(
            _job_calibrate,
            CronTrigger(day_of_week=str(s.get("calibrate_dow", "mon")),
                        hour=int(s.get("calibrate_hour_utc", 8)),
                        minute=int(s.get("calibrate_minute_utc", 30)), timezone="UTC"),
            id="forecast_calibrate", replace_existing=True,
            misfire_grace_time=grace, coalesce=True,
        )
        sch.start()
        _forecast_scheduler = sch
        _log.info("[forecast] scheduler avviato (predict/verify/calibrate)")
    except Exception:
        _log.exception("[forecast] scheduler startup failed")


@app.get("/api/forecast/status", tags=["forecast"])
def forecast_status() -> JSONResponse:
    """Stato operativo del forecast: ultima predizione, freschezza, governance, conteggi."""
    try:
        from src.forecast.calibration import load_weights_config
        from src.forecast.prediction_db import PredictionDB

        db = PredictionDB()
        recent = db.get_recent(limit=1)
        last = recent[0].created_at if recent else None

        fresh = None
        if last:
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(last).replace(tzinfo=timezone.utc)).total_seconds() / 3600
            max_h = float(get_settings().get("forecast", {}).get("freshness_max_hours", 30))
            fresh = age_h <= max_h

        gov = load_weights_config().get("governance", {})
        return _ok({
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
        raise _error(f"forecast status error: {exc}")


@app.post("/api/weights/{version_id}/activate", tags=["forecast"])
async def activate_weights(version_id: int, request: Request) -> JSONResponse:
    """Attivazione human-gated di una versione pesi (chiamata dal workflow Tuning).

    Body JSON opzionale: {"source": "dealer_flow"}. È l'unico modo per cambiare i pesi attivi.
    """
    try:
        from src.forecast.prediction_db import PredictionDB
        body = await request.json() if await request.body() else {}
        source = body.get("source", "dealer_flow")
        db = PredictionDB()
        db.activate_weight_version(version_id, source)
        active = db.get_active_weights(source)
        return _ok({"activated": version_id, "source": source,
                    "active_weights": active[1] if active else None})
    except Exception as exc:
        traceback.print_exc()
        raise _error(f"activate weights error: {exc}")
