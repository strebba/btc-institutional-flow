"""FastAPI backend per btc-institutional-flow.

Endpoint:
  GET /api/health        - Health check
  GET /api/health/edgar  - EDGAR DB health
  GET /api/gex           - GEX snapshot: regime, gamma flip, walls, profilo per strike
  GET /api/flows         - ETF flows IBIT: serie storica, correlazione, Granger
  GET /api/barriers      - Barriere attive da SEC EDGAR
  GET /api/notes[/by-url] - Drill-down note strutturate
  GET /api/signals       - Segnale composito LONG/CAUTION/RISK_OFF + backtest
  GET /api/pillars/series - Serie storica pilastro
  GET /api/macro         - Indicatori macro: funding rate, OI, long/short, liquidazioni
  GET /api/ifi           - [DEPRECATO] IFI series
  POST /api/telegram/webhook - Webhook Telegram
  /api/predictions/*     - Forecast spine
  /api/calibration       - Calibration report
  /api/forecast/status   - Forecast operational status
  /api/weights/{id}/activate - Human-gated weight activation

Schedulers APScheduler in-process: alert Telegram, IFI daily, forecast predict/verify/calibrate.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.api.routers import health, gex, flows, barriers, signals, forecast
from src.api.scheduler import (
    _alert_monitor,
    start_alert_scheduler,
    start_ifi_scheduler,
    start_forecast_scheduler,
    stop_all_schedulers,
)

_log = logging.getLogger("api.main")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTC Institutional Flow API",
    description="GEX, ETF flows, SEC barriers, composite signals for BTC institutional analysis.",
    version="1.0.0",
)

# CORS
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

# API key auth middleware (opzionale)
# Usa os.getenv ogni volta per consentire ai test di monkeypatch l'env var.


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    _key = os.getenv("API_KEY")
    if _key and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        if request.headers.get("X-API-Key") != _key:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ─── Include routers ─────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(gex.router)
app.include_router(flows.router)
app.include_router(barriers.router)
app.include_router(signals.router)
app.include_router(forecast.router)

# ─── Root redirect ───────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


# ─── Telegram webhook ────────────────────────────────────────────────────────


@app.post("/api/telegram/webhook", include_in_schema=False)
async def telegram_webhook(request: Request) -> JSONResponse:
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

    if not text or not chat_id:
        return JSONResponse({"ok": True})

    monitor = _alert_monitor
    if monitor is None:
        from src.alerts.gex_alert_monitor import GexAlertMonitor
        monitor = GexAlertMonitor()

    def _help_message() -> str:
        return (
            "<b>BTC Institutional Flow — Comandi disponibili</b>\n\n"
            "/recap — Invia il recap GEX + IFI + ETF flows aggiornato\n"
            "/signal — Segnale direzionale a 4 pilastri (bias long/short)\n"
            "/status — Stato del bot (ultimo recap, freschezza dati)\n"
            "/help — Mostra questo messaggio"
        )

    async def _status_message(mon: Any) -> str:
        from src.alerts.alert_db import AlertDB

        alert_db = AlertDB()
        last = alert_db.get_last_sent("daily_recap")
        lines: list[str] = []
        lines.append("📡 <b>BTC Institutional Flow — Stato</b>")
        now = datetime.now(timezone.utc)
        lines.append(f"<i>{now.strftime('%Y-%m-%d · %H:%M UTC')}</i>")
        lines.append("")

        if last:
            ts_str = last[0].strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"Ultimo daily recap: {ts_str}")
        else:
            lines.append("Nessun daily recap ancora inviato")

        gex_db = mon._gex_db
        count = await asyncio.to_thread(gex_db.count)
        lines.append(f"Snapshot GEX nel DB: {count}")
        last_regime = await asyncio.to_thread(gex_db.get_last_regime_label)
        if last_regime:
            lines.append(f"Regime corrente: {last_regime.replace('_', ' ')}")

        upcoming_recap_cfg = mon._cfg.get("daily_recap", {})
        recap_hour = int(upcoming_recap_cfg.get("hour_utc", 7))
        recap_min = int(upcoming_recap_cfg.get("minute_utc", 0))
        lines.append(f"Prossimo recap: {recap_hour:02d}:{recap_min:02d} UTC")

        return "\n".join(lines)

    cmd = text.split()[0].split("@")[0]

    if cmd == "/help":
        if monitor._telegram is not None:
            await monitor._telegram.send_to(chat_id, _help_message())
        return JSONResponse({"ok": True})

    if cmd == "/status":
        if monitor._telegram is not None:
            await monitor._telegram.send_to(chat_id, await _status_message(monitor))
        return JSONResponse({"ok": True})

    if cmd == "/signal":
        try:
            message = await monitor.build_signal_message()
            if message:
                if monitor._telegram is not None:
                    await monitor._telegram.send_to(chat_id, message)
                    _log.info("[webhook] /signal inviato a chat %s", chat_id)
                else:
                    _log.warning("[webhook] /signal impossibile: telegram non configurato")
            else:
                if monitor._telegram is not None:
                    await monitor._telegram.send_to(chat_id, "<i>Nessun dato GEX disponibile per il segnale.</i>")
        except Exception:
            _log.exception("[webhook] errore gestione /signal")
        return JSONResponse({"ok": True})

    if cmd != "/recap":
        return JSONResponse({"ok": True})

    try:
        message = await monitor.build_recap_message()
        if message:
            if monitor._telegram is not None:
                await monitor._telegram.send_to(chat_id, message)
                _log.info("[webhook] /recap inviato a chat %s", chat_id)
            else:
                _log.warning("[webhook] /recap impossibile: telegram non configurato")
        else:
            if monitor._telegram is not None:
                await monitor._telegram.send_to(chat_id, "<i>Nessun dato GEX disponibile al momento.</i>")
    except Exception:
        _log.exception("[webhook] errore gestione /recap")

    return JSONResponse({"ok": True})


# ─── Health check scheduler ─────────────────────────────────────────────────


@app.get("/api/health/scheduler", tags=["meta"])
def health_scheduler() -> JSONResponse:
    from src.api.helpers import ok
    from src.api.scheduler import _alert_scheduler, _ifi_scheduler, _forecast_scheduler
    return ok({
        "alert": _alert_scheduler is not None and getattr(_alert_scheduler, "running", False),
        "ifi": _ifi_scheduler is not None and getattr(_ifi_scheduler, "running", False),
        "forecast": _forecast_scheduler is not None and getattr(_forecast_scheduler, "running", False),
    })


# ─── Lifespan (startup / shutdown) ──────────────────────────────────────────


@app.on_event("startup")
async def _startup_schedulers():
    asyncio.create_task(start_alert_scheduler())
    asyncio.create_task(start_ifi_scheduler())
    asyncio.create_task(start_forecast_scheduler())


@app.on_event("shutdown")
async def _shutdown_schedulers():
    stop_all_schedulers()
