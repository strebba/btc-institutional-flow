"""APScheduler in-process: alert Telegram, IFI update, forecast predict/verify/calibrate.

Espone variabili globali _alert_scheduler, _ifi_scheduler, _forecast_scheduler
per health checking dall'esterno (es. /api/forecast/status).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import get_settings, setup_logging

_log = setup_logging("api.scheduler")

_alert_scheduler = None  # type: ignore[var-annotated]
_alert_monitor = None   # type: ignore[var-annotated]
_ifi_scheduler = None  # type: ignore[var-annotated]
_forecast_scheduler = None  # type: ignore[var-annotated]


# ─── Alert scheduler ─────────────────────────────────────────────────────────


async def _catch_up_daily_recap(monitor: Any, recap_cfg: dict) -> None:
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
        _log.info("[alerts] catch-up skip: recap gia' inviato oggi (UTC)")
        return

    _log.info("[alerts] catch-up: daily_recap non inviato oggi — lancio ora")
    await monitor.send_daily_recap()


async def _register_telegram_webhook(telegram: Any) -> None:
    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").rstrip("/")
    if not webhook_url or telegram is None:
        return
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    await telegram.set_webhook(f"{webhook_url}/api/telegram/webhook", secret_token=secret)
    await telegram.set_commands([
        {"command": "recap", "description": "Invia il recap GEX + IFI aggiornato"},
        {"command": "status", "description": "Stato del bot e freschezza dati"},
        {"command": "help", "description": "Mostra tutti i comandi disponibili"},
    ])


async def start_alert_scheduler():
    global _alert_scheduler, _alert_monitor

    cfg = get_settings().get("alerts", {})
    if not cfg.get("telegram_enabled", True):
        _log.info("[alerts] disabilitati via settings.alerts.telegram_enabled=false")
        return

    missing = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.getenv(v)]
    if missing:
        _log.warning("[alerts] env var mancanti %s — scheduler non avviato", missing)
        return

    try:
        from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        from src.alerts.gex_alert_monitor import GexAlertMonitor

        _alert_monitor = GexAlertMonitor()
        monitor = _alert_monitor
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
                _log.warning("[alerts] job %s MISSED at %s", event.job_id, event.scheduled_run_time)
            elif getattr(event, "exception", None):
                _log.error("[alerts] job %s failed: %s", event.job_id, event.exception)
            else:
                _log.info("[alerts] job %s executed", event.job_id)

        scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
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
        asyncio.create_task(monitor.send_startup_message())
    except Exception:
        _log.exception("[alerts] scheduler startup failed")


# ─── IFI scheduler ───────────────────────────────────────────────────────────


async def _auto_ifi_update(*, backfill: bool = False, days: int = 1) -> None:
    try:
        from src.analytics.ifi_updater import run as _ifi_run
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _ifi_run(backfill=backfill, days=days))
        _log.info("[ifi] aggiornamento completato (backfill=%s, days=%d, result=%d)", backfill, days, result)
    except Exception:
        _log.exception("[ifi] aggiornamento fallito")


async def start_ifi_scheduler():
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


# ─── Forecast scheduler ──────────────────────────────────────────────────────


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


async def start_forecast_scheduler():
    global _forecast_scheduler
    fc = get_settings().get("forecast", {})
    if not fc.get("enabled", True):
        _log.info("[forecast] scheduler disabilitato (settings.forecast.enabled=false)")
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        s = fc.get("schedule", {})
        grace = int(timedelta(hours=24).total_seconds())
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


# ─── Shutdown ────────────────────────────────────────────────────────────────


def stop_all_schedulers():
    global _alert_scheduler, _ifi_scheduler, _forecast_scheduler
    for name, scheduler in [("alerts", _alert_scheduler), ("ifi", _ifi_scheduler), ("forecast", _forecast_scheduler)]:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            _log.info("[%s] scheduler stopped", name)
    _alert_scheduler = None
    _ifi_scheduler = None
    _forecast_scheduler = None
