"""APScheduler in-process: alert Telegram, barrier snapshot, forecast predict/verify/calibrate.

Espone variabili globali _alert_scheduler, _maintenance_scheduler, _forecast_scheduler
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
_maintenance_scheduler = None  # type: ignore[var-annotated]
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
    sent = await asyncio.to_thread(alert_db.sent_today, ALERT_DAILY_RECAP)
    if sent:
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
        {"command": "recap", "description": "Invia il recap livelli GEX + flussi ETF"},
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


# ─── Maintenance scheduler (snapshot barriere) ───────────────────────────────


async def _job_snapshot_barriers() -> None:
    """Salva l'istantanea giornaliera delle barriere attive (storico EDGAR)."""
    try:
        from src.edgar.structured_notes_db import StructuredNotesDB
        loop = asyncio.get_running_loop()
        n = await loop.run_in_executor(None, StructuredNotesDB().snapshot_active_barriers)
        _log.info("[barriers] snapshot giornaliero: %d barriere salvate", n)
    except Exception:
        _log.exception("[barriers] snapshot fallito")


async def start_maintenance_scheduler():
    global _maintenance_scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = AsyncIOScheduler(timezone="UTC")
        scheduler.add_job(
            _job_snapshot_barriers,
            CronTrigger(hour=22, minute=30, timezone="UTC"),
            id="barrier_snapshot_daily",
            replace_existing=True,
            misfire_grace_time=int(timedelta(hours=3).total_seconds()),
        )
        scheduler.start()
        _maintenance_scheduler = scheduler
        _log.info("[maintenance] scheduler started — barrier snapshot 22:30 UTC")
    except Exception:
        _log.exception("[maintenance] scheduler startup failed")


# ─── Forecast scheduler ──────────────────────────────────────────────────────

# Valori espliciti. Qualsiasi altra stringa (typo) lascia il default: acceso.
_CATCHUP_ON = {"1", "true", "yes", "on"}
_CATCHUP_OFF = {"0", "false", "no", "off"}


def forecast_catchup_enabled() -> bool:
    """Catch-up all'avvio. Default acceso; ``FORECAST_CATCHUP_ENABLED`` lo spegne."""
    raw = os.getenv("FORECAST_CATCHUP_ENABLED")
    if raw is None or not raw.strip():
        return True
    value = raw.strip().lower()
    if value in _CATCHUP_OFF:
        return False
    if value not in _CATCHUP_ON:
        _log.warning(
            "[forecast] FORECAST_CATCHUP_ENABLED=%r non riconosciuto — catch-up resta attivo",
            raw.strip(),
        )
    return True


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _slot_reached(now: datetime, hour: int, minute: int) -> bool:
    """True se l'orario UTC di oggi ha già raggiunto lo slot (come il cron giornaliero)."""
    return (now.hour, now.minute) >= (int(hour), int(minute))


def _forecast_slots(fc: dict | None) -> dict[str, int]:
    """Orari dello scheduler. Stessi default di ``start_forecast_scheduler``."""
    fc = fc if fc is not None else get_settings().get("forecast", {})
    schedule = fc.get("schedule", {}) or {}
    return {
        "predict_hour": int(schedule.get("predict_hour_utc", 7)),
        "predict_minute": int(schedule.get("predict_minute_utc", 30)),
        "verify_hour": int(schedule.get("verify_hour_utc", 8)),
        "verify_minute": int(schedule.get("verify_minute_utc", 0)),
        "horizon": int(fc.get("horizon_days", 5)),
    }


def run_forecast_startup_catchup(
    fc: dict | None = None,
    *,
    now: datetime | None = None,
    db: Any = None,
) -> dict[str, Any]:
    """Se il processo parte dopo lo slot del giorno, esegue predict e/o verify una volta.

    APScheduler in-memory, al boot, calcola solo il prossimo fire futuro: uno slot
    già passato oggi non è un misfire (``misfire_grace_time`` vale solo per un
    next_run_time già persistito). Senza questo, un avvio dopo le 07:30 UTC non
    crea la cohort del giorno.

    Idempotenza: prima di predict si controlla la cohort v1 (data UTC + source +
    orizzonte). Se due processi passano il controllo insieme, ``idx_pred_day_unique``
    fa ignorare il secondo insert. ``run_daily_predict`` scrive anche la cohort
    ombra ``dealer_flow_v2`` (source diversa: l'indice di v1 non si tocca). Se v1
    c'è già e v2 no, v2 si ricostruisce dagli input salvati, senza un secondo
    gather e senza riscrivere v1. Verify è quello esistente: marca solo le
    predizioni mature ancora open e non riscrive gli esiti già presenti.
    """
    from src.forecast.jobs import run_daily_predict, run_daily_verify, run_shadow_catchup
    from src.forecast.prediction_db import PredictionDB
    from src.forecast.sources.dealer_flow import SOURCE as DEALER_FLOW
    from src.forecast.sources.dealer_flow_v2 import SOURCE as DEALER_FLOW_V2

    now_utc = _as_utc(now or datetime.now(timezone.utc))
    date = now_utc.strftime("%Y-%m-%d")
    slots = _forecast_slots(fc)
    result: dict[str, Any] = {
        "enabled": forecast_catchup_enabled(),
        "date": date,
        "predict": "skipped_disabled",
        "predict_v2": "skipped_disabled",
        "verify": "skipped_disabled",
        "predict_result": None,
        "predict_v2_result": None,
        "verify_result": None,
    }
    if not result["enabled"]:
        _log.info(
            "[forecast] catch-up skip: FORECAST_CATCHUP_ENABLED disabilitato (data %s)",
            date,
        )
        return result

    db = db or PredictionDB()
    stamp = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    predict_at = f"{slots['predict_hour']:02d}:{slots['predict_minute']:02d}"
    verify_at = f"{slots['verify_hour']:02d}:{slots['verify_minute']:02d}"

    if not _slot_reached(now_utc, slots["predict_hour"], slots["predict_minute"]):
        result["predict"] = "skipped_before_slot"
        result["predict_v2"] = "skipped_before_slot"
        _log.info(
            "[forecast] catch-up predict skip: prima dello slot %s UTC (now %s)",
            predict_at, stamp,
        )
    elif not db.has_cohort(date, source=DEALER_FLOW, horizon_days=slots["horizon"]):
        result["predict"] = "ran"
        result["predict_v2"] = "ran_with_v1"
        _log.info(
            "[forecast] catch-up predict: slot %s UTC passato, nessuna cohort per %s"
            " — eseguo run_daily_predict",
            predict_at, date,
        )
        try:
            result["predict_result"] = run_daily_predict(
                horizon=slots["horizon"], db=db, asof=now_utc,
            )
        except Exception:
            result["predict"] = "error"
            result["predict_v2"] = "error"
            _log.exception("[forecast] catch-up predict fallito per %s", date)
        else:
            _log.info("[forecast] catch-up predict risultato: %s", result["predict_result"])
    else:
        # v1 c'è già: non si rilancia (idx_pred_day_unique) e non si rifà il gather.
        result["predict"] = "skipped_exists"
        _log.info(
            "[forecast] catch-up predict skip: cohort %s già presente per %s",
            DEALER_FLOW, date,
        )
        if db.has_cohort(date, source=DEALER_FLOW_V2, horizon_days=slots["horizon"]):
            result["predict_v2"] = "skipped_exists"
            _log.info(
                "[forecast] catch-up shadow v2 skip: cohort %s già presente per %s",
                DEALER_FLOW_V2, date,
            )
        else:
            result["predict_v2"] = "ran"
            _log.info(
                "[forecast] catch-up shadow v2: cohort %s presente senza %s per %s"
                " — copio gli input già salvati, senza un secondo gather",
                DEALER_FLOW, DEALER_FLOW_V2, date,
            )
            try:
                result["predict_v2_result"] = run_shadow_catchup(
                    db, day=date, horizon=slots["horizon"],
                )
            except Exception:
                result["predict_v2"] = "error"
                _log.exception("[forecast] catch-up shadow v2 fallito per %s", date)
            else:
                _log.info("[forecast] catch-up shadow v2 risultato: %s", result["predict_v2_result"])

    if not _slot_reached(now_utc, slots["verify_hour"], slots["verify_minute"]):
        result["verify"] = "skipped_before_slot"
        _log.info(
            "[forecast] catch-up verify skip: prima dello slot %s UTC (now %s)",
            verify_at, stamp,
        )
    else:
        result["verify"] = "ran"
        _log.info(
            "[forecast] catch-up verify: slot %s UTC passato — eseguo run_daily_verify"
            " (solo predizioni mature)",
            verify_at,
        )
        try:
            result["verify_result"] = run_daily_verify(db=db)
        except Exception:
            result["verify"] = "error"
            _log.exception("[forecast] catch-up verify fallito per %s", date)
        else:
            _log.info("[forecast] catch-up verify risultato: %s", result["verify_result"])

    return result


async def _catch_up_forecast_jobs(fc: dict) -> None:
    try:
        await asyncio.to_thread(run_forecast_startup_catchup, fc)
    except Exception:
        _log.exception("[forecast] catch-up fallito")


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
        _log.info(
            "[forecast] scheduler disabilitato "
            "(settings.forecast.enabled=false o FORECAST_SCHEDULER_ENABLED=false)"
        )
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
        _log.info(
            "[forecast] scheduler avviato — predict %02d:%02d verify %02d:%02d UTC, "
            "catch-up=%s (FORECAST_CATCHUP_ENABLED)",
            int(s.get("predict_hour_utc", 7)),
            int(s.get("predict_minute_utc", 30)),
            int(s.get("verify_hour_utc", 8)),
            int(s.get("verify_minute_utc", 0)),
            "on" if forecast_catchup_enabled() else "off",
        )
        # Il cron in-memory non recupera lo slot di oggi se il processo parte dopo.
        asyncio.create_task(_catch_up_forecast_jobs(fc))
    except Exception:
        _log.exception("[forecast] scheduler startup failed")


# ─── Shutdown ────────────────────────────────────────────────────────────────


def stop_all_schedulers():
    global _alert_scheduler, _maintenance_scheduler, _forecast_scheduler
    for name, scheduler in [("alerts", _alert_scheduler), ("maintenance", _maintenance_scheduler), ("forecast", _forecast_scheduler)]:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            _log.info("[%s] scheduler stopped", name)
    _alert_scheduler = None
    _maintenance_scheduler = None
    _forecast_scheduler = None
