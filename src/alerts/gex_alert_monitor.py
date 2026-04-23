"""GexAlertMonitor — orchestra daily recap + ETF flow event alerts.

Due metodi principali chiamati da APScheduler:
  - send_daily_recap(): cron 07:00 UTC, 1 invio/giorno garantito (cooldown 20h)
  - check_etf_flows(): interval 4h, invia solo se soglia conservativa superata
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from src.alerts.alert_db import AlertDB
from src.alerts.telegram_client import TelegramClient
from src.alerts.templates import (
    EtfFlowEvent,
    FlowsSummary,
    format_daily_recap,
    format_etf_flow_alert,
)
from src.config import get_settings, setup_logging
from src.flows.models import AggregateFlows
from src.gex.gex_db import GexDB
from src.gex.models import GexSnapshot
from src.gex.regime_detector import RegimeDetector

_log = setup_logging("alerts.monitor")

ALERT_DAILY_RECAP = "daily_recap"
ALERT_FLOW_SINGLE_DAY = "etf_flow_single_day"
ALERT_FLOW_CUMULATIVE = "etf_flow_cumulative_7d"
ALERT_FLOW_STREAK = "etf_flow_streak"


# ─── Helper functions (pure, testabili in isolamento) ────────────────────────


def summarize_flows(
    aggregates: list[AggregateFlows],
    *,
    today: Optional[date] = None,
) -> Optional[FlowsSummary]:
    """Estrae last-day + cumulativo 7d + streak da una lista di aggregates.

    Args:
        aggregates: lista ordinata per data crescente (da FarsideScraper.aggregate).
        today: data "oggi" per escludere il giorno corrente parziale (default = ultimo).

    Returns:
        FlowsSummary pronta per il template, o None se lista vuota.
    """
    if not aggregates:
        return None

    # Ordina per sicurezza
    aggs = sorted(aggregates, key=lambda a: a.date)

    # Last day = ultima riga disponibile
    last = aggs[-1]

    # Cumulativo 7d
    cutoff = last.date - timedelta(days=6)  # include last → 7 giorni totali
    last_7 = [a for a in aggs if a.date >= cutoff]
    cumul_7d = sum(a.total_flow_usd for a in last_7)

    # Streak: giorni consecutivi con lo stesso segno del last_day, partendo dalla coda
    streak = 0
    target_sign = 1 if last.total_flow_usd > 0 else (-1 if last.total_flow_usd < 0 else 0)
    if target_sign != 0:
        for a in reversed(aggs):
            a_sign = 1 if a.total_flow_usd > 0 else (-1 if a.total_flow_usd < 0 else 0)
            if a_sign == target_sign:
                streak += 1
            else:
                break
        streak *= target_sign

    return FlowsSummary(
        last_day_total_usd=last.total_flow_usd,
        last_day_by_ticker=dict(last.flows_by_ticker),
        cumul_7d_usd=cumul_7d,
        streak_days=streak,
        last_day_date=last.date.isoformat(),
    )


def evaluate_etf_flow_triggers(
    flows: FlowsSummary,
    *,
    single_day_threshold: float,
    cumulative_7d_threshold: float,
    streak_min_days: int,
) -> list[EtfFlowEvent]:
    """Valuta le soglie conservative e ritorna la lista di eventi da notificare.

    Args:
        flows: FlowsSummary calcolata da summarize_flows.
        single_day_threshold: soglia in USD per single-day (es. 500_000_000).
        cumulative_7d_threshold: soglia in USD per 7d cumulativo (es. 2_000_000_000).
        streak_min_days: minimo giorni streak per triggerare (es. 3).

    Returns:
        Lista di EtfFlowEvent ordinata per priorità (single_day, cumul, streak).
    """
    events: list[EtfFlowEvent] = []

    base = dict(
        last_day_total_usd=flows.last_day_total_usd or 0.0,
        last_day_by_ticker=dict(flows.last_day_by_ticker),
        cumul_7d_usd=flows.cumul_7d_usd or 0.0,
        streak_days=flows.streak_days,
        event_date=flows.last_day_date,
    )

    if flows.last_day_total_usd is not None and abs(flows.last_day_total_usd) >= single_day_threshold:
        events.append(EtfFlowEvent(trigger="single_day", threshold_usd=single_day_threshold, **base))

    if flows.cumul_7d_usd is not None and abs(flows.cumul_7d_usd) >= cumulative_7d_threshold:
        events.append(EtfFlowEvent(trigger="cumulative_7d", threshold_usd=cumulative_7d_threshold, **base))

    if abs(flows.streak_days) >= streak_min_days:
        events.append(EtfFlowEvent(trigger="streak", threshold_usd=float(streak_min_days), **base))

    return events


# ─── Monitor ─────────────────────────────────────────────────────────────────


class GexAlertMonitor:
    """Orchestra l'invio degli alert Telegram.

    Args:
        telegram: client pronto (o lo carica da env).
        gex_db: GexDB (per snapshot storici).
        alert_db: AlertDB (per dedup/cooldown).
        fetch_flows: callable sincrono che ritorna list[AggregateFlows].
            Default: usa FarsideScraper. Iniettabile per test.
        config: dict con sezione alerts (default da settings.yaml).
    """

    def __init__(
        self,
        *,
        telegram: Optional[TelegramClient] = None,
        gex_db: Optional[GexDB] = None,
        alert_db: Optional[AlertDB] = None,
        fetch_flows=None,
        config: Optional[dict] = None,
    ) -> None:
        cfg_all = get_settings()
        self._cfg = config or cfg_all.get("alerts", {})

        self._telegram = telegram or self._build_telegram_from_env()
        self._gex_db = gex_db or GexDB()
        self._alert_db = alert_db or AlertDB()
        self._fetch_flows = fetch_flows or self._default_fetch_flows

    @staticmethod
    def _build_telegram_from_env() -> Optional[TelegramClient]:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat:
            _log.warning(
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID mancanti — alert disabilitati"
            )
            return None
        return TelegramClient(bot_token=token, chat_id=chat)

    @staticmethod
    def _default_fetch_flows() -> list[AggregateFlows]:
        from src.flows.scraper import FarsideScraper

        scraper = FarsideScraper()
        return scraper.aggregate(scraper.fetch())

    # ─── Daily recap ─────────────────────────────────────────────────────────

    async def send_daily_recap(self) -> bool:
        """Invia il recap mattutino. Ritorna True se inviato con successo."""
        if self._telegram is None:
            _log.info("send_daily_recap skip: telegram non configurato")
            return False

        # Cooldown 20h per proteggere da doppie esecuzioni ravvicinate
        cooldown_h = float(self._cfg.get("cooldown_hours", 24)) - 4
        if self._alert_db.within_cooldown(ALERT_DAILY_RECAP, hours=cooldown_h):
            _log.info("send_daily_recap skip: entro cooldown %.0fh", cooldown_h)
            return False

        # GEX: ultimi 2 snapshot per delta
        latest = self._gex_db.get_latest_n(2)
        if not latest:
            _log.warning("send_daily_recap skip: nessun GEX snapshot nel DB")
            return False

        snap = latest[-1]
        prev = latest[-2] if len(latest) >= 2 else None

        # Regime: pre-popola storico per percentile corretto
        detector = RegimeDetector()
        history = self._gex_db.get_latest_n(90)
        # detect() aggiunge lo snap allo storico, quindi escludiamo l'ultimo se coincide
        prepop = history[:-1] if history and history[-1].timestamp == snap.timestamp else history
        detector.load_history_from_db(prepop)
        regime = detector.detect(snap)

        # Flows (best-effort, errori non bloccano il recap GEX)
        flows_summary: Optional[FlowsSummary] = None
        try:
            aggs = self._fetch_flows()
            flows_summary = summarize_flows(aggs)
        except Exception as exc:
            _log.warning("flows fetch failed in daily_recap: %s", exc)

        message = format_daily_recap(snap, prev, regime, flows_summary)

        sent = await self._telegram.send_message(message)
        if sent:
            self._alert_db.record_sent(ALERT_DAILY_RECAP, message)
            _log.info("daily_recap inviato")
        return sent

    # ─── ETF flow events ────────────────────────────────────────────────────

    async def check_etf_flows(self) -> int:
        """Controlla i flussi ETF e invia alert se superano soglie conservative.

        Returns:
            Numero di alert inviati in questo giro.
        """
        if self._telegram is None:
            return 0

        try:
            aggs = self._fetch_flows()
        except Exception as exc:
            _log.warning("flows fetch failed in check_etf_flows: %s", exc)
            return 0

        summary = summarize_flows(aggs)
        if summary is None:
            return 0

        flow_cfg = self._cfg.get("etf_flow_check", {})
        events = evaluate_etf_flow_triggers(
            summary,
            single_day_threshold=float(flow_cfg.get("single_day_usd_threshold", 500_000_000)),
            cumulative_7d_threshold=float(flow_cfg.get("cumulative_7d_usd_threshold", 2_000_000_000)),
            streak_min_days=int(flow_cfg.get("streak_min_days", 3)),
        )
        if not events:
            return 0

        # Arricchisci con contesto GEX corrente (best-effort)
        latest = self._gex_db.get_latest_n(1)
        snap = latest[-1] if latest else None

        cooldown_map = {
            "single_day": ALERT_FLOW_SINGLE_DAY,
            "cumulative_7d": ALERT_FLOW_CUMULATIVE,
            "streak": ALERT_FLOW_STREAK,
        }
        cooldown_h = float(self._cfg.get("cooldown_hours", 24))

        sent_count = 0
        for ev in events:
            alert_type = cooldown_map.get(ev.trigger, ev.trigger)
            if self._alert_db.within_cooldown(alert_type, hours=cooldown_h):
                _log.info("flow alert %s skip: cooldown attivo", alert_type)
                continue

            if snap is not None:
                ev.spot_price = snap.spot_price
            # Regime dal DB (ultimo snapshot)
            if snap is not None:
                # regime stringa nel DB non è esposta su GexSnapshot; la rileggiamo dalla row
                ev.gex_regime = self._last_regime_label()

            message = format_etf_flow_alert(ev)
            if self._alert_db.is_duplicate(alert_type, message):
                _log.info("flow alert %s skip: payload duplicato", alert_type)
                continue

            if await self._telegram.send_message(message):
                self._alert_db.record_sent(alert_type, message)
                sent_count += 1
                _log.info("flow alert inviato: type=%s", alert_type)

        return sent_count

    def _last_regime_label(self) -> Optional[str]:
        """Legge la colonna regime dell'ultimo snapshot dal DB."""
        try:
            with self._gex_db._conn() as conn:
                row = conn.execute(
                    "SELECT regime FROM gex_snapshots ORDER BY date DESC LIMIT 1"
                ).fetchone()
            return row["regime"] if row else None
        except Exception:
            return None
