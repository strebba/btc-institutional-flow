"""Formattazione messaggi HTML per Telegram.

Due template:
  - format_daily_recap(snap, prev, regime, flows_summary) → recap mattutino
  - format_etf_flow_alert(event) → alert event-driven su flow ETF
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.gex.models import GexSnapshot, RegimeState

DASHBOARD_URL = "https://seashell-app-h7hc4.ondigitalocean.app/btc"

_REGIME_EMOJI = {
    "positive_gamma": "🟢",
    "negative_gamma": "🔴",
    "neutral": "🟡",
}


def _fmt_money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.0f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.0f}K"
    return f"{sign}${a:.0f}"


def _fmt_signed_money(v: float) -> str:
    """Come _fmt_money ma con '+' esplicito per positivi."""
    if v > 0:
        return "+" + _fmt_money(v)
    return _fmt_money(v)


def _fmt_price(v: Optional[float]) -> str:
    return f"${v:,.0f}" if v is not None else "—"


def _pct_change(new: float, old: float) -> Optional[float]:
    if old == 0 or old is None:
        return None
    return (new - old) / abs(old) * 100


# ─── Daily recap ─────────────────────────────────────────────────────────────


@dataclass
class FlowsSummary:
    """Sintesi flussi ETF per il recap mattutino.

    Attributes:
        last_day_total_usd: flusso netto del giorno precedente (tutti gli ETF).
        last_day_by_ticker: breakdown per ticker del giorno precedente.
        cumul_7d_usd: cumulativo 7 giorni.
        streak_days: giorni consecutivi stessa direzione (+ se inflow, - se outflow).
        last_day_date: data del giorno precedente (YYYY-MM-DD) per il display.
    """

    last_day_total_usd: Optional[float]
    last_day_by_ticker: dict[str, float]
    cumul_7d_usd: Optional[float]
    streak_days: int
    last_day_date: Optional[str] = None


def format_daily_recap(
    snapshot: GexSnapshot,
    prev_snapshot: Optional[GexSnapshot],
    regime: RegimeState,
    flows: Optional[FlowsSummary],
    *,
    now: Optional[datetime] = None,
) -> str:
    """Genera il messaggio HTML del daily recap.

    Args:
        snapshot: GexSnapshot più recente.
        prev_snapshot: snapshot del giorno precedente (per delta %), può essere None.
        regime: RegimeState corrente da RegimeDetector.detect().
        flows: sintesi flussi ETF, può essere None se il fetch è fallito.
        now: sovrascrive datetime.now() per test deterministici.

    Returns:
        Stringa HTML pronta per Telegram parse_mode=HTML.
    """
    now = now or datetime.now(tz=timezone.utc)
    emoji = _REGIME_EMOJI.get(regime.regime, "⚪")

    lines: list[str] = []
    lines.append("📊 <b>BTC Institutional Flow — Daily Recap</b>")
    lines.append(f"<i>{now.strftime('%Y-%m-%d · %H:%M UTC')}</i>")
    lines.append("")

    # ── GEX block ──
    lines.append(f"{emoji} <b>Regime</b>: {regime.regime.upper().replace('_', ' ')}")
    lines.append(f"Spot: <b>{_fmt_price(snapshot.spot_price)}</b>")

    gex_line = f"Net GEX: <b>{_fmt_signed_money(snapshot.total_net_gex)}</b>"
    if prev_snapshot is not None:
        delta = _pct_change(snapshot.total_net_gex, prev_snapshot.total_net_gex)
        if delta is not None:
            gex_line += f"  (prev {_fmt_signed_money(prev_snapshot.total_net_gex)} · Δ {delta:+.0f}%)"
    lines.append(gex_line)

    lines.append(f"Gamma Flip: <b>{_fmt_price(snapshot.gamma_flip_price)}</b>")
    lines.append(
        f"Call Wall: {_fmt_price(snapshot.call_wall)}  |  "
        f"Put Wall: {_fmt_price(snapshot.put_wall)}"
    )

    if regime.gex_percentile is not None:
        lines.append(f"GEX percentile 90d: {regime.gex_percentile:.0f}%")

    # ── Flows block ──
    lines.append("")
    lines.append("💰 <b>ETF Flows</b>")
    if flows is None or flows.last_day_total_usd is None:
        lines.append("<i>Dati flows non disponibili</i>")
    else:
        arrow = "🟢" if flows.last_day_total_usd >= 0 else "🔴"
        label = f"Net ({flows.last_day_date})" if flows.last_day_date else "Net yesterday"
        lines.append(f"{label}: <b>{_fmt_signed_money(flows.last_day_total_usd)}</b> {arrow}")

        top = sorted(
            flows.last_day_by_ticker.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:3]
        if top:
            parts = [f"{tk} {_fmt_signed_money(v)}" for tk, v in top if v != 0]
            if parts:
                lines.append("Top: " + " · ".join(parts))

        if flows.cumul_7d_usd is not None:
            lines.append(f"7d cumul: <b>{_fmt_signed_money(flows.cumul_7d_usd)}</b>")

        if abs(flows.streak_days) >= 3:
            direction = "inflow" if flows.streak_days > 0 else "outflow"
            lines.append(f"Streak: {abs(flows.streak_days)} gg consecutivi {direction}")

    # ── Alert dal RegimeDetector ──
    if regime.alerts:
        lines.append("")
        lines.append("⚠️  <b>Alert</b>")
        for a in regime.alerts:
            lines.append(f"• {a}")

    # ── Footer ──
    lines.append("")
    lines.append(f'<a href="{DASHBOARD_URL}">Apri dashboard →</a>')

    return "\n".join(lines)


# ─── ETF flow event ──────────────────────────────────────────────────────────


@dataclass
class EtfFlowEvent:
    """Evento di flusso ETF che supera le soglie conservative.

    Attributes:
        trigger: 'single_day' | 'cumulative_7d' | 'streak'.
        last_day_total_usd: flusso netto del giorno.
        last_day_by_ticker: breakdown per ticker.
        cumul_7d_usd: cumulativo 7 giorni.
        streak_days: giorni consecutivi (segno = direzione).
        threshold_usd: soglia superata (valore dalla config).
        spot_price: BTC spot al momento del check, opzionale.
        gex_regime: regime GEX corrente, opzionale.
        event_date: data dell'evento (YYYY-MM-DD), opzionale.
    """

    trigger: str
    last_day_total_usd: float
    last_day_by_ticker: dict[str, float]
    cumul_7d_usd: float
    streak_days: int
    threshold_usd: float
    spot_price: Optional[float] = None
    gex_regime: Optional[str] = None
    event_date: Optional[str] = None


_TRIGGER_TITLE = {
    "single_day": "Single-Day Move",
    "cumulative_7d": "7-Day Cumulative",
    "streak": "Streak Conferma",
}


def format_etf_flow_alert(
    event: EtfFlowEvent,
    *,
    now: Optional[datetime] = None,
) -> str:
    """Genera messaggio HTML per alert event-driven su flow ETF."""
    now = now or datetime.now(tz=timezone.utc)

    direction_emoji = "🟢" if event.last_day_total_usd >= 0 else "🔴"
    kind = "Inflow" if event.last_day_total_usd >= 0 else "Outflow"
    subtitle = _TRIGGER_TITLE.get(event.trigger, event.trigger)

    lines: list[str] = []
    lines.append(f"🚨 <b>ETF Flow Alert — Large {kind}</b>")
    lines.append(f"<i>{now.strftime('%Y-%m-%d · %H:%M UTC')} · {subtitle}</i>")
    lines.append("")

    date_label = f" ({event.event_date})" if event.event_date else ""
    lines.append(
        f"Net flow today{date_label}: "
        f"<b>{_fmt_signed_money(event.last_day_total_usd)}</b> {direction_emoji}"
    )
    top = sorted(
        event.last_day_by_ticker.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:5]
    for tk, v in top:
        if v == 0:
            continue
        lines.append(f"  {tk}: {_fmt_signed_money(v)}")

    lines.append("")
    lines.append(f"7d cumulativo: <b>{_fmt_signed_money(event.cumul_7d_usd)}</b>")
    if event.streak_days and abs(event.streak_days) >= 1:
        direction = "inflow" if event.streak_days > 0 else "outflow"
        lines.append(f"Streak: {abs(event.streak_days)} gg consecutivi di {direction}")
    lines.append(f"Soglia superata: ±{_fmt_money(event.threshold_usd)}")

    if event.spot_price is not None or event.gex_regime is not None:
        lines.append("")
        if event.spot_price is not None:
            lines.append(f"BTC Spot: {_fmt_price(event.spot_price)}")
        if event.gex_regime is not None:
            lines.append(f"GEX regime: {event.gex_regime}")

    lines.append("")
    lines.append(f'<a href="{DASHBOARD_URL}">Apri dashboard →</a>')

    return "\n".join(lines)
