"""Formattazione messaggi HTML per Telegram.

Tre template:
  - format_daily_recap(snap, prev, regime, flows_summary) → recap mattutino
  - format_etf_flow_alert(event) → alert event-driven su flow ETF
  - format_signal_message(...) → segnale direzionale giornaliero a 4 pilastri
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.analytics.pillars import PillarScore
from src.gex.models import GexSnapshot, GammaRegime

_REGIME_EMOJI = {
    "positive_gamma": "🟢",
    "negative_gamma": "🔴",
    "neutral": "🟡",
}

_IFI_EMOJI = {
    "Accumulation": "🟢",
    "Momentum": "🔵",
    "Neutral": "⚪",
    "Distribution": "🟠",
    "Outflow": "🔴",
}

_IFI_READING = {
    "Accumulation": "Istituzionali in forte accumulo — domanda strutturale superiore alla media storica",
    "Momentum": "Flussi in accelerazione, momentum positivo — trend in corso ma non ancora estremo",
    "Neutral": "Fase di equilibrio — nessuna convinzione direzionale, segnali misti",
    "Distribution": "Segnali di distribuzione — indebolimento strutturale, flussi in calo",
    "Outflow": "Deflusso istituzionale marcato — pressione ribassista strutturale dominante",
}


def _esc(text: str) -> str:
    """Escape HTML entities nei valori dinamici inseriti nei template."""
    return html.escape(text, quote=False)


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


def _mini_bar(v: Optional[float], width: int = 5) -> str:
    if v is None:
        return "—    "
    filled = round(max(0.0, min(1.0, v)) * width)
    return "█" * filled + "░" * (width - filled)


# ─── Daily recap ─────────────────────────────────────────────────────────────


@dataclass
class IFISummary:
    """Sintesi IFI per il recap mattutino."""

    score: float
    regime: str
    date: str
    flow_score: Optional[float] = None
    trend_score: Optional[float] = None
    price_score: Optional[float] = None


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
    regime: GammaRegime,
    flows: Optional[FlowsSummary],
    *,
    ifi: Optional[IFISummary] = None,
    now: Optional[datetime] = None,
) -> str:
    """Genera il messaggio HTML del daily recap.

    Args:
        snapshot: GexSnapshot più recente.
        prev_snapshot: snapshot del giorno precedente (per delta %), può essere None.
        regime: GammaRegime corrente da RegimeDetector.detect().
        flows: sintesi flussi ETF, può essere None se il fetch è fallito.
        ifi: sintesi IFI, può essere None se il DB è vuoto.
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
    lines.append(f"{emoji} <b>Regime</b>: {_esc(regime.regime.upper().replace('_', ' '))}")
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

    # ── IFI block ──
    lines.append("")
    lines.append("📈 <b>IFI — Institutional Flow Index</b>")
    if ifi is None:
        lines.append("<i>Dati IFI non disponibili</i>")
    else:
        ifi_emoji = _IFI_EMOJI.get(ifi.regime, "⚪")
        lines.append(
            f"Score: <b>{ifi.score:.0f}/100</b>  ·  {ifi_emoji} <b>{_esc(ifi.regime)}</b>"
        )
        reading = _IFI_READING.get(ifi.regime, "")
        if reading:
            lines.append(f"<i>{_esc(reading)}</i>")
        factors = []
        if ifi.flow_score is not None:
            factors.append(f"Flows {_mini_bar(ifi.flow_score)} {ifi.flow_score:.2f}")
        if ifi.trend_score is not None:
            factors.append(f"Trend {_mini_bar(ifi.trend_score)} {ifi.trend_score:.2f}")
        if ifi.price_score is not None:
            factors.append(f"Price {_mini_bar(ifi.price_score)} {ifi.price_score:.2f}")
        if factors:
            lines.append("  ·  ".join(factors))

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
            parts = [f"{_esc(tk)} {_fmt_signed_money(v)}" for tk, v in top if v != 0]
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
            lines.append(f"• {_esc(a)}")

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
        lines.append(f"  {_esc(tk)}: {_fmt_signed_money(v)}")

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
            lines.append(f"GEX regime: {_esc(event.gex_regime)}")

    return "\n".join(lines)


# ─── Directional signal (comando /signal) ─────────────────────────────────────

_BIAS_THRESHOLDS: list[tuple[float, str, str]] = [
    (40.0,  "LONG",          "🟢"),
    (15.0,  "LEGGERO LONG",  "🟢"),
    (-15.0, "NEUTRALE/FLAT", "🟡"),
    (-40.0, "LEGGERO SHORT", "🔴"),
]

_BIAS_READING: dict[str, str] = {
    "LONG":          "Condizioni favorevoli per esposizione long",
    "LEGGERO LONG":  "Condizioni moderatamente favorevoli — size ridotto",
    "NEUTRALE/FLAT": "Segnali contrastanti — restare piatti o attendere",
    "LEGGERO SHORT": "Condizioni moderatamente sfavorevoli — ridurre long",
    "SHORT":         "Condizioni favorevoli per esposizione short / cash",
}

_PILLAR_LABELS: dict[str, str] = {
    "gex": "GEX", "barrier": "BARRIER", "etf_flows": "ETF FLOWS", "macro": "MACRO",
}


def _pillar_arrow(score: Optional[float]) -> str:
    """Freccia direzionale per un pilastro 0-100."""
    if score is None:
        return "→"
    if score >= 65:
        return "↑"
    if score >= 55:
        return "↗"
    if score > 45:
        return "→"
    if score > 35:
        return "↘"
    return "↓"


def _bias_bar(bias: float, width: int = 30) -> str:
    """Barra visuale del bias da -100 a +100."""
    midpoint = width // 2
    pos = int(round((bias + 100) / 200 * width))
    pos = max(0, min(width - 1, pos))
    bar = ["░"] * width
    bar[midpoint] = "│"
    if pos != midpoint:
        bar[pos] = "█"
    else:
        bar[midpoint] = "╋"
    return "".join(bar)


def format_signal_message(
    *,
    score: float,
    bias: float,
    pillars: list[PillarScore],
    spot_price: Optional[float] = None,
    flip_price: Optional[float] = None,
    call_wall: Optional[float] = None,
    put_wall: Optional[float] = None,
    regime_label: str = "",
    barriers_count: int = 0,
    nearest_barrier: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> str:
    """Genera il messaggio HTML del segnale direzionale per il comando /signal.

    Args:
        score: punteggio composito 0-100.
        bias: bias direzionale netto -100/+100 (derivato dai pilastri).
        pillars: lista PillarScore dai 4 pilastri.
        spot_price: prezzo BTC spot corrente.
        flip_price: gamma flip price.
        call_wall: call wall Deribit.
        put_wall: put wall Deribit.
        regime_label: label regime GEX (positive_gamma, negative_gamma, neutral).
        barriers_count: numero di barriere EDGAR attive.
        nearest_barrier: dict con barrier_type, level_price_btc, distance_pct.
        now: sovrascrive datetime.now() per test deterministici.
    """
    now = now or datetime.now(tz=timezone.utc)

    verdict, verdict_emoji = "SHORT", "🔴"
    for thr, label, emoji in _BIAS_THRESHOLDS:
        if bias >= thr:
            verdict, verdict_emoji = label, emoji
            break

    reading = _BIAS_READING.get(verdict, "")
    regime_emoji = _REGIME_EMOJI.get(regime_label, "⚪")
    regime_display = regime_label.replace("_", " ").upper() if regime_label else "N/D"

    lines: list[str] = []
    lines.append("🚦 <b>BTC Directional Signal</b>")
    lines.append(f"<i>{now.strftime('%Y-%m-%d · %H:%M UTC')}</i>")
    lines.append("")

    lines.append("━━━ <b>VERDETTO</b> ━━━")
    lines.append(
        f"{verdict_emoji} <b>{verdict} ({bias:+.0f})</b>"
        + (f" — {_esc(reading)}" if reading else "")
    )
    lines.append("")

    lines.append(f"<code>{_bias_bar(bias)}</code>")
    lines.append("<code>SHORT ←────────────────|────────────────→ LONG</code>")
    lines.append("")

    lines.append("━━━ <b>PILASTRI</b> ━━━")
    for p in pillars:
        arrow = _pillar_arrow(p.score)
        label = _PILLAR_LABELS.get(p.name, p.name.upper())
        s = f"{p.score:.0f}/100" if p.score is not None else "n/d"
        w = f"{p.weight*100:.0f}%"
        line = f"{arrow} <b>{_esc(label)}</b>  {s} ({w})"
        if p.reason:
            line += f"  — {_esc(p.reason)}"
        lines.append(line)

    lines.append("")

    lines.append("━━━ <b>LIVELLI CHIAVE</b> ━━━")
    if spot_price is not None:
        lines.append(f"BTC: <b>{_fmt_price(spot_price)}</b>")
    if flip_price is not None and spot_price is not None and spot_price > 0:
        dist = (spot_price - flip_price) / spot_price * 100
        lines.append(f"Gamma Flip: {_fmt_price(flip_price)} ({dist:+.1f}%)")
    if call_wall is not None or put_wall is not None:
        cw = _fmt_price(call_wall) if call_wall else "—"
        pw = _fmt_price(put_wall) if put_wall else "—"
        lines.append(f"Call Wall: {cw}  ·  Put Wall: {pw}")
    if nearest_barrier:
        nb = nearest_barrier
        nb_type = _esc(str(nb.get("barrier_type", "?")))
        nb_price = nb.get("level_price_btc")
        nb_dist = nb.get("distance_pct")
        nb_str = (
            f"{nb_type} ${nb_price:,.0f}"
            if nb_price
            else f"{nb_type}"
        )
        if nb_dist is not None:
            nb_str += f" ({nb_dist:+.1f}%)"
        lines.append(f"Nearest Barrier: {nb_str}")
    elif barriers_count > 0:
        lines.append(f"Barriere attive: {barriers_count} — nessuna nel kernel")

    lines.append("")
    lines.append(
        f"<code>{regime_emoji} GEX: {_esc(regime_display)}"
        + (f" · {barriers_count} barriere attive" if barriers_count else "")
        + "</code>"
    )

    return "\n".join(lines)
