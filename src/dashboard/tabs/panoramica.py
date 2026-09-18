"""Panoramica: la risposta a colpo d'occhio.

Ordine di lettura: stato sintetico (tape) → segnale composito (hero) →
posizionamento e flussi/derivati → prossimo trigger meccanico.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import gex_walls, flows_chart
from src.dashboard.components import eyebrow, hero, pillar_bars, tape
from src.dashboard.data_loader import compute_composite, load_macro
from src.config import setup_logging

_log = setup_logging("dashboard.tabs.panoramica")

_REGIME_LABEL = {
    "positive_gamma": "Gamma positiva",
    "negative_gamma": "Gamma negativa",
    "neutral": "Gamma neutrale",
}


def _tab_panoramica(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    macro = load_macro()
    try:
        result = compute_composite(snap, merged_df, barriers, macro)
    except Exception as e:
        _log.warning("compute_composite fallito: %s", e)
        st.warning(f"Segnale composito non disponibile: {e}")
        _fallback(snap, merged_df, barriers)
        return

    pillars = [
        {"name": p.name, "score": p.score, "weight": p.weight, "reason": p.reason}
        for p in result.pillars
    ]

    tape(_status_tape(snap, merged_df, barriers, macro))

    col_hero, col_pillars = st.columns([1, 1.6], vertical_alignment="center")
    with col_hero:
        hero(result.score, result.signal, _hero_caption(snap, merged_df, macro))
    with col_pillars:
        pillar_bars(pillars)

    st.space("small")

    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        eyebrow("Posizionamento · spot vs livelli meccanici")
        st.plotly_chart(gex_walls(snap), width="stretch")
    with right:
        eyebrow("Flussi e derivati")
        _flows_block(merged_df, macro)

    _next_trigger(snap, barriers)


def _status_tape(
    snap: dict, merged_df: pd.DataFrame, barriers: list[dict], macro: dict
) -> str:
    regime = snap.get("regime") or "unknown"
    parts = [_REGIME_LABEL.get(regime, "Regime n/d")]
    flow_3d = _flow_3d_m(merged_df)
    if flow_3d is not None:
        parts.append(f"IBIT 3gg {flow_3d:+,.0f}M")
    funding = macro.get("funding_rate_annualized_pct")
    if funding is not None:
        parts.append(f"funding {funding:+.1f}%")
    n = len(barriers)
    parts.append(f"{n} barriera attiva" if n == 1 else f"{n} barriere attive")
    return "  ·  ".join(parts)


def _hero_caption(snap: dict, merged_df: pd.DataFrame, macro: dict) -> str:
    spot = snap.get("spot_price") or 0
    ret = _last_return_pct(merged_df)
    bits = [f"BTC ${spot:,.0f}"]
    if ret is not None:
        bits.append(f"{ret:+.2f}% 24h")
    flip = snap.get("gamma_flip_price")
    if spot and flip:
        bits.append(f"flip a {(flip - spot) / spot * 100:+.1f}%")
    return "  ·  ".join(bits)


def _flows_block(merged_df: pd.DataFrame, macro: dict) -> None:
    ibit_today = _last_flow_m(merged_df, "ibit_flow")
    total_today = _last_flow_m(merged_df, "total_flow")
    funding = macro.get("funding_rate_annualized_pct")
    oi_change = macro.get("oi_change_7d_pct")

    with st.container(horizontal=True):
        if ibit_today is not None:
            st.metric("IBIT oggi", f"${ibit_today:+,.0f}M", border=True)
        if total_today is not None:
            st.metric("ETF oggi", f"${total_today:+,.0f}M", border=True)
        if funding is not None:
            st.metric("Funding (ann.)", f"{funding:+.1f}%", border=True)
        if oi_change is not None:
            st.metric("OI 7gg", f"{oi_change:+.1f}%", border=True)

    if funding is None and oi_change is None:
        st.caption("Macro non disponibile (CoinGlass non configurato): i pesi del pilastro sono riscalati.")

    if not merged_df.empty and "ibit_flow" in merged_df.columns:
        series = merged_df["ibit_flow"].dropna().tail(30) / 1e6
        if not series.empty:
            st.caption("Flusso IBIT, ultimi 30 giorni (M$)")
            st.bar_chart(series, height=160, color="#00FF9D")


def _next_trigger(snap: dict, barriers: list[dict]) -> None:
    spot = snap.get("spot_price") or 0
    valid = [b for b in barriers if (b.get("level_price_btc") or 0) > 0 and spot > 0]
    if not valid:
        return

    closest = min(valid, key=lambda b: abs((b.get("level_price_btc") or 0) - spot))
    level = closest.get("level_price_btc") or 0
    distance = abs(spot - level) / spot * 100
    btype = (closest.get("barrier_type") or "barrier").replace("_", " ")
    issuer = closest.get("issuer") or "N/A"

    flip = snap.get("gamma_flip_price")
    flip_txt = ""
    if spot and flip:
        flip_txt = f" Gamma flip a {(flip - spot) / spot * 100:+.1f}% (${flip:,.0f})."

    msg = (
        f"Barriera più vicina: **{btype}** di {issuer} a ${level:,.0f} "
        f"(**{distance:.1f}%** dal prezzo).{flip_txt}"
    )
    if distance < 3:
        st.error(msg, icon=":material/error:")
    elif distance < 8:
        st.warning(msg, icon=":material/warning:")
    else:
        st.info(msg, icon=":material/info:")


def _fallback(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    """Se il composite fallisce, mostra almeno posizionamento e flussi."""
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(gex_walls(snap), width="stretch")
    with right:
        if not merged_df.empty:
            st.plotly_chart(flows_chart(merged_df), width="stretch")


def _last_flow_m(merged_df: pd.DataFrame, column: str) -> float | None:
    if merged_df.empty or column not in merged_df.columns:
        return None
    last = merged_df[column].dropna()
    return float(last.iloc[-1]) / 1e6 if not last.empty else None


def _flow_3d_m(merged_df: pd.DataFrame) -> float | None:
    if merged_df.empty or "ibit_flow_3d" not in merged_df.columns:
        return None
    last = merged_df["ibit_flow_3d"].dropna()
    return float(last.iloc[-1]) / 1e6 if not last.empty else None


def _last_return_pct(merged_df: pd.DataFrame) -> float | None:
    if merged_df.empty or "btc_return" not in merged_df.columns:
        return None
    last = merged_df["btc_return"].dropna()
    return float(last.iloc[-1]) * 100 if not last.empty else None
