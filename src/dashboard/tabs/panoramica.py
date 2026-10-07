"""Panoramica: la risposta a colpo d'occhio.

Ordine di lettura: stato sintetico (tape) → segnale composito (hero) →
posizionamento e flussi/derivati → prossimo trigger meccanico.

Per un asset senza segnale composito (ETH in fase 1) restano tape,
posizionamento e flussi/derivati: niente hero, pilastri né barriere.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.assets import AssetSpec, get_asset
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


def _tab_panoramica(
    snap: dict, merged_df: pd.DataFrame, barriers: list[dict], spec: AssetSpec | None = None
) -> None:
    spec = spec or get_asset("BTC")
    macro = load_macro(spec.key)
    if not spec.has("signal"):
        _panoramica_dati(snap, merged_df, macro, spec)
        return
    try:
        result = compute_composite(snap, merged_df, barriers, macro)
    except Exception as e:
        _log.warning("compute_composite fallito: %s", e)
        st.warning(f"Segnale composito non disponibile: {e}")
        _fallback(snap, merged_df, spec)
        return

    pillars = [
        {"name": p.name, "score": p.score, "weight": p.weight, "reason": p.reason}
        for p in result.pillars
    ]

    tape(_status_tape(snap, merged_df, barriers, macro, spec))

    col_hero, col_pillars = st.columns([1, 1.6], vertical_alignment="center")
    with col_hero:
        hero(result.score, result.signal, _hero_caption(snap, merged_df, spec))
    with col_pillars:
        pillar_bars(pillars)

    st.space("small")

    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        eyebrow("Posizionamento · spot vs livelli meccanici")
        st.plotly_chart(gex_walls(snap, asset=spec.key), width="stretch")
    with right:
        eyebrow("Flussi e derivati")
        _flows_block(merged_df, macro, spec)

    _next_trigger(snap, barriers)


def _panoramica_dati(snap: dict, merged_df: pd.DataFrame, macro: dict, spec: AssetSpec) -> None:
    """Panoramica senza segnale composito: stato, posizionamento, flussi e derivati."""
    tape(_status_tape(snap, merged_df, None, macro, spec))
    st.info(
        f"Segnale composito disponibile solo per BTC. Per {spec.key} la dashboard "
        f"mostra posizionamento dei dealer, flussi ETF e derivati, senza punteggio.",
        icon=":material/info:",
    )
    st.caption(_hero_caption(snap, merged_df, spec))

    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        eyebrow("Posizionamento · spot vs livelli meccanici")
        st.plotly_chart(gex_walls(snap, asset=spec.key), width="stretch")
    with right:
        eyebrow("Flussi e derivati")
        _flows_block(merged_df, macro, spec)


def _status_tape(
    snap: dict,
    merged_df: pd.DataFrame,
    barriers: list[dict] | None,
    macro: dict,
    spec: AssetSpec,
) -> str:
    regime = snap.get("regime") or "unknown"
    parts = [_REGIME_LABEL.get(regime, "Regime n/d")]
    flow_3d = _flow_3d_m(merged_df, spec)
    if flow_3d is not None:
        parts.append(f"{spec.lead_etf} 3gg {flow_3d:+,.0f}M")
    funding = macro.get("funding_rate_annualized_pct")
    if funding is not None:
        parts.append(f"funding {funding:+.1f}%")
    if barriers is not None:
        n = len(barriers)
        parts.append(f"{n} barriera attiva" if n == 1 else f"{n} barriere attive")
    return "  ·  ".join(parts)


def _hero_caption(snap: dict, merged_df: pd.DataFrame, spec: AssetSpec) -> str:
    spot = snap.get("spot_price") or 0
    ret = _last_return_pct(merged_df, spec)
    bits = [f"{spec.key} ${spot:,.0f}"]
    if ret is not None:
        bits.append(f"{ret:+.2f}% 24h")
    flip = snap.get("gamma_flip_price")
    if spot and flip:
        bits.append(f"flip a {(flip - spot) / spot * 100:+.1f}%")
    return "  ·  ".join(bits)


def _flows_block(merged_df: pd.DataFrame, macro: dict, spec: AssetSpec) -> None:
    lead_today = _last_flow_m(merged_df, spec.lead_flow_col)
    total_today = _last_flow_m(merged_df, "total_flow")
    funding = macro.get("funding_rate_annualized_pct")
    oi_change = macro.get("oi_change_7d_pct")

    with st.container(horizontal=True):
        if lead_today is not None:
            st.metric(f"{spec.lead_etf} oggi", f"${lead_today:+,.0f}M", border=True)
        if total_today is not None:
            st.metric("ETF oggi", f"${total_today:+,.0f}M", border=True)
        if funding is not None:
            st.metric("Funding (ann.)", f"{funding:+.1f}%", border=True)
        if oi_change is not None:
            st.metric("OI 7gg", f"{oi_change:+.1f}%", border=True)

    if funding is None and oi_change is None:
        if spec.has("signal"):
            st.caption("Macro non disponibile (CoinGlass non configurato): i pesi del pilastro sono riscalati.")
        else:
            st.caption("Macro non disponibile (CoinGlass non configurato).")

    if not merged_df.empty and spec.lead_flow_col in merged_df.columns:
        series = merged_df[spec.lead_flow_col].dropna().tail(30) / 1e6
        if not series.empty:
            st.caption(f"Flusso {spec.lead_etf}, ultimi 30 giorni (M$)")
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


def _fallback(snap: dict, merged_df: pd.DataFrame, spec: AssetSpec) -> None:
    """Se il composite fallisce, mostra almeno posizionamento e flussi."""
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(gex_walls(snap, asset=spec.key), width="stretch")
    with right:
        if not merged_df.empty:
            st.plotly_chart(flows_chart(merged_df, spec), width="stretch")


def _last_flow_m(merged_df: pd.DataFrame, column: str) -> float | None:
    if merged_df.empty or column not in merged_df.columns:
        return None
    last = merged_df[column].dropna()
    return float(last.iloc[-1]) / 1e6 if not last.empty else None


def _flow_3d_m(merged_df: pd.DataFrame, spec: AssetSpec) -> float | None:
    col = f"{spec.lead_flow_col}_3d"
    if merged_df.empty or col not in merged_df.columns:
        return None
    last = merged_df[col].dropna()
    return float(last.iloc[-1]) / 1e6 if not last.empty else None


def _last_return_pct(merged_df: pd.DataFrame, spec: AssetSpec) -> float | None:
    if merged_df.empty or spec.return_col not in merged_df.columns:
        return None
    last = merged_df[spec.return_col].dropna()
    return float(last.iloc[-1]) * 100 if not last.empty else None
