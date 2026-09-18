from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

_BadgeColor = Literal["red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"]

_REGIME_LABEL = {
    "positive_gamma": "Stabilizzante",
    "negative_gamma": "Amplificante",
    "neutral": "Neutrale",
}
_REGIME_COLOR: dict[str, _BadgeColor] = {
    "positive_gamma": "green",
    "negative_gamma": "red",
    "neutral": "gray",
}


def _render_header(snap: dict, merged_df: pd.DataFrame) -> None:
    regime = snap.get("regime", "unknown")
    regime_label = _REGIME_LABEL.get(regime, regime.replace("_", " ").title())
    regime_color = _REGIME_COLOR.get(regime, "gray")

    title_col, badge_col = st.columns([4, 1], vertical_alignment="center")
    with title_col:
        st.title("ibit-gamma-tracker", icon=":material/currency_bitcoin:")
        st.caption("Analisi dealer hedging su note strutturate IBIT · BTC")
    with badge_col:
        st.badge(regime_label, icon=":material/circle:", color=regime_color)

    spot = snap.get("spot_price") or 0
    gex_m = (snap.get("total_net_gex") or 0) / 1e6

    with st.container(horizontal=True):
        st.metric(
            "BTC Spot",
            f"${spot:,.0f}",
            border=True,
            chart_data=_btc_spark(merged_df),
            chart_type="line",
        )
        st.metric(
            "GEX Totale",
            f"{gex_m:+,.1f}M$",
            border=True,
            help="Gamma exposure netta aggregata (Deribit).",
        )
        st.metric(
            "Gamma Flip",
            f"${snap.get('gamma_flip_price') or 0:,.0f}",
            delta=f"{_dist_pct(snap.get('gamma_flip_price'), spot):+.1f}% da spot",
            border=True,
            help="Prezzo al quale il GEX cambia segno.",
        )
        st.metric(
            "Put Wall",
            f"${snap.get('put_wall') or 0:,.0f}",
            delta=f"{_dist_pct(snap.get('put_wall'), spot):+.1f}% da spot",
            delta_color="inverse",
            border=True,
            help="Supporto meccanico: qui i dealer comprano.",
        )
        st.metric(
            "Call Wall",
            f"${snap.get('call_wall') or 0:,.0f}",
            delta=f"{_dist_pct(snap.get('call_wall'), spot):+.1f}% da spot",
            border=True,
            help="Resistenza meccanica: qui i dealer vendono.",
        )

    for alert in snap.get("alerts", []):
        st.warning(f"{alert}", icon=":material/warning:")


def _dist_pct(level: float | None, spot: float) -> float:
    """Distanza percentuale di `level` dallo spot (0 se non calcolabile)."""
    if not level or not spot:
        return 0.0
    return (level - spot) / spot * 100


def _btc_spark(merged_df: pd.DataFrame) -> list[float]:
    """Ultimi 30 prezzi di chiusura BTC per la sparkline del KPI spot."""
    if merged_df.empty or "btc_close" not in merged_df.columns:
        return []
    return merged_df["btc_close"].dropna().tail(30).tolist()
