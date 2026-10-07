from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from src.assets import AssetSpec, get_asset

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


def _render_header(snap: dict, merged_df: pd.DataFrame, spec: AssetSpec | None = None) -> None:
    spec = spec or get_asset("BTC")
    regime = snap.get("regime", "unknown")
    regime_label = _REGIME_LABEL.get(regime, regime.replace("_", " ").title())
    regime_color = _REGIME_COLOR.get(regime, "gray")

    title_col, badge_col = st.columns([4, 1], vertical_alignment="center")
    with title_col:
        st.title("ibit-gamma-tracker", icon=":material/currency_bitcoin:")
        if spec.has("barriers"):
            st.caption("Analisi dealer hedging su note strutturate IBIT · BTC")
        else:
            st.caption(f"Posizionamento dealer e flussi ETF {spec.lead_etf} · {spec.key}")
    with badge_col:
        st.badge(regime_label, icon=":material/circle:", color=regime_color)

    spot = snap.get("spot_price") or 0
    gex_m = (snap.get("total_net_gex") or 0) / 1e6

    with st.container(horizontal=True):
        st.metric(
            f"{spec.key} Spot",
            f"${spot:,.0f}",
            border=True,
            chart_data=_spot_spark(merged_df, spec),
            chart_type="line",
        )
        st.metric(
            "GEX Totale",
            f"{gex_m:+,.1f}M$",
            border=True,
            help="Gamma exposure netta aggregata (Deribit).",
        )
        _level_metric(
            "Gamma Flip", snap.get("gamma_flip_price"), spot,
            help="Prezzo al quale il GEX cambia segno.",
        )
        _level_metric(
            "Put Wall", snap.get("put_wall"), spot, delta_color="inverse",
            help="Supporto meccanico: qui i dealer comprano.",
        )
        _level_metric(
            "Call Wall", snap.get("call_wall"), spot,
            help="Resistenza meccanica: qui i dealer vendono.",
        )

    for alert in snap.get("alerts", []):
        st.warning(f"{alert}", icon=":material/warning:")


def _level_metric(
    label: str,
    level: float | None,
    spot: float,
    *,
    help: str,
    delta_color: Literal["normal", "inverse"] = "normal",
) -> None:
    """KPI di un livello GEX. Se il livello non esiste mostra "n/d", non "$0 a -100%"."""
    if not level:
        st.metric(label, "n/d", border=True, help=help)
        return
    st.metric(
        label,
        f"${level:,.0f}",
        delta=f"{_dist_pct(level, spot):+.1f}% da spot",
        delta_color=delta_color,
        border=True,
        help=help,
    )


def _dist_pct(level: float | None, spot: float) -> float:
    """Distanza percentuale di `level` dallo spot (0 se non calcolabile)."""
    if not level or not spot:
        return 0.0
    return (level - spot) / spot * 100


def _spot_spark(merged_df: pd.DataFrame, spec: AssetSpec) -> list[float]:
    """Ultimi 30 prezzi di chiusura dell'asset per la sparkline del KPI spot."""
    if merged_df.empty or spec.price_col not in merged_df.columns:
        return []
    return merged_df[spec.price_col].dropna().tail(30).tolist()
