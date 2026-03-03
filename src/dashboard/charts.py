"""Funzioni di visualizzazione Plotly condivise tra dashboard e analytics.

Ogni funzione restituisce un plotly.graph_objects.Figure pronto per
st.plotly_chart() o l'esportazione standalone.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import get_settings

_theme = get_settings()["dashboard"]["theme"]
_BG   = _theme["background"]
_TEXT = _theme["text"]
_GRID = _theme["grid"]
_POS  = _theme["positive"]
_NEG  = _theme["negative"]
_NEU  = _theme["neutral"]

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID),
    margin=dict(l=50, r=20, t=50, b=40),
)


def _axis_style(**kwargs) -> dict:
    return dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        tickfont=dict(color=_TEXT),
        title_font=dict(color=_TEXT),
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GEX Charts
# ──────────────────────────────────────────────────────────────────────────────

def gex_profile(gex_by_strike: list[dict], spot: float) -> go.Figure:
    """Grafico a barre del profilo GEX per strike.

    Args:
        gex_by_strike: lista di dict {strike, net_gex, call_gex, put_gex}.
        spot: prezzo spot BTC corrente.

    Returns:
        Figure Plotly.
    """
    if not gex_by_strike:
        fig = go.Figure()
        fig.update_layout(title="GEX Profile — Dati non disponibili", **_LAYOUT_BASE)
        return fig

    strikes  = [d["strike"] for d in gex_by_strike]
    net_gex  = [d.get("net_gex", 0) / 1e6 for d in gex_by_strike]
    colors   = [_POS if v >= 0 else _NEG for v in net_gex]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=strikes,
        y=net_gex,
        marker_color=colors,
        name="Net GEX (M$)",
        hovertemplate="Strike: $%{x:,.0f}<br>GEX: %{y:.2f}M$<extra></extra>",
    ))
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color=_NEU,
        annotation_text=f"Spot ${spot:,.0f}",
        annotation_font_color=_NEU,
    )
    fig.update_layout(
        title="Gamma Exposure per Strike",
        xaxis=_axis_style(title="Strike BTC ($)"),
        yaxis=_axis_style(title="Net GEX (M$)"),
        **_LAYOUT_BASE,
    )
    return fig


def gex_walls(snapshot_dict: dict) -> go.Figure:
    """Indicatore a gauge del GEX totale con put/call wall.

    Args:
        snapshot_dict: dict da GexCalculator.gex_to_dict().

    Returns:
        Figure Plotly.
    """
    gex_m = snapshot_dict.get("total_net_gex", 0) / 1e6
    spot  = snapshot_dict.get("spot_price", 0)
    put_w = snapshot_dict.get("put_wall", 0)
    call_w= snapshot_dict.get("call_wall", 0)
    flip  = snapshot_dict.get("gamma_flip_price", 0)

    fig = go.Figure()

    # Livelli chiave come linee orizzontali su un grafico prezzo
    for level, label, color in [
        (call_w, f"Call Wall ${call_w:,.0f}", _POS),
        (flip,   f"Gamma Flip ${flip:,.0f}",  _NEU),
        (spot,   f"Spot ${spot:,.0f}",         _TEXT),
        (put_w,  f"Put Wall ${put_w:,.0f}",    _NEG),
    ]:
        if level > 0:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[level, level],
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash="dash" if level != spot else "solid"),
            ))
            fig.add_annotation(
                x=1.01, y=level,
                text=label,
                showarrow=False,
                font=dict(color=color, size=11),
                xref="paper",
            )

    fig.update_layout(
        title=f"Livelli chiave — GEX totale: {gex_m:+.1f}M$",
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=_axis_style(title="Prezzo BTC ($)", tickformat="$,.0f"),
        height=300,
        **_LAYOUT_BASE,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ETF Flows Charts
# ──────────────────────────────────────────────────────────────────────────────

def flows_chart(merged_df: pd.DataFrame) -> go.Figure:
    """Grafico a 3 pannelli: flussi IBIT, prezzo BTC, correlazione rolling.

    Args:
        merged_df: DataFrame da FlowCorrelation.merge().

    Returns:
        Figure Plotly.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["IBIT Flows (M$)", "BTC Price ($)", "Correlazione rolling 30d"],
        vertical_spacing=0.06,
        row_heights=[0.3, 0.4, 0.3],
    )

    df = merged_df.dropna(subset=["ibit_flow"])

    if not df.empty:
        colors = [_POS if v >= 0 else _NEG for v in df["ibit_flow"]]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["ibit_flow"] / 1e6,
            marker_color=colors,
            name="IBIT Flow (M$)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}M$<extra></extra>",
        ), row=1, col=1)

    if "btc_close" in merged_df.columns:
        fig.add_trace(go.Scatter(
            x=merged_df.index,
            y=merged_df["btc_close"],
            name="BTC",
            line=dict(color=_NEU, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    if "ibit_flow" in merged_df.columns and "btc_return" in merged_df.columns:
        valid = merged_df[["ibit_flow", "btc_return"]].dropna()
        if len(valid) >= 30:
            corr = valid["ibit_flow"].rolling(30, min_periods=15).corr(valid["btc_return"])
            color_corr = [_POS if v >= 0 else _NEG for v in corr.fillna(0)]
            fig.add_trace(go.Bar(
                x=corr.index,
                y=corr,
                marker_color=color_corr,
                name="Corr 30d",
                hovertemplate="%{x|%Y-%m-%d}<br>corr=%{y:.3f}<extra></extra>",
            ), row=3, col=1)

    fig.add_hline(y=0, row=3, col=1, line_dash="dot", line_color=_GRID)

    fig.update_layout(
        height=600,
        showlegend=False,
        **_LAYOUT_BASE,
    )
    for i in range(1, 4):
        fig.update_xaxes(**_axis_style(), row=i, col=1)
        fig.update_yaxes(**_axis_style(), row=i, col=1)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Analytics Charts
# ──────────────────────────────────────────────────────────────────────────────

def granger_heatmap(granger_df: pd.DataFrame) -> go.Figure:
    """Heatmap dei p-values del test di Granger per lag e direzione.

    Args:
        granger_df: DataFrame da GrangerAnalysis.to_dataframe().

    Returns:
        Figure Plotly.
    """
    if granger_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Granger — Dati non disponibili", **_LAYOUT_BASE)
        return fig

    pivot = granger_df.pivot(index="direction", columns="lag", values="p_value")
    z = pivot.values
    x = [str(c) for c in pivot.columns]
    y = list(pivot.index)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, _POS], [0.05, "#ffcc00"], [1.0, _NEG]],
        zmin=0, zmax=0.15,
        colorbar=dict(title="p-value", tickfont=dict(color=_TEXT)),
        hovertemplate="Lag %{x} — %{y}<br>p=%{z:.4f}<extra></extra>",
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig.add_shape(
        type="line", x0=-0.5, x1=len(x)-0.5, y0=-0.5, y1=-0.5,
        line=dict(color=_TEXT, width=0),
    )
    fig.update_layout(
        title="Granger Causality p-values per Lag (verde = significativo, p<0.05)",
        xaxis=_axis_style(title="Lag (giorni)"),
        yaxis=_axis_style(title=""),
        height=250,
        **_LAYOUT_BASE,
    )
    return fig


def regime_bars(regime_result) -> go.Figure:
    """Grafico a barre comparativo per regime positive vs negative gamma.

    Args:
        regime_result: RegimeComparisonResult.

    Returns:
        Figure Plotly.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Return medio (%/day)", "Volatilità media (%)", "Sharpe ratio"],
    )

    labels, mean_rets, vols, sharpes, bar_colors = [], [], [], [], []
    for stats, color in [
        (regime_result.positive_stats, _POS),
        (regime_result.negative_stats, _NEG),
    ]:
        if stats:
            labels.append(stats.regime.replace("_", " ").title())
            mean_rets.append(stats.mean_return * 100)
            vols.append(stats.mean_vol * 100)
            sharpes.append(stats.sharpe)
            bar_colors.append(color)

    if not labels:
        fig.update_layout(title="Regime — Dati insufficienti", **_LAYOUT_BASE)
        return fig

    for col_idx, (y_vals, title) in enumerate(zip(
        [mean_rets, vols, sharpes],
        ["Return medio (%/day)", "Vol media (%)", "Sharpe"],
    ), 1):
        fig.add_trace(go.Bar(
            x=labels, y=y_vals,
            marker_color=bar_colors,
            name=title,
            showlegend=False,
            hovertemplate=f"{title}: %{{y:.3f}}<extra></extra>",
        ), row=1, col=col_idx)

    fig.update_layout(
        title=f"Regime Analysis — p-value: {regime_result.p_value:.4f}"
              + (" *** SIGNIFICATIVO" if regime_result.significant else ""),
        height=350,
        **_LAYOUT_BASE,
    )
    for i in range(1, 4):
        fig.update_xaxes(**_axis_style(), row=1, col=i)
        fig.update_yaxes(**_axis_style(), row=1, col=i)

    return fig


def backtest_equity(backtest_results: dict) -> go.Figure:
    """Equity curve + rendimenti giornalieri del backtest.

    Args:
        backtest_results: dict da Backtest.run() con "strategy" e "buy_and_hold".

    Returns:
        Figure Plotly.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Equity Curve (base=1.0)", "Rendimenti giornalieri"],
        vertical_spacing=0.1,
    )

    colors = {"strategy": _POS, "buy_and_hold": _NEU}

    for key, metrics in backtest_results.items():
        if metrics.equity_curve.empty:
            continue
        color = colors.get(key, _TEXT)
        fig.add_trace(go.Scatter(
            x=metrics.equity_curve.index,
            y=metrics.equity_curve,
            name=metrics.strategy_name,
            line=dict(color=color, width=2),
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=metrics.daily_returns.index,
            y=metrics.daily_returns,
            name=f"{metrics.strategy_name} daily",
            marker_color=color,
            opacity=0.5,
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        height=600,
        **_LAYOUT_BASE,
    )
    for i in range(1, 3):
        fig.update_xaxes(**_axis_style(), row=i, col=1)
        fig.update_yaxes(**_axis_style(), row=i, col=1)

    return fig


def event_study_car(event_results: list) -> Optional[go.Figure]:
    """CAR medio ± CI per ogni tipo di barriera.

    Args:
        event_results: lista di EventStudyResult.

    Returns:
        Figure Plotly o None se nessun evento.
    """
    if not event_results or all(r.n_events == 0 for r in event_results):
        return None

    fig = go.Figure()
    colors = [_POS, _NEG, _NEU, "#ff8800", "#cc44ff"]
    w = event_results[0].car_by_day and max(abs(k) for k in event_results[0].car_by_day.keys()) or 5
    days = list(range(-w, w + 1))

    for i, res in enumerate(event_results):
        if res.n_events == 0:
            continue
        color = colors[i % len(colors)]
        car_vals = [res.car_by_day.get(d, 0.0) for d in days]
        ci_spread = (res.ci_upper - res.ci_lower) / 2
        ci_upper = [v + ci_spread for v in car_vals]
        ci_lower = [v - ci_spread for v in car_vals]
        label = f"{res.barrier_type} (n={res.n_events})"
        if res.significant:
            label += " ***"

        fig.add_trace(go.Scatter(
            x=days, y=car_vals,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=days + days[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=_GRID)
    fig.add_vline(x=0, line_dash="dot", line_color=_TEXT, annotation_text="Event")
    fig.update_layout(
        title="Cumulative Abnormal Returns intorno ai Barrier Levels",
        xaxis=_axis_style(title="Giorni dall'evento"),
        yaxis=_axis_style(title="CAR (log return cumulativo)"),
        **_LAYOUT_BASE,
    )
    return fig
