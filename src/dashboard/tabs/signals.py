from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import composite_gauge, pillar_gauges, backtest_equity

from src.config import setup_logging
from src.dashboard.data_loader import compute_composite, run_backtest
from src.dashboard.data_loader import load_macro

_log = setup_logging("dashboard.tabs.signals")

_LABELS = {
    "gex": "GEX (dealer gamma)",
    "barrier": "Barrier (note EDGAR)",
    "etf_flows": "ETF Flows",
    "macro": "Macro (derivati)",
}


def _tab_signals(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.header("Segnali Operativi", icon=":material/traffic:")
    st.caption(
        "Blend pesato di 4 pilastri — GEX, Barrier, ETF Flows, Macro. Strumento di "
        "lettura della microstruttura, non raccomandazione di investimento."
    )

    macro = load_macro()
    try:
        result = compute_composite(snap, merged_df, barriers, macro)
    except Exception as e:
        _log.warning("compute_composite fallito: %s", e)
        st.warning(f"Segnale composito temporaneamente non disponibile: {e}")
        st.info("Verifica che GEX, flussi e barriere siano caricati; prova il refresh.")
        return
    signal = result.signal
    pillars = [
        {"name": p.name, "score": p.score, "weight": p.weight, "reason": p.reason}
        for p in result.pillars
    ]

    g1, g2 = st.columns([1, 1], vertical_alignment="center")
    with g1:
        st.plotly_chart(composite_gauge(result.score, signal), width="stretch")
    with g2:
        if signal == "LONG":
            st.success(
                "**Regime favorevole** — condizioni strutturali verso stabilità o rialzo.",
                icon=":material/trending_up:",
            )
        elif signal == "RISK_OFF":
            st.error(
                "**Regime di rischio** — ridurre leva, allargare stop, valutare coperture.",
                icon=":material/trending_down:",
            )
        else:
            st.warning(
                "**Regime misto** — segnali contrastanti tra i pilastri.",
                icon=":material/swap_vert:",
            )

    st.plotly_chart(pillar_gauges(pillars), width="stretch")

    _pillar_table(pillars)
    if not macro:
        st.caption(
            "Pilastro Macro non disponibile (CoinGlass non configurato): i pesi sono "
            "riscalati sugli altri pilastri."
        )

    with st.expander("Come viene calcolato il segnale", icon=":material/settings:"):
        st.markdown(
            "Ogni pilastro produce un sotto-score 0-100; il segnale è il blend pesato "
            "(i pesi si riscalano se un pilastro manca).\n\n"
            "**GEX** — regime dealer gamma + contesto gamma-flip.\n"
            "**Barrier** — livelli di hedging meccanico dalle note EDGAR, direzionali e "
            "pesati per nozionale.\n"
            "**ETF Flows** — domanda spot istituzionale (momentum, accelerazione, 3gg).\n"
            "**Macro** — funding, OI, long/short, put/call, liquidazioni (chiave contrarian).\n\n"
            "**Soglie**: score ≥ 65 → LONG · 40-64 → CAUTION · < 40 → RISK_OFF."
        )

    st.divider()
    st.subheader("Backtest della strategia")
    st.caption("Periodo di test: dal lancio delle opzioni IBIT (Nov 2024) ad oggi.")

    if merged_df.empty:
        st.warning("Dati insufficienti per il backtest.")
        return

    with st.spinner("Esecuzione backtest..."):
        try:
            bt, results = run_backtest(merged_df, barriers)
        except Exception as e:
            st.error(f"Backtest non disponibile: {e}")
            return

    if not results:
        st.error("Backtest non disponibile.")
        return

    strat = results["strategy"]
    bah = results["buy_and_hold"]
    delta_sharpe = strat.sharpe_ratio - bah.sharpe_ratio

    with st.container(horizontal=True):
        st.metric("Sharpe Ratio", f"{strat.sharpe_ratio:.2f}", border=True)
        st.metric("Max Drawdown", f"{strat.max_drawdown * 100:.1f}%", border=True)
        st.metric("Win Rate", f"{strat.win_rate * 100:.0f}%", border=True)
        st.metric("vs Buy&Hold", f"{delta_sharpe:+.2f}", border=True)

    st.dataframe(bt.summary_table(results), width="stretch", hide_index=True)

    st.subheader("Equity curve")
    st.plotly_chart(backtest_equity(results), width="stretch")

    with st.expander("Limiti del backtest"):
        st.markdown(
            "Storico breve (da Nov 2024), nessuno slippage/commissioni, possibile "
            "overfitting. Usa i risultati come indicazione direzionale."
        )

    if strat.days_long == 0 and strat.days_short == 0:
        st.info(
            "Strategia flat su tutto il periodo: lo storico GEX non è ancora disponibile. "
            "Il backtest mostrerà segnali reali man mano che si accumulano snapshot."
        )


def _pillar_table(pillars: list[dict]) -> None:
    """Tabella leggibile dei 4 pilastri con score come progress bar."""
    rows = pd.DataFrame(
        [
            {
                "Pilastro": _LABELS.get(p["name"], p["name"]),
                "Score": p["score"] if p["score"] is not None else 0.0,
                "Peso": f"{p['weight'] * 100:.0f}%",
                "Lettura": p["reason"] or "—",
            }
            for p in pillars
        ]
    )
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Pilastro": st.column_config.TextColumn("Pilastro"),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f"
            ),
            "Peso": st.column_config.TextColumn("Peso"),
            "Lettura": st.column_config.TextColumn("Lettura"),
        },
    )
