from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import flows_chart, flows_stacked_chart, granger_heatmap

from src.dashboard.data_loader import run_granger


def _tab_flows(merged_df: pd.DataFrame) -> None:
    st.header("Flussi ETF Bitcoin", icon=":material/water:")
    st.caption(
        "Denaro istituzionale in entrata/uscita dagli ETF spot BTC ogni giorno "
        "(10+ ETF quotati)."
    )

    if merged_df.empty:
        st.warning("Dati flussi non disponibili.", icon=":material/warning:")
        return

    etf_tickers = sorted(
        [
            c.replace("_flow", "").upper()
            for c in merged_df.columns
            if c.endswith("_flow") and c != "total_flow"
        ]
    )

    today_flow = week_flow = total_today = corr_30d = 0.0
    if "ibit_flow" in merged_df.columns:
        ibit_col = merged_df["ibit_flow"].dropna()
        if not ibit_col.empty:
            today_flow = float(ibit_col.iloc[-1]) / 1e6
            week_flow = (
                float(
                    ibit_col[
                        ibit_col.index >= ibit_col.index.max() - pd.Timedelta(days=7)
                    ].sum()
                )
                / 1e6
            )
    if "total_flow" in merged_df.columns:
        total_col = merged_df["total_flow"].dropna()
        total_today = float(total_col.iloc[-1]) / 1e6 if not total_col.empty else 0.0
    if "ibit_flow" in merged_df.columns and "btc_return" in merged_df.columns:
        valid = merged_df[["ibit_flow", "btc_return"]].dropna()
        if len(valid) >= 30:
            corr_30d = float(
                valid["ibit_flow"]
                .rolling(30, min_periods=15)
                .corr(valid["btc_return"])
                .dropna()
                .iloc[-1]
            )

    n_inflow = n_outflow = 0
    for tk in etf_tickers:
        col = f"{tk.lower()}_flow"
        if col in merged_df.columns:
            last = merged_df[col].dropna()
            if not last.empty:
                if last.iloc[-1] > 0:
                    n_inflow += 1
                else:
                    n_outflow += 1

    with st.container(horizontal=True):
        st.metric("IBIT oggi", f"${today_flow:+,.0f}M", border=True)
        st.metric("IBIT 7gg", f"${week_flow:+,.0f}M", border=True)
        st.metric("Tutti gli ETF oggi", f"${total_today:+,.0f}M", border=True)
        st.metric("Corr. 30gg", f"{corr_30d:.2f}", border=True)

    _flow_alerts(merged_df, etf_tickers)

    if etf_tickers:
        with st.container(horizontal=True):
            for tk in etf_tickers:
                col = f"{tk.lower()}_flow"
                if col in merged_df.columns:
                    last = merged_df[col].dropna()
                    if not last.empty:
                        st.metric(tk, f"${float(last.iloc[-1]) / 1e6:+,.0f}M")

    st.plotly_chart(flows_chart(merged_df), width="stretch")

    if etf_tickers:
        st.subheader("Flussi per ETF (stacked)")
        st.plotly_chart(flows_stacked_chart(merged_df, etf_tickers), width="stretch")

    with st.expander("Riepilogo 30 giorni", icon=":material/calendar_today:"):
        r30 = merged_df[merged_df.index >= merged_df.index.max() - pd.Timedelta(days=30)]
        with st.container(horizontal=True):
            if "ibit_flow" in r30.columns:
                st.metric("Flusso IBIT (30d)", f"{r30['ibit_flow'].sum() / 1e6:+,.0f}M$", border=True)
                st.metric("Giorni inflow", f"{int((r30['ibit_flow'] > 0).sum())}/30", border=True)
            if "total_flow" in r30.columns:
                st.metric("Flusso totale ETF (30d)", f"{r30['total_flow'].sum() / 1e6:+,.0f}M$", border=True)
            if "btc_return" in r30.columns:
                btc_cum = float((1 + r30["btc_return"].dropna()).prod() - 1)
                st.metric("BTC Return (30d)", f"{btc_cum * 100:+.1f}%", border=True)

        if etf_tickers:
            ticker_rows = []
            for tk in etf_tickers:
                col = f"{tk.lower()}_flow"
                if col in r30.columns:
                    series = r30[col].dropna()
                    if not series.empty:
                        ticker_rows.append(
                            {
                                "ETF": tk,
                                "Net 30d (M$)": series.sum() / 1e6,
                                "Avg daily (M$)": series.mean() / 1e6,
                                "Max inflow (M$)": series.max() / 1e6,
                                "Max outflow (M$)": series.min() / 1e6,
                                "Inflow days": int((series > 0).sum()),
                            }
                        )
            if ticker_rows:
                st.dataframe(
                    pd.DataFrame(ticker_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "ETF": st.column_config.TextColumn("ETF"),
                        "Net 30d (M$)": st.column_config.NumberColumn(
                            "Net 30d", format="$%.0fM"
                        ),
                        "Avg daily (M$)": st.column_config.NumberColumn(
                            "Avg daily", format="$%.1fM"
                        ),
                        "Max inflow (M$)": st.column_config.NumberColumn(
                            "Max inflow", format="$%.0fM"
                        ),
                        "Max outflow (M$)": st.column_config.NumberColumn(
                            "Max outflow", format="$%.0fM"
                        ),
                        "Inflow days": st.column_config.NumberColumn("Inflow days"),
                    },
                )

    with st.expander("I flussi ETF predicono il prezzo?", icon=":material/query_stats:"):
        with st.spinner("Calcolo Granger causality..."):
            try:
                _, granger_df, granger_text = run_granger(merged_df)
                st.plotly_chart(granger_heatmap(granger_df), width="stretch")
                if not granger_df.empty:
                    st.caption(
                        "Un p-value < 0.05 (verde) significa che i flussi hanno potere "
                        "predittivo statisticamente significativo con quel ritardo."
                    )
                with st.expander("Tabella p-values"):
                    st.text(granger_text)
            except Exception as e:
                st.info(f"Granger causality non disponibile: {e}")


def _flow_alerts(merged_df: pd.DataFrame, etf_tickers: list[str]) -> None:
    """Alert concisi (una riga), dettaglio in expander."""
    alerts: list[tuple[str, str, str]] = []  # (icon, message, detail)

    if "ibit_flow_3d" in merged_df.columns:
        ibit_3d_col = merged_df["ibit_flow_3d"].dropna()
        if not ibit_3d_col.empty:
            ibit_3d = float(ibit_3d_col.iloc[-1]) / 1e6
            if ibit_3d < -500:
                alerts.append(
                    (
                        "error",
                        f"Deflussi pesanti: ${abs(ibit_3d):,.0f}M da IBIT in 3 giorni.",
                        "Livello che storicamente precede ulteriore debolezza di prezzo.",
                    )
                )
            elif ibit_3d < -200:
                alerts.append(
                    (
                        "warning",
                        f"Deflussi significativi: ${abs(ibit_3d):,.0f}M da IBIT in 3 giorni.",
                        "Controlla se è accompagnato da GEX negativo (tab GEX).",
                    )
                )
            elif ibit_3d > 300:
                alerts.append(
                    (
                        "success",
                        f"Afflussi solidi: +${ibit_3d:,.0f}M in IBIT in 3 giorni.",
                        "Domanda istituzionale forte; favorevole se con GEX positivo.",
                    )
                )

    inflow_etfs = [tk for tk in etf_tickers if _last_flow_m(merged_df, tk) > 50]
    outflow_etfs = [tk for tk in etf_tickers if _last_flow_m(merged_df, tk) < -50]
    if inflow_etfs and outflow_etfs:
        alerts.append(
            (
                "warning",
                f"Divergenza: {', '.join(inflow_etfs)} in inflow, {', '.join(outflow_etfs)} in outflow.",
                "Possibile rotazione tra emittenti: controlla il flusso aggregato.",
            )
        )

    for kind, msg, detail in alerts:
        fn = {"error": st.error, "warning": st.warning, "success": st.success}[kind]
        icon = {"error": ":material/error:", "warning": ":material/warning:", "success": ":material/check_circle:"}[kind]
        fn(msg, icon=icon)
        with st.expander("Dettaglio"):
            st.markdown(detail)


def _last_flow_m(merged_df: pd.DataFrame, tk: str) -> float:
    col = f"{tk.lower()}_flow"
    if col not in merged_df.columns:
        return 0.0
    last = merged_df[col].dropna()
    return float(last.iloc[-1]) / 1e6 if not last.empty else 0.0
