from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import flows_chart, flows_stacked_chart, granger_heatmap

from src.dashboard.data_loader import run_granger

from src.config import get_settings

_settings = get_settings()
_theme = _settings["dashboard"]["theme"]
_TEXT_MUTED = _theme.get("text_muted", "#888888")

def _tab_flows(merged_df: pd.DataFrame) -> None:

    st.header("💰 Flussi ETF Bitcoin")
    st.markdown("""
I flussi degli ETF spot Bitcoin mostrano quanto denaro istituzionale entra o esce
dal mercato ogni giorno. Il mercato include ora 10+ ETF spot BTC.
""")

    if merged_df.empty:
        st.warning("Dati flussi non disponibili.")
        return

    # Detect all ETF tickers from merged columns
    etf_tickers = sorted(
        [
            c.replace("_flow", "").upper()
            for c in merged_df.columns
            if c.endswith("_flow") and c != "total_flow"
        ]
    )

    # KPI row
    today_flow = week_flow = total_today = 0.0
    corr_30d = 0.0

    if "ibit_flow" in merged_df.columns:
        ibit_col = merged_df["ibit_flow"].dropna()
        if not ibit_col.empty:
            today_flow = float(ibit_col.iloc[-1]) / 1e6
        week_flow = float(ibit_col[ibit_col.index >= ibit_col.index.max() - pd.Timedelta(days=7)].sum()) / 1e6 if not ibit_col.empty else 0.0

    if "total_flow" in merged_df.columns:
        total_col = merged_df["total_flow"].dropna()
        total_today = float(total_col.iloc[-1]) / 1e6 if not total_col.empty else 0.0

    if "ibit_flow" in merged_df.columns and "btc_return" in merged_df.columns:
        valid = merged_df[["ibit_flow", "btc_return"]].dropna()
        if len(valid) >= 30:
            corr_30d = (
                float(
                    valid["ibit_flow"]
                    .rolling(30, min_periods=15)
                    .corr(valid["btc_return"])
                    .dropna()
                    .iloc[-1]
                )
                if len(valid) >= 30
                else 0.0
            )

    # Number of ETFs with inflow today
    n_inflow = 0
    n_outflow = 0
    if etf_tickers:
        for tk in etf_tickers:
            col = f"{tk.lower()}_flow"
            if col in merged_df.columns:
                last_val = merged_df[col].dropna()
                if not last_val.empty:
                    if last_val.iloc[-1] > 0:
                        n_inflow += 1
                    else:
                        n_outflow += 1

    # KPI row a 3 colonne per leggibilità su tutti gli schermi
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "IBIT Flusso Oggi",
        f"${today_flow:+,.0f}M",
        help="Flusso netto IBIT nelle ultime 24h di trading",
    )
    col1.metric(
        "IBIT Flusso 7gg",
        f"${week_flow:+,.0f}M",
        help="Flusso netto cumulativo ultimi 7 giorni di borsa",
    )
    col2.metric(
        "Tutti gli ETF Oggi",
        f"${total_today:+,.0f}M",
        help="Flusso netto aggregato di tutti gli ETF spot BTC",
    )
    col2.metric(
        "Correlazione 30gg",
        f"{corr_30d:.2f}",
        help="Correlazione rolling 30gg flussi IBIT ↔ BTC return next-day",
    )
    col3.metric(
        "ETF con Inflow",
        str(n_inflow),
        help=f"ETF con flusso positivo oggi (su {len(etf_tickers)})",
    )
    col3.metric(
        "ETF con Outflow",
        str(n_outflow),
        help=f"ETF con flusso negativo oggi (su {len(etf_tickers)})",
    )

    # Per-ticker KPI strip (layout a griglia adattivo)
    if etf_tickers:
        st.markdown(
            f"""
<p style="font-size:10px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.07em;color:{_TEXT_MUTED};margin:0.5rem 0 0.3rem">
  Flussi per ETF (oggi)
</p>
""",
            unsafe_allow_html=True,
        )
        cols_per_row = 6
        for row_start in range(0, len(etf_tickers), cols_per_row):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = row_start + i
                if idx >= len(etf_tickers):
                    break
                tk = etf_tickers[idx]
                col = f"{tk.lower()}_flow"
                if col in merged_df.columns:
                    last_val = merged_df[col].dropna()
                    if not last_val.empty:
                        val_m = float(last_val.iloc[-1]) / 1e6
                        cols[i].metric(tk, f"${val_m:+,.0f}M")

    # Alert contestuale flussi
    if "ibit_flow_3d" in merged_df.columns:
        ibit_3d_col = merged_df["ibit_flow_3d"].dropna()
        if not ibit_3d_col.empty:
            ibit_3d = float(ibit_3d_col.iloc[-1]) / 1e6
            if ibit_3d < -500:
                st.error(f"""
🚨 **Deflussi pesanti**: IBIT ha perso **${abs(ibit_3d):,.0f}M** negli ultimi 3 giorni.
Questo livello di uscite indica probabile ribilanciamento istituzionale o liquidazione
di posizioni hedge. Storicamente, deflussi di questa entità precedono ulteriore
debolezza nel prezzo a breve termine.
""")
            elif ibit_3d < -200:
                st.warning(f"""
⚡ **Deflussi significativi**: ${abs(ibit_3d):,.0f}M usciti da IBIT in 3 giorni.
Il livello merita attenzione — controlla se è accompagnato da GEX negativo
(Tab Gamma Exposure) per un segnale più forte.
""")
            elif ibit_3d > 300:
                st.success(f"""
💪 **Afflussi solidi**: +${ibit_3d:,.0f}M in IBIT negli ultimi 3 giorni.
La domanda istituzionale è forte. Se accompagnata da GEX positivo,
il regime è favorevole per la stabilità del prezzo.
""")

    # Check for divergence between major ETFs
    inflow_etfs = []
    outflow_etfs = []
    for tk in etf_tickers:
        col = f"{tk.lower()}_flow"
        if col in merged_df.columns:
            last_val = merged_df[col].dropna()
            if not last_val.empty:
                val = float(last_val.iloc[-1])
                if val > 50e6:
                    inflow_etfs.append(tk)
                elif val < -50e6:
                    outflow_etfs.append(tk)

    if inflow_etfs and outflow_etfs:
        st.warning(f"""
⚖️ **Divergenza rilevata**: {", ".join(inflow_etfs)} in inflow mentre {", ".join(outflow_etfs)} in outflow.
Questo può indicare rotazione tra emittenti piuttosto che flusso netto reale.
Controlla il **flusso totale aggregato** per il quadro completo.
""")

    # Grafico principale
    st.plotly_chart(flows_chart(merged_df), width="stretch")
    st.caption("""
📉 **Correlazione Flussi-Prezzo**: Questo grafico mostra quanto i flussi ETF
"predicono" il rendimento di BTC il giorno successivo. Una correlazione alta
(>0.3) significa che il denaro istituzionale sta guidando il prezzo.
La correlazione tende ad aumentare durante i periodi di stress.
""")

    # Stacked bar chart per tutti gli ETF
    if etf_tickers:
        st.subheader("Flussi per ETF (Stacked)")
        st.plotly_chart(flows_stacked_chart(merged_df, etf_tickers), width="stretch")
        st.caption("""
📊 **Stacked Bar**: Ogni barra mostra il contributo di ogni ETF al flusso totale giornaliero.
Verde = inflow, Rosso = outflow. Utile per vedere quale emittente guida il flusso netto.
""")

    # Riepilogo 30gg
    st.subheader("Riepilogo ultimi 30 giorni")
    r30 = merged_df[merged_df.index >= merged_df.index.max() - pd.Timedelta(days=30)]
    rc1, rc2, rc3 = st.columns(3)
    if "ibit_flow" in r30.columns:
        total_30 = r30["ibit_flow"].sum() / 1e6
        rc1.metric("Flusso IBIT (30d)", f"{total_30:+.0f}M$")
        pos_days = int((r30["ibit_flow"] > 0).sum())
        rc2.metric("Giorni inflow", f"{pos_days}/30")
    if "total_flow" in r30.columns:
        total_30_all = r30["total_flow"].sum() / 1e6
        rc3.metric("Flusso Totale ETF (30d)", f"{total_30_all:+.0f}M$")
    if "btc_return" in r30.columns:
        btc_cum = float((1 + r30["btc_return"].dropna()).prod() - 1)
        st.metric("BTC Return (30d)", f"{btc_cum * 100:+.1f}%")

    # Per-ticker 30-day summary
    if etf_tickers:
        st.subheader("Riepilogo 30gg per ETF")
        ticker_rows = []
        for tk in etf_tickers:
            col = f"{tk.lower()}_flow"
            if col in r30.columns:
                series = r30[col].dropna()
                if not series.empty:
                    ticker_rows.append(
                        {
                            "ETF": tk,
                            "Net 30d (M$)": f"{series.sum() / 1e6:+,.0f}",
                            "Avg Daily (M$)": f"{series.mean() / 1e6:+,.1f}",
                            "Max Inflow (M$)": f"{series.max() / 1e6:+,.0f}",
                            "Max Outflow (M$)": f"{series.min() / 1e6:+,.0f}",
                            "Inflow Days": f"{(series > 0).sum()}",
                        }
                    )
        if ticker_rows:
            st.dataframe(pd.DataFrame(ticker_rows), width="stretch")

    # Granger expander
    with st.expander("📊 Test Statistico: i flussi ETF predicono il prezzo?"):
        with st.spinner("Calcolo Granger causality..."):
            try:
                _, granger_df, granger_text = run_granger(merged_df)
                st.plotly_chart(granger_heatmap(granger_df), width="stretch")

                # Tabella p-values formattata
                if not granger_df.empty:
                    st.markdown("""
**Test di Granger Causality** — Verifica se conoscere i flussi ETF di oggi
migliora la previsione del prezzo BTC di domani (rispetto a usare solo la storia dei prezzi).

Un **p-value < 0.05** (🟢) significa che i flussi hanno potere predittivo
statisticamente significativo con quel ritardo.
""")
                with st.expander("📋 Tabella completa p-values"):
                    st.text(granger_text)
            except Exception as e:
                st.info(f"Granger causality non disponibile: {e}")


