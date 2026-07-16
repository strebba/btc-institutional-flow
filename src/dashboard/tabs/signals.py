from __future__ import annotations

import pandas as pd
import streamlit as st


from src.dashboard.charts import composite_gauge, pillar_gauges, backtest_equity

from src.config import setup_logging
from src.dashboard.data_loader import compute_composite, run_backtest
from src.dashboard.data_loader import load_macro

_log = setup_logging("dashboard.tabs.signals")

def _tab_signals(snap: dict, merged_df: pd.DataFrame, barriers: list[dict]) -> None:

    st.header("🚦 Segnali Operativi")
    st.markdown("""
Il segnale combina i **4 pilastri** dell'analisi — **GEX**, **Barrier**, **ETF Flows**,
**Macro** — ognuno con un sotto-score 0-100. Il punteggio finale è un blend pesato.
**Non sono raccomandazioni di investimento** ma strumenti di lettura della microstruttura.
""")
    st.warning("""
⚠️ **Disclaimer**: modello sperimentale con dati limitati. Non sostituisce l'analisi
personale. Usa sempre gestione del rischio. Il backtest storico non garantisce
performance futura.
""")

    # Calcola il segnale composito (stessa logica di /api/signals)
    macro = load_macro()
    try:
        result = compute_composite(snap, merged_df, barriers, macro)
    except Exception as e:
        _log.warning("compute_composite fallito: %s", e)
        st.warning(f"Segnale composito temporaneamente non disponibile: {e}")
        st.info(
            "Verifica che i dati GEX, flussi e barriere siano caricati. "
            "Se il problema persiste, prova il refresh manuale dalla sidebar."
        )
        return
    signal = result.signal
    pillars = [
        {"name": p.name, "score": p.score, "weight": p.weight, "reason": p.reason}
        for p in result.pillars
    ]

    # ── Gauge top-level + banner ───────────────────────────────────────────────
    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(composite_gauge(result.score, signal), width="stretch")
    with g2:
        if signal == "LONG":
            st.success(
                "### 🟢 REGIME FAVOREVOLE\n\n"
                "Le condizioni strutturali favoriscono stabilità o rialzo graduale: "
                "il mercato tende al *mean-reverting*, i cali vengono assorbiti meccanicamente."
            )
        elif signal == "RISK_OFF":
            st.error(
                "### 🔴 REGIME DI RISCHIO\n\n"
                "Volatilità elevata e ribasso potenzialmente amplificato. "
                "**Azione**: ridurre leva, allargare stop, valutare coperture."
            )
        else:
            st.warning(
                "### 🟡 REGIME MISTO\n\n"
                "Segnali contrastanti tra i pilastri. Riduci esposizione e monitora: "
                "un cambiamento in un pilastro può spostare il segnale."
            )

    # ── Sotto-gauge dei 4 pilastri ─────────────────────────────────────────────
    st.plotly_chart(pillar_gauges(pillars), width="stretch")

    # ── Tabella leggibile dei pilastri ─────────────────────────────────────────
    def _emoji(score):
        if score is None:
            return "⚪️ n/d"
        if score >= 65:
            return "🟢"
        if score < 40:
            return "🔴"
        return "🟡"

    labels = {"gex": "GEX (dealer gamma)", "barrier": "Barrier (note EDGAR)",
              "etf_flows": "ETF Flows (domanda spot)", "macro": "Macro (derivati)"}
    rows = "\n".join(
        f"| {labels.get(p['name'], p['name'])} | {_emoji(p['score'])} "
        f"{('%.0f/100' % p['score']) if p['score'] is not None else ''} "
        f"| {p['weight']*100:.0f}% | {p['reason'] or '—'} |"
        for p in pillars
    )
    st.markdown(
        "| Pilastro | Score | Peso | Lettura |\n|:---|:---:|:---:|:---|\n" + rows
    )
    if not macro:
        st.caption(
            "ℹ️ Pilastro **Macro** non disponibile (CoinGlass non configurato in locale): "
            "i pesi sono riscalati sugli altri pilastri."
        )

    with st.expander("⚙️ Come viene calcolato il segnale (4 pilastri)"):
        st.markdown("""
Ogni pilastro produce un sotto-score 0-100; il segnale finale è il **blend pesato**
dei pilastri disponibili (i pesi si riscalano se un pilastro manca).

**1. GEX** — *Deribit options* — regime dealer gamma + contesto gamma-flip.
GEX positivo = dealer stabilizzano (cali comprati); negativo = amplificano.

**2. Barrier** — *note strutturate IBIT da SEC EDGAR* — livelli di hedging meccanico,
**direzionali e pesati per notional**: knock-in *sotto* lo spot = accelerante ribasso;
autocall *sopra* = resistenza. La vicinanza (kernel ~10%) aumenta il peso.

**3. ETF Flows** — *Farside/CoinGlass* — domanda spot istituzionale (momentum,
accelerazione, prezzo aggiustato per volatilità + flusso 3gg).

**4. Macro** — *CoinGlass derivati* — funding, OI, long/short, put/call, liquidazioni
(letti in chiave contrarian).

**Soglie**: score ≥ 65 → 🟢 LONG · 40-65 → 🟡 CAUTION · < 40 → 🔴 RISK_OFF.
""")

    st.divider()

    # Backtest results
    st.subheader("📈 Backtest della Strategia")
    st.caption("Periodo di test: dal lancio IBIT opzioni (Nov 2024) ad oggi.")

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

    # KPI backtest
    delta_sharpe = strat.sharpe_ratio - bah.sharpe_ratio
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric(
        "Sharpe Ratio",
        f"{strat.sharpe_ratio:.2f}",
        help="Rendimento risk-adjusted. >1 è buono, >2 è eccellente.",
    )
    bc2.metric(
        "Max Drawdown",
        f"{strat.max_drawdown * 100:.1f}%",
        help="Massima perdita dal picco. Più basso è, meglio è.",
    )
    bc3.metric(
        "Win Rate",
        f"{strat.win_rate * 100:.0f}%",
        help="Percentuale di giorni in profitto.",
    )
    bc4.metric(
        "vs Buy&Hold",
        f"{delta_sharpe:+.2f}",
        help="Delta Sharpe rispetto al buy-and-hold BTC.",
    )

    # Tabella comparativa
    table = bt.summary_table(results)
    st.dataframe(table, use_container_width=True)

    # Equity curve
    st.subheader("Equity Curve")
    st.plotly_chart(backtest_equity(results), use_container_width=True)

    st.caption("""
⚠️ Il backtest ha limitazioni significative: lo storico è breve (dal Nov 2024),
non include slippage e commissioni, e potrebbe soffrire di overfitting.
Usa i risultati come indicazione direzionale, non come garanzia di performance.
""")

    if strat.days_long == 0 and strat.days_short == 0:
        st.info(
            "La strategia è flat su tutto il periodo: il GEX storico non è ancora "
            "disponibile. Il backtest mostrerà segnali reali man mano che il sistema "
            "accumula snapshot GEX giornalieri."
        )


