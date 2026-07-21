"""Validation tab: Information Coefficient, Walk-Forward, Factor Decomposition, Parameter Sensitivity.

Integra i nuovi moduli di quantitative research nella dashboard.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import setup_logging
from src.dashboard.data_loader import (
    run_walk_forward, run_factor_decomp, run_sensitivity, run_signal_ic,
)

_log = setup_logging("dashboard.tabs.validation")


def _tab_validation(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.header("🔬 Quantitative Validation")
    st.markdown("""
Validazione statistica del CompositeSignal a 4 pilastri. Quattro test indipendenti:
**Information Coefficient** (potere predittivo), **Walk-Forward** (robustezza OOS),
**Factor Decomposition** (alpha vs beta), **Parameter Sensitivity** (stabilità dei pesi).
""")

    if merged_df.empty or "btc_return" not in merged_df.columns:
        st.warning("Dati insufficienti per la validazione quantitativa.")
        return

    # ── Information Coefficient ─────────────────────────────────────────────────
    st.divider()
    st.subheader("1. Information Coefficient (IC)")
    st.caption(
        "Misura la correlazione tra il segnale composito oggi e il rendimento BTC "
        "di domani. IC > 0 e |t-stat| > 2 indicano che il segnale anticipa "
        "statisticamente il prezzo. Reference: patterns.md Alpha Signal Research."
    )

    with st.spinner("Calcolo IC in corso..."):
        try:
            ic_data = run_signal_ic(merged_df, barriers)
        except Exception as e:
            st.error(f"IC non disponibile: {e}")
            ic_data = None

    if ic_data is None or ic_data.get("rolling_ic") is None or ic_data["rolling_ic"].get("ic_mean") is None:
        st.info("Dati insufficienti per il calcolo dell'IC (servono almeno 20 giorni).")
    else:
        ric = ic_data["rolling_ic"]
        null_model = ic_data.get("null_model", {})
        raw_ic = ic_data.get("raw_ic")

        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric(
            "IC Mean (rolling 60d)",
            f"{ric['ic_mean']:.4f}",
            help="Correlazione di Spearman media su finestre rolling di 60 giorni. IC > 0.02 è considerato promettente.",
        )
        ic2.metric(
            "t-stat",
            f"{ric['t_stat']:.2f}",
            help="t-statistic dell'IC medio. |t| > 2 indica significatività statistica al 95%.",
        )
        ic3.metric(
            "Information Ratio",
            f"{ric['ir']:.2f}",
            help="IC medio / std(IC). IR > 0.5 indica un segnale stabile e consistente.",
        )
        ic4.metric(
            "% Periodi Positivi",
            f"{ric['pct_positive']:.0%}",
            help="Frazione di finestre rolling con IC positivo. > 50% = direzione consistente.",
        )

        if ric["is_significant"]:
            null_info = ""
            if null_model.get("actual_ic") and null_model.get("null_ic_95pct"):
                null_info = (
                    f" (IC = {null_model['actual_ic']:.4f} vs IC nullo = "
                    f"{null_model['null_ic_mean']:.4f} ± {null_model['null_ic_std']:.4f})"
                )
            st.success(
                f"✓ Il segnale è STATISTICAMENTE SIGNIFICATIVO "
                f"(IC medio = {ric['ic_mean']:.4f}, t = {ric['t_stat']:.1f}). "
                f"Il CompositeSignal anticipa la direzione del prezzo BTC meglio "
                f"del caso.{null_info}"
            )
        else:
            st.warning(
                f"⚠ Il segnale NON è statisticamente significativo "
                f"(IC medio = {ric['ic_mean']:.4f}, t = {ric['t_stat']:.1f}). "
                "Il potere predittivo non è distinguibile dal rumore."
            )

        with st.expander("📊 Dettaglio IC e Null Model"):
            if null_model:
                nm_col1, nm_col2 = st.columns(2)
                with nm_col1:
                    st.metric("IC Reale (raw)", f"{raw_ic:.4f}" if raw_ic is not None else "N/A")
                    st.metric("IC Nullo (mean)", f"{null_model.get('null_ic_mean', 'N/A'):.5f}" if null_model.get('null_ic_mean') is not None else "N/A")
                with nm_col2:
                    st.metric("Null IC 95° percentile", f"{null_model.get('null_ic_95pct', 'N/A'):.5f}" if null_model.get('null_ic_95pct') is not None else "N/A")
                    st.metric("p-value empirico", f"{null_model.get('p_value_empirico', 'N/A'):.4f}" if null_model.get('p_value_empirico') is not None else "N/A")
                st.caption(
                    "Il null model permuta i punteggi del segnale distruggendo la "
                    "struttura temporale ma preservando la distribuzione. Se l'IC "
                    "reale supera il 95esimo percentile dell'IC nullo, il segnale "
                    "ha potere predittivo genuino."
                )

            decay_df = ic_data.get("alpha_decay_df")
            if decay_df is not None and not decay_df.empty:
                st.subheader("Alpha Decay (IC per orizzonte)")
                st.caption(
                    "Mostra per quanti giorni il segnale mantiene potere predittivo. "
                    "Un decay rapido (IC → 0 entro 3-5 giorni) indica un segnale "
                    "di breve termine."
                )
                st.dataframe(decay_df.set_index("horizon"), width="stretch")

    # ── Walk-Forward ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("2. Walk-Forward Backtest")
    st.caption(
        "Rolling train (2 anni) → test (3 mesi). Misura se il segnale "
        "sopravvive out-of-sample. Reference: patterns.md Proper Backtest Framework."
    )

    with st.spinner("Walk-forward in corso..."):
        try:
            wf_analysis = run_walk_forward(merged_df, barriers)
        except Exception as e:
            st.error(f"Walk-forward non disponibile: {e}")
            wf_analysis = None

    if wf_analysis is None:
        st.info("Nessun dato per il walk-forward.")
    elif wf_analysis["total_periods"] == 0:
        st.warning(
            "Dataset troppo corto per il walk-forward (servono almeno ~600 giorni). "
            f"Disponibili: {len(merged_df)}."
        )
    else:
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric(
            "Train Sharpe (avg)",
            f"{wf_analysis['avg_train_sharpe']:.2f}",
            help="Sharpe medio sulle finestre di training (in-sample).",
        )
        wc2.metric(
            "Test Sharpe (avg)",
            f"{wf_analysis['avg_test_sharpe']:.2f}",
            delta=f"{(wf_analysis['avg_test_sharpe'] - wf_analysis['avg_train_sharpe']):+.2f}",
            help="Sharpe medio sulle finestre di test (out-of-sample). Più basso del train = overfitting.",
        )
        wc3.metric(
            "Sharpe Degradation",
            f"{wf_analysis['sharpe_degradation']:.0%}",
            help="Quanto peggiora lo Sharpe OOS. <50% è accettabile, >50% = overfitting severo.",
        )
        wc4.metric(
            "% Periodi Profittevoli",
            f"{wf_analysis['pct_profitable_periods']:.0%}",
            help="Frazione di finestre OOS con Sharpe positivo. <50% = il segnale è rumore.",
        )

        viability = wf_analysis["is_viable"]
        if viability:
            st.success(
                f"✓ VIABLE: il segnale sopravvive out-of-sample ({wf_analysis['total_periods']} finestre). "
                f"Sharpe OOS medio = {wf_analysis['avg_test_sharpe']:.2f}, "
                f"worst = {wf_analysis['worst_test_sharpe']:.2f}."
            )
        else:
            st.error(
                f"✗ NOT VIABLE: il segnale NON sopravvive out-of-sample ({wf_analysis['total_periods']} finestre). "
                f"Sharpe OOS medio = {wf_analysis['avg_test_sharpe']:.2f}. "
                "Possibile overfitting o rumore."
            )

        with st.expander("📊 Dettaglio walk-forward"):
            from src.analytics.walk_forward import WalkForwardBacktest
            wfb = WalkForwardBacktest()
            st.dataframe(wfb.summary_table(wf_analysis), width="stretch")

    # ── Factor Decomposition ───────────────────────────────────────────────────
    st.divider()
    st.subheader("3. Factor Decomposition")
    st.caption(
        "Regressione OLS dei rendimenti della strategia contro fattori noti "
        "(market, momentum, volatility). Separa alpha vero da beta mascherato. "
        "Reference: patterns.md Factor Model Construction."
    )

    with st.spinner("Factor decomposition in corso..."):
        try:
            fd_result = run_factor_decomp(merged_df, barriers)
        except Exception as e:
            st.error(f"Factor decomposition non disponibile: {e}")
            fd_result = None

    if fd_result is None:
        st.info("Dati insufficienti per la factor decomposition.")
    else:
        total_ret = fd_result["decomposition"]["total_return"]
        true_alpha = fd_result["decomposition"]["true_alpha"]
        r_sq = fd_result["exposures"]["r_squared"]
        alpha_t = fd_result["exposures"]["alpha_t_stat"]
        is_true_alpha = fd_result["decomposition"]["is_true_alpha"]

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric(
            "Strategy Return (ann)",
            f"{total_ret*100:+.1f}%",
            help="Rendimento annualizzato lordo della strategia.",
        )
        fc2.metric(
            "True Alpha (ann)",
            f"{true_alpha*100:+.1f}%",
            help=f"Intercetta OLS: rendimento non spiegato dai fattori. t={alpha_t:.1f}",
        )
        fc3.metric(
            "R²",
            f"{r_sq:.2f}",
            help="Frazione di varianza spiegata dai fattori. >0.7 = quasi tutta beta.",
        )
        fc4.metric(
            "Alpha % of Return",
            f"{fd_result['decomposition']['alpha_pct_of_return']:.0%}",
            help="Quanto del rendimento è vero alpha vs factor exposure.",
        )

        if is_true_alpha:
            st.success("✓ TRUE ALPHA: |t|>2 e alpha > 2%. Il segnale ha potere predittivo oltre i fattori noti.")
        else:
            reasons = []
            if abs(alpha_t) < 2:
                reasons.append(f"alpha non significativo (t={alpha_t:.1f})")
            if abs(true_alpha) <= 0.02:
                reasons.append("alpha troppo piccolo")
            if r_sq > 0.7:
                reasons.append(f"R² alto ({r_sq:.2f})")
            st.warning(f"✗ NO TRUE ALPHA: {'. '.join(reasons)}. I rendimenti sono spiegati da esposizione a fattori noti.")

        with st.expander("📊 Dettaglio factor exposures"):
            exp_data = []
            for factor, exp in fd_result["exposures"].get("factor_exposures", {}).items():
                exp_data.append({
                    "Factor": factor,
                    "Beta": exp["beta"],
                    "t-stat": exp["t_stat"],
                    "p-value": exp["p_value"],
                    "Significant": "***" if exp["significant"] else "",
                })
            if exp_data:
                st.dataframe(pd.DataFrame(exp_data).set_index("Factor"), width="stretch")
            else:
                st.caption("Nessuna esposizione significativa rilevata.")

    # ── Parameter Sensitivity ──────────────────────────────────────────────────
    st.divider()
    st.subheader("4. Parameter Sensitivity")
    st.caption(
        "Test ±20% su ogni peso dei pilastri. Se lo Sharpe varia >20%, "
        "il peso è fragile. Reference: sharp_edges.md curve-fitting-excuses."
    )

    with st.spinner("Sensitivity in corso..."):
        try:
            sens_result = run_sensitivity(merged_df, barriers)
        except Exception as e:
            st.error(f"Sensitivity non disponibile: {e}")
            sens_result = None

    if sens_result is None:
        st.info("Dati insufficienti per la sensitivity analysis.")
    else:
        unstable = [p for p, m in sens_result.items() if not m["is_stable"]]

        if unstable:
            st.warning(
                f"⚠️  Pesi instabili rilevati: **{', '.join(unstable)}**. "
                "Lo Sharpe varia >20% per una variazione ±20% del peso."
            )
        else:
            st.success("✓ Tutti i pesi sono stabili (variazione Sharpe <20% per ±20% del peso).")

        sens_data = []
        for pillar, metrics in sens_result.items():
            sens_data.append({
                "Pillar": pillar,
                "Base Sharpe": metrics["base_sharpe"],
                "Low (-20%)": metrics["low_sharpe"],
                "High (+20%)": metrics["high_sharpe"],
                "Range": metrics["range"],
                "Stable": "✓" if metrics["is_stable"] else "✗",
            })
        st.dataframe(pd.DataFrame(sens_data).set_index("Pillar"), width="stretch")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.divider()
    st.warning("""
⚠️ **Limiti della validazione**: i test sono eseguiti sullo stesso dataset usato
per il backtest (data snooping risk). Lo storico GEX è breve (~2-3 anni). I risultati
vanno interpretati come diagnostica relativa, non come garanzia di performance futura.
La validazione definitiva richiede un holdout set mai toccato (es. dati futuri).
""")
