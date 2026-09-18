"""Validation: Information Coefficient, Walk-Forward, Factor Decomposition, Parameter Sensitivity."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import setup_logging
from src.dashboard.data_loader import (
    run_factor_decomp,
    run_sensitivity,
    run_signal_ic,
    run_walk_forward,
)

_log = setup_logging("dashboard.tabs.validation")


def _tab_validation(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.header("Quantitative Validation", icon=":material/science:")
    st.caption(
        "Quattro test indipendenti sul CompositeSignal: Information Coefficient, "
        "Walk-Forward, Factor Decomposition, Parameter Sensitivity."
    )

    if merged_df.empty or "btc_return" not in merged_df.columns:
        st.warning("Dati insufficienti per la validazione quantitativa.")
        return

    _section_ic(merged_df, barriers)
    _section_walk_forward(merged_df, barriers)
    _section_factor_decomp(merged_df, barriers)
    _section_sensitivity(merged_df, barriers)

    st.warning(
        "Limiti: i test girano sullo stesso dataset del backtest (rischio data "
        "snooping). Diagnostica relativa, non garanzia di performance futura.",
        icon=":material/warning:",
    )


def _section_ic(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.subheader("1. Information Coefficient (IC)")
    st.caption("Correlazione tra segnale composito oggi e rendimento BTC domani.")

    with st.spinner("Calcolo IC in corso..."):
        try:
            ic_data = run_signal_ic(merged_df, barriers)
        except Exception as e:
            st.error(f"IC non disponibile: {e}")
            ic_data = None

    if ic_data is None or ic_data.get("rolling_ic") is None or ic_data["rolling_ic"].get("ic_mean") is None:
        st.info("Dati insufficienti per l'IC (servono almeno 20 giorni).")
        return

    ric = ic_data["rolling_ic"]
    null_model = ic_data.get("null_model", {})
    raw_ic = ic_data.get("raw_ic")

    with st.container(horizontal=True):
        st.metric("IC Mean (60d)", f"{ric['ic_mean']:.4f}", border=True)
        st.metric("t-stat", f"{ric['t_stat']:.2f}", border=True)
        st.metric("Information Ratio", f"{ric['ir']:.2f}", border=True)
        st.metric("% Periodi Positivi", f"{ric['pct_positive']:.0%}", border=True)

    if ric["is_significant"]:
        st.success(
            f"Segnale statisticamente significativo (IC = {ric['ic_mean']:.4f}, "
            f"t = {ric['t_stat']:.1f}).",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            f"Segnale non significativo (IC = {ric['ic_mean']:.4f}, t = {ric['t_stat']:.1f}).",
            icon=":material/warning:",
        )

    with st.expander("Dettaglio IC e null model"):
        if null_model:
            nm1, nm2 = st.columns(2)
            with nm1:
                st.metric("IC Reale (raw)", f"{raw_ic:.4f}" if raw_ic is not None else "N/A")
                st.metric("IC Nullo (mean)", f"{null_model.get('null_ic_mean', 'N/A'):.5f}" if null_model.get("null_ic_mean") is not None else "N/A")
            with nm2:
                st.metric("Null IC 95° pct", f"{null_model.get('null_ic_95pct', 'N/A'):.5f}" if null_model.get("null_ic_95pct") is not None else "N/A")
                st.metric("p-value empirico", f"{null_model.get('p_value_empirico', 'N/A'):.4f}" if null_model.get("p_value_empirico") is not None else "N/A")

        decay_df = ic_data.get("alpha_decay_df")
        if decay_df is not None and not decay_df.empty:
            st.subheader("Alpha decay (IC per orizzonte)")
            st.dataframe(decay_df.set_index("horizon"), width="stretch")


def _section_walk_forward(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.subheader("2. Walk-Forward Backtest")
    st.caption("Rolling train (2 anni) → test (3 mesi): robustezza out-of-sample.")

    with st.spinner("Walk-forward in corso..."):
        try:
            wf_analysis = run_walk_forward(merged_df, barriers)
        except Exception as e:
            st.error(f"Walk-forward non disponibile: {e}")
            wf_analysis = None

    if wf_analysis is None:
        st.info("Nessun dato per il walk-forward.")
        return
    if wf_analysis["total_periods"] == 0:
        st.warning(f"Dataset troppo corto (servono ~600 giorni; disponibili {len(merged_df)}).")
        return

    with st.container(horizontal=True):
        st.metric("Train Sharpe", f"{wf_analysis['avg_train_sharpe']:.2f}", border=True)
        st.metric(
            "Test Sharpe",
            f"{wf_analysis['avg_test_sharpe']:.2f}",
            delta=f"{wf_analysis['avg_test_sharpe'] - wf_analysis['avg_train_sharpe']:+.2f}",
            border=True,
        )
        st.metric("Sharpe Degradation", f"{wf_analysis['sharpe_degradation']:.0%}", border=True)
        st.metric("% Periodi Profittevoli", f"{wf_analysis['pct_profitable_periods']:.0%}", border=True)

    if wf_analysis["is_viable"]:
        st.success(
            f"VIABLE — il segnale sopravvive OOS ({wf_analysis['total_periods']} finestre).",
            icon=":material/check_circle:",
        )
    else:
        st.error(
            f"NOT VIABLE — il segnale non sopravvive OOS ({wf_analysis['total_periods']} finestre).",
            icon=":material/error:",
        )

    with st.expander("Dettaglio walk-forward"):
        from src.analytics.walk_forward import WalkForwardBacktest

        wfb = WalkForwardBacktest()
        st.dataframe(wfb.summary_table(wf_analysis), width="stretch")


def _section_factor_decomp(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.subheader("3. Factor Decomposition")
    st.caption("Regressione OLS dei rendimenti contro fattori noti: alpha vs beta.")

    with st.spinner("Factor decomposition in corso..."):
        try:
            fd_result = run_factor_decomp(merged_df, barriers)
        except Exception as e:
            st.error(f"Factor decomposition non disponibile: {e}")
            fd_result = None

    if fd_result is None:
        st.info("Dati insufficienti per la factor decomposition.")
        return

    total_ret = fd_result["decomposition"]["total_return"]
    true_alpha = fd_result["decomposition"]["true_alpha"]
    r_sq = fd_result["exposures"]["r_squared"]
    is_true_alpha = fd_result["decomposition"]["is_true_alpha"]

    with st.container(horizontal=True):
        st.metric("Strategy Return (ann)", f"{total_ret * 100:+.1f}%", border=True)
        st.metric("True Alpha (ann)", f"{true_alpha * 100:+.1f}%", border=True)
        st.metric("R²", f"{r_sq:.2f}", border=True)
        st.metric("Alpha % of Return", f"{fd_result['decomposition']['alpha_pct_of_return']:.0%}", border=True)

    if is_true_alpha:
        st.success("TRUE ALPHA — il segnale predice oltre i fattori noti.", icon=":material/check_circle:")
    else:
        st.warning("NO TRUE ALPHA — i rendimenti sono spiegati da esposizione a fattori noti.", icon=":material/warning:")

    with st.expander("Dettaglio factor exposures"):
        exp_data = [
            {
                "Factor": factor,
                "Beta": exp["beta"],
                "t-stat": exp["t_stat"],
                "p-value": exp["p_value"],
                "Significant": "***" if exp["significant"] else "",
            }
            for factor, exp in fd_result["exposures"].get("factor_exposures", {}).items()
        ]
        if exp_data:
            st.dataframe(pd.DataFrame(exp_data).set_index("Factor"), width="stretch")
        else:
            st.caption("Nessuna esposizione significativa rilevata.")


def _section_sensitivity(merged_df: pd.DataFrame, barriers: list[dict]) -> None:
    st.subheader("4. Parameter Sensitivity")
    st.caption("Test ±20% su ogni peso dei pilastri: stabilità dei pesi.")

    with st.spinner("Sensitivity in corso..."):
        try:
            sens_result = run_sensitivity(merged_df, barriers)
        except Exception as e:
            st.error(f"Sensitivity non disponibile: {e}")
            sens_result = None

    if sens_result is None:
        st.info("Dati insufficienti per la sensitivity analysis.")
        return

    unstable = [p for p, m in sens_result.items() if not m["is_stable"]]
    if unstable:
        st.warning(f"Pesi instabili rilevati: {', '.join(unstable)}.", icon=":material/warning:")
    else:
        st.success("Tutti i pesi sono stabili (±20% del peso → Sharpe <20%).", icon=":material/check_circle:")

    sens_data = [
        {
            "Pillar": pillar,
            "Base Sharpe": metrics["base_sharpe"],
            "Low (-20%)": metrics["low_sharpe"],
            "High (+20%)": metrics["high_sharpe"],
            "Range": metrics["range"],
            "Stable": "✓" if metrics["is_stable"] else "✗",
        }
        for pillar, metrics in sens_result.items()
    ]
    st.dataframe(pd.DataFrame(sens_data).set_index("Pillar"), width="stretch")
