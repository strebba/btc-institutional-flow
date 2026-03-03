"""Test di Granger causality tra flussi ETF e rendimenti BTC.

Usa statsmodels.tsa.stattools.grangercausalitytests per testare:
  H1: IBIT_flows Granger-cause BTC_returns (i flussi predicono i rendimenti)
  H2: BTC_returns Granger-cause IBIT_flows (i rendimenti predicono i flussi)

Per lag da 1 a max_lags giorni.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from src.config import get_settings, setup_logging

_log = setup_logging("analytics.granger")


@dataclass
class GrangerResult:
    """Risultato del test di Granger causality.

    Attributes:
        direction: "flows→returns" o "returns→flows".
        lag: numero di lag.
        f_stat: F-statistic del test.
        p_value: p-value del test.
        significant: True se p_value < alpha.
        alpha: livello di significatività usato.
    """

    direction: str
    lag: int
    f_stat: float
    p_value: float
    significant: bool
    alpha: float = 0.05


class GrangerAnalysis:
    """Esegue il test di Granger causality su serie temporali finanziarie.

    Args:
        cfg: configurazione analytics (da settings.yaml).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["analytics"]

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def _check_stationarity(self, series: pd.Series, name: str) -> bool:
        """Verifica la stazionarietà con il test ADF (Augmented Dickey-Fuller).

        Args:
            series: serie temporale.
            name: nome per il logging.

        Returns:
            bool: True se la serie è stazionaria (p < 0.05).
        """
        clean = series.dropna()
        if len(clean) < 20:
            _log.warning("%s: troppo pochi dati per ADF (%d)", name, len(clean))
            return True  # assume stazionaria

        result  = adfuller(clean, autolag="AIC")
        p_value = result[1]
        is_stat = p_value < 0.05
        _log.info(
            "ADF %s: p=%.4f → %s",
            name, p_value, "stazionaria" if is_stat else "NON stazionaria",
        )
        return is_stat

    def _prepare_data(
        self,
        merged_df: pd.DataFrame,
        flow_col: str = "ibit_flow",
        return_col: str = "btc_return",
    ) -> pd.DataFrame:
        """Prepara e stazionarizza le serie per il test di Granger.

        Args:
            merged_df: DataFrame con flussi e rendimenti.
            flow_col: colonna dei flussi.
            return_col: colonna dei rendimenti.

        Returns:
            pd.DataFrame: DataFrame pulito con le due serie.
        """
        if flow_col not in merged_df.columns or return_col not in merged_df.columns:
            raise ValueError(f"Colonne mancanti: {flow_col}, {return_col}")

        df = merged_df[[flow_col, return_col]].dropna().copy()

        # Normalizza i flussi (in miliardi) per avere scale simili
        df[flow_col] = df[flow_col] / 1e9

        # Verifica stazionarietà
        stat_flows   = self._check_stationarity(df[flow_col],   flow_col)
        stat_returns = self._check_stationarity(df[return_col], return_col)

        # Se non stazionari, differenzia
        if not stat_flows:
            _log.info("Differenziando %s per stazionarietà", flow_col)
            df[flow_col] = df[flow_col].diff()
        if not stat_returns:
            _log.info("Differenziando %s per stazionarietà", return_col)
            df[return_col] = df[return_col].diff()

        return df.dropna()

    # ──────────────────────────────────────────────────────────────────────────
    # Granger test
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        merged_df: pd.DataFrame,
        flow_col: str = "ibit_flow",
        return_col: str = "btc_return",
        max_lags: int | None = None,
        alpha: float = 0.05,
    ) -> dict[str, list[GrangerResult]]:
        """Esegue il test di Granger causality bidirezionale.

        Args:
            merged_df: DataFrame con almeno flow_col e return_col.
            flow_col: nome colonna flussi IBIT.
            return_col: nome colonna rendimenti BTC.
            max_lags: massimo numero di lag (default da settings.yaml).
            alpha: livello di significatività.

        Returns:
            dict: "flows→returns" e "returns→flows", ognuno con lista di GrangerResult.
        """
        max_lags = max_lags or self._cfg.get("granger_max_lags", 10)

        try:
            df = self._prepare_data(merged_df, flow_col, return_col)
        except (ValueError, KeyError) as e:
            _log.error("Errore preparazione dati Granger: %s", e)
            return {"flows→returns": [], "returns→flows": []}

        if len(df) < max_lags * 3:
            _log.warning(
                "Dataset troppo corto (%d righe) per %d lag — riduco a %d",
                len(df), max_lags, len(df) // 3,
            )
            max_lags = max(1, len(df) // 3)

        results: dict[str, list[GrangerResult]] = {
            "flows→returns": [],
            "returns→flows": [],
        }

        # H1: flows Granger-cause returns → [returns, flows] come ordine
        _log.info("Test H1: IBIT flows → BTC returns")
        try:
            test1 = grangercausalitytests(
                df[[return_col, flow_col]].values,
                maxlag=max_lags,
                verbose=False,
            )
            for lag, lag_result in test1.items():
                f_stat  = lag_result[0]["ssr_ftest"][0]
                p_value = lag_result[0]["ssr_ftest"][1]
                results["flows→returns"].append(GrangerResult(
                    direction="flows→returns",
                    lag=lag,
                    f_stat=float(f_stat),
                    p_value=float(p_value),
                    significant=p_value < alpha,
                    alpha=alpha,
                ))
        except Exception as e:
            _log.error("Errore test H1: %s", e)

        # H2: returns Granger-cause flows → [flows, returns] come ordine
        _log.info("Test H2: BTC returns → IBIT flows")
        try:
            test2 = grangercausalitytests(
                df[[flow_col, return_col]].values,
                maxlag=max_lags,
                verbose=False,
            )
            for lag, lag_result in test2.items():
                f_stat  = lag_result[0]["ssr_ftest"][0]
                p_value = lag_result[0]["ssr_ftest"][1]
                results["returns→flows"].append(GrangerResult(
                    direction="returns→flows",
                    lag=lag,
                    f_stat=float(f_stat),
                    p_value=float(p_value),
                    significant=p_value < alpha,
                    alpha=alpha,
                ))
        except Exception as e:
            _log.error("Errore test H2: %s", e)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Interpretazione
    # ──────────────────────────────────────────────────────────────────────────

    def interpret(self, results: dict[str, list[GrangerResult]]) -> str:
        """Genera un'interpretazione testuale dei risultati.

        Args:
            results: dict dal metodo run().

        Returns:
            str: interpretazione leggibile.
        """
        lines: list[str] = ["=== Granger Causality Test ===\n"]

        for direction, res_list in results.items():
            if not res_list:
                continue
            sig = [r for r in res_list if r.significant]
            lines.append(f"Direzione: {direction}")

            if sig:
                min_lag = min(r.lag for r in sig)
                min_p   = min(r.p_value for r in sig)
                lines.append(
                    f"  ✓ SIGNIFICATIVO: causalità di Granger rilevata "
                    f"(p={min_p:.4f} al lag {min_lag})"
                )
                label = (
                    "I flussi IBIT predicono i rendimenti BTC"
                    if direction == "flows→returns"
                    else "I rendimenti BTC predicono i flussi IBIT"
                )
                lines.append(f"  → {label} con lag {min_lag}d")
            else:
                lines.append("  ✗ Non c'è evidenza di causalità di Granger (p > 0.05 per tutti i lag)")
            lines.append("")

        # Tabella risultati
        lines.append("Tabella p-values:\n")
        lines.append(f"{'Lag':>4}  {'flows→ret p':>12}  {'ret→flows p':>12}  {'flows→ret sig':>13}  {'ret→flows sig':>13}")
        lines.append("-" * 60)

        f2r = {r.lag: r for r in results.get("flows→returns", [])}
        r2f = {r.lag: r for r in results.get("returns→flows", [])}
        all_lags = sorted(set(list(f2r.keys()) + list(r2f.keys())))

        for lag in all_lags:
            r1 = f2r.get(lag)
            r2 = r2f.get(lag)
            p1 = f"{r1.p_value:.4f}" if r1 else "  n/a "
            p2 = f"{r2.p_value:.4f}" if r2 else "  n/a "
            s1 = "***" if r1 and r1.significant else "   "
            s2 = "***" if r2 and r2.significant else "   "
            lines.append(f"{lag:>4}  {p1:>12}  {p2:>12}  {s1:>13}  {s2:>13}")

        return "\n".join(lines)

    def to_dataframe(self, results: dict[str, list[GrangerResult]]) -> pd.DataFrame:
        """Converte i risultati in un DataFrame.

        Args:
            results: dict dal metodo run().

        Returns:
            pd.DataFrame: con colonne lag, direction, f_stat, p_value, significant.
        """
        rows = []
        for direction, res_list in results.items():
            for r in res_list:
                rows.append({
                    "lag":         r.lag,
                    "direction":   r.direction,
                    "f_stat":      r.f_stat,
                    "p_value":     r.p_value,
                    "significant": r.significant,
                })
        return pd.DataFrame(rows)
