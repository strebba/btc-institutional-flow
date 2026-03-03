"""Analisi del regime di gamma e sua relazione con la volatilità e i rendimenti BTC.

Calcola:
  - Rolling correlation GEX ↔ BTC realized volatility
  - Rendimenti medi BTC in regime positive vs negative gamma
  - Test di significatività (Welch t-test) sulla differenza tra regimi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_settings, setup_logging
from src.gex.models import GexSnapshot

_log = setup_logging("analytics.regime")


@dataclass
class RegimeStats:
    """Statistiche condizionali per un regime di gamma.

    Attributes:
        regime: nome del regime ("positive_gamma" | "negative_gamma" | "neutral").
        n_obs: numero di osservazioni nel regime.
        mean_return: rendimento medio BTC nel regime.
        std_return: deviazione standard dei rendimenti.
        mean_vol: volatilità realizzata media nel regime.
        sharpe: Sharpe ratio approssimato (mean/std * sqrt(252)).
        cum_return: rendimento cumulativo nel regime.
    """

    regime: str
    n_obs: int
    mean_return: float
    std_return: float
    mean_vol: float
    sharpe: float
    cum_return: float


@dataclass
class RegimeComparisonResult:
    """Risultato del confronto statistico tra regimi.

    Attributes:
        positive_stats: statistiche per il regime positive_gamma.
        negative_stats: statistiche per il regime negative_gamma.
        t_stat: Welch t-statistic sulla differenza di rendimenti medi.
        p_value: p-value del Welch t-test.
        significant: True se p < 0.05.
        gex_vol_correlation: correlazione rolling GEX ↔ BTC vol.
        interpretation: stringa interpretativa.
    """

    positive_stats: Optional[RegimeStats]
    negative_stats: Optional[RegimeStats]
    t_stat: float
    p_value: float
    significant: bool
    gex_vol_correlation: Optional[pd.Series]
    interpretation: str


class RegimeAnalysis:
    """Analizza la relazione tra il regime GEX e i rendimenti/volatilità BTC.

    Args:
        cfg: configurazione analytics (da settings.yaml).
        deribit_cfg: configurazione deribit per i threshold.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        deribit_cfg: dict | None = None,
    ) -> None:
        settings         = get_settings()
        self._cfg        = cfg or settings["analytics"]
        self._deribit_cfg = deribit_cfg or settings["deribit"]

    # ──────────────────────────────────────────────────────────────────────────
    # Data preparation
    # ──────────────────────────────────────────────────────────────────────────

    def build_gex_series(
        self,
        snapshots: list[GexSnapshot],
    ) -> pd.Series:
        """Costruisce una serie temporale del GEX totale dagli snapshot.

        Args:
            snapshots: lista di GexSnapshot ordinati per timestamp.

        Returns:
            pd.Series: serie con DatetimeIndex e valori GEX totale.
        """
        if not snapshots:
            return pd.Series(dtype=float)

        data = {
            pd.Timestamp(s.timestamp.date()): s.total_net_gex
            for s in snapshots
        }
        return pd.Series(data).sort_index()

    def classify_regime(self, gex_value: float) -> str:
        """Classifica il regime in base al valore GEX.

        Args:
            gex_value: valore GEX totale netto.

        Returns:
            str: "positive_gamma" | "negative_gamma" | "neutral".
        """
        threshold = self._deribit_cfg.get("gex_threshold_usd", 1_000_000)
        if gex_value > threshold:
            return "positive_gamma"
        elif gex_value < -threshold:
            return "negative_gamma"
        else:
            return "neutral"

    # ──────────────────────────────────────────────────────────────────────────
    # Regime analysis from GEX snapshots + price data
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        merged_df: pd.DataFrame,
        gex_series: Optional[pd.Series] = None,
        gex_col: str = "total_gex",
    ) -> RegimeComparisonResult:
        """Analizza i rendimenti BTC condizionati al regime GEX.

        Se gex_series è None, tenta di usare la colonna gex_col nel merged_df.

        Args:
            merged_df: DataFrame con btc_return, btc_vol_7d (colonne obbligatorie).
            gex_series: serie temporale del GEX (opzionale, per join).
            gex_col: nome della colonna GEX nel merged_df (se gex_series è None).

        Returns:
            RegimeComparisonResult.
        """
        df = merged_df.copy()

        # Join con GEX series se fornita
        if gex_series is not None and not gex_series.empty:
            df = df.join(gex_series.rename("_gex"), how="left")
            df["_gex"] = df["_gex"].ffill()  # forward fill per giorni senza GEX
        elif gex_col in df.columns:
            df["_gex"] = df[gex_col]
        else:
            _log.warning("GEX non disponibile nel DataFrame — uso GEX sintetico zero")
            df["_gex"] = 0.0

        # Classifica regime
        df["_regime"] = df["_gex"].apply(self.classify_regime)

        # Separa per regime
        pos_df = df[df["_regime"] == "positive_gamma"]
        neg_df = df[df["_regime"] == "negative_gamma"]

        def _compute_stats(regime_df: pd.DataFrame, regime_name: str) -> Optional[RegimeStats]:
            ret_col = "btc_return"
            vol_col = "btc_vol_7d"
            if ret_col not in regime_df.columns or len(regime_df) < 3:
                return None
            rets = regime_df[ret_col].dropna()
            vols = regime_df[vol_col].dropna() if vol_col in regime_df.columns else pd.Series()
            if len(rets) < 2:
                return None
            mean_r  = float(rets.mean())
            std_r   = float(rets.std())
            sharpe  = mean_r / std_r * (252 ** 0.5) if std_r > 0 else 0.0
            cum_ret = float(np.exp(rets.sum()) - 1)
            return RegimeStats(
                regime=regime_name,
                n_obs=len(rets),
                mean_return=mean_r,
                std_return=std_r,
                mean_vol=float(vols.mean()) if len(vols) > 0 else 0.0,
                sharpe=sharpe,
                cum_return=cum_ret,
            )

        pos_stats = _compute_stats(pos_df, "positive_gamma")
        neg_stats = _compute_stats(neg_df, "negative_gamma")

        # Welch t-test sulla differenza di rendimenti tra regimi
        t_stat, p_value = 0.0, 1.0
        if pos_stats and neg_stats:
            pos_rets = pos_df["btc_return"].dropna().values
            neg_rets = neg_df["btc_return"].dropna().values
            if len(pos_rets) >= 2 and len(neg_rets) >= 2:
                t_stat, p_value = stats.ttest_ind(pos_rets, neg_rets, equal_var=False)

        # Rolling correlation GEX ↔ BTC vol
        gex_vol_corr = None
        if "btc_vol_7d" in df.columns:
            valid = df[["_gex", "btc_vol_7d"]].dropna()
            if len(valid) >= 30:
                gex_vol_corr = (
                    valid.rolling(30, min_periods=15)
                    .corr()
                    .unstack()["btc_vol_7d"]["_gex"]
                )

        # Interpretazione
        interpretation = self._interpret(pos_stats, neg_stats, t_stat, p_value)

        return RegimeComparisonResult(
            positive_stats=pos_stats,
            negative_stats=neg_stats,
            t_stat=float(t_stat),
            p_value=float(p_value),
            significant=p_value < 0.05,
            gex_vol_correlation=gex_vol_corr,
            interpretation=interpretation,
        )

    def _interpret(
        self,
        pos: Optional[RegimeStats],
        neg: Optional[RegimeStats],
        t_stat: float,
        p_value: float,
    ) -> str:
        """Genera l'interpretazione testuale del confronto tra regimi.

        Args:
            pos: statistiche regime positive_gamma.
            neg: statistiche regime negative_gamma.
            t_stat: Welch t-statistic.
            p_value: p-value.

        Returns:
            str: interpretazione.
        """
        lines = ["=== Analisi Regime Gamma ===\n"]

        if pos:
            lines.append(
                f"Regime POSITIVE GAMMA ({pos.n_obs} giorni):\n"
                f"  Return medio:   {pos.mean_return*100:+.3f}% / giorno\n"
                f"  Vol media:      {pos.mean_vol*100:.1f}%\n"
                f"  Sharpe:         {pos.sharpe:.2f}\n"
                f"  Return cumulativo: {pos.cum_return*100:+.1f}%"
            )

        if neg:
            lines.append(
                f"\nRegime NEGATIVE GAMMA ({neg.n_obs} giorni):\n"
                f"  Return medio:   {neg.mean_return*100:+.3f}% / giorno\n"
                f"  Vol media:      {neg.mean_vol*100:.1f}%\n"
                f"  Sharpe:         {neg.sharpe:.2f}\n"
                f"  Return cumulativo: {neg.cum_return*100:+.1f}%"
            )

        if pos and neg:
            diff = pos.mean_return - neg.mean_return
            lines.append(
                f"\nDifferenza di rendimento (pos - neg): {diff*100:+.4f}%/giorno\n"
                f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f}"
            )
            if p_value < 0.05:
                direction = "superiori" if diff > 0 else "inferiori"
                lines.append(
                    f"✓ SIGNIFICATIVO: rendimenti in positive gamma "
                    f"statisticamente {direction} (p={p_value:.4f})"
                )
                lines.append(
                    "→ La teoria di Hayes è supportata: il regime di gamma "
                    "influenza significativamente i rendimenti BTC"
                    if diff > 0
                    else "→ Rendimenti maggiori in negative gamma (risultato controintuitivo)"
                )
            else:
                lines.append(
                    "✗ Non significativo (p > 0.05): "
                    "nessuna differenza statisticamente rilevante tra i regimi"
                )

        return "\n".join(lines)
