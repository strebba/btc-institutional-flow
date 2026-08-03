"""Factor decomposition: separa alpha da esposizione a fattori noti.

La maggior parte degli "alpha" è solo beta mascherato. Questo modulo
regredisce i rendimenti della strategia contro fattori di mercato
(market return, momentum, volatility) per identificare il vero alpha.

Reference: patterns.md "Factor Model Construction" — solo ~5% delle
strategie ha vero alpha dopo factor adjustment.
"""
from __future__ import annotations


import pandas as pd
import statsmodels.api as sm

from src.config import setup_logging

_log = setup_logging("analytics.factor_decomposition")


class FactorDecomposition:
    """Analizza l'esposizione a fattori e decomponi i rendimenti.

    Args:
        cfg: configurazione analytics (da settings.yaml).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        from src.config import get_settings

        self._cfg = cfg or get_settings()["analytics"]

    def calculate_factor_exposures(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> dict:
        """Calcola l'esposizione della strategia a fattori comuni.

        Regressione OLS: strategy_return = α + Σ(β_i × factor_i) + ε

        Args:
            returns: rendimenti giornalieri della strategia.
            factors: DataFrame con una colonna per fattore (es. market, momentum, vol).

        Returns:
            dict con factor_exposures (per fattore), alpha_annual, alpha_t_stat,
            alpha_significant, r_squared, unexplained_variance.
        """
        if returns.empty or factors.empty:
            return {
                "factor_exposures": {},
                "alpha_annual": 0.0,
                "alpha_t_stat": 0.0,
                "alpha_significant": False,
                "r_squared": 0.0,
                "unexplained_variance": 1.0,
            }

        aligned = pd.concat([returns, factors], axis=1).dropna()
        if aligned.empty or aligned.shape[1] < 2:
            return {
                "factor_exposures": {},
                "alpha_annual": 0.0,
                "alpha_t_stat": 0.0,
                "alpha_significant": False,
                "r_squared": 0.0,
                "unexplained_variance": 1.0,
            }

        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]

        if X.shape[1] == 0 or len(y) < 20:
            return {
                "factor_exposures": {},
                "alpha_annual": 0.0,
                "alpha_t_stat": 0.0,
                "alpha_significant": False,
                "r_squared": 0.0,
                "unexplained_variance": 1.0,
            }

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        exposures: dict[str, dict] = {}
        for col in X.columns:
            exposures[col] = {
                "beta": round(float(model.params.get(col, 0.0)), 4),
                "t_stat": round(float(model.tvalues.get(col, 0.0)), 3),
                "p_value": round(float(model.pvalues.get(col, 1.0)), 4),
                "significant": bool(model.pvalues.get(col, 1.0) < 0.05),
            }

        const_key = "const"
        alpha_annual = float(model.params.get(const_key, 0.0)) * 365
        alpha_t_stat = float(model.tvalues.get(const_key, 0.0))
        alpha_sig = bool(abs(alpha_t_stat) > 2.0)

        _log.info(
            "Factor regression: alpha=%.4f ann, t=%.2f, sig=%s, R²=%.3f, n=%d",
            alpha_annual, alpha_t_stat, alpha_sig, model.rsquared, len(y),
        )

        return {
            "factor_exposures": exposures,
            "alpha_annual": round(alpha_annual, 4),
            "alpha_t_stat": round(alpha_t_stat, 3),
            "alpha_significant": alpha_sig,
            "r_squared": round(float(model.rsquared), 4),
            "unexplained_variance": round(1.0 - float(model.rsquared), 4),
        }

    def decompose_strategy_returns(
        self,
        strategy_returns: pd.Series,
        factors: pd.DataFrame,
    ) -> dict:
        """Decompone i rendimenti in componente fattoriale e alpha puro.

        Args:
            strategy_returns: rendimenti giornalieri della strategia.
            factors: DataFrame con una colonna per fattore.

        Returns:
            dict con total_return, factor_contributions, true_alpha,
            alpha_pct_of_return, is_true_alpha.
        """
        analysis = self.calculate_factor_exposures(strategy_returns, factors)

        if not analysis["factor_exposures"]:
            total = float(strategy_returns.mean() * 365) if not strategy_returns.empty else 0.0
            return {
                "total_return": round(total, 4),
                "factor_contributions": {},
                "total_factor_return": 0.0,
                "true_alpha": round(total, 4),
                "alpha_pct_of_return": 1.0 if total != 0 else 0.0,
                "is_true_alpha": False,
            }

        aligned = pd.concat([strategy_returns, factors], axis=1).dropna()
        factor_contrib: dict[str, float] = {}
        for factor, exp in analysis["factor_exposures"].items():
            if exp["significant"] and factor in aligned.columns:
                factor_mean = float(aligned[factor].mean())
                contrib = exp["beta"] * factor_mean * 365
                factor_contrib[factor] = round(contrib, 4)

        total_return = float(strategy_returns.mean() * 365) if not strategy_returns.empty else 0.0
        total_factor_return = sum(factor_contrib.values())
        true_alpha = total_return - total_factor_return

        is_true_alpha = bool(
            analysis["alpha_significant"]
            and abs(true_alpha) > 0.02
        )

        return {
            "total_return": round(total_return, 4),
            "factor_contributions": factor_contrib,
            "total_factor_return": round(total_factor_return, 4),
            "true_alpha": round(true_alpha, 4),
            "alpha_pct_of_return": round(true_alpha / total_return, 4) if total_return != 0 else 0.0,
            "is_true_alpha": is_true_alpha,
        }

    def build_default_factors(
        self,
        btc_returns: pd.Series,
        window: int = 20,
    ) -> pd.DataFrame:
        """Costruisce i fattori di default per BTC dai rendimenti.

        Args:
            btc_returns: rendimenti giornalieri BTC (log returns).
            window: finestra per momentum (default 20gg).

        Returns:
            DataFrame con colonne: market, momentum, volatility.
        """
        factors = pd.DataFrame(index=btc_returns.index)

        factors["market"] = btc_returns

        factors["momentum"] = btc_returns.rolling(window, min_periods=5).sum()

        factors["volatility"] = btc_returns.rolling(7, min_periods=3).std()

        return factors.dropna()

    def summary_report(
        self,
        decomposition: dict,
        exposures: dict,
    ) -> str:
        """Genera un report testuale della decomposizione.

        Args:
            decomposition: output di decompose_strategy_returns().
            exposures: output di calculate_factor_exposures().

        Returns:
            str: report formattato.
        """
        if not decomposition or decomposition.get("total_return", 0) == 0 and not exposures.get("factor_exposures"):
            return "FACTOR DECOMPOSITION — Dati insufficienti."

        lines = ["=== FACTOR DECOMPOSITION ===\n"]

        lines.append(f"Total Return (ann):  {decomposition['total_return']*100:+.2f}%")
        lines.append(f"Factor Return (ann): {decomposition['total_factor_return']*100:+.2f}%")
        lines.append(f"True Alpha (ann):    {decomposition['true_alpha']*100:+.2f}%")
        lines.append(f"Alpha / Total:       {decomposition['alpha_pct_of_return']*100:.0f}%")
        lines.append(f"R²:                  {exposures.get('r_squared', 0):.3f}")
        lines.append(f"Unexplained:         {exposures.get('unexplained_variance', 1):.3f}")
        lines.append("")

        if decomposition["is_true_alpha"]:
            lines.append("VERDICT: TRUE ALPHA detected (|t|>2, alpha>2%).")
        else:
            reason = []
            if not exposures.get("alpha_significant", False):
                reason.append("alpha not significant (|t|<2)")
            if abs(decomposition["true_alpha"]) <= 0.02:
                reason.append(f"alpha too small ({decomposition['true_alpha']*100:+.2f}%)")
            if exposures.get("r_squared", 0) > 0.7:
                reason.append(f"high R² ({exposures['r_squared']:.2f})")
            lines.append(f"VERDICT: NO TRUE ALPHA — {'; '.join(reason) if reason else 'insufficient evidence'}.")
            lines.append("→ The strategy return is mostly factor exposure (disguised beta).")

        lines.append("")
        lines.append("Factor exposures:")
        lines.append(f"{'Factor':<14} {'Beta':>8} {'t-stat':>8} {'p-value':>8} {'Sig':>5}")
        lines.append("-" * 48)
        for factor, exp in exposures.get("factor_exposures", {}).items():
            sig = "***" if exp["significant"] else "   "
            lines.append(
                f"{factor:<14} {exp['beta']:>8.3f} {exp['t_stat']:>8.2f} {exp['p_value']:>8.4f} {sig:>5}"
            )

        return "\n".join(lines)
