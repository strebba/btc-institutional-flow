"""Parameter sensitivity: testa la robustezza dei pesi.

La reference sharp_edges.md "curve-fitting-excuses" richiede che ogni
parametro abbia un rationale economico e che i risultati siano stabili
per variazioni di ±20%. Pesi con sensitività >20% sono fragili.

Il modulo testa sia i pesi dei pilastri (PILLAR_WEIGHTS) che i pesi
interni dei sotto-fattori (GEX, ETF, MACRO factor weights).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import pandas as pd

from src.analytics.pillars import (
    CompositeSignal,
    PILLAR_WEIGHTS,
    GEX_FACTOR_WEIGHTS,
    ETF_FACTOR_WEIGHTS,
    MACRO_FACTOR_WEIGHTS,
)
from src.analytics.backtest import Backtest
from src.config import setup_logging


@contextmanager
def _temp_weights(module, attr_name: str, test_weights: dict):
    """Temporarily replace factor weights, restoring on exit (even on exception)."""
    original = getattr(module, attr_name)
    setattr(module, attr_name, test_weights)
    try:
        yield
    finally:
        setattr(module, attr_name, original)

_log = setup_logging("analytics.sensitivity")


class ParameterSensitivity:
    """Testa la sensitività dello Sharpe del CompositeSignal alle variazioni
    dei pesi dei pilastri e dei sotto-fattori.

    Reference: patterns.md "Parameter Discipline" — ogni parametro deve
    passare il test di sensibilità ±20%.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        from src.config import get_settings

        self._cfg = cfg or get_settings()["backtest"]
        self._bt = Backtest(self._cfg)

    def _sharpe_for_signal(
        self,
        merged_df: pd.DataFrame,
        composite: CompositeSignal,
    ) -> float:
        """Calcola lo Sharpe della strategia con un dato CompositeSignal."""
        if "btc_return" not in merged_df.columns:
            return 0.0

        signals = self._bt._generate_signals(merged_df, composite=composite)
        signals_lagged = signals.shift(1).fillna(0.0)
        rets = merged_df["btc_return"].dropna()
        strat_rets = rets * signals_lagged.reindex(rets.index).fillna(0.0)
        metrics = self._bt._compute_metrics(strat_rets, "sensitivity")
        return metrics.sharpe_ratio

    def pillar_sensitivity(
        self,
        merged_df: pd.DataFrame,
        delta: float = 0.20,
        base_weights: Optional[dict[str, float]] = None,
        gex_series: Optional[pd.Series] = None,
        active_barriers: Optional[list[dict]] = None,
        barrier_history: Optional[pd.DataFrame] = None,
    ) -> dict[str, dict]:
        """Testa la sensitività di ogni peso dei pilastri (±delta%).

        Per ogni peso del pilastro:
          1. Calcola lo Sharpe base con i pesi originali.
          2. Calcola lo Sharpe con peso × (1 - delta) e peso × (1 + delta).
          3. Flagga come unstable se la variazione di Sharpe > 20%.

        Args:
            merged_df: DataFrame con colonne minime per CompositeSignal.
            delta: variazione frazionaria (default 0.20 = ±20%).
            base_weights: pesi pilastro personalizzati (default PILLAR_WEIGHTS).
            gex_series: serie GEX opzionale.
            active_barriers: barriere attive.
            barrier_history: storico barriere.

        Returns:
            dict: per ogni pillar_name → {base_sharpe, low_sharpe, high_sharpe,
            range, is_stable}.
        """
        weights = base_weights or PILLAR_WEIGHTS.copy()
        df = merged_df.copy()

        if df.empty or "btc_return" not in df.columns:
            _log.warning("DataFrame insufficiente per sensitivity")
            return {}

        if gex_series is not None and not gex_series.empty:
            df = df.join(gex_series.rename("_gex"), how="left")
            df["_gex"] = df["_gex"].ffill().fillna(0.0)

        base_composite = CompositeSignal(weights.copy())
        base_sharpe = self._sharpe_for_signal(df, base_composite)

        sensitivity: dict[str, dict] = {}

        for pillar in weights:
            if not isinstance(weights[pillar], (int, float)):
                continue
            w_base = weights[pillar]

            sharpes = []
            for mult in [1 - delta, 1 + delta]:
                test_weights = weights.copy()
                test_weights[pillar] = max(0.0, w_base * mult)
                s = sum(test_weights.values())
                test_weights = {k: v / s for k, v in test_weights.items()}

                test_composite = CompositeSignal(test_weights)
                sharpe = self._sharpe_for_signal(df, test_composite)
                sharpes.append(sharpe)

            low_s, high_s = sharpes[0], sharpes[1]
            sharpe_range = abs(high_s - low_s)
            is_stable = base_sharpe != 0 and (sharpe_range / abs(base_sharpe)) < 0.20

            sensitivity[pillar] = {
                "base_sharpe": round(base_sharpe, 3),
                "low_sharpe": round(low_s, 3),
                "high_sharpe": round(high_s, 3),
                "range": round(sharpe_range, 3),
                "is_stable": is_stable,
            }

        _log.info("Pillar sensitivity completata: %s",
                   {k: v["is_stable"] for k, v in sensitivity.items()})
        return sensitivity

    def subfactor_sensitivity(
        self,
        merged_df: pd.DataFrame,
        delta: float = 0.20,
        gex_series: Optional[pd.Series] = None,
        active_barriers: Optional[list[dict]] = None,
        barrier_history: Optional[pd.DataFrame] = None,
    ) -> dict[str, dict[str, dict]]:
        """Testa la sensitività dei pesi interni dei sotto-fattori.

        Tre gruppi di pesi:
          - GEX: regime vs flip
          - ETF: flow_momentum, flow_trend, price_momentum, flow_3d
          - MACRO: funding, oi_change, long_short, put_call, liquidations

        Per ogni fattore in ogni gruppo, testa ±delta% mantenendo gli altri
        invariati e ricalcolando lo Sharpe.

        Args:
            merged_df: DataFrame con colonne minime.
            delta: variazione frazionaria.
            gex_series: serie GEX opzionale.
            active_barriers: barriere attive.
            barrier_history: storico barriere.

        Returns:
            dict: {group_name: {factor_name: {base_sharpe, low_sharpe, high_sharpe, ...}}}
        """
        from src.analytics import pillars as _pillars_mod

        df = merged_df.copy()
        if df.empty or "btc_return" not in df.columns:
            return {}

        if gex_series is not None and not gex_series.empty:
            df = df.join(gex_series.rename("_gex"), how="left")
            df["_gex"] = df["_gex"].ffill().fillna(0.0)

        base_composite = CompositeSignal()
        base_sharpe = self._sharpe_for_signal(df, base_composite)

        groups: dict[str, dict[str, float]] = {
            "gex": GEX_FACTOR_WEIGHTS.copy(),
            "etf_flows": ETF_FACTOR_WEIGHTS.copy(),
            "macro": MACRO_FACTOR_WEIGHTS.copy(),
        }
        attr_map = {
            "gex": "GEX_FACTOR_WEIGHTS",
            "etf_flows": "ETF_FACTOR_WEIGHTS",
            "macro": "MACRO_FACTOR_WEIGHTS",
        }

        result: dict[str, dict[str, dict]] = {}

        for group_name, group_weights in groups.items():
            attr_name = attr_map[group_name]
            result[group_name] = {}
            for factor in group_weights:
                w_base = group_weights[factor]

                sharpes = []
                for mult in [1 - delta, 1 + delta]:
                    test_weights = group_weights.copy()
                    test_weights[factor] = max(0.0, w_base * mult)
                    s = sum(test_weights.values())
                    test_weights = {k: v / s for k, v in test_weights.items()}

                    with _temp_weights(_pillars_mod, attr_name, test_weights):
                        test_composite = CompositeSignal()
                        sharpe = self._sharpe_for_signal(df, test_composite)
                    sharpes.append(sharpe)

                low_s, high_s = sharpes[0], sharpes[1]
                sharpe_range = abs(high_s - low_s)
                is_stable = base_sharpe != 0 and (sharpe_range / abs(base_sharpe)) < 0.20

                result[group_name][factor] = {
                    "base_sharpe": round(base_sharpe, 3),
                    "low_sharpe": round(low_s, 3),
                    "high_sharpe": round(high_s, 3),
                    "range": round(sharpe_range, 3),
                    "is_stable": is_stable,
                }

        _log.info("Subfactor sensitivity completata")
        return result

    def summary_report(
        self,
        sensitivity_results: dict,
        title: str = "Parameter Sensitivity",
    ) -> str:
        """Genera un report testuale dei risultati di sensitivity.

        Args:
            sensitivity_results: output di pillar_sensitivity() o subfactor_sensitivity().
            title: titolo del report.

        Returns:
            str: report formattato.
        """
        if not sensitivity_results:
            return f"{title} — No results."

        lines = [f"=== {title} ===", ""]

        for param, metrics in sensitivity_results.items():
            if isinstance(metrics, dict) and "base_sharpe" in metrics:
                self._add_row(lines, param, metrics)
            elif isinstance(metrics, dict):
                lines.append(f"\n  Group: {param}")
                for factor, fm in metrics.items():
                    prefix = f"    {factor}"
                    self._add_row(lines, prefix, fm)

        unstable = self._unstable_params(sensitivity_results)
        if unstable:
            lines.append(f"\n⚠️  UNSTABLE parameters: {', '.join(unstable)}")
            lines.append("   Sharpe varia >20% per una variazione ±20% del peso.")
            lines.append("   Questi pesi sono fragili — rivedere il rationale o la calibrazione.")
        else:
            lines.append("\n✓  All parameters STABLE — Sharpe variation <20% for ±20% weight change.")

        return "\n".join(lines)

    @staticmethod
    def _add_row(lines: list[str], label: str, metrics: dict) -> None:
        status = "STABLE" if metrics["is_stable"] else "UNSTABLE"
        lines.append(
            f"  {label:<20} base={metrics['base_sharpe']:>7.3f}  "
            f"low={metrics['low_sharpe']:>7.3f}  high={metrics['high_sharpe']:>7.3f}  "
            f"range={metrics['range']:>7.3f}  {status}"
        )

    @staticmethod
    def _unstable_params(results: dict, prefix: str = "") -> list[str]:
        params = []
        for param, metrics in results.items():
            if isinstance(metrics, dict) and "is_stable" in metrics:
                if not metrics["is_stable"]:
                    params.append(f"{prefix}{param}")
            elif isinstance(metrics, dict):
                params.extend(
                    ParameterSensitivity._unstable_params(metrics, prefix=f"{param}.")
                )
        return params
