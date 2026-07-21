"""Validazione statistica del CompositeSignal: Information Coefficient e alpha decay.

Risponde alla domanda: "Il segnale composito a 4 pilastri anticipa il prezzo BTC
con significatività statistica?"

Reference: patterns.md "Alpha Signal Research" — IC, t-stat, Information Ratio.
Reference: sharp_edges.md "multiple-testing-trap" — no p-hacking, IC su finestra rolling.

Best practice:
- No look-ahead: signal_t vs forward_return_{t+1}.
- Rolling IC: non una singola correlazione sull'intero dataset (data snooping),
  ma IC medio su finestre rolling walk-forward.
- Null model: confronto con IC di un segnale permutato (stessa distribuzione, nessuna
  struttura temporale).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import setup_logging
from src.forecast.validation import forward_returns

_log = setup_logging("analytics.signal_validation")

_SEED = 42


def spearman_ic(scores: pd.Series, forward_rets: pd.Series) -> Optional[float]:
    """Information Coefficient = Spearman rank correlation tra segnale e forward return.

    Args:
        scores: segnale al tempo t (0-100).
        forward_rets: rendimento realizzato t→t+1, indicizzato a t.

    Returns:
        float IC in [-1, 1] o None se dati insufficienti.
    """
    aligned = pd.concat([scores, forward_rets], axis=1).dropna()
    if len(aligned) < 10:
        return None
    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]
    if x.nunique() < 3 or y.nunique() < 3:
        return None
    rx = x.rank()
    ry = y.rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def rolling_information_coefficient(
    scores: pd.Series,
    forward_rets: pd.Series,
    window: int = 60,
    min_periods: int = 20,
) -> dict:
    """Calcola l'IC su finestra rolling e restituisce metriche aggregate.

    Reference: patterns.md "calculate_information_coefficient" — l'IC va
    calcolato su sub-periodi per stimare la stabilità del segnale.

    Args:
        scores: segnale 0-100.
        forward_rets: rendimento forward t→t+1.
        window: giorni della finestra rolling.
        min_periods: minimo osservazioni per finestra.

    Returns:
        dict: ic_mean, ic_std, ir (IC_mean/IC_std), t_stat, pct_positive,
        ic_series, is_significant (|t_stat| > 2), n_windows.
    """
    aligned = pd.concat(
        [scores.rename("score"), forward_rets.rename("fwd")], axis=1,
    ).dropna()
    if len(aligned) < min_periods:
        return {
            "ic_mean": None, "ic_std": None, "ir": None, "t_stat": None,
            "pct_positive": None, "ic_series": [], "is_significant": False,
            "n_windows": 0,
        }

    ic_values: list[float] = []
    for i in range(min_periods, len(aligned) + 1, 1):
        window_df = aligned.iloc[max(0, i - window) : i]
        if len(window_df) < min_periods:
            continue
        ic = spearman_ic(window_df.iloc[:, 0], window_df.iloc[:, 1])
        if ic is not None and not np.isnan(ic):
            ic_values.append(ic)

    if len(ic_values) < 3:
        return {
            "ic_mean": None, "ic_std": None, "ir": None, "t_stat": None,
            "pct_positive": None, "ic_series": ic_values, "is_significant": False,
            "n_windows": len(ic_values),
        }

    ic_array = np.array(ic_values, dtype=float)
    ic_mean = float(np.mean(ic_array))
    ic_std = float(np.std(ic_array, ddof=1))
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    t_stat = ic_mean / (ic_std / np.sqrt(len(ic_array))) if ic_std > 0 else 0.0
    pct_pos = float((ic_array > 0).mean())

    return {
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "ir": round(ir, 3),
        "t_stat": round(t_stat, 3),
        "pct_positive": round(pct_pos, 3),
        "ic_series": [round(float(v), 4) for v in ic_values],
        "is_significant": abs(t_stat) > 2.0,
        "n_windows": len(ic_values),
    }


def alpha_decay(
    scores: pd.Series,
    returns: pd.Series,
    max_horizon: int = 15,
) -> pd.DataFrame:
    """Calcola l'IC per orizzonti 1..max_horizon per analizzare il decay dell'alpha.

    Reference: patterns.md "alpha_decay_analysis".

    Args:
        scores: segnale 0-100, indicizzato per data.
        returns: rendimenti BTC giornalieri (non forward!).
        max_horizon: massimo orizzonte in giorni.

    Returns:
        DataFrame con colonne horizon, ic, t_stat, pct_positive.
    """
    rows = []
    for h in range(1, max_horizon + 1):
        fwd = forward_returns(returns, h)
        ic = spearman_ic(scores, fwd)
        ic_result = rolling_information_coefficient(scores, fwd)
        rows.append({
            "horizon": h,
            "ic": round(ic, 4) if ic is not None else None,
            "t_stat": ic_result.get("t_stat"),
            "pct_positive": ic_result.get("pct_positive"),
            "is_significant": ic_result.get("is_significant", False),
        })

    return pd.DataFrame(rows)


def null_model_ic(
    scores: pd.Series,
    forward_rets: pd.Series,
    n_permutations: int = 100,
) -> dict:
    """Calcola l'IC di un segnale nullo (permutato) per confronto.

    Permuta i punteggi per distruggere la struttura temporale ma preservare
    la distribuzione. Ripete n_permutations volte per stimare la distribuzione
    nulla dell'IC.

    Args:
        scores: segnale originale 0-100.
        forward_rets: rendimento forward t→t+1.
        n_permutations: numero di permutazioni.

    Returns:
        dict: null_ic_mean, null_ic_std, null_ic_95pct (95esimo percentile),
        actual_ic, p_value_empirico.
    """
    aligned = pd.concat(
        [scores.rename("score"), forward_rets.rename("fwd")], axis=1,
    ).dropna()
    if len(aligned) < 10:
        return {
            "null_ic_mean": None, "null_ic_std": None,
            "null_ic_95pct": None, "actual_ic": None,
            "p_value_empirico": None,
        }

    actual = spearman_ic(aligned.iloc[:, 0], aligned.iloc[:, 1])

    rng = np.random.default_rng(_SEED)
    null_ics: list[float] = []
    score_values = aligned.iloc[:, 0].values
    fwd_values = aligned.iloc[:, 1].values

    for _ in range(n_permutations):
        perm = rng.permutation(score_values)
        ic = spearman_ic(
            pd.Series(perm, index=aligned.index),
            pd.Series(fwd_values, index=aligned.index),
        )
        if ic is not None and not np.isnan(ic):
            null_ics.append(ic)

    if not null_ics:
        return {
            "null_ic_mean": None, "null_ic_std": None,
            "null_ic_95pct": None, "actual_ic": actual,
            "p_value_empirico": None,
        }

    null_array = np.array(null_ics, dtype=float)
    null_mean = float(np.mean(null_array))
    null_std = float(np.std(null_array, ddof=1))
    null_95 = float(np.percentile(null_array, 95))

    p_value = float((null_array >= actual).mean()) if actual is not None and actual > 0 else 1.0

    return {
        "null_ic_mean": round(null_mean, 5),
        "null_ic_std": round(null_std, 5),
        "null_ic_95pct": round(null_95, 5),
        "actual_ic": round(actual, 5) if actual is not None else None,
        "p_value_empirico": round(p_value, 4),
    }


def signal_validation_summary(
    df: pd.DataFrame,
    scores_col: str = "composite_score",
    return_col: str = "btc_return",
    window: int = 60,
    min_periods: int = 20,
) -> dict:
    """Validazione completa del segnale: IC, decay, null model.

    Single entry point per dashboard e API. Incapsula tutta la logica di
    validazione statistica in un'unica chiamata.

    Args:
        df: DataFrame con colonna del segnale e colonna dei rendimenti.
        scores_col: nome colonna con i punteggi 0-100.
        return_col: nome colonna con i rendimenti BTC giornalieri.
        window: finestra rolling per IC.
        min_periods: minimo osservazioni per finestra.

    Returns:
        dict con rolling_ic, alpha_decay_df, null_model, raw_ic.
    """
    scores = df.get(scores_col, pd.Series(dtype=float))
    returns = df.get(return_col, pd.Series(dtype=float))

    if scores.empty or returns.empty or len(scores) < min_periods:
        _log.warning("signal_validation_summary: dati insufficienti (%d righe)", len(scores))
        return {
            "rolling_ic": {"ic_mean": None, "t_stat": None, "is_significant": False},
            "alpha_decay_df": pd.DataFrame(),
            "null_model": {},
            "raw_ic": None,
        }

    fwd_1d = forward_returns(returns, horizon=1)
    raw_ic = spearman_ic(scores, fwd_1d)

    rolling = rolling_information_coefficient(scores, fwd_1d, window=window, min_periods=min_periods)
    decay_df = alpha_decay(scores, returns, max_horizon=15)
    null_model = null_model_ic(scores, fwd_1d, n_permutations=100)

    _log.info(
        "Signal validation: IC=%.4f, rolling_IC=%.4f, t=%.2f, sig=%s",
        raw_ic or 0.0,
        rolling.get("ic_mean") or 0.0,
        rolling.get("t_stat") or 0.0,
        rolling.get("is_significant"),
    )

    return {
        "rolling_ic": rolling,
        "alpha_decay_df": decay_df,
        "null_model": null_model,
        "raw_ic": raw_ic,
    }
