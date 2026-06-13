"""Scaffolding per la validazione della confluenza barriere↔GEX (Step 2).

La confluenza resta un **overlay informativo** (vedi /api/barriers + dashboard): NON
è wired nel segnale live e NON pesa sullo score. Per promuoverla a sotto-score del
pilastro Barrier (Step 3) serve prima dimostrare che ha potere predittivo — il che
richiede **storia GEX sufficiente** (wall storici allineati al prezzo).

Questo modulo fornisce:
- ``confluence_backtest_ready``: gate sui dati (quanti giorni di wall servono);
- ``confluence_boost_series``: la serie giornaliera di boost direzionale di confluenza
  (riusa detect_clusters/compute_confluence/barrier_confluence_scores);
- ``run_confluence_backtest``: probe predittiva minima (information coefficient sul
  rendimento del giorno dopo) che si **attiva solo a dati sufficienti**, altrimenti
  ritorna un report ``ready=False`` esplicito.

Finché ``cron_gex.py`` non accumula storia, ``run_confluence_backtest`` ritorna
sempre ``ready=False``: è il comportamento corretto, non un errore.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.edgar.barrier_utils import (
    barrier_confluence_scores,
    compute_confluence,
    detect_clusters,
)

_log = logging.getLogger(__name__)

# Giorni minimi di storia wall GEX per un backtest minimamente significativo.
# Sotto questa soglia un IC è rumore: il gate blocca la validazione.
MIN_CONFLUENCE_HISTORY_DAYS = 60


def confluence_backtest_ready(walls_df: Optional[pd.DataFrame]) -> bool:
    """True se la storia dei wall GEX basta per una validazione minimamente sensata.

    Args:
        walls_df: output di ``GexDB.get_walls_series()`` (cols put_wall/call_wall/
            gamma_flip_price). Servono almeno ``MIN_CONFLUENCE_HISTORY_DAYS`` righe
            con almeno un wall non-null.
    """
    if walls_df is None or walls_df.empty:
        return False
    usable = walls_df.dropna(how="all")
    return len(usable) >= MIN_CONFLUENCE_HISTORY_DAYS


def confluence_boost_series(
    btc_close: pd.Series,
    active_barriers: Optional[list[dict]],
    walls_df: Optional[pd.DataFrame],
) -> pd.Series:
    """Serie giornaliera di boost direzionale di confluenza, in [-1, 1].

    Per ogni data con wall GEX disponibili: clusterizza le barriere correnti allo
    spot del giorno, calcola la confluenza coi wall di quel giorno e riduce a uno
    scalare ``bullish - bearish`` (>0 = confluenza rialzista, <0 = ribassista).
    Date senza wall → 0.0 (neutro).

    Limitazione nota (come `CompositeSignal._barrier_series`): usa le barriere
    *correnti* sul prezzo storico — lo storico delle barriere EDGAR non è disponibile.

    Args:
        btc_close: serie prezzo BTC (DatetimeIndex).
        active_barriers: barriere attive dal DB.
        walls_df: storia wall GEX da ``GexDB.get_walls_series()``.

    Returns:
        pd.Series allineata a ``btc_close.index``, valori in [-1, 1].
    """
    boost = pd.Series(0.0, index=btc_close.index)
    if not active_barriers or walls_df is None or walls_df.empty:
        return boost

    walls = walls_df.reindex(btc_close.index)
    for d, price in btc_close.items():
        if pd.isna(price) or price <= 0 or d not in walls.index:
            continue
        row = walls.loc[d]
        if isinstance(row, pd.DataFrame):  # indice duplicato → prendi la prima
            row = row.iloc[0]
        pw, cw, gf = row.get("put_wall"), row.get("call_wall"), row.get("gamma_flip_price")
        if pd.isna(pw) and pd.isna(cw) and pd.isna(gf):
            continue
        clusters = detect_clusters(active_barriers, float(price))
        conf = compute_confluence(
            clusters,
            put_wall=None if pd.isna(pw) else float(pw),
            call_wall=None if pd.isna(cw) else float(cw),
            gamma_flip=None if pd.isna(gf) else float(gf),
        )
        bear, bull = barrier_confluence_scores(conf)
        boost.loc[d] = bull - bear
    return boost


def run_confluence_backtest(
    btc_close: pd.Series,
    active_barriers: Optional[list[dict]],
    walls_df: pd.DataFrame,
) -> dict:
    """Probe predittiva minima: information coefficient della confluenza.

    Gated: se la storia wall è insufficiente ritorna ``{"ready": False, ...}`` SENZA
    calcolare nulla (un IC su pochi giorni è rumore). Quando i dati bastano, misura
    la correlazione tra il boost di confluenza del giorno t e il rendimento BTC del
    giorno t+1 — primo segnale grezzo di potere predittivo, non una strategia.

    Returns:
        dict con almeno ``ready`` (bool). Se ready: ``n_history``, ``n_confluence_days``,
        ``boost_mean``, ``ic_next_day`` (float|None). Altrimenti: ``n_history``,
        ``required``, ``reason``.
    """
    n_hist = 0 if walls_df is None else len(walls_df.dropna(how="all"))
    if not confluence_backtest_ready(walls_df):
        return {
            "ready": False,
            "n_history": n_hist,
            "required": MIN_CONFLUENCE_HISTORY_DAYS,
            "reason": (
                f"Storia wall GEX insufficiente ({n_hist}/{MIN_CONFLUENCE_HISTORY_DAYS} "
                "giorni). La confluenza resta overlay; accumula storia con cron_gex.py."
            ),
        }

    boost = confluence_boost_series(btc_close, active_barriers, walls_df)
    next_ret = btc_close.pct_change().shift(-1)
    nz = boost != 0.0
    n_conf = int(nz.sum())

    ic: Optional[float] = None
    if n_conf >= 2:
        paired = pd.concat([boost[nz], next_ret[nz]], axis=1).dropna()
        if len(paired) >= 2 and paired.iloc[:, 0].std() > 0 and paired.iloc[:, 1].std() > 0:
            ic = float(np.corrcoef(paired.iloc[:, 0], paired.iloc[:, 1])[0, 1])

    return {
        "ready": True,
        "n_history": n_hist,
        "n_confluence_days": n_conf,
        "boost_mean": float(boost.mean()),
        "ic_next_day": ic,
    }
