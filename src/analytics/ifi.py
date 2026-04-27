"""Institutional Flow Index (IFI) — serie temporale giornaliera 0-100.

Differenze chiave rispetto a SignalModel:
- Output: serie storica completa (non snapshot puntuale)
- Scaling: sigmoid continua su tutti i fattori (curva liscia, chartabile D/W)
- Backbone: ETF flows Farside (da gen 2024, ~470gg di storia)
- Fattori CoinGlass (funding/OI/L/S) supplementari: weight-rescaling automatico
  per i giorni dove non sono disponibili (history >333-500gg)

Fattori e pesi nominali (sommano a 1.0):
  flow_momentum  0.40  z-score cumulative flow 7d vs distribuzione 90d
  flow_trend     0.20  rapporto flow_7d / flow_30d (accelerazione)
  price_momentum 0.15  rendimento BTC 30d / volatilità 30d
  funding        0.10  contrarian funding rate annualizzato
  oi_momentum    0.10  variazione OI 30d normalizzata
  ls_squeeze     0.05  L/S ratio inverso (squeeze potential)

Regime labels:
  IFI ≥ 70  → Accumulation
  IFI 55-70 → Momentum
  IFI 45-55 → Neutral
  IFI 30-45 → Distribution
  IFI < 30  → Outflow
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import setup_logging

_log = setup_logging("analytics.ifi")

# ─── Pesi nominali ────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "flow_momentum":  0.40,
    "flow_trend":     0.20,
    "price_momentum": 0.15,
    "funding":        0.10,
    "oi_momentum":    0.10,
    "ls_squeeze":     0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Pesi non sommano a 1.0"

# ─── Regimi ───────────────────────────────────────────────────────────────────

REGIME_ACCUMULATION = "Accumulation"
REGIME_MOMENTUM     = "Momentum"
REGIME_NEUTRAL      = "Neutral"
REGIME_DISTRIBUTION = "Distribution"
REGIME_OUTFLOW      = "Outflow"

ALL_REGIMES = (
    REGIME_ACCUMULATION,
    REGIME_MOMENTUM,
    REGIME_NEUTRAL,
    REGIME_DISTRIBUTION,
    REGIME_OUTFLOW,
)

# ─── Output ──────────────────────────────────────────────────────────────────

@dataclass
class IFIResult:
    score:        float
    regime:       str
    components:   dict[str, Optional[float]]
    weights_used: dict[str, float]


# ─── Sigmoid helper ───────────────────────────────────────────────────────────

def _sigmoid(x: "np.ndarray | pd.Series") -> "np.ndarray | pd.Series":
    """Sigmoid continua: mappa R → (0, 1). Clampata a ±10 per stabilità numerica."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


# ─── Factor scoring (vectorized) ──────────────────────────────────────────────

def _score_flow_momentum(total_flow: pd.Series) -> pd.Series:
    """Z-score del cumulative flow 7d rispetto alla distribuzione rolling 90d.

    Z > 1.5σ → domanda istituzionale anormalmente alta → IFI alto.
    Z < −1.5σ → outflow estremo → IFI basso.
    """
    flow_7d = total_flow.rolling(7, min_periods=3).sum()
    mean_90 = flow_7d.rolling(90, min_periods=30).mean()
    std_90  = flow_7d.rolling(90, min_periods=30).std().replace(0, np.nan)
    z = (flow_7d - mean_90) / std_90
    return pd.Series(_sigmoid(z.fillna(0.0).to_numpy()), index=total_flow.index)


def _score_flow_trend(total_flow: pd.Series) -> pd.Series:
    """Rapporto normalizzato flow_7d / flow_30d — accelerazione vs decelerazione.

    Rapporto > 1 (7d > un quarto del 30d) = accelerazione = bullish.
    Rapporto < 1 = decelerazione = bearish.
    """
    flow_7d  = total_flow.rolling(7,  min_periods=3).sum()
    flow_30d = total_flow.rolling(30, min_periods=10).sum()
    denom = (flow_30d.abs() / 4.0).clip(lower=1e6)  # minimo $1M
    ratio = flow_7d / denom
    return pd.Series(_sigmoid((ratio - 1.0).fillna(0.0).to_numpy()), index=total_flow.index)


def _score_price_momentum(btc_close: pd.Series, btc_vol_7d: pd.Series) -> pd.Series:
    """Rendimento BTC 30d normalizzato per volatilità — Sharpe-like a 30gg.

    Rendimento alto con vol bassa → segnale forte. Normalizzato per volatilità
    annualizzata convertita a orizzonte 30gg: vol_ann * sqrt(30/252).
    """
    ret_30d = btc_close.pct_change(30)
    vol_adj = btc_vol_7d.replace(0, np.nan).fillna(0.40)
    vol_30d = vol_adj * (30 / 252) ** 0.5
    sharpe_like = ret_30d / vol_30d.replace(0, 0.10)
    return pd.Series(_sigmoid(sharpe_like.fillna(0.0).to_numpy()), index=btc_close.index)


def _score_funding(funding_rate_ann: pd.Series) -> pd.Series:
    """Contrarian funding rate — alto annualizzato = surriscaldato = bearish per IFI.

    Centrato a 20% (healthy range). Scala: 40% per unità di z.
    """
    z = (funding_rate_ann - 20.0) / 40.0
    return pd.Series(_sigmoid(-z.fillna(0.0).to_numpy()), index=funding_rate_ann.index)


def _score_oi_momentum(oi_usd: pd.Series) -> pd.Series:
    """Variazione OI aggregato 30d normalizzata — momentum posizionamento futures.

    +15% cambio in 30gg ≈ 1σ tipico → z=1.
    """
    oi_change = oi_usd.pct_change(30)
    z = oi_change / 0.15
    return pd.Series(_sigmoid(z.fillna(0.0).to_numpy()), index=oi_usd.index)


def _score_ls_squeeze(ls_ratio: pd.Series) -> pd.Series:
    """L/S ratio inverso — squeeze potential.

    L/S < 1 (più short che long) → squeeze potenziale alto → bullish contrarian.
    Centrato a 1.0, scala 0.5.
    """
    z = (1.0 - ls_ratio) / 0.5
    return pd.Series(_sigmoid(z.fillna(0.0).to_numpy()), index=ls_ratio.index)


# ─── Regime classifier ────────────────────────────────────────────────────────

def regime_label(score: float) -> str:
    if score >= 70: return REGIME_ACCUMULATION
    if score >= 55: return REGIME_MOMENTUM
    if score >= 45: return REGIME_NEUTRAL
    if score >= 30: return REGIME_DISTRIBUTION
    return REGIME_OUTFLOW


# ─── Column resolution helper ─────────────────────────────────────────────────

def _col(df: pd.DataFrame, *names: str) -> "pd.Series | None":
    for name in names:
        if name in df.columns and df[name].notna().sum() >= 5:
            return df[name]
    return None


# ─── IFIModel ─────────────────────────────────────────────────────────────────

class IFIModel:
    """Calcola l'Institutional Flow Index come serie temporale giornaliera.

    Input DataFrame atteso (da FlowCorrelation.merge() con dati CoinGlass):
      Obbligatori:
        - total_flow / total_flow_usd : flussi ETF totali giornalieri (USD)
        - btc_close                   : prezzo BTC giornaliero
        - btc_vol_7d                  : vol realizzata 7gg annualizzata
      Opzionali (CoinGlass, max 333-500gg di storia):
        - funding_rate                : funding rate annualizzato in %
        - oi_usd                      : open interest totale USD
        - long_short_ratio            : rapporto long/short

    Usage:
        model = IFIModel()
        scores = model.compute_series(df)   # pd.Series 0-100
        result = model.compute_latest(df)   # IFIResult (singolo punto)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or WEIGHTS.copy()

    def _build_factor_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Costruisce il DataFrame dei fattori normalizzati (tutti in [0,1])."""
        factors: dict[str, pd.Series] = {}

        flow = _col(df, "total_flow_usd", "total_flow")
        if flow is not None and flow.notna().sum() >= 30:
            factors["flow_momentum"] = _score_flow_momentum(flow.fillna(0.0))
            factors["flow_trend"]    = _score_flow_trend(flow.fillna(0.0))

        btc_close = _col(df, "btc_close")
        btc_vol   = _col(df, "btc_vol_7d")
        if btc_close is not None and btc_vol is not None:
            factors["price_momentum"] = _score_price_momentum(btc_close, btc_vol)

        funding = _col(df, "funding_rate")
        if funding is not None:
            factors["funding"] = _score_funding(funding)

        oi = _col(df, "oi_usd")
        if oi is not None:
            factors["oi_momentum"] = _score_oi_momentum(oi)

        ls = _col(df, "long_short_ratio")
        if ls is not None:
            factors["ls_squeeze"] = _score_ls_squeeze(ls)

        return pd.DataFrame(factors, index=df.index)

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        """Computa IFI per ogni giorno del DataFrame.

        Gestisce automaticamente i fattori mancanti (NaN) con weight-rescaling
        proporzionale: ogni giorno usa solo i fattori disponibili in quel giorno.

        Returns:
            pd.Series di score float 0-100 con stesso DatetimeIndex di df.
        """
        if df.empty:
            return pd.Series(dtype=float)

        factor_df = self._build_factor_df(df)

        if factor_df.empty:
            _log.warning("IFI: nessun fattore disponibile — serie a 50.0")
            return pd.Series(50.0, index=df.index)

        # Matrice dei pesi nominali (shape: n_rows × n_factors)
        weight_row = {k: self._weights.get(k, 0.0) for k in factor_df.columns}
        weight_df  = pd.DataFrame(
            [weight_row] * len(factor_df),
            index=factor_df.index,
            columns=factor_df.columns,
        )

        # Azzera i pesi dove il fattore è NaN → rescaling implicito nella normalizzazione
        masked_w   = weight_df.where(factor_df.notna(), 0.0)
        weight_sum = masked_w.sum(axis=1).replace(0.0, np.nan)

        score_01 = (factor_df.fillna(0.0) * masked_w).sum(axis=1) / weight_sum
        scores   = (score_01.fillna(0.5) * 100.0).round(2)

        _log.info(
            "IFI compute_series: %d righe | media=%.1f std=%.1f range=[%.1f,%.1f]",
            len(scores), scores.mean(), scores.std(), scores.min(), scores.max(),
        )
        return scores

    def compute_latest(self, df: pd.DataFrame) -> IFIResult:
        """Computa IFI per l'ultimo giorno disponibile nel DataFrame."""
        scores = self.compute_series(df)
        if scores.empty:
            return IFIResult(50.0, REGIME_NEUTRAL, {}, {})

        score  = float(scores.iloc[-1])
        regime = regime_label(score)

        factor_df = self._build_factor_df(df)
        components: dict[str, Optional[float]] = {}
        for k in self._weights:
            if k in factor_df.columns:
                v = factor_df[k].iloc[-1]
                components[k] = round(float(v), 4) if pd.notna(v) else None
            else:
                components[k] = None

        available_w = sum(
            self._weights[k] for k, v in components.items() if v is not None
        )
        weights_used = (
            {k: round(self._weights[k] / available_w, 4)
             for k, v in components.items() if v is not None}
            if available_w > 0 else {}
        )

        return IFIResult(
            score=score,
            regime=regime,
            components=components,
            weights_used=weights_used,
        )
