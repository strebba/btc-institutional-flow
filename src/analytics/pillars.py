"""Architettura a 4 PILASTRI per il segnale BTC istituzionale.

Consolida i tre motori storici (regole binarie dashboard, SignalModel a 7 fattori,
IFI a 6 fattori) in un'unica struttura leggibile composta da quattro pilastri,
ciascuno con un sotto-score 0-100, da cui si compone un punteggio finale 0-100:

  ┌─────────────┬──────────────────────────────┬────────────────────────────────┐
  │ Pilastro    │ Fonte dati                   │ Significato                    │
  ├─────────────┼──────────────────────────────┼────────────────────────────────┤
  │ GEX         │ Deribit options (snapshot)   │ Dealer gamma: stabilizza/amplif│
  │ BARRIER     │ EDGAR note strutturate       │ Livelli di hedging meccanico   │
  │ ETF FLOWS   │ Farside/CoinGlass flows       │ Domanda spot istituzionale     │
  │ MACRO       │ CoinGlass derivati           │ Posizionamento/sentiment       │
  └─────────────┴──────────────────────────────┴────────────────────────────────┘

Il modulo NON riscrive la logica di scoring: RIUSA le funzioni testate di
`signal_model` (regime GEX, flow 3d, funding, OI, long/short, put/call, liquidazioni)
e di `ifi` (flow momentum/trend, price momentum, sigmoid). IFI e SignalModel restano
come librerie di scoring; questo modulo è l'unico orchestratore top-level.

Due modalità, come SignalModel/IFI:
  - `CompositeSignal.compute(inputs)`      → live (snapshot, ricco): CompositeResult
  - `CompositeSignal.compute_series(df)`   → backtest/storico (vettoriale): DataFrame
                                             con colonne *_score chartabili (preserva
                                             la capacità serie-storica di IFI).

Soglie output (riusate da signal_model):
  score ≥ 65 → LONG | 40-64 → CAUTION | < 40 → RISK_OFF
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config import setup_logging
from src.analytics import ifi as _ifi
from src.analytics import signal_model as _sm

_log = setup_logging("analytics.pillars")

# ─── Soglie segnale (riusate da signal_model) ─────────────────────────────────

SIGNAL_LONG     = _sm.SIGNAL_LONG
SIGNAL_CAUTION  = _sm.SIGNAL_CAUTION
SIGNAL_RISK_OFF = _sm.SIGNAL_RISK_OFF
LONG_THRESHOLD     = _sm.LONG_THRESHOLD
RISK_OFF_THRESHOLD = _sm.RISK_OFF_THRESHOLD

# ─── Pesi dei 4 pilastri (devono sommare a 1.0) ───────────────────────────────

PILLAR_WEIGHTS: dict[str, float] = {
    "gex":       0.25,
    "barrier":   0.25,
    "etf_flows": 0.30,
    "macro":     0.20,
}
assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9, "Pesi pilastri non sommano a 1.0"

# ─── Pesi interni di ciascun pilastro (rescalati sui fattori disponibili) ──────

GEX_FACTOR_WEIGHTS: dict[str, float] = {"regime": 0.70, "flip": 0.30}
ETF_FACTOR_WEIGHTS: dict[str, float] = {
    "flow_momentum": 0.40, "flow_trend": 0.25, "price_momentum": 0.15, "flow_3d": 0.20,
}
MACRO_FACTOR_WEIGHTS: dict[str, float] = {
    "funding": 0.30, "oi_change": 0.20, "long_short": 0.20,
    "put_call": 0.15, "liquidations": 0.15,
}

# Parametri Barrier pillar
_BARRIER_SIGMA = 0.10        # ampiezza kernel di prossimità (±10% → peso ~0.37)
_DIR_ACCELERANT = 0.15       # knock_in/buffer sotto spot → accelerante ribasso
_DIR_RESISTANCE = 0.40       # autocall/knock_out sopra spot → resistenza/soft-cap
_DIR_NEUTRAL    = 0.50       # barriera dal lato neutro


# ─── Output ───────────────────────────────────────────────────────────────────

@dataclass
class PillarScore:
    """Sotto-score 0-100 di un singolo pilastro.

    Attributes:
        name: identificatore pilastro (gex|barrier|etf_flows|macro).
        score: punteggio 0-100, oppure None se nessun dato disponibile.
        weight: peso effettivo nel composito (dopo rescaling), 0-1.
        components: contributo grezzo (0-1) dei fattori interni.
        reason: stringa human-readable.
    """

    name:       str
    score:      Optional[float]
    weight:     float = 0.0
    components: dict[str, Optional[float]] = field(default_factory=dict)
    reason:     str = ""


@dataclass
class CompositeResult:
    """Risultato del segnale composito a 4 pilastri."""

    score:             float
    signal:            str
    pillars:           list[PillarScore]
    reason:            str
    weights_used:      dict[str, float]
    legacy_components: dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def components(self) -> dict[str, Optional[float]]:
        """Alias dei 7 fattori storici (duck-typing con SignalResult per SignalDB)."""
        return self.legacy_components


@dataclass
class CompositeInputs:
    """Input live per il segnale composito.

    Tutti opzionali: i pilastri/fattori senza dati vengono esclusi e i pesi
    riscalati proporzionalmente.
    """

    # GEX
    gex_usd:                       Optional[float] = None
    gamma_flip_price:              Optional[float] = None
    put_wall:                      Optional[float] = None
    call_wall:                     Optional[float] = None
    # Barrier
    active_barriers:               Optional[list[dict]] = None
    # ETF flows
    etf_flow_3d_usd:               Optional[float] = None
    flow_history_df:               Optional[pd.DataFrame] = None
    flow_is_estimate:              bool = False
    # Macro
    funding_rate_annualized_pct:   Optional[float] = None
    oi_change_7d_pct:              Optional[float] = None
    long_short_ratio:              Optional[float] = None
    put_call_ratio:                Optional[float] = None
    liquidations_long_24h_usd:     Optional[float] = None
    liquidations_short_24h_usd:    Optional[float] = None
    # Comune
    spot_price:                    Optional[float] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))


def _blend(parts: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    """Media pesata 0-1 dei fattori disponibili, con rescaling sui pesi presenti."""
    avail = {k: v for k, v in parts.items() if v is not None}
    if not avail:
        return None
    wsum = sum(weights.get(k, 0.0) for k in avail)
    if wsum <= 0:
        return None
    return sum(avail[k] * weights.get(k, 0.0) for k in avail) / wsum


def _to_score(v: Optional[float]) -> Optional[float]:
    return round(v * 100.0, 1) if v is not None else None


def score_to_signal(score: float) -> str:
    if score >= LONG_THRESHOLD:
        return SIGNAL_LONG
    if score < RISK_OFF_THRESHOLD:
        return SIGNAL_RISK_OFF
    return SIGNAL_CAUTION


# ─── Pillar 1: GEX ────────────────────────────────────────────────────────────

def score_gex_pillar(
    *,
    gex_usd: Optional[float],
    gamma_flip_price: Optional[float] = None,
    spot_price: Optional[float] = None,
) -> PillarScore:
    """GEX pillar: regime dealer gamma + contesto gamma-flip.

    - regime: riusa signal_model._score_gex (positivo = dealer stabilizzano).
    - flip:   spot sopra il gamma flip = zona di gamma positiva (supportiva).
    """
    factors: dict[str, Optional[float]] = {"regime": None, "flip": None}

    if gex_usd is not None:
        factors["regime"] = _sm._score_gex(gex_usd)

    if gamma_flip_price and spot_price and spot_price > 0:
        # Sopra il flip → sigmoid > 0.5 (supportivo); sotto → < 0.5 (destabilizzante)
        factors["flip"] = _sigmoid_scalar((spot_price - gamma_flip_price) / (0.05 * spot_price))

    blended = _blend(factors, GEX_FACTOR_WEIGHTS)
    score = _to_score(blended)

    reason = ""
    if gex_usd is not None:
        regime = "positivo" if gex_usd > 0 else "negativo"
        reason = f"GEX {gex_usd/1e6:+.0f}M ({regime})"
        if factors["flip"] is not None:
            reason += ", spot " + ("sopra" if factors["flip"] > 0.5 else "sotto") + " gamma flip"

    return PillarScore("gex", score, components=factors, reason=reason)


# ─── Pillar 2: BARRIER (direzionale, notional-weighted) ───────────────────────

def _barrier_direction(barrier_type: str, below_spot: bool) -> float:
    """Score direzionale 0-1 di una barriera (alto = supportivo, basso = ribassista)."""
    bt = (barrier_type or "").lower()
    if below_spot and bt in ("knock_in", "buffer", "knock_out"):
        return _DIR_ACCELERANT   # rottura al ribasso → dealer vendono → accelera il calo
    if (not below_spot) and bt in ("autocall", "knock_out", "call"):
        return _DIR_RESISTANCE   # sopra spot → soft-cap / resistenza al rialzo
    return _DIR_NEUTRAL


def score_barrier_pillar(
    *,
    active_barriers: Optional[list[dict]],
    spot_price: Optional[float],
) -> PillarScore:
    """Barrier pillar: media direzionale pesata per notional e prossimità.

    Per ogni barriera attiva con prezzo BTC e spot validi:
      d    = (level - spot)/spot
      prox = exp(-(d/σ)²)                       (σ=10%; barriere lontane ~0)
      dir  = direzionalità per tipo/lato        (knock_in sotto=ribasso, autocall sopra=resist.)
      w    = notional / Σnotional               (equal-weight se notional assente)
      score = 100 · Σ w·prox·dir / Σ w·prox
    Nessuna barriera nel kernel → 50 (neutro).
    """
    components: dict[str, Optional[float]] = {
        "notional_weighted_distance": None,
        "n_active": 0,
        "n_in_kernel": 0,
    }
    if not active_barriers or not spot_price or spot_price <= 0:
        return PillarScore("barrier", None, components=components,
                           reason="Nessuna barriera attiva / spot non disponibile")

    rows = []
    total_notional = 0.0
    for b in active_barriers:
        lvl = b.get("level_price_btc") or 0.0
        if lvl <= 0:
            continue
        notional = b.get("notional_usd") or 0.0
        total_notional += max(0.0, notional)
        rows.append((b, lvl, notional))

    components["n_active"] = len(rows)
    if not rows:
        return PillarScore("barrier", 50.0, components=components,
                           reason="Barriere senza prezzo BTC calcolabile")

    num = den = 0.0
    nearest = None
    nearest_absd = float("inf")
    dir_counts = {"accelerante_ribasso": 0, "resistenza": 0, "neutro": 0}
    n_in_kernel = 0
    nw_dist_num = nw_dist_den = 0.0

    for b, lvl, notional in rows:
        d = (lvl - spot_price) / spot_price
        prox = math.exp(-((d / _BARRIER_SIGMA) ** 2))
        below = d < 0
        dir_score = _barrier_direction(b.get("barrier_type", ""), below)
        # peso notional: equal-weight se nessun notional disponibile
        w = (notional / total_notional) if total_notional > 0 else (1.0 / len(rows))
        num += w * prox * dir_score
        den += w * prox
        nw_dist_num += w * abs(d)
        nw_dist_den += w
        if prox > 0.05:
            n_in_kernel += 1
            if dir_score == _DIR_ACCELERANT:
                dir_counts["accelerante_ribasso"] += 1
            elif dir_score == _DIR_RESISTANCE:
                dir_counts["resistenza"] += 1
            else:
                dir_counts["neutro"] += 1
        if abs(d) < nearest_absd:
            nearest_absd = abs(d)
            nearest = {
                "barrier_type": b.get("barrier_type"),
                "level_price_btc": round(lvl, 2),
                "distance_pct": round(d * 100, 2),
                "issuer": b.get("issuer"),
                "notional_usd": notional or None,
            }

    score_01 = (num / den) if den > 0 else 0.5
    components["nearest_barrier"] = nearest
    components["n_in_kernel"] = n_in_kernel
    components["notional_weighted_distance"] = (
        round(nw_dist_num / nw_dist_den, 4) if nw_dist_den > 0 else None
    )
    dominant = max(dir_counts, key=dir_counts.get) if n_in_kernel else "nessuna_vicina"
    components["dominant_direction"] = dominant

    if n_in_kernel == 0:
        reason = "Nessuna barriera entro ~25% dallo spot (neutro)"
        score_01 = 0.5
    else:
        nb = nearest or {}
        reason = (
            f"{nb.get('barrier_type')} ${nb.get('level_price_btc')} "
            f"a {nb.get('distance_pct')}% → {dominant.replace('_', ' ')}"
        )

    return PillarScore("barrier", _to_score(score_01), components=components, reason=reason)


# ─── Pillar 3: ETF FLOWS ──────────────────────────────────────────────────────

def score_etf_flows_pillar(
    *,
    etf_flow_3d_usd: Optional[float] = None,
    history_df: Optional[pd.DataFrame] = None,
    is_estimate: bool = False,
) -> PillarScore:
    """ETF flows pillar: domanda spot istituzionale.

    Riusa i fattori IFI (flow_momentum, flow_trend, price_momentum) quando è
    disponibile la serie storica, più il livello flow 3d (signal_model._score_etf_flow).
    Senza storico, ricade sul solo flow 3d.
    """
    factors: dict[str, Optional[float]] = {
        "flow_momentum": None, "flow_trend": None, "price_momentum": None, "flow_3d": None,
    }

    if etf_flow_3d_usd is not None:
        factors["flow_3d"] = _sm._score_etf_flow(etf_flow_3d_usd)

    if history_df is not None and not history_df.empty:
        flow = _ifi._col(history_df, "total_flow_usd", "total_flow")
        if flow is not None and flow.notna().sum() >= 30:
            flow = flow.fillna(0.0)
            factors["flow_momentum"] = float(_ifi._score_flow_momentum(flow).iloc[-1])
            factors["flow_trend"]    = float(_ifi._score_flow_trend(flow).iloc[-1])
        btc_close = _ifi._col(history_df, "btc_close")
        btc_vol   = _ifi._col(history_df, "btc_vol_7d")
        if btc_close is not None and btc_vol is not None:
            factors["price_momentum"] = float(
                _ifi._score_price_momentum(btc_close, btc_vol).iloc[-1]
            )

    blended = _blend(factors, ETF_FACTOR_WEIGHTS)
    # Stima yfinance a bassa qualità → comprimi verso il neutro (riduci confidenza)
    if blended is not None and is_estimate:
        blended = 0.5 + (blended - 0.5) * 0.5

    reason = ""
    if etf_flow_3d_usd is not None:
        reason = f"ETF flow 3d {etf_flow_3d_usd/1e6:+.0f}M"
        if is_estimate:
            reason += " (stima yfinance, bassa qualità)"

    return PillarScore("etf_flows", _to_score(blended), components=factors, reason=reason)


# ─── Pillar 4: MACRO ──────────────────────────────────────────────────────────

def score_macro_pillar(
    *,
    funding_rate_annualized_pct: Optional[float] = None,
    oi_change_7d_pct: Optional[float] = None,
    long_short_ratio: Optional[float] = None,
    put_call_ratio: Optional[float] = None,
    liquidations_long_24h_usd: Optional[float] = None,
    liquidations_short_24h_usd: Optional[float] = None,
) -> PillarScore:
    """Macro pillar: posizionamento/sentiment dai derivati CoinGlass.

    Riusa le funzioni contrarian di signal_model: funding, OI change, long/short,
    put/call, liquidazioni.
    """
    factors: dict[str, Optional[float]] = {
        "funding": None, "oi_change": None, "long_short": None,
        "put_call": None, "liquidations": None,
    }

    if funding_rate_annualized_pct is not None:
        factors["funding"] = _sm._score_funding_rate(funding_rate_annualized_pct)
    if oi_change_7d_pct is not None:
        factors["oi_change"] = _sm._score_oi_change(oi_change_7d_pct)
    if long_short_ratio is not None:
        factors["long_short"] = _sm._score_long_short_ratio(long_short_ratio)
    if put_call_ratio is not None:
        factors["put_call"] = _sm._score_put_call_ratio(put_call_ratio)
    if liquidations_long_24h_usd is not None and liquidations_short_24h_usd is not None:
        factors["liquidations"] = _sm._score_liquidations(
            liquidations_long_24h_usd, liquidations_short_24h_usd
        )

    blended = _blend(factors, MACRO_FACTOR_WEIGHTS)

    reason_parts = []
    if funding_rate_annualized_pct is not None:
        reason_parts.append(f"funding {funding_rate_annualized_pct:.0f}% ann")
    if long_short_ratio is not None:
        reason_parts.append(f"L/S {long_short_ratio:.2f}")
    reason = " | ".join(reason_parts)

    return PillarScore("macro", _to_score(blended), components=factors, reason=reason)


# ─── Orchestratore ────────────────────────────────────────────────────────────

class CompositeSignal:
    """Compone i 4 pilastri in un segnale finale 0-100 + label.

    Usage:
        cs = CompositeSignal()
        result = cs.compute(inputs)        # live → CompositeResult
        df = cs.compute_series(merged_df)  # backtest → DataFrame *_score
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or PILLAR_WEIGHTS.copy()

    # ─── Live ─────────────────────────────────────────────────────────────────

    def compute(self, inputs: CompositeInputs) -> CompositeResult:
        pillars = [
            score_gex_pillar(
                gex_usd=inputs.gex_usd,
                gamma_flip_price=inputs.gamma_flip_price,
                spot_price=inputs.spot_price,
            ),
            score_barrier_pillar(
                active_barriers=inputs.active_barriers,
                spot_price=inputs.spot_price,
            ),
            score_etf_flows_pillar(
                etf_flow_3d_usd=inputs.etf_flow_3d_usd,
                history_df=inputs.flow_history_df,
                is_estimate=inputs.flow_is_estimate,
            ),
            score_macro_pillar(
                funding_rate_annualized_pct=inputs.funding_rate_annualized_pct,
                oi_change_7d_pct=inputs.oi_change_7d_pct,
                long_short_ratio=inputs.long_short_ratio,
                put_call_ratio=inputs.put_call_ratio,
                liquidations_long_24h_usd=inputs.liquidations_long_24h_usd,
                liquidations_short_24h_usd=inputs.liquidations_short_24h_usd,
            ),
        ]

        # Blend pesato sui pilastri disponibili (rescaling)
        avail = {p.name: p.score for p in pillars if p.score is not None}
        if not avail:
            _log.warning("Nessun pilastro disponibile per il segnale composito")
            return CompositeResult(
                score=50.0, signal=SIGNAL_CAUTION, pillars=pillars,
                reason="Dati insufficienti", weights_used={},
            )

        wsum = sum(self._weights[k] for k in avail)
        weights_used = {k: round(self._weights[k] / wsum, 4) for k in avail}
        for p in pillars:
            p.weight = weights_used.get(p.name, 0.0)

        score = round(sum(avail[k] * weights_used[k] for k in avail), 1)
        signal = score_to_signal(score)

        reason = f"{signal} [{score:.0f}/100] — " + " | ".join(
            f"{p.name}:{p.score:.0f}" for p in pillars if p.score is not None
        )

        return CompositeResult(
            score=score,
            signal=signal,
            pillars=pillars,
            reason=reason,
            weights_used=weights_used,
            legacy_components=self._legacy_components(pillars),
        )

    @staticmethod
    def _legacy_components(pillars: list[PillarScore]) -> dict[str, Optional[float]]:
        """Espone i 7 fattori storici (retro-compat /api/signals) dai pilastri."""
        by_name = {p.name: p.components for p in pillars}
        gex = by_name.get("gex", {})
        etf = by_name.get("etf_flows", {})
        mac = by_name.get("macro", {})
        return {
            "gex":          gex.get("regime"),
            "etf_flow":     etf.get("flow_3d"),
            "funding_rate": mac.get("funding"),
            "oi_change":    mac.get("oi_change"),
            "long_short":   mac.get("long_short"),
            "put_call":     mac.get("put_call"),
            "liquidations": mac.get("liquidations"),
        }

    # ─── Backtest / serie storica ─────────────────────────────────────────────

    def compute_series(
        self,
        df: pd.DataFrame,
        active_barriers: Optional[list[dict]] = None,
        barrier_history: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calcola i sotto-score dei pilastri per ogni giorno + il composito.

        Colonne df attese (tutte opzionali; pesi rescalati per ciò che manca):
          GEX:    total_net_gex | _gex | total_gex
          ETF:    total_flow_usd | total_flow, btc_close, btc_vol_7d
          MACRO:  funding_rate, oi_usd | oi_change_7d_pct, long_short_ratio
        Barrier: usa ``barrier_history`` (storico) se fornito, altrimenti applica
        ``active_barriers`` (correnti) su tutto lo storico prezzi.

        Returns:
            DataFrame con colonne gex_score, barrier_score, etf_flows_score,
            macro_score, composite_score (0-100), stesso indice di df.
        """
        idx = df.index
        out = pd.DataFrame(index=idx)

        # GEX (regime vettorizzato; il flip storico non è disponibile)
        gex_col = _ifi._col(df, "total_net_gex", "_gex", "total_gex")
        if gex_col is not None:
            out["gex_score"] = gex_col.apply(
                lambda v: _sm._score_gex(float(v)) * 100 if pd.notna(v) else np.nan
            )
        else:
            out["gex_score"] = np.nan

        # ETF flows (riusa i fattori IFI vettoriali)
        etf_parts = pd.DataFrame(index=idx)
        flow = _ifi._col(df, "total_flow_usd", "total_flow")
        if flow is not None and flow.notna().sum() >= 30:
            flow = flow.fillna(0.0)
            etf_parts["flow_momentum"] = _ifi._score_flow_momentum(flow)
            etf_parts["flow_trend"]    = _ifi._score_flow_trend(flow)
        btc_close = _ifi._col(df, "btc_close")
        btc_vol   = _ifi._col(df, "btc_vol_7d")
        if btc_close is not None and btc_vol is not None:
            etf_parts["price_momentum"] = _ifi._score_price_momentum(btc_close, btc_vol)
        if "ibit_flow_3d" in df.columns:
            etf_parts["flow_3d"] = df["ibit_flow_3d"].apply(
                lambda v: _sm._score_etf_flow(float(v)) if pd.notna(v) else np.nan
            )
        out["etf_flows_score"] = self._weighted_row(etf_parts, ETF_FACTOR_WEIGHTS) * 100

        # MACRO (riusa i fattori IFI vettoriali dove disponibili)
        macro_parts = pd.DataFrame(index=idx)
        funding = _ifi._col(df, "funding_rate")
        if funding is not None:
            macro_parts["funding"] = _ifi._score_funding(funding)
        oi = _ifi._col(df, "oi_usd")
        if oi is not None:
            macro_parts["oi_change"] = _ifi._score_oi_momentum(oi)
        ls = _ifi._col(df, "long_short_ratio")
        if ls is not None:
            macro_parts["long_short"] = _ifi._score_ls_squeeze(ls)
        out["macro_score"] = self._weighted_row(macro_parts, MACRO_FACTOR_WEIGHTS) * 100

        # BARRIER (barriere correnti o storico su prezzo storico)
        out["barrier_score"] = self._barrier_series(btc_close, active_barriers, barrier_history)

        # Composito: blend pesato per riga con rescaling sui pilastri disponibili
        pillar_scores = out[["gex_score", "barrier_score", "etf_flows_score", "macro_score"]] / 100
        pillar_scores = pillar_scores.rename(columns=lambda c: c.replace("_score", ""))
        wmap = {"gex": self._weights["gex"], "barrier": self._weights["barrier"],
                "etf_flows": self._weights["etf_flows"], "macro": self._weights["macro"]}
        out["composite_score"] = (self._weighted_row(pillar_scores, wmap) * 100).round(2)
        return out

    @staticmethod
    def _weighted_row(parts: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
        """Media pesata per riga (0-1) con rescaling sui valori non-NaN."""
        if parts.empty or parts.shape[1] == 0:
            return pd.Series(0.5, index=parts.index)
        w = pd.Series({c: weights.get(c, 0.0) for c in parts.columns})
        mask = parts.notna()
        wmat = mask.mul(w, axis=1)
        wsum = wmat.sum(axis=1).replace(0.0, np.nan)
        num = (parts.fillna(0.0) * wmat).sum(axis=1)
        return (num / wsum).fillna(0.5)

    def _barrier_series(
        self,
        btc_close: Optional[pd.Series],
        active_barriers: Optional[list[dict]],
        barrier_history: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Serie del barrier score: barriere correnti o storico su prezzo storico.

        Args:
            btc_close: serie prezzi BTC.
            active_barriers: barriere attive correnti (usate se manca storico).
            barrier_history: storico barriere da StructuredNotesDB.get_barrier_history().
                Se fornito, per ogni data storica recupera le barriere attive in quel
                giorno invece di applicare quelle correnti.

        Returns:
            Serie 0-100 con stesso indice di btc_close.
        """
        if btc_close is None:
            return pd.Series(dtype=float)
        if not active_barriers and barrier_history is None:
            return pd.Series(50.0, index=btc_close.index)

        # Se abbiamo lo storico, per ogni data recuperiamo le barriere di quel giorno
        if barrier_history is not None and not barrier_history.empty:
            return self._barrier_series_from_history(btc_close, barrier_history)

        # Fallback: barriere correnti su tutto lo storico
        return btc_close.apply(
            lambda p: (
                score_barrier_pillar(active_barriers=active_barriers, spot_price=float(p)).score
                if pd.notna(p) and p > 0 else np.nan
            )
        )

    def _barrier_series_from_history(
        self,
        btc_close: pd.Series,
        barrier_history: pd.DataFrame,
    ) -> pd.Series:
        """Calcola il barrier score usando lo storico barriere per ogni data."""
        # Raggruppa barriere per snapshot_date
        history_by_date: dict[pd.Timestamp, list[dict]] = {}
        for _, row in barrier_history.iterrows():
            d = row["snapshot_date"]
            history_by_date.setdefault(d, []).append({
                "barrier_type": row["barrier_type"],
                "level_price_btc": row["level_price_btc"],
                "notional_usd": row["notional_usd"],
                "issuer": row["issuer"],
            })

        scores = pd.Series(np.nan, index=btc_close.index, dtype=float)
        for ts, price in btc_close.items():
            if pd.isna(price) or price <= 0:
                continue
            # Cerca la data di snapshot più vicina (entro 1 giorno)
            if ts in history_by_date:
                barriers = history_by_date[ts]
            else:
                # Nearest neighbor: giorno prima o dopo
                candidates = sorted(history_by_date.keys())
                nearest = min(candidates, key=lambda d: abs((d - ts).days), default=None)
                if nearest is None or abs((nearest - ts).days) > 1:
                    scores[ts] = 50.0
                    continue
                barriers = history_by_date[nearest]

            if barriers:
                result = score_barrier_pillar(active_barriers=barriers, spot_price=float(price))
                scores[ts] = result.score if result.score is not None else 50.0
            else:
                scores[ts] = 50.0

        return scores.fillna(50.0)

    def signals_from_scores(self, scores: pd.Series) -> pd.Series:
        """Converte score 0-100 in segnali +1/0/-1 (riusa la convenzione signal_model)."""
        result = pd.Series(0.0, index=scores.index)
        result[scores >= LONG_THRESHOLD]    =  1.0
        result[scores < RISK_OFF_THRESHOLD] = -1.0
        return result
