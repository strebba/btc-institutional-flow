"""Modello di segnale multi-fattore per BTC istituzionale.

Sostituisce le 3 regole binarie del backtest originale con uno scoring
ponderato a 7 fattori (0-100). Ogni fattore è normalizzato 0-1 prima
di essere pesato; il punteggio finale è convertito in etichetta:

  score ≥ 65 → LONG
  score 40-64 → CAUTION
  score < 40  → RISK_OFF

Progettato per essere usato sia dal segnale live (/api/signals) sia
dal backtest storico: accetta sia valori singoli (live) sia pd.Series
(backtest vectorizzato).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config import setup_logging

_log = setup_logging("analytics.signal_model")

# ─── Soglie output ────────────────────────────────────────────────────────────

SIGNAL_LONG     = "LONG"
SIGNAL_CAUTION  = "CAUTION"
SIGNAL_RISK_OFF = "RISK_OFF"

LONG_THRESHOLD     = 65.0
RISK_OFF_THRESHOLD = 40.0

# ─── Pesi dei 7 fattori (devono sommare a 1.0) ────────────────────────────────

WEIGHTS: dict[str, float] = {
    "gex":            0.15,
    "etf_flow":       0.15,
    "funding_rate":   0.20,
    "oi_change":      0.15,
    "long_short":     0.15,
    "put_call":       0.10,
    "liquidations":   0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Pesi non sommano a 1.0"


# ─── Input model ─────────────────────────────────────────────────────────────

@dataclass
class SignalInputs:
    """Tutti gli input necessari per calcolare il segnale composito.

    Tutti i campi sono opzionali: se None il fattore viene escluso dal
    calcolo e i pesi degli altri fattori vengono riscalati proporzionalmente.

    Attributes:
        gex_usd: GEX totale netto in USD (positivo = positive gamma).
        etf_flow_3d_usd: Flussi ETF cumulati 3 giorni in USD.
        funding_rate_annualized_pct: Funding rate 8h annualizzato in %.
        oi_change_7d_pct: Variazione % OI futures aggregato su 7 giorni.
        long_short_ratio: Rapporto account long/short globale (es. 1.5).
        put_call_ratio: Rapporto put OI / call OI da Deribit.
        liquidations_long_24h_usd: Liquidazioni long nelle ultime 24h in USD.
        liquidations_short_24h_usd: Liquidazioni short nelle ultime 24h in USD.
        spot_price: Prezzo spot BTC (usato per proximity check barriere).
        near_active_barrier: True se il prezzo è entro 5% da una barriera.
    """

    gex_usd:                       Optional[float] = None
    etf_flow_3d_usd:               Optional[float] = None
    funding_rate_annualized_pct:   Optional[float] = None
    oi_change_7d_pct:              Optional[float] = None
    long_short_ratio:              Optional[float] = None
    put_call_ratio:                Optional[float] = None
    liquidations_long_24h_usd:     Optional[float] = None
    liquidations_short_24h_usd:    Optional[float] = None
    spot_price:                    Optional[float] = None
    near_active_barrier:           bool = False


# ─── Output model ────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Risultato del calcolo del segnale composito.

    Attributes:
        score: punteggio aggregato 0-100 (65+ = LONG, 40-65 = CAUTION, <40 = RISK_OFF).
        signal: etichetta segnale (LONG | CAUTION | RISK_OFF).
        components: dict con il contributo normalizzato (0-1) di ogni fattore.
        weights_used: pesi effettivi usati (dopo riscalatura per campi None).
        reason: stringa descrittiva human-readable.
    """

    score:        float
    signal:       str
    components:   dict[str, Optional[float]]
    weights_used: dict[str, float]
    reason:       str


# ─── Funzioni di scoring per singolo fattore ─────────────────────────────────

def _score_gex(gex_usd: float) -> float:
    """GEX positivo = market makers coprono vendendo puts → supporto prezzi."""
    if gex_usd > 5e8:   return 1.0   # > +500M: regime fortemente positivo
    if gex_usd > 0:     return 0.65  # positivo ma moderato
    if gex_usd > -5e8:  return 0.35  # leggermente negativo
    return 0.0                        # < -500M: regime fortemente negativo


def _score_etf_flow(flow_3d_usd: float) -> float:
    """Flussi ETF 3gg: proxy domanda istituzionale."""
    # Clamp a ±1B, interpolazione lineare
    v = max(-1e9, min(1e9, flow_3d_usd))
    return (v + 1e9) / 2e9  # 0.0 (−1B) → 1.0 (+1B)


def _score_funding_rate(rate_ann_pct: float) -> float:
    """Funding rate annualizzato: alto = mercato surriscaldato = bearish contrarian.

    Funding negativo = paura/hedging → segnale contrarian rialzista.
    """
    if rate_ann_pct < 0:      return 0.80  # mercato in paura/shorting aggressivo
    if rate_ann_pct < 15:     return 0.70  # neutro bullish
    if rate_ann_pct < 30:     return 0.55  # moderatamente surriscaldato
    if rate_ann_pct < 50:     return 0.35  # surriscaldato
    if rate_ann_pct < 75:     return 0.20  # molto surriscaldato
    return 0.10                             # estremo (>75% ann.) = top signal


def _score_oi_change(oi_change_7d_pct: float) -> float:
    """OI crescente con prezzo in salita = conferma; OI in calo = deleveraging."""
    if oi_change_7d_pct > 15:   return 0.85  # forte aumento OI = momentum
    if oi_change_7d_pct > 5:    return 0.65
    if oi_change_7d_pct > -5:   return 0.50  # stabile
    if oi_change_7d_pct > -15:  return 0.35  # deleveraging moderato
    return 0.15                               # forte deleveraging (>-15%)


def _score_long_short_ratio(ratio: float) -> float:
    """Rapporto long/short: >2 = retail crowded long = contrarian bearish."""
    if ratio < 0.7:    return 0.85  # più short che long = contrarian long
    if ratio < 1.0:    return 0.65
    if ratio < 1.3:    return 0.50  # bilanciato
    if ratio < 1.7:    return 0.38
    if ratio < 2.2:    return 0.25  # retail crowded long
    return 0.12                      # estremamente crowded


def _score_put_call_ratio(pcr: float) -> float:
    """Put/call ratio: alto = hedging/paura = supporto rialzista."""
    if pcr > 1.8:    return 0.85  # forte hedging put = paura = fondo potenziale
    if pcr > 1.3:    return 0.68
    if pcr > 0.9:    return 0.50  # neutro
    if pcr > 0.6:    return 0.32
    return 0.15                    # complacency (poca copertura) = top segnale


def _score_liquidations(long_usd: float, short_usd: float) -> float:
    """Liquidazioni: evento estremo = segnale contrarian direzionale.

    - Molte liquidazioni long (cascata bearish) → near-bottom contrarian
    - Molte liquidazioni short (squeeze rialzista) → near-top contrarian
    - Basse liquidazioni = mercato stabile
    """
    total = long_usd + short_usd
    if total < 1e8:   return 0.50  # mercato tranquillo, neutro

    # Bias direzionale: se prevalgono le long = cascata verso il basso = possibile bottom
    long_ratio = long_usd / total if total > 0 else 0.5

    if total > 1e9:
        # Evento estremo: capitulation se prevalentemente long, squeeze se short
        return 0.72 if long_ratio > 0.6 else 0.28
    if total > 5e8:
        return 0.62 if long_ratio > 0.6 else 0.35
    # Liquidazioni moderate (100M-500M)
    return 0.55 if long_ratio > 0.55 else 0.42


# ─── SignalModel ──────────────────────────────────────────────────────────────

class SignalModel:
    """Calcola il segnale composito multi-fattore per BTC.

    Supporta due modalità:
    - Singolo punto (live): accetta SignalInputs, restituisce SignalResult.
    - Backtest (vectorized): accetta pd.DataFrame con colonne nominate,
      restituisce pd.Series di score float.

    Usage:
        model = SignalModel()
        result = model.compute(inputs)
        print(result.score, result.signal, result.components)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or WEIGHTS.copy()

    # ─── Live (single point) ──────────────────────────────────────────────────

    def compute(self, inputs: SignalInputs) -> SignalResult:
        """Calcola il segnale composito da un singolo set di input.

        Args:
            inputs: SignalInputs con i valori dei 7 fattori.

        Returns:
            SignalResult con score, signal, components, weights_used, reason.
        """
        raw: dict[str, Optional[float]] = {}

        # Calcola raw score 0-1 per ogni fattore disponibile
        if inputs.gex_usd is not None:
            raw["gex"] = _score_gex(inputs.gex_usd)

        if inputs.etf_flow_3d_usd is not None:
            raw["etf_flow"] = _score_etf_flow(inputs.etf_flow_3d_usd)

        if inputs.funding_rate_annualized_pct is not None:
            raw["funding_rate"] = _score_funding_rate(inputs.funding_rate_annualized_pct)

        if inputs.oi_change_7d_pct is not None:
            raw["oi_change"] = _score_oi_change(inputs.oi_change_7d_pct)

        if inputs.long_short_ratio is not None:
            raw["long_short"] = _score_long_short_ratio(inputs.long_short_ratio)

        if inputs.put_call_ratio is not None:
            raw["put_call"] = _score_put_call_ratio(inputs.put_call_ratio)

        if (inputs.liquidations_long_24h_usd is not None
                and inputs.liquidations_short_24h_usd is not None):
            raw["liquidations"] = _score_liquidations(
                inputs.liquidations_long_24h_usd,
                inputs.liquidations_short_24h_usd,
            )

        # Riscala pesi per i fattori disponibili
        available_weight = sum(self._weights[k] for k in raw)
        if available_weight <= 0:
            _log.warning("Nessun fattore disponibile per il segnale")
            return SignalResult(
                score=50.0, signal=SIGNAL_CAUTION,
                components={k: None for k in WEIGHTS},
                weights_used={}, reason="Dati insufficienti",
            )

        scaled_weights = {
            k: self._weights[k] / available_weight
            for k in raw
        }

        # Score aggregato (0-100)
        score_01 = sum(raw[k] * scaled_weights[k] for k in raw)
        score = round(score_01 * 100.0, 1)

        # Override: barriera attiva → abbassa score se siamo vicini al LONG threshold
        if inputs.near_active_barrier and score >= LONG_THRESHOLD:
            score = LONG_THRESHOLD - 1.0

        # Etichetta
        signal = _score_to_signal(score)

        # Reason string
        reason = _build_reason(inputs, raw, score)

        # Componenti full (inclusi quelli None)
        components = {k: raw.get(k) for k in WEIGHTS}

        return SignalResult(
            score=score,
            signal=signal,
            components=components,
            weights_used=scaled_weights,
            reason=reason,
        )

    # ─── Backtest (vectorized) ────────────────────────────────────────────────

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        """Calcola lo score per ogni riga di un DataFrame storico.

        Colonne attese nel DataFrame (tutte opzionali):
          - total_net_gex / _gex: GEX in USD
          - ibit_flow_3d: flussi ETF 3gg in USD
          - funding_rate: funding rate annualizzato in %
          - oi_change_7d_pct: variazione OI 7gg in %
          - long_short_ratio: rapporto long/short
          - put_call_ratio: put/call ratio Deribit
          - liquidations_long_24h / liquidations_short_24h: liquidazioni USD

        Returns:
            pd.Series di score float (0-100) con stesso DatetimeIndex di df.
        """
        scores = pd.Series(50.0, index=df.index, dtype=float)

        def _col(name: str, *aliases: str) -> Optional[pd.Series]:
            for n in (name, *aliases):
                if n in df.columns:
                    return df[n]
            return None

        gex_s    = _col("total_net_gex", "_gex", "total_gex")
        flow_s   = _col("ibit_flow_3d")
        fund_s   = _col("funding_rate")
        oi_s     = _col("oi_change_7d_pct")
        ls_s     = _col("long_short_ratio")
        pcr_s    = _col("put_call_ratio")
        liq_l_s  = _col("liquidations_long_24h")
        liq_s_s  = _col("liquidations_short_24h")

        for i in range(len(df)):
            inputs = SignalInputs(
                gex_usd                     = float(gex_s.iloc[i])   if gex_s  is not None and not _isnan(gex_s.iloc[i])   else None,
                etf_flow_3d_usd             = float(flow_s.iloc[i])  if flow_s is not None and not _isnan(flow_s.iloc[i])  else None,
                funding_rate_annualized_pct = float(fund_s.iloc[i])  if fund_s is not None and not _isnan(fund_s.iloc[i])  else None,
                oi_change_7d_pct            = float(oi_s.iloc[i])    if oi_s   is not None and not _isnan(oi_s.iloc[i])    else None,
                long_short_ratio            = float(ls_s.iloc[i])    if ls_s   is not None and not _isnan(ls_s.iloc[i])    else None,
                put_call_ratio              = float(pcr_s.iloc[i])   if pcr_s  is not None and not _isnan(pcr_s.iloc[i])   else None,
                liquidations_long_24h_usd   = float(liq_l_s.iloc[i]) if liq_l_s is not None and not _isnan(liq_l_s.iloc[i]) else None,
                liquidations_short_24h_usd  = float(liq_s_s.iloc[i]) if liq_s_s is not None and not _isnan(liq_s_s.iloc[i]) else None,
            )
            result = self.compute(inputs)
            scores.iloc[i] = result.score

        return scores

    def score_to_signal(self, score: float) -> str:
        """Converte uno score numerico in etichetta segnale."""
        return _score_to_signal(score)

    def signals_from_scores(self, scores: pd.Series) -> pd.Series:
        """Converte una Serie di score in segnali +1/0/-1 per il backtest.

        +1 = LONG (score ≥ LONG_THRESHOLD)
         0 = FLAT/CAUTION
        -1 = RISK_OFF (score < RISK_OFF_THRESHOLD)
        """
        result = pd.Series(0.0, index=scores.index)
        result[scores >= LONG_THRESHOLD]     =  1.0
        result[scores < RISK_OFF_THRESHOLD]  = -1.0
        return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _score_to_signal(score: float) -> str:
    if score >= LONG_THRESHOLD:
        return SIGNAL_LONG
    if score < RISK_OFF_THRESHOLD:
        return SIGNAL_RISK_OFF
    return SIGNAL_CAUTION


def _build_reason(inputs: SignalInputs, raw: dict[str, float], score: float) -> str:
    """Costruisce una stringa leggibile che spiega il segnale."""
    parts = []

    gex = inputs.gex_usd
    if gex is not None:
        regime = "positive" if gex > 0 else "negative"
        parts.append(f"GEX {regime} ({gex/1e6:+.0f}M)")

    flow = inputs.etf_flow_3d_usd
    if flow is not None:
        parts.append(f"ETF flow 3d {flow/1e6:+.0f}M")

    fr = inputs.funding_rate_annualized_pct
    if fr is not None:
        label = "surriscaldato" if fr > 50 else ("alto" if fr > 25 else ("neutro" if fr >= 0 else "negativo"))
        parts.append(f"funding {fr:.1f}% ann ({label})")

    oi = inputs.oi_change_7d_pct
    if oi is not None:
        trend = "↑" if oi > 0 else "↓"
        parts.append(f"OI 7d {oi:+.1f}% {trend}")

    ls = inputs.long_short_ratio
    if ls is not None:
        sentiment = "crowded long" if ls > 1.8 else ("bilanciato" if ls < 1.3 else "long bias")
        parts.append(f"L/S {ls:.2f} ({sentiment})")

    pcr = inputs.put_call_ratio
    if pcr is not None:
        hedge = "alto hedging" if pcr > 1.3 else ("neutro" if pcr > 0.8 else "complacency")
        parts.append(f"PCR {pcr:.2f} ({hedge})")

    if inputs.near_active_barrier:
        parts.append("⚠ vicino a barriera attiva")

    signal = _score_to_signal(score)
    return f"{signal} [{score:.0f}/100] — " + " | ".join(parts) if parts else f"{signal} [{score:.0f}/100]"
