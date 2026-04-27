"""WAGMI Index — indicatore composito 0-100 per BTC istituzionale.

Differenze chiave rispetto a SignalModel:
  - `gamma_structure`: integra posizione rispetto al gamma flip (non solo sign del GEX)
  - `squeeze_setup`: combina L/S ratio e funding rate — la divergenza è informativa
  - `options_structure`: distanza da call wall, put wall e max pain come input strutturale
  - Pesi spostati verso fattori momentum/strutturali, meno contrarian puro

Labels:
  score ≥ 80  → FULL WAGMI   (bull forte)
  score 65-79 → WAGMI        (bullish confermato)
  score 51-64 → MEH          (neutro)
  score 31-50 → DEGEN        (rischioso)
  score ≤ 30  → NGMI         (ribassista)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.config import setup_logging

_log = setup_logging("analytics.wagmi")

# ─── Soglie output ────────────────────────────────────────────────────────────

LABEL_FULL_WAGMI = "FULL_WAGMI"
LABEL_WAGMI      = "WAGMI"
LABEL_MEH        = "MEH"
LABEL_DEGEN      = "DEGEN"
LABEL_NGMI       = "NGMI"

# ─── Pesi (devono sommare a 1.0) ─────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "gamma_structure":  0.25,
    "etf_flow":         0.20,
    "squeeze_setup":    0.15,
    "funding_rate":     0.12,
    "oi_trend":         0.10,
    "options_structure":0.10,
    "liquidation_bias": 0.08,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Pesi non sommano a 1.0"


# ─── Input ───────────────────────────────────────────────────────────────────

@dataclass
class WagmiInputs:
    """Input per il calcolo del WAGMI Index.

    Tutti i campi sono opzionali; se None il fattore viene escluso e i pesi
    vengono riscalati proporzionalmente sugli altri fattori disponibili.
    """
    # GEX + gamma flip
    gex_usd:                     Optional[float] = None
    spot_price:                  Optional[float] = None
    gamma_flip_price:            Optional[float] = None

    # ETF flows
    etf_flow_3d_usd:             Optional[float] = None

    # Futures sentiment
    long_short_ratio:            Optional[float] = None
    funding_rate_annualized_pct: Optional[float] = None
    oi_change_7d_pct:            Optional[float] = None

    # Options structure
    call_wall_price:             Optional[float] = None
    put_wall_price:              Optional[float] = None
    max_pain_price:              Optional[float] = None

    # Liquidations
    liquidations_long_24h_usd:  Optional[float] = None
    liquidations_short_24h_usd: Optional[float] = None


# ─── Output ──────────────────────────────────────────────────────────────────

@dataclass
class WagmiResult:
    """Risultato del calcolo del WAGMI Index."""
    score:        float
    label:        str
    components:   dict[str, Optional[float]]
    weights_used: dict[str, float]
    narrative:    str
    key_level:    Optional[float]       # livello critico più rilevante (es. gamma flip)
    key_level_label: Optional[str]


# ─── Scoring per fattore ─────────────────────────────────────────────────────

def _score_gamma_structure(
    gex_usd: float,
    spot: float,
    gamma_flip: float,
) -> float:
    """Posizione rispetto al gamma flip + segno del GEX.

    Sopra il flip con GEX positivo → dealer long gamma → vol compressa → trend bull.
    Sotto il flip → dealer short gamma → vol amplificata → movimenti bruschi.
    """
    dist_pct = (spot - gamma_flip) / gamma_flip * 100  # positivo = sopra il flip

    if gex_usd > 0:
        if dist_pct >= 5:    return 0.92   # ben sopra flip, GEX positivo
        if dist_pct >= 2:    return 0.82
        if dist_pct >= 0:    return 0.70   # sopra flip ma di poco
        if dist_pct >= -3:   return 0.48   # just below flip — zona critica
        if dist_pct >= -10:  return 0.32
        return 0.18                         # lontano sotto il flip
    else:
        # GEX negativo: dealer short gamma, indipendentemente dal flip
        if dist_pct >= 0:    return 0.38   # sopra flip ma GEX negativo = insolito
        if dist_pct >= -5:   return 0.22
        return 0.10                         # worst case


def _score_etf_flow(flow_3d_usd: float) -> float:
    """Flussi ETF aggregati 3gg — proxy domanda istituzionale."""
    if flow_3d_usd > 500e6:    return 0.92
    if flow_3d_usd > 250e6:    return 0.78
    if flow_3d_usd > 50e6:     return 0.62
    if flow_3d_usd > -50e6:    return 0.50
    if flow_3d_usd > -250e6:   return 0.32
    return 0.12


def _score_squeeze_setup(long_short_ratio: float, funding_rate_ann: float) -> float:
    """Divergenza L/S ratio vs funding rate — segnala potenziale squeeze.

    Intuizione: funding alto + L/S basso (più short che long) significa che
    gli short stanno pagando il funding. Se il prezzo sale, la cascata di
    short cover è amplificata. È una situazione instabile bullish.

    Funding alto + L/S alto = retail crowded long, vulnerabile a flush.
    """
    ls = long_short_ratio
    fr = funding_rate_ann

    # Divergenza bullish: molti short (ls<1) + funding che sale (short pagano)
    if ls < 0.80 and fr > 30:     return 0.88   # squeeze setup ideale
    if ls < 0.80 and fr > 0:      return 0.78
    if ls < 0.80:                  return 0.70   # ls basso ma funding neutro
    if ls < 1.0:                   return 0.60
    if ls < 1.3 and fr < 20:       return 0.50   # bilanciato, funding sano
    if ls < 1.3:                   return 0.42
    if ls < 1.8:                   return 0.30   # retail long bias
    return 0.15                                    # crowded long → vulnerabile


def _score_funding_rate(rate_ann_pct: float) -> float:
    """Funding rate annualizzato — segnale contrarian (alta = surriscaldato)."""
    if rate_ann_pct < 0:      return 0.80
    if rate_ann_pct < 15:     return 0.70
    if rate_ann_pct < 30:     return 0.56
    if rate_ann_pct < 50:     return 0.38
    if rate_ann_pct < 75:     return 0.22
    return 0.12


def _score_oi_trend(oi_change_7d_pct: float) -> float:
    """Variazione OI 7gg — momentum del posizionamento futures."""
    if oi_change_7d_pct > 15:    return 0.82
    if oi_change_7d_pct > 7:     return 0.68
    if oi_change_7d_pct > 2:     return 0.58
    if oi_change_7d_pct > -2:    return 0.50
    if oi_change_7d_pct > -10:   return 0.35
    return 0.18


def _score_options_structure(
    spot: float,
    call_wall: float,
    put_wall: float,
    max_pain: float,
) -> float:
    """Posizione del prezzo rispetto ai livelli chiave delle options.

    Call wall → magnete a breve termine, breakout = bullish.
    Put wall → supporto strutturale.
    Max pain → attrazione gravitazionale verso expiry.
    """
    dist_call_pct = (call_wall - spot) / spot * 100   # positivo = spazio alla call wall
    dist_put_pct  = (spot - put_wall)  / spot * 100   # positivo = buffer sopra put wall
    above_max_pain = spot > max_pain

    if spot >= call_wall:
        return 0.88   # sopra call wall = breakout/squeeze
    if dist_call_pct < 2 and above_max_pain:
        return 0.75   # quasi alla call wall, sopra max pain = momentum positivo
    if dist_call_pct < 5 and above_max_pain:
        return 0.65
    if above_max_pain:
        return 0.55   # sopra max pain ma lontano dalla call wall
    if dist_put_pct > 15:
        return 0.42   # sotto max pain ma buffer ampio da put wall
    if dist_put_pct > 5:
        return 0.32
    return 0.18        # vicino a put wall = pericolo


def _score_liquidation_bias(long_usd: float, short_usd: float) -> float:
    """Asimmetria liquidazioni — evento direzionale contrarian.

    Cascade long (long_usd >> short_usd) → capitulation → potenziale bottom.
    Squeeze short (short_usd >> long_usd) → top signal contrarian.
    """
    total = long_usd + short_usd
    if total < 10e6:
        return 0.50   # mercato tranquillo

    long_ratio = long_usd / total if total > 0 else 0.5

    if total > 500e6:
        # Evento estremo
        return 0.72 if long_ratio > 0.60 else 0.28
    if total > 100e6:
        return 0.60 if long_ratio > 0.60 else 0.38
    # Moderato
    return 0.54 if long_ratio > 0.55 else 0.44


# ─── Label + narrative ───────────────────────────────────────────────────────

def _score_to_label(score: float) -> str:
    if score >= 80:  return LABEL_FULL_WAGMI
    if score >= 65:  return LABEL_WAGMI
    if score >= 51:  return LABEL_MEH
    if score >= 31:  return LABEL_DEGEN
    return LABEL_NGMI


def _build_narrative(inputs: WagmiInputs, raw: dict, score: float, label: str) -> str:
    parts: list[str] = []

    if inputs.gex_usd is not None and inputs.gamma_flip_price and inputs.spot_price:
        dist = (inputs.spot_price - inputs.gamma_flip_price) / inputs.gamma_flip_price * 100
        regime = "positive" if inputs.gex_usd > 0 else "negative"
        pos = f"{'sopra' if dist >= 0 else 'sotto'} gamma flip ({dist:+.1f}%)"
        parts.append(f"GEX {regime} ({inputs.gex_usd/1e6:+.0f}M), {pos}")

    if inputs.etf_flow_3d_usd is not None:
        parts.append(f"ETF 3d {inputs.etf_flow_3d_usd/1e6:+.0f}M")

    if inputs.long_short_ratio is not None and inputs.funding_rate_annualized_pct is not None:
        squeeze = inputs.long_short_ratio < 1.0 and inputs.funding_rate_annualized_pct > 30
        label_squeeze = "⚡ squeeze setup" if squeeze else "neutro"
        parts.append(f"L/S {inputs.long_short_ratio:.2f} | funding {inputs.funding_rate_annualized_pct:.0f}% ({label_squeeze})")

    if inputs.call_wall_price and inputs.spot_price:
        dist_cw = (inputs.call_wall_price - inputs.spot_price) / inputs.spot_price * 100
        parts.append(f"call wall ${inputs.call_wall_price:,.0f} ({dist_cw:+.1f}%)")

    return f"{label} [{score:.0f}/100] — " + " | ".join(parts) if parts else f"{label} [{score:.0f}/100]"


# ─── WagmiModel ──────────────────────────────────────────────────────────────

class WagmiModel:
    """Calcola il WAGMI Index composito per BTC.

    Usage:
        model = WagmiModel()
        result = model.compute(inputs)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or WEIGHTS.copy()

    def compute(self, inputs: WagmiInputs) -> WagmiResult:
        raw: dict[str, Optional[float]] = {}

        # Gamma structure — richiede tutti e 3 i valori
        if (inputs.gex_usd is not None
                and inputs.spot_price is not None
                and inputs.gamma_flip_price is not None
                and inputs.gamma_flip_price > 0):
            raw["gamma_structure"] = _score_gamma_structure(
                inputs.gex_usd, inputs.spot_price, inputs.gamma_flip_price
            )

        if inputs.etf_flow_3d_usd is not None:
            raw["etf_flow"] = _score_etf_flow(inputs.etf_flow_3d_usd)

        # Squeeze setup — richiede entrambi L/S e funding
        if (inputs.long_short_ratio is not None
                and inputs.funding_rate_annualized_pct is not None):
            raw["squeeze_setup"] = _score_squeeze_setup(
                inputs.long_short_ratio, inputs.funding_rate_annualized_pct
            )

        if inputs.funding_rate_annualized_pct is not None:
            raw["funding_rate"] = _score_funding_rate(inputs.funding_rate_annualized_pct)

        if inputs.oi_change_7d_pct is not None:
            raw["oi_trend"] = _score_oi_trend(inputs.oi_change_7d_pct)

        if (inputs.call_wall_price is not None
                and inputs.put_wall_price is not None
                and inputs.max_pain_price is not None
                and inputs.spot_price is not None):
            raw["options_structure"] = _score_options_structure(
                inputs.spot_price,
                inputs.call_wall_price,
                inputs.put_wall_price,
                inputs.max_pain_price,
            )

        if (inputs.liquidations_long_24h_usd is not None
                and inputs.liquidations_short_24h_usd is not None):
            raw["liquidation_bias"] = _score_liquidation_bias(
                inputs.liquidations_long_24h_usd,
                inputs.liquidations_short_24h_usd,
            )

        # Riscala pesi sui fattori disponibili
        available_weight = sum(self._weights[k] for k in raw)
        if available_weight <= 0:
            _log.warning("Nessun fattore disponibile per WAGMI")
            return WagmiResult(
                score=50.0, label=LABEL_MEH,
                components={k: None for k in WEIGHTS},
                weights_used={}, narrative="Dati insufficienti",
                key_level=None, key_level_label=None,
            )

        scaled = {k: self._weights[k] / available_weight for k in raw}
        score = round(sum(raw[k] * scaled[k] for k in raw) * 100.0, 1)

        label = _score_to_label(score)

        # Livello critico più rilevante da comunicare
        key_level = inputs.gamma_flip_price
        key_level_label = f"gamma flip ${key_level:,.0f}" if key_level else None

        narrative = _build_narrative(inputs, raw, score, label)
        components = {k: raw.get(k) for k in WEIGHTS}

        _log.info(
            "WAGMI score=%.1f label=%s | gamma_structure=%.2f etf_flow=%.2f squeeze=%.2f",
            score, label,
            raw.get("gamma_structure", float("nan")),
            raw.get("etf_flow", float("nan")),
            raw.get("squeeze_setup", float("nan")),
        )

        return WagmiResult(
            score=score,
            label=label,
            components=components,
            weights_used=scaled,
            narrative=narrative,
            key_level=key_level,
            key_level_label=key_level_label,
        )
