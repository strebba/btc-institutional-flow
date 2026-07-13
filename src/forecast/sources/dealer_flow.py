"""Adapter dealer-flow: SignalResult + GexSnapshot → list[Prediction].

Produce fino a 3 previsioni verificabili dal segnale multi-fattore:

1. DIRECTION — direzione attesa di BTC sull'orizzonte, dal regime del segnale
   (LONG→up, RISK_OFF→down, CAUTION→flat). Confidence dalla distanza dello score da 50.

2. LEVEL — il "magnete" dominante dato il regime gamma:
   - GEX > 0 (positive gamma, mean-reverting): il prezzo tende a essere *pinnato* verso
     max_pain → target = max_pain, mode="reach".
   - GEX < 0 (negative gamma, trend-amplifying): rottura del gamma_flip nella direzione
     del segnale → target = gamma_flip, mode="break".
   Se i livelli non sono disponibili, la previsione level viene omessa.

3. PROB — probabilità che il return BTC sull'orizzonte sia positivo, mappata dallo score.

L'orizzonte di default è 5 giorni: coerente con il lead documentato dei flussi IBIT
(Granger ~5-7gg). Funzione pura: nessun I/O, testabile offline.
"""
from __future__ import annotations

from typing import Optional

from src.analytics.factor_scorers import (
    SIGNAL_LONG,
    SIGNAL_RISK_OFF,
    SignalResult,
)
from src.forecast.models import (
    DIR_DOWN,
    DIR_FLAT,
    DIR_UP,
    Prediction,
    TARGET_DIRECTION,
    TARGET_LEVEL,
    TARGET_PROB,
)

SOURCE = "dealer_flow"
DEFAULT_HORIZON_DAYS = 5
_FLAT_BAND_PCT = 1.0  # |return| < 1% su orizzonte = "flat"


def _confidence_from_score(score: float) -> float:
    """Distanza dello score da 50, normalizzata a [0,1]. 50→0, 0/100→1."""
    return max(0.0, min(1.0, abs(score - 50.0) / 50.0))


def _prob_up_from_score(score: float) -> float:
    """Mappa lineare score→P(return>0), centrata su 0.5, clampata a [0.05, 0.95]."""
    return max(0.05, min(0.95, 0.5 + (score - 50.0) / 100.0))


def build_dealer_flow_predictions(
    result: SignalResult,
    *,
    spot_price: float,
    gamma_flip: Optional[float] = None,
    max_pain: Optional[float] = None,
    put_wall: Optional[float] = None,
    call_wall: Optional[float] = None,
    total_net_gex: Optional[float] = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    asset: str = "BTC",
    created_at: Optional[str] = None,
    weights_version: Optional[int] = None,
    model_version: str = "v1",
) -> list[Prediction]:
    """Costruisce le previsioni dealer-flow da un SignalResult e dai livelli GEX."""
    if not spot_price or spot_price <= 0:
        raise ValueError("spot_price deve essere positivo")

    conf = _confidence_from_score(result.score)
    common = dict(
        source=SOURCE,
        asset=asset,
        horizon_days=horizon_days,
        score_ref=result.score,
        components=result.components,
        weights_version=weights_version,
        model_version=model_version,
    )
    if created_at is not None:
        common["created_at"] = created_at

    preds: list[Prediction] = []

    # ── 1. DIRECTION ──────────────────────────────────────────────────────────
    if result.signal == SIGNAL_LONG:
        direction = DIR_UP
    elif result.signal == SIGNAL_RISK_OFF:
        direction = DIR_DOWN
    else:
        direction = DIR_FLAT

    preds.append(Prediction(
        target_type=TARGET_DIRECTION,
        target_spec={
            "direction": direction,
            "ref_price": spot_price,
            "flat_band_pct": _FLAT_BAND_PCT,
        },
        confidence=conf,
        rationale=(
            f"Segnale {result.signal} (score {result.score:.0f}) → direzione attesa "
            f"{direction} su {horizon_days}g. {result.reason}"
        ),
        **common,
    ))

    # ── 2. LEVEL (magnete dominante per regime) ────────────────────────────────
    gex = total_net_gex if total_net_gex is not None else 0.0
    if gex >= 0 and max_pain:
        # Positive gamma → pin verso max pain
        side = "above" if max_pain >= spot_price else "below"
        preds.append(Prediction(
            target_type=TARGET_LEVEL,
            target_spec={
                "level_name": "max_pain",
                "level_price": float(max_pain),
                "mode": "reach",
                "side": side,
                "ref_price": spot_price,
            },
            confidence=conf,
            rationale=(
                f"Positive gamma ({gex/1e6:+.0f}M): regime mean-reverting, prezzo atteso "
                f"gravitare verso max pain {max_pain:,.0f} entro {horizon_days}g."
            ),
            **common,
        ))
    elif gex < 0 and gamma_flip:
        # Negative gamma → rottura del gamma flip nella direzione del segnale
        side = "below" if result.signal == SIGNAL_RISK_OFF else "above"
        preds.append(Prediction(
            target_type=TARGET_LEVEL,
            target_spec={
                "level_name": "gamma_flip",
                "level_price": float(gamma_flip),
                "mode": "break",
                "side": side,
                "ref_price": spot_price,
            },
            confidence=conf,
            rationale=(
                f"Negative gamma ({gex/1e6:+.0f}M): regime amplificante, attesa rottura "
                f"del gamma flip {gamma_flip:,.0f} ({side}) entro {horizon_days}g."
            ),
            **common,
        ))

    # ── 3. PROB ────────────────────────────────────────────────────────────────
    p_up = _prob_up_from_score(result.score)
    preds.append(Prediction(
        target_type=TARGET_PROB,
        target_spec={"event": "btc_return_positive", "p": p_up},
        confidence=abs(p_up - 0.5) * 2.0,
        rationale=(
            f"P(BTC return > 0 a {horizon_days}g) = {p_up:.0%}, da score {result.score:.0f}."
        ),
        **common,
    ))

    return preds
