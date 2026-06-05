"""Adapter portfolio (PTF Core/VC): drift di allocazione → predizioni di mean-reversion.

Per ogni asset il cui peso corrente devia dal target oltre una soglia, emette una predizione
`direction` di rientro verso il target (overweight → atteso ribasso relativo / trim; underweight
→ atteso rialzo / add). Verificabile sullo stesso spine se esiste un price provider per l'asset;
gli asset senza dati prezzo restano semplicemente non verificati (il verifier li salta).

Input `holdings`: lista di dict {asset, current_weight, target_weight, price?}.
"""
from __future__ import annotations

from typing import Optional

from src.forecast.models import DIR_DOWN, DIR_UP, Prediction, TARGET_DIRECTION

SOURCE = "portfolio"
DEFAULT_HORIZON_DAYS = 30
DEFAULT_DRIFT_THRESHOLD = 0.05  # 5 punti percentuali di peso


def build_portfolio_predictions(
    holdings: list[dict],
    *,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    sleeve: Optional[str] = None,
    created_at: Optional[str] = None,
) -> list[Prediction]:
    """Emette una predizione di mean-reversion per ogni asset oltre la soglia di drift."""
    preds: list[Prediction] = []
    for h in holdings:
        asset = h["asset"]
        cur = float(h["current_weight"])
        tgt = float(h["target_weight"])
        drift = cur - tgt
        if abs(drift) < drift_threshold:
            continue

        # overweight (drift>0) → atteso rientro al ribasso; underweight → al rialzo
        direction = DIR_DOWN if drift > 0 else DIR_UP
        confidence = max(0.0, min(1.0, abs(drift) / (2 * drift_threshold)))
        price = h.get("price")

        spec: dict = {"direction": direction, "flat_band_pct": 2.0,
                      "drift": round(drift, 4), "current_weight": cur, "target_weight": tgt}
        if price is not None:
            spec["ref_price"] = float(price)

        common: dict = dict(source=SOURCE, asset=asset, horizon_days=horizon_days)
        if created_at is not None:
            common["created_at"] = created_at

        preds.append(Prediction(
            target_type=TARGET_DIRECTION,
            target_spec=spec,
            confidence=confidence,
            rationale=(
                f"{asset} {'overweight' if drift > 0 else 'underweight'} "
                f"{drift:+.1%} vs target ({cur:.1%}→{tgt:.1%})"
                + (f" [{sleeve}]" if sleeve else "")
                + f": atteso rientro {direction} su {horizon_days}g."
            ),
            components={"weight_drift": drift},
            **common,
        ))
    return preds
