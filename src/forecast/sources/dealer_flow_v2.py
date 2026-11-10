"""Shadow dealer-flow v2. Stesse feature e stessi pesi di v1, regole diverse.

Non sostituisce ``dealer_flow``. Le righe hanno ``source='dealer_flow_v2'`` così
``idx_pred_day_unique`` (date, source, target_type, horizon) resta libero per
la cohort live. Le regole sono quelle pre-registrate in
``docs/forecast-v2-preregistration.md`` (2026-11-10): niente look-ahead.

1. Direzione = segno di (score − 50). Se la confidence |score−50|/50 è sotto
   0.10 (score in (45, 55)) non si scrive una direction: è un'astensione.
   La banda "flat" usata in verifica è 0,5 σ dell'orizzonte, stimata sulle
   ultime 20 variazioni log giornaliere strettamente precedenti alla data
   della predizione. Sotto le 10 osservazioni si usa il prior 4%.

2. Max pain solo se dista al più il 4% dal ref. Altrimenti niente level.
   Il gamma flip in negative gamma resta quello di v1.

3. La probabilità è Platt (logistica su score) stimata solo sugli esiti prob
   ``dealer_flow`` già maturi prima dell'istante della predizione, con ridge.
   Sotto i 30 esiti: shrink a metà tra la p grezza di v1 e il base rate
   espandente (0,5 se non c'è storia).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np

from src.forecast.models import (
    DIR_DOWN,
    DIR_UP,
    Prediction,
    TARGET_DIRECTION,
    TARGET_LEVEL,
    TARGET_PROB,
)
from src.forecast.sources.dealer_flow import (
    _confidence_from_score,
    _prob_up_from_score,
)

SOURCE = "dealer_flow_v2"
MODEL_VERSION = "v2"

# |score-50|/50 sotto questa soglia: nessuna direction.
ABSTAIN_CONFIDENCE = 0.10
# Max pain più lontano di così dal ref: nessuna level.
MAX_PAIN_MAX_DISTANCE = 0.04
VOL_LOOKBACK = 20
VOL_MIN_OBS = 10
FLAT_BAND_SIGMA = 0.5
FLAT_BAND_FALLBACK_PCT = 4.0
FLAT_BAND_FLOOR_PCT = 0.1
PLATT_MIN_N = 30
PLATT_RIDGE = 1.0
_P_MIN, _P_MAX = 0.05, 0.95


def flat_band_pct_from_returns(log_returns: Sequence[float], horizon_days: int) -> float:
    """0,5 σ dell'orizzonte, in punti percentuali. Prior 4% se i dati non bastano."""
    clean = [float(r) for r in log_returns if r is not None and math.isfinite(float(r))]
    if len(clean) < VOL_MIN_OBS:
        return FLAT_BAND_FALLBACK_PCT
    window = clean[-VOL_LOOKBACK:]
    sigma_d = float(np.std(window, ddof=1))
    sigma_h = sigma_d * math.sqrt(int(horizon_days))
    return max(FLAT_BAND_FLOOR_PCT, 100.0 * FLAT_BAND_SIGMA * sigma_h)


def trailing_log_returns(fetcher, before: date) -> list[float]:
    """Log-return giornalieri con data < ``before``. Non scarica nulla.

    Se la colonna ``partial`` c'è (barre della sessione ancora aperta) quelle
    righe non entrano. Senza la colonna si usano i close già salvati.
    """
    if before <= date(2010, 1, 1):
        return []
    start = before - timedelta(days=VOL_LOOKBACK * 4)
    end = before - timedelta(days=1)
    frame = fetcher._load_from_db("BTC-USD", start, end)
    if frame is None or frame.empty or "close" not in frame.columns:
        return []
    # Stesso filtro del verifier: partial=1 è la sessione UTC non chiusa
    # (schema prices.partial di #25). Senza la colonna il frame è già completo.
    from src.flows.price_fetcher import complete_bars
    frame = complete_bars(frame)
    if frame.empty:
        return []
    if "daily_return" in frame.columns and int(frame["daily_return"].notna().sum()) >= VOL_MIN_OBS:
        return [float(v) for v in frame["daily_return"].dropna().tolist()]
    closes = frame["close"].astype(float)
    rets = np.log(closes / closes.shift(1)).dropna()
    return [float(v) for v in rets.tolist()]


def filter_matured_prob_rows(rows: Sequence[dict], before: datetime) -> list[tuple[float, int]]:
    """(score, y) degli esiti prob v1 la cui finestra è chiusa prima di ``before``.

    ``y = 1`` se il return realizzato è positivo, come il verifier. Una riga
    con ``created_at + horizon`` successivo a ``before`` non è ancora
    osservabile: non entra, anche se nel DB c'è già un realized_return.
    """
    if before.tzinfo is None:
        before = before.replace(tzinfo=timezone.utc)
    out: list[tuple[float, int]] = []
    for row in rows:
        if row.get("source") not in (None, "dealer_flow"):
            continue
        if row.get("target_type") != TARGET_PROB:
            continue
        if row.get("score_ref") is None or row.get("realized_return") is None:
            continue
        created = datetime.fromisoformat(str(row["created_at"])).replace(tzinfo=timezone.utc)
        horizon = int(row.get("horizon_days") or 0)
        # Maturità uguale a ``before`` non è ancora alle spalle della predizione.
        if created + timedelta(days=horizon) >= before:
            continue
        out.append((float(row["score_ref"]), 1 if float(row["realized_return"]) > 0 else 0))
    return out


def matured_prob_history(db, before: datetime) -> list[tuple[float, int]]:
    """Storia dal DB. Finestra lunga: il filtro di maturità è ``before``, non i giorni."""
    rows = db.get_with_outcomes(days=3650, source="dealer_flow")
    return filter_matured_prob_rows(rows, before)


def calibrate_prob(score: float, history: Sequence[tuple[float, int]]) -> tuple[float, str]:
    """(p, metodo). ``platt`` solo con almeno ``PLATT_MIN_N`` esiti già maturi."""
    p_raw = _prob_up_from_score(score)
    n = len(history)
    pi = (sum(y for _, y in history) / n) if n else 0.5
    if n < PLATT_MIN_N:
        return _clip(0.5 * p_raw + 0.5 * pi), "shrink_base_rate"
    try:
        a, b = _fit_platt(history)
    except Exception:
        return _clip(0.5 * p_raw + 0.5 * pi), "shrink_base_rate"
    z = a + b * (float(score) - 50.0)
    return _clip(_sigmoid(z)), "platt"


def build_dealer_flow_v2_predictions(
    *,
    score: float,
    spot_price: float,
    p_up: float,
    flat_band_pct: float,
    components: Optional[dict] = None,
    gamma_flip: Optional[float] = None,
    max_pain: Optional[float] = None,
    total_net_gex: Optional[float] = None,
    horizon_days: int = 5,
    asset: str = "BTC",
    created_at: Optional[str] = None,
    weights_version: Optional[int] = None,
    calibration: str = "shrink_base_rate",
    calibration_n: int = 0,
) -> list[Prediction]:
    """Costruisce la cohort ombra. Pura: p e banda arrivano già stimati."""
    if not spot_price or spot_price <= 0:
        raise ValueError("spot_price deve essere positivo")

    conf = _confidence_from_score(score)
    common = dict(
        source=SOURCE,
        asset=asset,
        horizon_days=horizon_days,
        score_ref=float(score),
        components=components or {},
        weights_version=weights_version,
        model_version=MODEL_VERSION,
    )
    if created_at is not None:
        common["created_at"] = created_at

    preds: list[Prediction] = []

    # ── 1. DIRECTION: segno dello score, astensione se confidence bassa ──
    if conf >= ABSTAIN_CONFIDENCE and score != 50.0:
        direction = DIR_UP if score > 50.0 else DIR_DOWN
        preds.append(Prediction(
            target_type=TARGET_DIRECTION,
            target_spec={
                "direction": direction,
                "ref_price": float(spot_price),
                "flat_band_pct": float(flat_band_pct),
                "flat_band_rule": "half_sigma_horizon",
            },
            confidence=conf,
            rationale=(
                f"v2: segno dello score {score:.0f} → {direction} su {horizon_days}g "
                f"(confidence {conf:.2f} ≥ {ABSTAIN_CONFIDENCE:.2f}). "
                f"Banda flat {flat_band_pct:.2f}% = 0,5 σ dell'orizzonte."
            ),
            **common,
        ))

    # ── 2. LEVEL: max pain solo entro il 4%; gamma flip come v1 ──
    gex = total_net_gex if total_net_gex is not None else 0.0
    if gex >= 0 and max_pain:
        distance = abs(float(max_pain) - float(spot_price)) / float(spot_price)
        if distance <= MAX_PAIN_MAX_DISTANCE:
            side = "above" if float(max_pain) >= float(spot_price) else "below"
            preds.append(Prediction(
                target_type=TARGET_LEVEL,
                target_spec={
                    "level_name": "max_pain",
                    "level_price": float(max_pain),
                    "mode": "reach",
                    "side": side,
                    "ref_price": float(spot_price),
                    "max_distance": MAX_PAIN_MAX_DISTANCE,
                },
                confidence=conf,
                rationale=(
                    f"v2: max pain {float(max_pain):,.0f} a {distance:.1%} dal ref, "
                    f"entro il {MAX_PAIN_MAX_DISTANCE:.0%}: target reach a {horizon_days}g."
                ),
                **common,
            ))
    elif gex < 0 and gamma_flip:
        side = "below" if score < 50.0 else "above"
        preds.append(Prediction(
            target_type=TARGET_LEVEL,
            target_spec={
                "level_name": "gamma_flip",
                "level_price": float(gamma_flip),
                "mode": "break",
                "side": side,
                "ref_price": float(spot_price),
            },
            confidence=conf,
            rationale=(
                f"v2: negative gamma, rottura del gamma flip {float(gamma_flip):,.0f} "
                f"({side}) entro {horizon_days}g."
            ),
            **common,
        ))

    # ── 3. PROB già ricalibrata dal chiamante ──
    p = _clip(p_up)
    preds.append(Prediction(
        target_type=TARGET_PROB,
        target_spec={
            "event": "btc_return_positive",
            "p": p,
            "ref_price": float(spot_price),
            "calibration": calibration,
            "calibration_n": int(calibration_n),
        },
        confidence=abs(p - 0.5) * 2.0,
        rationale=(
            f"v2: P(return>0 a {horizon_days}g) = {p:.0%} "
            f"({calibration}, n={calibration_n}), score {score:.0f}."
        ),
        **common,
    ))
    return preds


def v2_from_stored_v1(
    stored: Sequence[Prediction],
    *,
    p_up: float,
    flat_band_pct: float,
    calibration: str,
    calibration_n: int,
) -> list[Prediction]:
    """Cohort ombra dagli input già scritti su v1 (stesso score, ref, livelli, pesi).

    Non rilegge Deribit. Se lo score non c'è, la direction è un'astensione
    (score trattato come 50 solo per la formula della p, che il chiamante ha
    già calcolato).
    """
    if not stored:
        return []
    direction = next((p for p in stored if p.target_type == TARGET_DIRECTION), None)
    level = next((p for p in stored if p.target_type == TARGET_LEVEL), None)
    prob = next((p for p in stored if p.target_type == TARGET_PROB), None)
    anchor = direction or prob or level or stored[0]
    spot = None
    for row in (direction, prob, level):
        if row is not None and row.target_spec.get("ref_price"):
            spot = float(row.target_spec["ref_price"])
            break
    if not spot:
        return []
    score = float(anchor.score_ref) if anchor.score_ref is not None else 50.0
    max_pain = gamma_flip = None
    gex = 0.0
    if level is not None:
        name = level.target_spec.get("level_name")
        price = level.target_spec.get("level_price")
        if name == "max_pain" and price:
            max_pain = float(price)
            gex = 1.0
        elif name == "gamma_flip" and price:
            gamma_flip = float(price)
            gex = -1.0
    return build_dealer_flow_v2_predictions(
        score=score,
        spot_price=spot,
        p_up=p_up,
        flat_band_pct=flat_band_pct,
        components=dict(anchor.components or {}),
        gamma_flip=gamma_flip,
        max_pain=max_pain,
        total_net_gex=gex,
        horizon_days=int(anchor.horizon_days),
        asset=anchor.asset,
        created_at=anchor.created_at,
        weights_version=anchor.weights_version,
        calibration=calibration,
        calibration_n=calibration_n,
    )


def _clip(p: float) -> float:
    return max(_P_MIN, min(_P_MAX, float(p)))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _log_sigmoid(z: float) -> float:
    if z >= 0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def _fit_platt(history: Sequence[tuple[float, int]]) -> tuple[float, float]:
    """Minimizza NLL + (λ/2)(a²+b²). Parte da (0, 0): senza segnale, p=0,5."""
    from scipy.optimize import minimize

    pairs = [(float(s), int(y)) for s, y in history]

    def nll(theta: np.ndarray) -> float:
        a, b = float(theta[0]), float(theta[1])
        loss = 0.5 * PLATT_RIDGE * (a * a + b * b)
        for score, y in pairs:
            z = a + b * (score - 50.0)
            loss -= _log_sigmoid(z) if y else _log_sigmoid(-z)
        return loss

    fit = minimize(nll, np.zeros(2), method="L-BFGS-B", bounds=[(-5.0, 5.0), (-2.0, 2.0)])
    if not fit.success:
        raise RuntimeError(fit.message)
    return float(fit.x[0]), float(fit.x[1])
