"""Verifica d'esito: assegna un Outcome a ogni predizione matura.

`score_prediction` è puro (accetta un DataFrame di prezzi) → testabile offline.
`score_due_predictions` orchestra: legge le predizioni due dal DB, recupera i prezzi
reali e persiste gli esiti.

Convenzioni prezzi: DataFrame con DatetimeIndex e colonne `close`, `high`, `low`
(come restituito da src/flows/price_fetcher.PriceFetcher._load_from_db).
La finestra di valutazione va da created_at (escluso) a created_at + horizon_days (incluso).

Barre usate (solo sessioni UTC chiuse):
- ``realized_price`` è il close della barra giornaliera del giorno di fine
  orizzonte (``created_at`` + ``horizon_days``, data UTC). Quella barra esiste
  come close solo dopo le 00:00 UTC del giorno successivo: se manca, o se è
  marcata ``partial=1`` (sessione ancora aperta, lo spot al momento del fetch),
  la predizione resta open. Non si usa l'ultimo close disponibile al posto del
  close di fine orizzonte.
- Il tocco di un livello usa il massimo degli ``high`` e il minimo dei ``low``
  delle sole barre complete dentro la finestra. Una barra parziale non allunga
  né accorcia il range.

Questi prezzi non sono gli input del modello: lo spot salvato in ``ref_price``
arriva da Deribit al momento della predizione. Qui si misura solo l'esito.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from src.config import setup_logging
from src.forecast.models import (
    DIR_DOWN,
    DIR_FLAT,
    DIR_UP,
    Outcome,
    Prediction,
    TARGET_DIRECTION,
    TARGET_LEVEL,
    TARGET_PROB,
)

_log = setup_logging("forecast.verifier")

# Provider prezzi: (asset, start, end) -> DataFrame[close, high, low] con DatetimeIndex
PriceProvider = Callable[[str, datetime, datetime], pd.DataFrame]


def _window(prices: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(prices.index)
    df = prices.copy()
    df.index = idx
    mask = (idx > pd.Timestamp(start.date())) & (idx <= pd.Timestamp(end.date()))
    return df.loc[mask]


def _bar_date(value) -> datetime:
    """Data (mezzanotte, naive) della barra. Serve solo ``.date()``."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _drop_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Esclude le barre ``partial=1``. Senza colonna, il frame resta com'è."""
    if df.empty or "partial" not in df.columns:
        return df
    flag = pd.to_numeric(df["partial"], errors="coerce").fillna(0)
    return df.loc[flag.eq(0)]


def score_prediction(pred: Prediction, prices: pd.DataFrame) -> Optional[Outcome]:
    """Valuta una singola predizione contro i prezzi della finestra. None se dati insufficienti.

    Dati insufficienti anche quando la barra di fine orizzonte non c'è o è
    ancora parziale: non si chiude l'esito sul close infragiornaliero.
    """
    created = datetime.fromisoformat(pred.created_at).replace(tzinfo=timezone.utc)
    end = created + timedelta(days=pred.horizon_days)
    win = _drop_partial(_window(prices, created, end))
    if win.empty or "close" not in win.columns:
        return None
    if _bar_date(win.index[-1]).date() != end.date():
        return None

    ref = float(pred.target_spec.get("ref_price") or 0.0)
    realized_price = float(win["close"].iloc[-1])
    if ref <= 0:
        ref = float(win["close"].iloc[0])
    realized_return = realized_price / ref - 1.0 if ref else 0.0

    high = float(win["high"].max()) if "high" in win.columns else realized_price
    low = float(win["low"].min()) if "low" in win.columns else realized_price

    if pred.target_type == TARGET_DIRECTION:
        return _score_direction(pred, ref, realized_price, realized_return)
    if pred.target_type == TARGET_LEVEL:
        return _score_level(pred, ref, realized_price, realized_return, high, low)
    if pred.target_type == TARGET_PROB:
        return _score_prob(pred, ref, realized_price, realized_return)
    return None


def _score_direction(pred, ref, realized_price, ret) -> Outcome:
    band = float(pred.target_spec.get("flat_band_pct", 1.0)) / 100.0
    realized_dir = DIR_UP if ret > band else (DIR_DOWN if ret < -band else DIR_FLAT)
    predicted = pred.target_spec.get("direction")
    hit = realized_dir == predicted
    return Outcome(
        prediction_id=pred.id or 0,
        hit=hit,
        realized_return=ret,
        realized_price=realized_price,
        ref_price=ref,
        signed_error=ret,
        detail=f"predetto={predicted}, realizzato={realized_dir} (ret {ret:+.2%})",
    )


def _score_level(pred, ref, realized_price, ret, high, low) -> Outcome:
    level = float(pred.target_spec["level_price"])
    mode = pred.target_spec.get("mode", "reach")
    side = pred.target_spec.get("side", "above")
    touched = low <= level <= high

    if mode in ("reach", "break"):
        hit = touched
        verb = "raggiunto" if mode == "reach" else "rotto"
        detail = f"{pred.target_spec.get('level_name')} {level:,.0f} {verb}={hit} (range {low:,.0f}-{high:,.0f})"
    else:  # respect: il livello non deve essere violato dal lato sbagliato
        if side == "below":      # supporto (es. put wall): non scendere sotto
            hit = low >= level
        else:                     # resistenza (es. call wall): non salire sopra
            hit = high <= level
        detail = f"{pred.target_spec.get('level_name')} {level:,.0f} rispettato({side})={hit} (range {low:,.0f}-{high:,.0f})"

    signed_error = (realized_price - level) / level if level else None
    return Outcome(
        prediction_id=pred.id or 0,
        hit=hit,
        realized_return=ret,
        realized_price=realized_price,
        ref_price=ref,
        signed_error=signed_error,
        detail=detail,
    )


def _score_prob(pred, ref, realized_price, ret) -> Outcome:
    p = float(pred.target_spec.get("p", 0.5))
    event = ret > 0.0  # btc_return_positive
    brier = (p - (1.0 if event else 0.0)) ** 2
    hit = (p > 0.5) == event
    return Outcome(
        prediction_id=pred.id or 0,
        hit=hit,
        realized_return=ret,
        realized_price=realized_price,
        ref_price=ref,
        signed_error=p - (1.0 if event else 0.0),
        brier=brier,
        detail=f"p={p:.0%}, evento(ret>0)={event}, brier={brier:.3f}",
    )


def prediction_for_scoring(pred: Prediction, anchor: Optional[Prediction]) -> Prediction:
    """Prob della cohort: stesso ref e stessa finestra della direction, se c'è."""
    if pred.target_type == TARGET_PROB and anchor is not None:
        return _align_prob_to_direction(pred, anchor)
    return pred


def verification_price_provider(fetcher) -> PriceProvider:
    """Provider del verify: barre giornaliere complete, mai il close della sessione aperta.

    ``fetcher.fetch`` riscrive la coda recente e marca ``partial`` il giorno UTC
    corrente. Qui quelle righe vengono tolte prima dello scoring.
    """
    from src.config import TICKER_MAP
    from src.flows.price_fetcher import complete_bars

    def provider(asset, start, end):
        frame = fetcher.fetch(
            TICKER_MAP.get(asset, asset),
            start_date=start.date(),
            end_date=(end + timedelta(days=1)).date(),
        )
        return complete_bars(frame)

    return provider


def _cohort_key(pred: Prediction) -> tuple:
    """Cohort giornaliera: stessa data, source, asset e orizzonte."""
    return (pred.created_at[:10], pred.source, pred.asset, int(pred.horizon_days))


def _align_prob_to_direction(pred: Prediction, direction: Prediction) -> Prediction:
    """Valuta la prob sullo stesso ref e sulla stessa finestra della direction.

    Le prob già persistite non hanno `ref_price` nello spec: senza questo
    aggancio `score_prediction` usa la prima close della finestra, diversa
    dallo spot salvato sulla direction della stessa cohort.
    """
    spec = dict(pred.target_spec)
    ref = direction.target_spec.get("ref_price")
    if ref:
        spec["ref_price"] = ref
    return replace(pred, target_spec=spec, created_at=direction.created_at)


def _direction_anchors(db, due: list[Prediction]) -> dict[tuple, Prediction]:
    """Direction della cohort, dalle predizioni due o già verificate."""
    anchors = {_cohort_key(p): p for p in due if p.target_type == TARGET_DIRECTION}
    if not any(p.target_type == TARGET_PROB and _cohort_key(p) not in anchors for p in due):
        return anchors
    for row in db.get_with_outcomes(days=400):
        if row.get("target_type") != TARGET_DIRECTION:
            continue
        sib = Prediction.from_row(row)
        anchors.setdefault(_cohort_key(sib), sib)
    return anchors


def score_due_predictions(
    db, price_provider: PriceProvider, asof: Optional[datetime] = None,
    *,
    only_ids: Optional[set[int]] = None,
) -> list[Outcome]:
    """Verifica le predizioni mature nel DB e persiste gli esiti.

    Le formule di ``score_prediction`` non cambiano. ``only_ids``, se passato,
    limita la scrittura alle predizioni con quegli id: le altre coorti mature
    restano aperte e gli esiti già presenti non vengono ricalcolati
    (``insert_outcome`` è INSERT OR IGNORE).

    Args:
        db: PredictionDB.
        price_provider: callable (asset, start, end) → DataFrame prezzi.
        asof: istante di riferimento (default now UTC).
        only_ids: sottoinsieme di id da verificare. None = tutte le mature.

    Returns:
        Lista degli Outcome creati.
    """
    due = db.get_due(asof)
    if only_ids is not None:
        due = [p for p in due if p.id in only_ids]
    anchors = _direction_anchors(db, due)
    created: list[Outcome] = []
    for pred in due:
        scoring = prediction_for_scoring(pred, anchors.get(_cohort_key(pred)))
        start = datetime.fromisoformat(scoring.created_at).replace(tzinfo=timezone.utc)
        end = scoring.matures_at
        try:
            prices = price_provider(pred.asset, start, end)
        except Exception as exc:
            _log.warning("Prezzi non disponibili per pred %s: %s", pred.id, exc)
            continue
        outcome = score_prediction(scoring, prices)
        if outcome is None:
            _log.info("Dati insufficienti per pred %s, resta open", pred.id)
            continue
        if db.insert_outcome(outcome):
            created.append(outcome)
            _log.info("Pred %s verificata: hit=%s — %s", pred.id, outcome.hit, outcome.detail)
    return created
