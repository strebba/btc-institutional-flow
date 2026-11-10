"""Ricostruzione point-in-time di una coorte dealer-flow mancata.

Il predict live (``gather_dealer_flow_context``) legge solo lo stato corrente.
Rilanciarlo oggi per il 2026-11-03 stamperebbe i dati di oggi su quella data.
Questo modulo usa soltanto osservazioni chiuse allo slot (default 07:30 UTC,
``forecast.schedule`` in settings) e non modifica le altre coorti.

Cosa è ricostruibile per il 2026-11-03
--------------------------------------
* Prezzo BTC: indice Deribit ``btc_usd`` (``public/get_index_chart_data``),
  ultimo print con timestamp <= slot. La finestra ``1m`` è oraria e ancorata
  a adesso, quindi il print delle 07:30 di solito non esiste: si usa quello
  delle 07:00, e i print successivi allo slot si scartano.
* Flussi ETF: print IBIT osservati (CoinGlass flow-history, altrimenti la
  tabella Farside). ``etf_flow_3d_usd`` è la somma IBIT sui tre giorni di
  calendario D-3..D-1. I print con data >= D si scartano. Non si usano le
  stime N-PORT / yfinance di ``FarsideScraper.fetch``.
* CoinGlass (funding 8h ricampionato daily, OI, long/short, liquidazioni):
  una barra daily è usabile solo se si è chiusa entro lo slot (open + 1 giorno
  <= slot). La barra del giorno D, che contiene i print dopo le 07:30, si
  scarta. Senza ``COINGLASS_API_KEY`` quei fattori restano assenti, come nel
  predict live quando CoinGlass non risponde.
* Pesi: ultima ``weight_versions`` con ``created_at`` <= slot e status
  ``active`` o ``retired``. Lo schema non ha ``activated_at``: una versione
  proposta prima dello slot e attivata dopo verrebbe selezionata. Le versioni
  create dopo lo slot non si usano. Se non ce n'è nessuna, si usano i prior
  di ``config/weights.yaml`` / ``factor_scorers.WEIGHTS`` e
  ``weights_version`` resta vuoto.
* Barriere: ``barrier_snapshots`` con ``snapshot_date`` < D, oppure = D solo
  se ``created_at`` <= slot. Lo snapshot delle 22:30 UTC del giorno D è dopo
  lo slot.

Cosa non è ricostruibile, e perché la coorte è degradata
--------------------------------------------------------
Il GEX non ha un'API storica in questo repo. Deribit qui è live
(``get_instruments`` + ticker). Gli endpoint opzioni CoinGlass
(``fetch_options_info``, ``fetch_options_max_pain``) sono snapshot correnti.
L'unica copia è ``gex_snapshots`` (una riga per data+asset, UPSERT: un
aggiornamento pomeridiano sovrascrive il mattino). Una riga del giorno D è
usabile solo se ``timestamp`` <= slot. Una riga di un giorno precedente non
è il gamma del 3/11. Se manca, si omettono GEX, put/call, call wall e la
predizione ``level``, e la coorte è marcata ``degraded_inputs``. Non si
rifiuta: gli altri fattori hanno storico e la coorte può comunque essere
scritta e, dopo, verificata.

Il flag sta in ``predictions.metadata`` (JSON, NULL sulle righe live) e non
nella source: la source resta ``dealer_flow``, così la riga occupa lo slot
unico del giorno e il verifier la tratta come le altre. ``created_at`` è lo
slot, non l'istante del rebuild.

Lo scoring non cambia formula. ``score_cohort`` chiama ``score_due_predictions``
con ``only_ids`` limitato agli id appena inseriti.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from src.analytics.factor_scorers import WEIGHTS as DEFAULT_WEIGHTS
from src.analytics.factor_scorers import SignalInputs, SignalModel
from src.config import PROJECT_ROOT, get_settings, setup_logging
from src.forecast.calibration import load_weights_config
from src.forecast.models import Prediction
from src.forecast.sources.dealer_flow import SOURCE, build_dealer_flow_predictions

_log = setup_logging("forecast.reconstruct")

# Stesse soglie di src/forecast/context.py (non importate: quel modulo non ha
# costanti pubbliche e il gather live non va eseguito da qui).
_BARRIER_PCT = 0.05
_CALL_WALL_PCT = 0.02
_DAY_SECONDS = 86_400

_GEX_SQL = """
SELECT date, timestamp, total_net_gex, gamma_flip_price, call_wall, max_pain, put_call_ratio
FROM gex_snapshots
WHERE asset = ?
"""
_WEIGHTS_SQL = """
SELECT id, created_at, weights, status
FROM weight_versions
WHERE source = ?
"""
_BARRIER_SQL = """
SELECT snapshot_date, created_at, level_price_btc
FROM barrier_snapshots
"""
_COHORT_SQL = """
SELECT 1 AS present FROM predictions WHERE date = ? AND source = ? LIMIT 1
"""


@dataclass(frozen=True)
class FlowPrint:
    """Print di flusso ETF già osservato (non una stima)."""

    day: date
    ticker: str
    flow_usd: float
    source: str = ""


@dataclass(frozen=True)
class TimedBar:
    """Barra OHLC il cui ``open_ts`` è l'apertura. È nota a open + durata."""

    open_ts: datetime
    value: float
    bar_seconds: int


@dataclass(frozen=True)
class LiqBar:
    """Barra di liquidazioni, stessa regola di chiusura di ``TimedBar``."""

    open_ts: datetime
    long_usd: float
    short_usd: float
    bar_seconds: int = _DAY_SECONDS


@dataclass(frozen=True)
class GexRow:
    """Riga di ``gex_snapshots`` ridotta ai campi usati dal segnale."""

    day: date
    timestamp: datetime
    total_net_gex: float
    gamma_flip_price: Optional[float]
    call_wall: Optional[float]
    max_pain: Optional[float]
    put_call_ratio: Optional[float]


@dataclass(frozen=True)
class WeightRow:
    """Versione pesi. ``created_at`` è l'inserimento, non l'attivazione."""

    id: int
    created_at: datetime
    weights: dict[str, float]
    status: str


@dataclass(frozen=True)
class BarrierRow:
    """Livello da ``barrier_snapshots``."""

    snapshot_date: date
    created_at: Optional[datetime]
    level_price_btc: Optional[float]


@dataclass
class MarketHistory:
    """Tutto ciò che i client hanno restituito, prima del taglio allo slot."""

    spot_prints: list[tuple[datetime, float]] = field(default_factory=list)
    flows: Optional[list[FlowPrint]] = None
    funding_bars: list[TimedBar] = field(default_factory=list)
    oi_bars: list[TimedBar] = field(default_factory=list)
    long_short_bars: list[TimedBar] = field(default_factory=list)
    liquidation_bars: list[LiqBar] = field(default_factory=list)
    gex_rows: list[GexRow] = field(default_factory=list)
    weight_rows: list[WeightRow] = field(default_factory=list)
    barrier_rows: list[BarrierRow] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)


@dataclass
class LocalPaths:
    """File SQLite locali. Mai un URL Postgres."""

    predictions_db: Path
    gex_db: Path
    notes_db: Path


@dataclass
class ReconstructResult:
    """Predizioni che *sarebbero* (o sono state) scritte, e la loro provenienza."""

    slot: datetime
    predictions: list[Prediction]
    provenance: dict[str, Any]
    degraded_inputs: list[str]
    refused: Optional[str]
    signal_score: Optional[float]
    signal_label: Optional[str]
    weights_version: Optional[int]


def as_utc(ts: datetime) -> datetime:
    """Normalizza a UTC. I timestamp naive del DB sono già UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def iso_z(ts: datetime) -> str:
    """ISO UTC con suffisso Z, per il metadata e il report."""
    return as_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(value: Any) -> Optional[datetime]:
    """Parse ISO. ``Z`` è accettato anche su Python 3.10."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def slot_on(day: date, hour: Optional[int] = None, minute: Optional[int] = None) -> datetime:
    """Slot predict di ``day``. Default: ora/minuto di ``settings.forecast.schedule``."""
    if hour is None or minute is None:
        sched = get_settings().get("forecast", {}).get("schedule", {})
        hour = int(sched.get("predict_hour_utc", 7))
        minute = int(sched.get("predict_minute_utc", 30))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)


def index_chart_range(slot: datetime, now: Optional[datetime] = None) -> str:
    """Range Deribit che copre lo slot. La finestra è sempre ancorata a ``now``."""
    now = as_utc(now or datetime.now(timezone.utc))
    age = now - as_utc(slot)
    if age <= timedelta(hours=20):
        return "1d"
    if age <= timedelta(days=27):
        return "1m"
    if age <= timedelta(days=360):
        return "1y"
    return "all"


def price_at_slot(
    prints: list[tuple[datetime, float]], slot: datetime,
) -> Optional[tuple[datetime, float]]:
    """Ultimo print con timestamp <= slot e prezzo positivo. I successivi si ignorano."""
    known = [(as_utc(ts), px) for ts, px in prints if as_utc(ts) <= slot and px > 0]
    if not known:
        return None
    return max(known, key=lambda item: item[0])


def _closed_bars(bars: list[TimedBar], slot: datetime) -> list[TimedBar]:
    kept = [
        bar for bar in bars
        if as_utc(bar.open_ts) + timedelta(seconds=bar.bar_seconds) <= slot
    ]
    kept.sort(key=lambda bar: as_utc(bar.open_ts))
    return kept


def _closed_liq(bars: list[LiqBar], slot: datetime) -> list[LiqBar]:
    kept = [
        bar for bar in bars
        if as_utc(bar.open_ts) + timedelta(seconds=bar.bar_seconds) <= slot
    ]
    kept.sort(key=lambda bar: as_utc(bar.open_ts))
    return kept


def etf_inputs(
    flows: Optional[list[FlowPrint]], slot: datetime,
) -> dict[str, Any]:
    """Somma IBIT su D-3..D-1 e lead Granger sul serie tagliata a D-1.

    I print con data >= D non entrano né nella somma né nella serie del lead
    (``iloc[-6]`` sul solo storico noto, come il predict live sulla serie
    troncata). I ticker diversi da IBIT non entrano nella somma.
    """
    slot_day = slot.date()
    window_start = slot_day - timedelta(days=3)
    window_end = slot_day - timedelta(days=1)
    window = {
        (window_start + timedelta(days=i)).isoformat(): 0.0
        for i in range(3)
    }
    daily: dict[date, float] = {}
    dropped = 0
    sources: set[str] = set()
    if flows:
        for flow in flows:
            if flow.ticker.upper() != "IBIT":
                continue
            if flow.day >= slot_day:
                dropped += 1
                continue
            daily[flow.day] = daily.get(flow.day, 0.0) + float(flow.flow_usd)
            if window_start <= flow.day <= window_end:
                window[flow.day.isoformat()] += float(flow.flow_usd)
            if flow.source:
                sources.add(flow.source)
    ordered_days = sorted(daily)
    granger = daily[ordered_days[-6]] if len(ordered_days) >= 6 else None
    granger_day = ordered_days[-6].isoformat() if len(ordered_days) >= 6 else None
    return {
        "known": bool(daily),
        "etf_flow_3d_usd": sum(window.values()) if daily else None,
        "window": window,
        "granger_lead_flow_usd": granger,
        "granger_lead_date": granger_day,
        "dropped_on_or_after_slot": dropped,
        "sources": sorted(sources),
    }


def select_gex(rows: list[GexRow], slot: datetime) -> tuple[Optional[GexRow], dict[str, int]]:
    """Snapshot del giorno dello slot con timestamp <= slot. Il più recente.

    Una riga dello stesso giorno ma successiva allo slot è lookahead (l'UPSERT
    pomeridiano). Una riga di un altro giorno non è il GEX di quello slot.
    """
    day = slot.date()
    eligible = [
        row for row in rows
        if row.day == day and as_utc(row.timestamp) <= slot
    ]
    counts = {
        "same_day_after_slot": sum(
            1 for row in rows if row.day == day and as_utc(row.timestamp) > slot
        ),
        "later_days": sum(1 for row in rows if row.day > day),
    }
    if not eligible:
        return None, counts
    return max(eligible, key=lambda row: as_utc(row.timestamp)), counts


def select_weights(
    rows: list[WeightRow], slot: datetime,
) -> tuple[Optional[WeightRow], int]:
    """Ultima versione attivata (active|retired) creata non dopo lo slot.

    Le ``proposed`` non sono mai state i pesi live. Le righe con
    ``created_at`` > slot si contano come scartate.
    """
    rejected = 0
    eligible: list[WeightRow] = []
    for row in rows:
        if as_utc(row.created_at) > slot:
            rejected += 1
            continue
        if row.status not in ("active", "retired"):
            continue
        eligible.append(row)
    if not eligible:
        return None, rejected
    best = max(eligible, key=lambda row: (as_utc(row.created_at), row.id))
    return best, rejected


def select_barrier_levels(
    rows: list[BarrierRow], slot: datetime,
) -> tuple[list[float], Optional[str]]:
    """Livelli dell'ultimo snapshot noto allo slot."""
    eligible: list[BarrierRow] = []
    for row in rows:
        if row.level_price_btc is None or row.level_price_btc <= 0:
            continue
        if row.snapshot_date < slot.date():
            eligible.append(row)
        elif (
            row.snapshot_date == slot.date()
            and row.created_at is not None
            and as_utc(row.created_at) <= slot
        ):
            eligible.append(row)
    if not eligible:
        return [], None
    latest = max(row.snapshot_date for row in eligible)
    levels = [
        float(row.level_price_btc)
        for row in eligible
        if row.snapshot_date == latest and row.level_price_btc is not None
    ]
    return levels, latest.isoformat()


def _near(spot: float, levels: list[float], pct: float) -> bool:
    if spot <= 0:
        return False
    return any(level > 0 and abs(spot - level) / spot < pct for level in levels)


def assemble(
    slot: datetime,
    history: MarketHistory,
    *,
    horizon_days: int = 5,
    model_version: str = "v1",
    fallback_weights: Optional[dict[str, float]] = None,
) -> ReconstructResult:
    """Costruisce le predizioni dello slot. Non scrive."""
    slot = as_utc(slot)
    fallback = dict(fallback_weights or DEFAULT_WEIGHTS)
    spot = price_at_slot(history.spot_prints, slot)
    etf = etf_inputs(history.flows, slot)
    funding = _closed_bars(history.funding_bars, slot)
    oi = _closed_bars(history.oi_bars, slot)
    longs = _closed_bars(history.long_short_bars, slot)
    liq = _closed_liq(history.liquidation_bars, slot)
    gex, gex_counts = select_gex(history.gex_rows, slot)
    chosen, weights_rejected = select_weights(history.weight_rows, slot)
    levels, barrier_day = select_barrier_levels(history.barrier_rows, slot)

    spot_dropped = sum(1 for ts, px in history.spot_prints if as_utc(ts) > slot and px > 0)
    funding_rate_8h = funding[-1].value if funding else None
    funding_ann = None
    if funding_rate_8h is not None:
        from src.flows.funding import annualize_funding_pct
        funding_ann = annualize_funding_pct(funding_rate_8h)

    oi_change = None
    if len(oi) >= 8 and oi[-8].value > 0:
        oi_change = (oi[-1].value - oi[-8].value) / oi[-8].value * 100.0

    degraded = ["gex", "put_call", "call_wall", "level"] if gex is None else []
    provenance: dict[str, Any] = {
        "slot": iso_z(slot),
        "reconstructed": True,
        "definition": {
            "etf_flow_3d_usd": "IBIT sum over calendar days D-3..D-1",
            "coinglass_bars": "daily bar usable only when open+duration <= slot",
            "spot": "last btc_usd index print with timestamp <= slot",
            "gex": (
                "gex_snapshots row for the slot date with timestamp <= slot; "
                "otherwise the component is omitted"
            ),
        },
        "sources": dict(history.notes),
        "degraded_inputs": degraded,
        "weights": {
            "version_id": chosen.id if chosen else None,
            "created_at": iso_z(chosen.created_at) if chosen else None,
            "source": "weight_versions" if chosen else "priors",
            "rejected_after_slot": weights_rejected,
        },
        "inputs": {
            "spot": {
                "price": None if spot is None else spot[1],
                "print_at": None if spot is None else iso_z(spot[0]),
                "dropped_after_slot": spot_dropped,
            },
            "etf_flow_3d_usd": etf["etf_flow_3d_usd"],
            "etf_window": etf["window"],
            "etf_sources": etf["sources"],
            "etf_dropped_on_or_after_slot": etf["dropped_on_or_after_slot"],
            "granger_lead_flow_usd": etf["granger_lead_flow_usd"],
            "granger_lead_date": etf["granger_lead_date"],
            "funding_rate_8h_pct": funding_rate_8h,
            "funding_rate_annualized_pct": funding_ann,
            "funding_bar_open": iso_z(funding[-1].open_ts) if funding else None,
            "funding_dropped_unclosed": len(history.funding_bars) - len(funding),
            "oi_change_7d_pct": oi_change,
            "oi_bar_open": iso_z(oi[-1].open_ts) if oi else None,
            "oi_dropped_unclosed": len(history.oi_bars) - len(oi),
            "long_short_ratio": longs[-1].value if longs else None,
            "long_short_bar_open": iso_z(longs[-1].open_ts) if longs else None,
            "long_short_dropped_unclosed": len(history.long_short_bars) - len(longs),
            "liquidations_long_24h_usd": liq[-1].long_usd if liq else None,
            "liquidations_short_24h_usd": liq[-1].short_usd if liq else None,
            "liquidations_bar_open": iso_z(liq[-1].open_ts) if liq else None,
            "liquidations_dropped_unclosed": len(history.liquidation_bars) - len(liq),
            "gex": None if gex is None else {
                "date": gex.day.isoformat(),
                "timestamp": iso_z(gex.timestamp),
                "total_net_gex": gex.total_net_gex,
                "gamma_flip_price": gex.gamma_flip_price,
                "call_wall": gex.call_wall,
                "max_pain": gex.max_pain,
                "put_call_ratio": gex.put_call_ratio,
            },
            "gex_same_day_after_slot": gex_counts["same_day_after_slot"],
            "gex_later_days": gex_counts["later_days"],
            "barrier_snapshot_date": barrier_day,
        },
    }

    def _empty(reason: str) -> ReconstructResult:
        return ReconstructResult(
            slot=slot, predictions=[], provenance=provenance,
            degraded_inputs=degraded, refused=reason,
            signal_score=None, signal_label=None,
            weights_version=chosen.id if chosen else None,
        )

    if spot is None:
        return _empty("spot_unavailable")
    if history.flows is None or not etf["known"]:
        return _empty("etf_flows_unavailable")

    spot_px = spot[1]
    near_barrier = _near(spot_px, levels, _BARRIER_PCT)
    near_call = False
    if (
        gex is not None
        and gex.total_net_gex > 0
        and gex.call_wall is not None
        and _near(spot_px, [gex.call_wall], _CALL_WALL_PCT)
    ):
        near_call = True
    provenance["inputs"]["near_barrier"] = near_barrier
    provenance["inputs"]["near_call_wall"] = near_call

    weights = dict(chosen.weights) if chosen else fallback
    version_id = chosen.id if chosen else None
    inputs = SignalInputs(
        gex_usd=None if gex is None else gex.total_net_gex,
        etf_flow_3d_usd=float(etf["etf_flow_3d_usd"]),
        funding_rate_annualized_pct=funding_ann,
        oi_change_7d_pct=oi_change,
        long_short_ratio=longs[-1].value if longs else None,
        put_call_ratio=None if gex is None else gex.put_call_ratio,
        liquidations_long_24h_usd=liq[-1].long_usd if liq else None,
        liquidations_short_24h_usd=liq[-1].short_usd if liq else None,
        granger_lead_flow_usd=etf["granger_lead_flow_usd"],
        spot_price=spot_px,
        near_active_barrier=near_barrier,
        near_call_wall=near_call,
    )
    try:
        signal = SignalModel(weights=weights).compute(inputs)
    except KeyError as exc:
        provenance["weights"]["error"] = f"missing component {exc}"
        return _empty("weights_incompatible")

    created_at = slot.strftime("%Y-%m-%dT%H:%M:%S")
    preds = build_dealer_flow_predictions(
        signal,
        spot_price=spot_px,
        gamma_flip=None if gex is None else gex.gamma_flip_price,
        max_pain=None if gex is None else gex.max_pain,
        total_net_gex=None if gex is None else gex.total_net_gex,
        horizon_days=horizon_days,
        created_at=created_at,
        weights_version=version_id,
        model_version=model_version,
    )
    meta = {
        "reconstructed": True,
        "degraded": bool(degraded),
        "degraded_inputs": list(degraded),
        "slot": iso_z(slot),
        "provenance": provenance,
    }
    for pred in preds:
        pred.metadata = json.loads(json.dumps(meta))
    return ReconstructResult(
        slot=slot,
        predictions=preds,
        provenance=provenance,
        degraded_inputs=degraded,
        refused=None,
        signal_score=signal.score,
        signal_label=signal.signal,
        weights_version=version_id,
    )


def prediction_to_public(pred: Prediction) -> dict[str, Any]:
    """Vista JSON della predizione, senza id (non ancora assegnato in dry-run)."""
    return {
        "created_at": pred.created_at,
        "date": pred.created_at[:10],
        "source": pred.source,
        "asset": pred.asset,
        "target_type": pred.target_type,
        "target_spec": pred.target_spec,
        "horizon_days": pred.horizon_days,
        "confidence": pred.confidence,
        "rationale": pred.rationale,
        "score_ref": pred.score_ref,
        "components": pred.components,
        "weights_version": pred.weights_version,
        "model_version": pred.model_version,
        "metadata": pred.metadata,
    }


def build_report(
    result: ReconstructResult,
    *,
    mode: str,
    cohort_exists: bool,
    inserted_ids: Optional[list[int]] = None,
    outcomes: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """Report stampato dal CLI. ``cohort_exists`` blocca la scrittura."""
    refused = result.refused
    if cohort_exists and refused is None:
        refused = "cohort_exists"
    show_preds = refused is None or refused == "cohort_exists"
    return {
        "mode": mode,
        "slot": iso_z(result.slot),
        "refused": refused,
        "cohort_exists": cohort_exists,
        "degraded_inputs": list(result.degraded_inputs),
        "weights_version": result.weights_version,
        "signal": None if result.signal_score is None else {
            "score": result.signal_score,
            "label": result.signal_label,
        },
        "predictions": [prediction_to_public(p) for p in result.predictions] if show_preds else [],
        "provenance": result.provenance,
        "inserted_ids": list(inserted_ids or []),
        "scored": [
            {"prediction_id": o.prediction_id, "hit": o.hit, "detail": o.detail}
            for o in (outcomes or [])
        ],
    }


def score_cohort(db: Any, price_provider: Any, ids: list[int], asof: Optional[datetime] = None) -> list:
    """Verifica solo gli id indicati. Le altre coorti mature non sono toccate."""
    from src.forecast.verifier import score_due_predictions

    if not ids:
        return []
    return score_due_predictions(db, price_provider, asof, only_ids=set(ids))


def assert_not_production_database() -> None:
    """Rifiuta se ``DATABASE_URL`` punta a Postgres.

    ``PredictionDB(path)`` ignora il path quando ``DATABASE_URL`` è impostata
    e scriverebbe sul database di produzione. Il tool non apre quel database,
    né in lettura né in scrittura.
    """
    from src.db import get_database_url

    if get_database_url():
        raise SystemExit(
            "DATABASE_URL è impostata: reconstruct_cohort non si collega a Postgres. "
            "Per un dry-run sul Mac, prefissa il comando con DATABASE_URL= "
            "e passa i file SQLite locali con --predictions-db / --gex-db / --notes-db."
        )


def read_sqlite(path: Path, sql: str, params: tuple = ()) -> list[dict]:
    """SELECT in sola lettura. File assente o tabella assente → lista vuota.

    Non crea il file e non applica migrazioni.
    """
    if not path.is_file():
        return []
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as exc:
        _log.warning("SQLite read-only %s: %s", path, exc)
        return []
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    except sqlite3.OperationalError as exc:
        _log.warning("SQLite %s: %s", path, exc)
        return []
    finally:
        conn.close()


def cohort_exists(path: Path, day: str, source: str = SOURCE) -> bool:
    """True se la coorte è già nel file SQLite. Non crea il file."""
    return bool(read_sqlite(path, _COHORT_SQL, (day, source)))


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gex_rows_from_dicts(rows: list[dict]) -> list[GexRow]:
    """Converte le righe SQL. Un timestamp illeggibile si salta (non è dimostrabile <= slot)."""
    parsed: list[GexRow] = []
    for row in rows:
        ts = parse_ts(row.get("timestamp"))
        raw_day = row.get("date")
        if ts is None or not raw_day:
            continue
        try:
            day = date.fromisoformat(str(raw_day)[:10])
            gex = float(row["total_net_gex"])
        except (TypeError, ValueError):
            continue
        parsed.append(GexRow(
            day=day,
            timestamp=ts,
            total_net_gex=gex,
            gamma_flip_price=_opt_float(row.get("gamma_flip_price")),
            call_wall=_opt_float(row.get("call_wall")),
            max_pain=_opt_float(row.get("max_pain")),
            put_call_ratio=_opt_float(row.get("put_call_ratio")),
        ))
    return parsed


def weight_rows_from_dicts(rows: list[dict]) -> list[WeightRow]:
    parsed: list[WeightRow] = []
    for row in rows:
        created = parse_ts(row.get("created_at"))
        if created is None:
            continue
        raw = row.get("weights")
        try:
            weights = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            continue
        parsed.append(WeightRow(
            id=int(row["id"]),
            created_at=created,
            weights={str(k): float(v) for k, v in weights.items()},
            status=str(row.get("status") or ""),
        ))
    return parsed


def barrier_rows_from_dicts(rows: list[dict]) -> list[BarrierRow]:
    parsed: list[BarrierRow] = []
    for row in rows:
        raw_day = row.get("snapshot_date")
        if not raw_day:
            continue
        try:
            day = date.fromisoformat(str(raw_day)[:10])
        except ValueError:
            continue
        parsed.append(BarrierRow(
            snapshot_date=day,
            created_at=parse_ts(row.get("created_at")),
            level_price_btc=_opt_float(row.get("level_price_btc")),
        ))
    return parsed


def load_local_snapshots(paths: LocalPaths) -> tuple[list[GexRow], list[WeightRow], list[BarrierRow]]:
    """Legge GEX, pesi e barriere in sola lettura."""
    gex = gex_rows_from_dicts(read_sqlite(paths.gex_db, _GEX_SQL, ("BTC",)))
    weights = weight_rows_from_dicts(
        read_sqlite(paths.predictions_db, _WEIGHTS_SQL, (SOURCE,))
    )
    barriers = barrier_rows_from_dicts(read_sqlite(paths.notes_db, _BARRIER_SQL))
    return gex, weights, barriers


def _series_daily_bars(series: Any) -> list[TimedBar]:
    """Serie daily (indice = apertura giornata) → barre chiuse il giorno dopo."""
    if series is None or len(series) == 0:
        return []
    import pandas as pd

    bars: list[TimedBar] = []
    for ts, val in series.items():
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        if pd.isna(val):
            continue
        open_ts = datetime(stamp.year, stamp.month, stamp.day, tzinfo=timezone.utc)
        bars.append(TimedBar(open_ts, float(val), _DAY_SECONDS))
    return bars


def _frame_liq_bars(frame: Any) -> list[LiqBar]:
    if frame is None or len(frame) == 0:
        return []
    import pandas as pd

    bars: list[LiqBar] = []
    for ts, row in frame.iterrows():
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        long_v = row["long_usd"] if "long_usd" in row else None
        short_v = row["short_usd"] if "short_usd" in row else None
        if pd.isna(long_v) or pd.isna(short_v):
            continue
        open_ts = datetime(stamp.year, stamp.month, stamp.day, tzinfo=timezone.utc)
        bars.append(LiqBar(open_ts, float(long_v), float(short_v), _DAY_SECONDS))
    return bars


def load_market_history(slot: datetime, paths: LocalPaths) -> MarketHistory:
    """Client storici + snapshot locali. I tagli allo slot li fa ``assemble``."""
    from src.assets import get_asset
    from src.gex.deribit_client import DeribitClient

    notes: dict[str, str] = {}
    prints: list[tuple[datetime, float]] = []
    chart_range = index_chart_range(slot)
    try:
        prints = DeribitClient().get_index_chart(get_asset("BTC").deribit_index, chart_range)
        notes["spot"] = (
            f"deribit public/get_index_chart_data btc_usd range={chart_range} "
            "(prints; last timestamp <= slot)"
        )
    except Exception as exc:
        _log.warning("Indice Deribit non disponibile: %s", exc)
        notes["spot"] = f"deribit index chart failed: {exc}"

    flows: Optional[list[FlowPrint]] = None
    try:
        from src.flows.scraper import FarsideScraper
        observed = FarsideScraper(asset="BTC").fetch_observed()
        if observed:
            flows = [
                FlowPrint(
                    day=row.date, ticker=row.ticker,
                    flow_usd=float(row.flow_usd), source=row.source or "",
                )
                for row in observed
            ]
            notes["etf"] = observed[0].source or "observed"
        else:
            notes["etf"] = "no observed IBIT prints"
    except Exception as exc:
        _log.warning("Flussi ETF non disponibili: %s", exc)
        notes["etf"] = f"failed: {exc}"

    funding: list[TimedBar] = []
    oi: list[TimedBar] = []
    longs: list[TimedBar] = []
    liq: list[LiqBar] = []
    try:
        from src.flows.coinglass_client import CoinGlassClient
        client = CoinGlassClient()
        if not client.has_api_key:
            notes["coinglass"] = "no COINGLASS_API_KEY; funding/OI/long-short/liquidations omitted"
        else:
            # Finestra che arriva fino a oggi e indietro oltre lo slot: il taglio
            # butta le barre non ancora chiuse alle 07:30.
            span = max(21, (datetime.now(timezone.utc).date() - slot.date()).days + 21)
            funding = _series_daily_bars(client.fetch_funding_rate_history(days=span))
            oi = _series_daily_bars(client.fetch_aggregated_oi_history(days=span))
            longs = _series_daily_bars(client.fetch_long_short_ratio(days=span))
            liq = _frame_liq_bars(client.fetch_liquidations(days=span))
            notes["coinglass"] = (
                "historical funding/OI/long-short/liquidations; "
                "daily bars kept only when closed by the slot"
            )
    except Exception as exc:
        _log.warning("CoinGlass storico non disponibile: %s", exc)
        notes["coinglass"] = f"failed: {exc}"

    gex_rows, weight_rows, barrier_rows = load_local_snapshots(paths)
    notes["gex"] = (
        "gex_snapshots only; no historical options chain. "
        "Same-day row used only when timestamp <= slot, else component skipped"
    )
    notes["barriers"] = "barrier_snapshots with snapshot_date before the slot (or created_at <= slot)"
    return MarketHistory(
        spot_prints=prints,
        flows=flows,
        funding_bars=funding,
        oi_bars=oi,
        long_short_bars=longs,
        liquidation_bars=liq,
        gex_rows=gex_rows,
        weight_rows=weight_rows,
        barrier_rows=barrier_rows,
        notes=notes,
    )


def _default_price_provider():
    """Stesso provider del verify giornaliero (barre UTC chiuse). Usato solo con ``--score``."""
    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.verifier import verification_price_provider

    return verification_price_provider(PriceFetcher())


def _model_version() -> str:
    gov = load_weights_config().get("governance", {})
    return str(gov.get("model_version") or "v1")


def _horizon() -> int:
    return int(get_settings().get("forecast", {}).get("horizon_days", 5))


def main(
    argv: Optional[list[str]] = None,
    *,
    loader: Any = None,
    price_provider: Any = None,
) -> int:
    """CLI. ``loader(slot, paths) -> MarketHistory`` sostituisce la rete nei test."""
    parser = argparse.ArgumentParser(
        description=(
            "Ricostruisce la coorte dealer-flow di una data passata usando solo "
            "dati noti allo slot predict (default 07:30 UTC). Non è schedulato."
        ),
    )
    parser.add_argument("--date", required=True, help="Giorno della coorte, YYYY-MM-DD")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Stampa predizioni e provenienza, non scrive")
    mode.add_argument("--write", action="store_true", help="Inserisce solo la coorte mancante")
    parser.add_argument(
        "--score", action="store_true",
        help="Dopo --write, verifica solo gli id inseriti (non le altre coorti mature)",
    )
    parser.add_argument("--predictions-db", required=True, help="SQLite con predictions e weight_versions")
    parser.add_argument("--gex-db", help="SQLite con gex_snapshots (default: data/structured_notes.db)")
    parser.add_argument("--notes-db", help="SQLite con barrier_snapshots (default: data/structured_notes.db)")
    parser.add_argument("--slot-time", help="HH:MM UTC. Default: forecast.schedule in settings")
    args = parser.parse_args(argv)

    if args.score and not args.write:
        print("error: --score richiede --write", file=sys.stderr)
        return 2

    try:
        assert_not_production_database()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        day = date.fromisoformat(args.date)
    except ValueError:
        print(f"error: data non valida {args.date}", file=sys.stderr)
        return 2

    if args.slot_time:
        try:
            hour_s, minute_s = args.slot_time.split(":")
            slot = slot_on(day, int(hour_s), int(minute_s))
        except ValueError:
            print(f"error: --slot-time atteso HH:MM, ricevuto {args.slot_time}", file=sys.stderr)
            return 2
    else:
        slot = slot_on(day)

    notes_default = PROJECT_ROOT / "data" / "structured_notes.db"
    paths = LocalPaths(
        predictions_db=Path(args.predictions_db),
        gex_db=Path(args.gex_db) if args.gex_db else notes_default,
        notes_db=Path(args.notes_db) if args.notes_db else notes_default,
    )
    load = loader or load_market_history
    try:
        history = load(slot, paths)
        result = assemble(
            slot, history,
            horizon_days=_horizon(),
            model_version=_model_version(),
            fallback_weights=dict(DEFAULT_WEIGHTS),
        )
    except Exception as exc:
        _log.exception("reconstruct fallito")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    exists = cohort_exists(paths.predictions_db, day.isoformat(), SOURCE)
    mode_name = "write" if args.write else "dry-run"
    if result.refused or exists:
        print(json.dumps(
            build_report(result, mode=mode_name, cohort_exists=exists),
            ensure_ascii=False, indent=2,
        ))
        return 2

    if not args.write:
        print(json.dumps(
            build_report(result, mode="dry-run", cohort_exists=False),
            ensure_ascii=False, indent=2,
        ))
        return 0

    from src.forecast.prediction_db import CohortExistsError, PredictionDB

    db = PredictionDB(paths.predictions_db)
    try:
        ids = db.insert_cohort(result.predictions)
    except CohortExistsError:
        print(json.dumps(
            build_report(result, mode="write", cohort_exists=True),
            ensure_ascii=False, indent=2,
        ))
        return 2

    outcomes = None
    if args.score:
        provider = price_provider or _default_price_provider()
        outcomes = score_cohort(db, provider, ids, datetime.now(timezone.utc))
    print(json.dumps(
        build_report(
            result, mode="write", cohort_exists=False,
            inserted_ids=ids, outcomes=outcomes,
        ),
        ensure_ascii=False, indent=2,
    ))
    return 0
