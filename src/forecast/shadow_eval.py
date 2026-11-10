"""Metriche pre-registrate di dealer_flow contro dealer_flow_v2.

Il campione di decisione è solo le cohort con data >= start (default
2026-11-10) già maturate. Il replay sulle date precedenti è indicativo:
le regole sono state scelte dopo aver visto quel campione, e non entra
nel verdetto.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from src.forecast.models import TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB, Prediction
from src.forecast.sources.dealer_flow_v2 import (
    ABSTAIN_CONFIDENCE,
    SOURCE,
    calibrate_prob,
    filter_matured_prob_rows,
    flat_band_pct_from_returns,
    v2_from_stored_v1,
)

PREREG_DATE = "2026-11-10"
MIN_COHORTS = 40
MIN_DIRECTION = 40
MIN_PROB = 40
MIN_LEVEL = 20
_Z = 1.96


@dataclass
class Proportion:
    name: str
    hits: int
    n: int
    rate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    note: str = ""


@dataclass
class ShadowReport:
    start: str
    n_cohorts: int
    coverage: Optional[float]
    decision: str
    lines: list[str] = field(default_factory=list)


def wilson(hits: int, n: int, z: float = _Z) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Intervallo di Wilson al 95% per una proporzione. (rate, low, high)."""
    if n <= 0:
        return None, None, None
    p = hits / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def proportion(name: str, hits: int, n: int, note: str = "") -> Proportion:
    rate, low, high = wilson(hits, n)
    return Proportion(name, hits, n, rate, low, high, note)


def _spec(row: dict) -> dict:
    raw = row.get("target_spec") or {}
    if isinstance(raw, str):
        raw = json.loads(raw) if raw else {}
    return raw if isinstance(raw, dict) else {}


def _day(row: dict) -> str:
    return str(row.get("date") or row.get("created_at") or "")[:10]


def _ret(row: dict) -> Optional[float]:
    if row.get("realized_return") is None:
        return None
    return float(row["realized_return"])


def _band(row: dict) -> float:
    return float(_spec(row).get("flat_band_pct", 1.0)) / 100.0


def realized_class(ret: float, band: float) -> str:
    if ret > band:
        return "up"
    if ret < -band:
        return "down"
    return "flat"


def _scored(rows: Sequence[dict], target: str) -> list[dict]:
    return [r for r in rows if r.get("target_type") == target and r.get("hit") is not None
            and _ret(r) is not None]


def direction_block(rows: Sequence[dict], label: str) -> dict[str, Any]:
    """Hit del modello (esiti già scritti) e baseline always-flat / persistence."""
    scored = sorted(_scored(rows, TARGET_DIRECTION), key=_day)
    hits = sum(int(r["hit"]) for r in scored)
    model = proportion(f"{label} direction", hits, len(scored))
    classes = [realized_class(_ret(r), _band(r)) for r in scored]  # type: ignore[arg-type]
    flat_hits = sum(1 for c in classes if c == "flat")
    always_flat = proportion(f"{label} always-flat", flat_hits, len(classes))
    pers_hits = pers_n = 0
    for i in range(1, len(classes)):
        pers_n += 1
        pers_hits += int(classes[i - 1] == classes[i])
    persistence = proportion(f"{label} persistence", pers_hits, pers_n)
    best_name, best_rate = _best_baseline(
        (("always-flat", always_flat.rate), ("persistence", persistence.rate)),
    )
    beats = None
    if model.n >= MIN_DIRECTION and model.ci_low is not None and best_rate is not None:
        beats = model.ci_low > best_rate
    return {
        "model": model, "always_flat": always_flat, "persistence": persistence,
        "best_name": best_name, "best_rate": best_rate, "beats": beats,
    }


def prob_block(rows: Sequence[dict], label: str) -> dict[str, Any]:
    scored = sorted(_scored(rows, TARGET_PROB), key=_day)
    briers = []
    expand = []
    ys: list[int] = []
    for row in scored:
        y = 1 if _ret(row) > 0 else 0  # type: ignore[operator]
        if row.get("brier") is not None:
            briers.append(float(row["brier"]))
        else:
            p = float(_spec(row).get("p", 0.5))
            briers.append((p - y) ** 2)
        pi = (sum(ys) / len(ys)) if ys else 0.5
        expand.append((pi - y) ** 2)
        ys.append(y)
    n = len(briers)
    mean = sum(briers) / n if n else None
    ref_half = 0.25
    ref_exp = sum(expand) / n if n else None
    se = None
    if n > 1 and mean is not None:
        var = sum((b - mean) ** 2 for b in briers) / (n - 1)
        se = math.sqrt(var / n)
    upper = mean + _Z * se if mean is not None and se is not None else None
    best_name, best_rate = _best_baseline(
        (("constant 0.5", ref_half), ("expanding base rate", ref_exp)),
        lower_is_better=True,
    )
    beats = None
    if n >= MIN_PROB and upper is not None and best_rate is not None:
        beats = upper < best_rate
    return {
        "n": n, "mean_brier": mean, "ci_high": upper,
        "brier_half": ref_half, "brier_expanding": ref_exp,
        "best_name": best_name, "best_rate": best_rate, "beats": beats,
        "label": label,
    }


def level_block(
    rows: Sequence[dict], prices, *, before: str, label: str,
) -> dict[str, Any]:
    scored = _scored(rows, TARGET_LEVEL)
    hits = sum(int(r["hit"]) for r in scored)
    model = proportion(f"{label} level", hits, len(scored))
    rates: list[float] = []
    for row in scored:
        spec = _spec(row)
        ref = spec.get("ref_price") or row.get("ref_price")
        level = spec.get("level_price")
        if not ref or not level or float(ref) <= 0:
            continue
        distance = abs(float(level) / float(ref) - 1.0)
        side = "above" if float(level) >= float(ref) else "below"
        rate = historical_touch_rate(prices, distance, side, before=before)
        if rate is not None:
            rates.append(rate)
    baseline = sum(rates) / len(rates) if rates else None
    beats = None
    if model.n >= MIN_LEVEL and model.ci_low is not None and baseline is not None:
        beats = model.ci_low > baseline
    return {"model": model, "baseline": baseline, "n_baseline": len(rates), "beats": beats, "label": label}


def historical_touch_rate(prices, distance: float, side: str, *, before: str, horizon: int = 5) -> Optional[float]:
    """Frequenza storica, solo barre con data < ``before``, di un tocco a distanza ``distance``.

    Finestra come il verifier: le ``horizon`` barre successive al close di riferimento.
    """
    if prices is None or getattr(prices, "empty", True):
        return None
    frame = prices
    if "close" not in frame.columns:
        return None
    idx = frame.index
    if before:
        frame = frame.loc[idx < _ts(before)]
    if len(frame) < horizon + 30:
        return None
    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float) if "high" in frame.columns else closes
    lows = frame["low"].to_numpy(dtype=float) if "low" in frame.columns else closes
    hits = n = 0
    for i in range(len(frame) - horizon):
        ref = closes[i]
        if not math.isfinite(ref) or ref <= 0:
            continue
        hi = float(np_max(highs[i + 1:i + 1 + horizon]))
        lo = float(np_min(lows[i + 1:i + 1 + horizon]))
        level = ref * (1.0 + distance) if side == "above" else ref * (1.0 - distance)
        hits += int(lo <= level <= hi)
        n += 1
    if n < 30:
        return None
    return hits / n


def coverage(v2_rows: Sequence[dict]) -> Optional[float]:
    """Giorni con una direction / giorni con una qualunque riga v2.

    L'astensione non scrive la direction: il rapporto è la copertura.
    """
    days = {_day(r) for r in v2_rows if _day(r)}
    if not days:
        return None
    directed = {_day(r) for r in v2_rows if r.get("target_type") == TARGET_DIRECTION and _day(r)}
    return len(directed & days) / len(days)


def decide(n_cohorts: int, direction: dict, prob: dict, level: dict) -> str:
    """``not_yet`` finché un head non ha il minimo. ``v2_better`` solo se tutti battono il baseline."""
    if n_cohorts < MIN_COHORTS:
        return "not_yet"
    heads = (direction["beats"], prob["beats"], level["beats"])
    if any(flag is None for flag in heads):
        return "not_yet"
    if all(heads):
        return "v2_better"
    return "v2_not_better"


def evaluate(
    v1_rows: Sequence[dict],
    v2_rows: Sequence[dict],
    *,
    start: str = PREREG_DATE,
    prices=None,
) -> ShadowReport:
    """Metriche sul campione forward. Le date sono quelle di v2 mature >= start."""
    v2_mature = [r for r in v2_rows if _day(r) >= start and r.get("hit") is not None]
    days = sorted({_day(r) for r in v2_mature if _day(r)})
    v1_same = [r for r in v1_rows if _day(r) in set(days)]
    v2_same = [r for r in v2_rows if _day(r) in set(days)]
    d1 = direction_block(v1_same, "v1")
    d2 = direction_block(v2_same, "v2")
    p1 = prob_block(v1_same, "v1")
    p2 = prob_block(v2_same, "v2")
    lv1 = level_block(v1_same, prices, before=start, label="v1")
    lv2 = level_block(v2_same, prices, before=start, label="v2")
    cov = coverage(v2_same)
    decision = decide(len(days), d2, p2, lv2)
    lines = [
        f"sample: matured v2 cohorts with date>={start}: {len(days)} "
        f"(minimum {MIN_COHORTS})",
        f"v2 direction coverage (cohorts with a direction call): {_pct(cov)}",
        _fmt_direction(d1),
        _fmt_direction(d2),
        _fmt_prob(p1),
        _fmt_prob(p2),
        _fmt_level(lv1),
        _fmt_level(lv2),
        f"decision: {decision}",
    ]
    if decision == "not_yet":
        lines.append(
            "not yet: need >=40 matured cohorts and every head at its minimum "
            f"(direction/prob {MIN_DIRECTION}, level {MIN_LEVEL}) with a baseline. "
            "Abstentions stay out of the direction hit rate; they are the coverage."
        )
    return ShadowReport(start, len(days), cov, decision, lines)


def replay_indicative(
    v1_rows: Sequence[dict],
    *,
    start: str = PREREG_DATE,
    all_rows: Optional[Sequence[dict]] = None,
    log_returns_before: Optional[dict[str, list[float]]] = None,
) -> list[str]:
    """Applica le regole v2 alle cohort v1 precedenti a ``start``. Non è il verdetto.

    Usa il realized_return già salvato (può essere quello della barra parziale).
    La banda e la Platt di ogni data usano solo dati anteriori a quella data.
    """
    lines = [
        "IN-SAMPLE / INDICATIVE ONLY — not used for the decision.",
        "Rules were chosen after seeing this history. Do not read a win here as evidence.",
    ]
    past = [r for r in v1_rows if _day(r) and _day(r) < start and r.get("source", "dealer_flow") == "dealer_flow"]
    by_day: dict[str, list[dict]] = {}
    for row in past:
        by_day.setdefault(_day(row), []).append(row)
    replay_rows: list[dict] = []
    history_pool = list(all_rows or v1_rows)
    for day in sorted(by_day):
        cohort = by_day[day]
        anchor = next((r for r in cohort if r.get("target_type") == TARGET_DIRECTION), cohort[0])
        score = anchor.get("score_ref")
        if score is None:
            continue
        when = datetime.fromisoformat(str(anchor["created_at"])).replace(tzinfo=timezone.utc)
        hist = filter_matured_prob_rows(history_pool, when)
        p_up, method = calibrate_prob(float(score), hist)
        rets = (log_returns_before or {}).get(day, [])
        horizon = int(anchor.get("horizon_days") or 5)
        band = flat_band_pct_from_returns(rets, horizon)
        stored = []
        for row in cohort:
            try:
                stored.append(Prediction.from_row(row))
            except (KeyError, TypeError, ValueError):
                continue
        if not stored:
            continue
        built = v2_from_stored_v1(
            stored, p_up=p_up, flat_band_pct=band,
            calibration=method, calibration_n=len(hist),
        )
        ret_row = next((r for r in cohort if _ret(r) is not None), None)
        if ret_row is None:
            continue
        ret = _ret(ret_row)
        for pred in built:
            replay_rows.append(_fake_outcome(pred, ret, day))
    if not replay_rows:
        lines.append("replay: no historical v1 cohorts with a score and a realized return.")
        return lines
    # Copertura sul replay: giorni con direction / giorni con prob.
    lines.append(f"replay cohorts: {len({_day(r) for r in replay_rows})} (dates < {start})")
    lines.append(f"replay direction coverage: {_pct(coverage(replay_rows))}")
    d = direction_block(replay_rows, "replay v2")
    p = prob_block(replay_rows, "replay v2")
    lines.append(_fmt_direction(d))
    lines.append(_fmt_prob(p))
    lines.append(
        f"abstain rule used: confidence < {ABSTAIN_CONFIDENCE:.2f} "
        f"(|score-50|/50). A score of 50 abstains; 55 and 45 do not."
    )
    return lines


def _fake_outcome(pred, ret: float, day: str) -> dict:
    """Esito sintetico dal return salvato. Il level non è ricostruibile senza high/low."""
    spec = pred.target_spec
    row = {
        "date": day,
        "created_at": pred.created_at,
        "source": pred.source,
        "target_type": pred.target_type,
        "target_spec": spec,
        "horizon_days": pred.horizon_days,
        "score_ref": pred.score_ref,
        "realized_return": ret,
        "hit": None,
        "brier": None,
    }
    if pred.target_type == TARGET_DIRECTION:
        band = float(spec.get("flat_band_pct", 1.0)) / 100.0
        row["hit"] = int(realized_class(ret, band) == spec.get("direction"))
    elif pred.target_type == TARGET_PROB:
        y = 1.0 if ret > 0 else 0.0
        p = float(spec.get("p", 0.5))
        row["brier"] = (p - y) ** 2
        row["hit"] = int((p > 0.5) == (y == 1.0))
    elif pred.target_type == TARGET_LEVEL:
        row["hit"] = None  # senza high/low il tocco non si ricostruisce
        row["realized_return"] = None
    return row


def format_report(report: ShadowReport, replay_lines: Optional[Sequence[str]] = None) -> str:
    lines = list(report.lines)
    if replay_lines:
        lines.append("")
        lines.extend(replay_lines)
    return "\n".join(lines)


def _best_baseline(pairs, *, lower_is_better: bool = False):
    usable = [(name, rate) for name, rate in pairs if rate is not None]
    if not usable:
        return None, None
    name, rate = (min if lower_is_better else max)(usable, key=lambda item: item[1])
    return name, rate


def _fmt_direction(block: dict) -> str:
    model: Proportion = block["model"]
    flat: Proportion = block["always_flat"]
    pers: Proportion = block["persistence"]
    return (
        f"{model.name}: hit {_pct(model.rate)} "
        f"Wilson [{_pct(model.ci_low)}, {_pct(model.ci_high)}] n={model.n} | "
        f"always-flat {_pct(flat.rate)} n={flat.n} | "
        f"persistence {_pct(pers.rate)} n={pers.n} | "
        f"best={block['best_name']} {_pct(block['best_rate'])} beats={block['beats']}"
    )


def _fmt_prob(block: dict) -> str:
    return (
        f"{block['label']} prob: Brier {_num(block['mean_brier'])} "
        f"CI upper {_num(block['ci_high'])} n={block['n']} | "
        f"vs 0.5 {_num(block['brier_half'])} | "
        f"vs expanding {_num(block['brier_expanding'])} | "
        f"best={block['best_name']} {_num(block['best_rate'])} beats={block['beats']}"
    )


def _fmt_level(block: dict) -> str:
    model: Proportion = block["model"]
    return (
        f"{block['label']} level: hit {_pct(model.rate)} "
        f"Wilson [{_pct(model.ci_low)}, {_pct(model.ci_high)}] n={model.n} | "
        f"historical touch at same distance {_pct(block['baseline'])} "
        f"(n_with_baseline={block['n_baseline']}) beats={block['beats']}"
    )


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _ts(day: str):
    import pandas as pd
    return pd.Timestamp(day)


def np_max(values) -> float:
    import numpy as np
    return float(np.max(values)) if len(values) else float("nan")


def np_min(values) -> float:
    import numpy as np
    return float(np.min(values)) if len(values) else float("nan")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI. Legge il DB, non lo modifica."""
    import argparse
    import sys
    from pathlib import Path

    from src.db import get_database_url

    parser = argparse.ArgumentParser(description="Metriche v1 / v2 / baseline sulle cohort maturate.")
    parser.add_argument("--db", required=True, help="SQLite con predictions e prices (data/runtime.db)")
    parser.add_argument("--start", default=PREREG_DATE, help="Prima data del campione di decisione")
    parser.add_argument("--replay", action="store_true", help="Aggiunge il replay in-sample, escluso dal verdetto")
    parser.add_argument("--allow-postgres", action="store_true")
    args = parser.parse_args(argv)
    if get_database_url() and not args.allow_postgres:
        print(
            "DATABASE_URL è impostata: compare_v1_v2 non si collega a Postgres. "
            "Sul Mac: DATABASE_URL= python scripts/compare_v1_v2.py --db data/runtime.db. "
            "Per Postgres serve --allow-postgres.",
            file=sys.stderr,
        )
        return 2
    from src.flows.price_fetcher import PriceFetcher
    from src.forecast.prediction_db import PredictionDB

    db_path = Path(args.db)
    db = PredictionDB(db_path)
    rows = db.get_with_outcomes(days=3650)
    v1 = [r for r in rows if r.get("source") == "dealer_flow"]
    v2 = [r for r in rows if r.get("source") == SOURCE]
    prices = load_prices(PriceFetcher(db_path=db_path), date.fromisoformat(args.start))
    report = evaluate(v1, v2, start=args.start, prices=prices)
    replay = replay_indicative(v1, start=args.start, all_rows=rows) if args.replay else None
    print(format_report(report, replay))
    return 0


def load_prices(fetcher, before: date):
    """Barre fino al giorno prima di ``before`` (il baseline non vede il campione)."""
    if before <= date(2010, 1, 1):
        return None
    frame = fetcher._load_from_db("BTC-USD", date(2010, 1, 1), before - timedelta(days=1))
    if frame is None or frame.empty:
        return None
    from src.flows.price_fetcher import complete_bars
    frame = complete_bars(frame)
    if frame.empty:
        return None
    return frame
