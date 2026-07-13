"""Calibrazione human-in-the-loop dei pesi del SignalModel dealer-flow.

Dai join predictions+outcomes calcola metriche di performance (hit-rate + test binomiale,
Brier, IC delle componenti) e **propone** un nuovo set di pesi entro guardrail rigorosi:
shrinkage verso i prior, vincolo Σ=1, cap del passo per ciclo, gate sul campione minimo
(più alto per le componenti GEX, validate in avanti). **Non attiva mai nulla**: la proposta
resta `proposed` finché Stefano non la attiva (workflow Tuning).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.analytics.signal_model import WEIGHTS as DEFAULT_WEIGHTS
from src.config import setup_logging
from src.forecast.models import TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB
from src.forecast.sources.dealer_flow import SOURCE as DEALER_FLOW

_log = setup_logging("forecast.calibration")

_WEIGHTS_YAML = Path(__file__).resolve().parents[2] / "config" / "weights.yaml"


# ─── Config ──────────────────────────────────────────────────────────────────

def load_weights_config() -> dict:
    """Carica config/weights.yaml (prior + guardrail). Fallback ai default se assente."""
    if _WEIGHTS_YAML.exists():
        with open(_WEIGHTS_YAML, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {
        "priors": dict(DEFAULT_WEIGHTS),
        "calibration": {
            "min_scored": 30, "min_scored_gex": 40, "max_step": 0.05,
            "shrinkage": 0.30, "learning_rate": 0.50, "min_oos_improvement": 0.0,
        },
        "governance": {"kill_switch": False, "freeze_weights": True},
    }


# ─── Statistiche ─────────────────────────────────────────────────────────────

def binomial_p_value(hits: int, n: int, p0: float = 0.5) -> Optional[float]:
    """p-value a una coda (H1: tasso > p0) del test binomiale esatto. None se n=0."""
    if n <= 0:
        return None
    # P(X >= hits) con X ~ Binom(n, p0)
    tail = sum(math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k) for k in range(hits, n + 1))
    return float(min(1.0, tail))


def spearman_ic(x: list[float], y: list[float]) -> Optional[float]:
    """Information coefficient = correlazione di rango (Spearman) tra x e y. None se insuff."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3 or np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


# ─── Metriche per source ─────────────────────────────────────────────────────

def compute_source_metrics(rows: list[dict]) -> dict:
    """Aggrega le righe predictions+outcomes (di una source) in metriche per target_type
    + IC delle componenti vs return realizzato."""
    by_type: dict[str, dict] = {}
    for tt in (TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB):
        sub = [r for r in rows if r["target_type"] == tt and r.get("hit") is not None]
        n = len(sub)
        hits = sum(int(r["hit"]) for r in sub)
        briers = [r["brier"] for r in sub if r.get("brier") is not None]
        by_type[tt] = {
            "scored": n,
            "hits": hits,
            "hit_rate": round(hits / n, 3) if n else None,
            "binomial_p": round(binomial_p_value(hits, n), 4) if n else None,
            "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
        }

    # IC componenti: correla il valore normalizzato della componente col return realizzato.
    scored = [r for r in rows if r.get("hit") is not None and r.get("realized_return") is not None]
    component_ic: dict[str, Optional[float]] = {}
    component_n: dict[str, int] = {}
    for comp in DEFAULT_WEIGHTS:
        xs, ys = [], []
        for r in scored:
            comps = json.loads(r["components"]) if r.get("components") else {}
            v = comps.get(comp)
            if v is not None:
                xs.append(float(v))
                ys.append(float(r["realized_return"]))
        component_n[comp] = len(xs)
        component_ic[comp] = spearman_ic(xs, ys)

    return {
        "total_scored": len(scored),
        "by_target_type": by_type,
        "component_ic": component_ic,
        "component_n": component_n,
    }


# ─── Proposta pesi (guardrail) ───────────────────────────────────────────────

def _renorm(w: dict[str, float]) -> dict[str, float]:
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 0 else dict(w)


def propose_weights(
    active: dict[str, float],
    component_ic: dict[str, Optional[float]],
    component_n: dict[str, int],
    cfg: dict,
) -> tuple[dict[str, float], str]:
    """Proietta i pesi attivi verso le componenti con IC positivo, entro i guardrail.

    Passi: tilt ∝ IC → renormalizza → cap del passo vs attivi → shrinkage verso prior → renormalizza.
    Le componenti con campione insufficiente (in particolare GEX) non vengono tiltate.
    """
    cal = cfg.get("calibration", {})
    priors = cfg.get("priors", dict(DEFAULT_WEIGHTS))
    lr = float(cal.get("learning_rate", 0.5))
    max_step = float(cal.get("max_step", 0.05))
    shrink = float(cal.get("shrinkage", 0.3))
    min_n = int(cal.get("min_scored", 30))
    min_n_gex = int(cal.get("min_scored_gex", 40))

    keys = list(priors.keys())
    active = {k: active.get(k, priors[k]) for k in keys}

    # Tilt verso IC positivo (componenti senza dati sufficienti restano neutre)
    tilted = {}
    notes = []
    for k in keys:
        ic = component_ic.get(k)
        n_req = min_n_gex if k == "gex" else min_n
        if ic is None or component_n.get(k, 0) < n_req:
            tilted[k] = active[k]
            continue
        tilted[k] = max(0.0, active[k] * (1.0 + lr * ic))
        notes.append(f"{k}: IC={ic:+.2f} (n={component_n.get(k,0)})")
    tilted = _renorm(tilted)

    # Cap del passo per peso rispetto agli attivi
    capped = {}
    for k in keys:
        delta = tilted[k] - active[k]
        if abs(delta) > max_step:
            delta = math.copysign(max_step, delta)
        capped[k] = max(0.0, active[k] + delta)
    capped = _renorm(capped)

    # Shrinkage verso i prior
    final = {k: (1 - shrink) * capped[k] + shrink * priors[k] for k in keys}
    final = _renorm(final)
    final = {k: round(v, 4) for k, v in final.items()}
    final = _renorm(final)

    rationale = (
        "; ".join(notes) if notes
        else "nessuna componente con campione/IC sufficiente: pesi quasi invariati"
    )
    return final, rationale


# ─── Report ──────────────────────────────────────────────────────────────────

@dataclass
class CalibrationReport:
    source: str
    metrics: dict
    active_weights: dict[str, float]
    proposed_weights: Optional[dict[str, float]]
    gate_ok: bool
    rationale: str
    notes: list[str] = field(default_factory=list)


def run_calibration(db, source: str = DEALER_FLOW, *, days: int = 180,
                    cfg: Optional[dict] = None) -> CalibrationReport:
    """Calcola metriche e (se il gate passa) propone una nuova versione pesi (status 'proposed').

    Non attiva nulla. Restituisce un CalibrationReport per il workflow Tuning.
    """
    cfg = cfg or load_weights_config()
    priors = cfg.get("priors", dict(DEFAULT_WEIGHTS))
    min_scored = int(cfg.get("calibration", {}).get("min_scored", 30))

    model_version = cfg.get("governance", {}).get("model_version", "v1")
    rows = db.get_with_outcomes(days=days, source=source, model_version=model_version)
    metrics = compute_source_metrics(rows)

    active_pair = db.get_active_weights(source)
    active = active_pair[1] if active_pair else dict(priors)

    notes: list[str] = []
    gate_ok = metrics["total_scored"] >= min_scored
    proposed: Optional[dict[str, float]] = None

    if not gate_ok:
        notes.append(
            f"GATE non superato: {metrics['total_scored']} esiti < min {min_scored}. "
            "Nessuna proposta (servono più dati in shadow)."
        )
        return CalibrationReport(source, metrics, active, None, False, "; ".join(notes), notes)

    proposed, rationale = propose_weights(
        active, metrics["component_ic"], metrics["component_n"], cfg
    )
    notes.append(f"Proposta generata su {metrics['total_scored']} esiti.")

    vid = db.insert_weight_version(
        source, proposed,
        rationale=f"calibrazione auto: {rationale}",
        activate=False,  # MAI auto-attivare
    )
    notes.append(f"Versione {vid} salvata come 'proposed' (attivazione manuale via Tuning).")

    return CalibrationReport(source, metrics, active, proposed, True, rationale, notes)
