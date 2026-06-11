"""Utility per analisi barriere: clustering, segno direzionale, confluenza con GEX."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.config import get_settings

_log = logging.getLogger(__name__)


# ─── Dataclass output ─────────────────────────────────────────────────────────


@dataclass
class BarrierCluster:
    """Gruppo di barriere vicine tra loro.

    Attributes:
        barriers: lista di dict delle barriere nel cluster.
        mean_price_btc: prezzo medio BTC del cluster.
        total_notional_usd: somma nozionali delle note associate.
        dominant_type: tipo di barriera prevalente (knock_in, autocall, ...).
        sign: segno direzionale (bullish/bearish/neutral).
        n_barriers: numero barriere nel cluster.
        distance_to_spot_pct: distanza % dal prezzo spot.
    """
    barriers: list[dict]
    mean_price_btc: float
    total_notional_usd: float
    dominant_type: str
    sign: str
    n_barriers: int
    distance_to_spot_pct: float


# ─── Threshold unificata ─────────────────────────────────────────────────────


def get_proximity_pct() -> float:
    """Restituisce la soglia di prossimità unificata da settings.yaml.

    Usata ovunque per definire "near barrier".
    """
    cfg = get_settings()
    return float(cfg.get("analytics", {}).get("barrier_proximity_pct", 2.0))


# ─── Segno direzionale ───────────────────────────────────────────────────────


def barrier_sign(barrier_type: str, level_price_btc: float, spot_price: float) -> str:
    """Determina il segno direzionale di una barriera.

    Args:
        barrier_type: knock_in, autocall, buffer, knock_out.
        level_price_btc: prezzo BTC della barriera.
        spot_price: prezzo spot BTC corrente.

    Returns:
        "bullish" | "bearish" | "neutral".
    """
    if level_price_btc <= 0 or spot_price <= 0:
        return "neutral"

    above = level_price_btc > spot_price

    # Barriere rialziste: il dealer deve COMPRARE per coprirsi
    if barrier_type == "autocall" and above:
        return "bullish"
    if barrier_type == "knock_out" and above:
        return "bullish"

    # Barriere ribassiste: il dealer deve VENDERE per coprirsi
    if barrier_type == "knock_in" and not above:
        return "bearish"
    if barrier_type == "buffer" and not above:
        return "bearish"

    # Caso atipico (es. knock-in sopra spot = autocall-like)
    return "neutral"


# ─── Clustering ──────────────────────────────────────────────────────────────


def detect_clusters(
    barriers: list[dict],
    spot_price: float,
    proximity_pct: Optional[float] = None,
) -> list[BarrierCluster]:
    """Raggruppa barriere vicine tra loro in cluster.

    Ogni barriera viene assegnata al primo cluster il cui prezzo medio
    dista meno di ``proximity_pct``% dal suo ``level_price_btc``.

    Args:
        barriers: lista di dict da StructuredNotesDB.get_active_barriers().
        spot_price: prezzo spot BTC corrente.
        proximity_pct: soglia % per considerare due barriere nello stesso cluster
            (default: da settings.yaml barrier_proximity_pct).

    Returns:
        list[BarrierCluster]: cluster ordinati per prezzo decrescente.
    """
    if not barriers:
        return []

    pct = proximity_pct if proximity_pct is not None else get_proximity_pct()
    valid = [b for b in barriers if (b.get("level_price_btc") or 0) > 0]
    valid.sort(key=lambda b: b["level_price_btc"])

    clusters: list[list[dict]] = []

    for b in valid:
        price = b["level_price_btc"]
        placed = False
        for cl in clusters:
            mean_p = sum(c["level_price_btc"] for c in cl) / len(cl)
            if mean_p > 0 and abs(price - mean_p) / mean_p * 100 < pct:
                cl.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])

    result: list[BarrierCluster] = []
    for cl in clusters:
        prices = [c["level_price_btc"] for c in cl]
        mean_p = sum(prices) / len(prices)
        notional = sum(
            c.get("notional_usd") or 0.0
            for c in cl
        )
        types = [c.get("barrier_type", "unknown") for c in cl]
        dominant = max(set(types), key=types.count)
        sign = barrier_sign(dominant, mean_p, spot_price)
        dist = (mean_p - spot_price) / spot_price * 100 if spot_price > 0 else 0.0
        result.append(BarrierCluster(
            barriers=cl,
            mean_price_btc=mean_p,
            total_notional_usd=notional,
            dominant_type=dominant,
            sign=sign,
            n_barriers=len(cl),
            distance_to_spot_pct=round(dist, 2),
        ))

    result.sort(key=lambda c: c.mean_price_btc, reverse=True)
    return result


# ─── Confluenza con GEX ──────────────────────────────────────────────────────


def compute_confluence(
    clusters: list[BarrierCluster],
    put_wall: Optional[float],
    call_wall: Optional[float],
    gamma_flip: Optional[float],
    confluence_pct: float = 1.0,
) -> list[dict]:
    """Identifica convergenze tra cluster di barriere e livelli GEX.

    Args:
        clusters: cluster da detect_clusters().
        put_wall: prezzo del put wall GEX.
        call_wall: prezzo del call wall GEX.
        gamma_flip: prezzo del gamma flip point GEX.
        confluence_pct: soglia % per considerare due livelli coincidenti.

    Returns:
        list[dict] con campi:
            - cluster_mean: prezzo medio cluster
            - cluster_sign: bullish/bearish
            - cluster_notional: nozionale totale
            - gex_level: livello GEX corrispondente
            - gex_level_name: "put_wall", "call_wall", "gamma_flip"
            - distance_pct: distanza % tra cluster e GEX level
            - confluence_type: "bearish_reinforced" | "bullish_reinforced" | "mixed"
    """
    if not clusters:
        return []

    confluence: list[dict] = []

    for cl in clusters:
        for gex_price, gex_name in [
            (put_wall, "put_wall"),
            (call_wall, "call_wall"),
            (gamma_flip, "gamma_flip"),
        ]:
            if not gex_price or gex_price <= 0:
                continue
            if cl.mean_price_btc <= 0:
                continue
            dist = abs(cl.mean_price_btc - gex_price) / gex_price * 100
            if dist <= confluence_pct:
                if cl.sign == "bearish" and gex_name == "put_wall":
                    ctype = "bearish_reinforced"
                elif cl.sign == "bullish" and gex_name == "call_wall":
                    ctype = "bullish_reinforced"
                else:
                    ctype = "mixed"
                confluence.append({
                    "cluster_mean_price_btc": round(cl.mean_price_btc, 1),
                    "cluster_sign": cl.sign,
                    "cluster_notional_usd": round(cl.total_notional_usd, 0),
                    "cluster_n_barriers": cl.n_barriers,
                    "gex_level_price": round(gex_price, 1),
                    "gex_level_name": gex_name,
                    "distance_pct": round(dist, 2),
                    "confluence_type": ctype,
                })

    return confluence


def barrier_confluence_scores(
    confluence: list[dict],
) -> tuple[float, float]:
    """Calcola score bearish e bullish dalle confluenze.

    Args:
        confluence: output di compute_confluence().

    Returns:
        (bearish_score, bullish_score): float 0-1, quanto pesa la confluence
        in direzione ribassista/rialzista.
    """
    bearish_weight = 0.0
    bullish_weight = 0.0

    for c in confluence:
        notional_b = c.get("cluster_notional_usd") or 0.0
        # Normalizza notional: 0→0, 200M→1.0
        w = min(1.0, notional_b / 200e6)
        n_bar = c.get("cluster_n_barriers") or 1
        w *= min(1.5, 1.0 + (n_bar - 1) * 0.2)

        if c["confluence_type"] == "bearish_reinforced":
            bearish_weight = max(bearish_weight, w)
        elif c["confluence_type"] == "bullish_reinforced":
            bullish_weight = max(bullish_weight, w)

    bearish_score = min(1.0, bearish_weight)
    bullish_score = min(1.0, bullish_weight)

    return bearish_score, bullish_score
