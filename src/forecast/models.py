"""Modelli dati del forecast spine: Prediction e Outcome.

Una `Prediction` è una previsione *verificabile* prodotta da una source (es. dealer_flow):
ha un tipo di target, una specifica del target, un orizzonte e una confidence. Quando
l'orizzonte matura, il `verifier` produce un `Outcome` che la valuta.

Tre tipi di target (vedi `TARGET_*`):
- `direction` — direzione/regime su orizzonte (up | down | flat rispetto a un prezzo di riferimento).
- `level`     — un livello di prezzo (gamma flip / wall / max pain / barriera) da raggiungere o rispettare.
- `prob`      — un evento probabilistico (es. "BTC return positivo a H giorni") con probabilità p ∈ [0,1].

target_spec è un dict serializzato in JSON nel DB. Schema per tipo:
- direction: {"direction": "up|down|flat", "ref_price": float, "flat_band_pct": float}
- level:     {"level_name": str, "level_price": float, "mode": "reach|break|respect",
              "side": "above|below", "ref_price": float}
- prob:      {"event": str, "p": float}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ─── Costanti ────────────────────────────────────────────────────────────────

TARGET_DIRECTION = "direction"
TARGET_LEVEL = "level"
TARGET_PROB = "prob"
TARGET_TYPES = (TARGET_DIRECTION, TARGET_LEVEL, TARGET_PROB)

STATUS_OPEN = "open"
STATUS_SCORED = "scored"

DIR_UP = "up"
DIR_DOWN = "down"
DIR_FLAT = "flat"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ─── Prediction ──────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """Una previsione verificabile.

    Attributes:
        source: dominio che ha prodotto la previsione (es. "dealer_flow", "ema", "portfolio").
        asset: asset di riferimento (es. "BTC").
        target_type: uno di TARGET_TYPES.
        target_spec: dict con la specifica del target (schema dipende da target_type).
        horizon_days: orizzonte di verifica in giorni.
        confidence: confidence 0-1 della previsione.
        created_at: ISO UTC alla creazione (chiave di dedup con source/target_type/horizon).
        rationale: motivazione quant generata dalla source.
        counter_analysis: contro-analisi (red-team) — popolata nel daily-review.
        human_overlay: overlay di conoscenza di Stefano — popolato nel daily-review.
        score_ref: score del SignalModel al momento della creazione (per la calibrazione).
        components: contributi normalizzati delle componenti del segnale (per la calibrazione).
        weights_version: id della versione pesi attiva usata (per audit della calibrazione).
        status: STATUS_OPEN finché non verificata, poi STATUS_SCORED.
        id: chiave primaria (assegnata dal DB).
    """

    source: str
    asset: str
    target_type: str
    target_spec: dict[str, Any]
    horizon_days: int
    confidence: float
    created_at: str = field(default_factory=_utc_now_iso)
    rationale: str = ""
    counter_analysis: str = ""
    human_overlay: str = ""
    score_ref: Optional[float] = None
    components: dict[str, Optional[float]] = field(default_factory=dict)
    weights_version: Optional[int] = None
    model_version: str = "v1"
    status: str = STATUS_OPEN
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.target_type not in TARGET_TYPES:
            raise ValueError(f"target_type non valido: {self.target_type!r}")
        self.confidence = float(max(0.0, min(1.0, self.confidence)))

    @property
    def matures_at(self) -> datetime:
        """Istante (UTC) in cui l'orizzonte è maturo e la predizione è verificabile."""
        created = datetime.fromisoformat(self.created_at).replace(tzinfo=timezone.utc)
        from datetime import timedelta
        return created + timedelta(days=self.horizon_days)

    def is_due(self, asof: Optional[datetime] = None) -> bool:
        """True se la predizione è open e l'orizzonte è maturo."""
        now = asof or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return self.status == STATUS_OPEN and now >= self.matures_at

    def to_row(self) -> dict[str, Any]:
        """Serializza per l'inserimento in SQLite (dict JSON-encoded dove serve)."""
        return {
            "created_at": self.created_at,
            "date": self.created_at[:10],
            "source": self.source,
            "asset": self.asset,
            "target_type": self.target_type,
            "target_spec": json.dumps(self.target_spec, ensure_ascii=False),
            "horizon_days": int(self.horizon_days),
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "counter_analysis": self.counter_analysis,
            "human_overlay": self.human_overlay,
            "score_ref": self.score_ref,
            "components": json.dumps(self.components, ensure_ascii=False),
            "weights_version": self.weights_version,
            "model_version": self.model_version,
            "status": self.status,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Prediction":
        return cls(
            id=row.get("id"),
            created_at=row["created_at"],
            source=row["source"],
            asset=row["asset"],
            target_type=row["target_type"],
            target_spec=json.loads(row["target_spec"]) if row.get("target_spec") else {},
            horizon_days=int(row["horizon_days"]),
            confidence=float(row["confidence"]),
            rationale=row.get("rationale") or "",
            counter_analysis=row.get("counter_analysis") or "",
            human_overlay=row.get("human_overlay") or "",
            score_ref=row.get("score_ref"),
            components=json.loads(row["components"]) if row.get("components") else {},
            weights_version=row.get("weights_version"),
            model_version=row.get("model_version") or "v1",
            status=row.get("status", STATUS_OPEN),
        )


# ─── Outcome ─────────────────────────────────────────────────────────────────

@dataclass
class Outcome:
    """Esito di una predizione verificata.

    Attributes:
        prediction_id: FK alla predizione.
        hit: True se la previsione si è avverata (per direction/level), o se la
            probabilità era dalla parte giusta (per prob: (p>0.5) == evento).
        realized_return: return realizzato dell'asset sull'orizzonte (log o semplice, vedi verifier).
        realized_price: prezzo a fine orizzonte.
        ref_price: prezzo di riferimento usato per il confronto.
        signed_error: errore con segno (es. return per direction; (realizzato-target)/target per level).
        brier: Brier score per i target prob ((p - esito)²), altrimenti None.
        detail: descrizione human-readable dell'esito.
        scored_at: ISO UTC del calcolo dell'esito.
    """

    prediction_id: int
    hit: bool
    realized_return: Optional[float] = None
    realized_price: Optional[float] = None
    ref_price: Optional[float] = None
    signed_error: Optional[float] = None
    brier: Optional[float] = None
    detail: str = ""
    scored_at: str = field(default_factory=_utc_now_iso)

    def to_row(self) -> dict[str, Any]:
        return {
            "prediction_id": int(self.prediction_id),
            "hit": 1 if self.hit else 0,
            "realized_return": self.realized_return,
            "realized_price": self.realized_price,
            "ref_price": self.ref_price,
            "signed_error": self.signed_error,
            "brier": self.brier,
            "detail": self.detail,
            "scored_at": self.scored_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Outcome":
        return cls(
            prediction_id=int(row["prediction_id"]),
            hit=bool(row["hit"]),
            realized_return=row.get("realized_return"),
            realized_price=row.get("realized_price"),
            ref_price=row.get("ref_price"),
            signed_error=row.get("signed_error"),
            brier=row.get("brier"),
            detail=row.get("detail") or "",
            scored_at=row.get("scored_at") or _utc_now_iso(),
        )
