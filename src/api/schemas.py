"""Pydantic response models for FastAPI endpoints.

Usato per validazione automatica delle risposte e generazione OpenAPI docs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ApiResponse(BaseModel):
    """Envelope standard di tutte le risposte API."""
    status: str
    timestamp: str
    data: dict[str, Any]


class HealthData(BaseModel):
    service: str
    healthy: bool


class EdgarHealthData(BaseModel):
    last_update: Optional[str] = None
    total_notes: int = 0
    total_barriers: int = 0
    active_barriers: int = 0
    stale_days: Optional[int] = None


class GexSnapshotModel(BaseModel):
    spot_price: float
    total_net_gex: float
    total_net_gex_m: float
    gamma_flip_price: Optional[float] = None
    put_wall: Optional[float] = None
    call_wall: Optional[float] = None
    max_pain: Optional[float] = None
    n_instruments: int = 0
    put_call_ratio: Optional[float] = None
    distance_to_put_wall_pct: Optional[float] = None
    distance_to_call_wall_pct: Optional[float] = None


class SchedulerHealth(BaseModel):
    alert: bool
    ifi: bool
    forecast: bool
