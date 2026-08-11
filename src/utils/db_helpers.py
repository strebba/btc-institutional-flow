"""Utility condivise per operazioni su database SQLite."""
from __future__ import annotations

import math
from typing import Any


def safe_db_value(val: Any) -> float | None:
    """Converte NaN/Inf in None per compatibilità SQLite."""
    if val is None:
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 6)
