"""Helper functions condivise dai router: sanitizzazione JSON, envelope risposte."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import JSONResponse


def sanitize(obj: Any) -> Any:
    """Converte ricorsivamente numpy types e NaN/Inf in tipi JSON-compatibili.

    Gestisce: np.integer -> int, np.floating -> float (NaN/Inf -> None),
    np.bool_ -> bool, np.ndarray -> list, pd.Timestamp -> str,
    dict -> dict, list -> list.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def ok(data: Any) -> JSONResponse:
    payload = {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "data": data}
    return JSONResponse(content=sanitize(payload))


def http_error(msg: str, code: int = 500) -> HTTPException:
    """Genera un'HTTPException con il messaggio e codice specificati."""
    return HTTPException(status_code=code, detail=msg)
