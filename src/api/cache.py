"""Module-wide cache con TTL configurabile e dedup in-flight per fetch Deribit."""
from __future__ import annotations

import threading
import time
from typing import Any

# ─── In-memory TTL cache ───────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}  # key → (timestamp, payload)
_cache_lock = threading.Lock()

_TTL: dict[str, int] = {
    "gex":            300,   # 5 min  — opzioni Deribit, ~90s fetch
    "_gex_data":      300,   # 5 min  — raw GexSnapshot objects (condivisi tra /gex e /signals)
    "gex_enrichment": 3600,  # 1 ora  — CoinGlass coverage score + multi-exchange PCR
    "flows":          900,   # 15 min — Farside scrape
    "barriers":       3600,  # 1 ora  — dati SEC EDGAR statici
    "signals":        300,   # 5 min  — dipende da gex + flows
    "macro":          3600,  # 1 ora  — dati CoinGlass giornalieri
    "ifi":            900,   # 15 min — serie giornaliera, cambia lentamente
    "pillars_series": 900,   # 15 min — compute_series + Farside scrape (costoso)
}

# Lock che impedisce fetch Deribit concorrenti: il secondo richiedente attende
# il primo e poi legge dalla cache invece di lanciare un nuovo fetch da 888 opzioni.
_gex_fetch_lock = threading.Lock()


def cache_get(key: str) -> Any | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < _TTL.get(key, 300):
            return entry[1]
    return None


def cache_set(key: str, payload: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), payload)


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()
