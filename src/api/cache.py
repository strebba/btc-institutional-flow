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
    "_gex_data":      300,   # 5 min  — raw GexSnapshot objects (condivisi tra /gex e /barriers)
    "gex_enrichment": 3600,  # 1 ora  — CoinGlass coverage score + multi-exchange PCR
    "flows":          900,   # 15 min — Farside scrape
    "barriers":       3600,  # 1 ora  — dati SEC EDGAR statici
    "macro":          3600,  # 1 ora  — dati CoinGlass giornalieri
}

# Lock che impedisce fetch Deribit concorrenti: il secondo richiedente attende
# il primo e poi legge dalla cache invece di lanciare un nuovo fetch da 888 opzioni.
#
# NOTA: questo è un threading.Lock, corretto perché tutti i chiamanti di
# _get_gex_data() sono endpoint sync (def, non async def). FastAPI esegue
# gli endpoint sync in un threadpool — il lock serializza correttamente
# attraverso i thread. Se in futuro un chiamante diventasse async (await),
# questo lock va convertito in asyncio.Lock per non bloccare l'event loop.
_gex_fetch_lock = threading.Lock()
_gex_fetch_locks: dict[str, threading.Lock] = {"BTC": _gex_fetch_lock}
_gex_locks_guard = threading.Lock()


def gex_fetch_lock(asset: str = "BTC") -> threading.Lock:
    """Lock di dedup del fetch Deribit per asset: BTC ed ETH non si attendono a vicenda."""
    with _gex_locks_guard:
        return _gex_fetch_locks.setdefault(asset, threading.Lock())


def asset_key(base: str, asset: str = "BTC") -> str:
    """Chiave di cache per asset. BTC conserva la chiave storica (``gex``), ETH ha ``gex:eth``."""
    return base if asset == "BTC" else f"{base}:{asset.lower()}"


def cache_get(key: str) -> Any | None:
    ttl = _TTL.get(key.split(":", 1)[0], 300)   # gex:eth eredita il TTL di gex
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry[0]) < ttl:
            return entry[1]
    return None


def cache_set(key: str, payload: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), payload)


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()
