"""Client per l'API pubblica di Deribit (senza autenticazione).

Endpoints usati (tutti GET, no auth):
  GET /public/get_instruments?currency=BTC&kind=option&expired=false
  GET /public/ticker?instrument_name={name}
  GET /public/get_index_price?index_name=btc_usd
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings, setup_logging

_log = setup_logging("gex.deribit")


class DeribitClient:
    """Client per le API pubbliche di Deribit.

    Args:
        cfg: blocco configurazione deribit (da settings.yaml).
    """

    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["deribit"]
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "ibit-gamma-tracker/1.0",
            "Accept":     "application/json",
        })
        self._min_interval = 1.0 / self._cfg["rate_limit_rps"]
        self._last_request: float = 0.0
        # Cache in memoria per la sessione (evita richieste duplicate)
        self._cache: dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Rispetta il rate limit configurato."""
        elapsed = time.monotonic() - self._last_request
        wait    = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=10))
    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """GET con throttle, retry e caching.

        Args:
            endpoint: es. "/get_instruments".
            params: query parameters.

        Returns:
            Any: campo "result" della risposta JSON Deribit.

        Raises:
            RuntimeError: se l'API ritorna un errore applicativo.
            requests.HTTPError: per errori HTTP.
        """
        cache_key = f"{endpoint}:{params}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._throttle()
        url  = f"{self.BASE_URL}{endpoint}"
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Deribit API error: {data['error']}")

        result = data.get("result")
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Svuota la cache in memoria."""
        self._cache.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API methods
    # ──────────────────────────────────────────────────────────────────────────

    def get_spot_price(self) -> float:
        """Restituisce il prezzo spot corrente di BTC-USD.

        Returns:
            float: prezzo in USD.
        """
        result = self._get("/get_index_price", {"index_name": "btc_usd"})
        return float(result["index_price"])

    def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False,
    ) -> list[dict]:
        """Restituisce la lista degli strumenti attivi.

        Args:
            currency: "BTC" o "ETH".
            kind: "option" o "future".
            expired: se True include gli strumenti scaduti.

        Returns:
            list[dict]: lista di strumenti con nome, strike, tipo, scadenza.
        """
        result = self._get("/get_instruments", {
            "currency": currency,
            "kind":     kind,
            "expired":  str(expired).lower(),
        })
        return result or []

    def get_ticker(self, instrument_name: str) -> dict:
        """Restituisce il ticker completo per un singolo strumento.

        Contiene: mark_price, mark_iv, greeks (delta, gamma, theta, vega),
        open_interest, best_bid, best_ask.

        Args:
            instrument_name: es. "BTC-25JAN25-100000-C".

        Returns:
            dict: dati ticker.
        """
        result = self._get("/ticker", {"instrument_name": instrument_name})
        return result or {}

    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 1,
    ) -> dict:
        """Restituisce il book ordini (con greche e OI).

        Args:
            instrument_name: nome strumento Deribit.
            depth: profondità del book (default 1).

        Returns:
            dict: book con gamma, delta, open_interest, etc.
        """
        result = self._get("/get_order_book", {
            "instrument_name": instrument_name,
            "depth":           depth,
        })
        return result or {}

    # ──────────────────────────────────────────────────────────────────────────
    # Batch fetching
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_all_options(self, currency: str = "BTC") -> list[dict]:
        """Scarica i dati di tutte le opzioni attive con greche e OI.

        Usa il ticker endpoint (più completo dell'order_book per le greche).
        Gestisce il rate limit tra le richieste.

        Args:
            currency: "BTC" o "ETH".

        Returns:
            list[dict]: lista di dati per ogni opzione con campi:
                instrument_name, strike, option_type, expiration_timestamp,
                gamma, open_interest, mark_price, mark_iv, delta, underlying_price.
        """
        instruments = self.get_instruments(currency=currency)
        _log.info("Strumenti %s attivi: %d", currency, len(instruments))

        results: list[dict] = []
        errors  = 0

        for i, instr in enumerate(instruments, 1):
            name = instr.get("instrument_name", "")
            if not name:
                continue

            try:
                ticker = self.get_ticker(name)
                greeks = ticker.get("greeks", {})

                results.append({
                    "instrument_name":     name,
                    "strike":              float(instr.get("strike", 0)),
                    "option_type":         instr.get("option_type", ""),   # "call" | "put"
                    "expiration_timestamp": instr.get("expiration_timestamp", 0),
                    "gamma":               float(greeks.get("gamma", 0) or 0),
                    "delta":               float(greeks.get("delta", 0) or 0),
                    "vega":                float(greeks.get("vega",  0) or 0),
                    "open_interest":       float(ticker.get("open_interest", 0) or 0),
                    "mark_price":          float(ticker.get("mark_price", 0) or 0),
                    "mark_iv":             float(ticker.get("mark_iv", 0) or 0),
                    "underlying_price":    float(ticker.get("underlying_price", 0) or 0),
                    "best_bid":            float(ticker.get("best_bid_price", 0) or 0),
                    "best_ask":            float(ticker.get("best_ask_price", 0) or 0),
                })

                if i % 50 == 0:
                    _log.info("Progresso: %d/%d opzioni scaricate", i, len(instruments))

            except Exception as e:
                errors += 1
                _log.debug("Errore per %s: %s", name, e)
                continue

        _log.info(
            "Fetch completato: %d opzioni valide, %d errori",
            len(results), errors,
        )
        return results
