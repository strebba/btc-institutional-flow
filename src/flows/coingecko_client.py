"""Client CoinGecko per funding rate e open interest dei perpetui BTC.

Serve a coprire due dei cinque fattori del pilastro macro quando CoinGlass non è
disponibile. È un ripiego dichiarato, non un sostituto: CoinGecko **non** espone
long/short ratio né liquidazioni, e non ha storico — l'open interest arriva come
livello istantaneo, non come serie.

L'endpoint ``/api/v3/derivatives`` funziona anche senza chiave. Una chiave del
tier Demo alza solo i limiti di frequenza, quindi è opzionale.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from src.config import get_settings, setup_logging
from src.flows.funding import annualize_funding_pct

_log = setup_logging("flows.coingecko")

__all__ = ["CoinGeckoClient", "CoinGeckoError", "annualize_funding_pct"]


class CoinGeckoError(Exception):
    """Errore nel raggiungere o interpretare l'API CoinGecko."""


def _num(value: Any) -> float | None:
    """Converte in float restituendo None invece di sollevare."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CoinGeckoClient:
    """Client HTTP per l'API pubblica CoinGecko.

    Args:
        cfg: configurazione (default da settings.yaml sezione 'coingecko').
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, cfg: dict | None = None) -> None:
        settings = get_settings()
        self._cfg = cfg or settings.get("coingecko", {}) or {}

        # La chiave è opzionale: l'endpoint derivatives risponde anche senza.
        self._api_key = (
            os.getenv("COINGECKO_API_KEY") or self._cfg.get("api_key", "")
        ).strip()

        self._timeout = self._cfg.get("timeout_s", 90)
        self._rate_limit = self._cfg.get("rate_limit_rps", 0.5)
        self._last_call_ts = 0.0

        self._session = requests.Session()
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key
        self._session.headers.update(headers)

        _log.info("CoinGecko client inizializzato (chiave presente: %s)", bool(self._api_key))

    @property
    def has_api_key(self) -> bool:
        """Vero se una chiave Demo è configurata.

        Informativo: l'endpoint usato qui risponde anche senza. La chiave alza
        solo i limiti di frequenza.
        """
        return bool(self._api_key)

    def _throttle(self) -> None:
        if self._rate_limit <= 0:
            return
        attesa = (1.0 / self._rate_limit) - (time.time() - self._last_call_ts)
        if attesa > 0:
            time.sleep(attesa)
        self._last_call_ts = time.time()

    def _get(self, path: str) -> Any:
        self._throttle()
        url = self._cfg.get("base_url", self.BASE_URL) + path
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise CoinGeckoError(f"CoinGecko {path}: {exc}") from exc
        except ValueError as exc:
            raise CoinGeckoError(f"CoinGecko {path}: risposta non JSON") from exc

    def fetch_btc_derivatives(self) -> list[dict]:
        """Perpetui BTC con open interest e funding rate utilizzabili.

        L'endpoint restituisce **tutti** gli asset — circa 25.000 contratti, 8 MB —
        quindi il filtro va fatto subito e il risultato va messo in cache a monte.

        Returns:
            Lista di dict con ``market``, ``open_interest`` e ``funding_rate``.
            Lista vuota se la risposta non ha la forma attesa: un payload
            malformato è un dato mancante, non un errore fatale.
        """
        data = self._get("/derivatives")
        if not isinstance(data, list):
            _log.warning("CoinGecko /derivatives: risposta inattesa (%s)", type(data).__name__)
            return []

        out: list[dict] = []
        for riga in data:
            if not isinstance(riga, dict):
                continue
            if riga.get("index_id") != "BTC" or riga.get("contract_type") != "perpetual":
                continue
            oi = _num(riga.get("open_interest"))
            funding = _num(riga.get("funding_rate"))
            if oi is None or funding is None or oi <= 0:
                continue
            out.append({"market": riga.get("market", "n/d"), "open_interest": oi,
                        "funding_rate": funding})

        _log.info("CoinGecko: %d perpetui BTC utilizzabili su %d contratti", len(out), len(data))
        return out

    def fetch_funding_and_oi(self) -> tuple[float | None, float | None, int]:
        """Funding annualizzato pesato per OI, open interest aggregato, n. contratti.

        La ponderazione per open interest è ciò che rende il numero confrontabile
        con quello di CoinGlass, che espone proprio un ``funding-rate/oi-weight``:
        una media semplice darebbe lo stesso peso a un exchange da 8 miliardi e a
        uno da 50 milioni.

        Returns:
            ``(funding_ann_pct, oi_usd, n_contratti)``, oppure ``(None, None, 0)``
            se non c'è nulla di utilizzabile. È un fallback: non solleva, perché
            il chiamante deve poter proseguire senza questo fattore.
        """
        try:
            righe = self.fetch_btc_derivatives()
        except CoinGeckoError as exc:
            _log.warning("CoinGecko non raggiungibile: %s", exc)
            return None, None, 0

        oi_totale = sum(r["open_interest"] for r in righe)
        if not righe or oi_totale <= 0:
            return None, None, 0

        pesato = sum(r["funding_rate"] * r["open_interest"] for r in righe) / oi_totale
        return annualize_funding_pct(pesato), oi_totale, len(righe)
