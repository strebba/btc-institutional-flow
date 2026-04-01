"""Client per CoinGlass API v4 (Startup tier).

Fornisce:
  - fetch_etf_flows()          → list[EtfFlowData]  (IBIT + tutti i BTC ETF spot US)
  - fetch_funding_rate_history() → pd.Series         (funding rate OI-weighted, daily)
  - fetch_aggregated_oi_history() → pd.Series        (futures OI aggregato, daily)

Autenticazione: header CG-API-KEY.
API key: env var COINGLASS_API_KEY > settings.yaml coinglass.api_key.

Docs: https://docs.coinglass.com
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone
from typing import Any

import requests

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False
    # No-op fallback: il decorator non fa nulla senza tenacity
    def retry(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator
    def stop_after_attempt(n):  # type: ignore[misc]
        return None
    def wait_exponential(**kwargs):  # type: ignore[misc]
        return None

import pandas as pd

from src.config import get_settings, setup_logging
from src.flows.models import EtfFlowData

_log = setup_logging("flows.coinglass")


class CoinGlassError(Exception):
    """Eccezione per errori API CoinGlass."""


class CoinGlassClient:
    """Client HTTP per CoinGlass API v4.

    Args:
        cfg: configurazione (default da settings.yaml sezione 'coinglass').
    """

    BASE_URL = "https://open-api-v4.coinglass.com"

    def __init__(self, cfg: dict | None = None) -> None:
        settings   = get_settings()
        self._cfg  = cfg or settings.get("coinglass", {})

        # API key: env var ha precedenza su settings.yaml
        self._api_key = (
            os.getenv("COINGLASS_API_KEY")
            or self._cfg.get("api_key", "")
        ).strip()

        self._timeout      = self._cfg.get("timeout_s", 15)
        self._rate_limit   = self._cfg.get("rate_limit_rps", 1.0)
        self._last_call_ts = 0.0

        self._session = requests.Session()
        self._session.headers.update({
            "CG-API-KEY": self._api_key,
            "Accept":     "application/json",
        })

    def _throttle(self) -> None:
        """Rispetta il rate limit configurato (req/s)."""
        if self._rate_limit > 0:
            elapsed = time.time() - self._last_call_ts
            wait    = (1.0 / self._rate_limit) - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_call_ts = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=10))
    def _get(self, path: str, params: dict | None = None) -> Any:
        """Esegue GET con retry su errori transitori.

        Args:
            path: percorso relativo es. '/api/etf/bitcoin/flow-history'.
            params: query string params.

        Returns:
            Il campo 'data' della risposta JSON CoinGlass.

        Raises:
            CoinGlassError: su errore API o chiave mancante.
        """
        if not self._api_key:
            raise CoinGlassError(
                "CoinGlass API key mancante. "
                "Imposta env var COINGLASS_API_KEY o config coinglass.api_key"
            )

        self._throttle()
        url = self._cfg.get("base_url", self.BASE_URL) + path
        resp = self._session.get(url, params=params or {}, timeout=self._timeout)

        if resp.status_code == 401:
            raise CoinGlassError("CoinGlass: API key non valida (401)")
        if resp.status_code == 429:
            raise CoinGlassError("CoinGlass: rate limit superato (429)")
        resp.raise_for_status()

        body = resp.json()
        if body.get("code") != "0":
            raise CoinGlassError(f"CoinGlass API error: {body.get('msg', body)}")

        return body.get("data", [])

    # ─── ETF Flows ────────────────────────────────────────────────────────────

    def fetch_etf_flows(self, days: int = 365) -> list[EtfFlowData]:
        """Scarica i flussi giornalieri di tutti i Bitcoin ETF spot USA.

        Args:
            days: numero di giorni storici da richiedere.

        Returns:
            list[EtfFlowData] con source='coinglass', ordinata per data crescente.
            Lista vuota se l'endpoint non risponde o non ci sono dati.
        """
        try:
            data = self._get("/api/etf/bitcoin/flow-history", {"limit": min(days, 500)})
        except CoinGlassError as e:
            _log.warning("CoinGlass ETF flows: %s", e)
            return []
        except Exception as e:
            _log.warning("CoinGlass ETF flows (rete): %s", e)
            return []

        if not isinstance(data, list):
            _log.warning("CoinGlass ETF flows: risposta inattesa %s", type(data))
            return []

        results: list[EtfFlowData] = []
        for entry in data:
            # Supporta due strutture di risposta:
            # v4: {"timestamp": ms, "etf_flows": [{"etf_ticker": "IBIT", "flow_usd": ...}]}
            # v3: {"date": "YYYY-MM-DD", "flows_by_ticker": {"IBIT": ...}}
            raw_date = entry.get("timestamp") or entry.get("date") or entry.get("t")
            if raw_date is None:
                continue

            try:
                if isinstance(raw_date, (int, float)):
                    flow_date = datetime.fromtimestamp(raw_date / 1000, tz=timezone.utc).date()
                else:
                    flow_date = date.fromisoformat(str(raw_date)[:10])
            except (ValueError, OSError):
                _log.debug("CoinGlass: data non parsabile: %s", raw_date)
                continue

            # v4 format: etf_flows è una lista di {etf_ticker, flow_usd}
            etf_flows_list: list[dict] = entry.get("etf_flows") or []
            if etf_flows_list:
                for etf_entry in etf_flows_list:
                    ticker   = etf_entry.get("etf_ticker") or etf_entry.get("ticker")
                    flow_val = etf_entry.get("flow_usd")
                    if not ticker or flow_val is None:
                        continue
                    try:
                        results.append(EtfFlowData(
                            date     = flow_date,
                            ticker   = str(ticker),
                            flow_usd = float(flow_val),
                            source   = "coinglass",
                        ))
                    except (TypeError, ValueError):
                        continue
            else:
                # v3 format: flows_by_ticker è un dict {TICKER: value}
                flows_by_ticker: dict = entry.get("flows_by_ticker") or {}
                for ticker, flow_val in flows_by_ticker.items():
                    if flow_val is None:
                        continue
                    try:
                        results.append(EtfFlowData(
                            date     = flow_date,
                            ticker   = str(ticker),
                            flow_usd = float(flow_val),
                            source   = "coinglass",
                        ))
                    except (TypeError, ValueError):
                        continue

        _log.info("CoinGlass ETF flows: %d record scaricati", len(results))
        return sorted(results, key=lambda r: r.date)

    # ─── Funding Rate ─────────────────────────────────────────────────────────

    def fetch_funding_rate_history(self, days: int = 365) -> pd.Series:
        """Scarica il funding rate OI-weighted (daily, annualizzato).

        Args:
            days: numero di giorni storici.

        Returns:
            pd.Series con DatetimeIndex (UTC) e valori float (funding rate 8h in %).
            Serie vuota su errore.
        """
        try:
            data = self._get(
                "/api/futures/fundingRate/oi-weight-ohlc-history",
                {"symbol": "BTC", "interval": "h8", "limit": min(days * 3, 1000)},
            )
        except (CoinGlassError, Exception) as e:
            _log.warning("CoinGlass funding rate: %s", e)
            return pd.Series(dtype=float, name="funding_rate")

        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float, name="funding_rate")

        # Risposta OHLC: [t, o, h, l, c] oppure dict con t/c
        rows: list[tuple] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 5:
                ts_ms, _, _, _, close = item[0], item[1], item[2], item[3], item[4]
            elif isinstance(item, dict):
                ts_ms = item.get("t") or item.get("time")
                close = item.get("c") or item.get("close")
            else:
                continue
            if ts_ms is None or close is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append((dt, float(close)))
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return pd.Series(dtype=float, name="funding_rate")

        # Ricampiona a daily usando l'ultimo valore del giorno
        df = pd.DataFrame(rows, columns=["ts", "rate"]).set_index("ts")
        daily = df["rate"].resample("1D").last().dropna()
        daily.name = "funding_rate"
        return daily

    # ─── Aggregated Futures OI ────────────────────────────────────────────────

    def fetch_aggregated_oi_history(self, days: int = 365) -> pd.Series:
        """Scarica l'open interest futures aggregato cross-exchange (daily, USD).

        Args:
            days: numero di giorni storici.

        Returns:
            pd.Series con DatetimeIndex (UTC) e valori float (OI in USD).
            Serie vuota su errore.
        """
        try:
            data = self._get(
                "/api/futures/openInterest/aggregated-history",
                {"symbol": "BTC", "interval": "1d", "limit": min(days, 500)},
            )
        except (CoinGlassError, Exception) as e:
            _log.warning("CoinGlass aggregated OI: %s", e)
            return pd.Series(dtype=float, name="futures_oi_usd")

        if not isinstance(data, list) or not data:
            return pd.Series(dtype=float, name="futures_oi_usd")

        rows: list[tuple] = []
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                ts_ms, oi = item[0], item[1]
            elif isinstance(item, dict):
                ts_ms = item.get("t") or item.get("time")
                oi    = item.get("openInterest") or item.get("oi") or item.get("c")
            else:
                continue
            if ts_ms is None or oi is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts_ms) / 1000, tz=timezone.utc)
                rows.append((dt, float(oi)))
            except (TypeError, ValueError, OSError):
                continue

        if not rows:
            return pd.Series(dtype=float, name="futures_oi_usd")

        df = pd.DataFrame(rows, columns=["ts", "oi"]).set_index("ts")
        daily = df["oi"].resample("1D").last().dropna()
        daily.name = "futures_oi_usd"
        return daily
