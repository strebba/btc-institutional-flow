"""Client per SoSoValue ETF API — flussi giornalieri Bitcoin ETF spot."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import requests

from src.config import get_settings, setup_logging
from src.flows.models import EtfFlowData

_log = setup_logging("flows.sosovalue")

_API_URL = "https://sosovalue.com/api/etf/btc-spot-etf-flow"

# Ticker riconosciuti da SoSoValue → ticker standard
_TICKER_MAP = {
    "IBIT": "IBIT", "FBTC": "FBTC", "BITB": "BITB", "ARKB": "ARKB",
    "BTCO": "BTCO", "EZBC": "EZBC", "BRRR": "BRRR", "HODL": "HODL",
    "BTCW": "BTCW", "GBTC": "GBTC", "BTC":  "BTC",
}


class SoSoValueClient:
    """Scarica flussi ETF Bitcoin da SoSoValue API (free tier con API key).

    Se `sosovalue_api_key` non è configurata in settings.yaml, ``fetch()``
    ritorna una lista vuota silenziosamente (non lancia eccezioni).

    Args:
        cfg: sezione ``flows`` di settings.yaml (opzionale, caricata in auto).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["flows"]
        self._api_key: str = self._cfg.get("sosovalue_api_key", "")

    def fetch(self, lookback_days: int = 400) -> list[EtfFlowData]:
        """Scarica i flussi ETF Bitcoin spot dall'API SoSoValue.

        Args:
            lookback_days: giorni di storico da richiedere.

        Returns:
            list[EtfFlowData]: flussi giornalieri, o lista vuota se API key
            assente o in caso di errore.
        """
        if not self._api_key:
            _log.debug("SoSoValue API key non configurata — skip")
            return []

        end_date   = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "ibit-gamma-tracker/1.0 (contact@example.com)",
            "Accept": "application/json",
        })

        params: dict[str, Any] = {
            "range":     "1Y",
            "startDate": start_date.isoformat(),
            "endDate":   end_date.isoformat(),
        }

        try:
            resp = session.get(_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log.warning("SoSoValue API errore: %s", e)
            return []

        return self._parse_response(data)

    def _parse_response(self, data: Any) -> list[EtfFlowData]:
        """Parsa la risposta JSON di SoSoValue.

        Formato atteso: array di dict o wrapper ``{"data": [...]}``.
        Ogni record ha un campo ``date`` (stringa ISO o unix-ms) e campi
        per ticker (valori in milioni USD).

        Args:
            data: risposta JSON deserializzata.

        Returns:
            list[EtfFlowData].
        """
        # Normalizza wrapper {"data": [...]} o {"items": [...]}
        if isinstance(data, dict):
            data = data.get("data", data.get("items", []))

        if not isinstance(data, list):
            _log.warning("SoSoValue: formato risposta inatteso (tipo %s)", type(data).__name__)
            return []

        results: list[EtfFlowData] = []
        for record in data:
            if not isinstance(record, dict):
                continue

            raw_date = (
                record.get("date")
                or record.get("Date")
                or record.get("timestamp")
            )
            if raw_date is None:
                continue

            try:
                if isinstance(raw_date, (int, float)):
                    # Unix timestamp in millisecondi
                    parsed_date = datetime.utcfromtimestamp(raw_date / 1000).date()
                else:
                    parsed_date = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d").date()
            except (ValueError, OSError):
                continue

            for api_ticker, std_ticker in _TICKER_MAP.items():
                raw_val = record.get(api_ticker)
                if raw_val is None:
                    continue
                try:
                    flow_usd = float(raw_val) * 1_000_000  # SoSoValue usa milioni USD
                except (TypeError, ValueError):
                    continue

                results.append(EtfFlowData(
                    date=parsed_date,
                    ticker=std_ticker,
                    flow_usd=flow_usd,
                    source="sosovalue",
                ))

        _log.info("SoSoValue: %d flussi parsati", len(results))
        return results
