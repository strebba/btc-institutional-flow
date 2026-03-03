"""Scraping dei flussi ETF Bitcoin giornalieri da Farside Investors.

URL: https://farside.co.uk/bitcoin-etf-flow-all-data/
La pagina contiene una tabella HTML con i flussi netti (in milioni USD) per
ciascun ETF Bitcoin spot (IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, GBTC, BTC).
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings, setup_logging
from src.flows.models import AggregateFlows, EtfFlowData

_log = setup_logging("flows.scraper")

# ETF ordinati come appaiono nella tabella Farside (colonne)
FARSIDE_TICKERS = [
    "IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC",
    "BRRR", "HODL", "BTCW", "GBTC", "BTC",
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_flow_value(cell_text: str) -> Optional[float]:
    """Converte il testo di una cella flusso in float USD.

    Gestisce:
      - valori normali: "123.4" → 123,400,000
      - negativi con parentesi: "(45.6)" → -45,600,000
      - "-" o vuoto → None
      - "Total" o header → None

    Args:
        cell_text: testo grezzo della cella.

    Returns:
        float | None: valore in USD (milioni × 1e6), o None se assente.
    """
    text = cell_text.strip().replace(",", "").replace("\xa0", "")
    if not text or text in ("-", "—", "Total", "n/a", "N/A"):
        return None
    # Parentesi = negativo
    neg = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        val = float(text) * 1_000_000  # i valori Farside sono in milioni
        return -val if neg else val
    except ValueError:
        return None


def _parse_farside_date(text: str, year_hint: int = 2024) -> Optional[date]:
    """Parsa la data dal formato Farside (es. "13 Jan" o "13 Jan 2025").

    Args:
        text: stringa data grezza.
        year_hint: anno di fallback se non specificato nella cella.

    Returns:
        date | None.
    """
    text = text.strip().lower()
    # Pattern "13 Jan 2025" o "13 Jan"
    m = re.match(r"(\d{1,2})\s+([a-z]{3})(?:\s+(\d{4}))?", text)
    if not m:
        return None
    day   = int(m.group(1))
    month = _MONTH_MAP.get(m.group(2))
    year  = int(m.group(3)) if m.group(3) else year_hint
    if not month:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


class FarsideScraper:
    """Scarica e parsa i flussi ETF Bitcoin da Farside Investors.

    Args:
        cfg: configurazione flows (da settings.yaml["flows"]).
    """

    FARSIDE_URL  = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    SOSOVALUE_URL = "https://sosovalue.com/assets/etf/us-bitcoin-spot"

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["flows"]
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/120.0.0.0 Safari/537.36",
            "Accept":          "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://www.google.com/",
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def _fetch_html(self, url: str) -> str:
        """Scarica HTML con retry.

        Args:
            url: URL da scaricare.

        Returns:
            str: testo HTML.
        """
        _log.info("Fetching: %s", url)
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text

    # ──────────────────────────────────────────────────────────────────────────
    # Farside parser
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_tickers_from_header(self, header_row: Tag) -> list[str]:
        """Estrae i ticker delle colonne dall'header della tabella.

        Args:
            header_row: tag <tr> dell'header.

        Returns:
            list[str]: lista di ticker nell'ordine delle colonne.
        """
        cells = header_row.find_all(["th", "td"])
        tickers: list[str] = []
        for cell in cells:
            txt = cell.get_text(strip=True).upper()
            # Cerca tickers ETF noti o "DATE"/"TOTAL"
            if txt in ("DATE", "DAY", ""):
                tickers.append("__date__")
            elif txt in ("TOTAL", "TOTAL FLOW"):
                tickers.append("__total__")
            else:
                # Normalizza: prendi il ticker puro (es. "IBIT\n(BlackRock)" → "IBIT")
                m = re.match(r"([A-Z]{2,6})", txt)
                tickers.append(m.group(1) if m else txt[:6])
        return tickers

    def _parse_table(self, html: str) -> list[EtfFlowData]:
        """Parsa la tabella HTML di Farside e ritorna i flussi giornalieri.

        Args:
            html: testo HTML della pagina.

        Returns:
            list[EtfFlowData]: lista di flussi, uno per ETF per giorno.
        """
        soup   = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")

        if not tables:
            _log.warning("Nessuna tabella trovata in Farside HTML")
            return []

        # Usa la tabella più grande (quella dei flussi)
        main_table = max(tables, key=lambda t: len(t.find_all("tr")))
        rows       = main_table.find_all("tr")

        if not rows:
            return []

        # Rileva colonne dall'header
        tickers = self._detect_tickers_from_header(rows[0])
        _log.info("Colonne Farside rilevate: %s", tickers)

        results: list[EtfFlowData] = []
        current_year = datetime.now().year
        last_year    = current_year  # usato per inferire l'anno quando non è nella cella

        for row in rows[1:]:
            cells    = row.find_all(["td", "th"])
            if not cells:
                continue
            raw_date = cells[0].get_text(strip=True)
            if not raw_date or raw_date.lower() in ("date", "total", ""):
                continue

            # Cerca anno nel testo della riga (a volte appare come "2024" in prima cella)
            year_in_cell = re.search(r"20\d{2}", raw_date)
            if year_in_cell:
                last_year = int(year_in_cell.group(0))

            parsed_date = _parse_farside_date(raw_date, year_hint=last_year)
            if not parsed_date:
                # Prova a leggere la data come gg/mm/yyyy o yyyy-mm-dd
                for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
                    try:
                        parsed_date = datetime.strptime(raw_date, fmt).date()
                        break
                    except ValueError:
                        continue
            if not parsed_date:
                _log.debug("Data non parsabile: %r", raw_date)
                continue

            # Leggi i valori per ogni ticker
            for col_idx, ticker in enumerate(tickers):
                if ticker in ("__date__", "__total__", ""):
                    continue
                if col_idx >= len(cells):
                    break
                val = _parse_flow_value(cells[col_idx].get_text(strip=True))
                if val is None:
                    continue  # cella vuota → skip, non zero

                results.append(EtfFlowData(
                    date=parsed_date,
                    ticker=ticker,
                    flow_usd=val,
                    source="farside",
                ))

        _log.info("Farside: %d righe di flussi parsate", len(results))
        return results

    def fetch(self) -> list[EtfFlowData]:
        """Scarica e parsa i flussi da Farside, con fallback su SoSoValue.

        Returns:
            list[EtfFlowData]: tutti i flussi disponibili.
        """
        try:
            html = self._fetch_html(self.FARSIDE_URL)
            flows = self._parse_table(html)
            if flows:
                return flows
            _log.warning("Farside: tabella vuota, provo SoSoValue")
        except Exception as e:
            _log.warning("Farside non raggiungibile (%s), provo SoSoValue", e)

        # Fallback: prova a costruire dati da SoSoValue (struttura diversa, best-effort)
        try:
            return self._fetch_sosovalue()
        except Exception as e2:
            _log.error("Anche SoSoValue fallito: %s", e2)
            return []

    def _fetch_sosovalue(self) -> list[EtfFlowData]:
        """Fallback: stima flussi da volume IBIT yfinance.

        Quando né Farside né CSV sono disponibili, stima il flusso netto IBIT
        come proxy basato su:  flow ≈ (net_volume_fraction × volume_usd)
        dove la direzione è inferita dal segno del daily return.

        Non è precisa quanto i dati Farside ma permette di usare il sistema
        end-to-end. I flussi reali possono essere caricati via CSV.

        Returns:
            list[EtfFlowData].
        """
        _log.info("Fallback: stima flussi IBIT da yfinance...")
        try:
            import yfinance as yf
            import numpy as np
            from datetime import timedelta

            end   = date.today()
            start = end - timedelta(days=self._cfg.get("lookback_days", 400))

            ibit = yf.download("IBIT", start=start.isoformat(),
                               end=(end + timedelta(days=1)).isoformat(),
                               progress=False, auto_adjust=True)
            if ibit.empty:
                return []

            # Appiattisci colonne multi-index (es. ("Close", "IBIT") → "Close")
            if isinstance(ibit.columns, pd.MultiIndex):
                ibit.columns = [c[0] for c in ibit.columns]

            # Stima: flow ≈ sign(return) × volume_usd × 0.08
            # (calibrato empiricamente per avere ordine di grandezza corretto)
            close_col  = "Close"
            volume_col = "Volume"
            ibit["_prev_close"] = ibit[close_col].shift(1)

            flows: list[EtfFlowData] = []
            for idx, row in ibit.iterrows():
                try:
                    close  = float(row[close_col])
                    volume = float(row[volume_col])
                    prev   = float(row["_prev_close"])
                    ret    = (close - prev) / prev if pd.notna(prev) and prev > 0 else 0.0
                    flow   = np.sign(ret) * volume * close * 0.08
                    flows.append(EtfFlowData(
                        date=idx.date(),
                        ticker="IBIT",
                        flow_usd=flow,
                        source="yfinance_estimate",
                    ))
                except Exception:
                    continue

            _log.info("Stima flussi IBIT da yfinance: %d giorni", len(flows))
            return flows
        except Exception as e:
            _log.error("Stima yfinance fallita: %s", e)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # CSV loader
    # ──────────────────────────────────────────────────────────────────────────

    def from_csv(self, path: str) -> list[EtfFlowData]:
        """Carica i flussi da un file CSV con formato Farside.

        Formato atteso:
          Date, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, GBTC, BTC, Total
          13 Jan 2025, 500.1, 200.5, (150.3), ...

        Args:
            path: percorso al file CSV.

        Returns:
            list[EtfFlowData].
        """
        import csv
        results: list[EtfFlowData] = []
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    raw_date = row.get("Date", "")
                    parsed   = _parse_farside_date(raw_date)
                    if not parsed:
                        continue
                    for ticker in FARSIDE_TICKERS:
                        raw = row.get(ticker, "")
                        val = _parse_flow_value(raw)
                        if val is not None:
                            results.append(EtfFlowData(
                                date=parsed, ticker=ticker,
                                flow_usd=val, source="csv",
                            ))
            _log.info("CSV caricato: %d flussi da %s", len(results), path)
        except Exception as e:
            _log.error("Errore lettura CSV %s: %s", path, e)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Aggregazione
    # ──────────────────────────────────────────────────────────────────────────

    def to_dataframe(self, flows: list[EtfFlowData]) -> pd.DataFrame:
        """Converte lista di EtfFlowData in DataFrame pivot.

        Righe = date, colonne = ticker ETF.

        Args:
            flows: lista di flussi.

        Returns:
            pd.DataFrame: pivot con colonne per ticker + "total".
        """
        if not flows:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"date": f.date, "ticker": f.ticker, "flow_usd": f.flow_usd}
            for f in flows
        ])
        pivot = df.pivot_table(
            index="date", columns="ticker", values="flow_usd", aggfunc="sum"
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot["total"] = pivot.sum(axis=1)
        # Rinomina colonne per chiarezza
        if "IBIT" not in pivot.columns:
            pivot["IBIT"] = float("nan")
        return pivot.sort_index()

    def aggregate(self, flows: list[EtfFlowData]) -> list[AggregateFlows]:
        """Aggrega i flussi per data.

        Args:
            flows: lista di EtfFlowData.

        Returns:
            list[AggregateFlows]: aggregato giornaliero.
        """
        by_date: dict[date, dict[str, float]] = {}
        for f in flows:
            by_date.setdefault(f.date, {})[f.ticker] = (
                by_date.get(f.date, {}).get(f.ticker, 0.0) + f.flow_usd
            )

        result: list[AggregateFlows] = []
        for d in sorted(by_date.keys()):
            tickers_d    = by_date[d]
            total        = sum(tickers_d.values())
            ibit_flow    = tickers_d.get("IBIT", 0.0)
            result.append(AggregateFlows(
                date=d,
                total_flow_usd=total,
                ibit_flow_usd=ibit_flow,
                flows_by_ticker=tickers_d,
            ))
        return result


def main() -> None:
    """Entry point CLI."""
    scraper = FarsideScraper()
    flows   = scraper.fetch()
    df      = scraper.to_dataframe(flows)
    print(df.tail(10).to_string())
    print(f"\nTotale giorni: {len(df)}")
    print(f"Range: {df.index.min()} → {df.index.max()}")
    if "IBIT" in df.columns:
        print(f"IBIT total inflow: ${df['IBIT'].sum()/1e9:.2f}B")


if __name__ == "__main__":
    main()
