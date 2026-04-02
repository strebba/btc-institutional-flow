"""Ricerca filing SEC EDGAR (424B2/424B3) relativi a IBIT.

Usa l'API EFTS (Full-Text Search) di EDGAR:
  GET https://efts.sec.gov/LATEST/search-index
"""
from __future__ import annotations

import time
from typing import Any, Generator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings, setup_logging

_log = setup_logging("edgar.search")


class EdgarSearcher:
    """Ricerca e scarica la lista di filing EDGAR per note strutturate su IBIT.

    Args:
        cfg: blocco di configurazione (da settings.yaml["edgar"]).
    """

    BASE_URL = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["edgar"]
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self._cfg["user_agent"]})
        self._min_interval = 1.0 / self._cfg["rate_limit_rps"]
        self._last_request: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Rispetta il rate limit configurato (default 8 req/s)."""
        elapsed = time.monotonic() - self._last_request
        wait    = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get(self, params: dict[str, Any]) -> dict:
        """Esegue una GET sull'API EFTS con throttle e retry.

        Args:
            params: parametri query string.

        Returns:
            dict: risposta JSON parsata.

        Raises:
            requests.HTTPError: se la risposta non è 2xx dopo i retry.
        """
        self._throttle()
        resp = self._session.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def search_filings(
        self,
        query: str,
        forms: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Generator[dict, None, None]:
        """Ricerca paginata su EFTS e yield di ogni hit.

        Args:
            query: stringa di ricerca full-text (es. "IBIT").
            forms: lista di form type (es. ["424B2", "424B3"]).
            start_date: data inizio nel formato YYYY-MM-DD.
            end_date: data fine nel formato YYYY-MM-DD (default: oggi).

        Yields:
            dict: singolo hit con campi _source (url, filing_date, entity_name …).
        """
        cfg   = self._cfg
        forms = forms or cfg["forms"]
        start = start_date or cfg["start_date"]

        params: dict[str, Any] = {
            "q":        f'"{query}"',
            "dateRange": "custom",
            "startdt":  start,
            "forms":    ",".join(forms),
            "from":     0,
            "size":     cfg["page_size"],
        }
        if end_date:
            params["enddt"] = end_date

        _log.info("EDGAR search: query=%r forms=%s from=%s", query, forms, start)

        total_fetched = 0
        while True:
            data  = self._get(params)
            hits  = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {}).get("value", 0)

            if not hits:
                break

            _log.debug("Pagina from=%d: %d hit (totale=%d)", params["from"], len(hits), total)

            for hit in hits:
                yield hit
                total_fetched += 1

            if total_fetched >= total:
                break

            params["from"] += len(hits)  # next page

    def collect_filing_urls(self) -> list[dict]:
        """Raccoglie tutti gli URL di filing per tutti i search_terms configurati.

        Deduplicazione per URL. Ogni elemento è un dict con:
          - url: link al documento primario
          - accession_no: numero di accession EDGAR
          - filing_date: data del filing
          - entity_name: nome emittente (dal filing header)
          - form_type: 424B2 | 424B3

        Returns:
            list[dict]: lista deduplicata di filing metadata.
        """
        seen:    set[str] = set()
        results: list[dict] = []

        for term in self._cfg["search_terms"]:
            for hit in self.search_filings(query=term):
                src    = hit.get("_source", {})
                # L'URL del documento primario è in file_num o file_date
                acc_no = src.get("file_num") or hit.get("_id", "")
                # EDGAR EFTS restituisce il path relativo del documento
                doc_path = src.get("period_of_report", "")

                # Costruisci URL canonico del filing (viewer EDGAR)
                # Il campo `_id` in EFTS è: accession_number (con trattini)
                raw_id   = hit.get("_id", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{src.get('entity_id', '')}/{raw_id.replace('-', '')}/{raw_id}.txt"
                )

                # URL più affidabile: usiamo il link diretto al documento
                # presente in `file_date` → preferisco costruire l'URL viewer
                viewer_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum={acc_no}"

                # EFTS v2 fornisce direttamente `_source.file_date` e un campo
                # `period_of_report`; il link al documento è in `_source.period_of_report`
                # In realtà usiamo l'endpoint più pratico:
                entity_id  = src.get("entity_id", "")
                accession  = raw_id  # es. "0001234567-24-000123"
                # URL del filing index page
                index_url  = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar"
                    f"?action=getcompany&CIK={entity_id}&type={src.get('form_type','424B2')}"
                )
                # URL diretto più utile: costruiamo con accession number senza trattini
                acc_clean  = accession.replace("-", "")
                direct_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{entity_id}/{acc_clean}/{accession}.htm"
                )

                if direct_url in seen:
                    continue
                seen.add(direct_url)

                results.append(
                    {
                        "url":          direct_url,
                        "accession_no": accession,
                        "filing_date":  src.get("period_of_report") or src.get("file_date", ""),
                        "entity_name":  src.get("entity_name", ""),
                        "form_type":    src.get("form_type", ""),
                    }
                )

        _log.info("Totale filing unici trovati: %d", len(results))
        return results


# ─── Alternativa più robusta: usa l'endpoint EFTS corretto ────────────────────

class EdgarEftsSearcher:
    """Versione alternativa che usa l'endpoint EFTS pubblico con response corretta.

    L'API reale è: https://efts.sec.gov/LATEST/search-index?q=...&forms=...
    ma per estrarre URL dei documenti è meglio usare:
    https://efts.sec.gov/LATEST/search-index?q=%22IBIT%22&forms=424B2&dateRange=custom&startdt=2024-01-01
    e leggere i campi `_source.period_of_report`, `_source.entity_id`, `_id`.

    Questa classe è quella che viene effettivamente usata nel progetto.
    """

    EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
    # Endpoint alternativo più semplice (EDGAR full-text search API v2)
    EFTS_V2  = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or get_settings()["edgar"]
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":      self._cfg["user_agent"],
            "Accept-Encoding": "gzip, deflate",
            "Accept":          "application/json",
        })
        self._min_interval = 1.0 / self._cfg["rate_limit_rps"]
        self._last_request: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait    = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    def _get(self, params: dict) -> dict:
        """GET con throttle, retry e logging.

        Args:
            params: query parameters.

        Returns:
            dict: risposta JSON.
        """
        self._throttle()
        _log.debug("GET %s params=%s", self.EFTS_URL, params)
        resp = self._session.get(self.EFTS_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _build_url(hit: dict) -> dict:
        """Costruisce metadati e URL corretto da un singolo hit EFTS.

        Struttura risposta EFTS:
          _id     = "{adsh}:{document_filename}"  es. "0001213900-26-003766:ea0272591-01_424b2.htm"
          _source.adsh           = accession number con trattini
          _source.ciks           = lista CIK; l'ultimo è l'emittente effettivo
          _source.display_names  = lista nomi entity
          _source.file_date      = data filing

        Args:
            hit: singolo hit dalla risposta EFTS.

        Returns:
            dict: metadati con url corretto.
        """
        src = hit.get("_source", {})
        raw_id = hit.get("_id", "")          # "0001213900-26-003766:ea0272591-01_424b2.htm"

        # Separa accession number e document filename
        if ":" in raw_id:
            adsh, doc_filename = raw_id.split(":", 1)
        else:
            adsh        = raw_id
            doc_filename = ""

        # Usa il campo adsh se disponibile (più affidabile)
        adsh = src.get("adsh", adsh)
        acc_clean = adsh.replace("-", "")    # "000121390026003766"

        # CIK dell'emittente: l'ultimo nella lista (l'entità che ha depositato le note)
        ciks = src.get("ciks", [])
        cik  = ciks[-1] if ciks else ""

        # Nome display (l'ultimo elemento è l'emittente effettivo)
        display_names = src.get("display_names", [])
        entity_name   = display_names[-1] if display_names else src.get("entity_name", "")
        # Pulizia: rimuovi la parte "(CIK XXXXXXXXXX)"
        entity_name = entity_name.split(" (CIK")[0].strip()

        # URL diretto al documento
        if doc_filename and cik:
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{doc_filename}"
            )
        else:
            # Fallback all'index del filing
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik) if cik else 0}/{acc_clean}/{adsh}-index.htm"
            )

        return {
            "url":          filing_url,
            "accession_no": adsh,
            "entity_id":    cik,
            "entity_name":  entity_name,
            "filing_date":  src.get("file_date", ""),
            "form_type":    src.get("form", "") or (src.get("root_forms", ["424B2"])[0]),
        }

    def search(
        self,
        query: str,
        forms: list[str] | None = None,
        start_date: str | None = None,
    ) -> list[dict]:
        """Ricerca paginata e ritorna tutti i filing trovati.

        Args:
            query: termine di ricerca.
            forms: form types da filtrare.
            start_date: data di inizio (YYYY-MM-DD).

        Returns:
            list[dict]: lista di filing con url, entity_name, filing_date, form_type.
        """
        cfg   = self._cfg
        forms = forms or cfg["forms"]
        start = start_date or cfg["start_date"]

        results: list[dict] = []
        page_size = cfg["page_size"]
        offset    = 0

        while True:
            params = {
                "q":        f'"{query}"',
                "dateRange": "custom",
                "startdt":  start,
                "forms":    ",".join(forms),
                "from":     offset,
                "size":     page_size,
            }
            data  = self._get(params)
            hits  = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {}).get("value", 0)

            if not hits:
                _log.info("Query %r: nessun altro risultato a offset=%d", query, offset)
                break

            _log.info("Query %r: %d/%d risultati (offset=%d)", query, len(hits), total, offset)

            for hit in hits:
                results.append(self._build_url(hit))

            offset += len(hits)
            if offset >= total:
                break

        return results

    def collect_all_filings(self) -> list[dict]:
        """Raccoglie filing per tutti i termini configurati, deduplicando.

        Returns:
            list[dict]: lista deduplicata di filing.
        """
        seen:    set[str] = set()
        all_res: list[dict] = []

        for term in self._cfg["search_terms"]:
            filings = self.search(query=term)
            for f in filings:
                key = f["accession_no"]
                if key in seen:
                    continue
                seen.add(key)
                all_res.append(f)

        _log.info("Totale filing unici: %d", len(all_res))
        return all_res


def main() -> None:
    """Entry point CLI: stampa i filing trovati."""
    import json
    searcher = EdgarEftsSearcher()
    filings  = searcher.collect_all_filings()
    print(json.dumps(filings[:10], indent=2))
    print(f"\nTotale: {len(filings)} filing trovati")


if __name__ == "__main__":
    main()
