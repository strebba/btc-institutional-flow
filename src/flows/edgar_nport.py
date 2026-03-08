"""Client per EDGAR N-PORT — flussi mensili IBIT da SEC (interpolati a daily)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import requests

from src.config import get_settings, setup_logging
from src.flows.models import EtfFlowData

_log = setup_logging("flows.edgar_nport")

_DEFAULT_CIK    = "0001980994"           # IBIT CIK su EDGAR
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVES_BASE   = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
_MAX_STALE_DAYS  = 45                   # warning se ultimo filing è troppo vecchio


class EdgarNportClient:
    """Scarica i filing N-PORT di IBIT da SEC EDGAR e stima flussi mensili.

    Logica:
    1. Ottiene la lista dei filing N-PORT dall'API EDGAR submissions.
    2. Per ogni filing scarica il documento XML primario.
    3. Estrae ``totalAssets`` e ``sharesOutstanding``.
    4. Calcola ``flow_mensile = Δ(shares) × avg_NAV`` e interpola linearmente
       al giornaliero.

    Args:
        cfg: configurazione completa del progetto (opzionale, caricata in auto).
        cik: CIK dell'ETF. Default: IBIT (0001980994).
    """

    def __init__(self, cfg: dict | None = None, cik: str | None = None) -> None:
        settings = cfg or get_settings()
        self._user_agent = settings.get("edgar", {}).get(
            "user_agent", "ibit-gamma-tracker/1.0 (contact@example.com)"
        )
        flows_cfg = settings.get("flows", {})
        raw_cik = cik or flows_cfg.get("edgar_nport_cik", _DEFAULT_CIK)
        self._cik = raw_cik.zfill(10)

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self._user_agent,
            "Accept":     "application/json",
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_monthly_flows(self, lookback_months: int = 13) -> list[EtfFlowData]:
        """Scarica e interpola flussi mensili da N-PORT filings EDGAR.

        Args:
            lookback_months: mesi di storico da considerare (default 13).

        Returns:
            list[EtfFlowData]: flussi giornalieri con
            ``source="edgar_nport_interpolated"``, o lista vuota in caso di
            errore o dati insufficienti.
        """
        filings = self._fetch_nport_index(lookback_months)
        if not filings:
            _log.warning("EDGAR N-PORT: nessun filing N-PORT trovato per CIK %s", self._cik)
            return []

        # Controlla staleness
        latest_date = max(f["date"] for f in filings)
        days_old = (date.today() - latest_date).days
        if days_old > _MAX_STALE_DAYS:
            _log.warning(
                "EDGAR N-PORT: ultimo filing del %s (%d gg fa) — dati potenzialmente stantii",
                latest_date.isoformat(), days_old,
            )

        monthly_points = self._extract_monthly_data(filings)
        if len(monthly_points) < 2:
            _log.warning(
                "EDGAR N-PORT: punti mensili insufficienti (%d) — impossibile interpolare",
                len(monthly_points),
            )
            return []

        return self._interpolate_to_daily(monthly_points)

    # ──────────────────────────────────────────────────────────────────────────
    # Fetch EDGAR index
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_nport_index(self, lookback_months: int) -> list[dict]:
        """Interroga EDGAR submissions API e ritorna i filing N-PORT nel periodo."""
        url = _SUBMISSIONS_URL.format(cik=self._cik)
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log.warning("EDGAR submissions fetch fallito: %s", e)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        dates_raw  = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])

        cutoff = date.today() - timedelta(days=lookback_months * 31)
        results: list[dict] = []
        cik_int = str(int(self._cik))

        for form, date_str, accession in zip(forms, dates_raw, accessions):
            if form != "NPORT-P":
                continue
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if filing_date < cutoff:
                continue
            results.append({
                "date":      filing_date,
                "accession": accession,
                "cik_int":   cik_int,
            })

        _log.info("EDGAR N-PORT: %d filing trovati (CIK %s)", len(results), self._cik)
        return sorted(results, key=lambda x: x["date"])

    # ──────────────────────────────────────────────────────────────────────────
    # Fetch + parse XML per ogni filing
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_monthly_data(self, filings: list[dict]) -> list[dict]:
        """Scarica e parsa ogni filing N-PORT; ritorna lista di punti mensili."""
        points: list[dict] = []
        for filing in filings:
            point = self._fetch_filing_data(filing)
            if point:
                points.append(point)
        return points

    def _fetch_filing_data(self, filing: dict) -> dict | None:
        """Scarica il documento XML primario di un filing N-PORT e lo parsa."""
        cik_int   = filing["cik_int"]
        accession = filing["accession"].replace("-", "")
        base_url  = _ARCHIVES_BASE.format(cik=cik_int, acc=accession)

        # 1. Prova a ottenere l'indice del filing per trovare il file XML
        xml_text = self._try_index_fetch(cik_int, accession, base_url)

        # 2. Fallback: nomi candidati comuni per file XML N-PORT
        if not xml_text:
            xml_text = self._try_candidate_files(base_url, accession)

        if not xml_text:
            _log.debug("EDGAR N-PORT: nessun XML trovato per %s", accession)
            return None

        return self._parse_nport_xml_text(xml_text, filing["date"])

    def _try_index_fetch(self, cik_int: str, accession: str, base_url: str) -> str | None:
        """Ottiene il file XML dal manifest JSON del filing."""
        index_url = f"{base_url}{accession}-index.json"
        try:
            resp = self._session.get(index_url, timeout=20)
            resp.raise_for_status()
            idx_data = resp.json()
        except Exception:
            return None

        for item in idx_data.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.endswith(".xml"):
                try:
                    r = self._session.get(base_url + name, timeout=30)
                    if r.status_code == 200:
                        return r.text
                except Exception:
                    continue
        return None

    def _try_candidate_files(self, base_url: str, accession: str) -> str | None:
        """Tenta nomi file XML comuni per filing N-PORT."""
        candidates = [
            f"{accession}.xml",
            "primary_doc.xml",
            "nport.xml",
            "nportp.xml",
        ]
        for name in candidates:
            try:
                r = self._session.get(base_url + name, timeout=20)
                if r.status_code == 200:
                    return r.text
            except Exception:
                continue
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # XML parsing (testabile senza rete)
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_nport_xml_text(self, xml_text: str, filing_date: date) -> dict | None:
        """Parsa testo XML N-PORT ed estrae totalAssets e sharesOutstanding.

        Usa ElementTree con fallback regex per massima robustezza ai diversi
        schemi N-PORT (il formato è cambiato tra le versioni EDGAR).

        Args:
            xml_text: contenuto XML grezzo.
            filing_date: data del filing (usata nel punto mensile restituito).

        Returns:
            dict con keys ``date``, ``total_assets``, ``shares_outstanding``,
            ``nav_per_share``; o ``None`` se i dati non sono disponibili.
        """
        total_assets       = self._extract_xml_value(xml_text, [
            "totAssets", "totalAssets", "tot-assets",
        ])
        shares_outstanding = self._extract_xml_value(xml_text, [
            "shrOutstanding", "sharesOutstanding", "shr-outstanding",
            "outstandingShares",
        ])
        nav_per_share      = self._extract_xml_value(xml_text, [
            "navPerShare", "nav-per-share", "netAssetValuePerShare",
        ])

        if total_assets is None and shares_outstanding is None:
            _log.debug("EDGAR N-PORT: totalAssets e sharesOutstanding non trovati")
            return None

        # Stima NAV/share se non disponibile
        if (
            nav_per_share is None
            and total_assets
            and shares_outstanding
            and shares_outstanding > 0
        ):
            nav_per_share = total_assets / shares_outstanding

        return {
            "date":               filing_date,
            "total_assets":       total_assets,
            "shares_outstanding": shares_outstanding,
            "nav_per_share":      nav_per_share,
        }

    @staticmethod
    def _extract_xml_value(xml_text: str, tag_names: list[str]) -> float | None:
        """Estrae il primo valore numerico che corrisponde a uno dei tag.

        Prova prima ElementTree (con e senza namespace), poi fallback regex.
        """
        # --- ElementTree ---
        try:
            root = ET.fromstring(xml_text)
            # Estrai namespace dalla radice (es. "http://www.sec.gov/edgar/nport")
            ns_match = re.match(r"\{([^}]+)\}", root.tag)
            ns = ns_match.group(1) if ns_match else None

            for tag in tag_names:
                # Senza namespace
                el = root.find(f".//{tag}")
                if el is None and ns:
                    el = root.find(f"{{{ns}}}{tag}")
                if el is None and ns:
                    el = root.find(f".//{{{ns}}}{tag}")
                if el is not None and el.text:
                    try:
                        return float(el.text.strip().replace(",", ""))
                    except ValueError:
                        pass
        except ET.ParseError:
            pass

        # --- Regex fallback ---
        for tag in tag_names:
            pattern = rf"<{re.escape(tag)}[^>]*>\s*([\d,.\-]+)\s*</"
            m = re.search(pattern, xml_text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    pass
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Interpolazione mensile → giornaliero (testabile senza rete)
    # ──────────────────────────────────────────────────────────────────────────

    def _interpolate_to_daily(self, monthly_points: list[dict]) -> list[EtfFlowData]:
        """Interpola i punti mensili in flussi giornalieri.

        Flusso mensile = Δ(shares) × avg_NAV.
        Distribuito uniformemente sui giorni tra filing consecutivi.

        Args:
            monthly_points: lista di dict con ``date``, ``shares_outstanding``,
                ``nav_per_share`` (ordinata per data).

        Returns:
            list[EtfFlowData]: flussi giornalieri con
            ``source="edgar_nport_interpolated"``.
        """
        results: list[EtfFlowData] = []

        for i in range(1, len(monthly_points)):
            prev = monthly_points[i - 1]
            curr = monthly_points[i]

            prev_shares = prev.get("shares_outstanding")
            curr_shares = curr.get("shares_outstanding")
            prev_nav    = prev.get("nav_per_share")
            curr_nav    = curr.get("nav_per_share")

            if prev_shares is None or curr_shares is None:
                continue

            delta_shares  = curr_shares - prev_shares
            avg_nav       = _avg(prev_nav, curr_nav)
            if avg_nav == 0:
                continue

            monthly_flow = delta_shares * avg_nav

            start_d  = prev["date"]
            end_d    = curr["date"]
            num_days = max((end_d - start_d).days, 1)
            daily_flow = monthly_flow / num_days

            for offset in range(num_days):
                day = start_d + timedelta(days=offset + 1)
                if day > date.today():
                    break
                results.append(EtfFlowData(
                    date=day,
                    ticker="IBIT",
                    flow_usd=daily_flow,
                    source="edgar_nport_interpolated",
                ))

        _log.info("EDGAR N-PORT: %d flussi giornalieri interpolati", len(results))
        return results


def _avg(a: float | None, b: float | None) -> float:
    """Media di due valori, gestisce None.

    Se entrambi i valori sono disponibili restituisce la media aritmetica.
    Se uno solo è disponibile restituisce quel valore (media di un singolo punto).
    """
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else (b or 0.0)
