"""Parser per i prospetti SEC 424B2/424B3 relativi a note strutturate su IBIT.

Estrae: emittente, date, nozionale, tipo prodotto, barrier levels, initial level,
coupon, e altri parametri chiave dai testi HTML dei filing EDGAR.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings, setup_logging
from src.edgar.models import BarrierLevel, StructuredNote

_log = setup_logging("edgar.parser")

# ─── Regex patterns ──────────────────────────────────────────────────────────

_RE_NOTIONAL = re.compile(
    r"(?:aggregate\s+principal\s+amount|notional\s+amount|total\s+offering)"
    r"\s*(?:of\s+(?:the\s+notes\s+is\s+)?|:?\s*)"
    r"\$?([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
    re.IGNORECASE,
)

_RE_INITIAL_LEVEL = re.compile(
    r"(?:initial\s+(?:value|level|price)|starting\s+(?:value|level)|"
    r"reference\s+(?:price|value|level)|initial\s+share\s+price)"
    r"\s*(?:of\s+the\s+underlying\s*)?:?\s*\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_RE_BARRIER_PCT = re.compile(
    r"([\d]+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?:initial|starting|reference)\s+"
    r"(?:value|level|price|share\s+price)",
    re.IGNORECASE,
)

_RE_BARRIER_LABEL = re.compile(
    r"(?:barrier|trigger|knock[\-\s]in\s+level|threshold|"
    r"buffer\s+level|downside\s+threshold|protection\s+level)"
    r"[:\s]+([^\n<]{3,80})",
    re.IGNORECASE,
)

_RE_AUTOCALL_PCT = re.compile(
    r"(?:auto[\-\s]?call\s+(?:trigger|level|threshold)|call\s+(?:level|trigger|value))"
    r"\s*:?\s*(?:1\s+)?([\d]+(?:\.\d+)?)\s*%",   # gestisce "10 0.00%" → "100.00%"
    re.IGNORECASE,
)

# Pattern JPMorgan: "Barrier Amount: 55.00% of the Initial Value, which is $28.138"
_RE_BARRIER_ABS = re.compile(
    r"(?:barrier\s+(?:amount|level)|trigger\s+level|knock[\-\s]in\s+level)"
    r"\s*:?\s*([\d]+(?:\.\d+)?)\s*%\s+of\s+the\s+(?:initial|starting)\s+value"
    r",\s*which\s+is\s+\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Pattern per nozionale con $ diretto (es. "$5,000,000 aggregate principal")
_RE_NOTIONAL_ALT = re.compile(
    r"\$([\d,]+(?:\.\d+)?)\s+(?:aggregate\s+principal|notional|face\s+value)",
    re.IGNORECASE,
)

_RE_COUPON = re.compile(
    r"(?:contingent\s+(?:coupon|payment|interest)|coupon\s+rate|"
    r"guaranteed\s+(?:minimum\s+)?return|interest\s+rate)"
    r"\s*:?\s*([\d]+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

_RE_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)

_RE_DATE_ISO = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b")

_PRODUCT_KEYWORDS: dict[str, list[str]] = {
    "autocallable":         ["auto-callable", "autocallable", "auto callable", "auto-call"],
    "barrier_note":         ["barrier note", "knock-in note", "knock-in level"],
    "buffered_note":        ["buffered note", "buffer", "principal buffer"],
    "principal_protected":  ["principal protected", "capital protected", "100% principal protection"],
    "contingent_coupon":    ["contingent coupon", "contingent payment", "contingent interest"],
    "leveraged_note":       ["leveraged", "participation rate", "upside participation"],
}

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ─── Helper functions ─────────────────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    """Converte una stringa data in oggetto date.

    Supporta formati: 'January 15, 2024', '2024-01-15', '2024/01/15'.

    Args:
        s: stringa data.

    Returns:
        date | None: data parsata o None se non riconosciuta.
    """
    s = s.strip()
    # ISO
    m = re.match(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # Month name
    m = re.match(
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{1,2}),\s+(20\d{2})",
        s, re.IGNORECASE,
    )
    if m:
        month = _MONTH_MAP[m.group(1).lower()]
        return date(int(m.group(3)), month, int(m.group(2)))
    return None


def _parse_notional(text: str) -> Optional[float]:
    """Estrae il nozionale in USD dalla stringa.

    Args:
        text: testo grezzo.

    Returns:
        float | None: nozionale in USD o None.
    """
    m = _RE_NOTIONAL.search(text)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    suffix = (m.group(2) or "").lower()
    if suffix in ("million", "m"):
        val *= 1_000_000
    elif suffix in ("billion", "b"):
        val *= 1_000_000_000
    return val


def _detect_product_type(text: str) -> Optional[str]:
    """Classifica il tipo di prodotto in base a keyword.

    Args:
        text: testo del filing.

    Returns:
        str | None: tipo prodotto o None.
    """
    text_lower = text.lower()
    matches: list[str] = []
    for product, keywords in _PRODUCT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matches.append(product)
    if not matches:
        return None
    # Preferenza: autocallable > barrier_note > buffered > etc.
    priority = [
        "autocallable", "barrier_note", "buffered_note",
        "contingent_coupon", "principal_protected", "leveraged_note",
    ]
    for p in priority:
        if p in matches:
            return p
    return matches[0]


def _detect_issuer(text: str, known_issuers: list[str]) -> Optional[str]:
    """Cerca il nome dell'emittente nel testo.

    Args:
        text: testo del filing.
        known_issuers: lista di emittenti noti.

    Returns:
        str | None: nome emittente o None.
    """
    for issuer in known_issuers:
        if issuer.lower() in text.lower():
            return issuer
    return None


def _extract_barrier_levels(text: str, initial_level: Optional[float]) -> list[BarrierLevel]:
    """Estrae tutti i barrier levels dal testo.

    Cerca pattern come "70% of the Initial Value" e li classifica
    in knock_in, autocall, o buffer in base al contesto.

    Args:
        text: testo grezzo del filing.
        initial_level: prezzo IBIT iniziale (per calcolare il prezzo assoluto).

    Returns:
        list[BarrierLevel]: lista di barrier levels trovati.
    """
    barriers: list[BarrierLevel] = []
    seen_pcts: set[float] = set()

    # Pattern: "XX% of the Initial/Starting Value" con contesto
    pattern = re.compile(
        r"([\d]+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?:initial|starting|reference)\s+"
        r"(?:value|level|price|share\s+price)",
        re.IGNORECASE,
    )

    for m in pattern.finditer(text):
        pct     = float(m.group(1))
        if pct in seen_pcts:
            continue
        seen_pcts.add(pct)

        # Guarda il contesto PRIMA del match (il tipo di barriera precede sempre il valore %)
        start   = max(0, m.start() - 350)
        context = text[start:m.start()].lower()

        # Classifica il tipo di barriera — ordine: più specifico prima
        if any(kw in context for kw in ["knock-out", "knock out", "knockout"]):
            btype = "knock_out"
        elif any(kw in context for kw in ["auto-call", "autocall", "auto call",
                                           "auto call trigger", "call trigger"]):
            btype = "autocall"
        elif any(kw in context for kw in ["knock-in", "knock in", "knockin",
                                            "barrier level", "downside threshold"]):
            btype = "knock_in"
        elif any(kw in context for kw in ["buffer", "protection level", "principal protection"]):
            btype = "buffer"
        else:
            # Euristica: pct < 100 → probabile knock-in, pct >= 100 → autocall
            btype = "knock_in" if pct < 100 else "autocall"

        price_ibit = (initial_level * pct / 100.0) if initial_level else None

        barriers.append(BarrierLevel(
            barrier_type=btype,
            level_pct=pct,
            level_price_ibit=price_ibit,
        ))

    # Deduplica per (type, pct)
    unique: list[BarrierLevel] = []
    seen_combo: set[tuple[str, float]] = set()
    for b in barriers:
        key = (b.barrier_type, b.level_pct)
        if key not in seen_combo:
            seen_combo.add(key)
            unique.append(b)

    return unique


# ─── Main parser class ────────────────────────────────────────────────────────

class ProspectusParser:
    """Scarica e parsa un prospetto SEC 424B2/424B3 per estrarre i dati chiave.

    Args:
        cfg: configurazione edgar (da settings.yaml["edgar"]).
    """

    SECTION_HINTS = [
        "key terms", "summary of terms", "hypothetical examples",
        "payoff diagram", "terms and conditions", "description of the notes",
    ]

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg     = cfg or get_settings()["edgar"]
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":      self._cfg["user_agent"],
            "Accept-Encoding": "gzip, deflate",
        })
        self._min_interval = 1.0 / self._cfg["rate_limit_rps"]
        self._last_request: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait    = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _fetch(self, url: str) -> str:
        """Scarica il contenuto HTML di un filing.

        Args:
            url: URL del documento.

        Returns:
            str: testo HTML.

        Raises:
            requests.HTTPError: se la risposta non è 2xx.
        """
        self._throttle()
        _log.debug("Fetching: %s", url)
        resp = self._session.get(url, timeout=45)
        resp.raise_for_status()
        return resp.text

    def _resolve_actual_url(self, filing_meta: dict) -> str:
        """Restituisce l'URL diretto al documento del filing.

        L'URL viene già costruito correttamente da EdgarEftsSearcher._build_url,
        quindi lo usiamo direttamente senza passare per l'index page.

        Args:
            filing_meta: dict con url, entity_id, accession_no.

        Returns:
            str: URL diretto al documento.
        """
        return filing_meta["url"]

    def _extract_relevant_text(self, html: str) -> str:
        """Converte HTML in testo pulito, preferendo le sezioni chiave.

        Gestisce anche il formato PDF-converted usato da JPMorgan/altri emittenti
        dove il testo è spezzato in molti <div> con position:absolute.

        Args:
            html: testo HTML grezzo.

        Returns:
            str: testo pulito (max 100k caratteri).
        """
        soup = BeautifulSoup(html, "lxml")

        # Rimuovi script, style, header/footer
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        # Estrae testo pulito — usa separatore spazio per gestire PDF-converted HTML
        full_text = soup.get_text(separator=" ", strip=True)
        # Normalizza spazi multipli
        full_text = re.sub(r" {2,}", " ", full_text)

        # Prova a isolare sezioni rilevanti
        relevant_chunks: list[str] = []
        text_lower = full_text.lower()
        for hint in self.SECTION_HINTS:
            idx = text_lower.find(hint)
            if idx != -1:
                chunk = full_text[idx: idx + 8000]
                relevant_chunks.append(chunk)

        if relevant_chunks:
            focused = "\n\n".join(relevant_chunks)
            return (full_text[:5000] + "\n\n" + focused)[:100_000]

        return full_text[:100_000]

    def parse(self, filing_meta: dict) -> StructuredNote:
        """Scarica e parsa un filing, restituendo una StructuredNote.

        Args:
            filing_meta: dict con url, entity_name, filing_date, form_type,
                         entity_id, accession_no.

        Returns:
            StructuredNote: nota strutturata con i dati estratti.
        """
        url = self._resolve_actual_url(filing_meta)
        _log.info("Parsing: %s (%s)", filing_meta.get("entity_name", "?"), url)

        try:
            html = self._fetch(url)
        except Exception as e:
            _log.warning("Download fallito per %s: %s — ritento con URL originale", url, e)
            try:
                html = self._fetch(filing_meta["url"])
                url  = filing_meta["url"]
            except Exception as e2:
                _log.error("Download definitivamente fallito: %s", e2)
                return StructuredNote(filing_url=filing_meta["url"])

        text = self._extract_relevant_text(html)

        # ── Estrai campi ──────────────────────────────────────────────────
        known_issuers = self._cfg.get("known_issuers", [])

        issuer        = _detect_issuer(text, known_issuers) or filing_meta.get("entity_name")
        notional = _parse_notional(text)
        # Fallback nozionale con pattern alternativo
        if notional is None:
            m_n = _RE_NOTIONAL_ALT.search(text)
            if m_n:
                try:
                    notional = float(m_n.group(1).replace(",", ""))
                except ValueError:
                    pass
        product_type  = _detect_product_type(text)

        # Initial level — strategia in cascata
        initial_level: Optional[float] = None

        # 1) Pattern esplicito "Initial Value: $XX"
        m_init = _RE_INITIAL_LEVEL.search(text)
        if m_init and m_init.group(1):
            try:
                val = float(m_init.group(1).replace(",", ""))
                if 1.0 < val < 500.0:
                    initial_level = val
            except ValueError:
                pass

        # 2) Ricava initial level da "Barrier Amount: XX% ... which is $YY.YY"
        if initial_level is None:
            m_abs = _RE_BARRIER_ABS.search(text)
            if m_abs:
                try:
                    barrier_pct   = float(m_abs.group(1)) / 100.0
                    barrier_price = float(m_abs.group(2).replace(",", ""))
                    if barrier_pct > 0 and 1.0 < barrier_price < 500.0:
                        initial_level = round(barrier_price / barrier_pct, 4)
                        _log.debug("Initial level calcolato da barriera assoluta: %.4f", initial_level)
                except (ValueError, ZeroDivisionError):
                    pass

        # Autocall trigger
        m_autocall = _RE_AUTOCALL_PCT.search(text)
        autocall_pct = float(m_autocall.group(1)) if m_autocall else None

        # Coupon
        m_coupon = _RE_COUPON.search(text)
        coupon_rate = float(m_coupon.group(1)) if m_coupon else None

        # Date: issue e maturity
        all_dates = _RE_DATE.findall(text)
        issue_date    = _parse_date(all_dates[0]) if len(all_dates) > 0 else None
        maturity_date = _parse_date(all_dates[-1]) if len(all_dates) > 1 else None

        # Prova date ISO se non trovate
        if not issue_date:
            iso_dates = _RE_DATE_ISO.findall(text)
            if iso_dates:
                issue_date    = _parse_date(iso_dates[0])
                maturity_date = _parse_date(iso_dates[-1]) if len(iso_dates) > 1 else None

        # Override issue_date con filing_date se disponibile
        if not issue_date and filing_meta.get("filing_date"):
            issue_date = _parse_date(filing_meta["filing_date"])

        # Barrier levels
        barriers = _extract_barrier_levels(text, initial_level)

        # Knock-in pct (il livello più basso dei knock_in trovati)
        knockin_pcts = [b.level_pct for b in barriers if b.barrier_type == "knock_in"]
        knockin_pct  = min(knockin_pcts) if knockin_pcts else None

        # Buffer pct (inferita: 100 - min_knockin oppure da keyword)
        m_buf = re.search(
            r"(?:buffer|protection)\s+of\s+([\d]+(?:\.\d+)?)\s*%", text, re.IGNORECASE
        )
        buffer_pct = float(m_buf.group(1)) if m_buf else (
            (100 - knockin_pct) if knockin_pct else None
        )

        # Partecipazione
        m_part = re.search(
            r"participation\s+rate\s*:?\s*([\d]+(?:\.\d+)?)\s*%", text, re.IGNORECASE
        )
        participation = float(m_part.group(1)) if m_part else None

        # Truncate raw text
        raw = text[:50_000]

        note = StructuredNote(
            filing_url=url,
            issuer=issuer,
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional_usd=notional,
            product_type=product_type,
            underlying="IBIT",
            initial_level=initial_level,
            autocall_trigger_pct=autocall_pct,
            knockin_barrier_pct=knockin_pct,
            buffer_pct=buffer_pct,
            participation_rate=participation,
            coupon_rate=coupon_rate,
            barriers=barriers,
            raw_text=raw,
        )

        _log.info(
            "Estratto: issuer=%s type=%s notional=%.0f initial=%.2f barriers=%d",
            issuer or "?",
            product_type or "?",
            notional or 0,
            initial_level or 0,
            len(barriers),
        )

        return note

    def parse_batch(
        self, filings: list[dict], max_items: int | None = None
    ) -> list[StructuredNote]:
        """Parsa un batch di filing, ritornando solo quelli con dati rilevanti.

        Args:
            filings: lista di filing metadata da EdgarEftsSearcher.
            max_items: limite di filing da processare (None = tutti).

        Returns:
            list[StructuredNote]: note estratte con almeno un campo non None.
        """
        results: list[StructuredNote] = []
        to_parse = filings[:max_items] if max_items else filings

        for i, filing in enumerate(to_parse, 1):
            _log.info("[%d/%d] %s", i, len(to_parse), filing.get("entity_name", "?"))
            note = self.parse(filing)
            # Tieni solo le note con almeno qualcosa di utile
            if any([note.issuer, note.notional_usd, note.product_type, note.initial_level]):
                results.append(note)
            else:
                _log.warning("Filing senza dati utili: %s", filing["url"])

        _log.info("Parsing completato: %d/%d note con dati", len(results), len(to_parse))
        return results
