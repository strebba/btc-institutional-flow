"""Parser per i prospetti SEC 424B2/424B3 relativi a note strutturate su IBIT.

Estrae: emittente, date, nozionale, tipo prodotto, barrier levels, initial level,
coupon, e altri parametri chiave dai testi HTML dei filing EDGAR.
"""
from __future__ import annotations

import re
import time
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

# Pattern aggiuntivi per nozionale — coprono formati come "$1,500,000 aggregate..."
# e "Total: $5.2 million" usati da alcuni emittenti
_RE_NOTIONAL_ALT2 = re.compile(
    r"\$([\d,]+(?:\.\d+)?)\s+(?:aggregate\s+principal|notional|face\s+value|"
    r"principal\s+amount)",
    re.IGNORECASE,
)

_RE_NOTIONAL_PLAIN_M = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)\b",
    re.IGNORECASE,
)

# Pattern prioritario e affidabile per la size dell'offering nei supplement finali:
# "Aggregate principal amount: $5,000,000".
_RE_NOTIONAL_AGGREGATE = re.compile(
    r"aggregate\s+principal\s+amount\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
    re.IGNORECASE,
)

# Soglia sotto la quale un "notional" è in realtà la denominazione per-nota
# (es. $1,000 "stated principal amount"), non la size dell'offering.
_MIN_NOTIONAL_USD = 50_000.0

# Filing preliminare ("subject to completion"): Initial Value e aggregate principal
# non sono ancora fissati → non vanno inventati.
_RE_PRELIMINARY = re.compile(
    r"subject\s+to\s+completion|preliminary\s+pricing\s+supplement|"
    r"preliminary\s+prospectus\s+supplement",
    re.IGNORECASE,
)

_RE_INITIAL_LEVEL = re.compile(
    r"(?:initial\s+(?:value|level|price)|starting\s+(?:value|level)|"
    r"reference\s+(?:price|value|level)|initial\s+share\s+price)"
    r"\s*(?:of\s+the\s+underlying\s*)?:?\s*\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Pattern aggiuntivi per initial_level usati da Morgan Stanley e formati recenti JPMorgan
_RE_INITIAL_LEVEL_ALT = re.compile(
    r"(?:"
    r"closing\s+(?:share\s+)?price\s+(?:of\s+(?:the\s+)?(?:underlying|IBIT)\s+)?:?\s*\$?([\d,]+\.?\d*)"
    r"|"
    r"valuation\s+date\s+(?:closing\s+)?(?:price|value)\s*:?\s*\$?([\d,]+\.?\d*)"
    r"|"
    r"(?:price|value)\s+of\s+(?:the\s+)?(?:underlying|IBIT|share)\s+on\s+the\s+"
    r"(?:pricing|trade)\s+date\s*:?\s*\$?([\d,]+\.?\d*)"
    r")",
    re.IGNORECASE,
)

# "Initial Value" … "$XX.XX" entro ~60 caratteri (tabella key-terms dei finali JPM).
# Prezzo vincolato a 1-3 cifre intere + 2-4 decimali per non agganciare nozionali.
_RE_INITIAL_VALUE_NEAR = re.compile(
    r"initial\s+value\b[^$]{0,60}?\$\s*(\d{1,3}\.\d{2,4})\b",
    re.IGNORECASE,
)

# Segnale forte e cross-issuer (Goldman: "Initial ETF price: $42.73, which is the
# closing price of the ETF on the pricing date"). Il prezzo di chiusura alla pricing
# date È l'initial level. Il suffisso "which is the closing price" esclude i valori
# ipotetici degli esempi (es. "$100.00 ... for illustrative purposes").
_RE_INITIAL_CLOSING_PRICE = re.compile(
    r"\$\s*(\d{1,3}\.\d{2,4})[^$]{0,45}?which\s+is\s+the\s+closing\s+price",
    re.IGNORECASE,
)

# Formato Morgan Stanley/UBS: "Strike value: $31.41, 84.90% of the initial underlying
# value" → initial = prezzo / (pct/100). È l'inverso di _RE_BARRIER_ABS (prezzo prima
# della percentuale).
_RE_BARRIER_ABS_REVERSE = re.compile(
    r"\$\s*([\d,]+\.\d{1,4})\s*,\s*([\d]+(?:\.\d+)?)\s*%\s+of\s+the\s+initial",
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

# Data di inception di IBIT (iShares Bitcoin Trust, quotato l'11-gen-2024).
# Una nota che referenzia IBIT non può avere pricing/issue date precedente: i
# filing più vecchi che matchano la ricerca sono falsi positivi del full-text
# search EDGAR (il termine "IBIT" matcha la sottostringa in "exhibit"/"prohibited").
_IBIT_INCEPTION = date(2024, 1, 1)

# Mappa di canonicalizzazione emittenti: substring (lowercase) → nome canonico.
# Serve a unificare le varianti del campo entity_name di EDGAR
# (es. "JPMorgan Chase Financial Co. LLC" e "HSBC USA INC /MD/") con i nomi
# usati in known_issuers. L'ordine conta: substring più specifiche prima.
_ISSUER_CANONICAL: list[tuple[str, str]] = [
    ("jpmorgan",          "JPMorgan"),
    ("jp morgan",         "JPMorgan"),
    ("morgan stanley",    "Morgan Stanley"),
    ("goldman",           "Goldman Sachs"),
    ("gs finance",        "Goldman Sachs"),  # "GS Finance Corp." (filer Goldman)
    ("citigroup",         "Citigroup"),
    ("citibank",          "Citigroup"),
    ("bank of nova scotia", "Bank of Nova Scotia"),
    ("scotiabank",        "Bank of Nova Scotia"),
    ("barclays",          "Barclays"),
    ("jefferies",         "Jefferies"),
    ("hsbc",              "HSBC"),
    ("merrill",           "Bank of America"),
    ("bank of america",   "Bank of America"),
    ("bofa",              "Bank of America"),
    ("wells fargo",       "Wells Fargo"),
    ("bnp paribas",       "BNP Paribas"),
    ("societe generale",  "Societe Generale"),
    ("credit suisse",     "Credit Suisse"),
    ("ubs",               "UBS"),
    ("royal bank of canada", "RBC"),
    ("toronto-dominion",  "TD"),
    ("td bank",           "TD"),
    ("marex",             "Marex"),
]

# Allowlist degli emittenti di note strutturate riconosciuti (i valori canonici).
# Serve a scartare i falsi positivi del full-text search "IBIT": prospetti di
# ETF/trust (iShares Bitcoin Trust, Franklin … Trust) e società non-bancarie
# (Flybondi, Biomotion, ReserveOne…) il cui 424B matcha "IBIT" ma NON sono note.
_NOTE_ISSUERS: frozenset[str] = frozenset(canonical for _, canonical in _ISSUER_CANONICAL)


# ─── Helper functions ─────────────────────────────────────────────────────────


def _known_issuer_or_none(name: Optional[str]) -> Optional[str]:
    """Ritorna l'emittente canonico SE `name` è un emittente di note noto.

    A differenza di `_canonicalize_issuer` (che ripiega sul nome ripulito), qui
    si ritorna None quando il nome non corrisponde a un emittente in allowlist:
    è il segnale che il filing NON è una nota strutturata su IBIT.
    """
    canonical = _canonicalize_issuer(name)
    return canonical if canonical in _NOTE_ISSUERS else None


def _canonicalize_issuer(name: Optional[str]) -> Optional[str]:
    """Normalizza il nome di un emittente a una forma canonica.

    Cerca una substring nota in `name` (case-insensitive) e ritorna il nome
    canonico corrispondente; se nessuna matcha, ritorna il nome ripulito.

    Args:
        name: nome grezzo (da known_issuers o entity_name di EDGAR).

    Returns:
        str | None: nome canonico, o None se input vuoto/None.
    """
    if not name:
        return None
    low = name.lower()
    for needle, canonical in _ISSUER_CANONICAL:
        if needle in low:
            return canonical
    # Nessun match: rimuovi suffissi di stato/forma societaria comuni e spazi doppi
    cleaned = re.sub(r"\s{2,}", " ", name).strip()
    return cleaned or None

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

    Tenta i pattern in ordine di specificità:
    1. _RE_NOTIONAL     — "aggregate principal amount ... $5M"
    2. _RE_NOTIONAL_ALT — "$5,000,000 aggregate principal"  (già esistente)
    3. _RE_NOTIONAL_ALT2 — "$5,000,000 notional/face value"
    4. _RE_NOTIONAL_PLAIN_M — "$5.2 million" (generico, usato per ultimo)

    Args:
        text: testo grezzo.

    Returns:
        float | None: nozionale in USD o None.
    """
    def _scaled(num: str, suffix: str) -> float:
        val = float(num.replace(",", ""))
        s = (suffix or "").lower()
        if s in ("million", "m"):
            val *= 1_000_000
        elif s in ("billion", "b"):
            val *= 1_000_000_000
        return val

    # Pattern 0 (prioritario) — "Aggregate principal amount: $X": la size reale
    # dell'offering nei supplement finali.
    m = _RE_NOTIONAL_AGGREGATE.search(text)
    if m:
        val = _scaled(m.group(1), m.group(2))
        if val >= _MIN_NOTIONAL_USD:
            return val

    # Pattern 1 — con suffisso milioni/miliardi
    m = _RE_NOTIONAL.search(text)
    if m:
        val = _scaled(m.group(1), m.group(2))
        if val >= _MIN_NOTIONAL_USD:
            return val

    # Pattern 2 — "$X,XXX aggregate principal" (già coperto da _RE_NOTIONAL_ALT)
    m = _RE_NOTIONAL_ALT.search(text)
    if m:
        val = float(m.group(1).replace(",", ""))
        if val >= _MIN_NOTIONAL_USD:
            return val

    # Pattern 3 — "$X notional/face value"
    m = _RE_NOTIONAL_ALT2.search(text)
    if m:
        val = float(m.group(1).replace(",", ""))
        if val >= _MIN_NOTIONAL_USD:
            return val

    # Pattern 4 — "$X million" generico (ultimo resort)
    m = _RE_NOTIONAL_PLAIN_M.search(text)
    if m:
        return float(m.group(1).replace(",", "")) * 1_000_000

    # Nessun pattern affidabile: meglio None che la denominazione per-nota ($1,000).
    return None


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
        # Scarta barriere con percentuale nulla o impossibile (>200%)
        if pct <= 0.0 or pct > 200.0:
            _log.debug("Barriera scartata (pct=%s fuori range)", pct)
            continue
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
        # Scarta se il prezzo IBIT calcolato è zero (initial_level=0)
        if price_ibit is not None and price_ibit <= 0.0:
            _log.debug("Barriera scartata (price_ibit=%s <= 0)", price_ibit)
            continue

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

        # Un "preliminary pricing supplement" non ha ancora Initial Value né la
        # size dell'offering: lo segnaliamo per non inventare valori più sotto.
        is_preliminary = bool(_RE_PRELIMINARY.search(text))

        # L'emittente si deduce dall'entity_name di EDGAR (il FILER autorevole),
        # non dal testo del corpo: lì compaiono nomi di banche come dealer/agent
        # (es. "UBS") che causavano mis-attribuzioni. Se l'entity_name non è un
        # emittente di note noto → issuer=None → il filing è un falso positivo
        # (prospetto ETF/trust o società non pertinente) e viene scartato in
        # parse_batch. Fallback al testo solo se l'entity_name manca del tutto.
        issuer = _known_issuer_or_none(filing_meta.get("entity_name"))
        if issuer is None and not filing_meta.get("entity_name"):
            issuer = _known_issuer_or_none(_detect_issuer(text, known_issuers))
        notional      = _parse_notional(text)
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

        # 2) Pattern alternativi: Morgan Stanley "Closing price: $XX" e varianti
        if initial_level is None:
            m_alt = _RE_INITIAL_LEVEL_ALT.search(text)
            if m_alt:
                raw = next((g for g in m_alt.groups() if g), None)
                if raw:
                    try:
                        val = float(raw.replace(",", ""))
                        if 1.0 < val < 500.0:
                            initial_level = val
                    except ValueError:
                        pass

        # 3) Ricava initial level da "Barrier Amount: XX% ... which is $YY.YY"
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

        # 4) "Initial Value" seguito a breve da un prezzo IBIT "$XX.XX" (supplement
        #    finali JPMorgan: il numero è nella tabella key-terms, anche a qualche
        #    parola di distanza dalla label). Formato prezzo vincolato (1-3 cifre,
        #    2-4 decimali) per non agganciare importi grandi (es. nozionali).
        if initial_level is None:
            m_iv = _RE_INITIAL_VALUE_NEAR.search(text)
            if m_iv:
                try:
                    val = float(m_iv.group(1))
                    if 1.0 < val < 500.0:
                        initial_level = val
                except ValueError:
                    pass

        # 5) "$XX.XX, which is the closing price ... on the pricing date" (Goldman e
        #    altri): il prezzo di chiusura alla pricing date è l'initial level.
        if initial_level is None:
            m_cp = _RE_INITIAL_CLOSING_PRICE.search(text)
            if m_cp:
                try:
                    val = float(m_cp.group(1))
                    if 1.0 < val < 500.0:
                        initial_level = val
                except ValueError:
                    pass

        # 6) "$XX.XX, YY.YY% of the initial …" (Morgan Stanley/UBS): ricava l'initial
        #    dal prezzo assoluto di una barriera e dalla sua percentuale.
        if initial_level is None:
            m_rev = _RE_BARRIER_ABS_REVERSE.search(text)
            if m_rev:
                try:
                    price = float(m_rev.group(1).replace(",", ""))
                    pct   = float(m_rev.group(2)) / 100.0
                    if pct > 0 and 1.0 < price < 500.0:
                        initial_level = round(price / pct, 2)
                except (ValueError, ZeroDivisionError):
                    pass

        # Autocall trigger
        m_autocall = _RE_AUTOCALL_PCT.search(text)
        autocall_pct = float(m_autocall.group(1)) if m_autocall else None

        # Coupon
        m_coupon = _RE_COUPON.search(text)
        coupon_rate = float(m_coupon.group(1)) if m_coupon else None

        # Date: issue e maturity.
        # La filing_date di EDGAR è la fonte autorevole per la data di pricing/emissione
        # di un 424B2 (pricing supplement): usarla evita di scambiare per "issue date"
        # una data di scadenza/osservazione citata in testa al documento (bug storico:
        # issue_date nel 2030/2031). Il testo serve solo a ricavare la maturity.
        text_dates = sorted({
            d for d in (
                _parse_date(s) for s in (_RE_DATE.findall(text) + _RE_DATE_ISO.findall(text))
            ) if d
        })

        issue_date: Optional[date] = None
        if filing_meta.get("filing_date"):
            issue_date = _parse_date(filing_meta["filing_date"])
        if issue_date is None and text_dates:
            # Fallback: la prima data plausibile nel testo
            issue_date = text_dates[0]

        # Maturity: la data futura più lontana rispetto all'issue, entro ~15 anni
        # (le note IBIT hanno tenor tipicamente 1-5y; il cap scarta date-rumore).
        maturity_date: Optional[date] = None
        if issue_date:
            future = [d for d in text_dates if d > issue_date and d.year <= issue_date.year + 15]
            maturity_date = max(future) if future else None
        elif text_dates:
            maturity_date = text_dates[-1]

        # Nei preliminari Initial Value e size dell'offering non sono ancora fissati:
        # non inventarli. Le barriere percentuali (es. "70% dell'Initial Value") restano
        # valide; senza initial_level il loro prezzo assoluto sarà None (atteso).
        if is_preliminary:
            notional      = None
            initial_level = None

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

        # Snippet di testo per debug. Tenuto corto (4k) di proposito: raw_text non è
        # mai riletto a valle (né API né DB read-path), quindi 50k gonfiavano solo il
        # file SQLite versionato (~14MB). 4k bastano per ispezionare un filing.
        raw = text[:4_000]

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
            is_preliminary=is_preliminary,
            barriers=barriers,
            raw_text=raw,
        )

        _log.info(
            "Estratto: issuer=%s type=%s notional=%.0f initial=%.2f barriers=%d%s",
            issuer or "?",
            product_type or "?",
            notional or 0,
            initial_level or 0,
            len(barriers),
            " [preliminary]" if is_preliminary else "",
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

            # Scarta i falsi positivi: una nota IBIT non può precedere l'inception
            # dell'ETF (cfr. _IBIT_INCEPTION). Questi filing matchano la ricerca solo
            # perché "IBIT" compare in "exhibit"/"prohibited".
            if note.issue_date and note.issue_date < _IBIT_INCEPTION:
                _log.info("Scartato (pre-IBIT %s): %s", note.issue_date, filing["url"])
                continue

            # Scarta i filing il cui FILER non è un emittente di note noto: prospetti
            # di ETF/trust (iShares, Franklin) e società non pertinenti il cui 424B
            # matcha "IBIT" ma non sono note strutturate (cfr. _known_issuer_or_none).
            if note.issuer is None:
                _log.info("Scartato (filer non-emittente: %s): %s",
                          filing.get("entity_name", "?"), filing["url"])
                continue

            # Superati i filtri: il filer è un emittente noto → è una nota vera.
            results.append(note)

        _log.info("Parsing completato: %d/%d note con dati", len(results), len(to_parse))
        return results
