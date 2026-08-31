"""Fetch unificato dei dati macro (funding/OI/L-S/liquidazioni).

Singola fonte di verità usata da /api/signals, /api/macro e dashboard data_loader.
Sostituisce i 3 blocchi duplicati di fetch macro che avevano caching inconsistente.

Due fonti, in ordine di preferenza:

* **CoinGlass** copre tutti e cinque i fattori, ma richiede una chiave a pagamento.
* **CoinGecko** ne copre due — funding rate e open interest — e funziona senza
  chiave. Non espone long/short ratio né liquidazioni, e non ha storico: la
  variazione a 7 giorni dell'OI arriva dagli snapshot che accumuliamo noi in
  ``macro_snapshots``.

Il ripiego non è un sostituto e non deve spacciarsi per tale: quando i numeri
vengono da CoinGecko lo stato è ``partial_coingecko``, non ``ok``, così la card
del punteggio può dire che due fattori su cinque mancano ancora.

Ogni campo può essere ``None``, ma ``source_status`` dice **perché**: senza quella
distinzione "non lo sappiamo" e "il mercato è piatto" sono lo stesso valore, e il
pilastro macro finisce per pubblicare un giudizio che nessun dato sostiene.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from src.flows.coingecko_client import CoinGeckoClient
from src.flows.coinglass_client import CoinGlassClient, CoinGlassError

_log = logging.getLogger(__name__)

#: Almeno un fattore è stato letto davvero.
STATUS_OK = "ok"

#: Nessuna chiave CoinGlass configurata — è un problema di configurazione, non di
#: mercato, e si risolve valorizzando COINGLASS_API_KEY nella spec DO.
STATUS_NO_API_KEY = "no_api_key"

#: Chiave presente ma nessun dato ottenuto (API giù, tier insufficiente, rate limit).
STATUS_UNAVAILABLE = "unavailable"

#: Funding e open interest letti da CoinGecko; long/short e liquidazioni mancano.
#: Non è ``ok`` (il pilastro è coperto a metà) e non è ``no_api_key`` (due fattori
#: su cinque ci sono davvero): serviva un quarto stato per non mentire in nessuna
#: delle due direzioni.
STATUS_PARTIAL_COINGECKO = "partial_coingecko"

#: Etichette di provenienza per ``MacroData.funding_source``.
SOURCE_COINGLASS = "coinglass"
SOURCE_COINGECKO = "coingecko"


@dataclass
class MacroData:
    funding_rate_annualized_pct: float | None = None
    oi_change_7d_pct: float | None = None
    long_short_ratio: float | None = None
    liquidations_long_24h_usd: float | None = None
    liquidations_short_24h_usd: float | None = None
    oi_usd: float | None = None
    funding_source: str | None = None
    source_status: str = STATUS_UNAVAILABLE

    def to_dict(self) -> dict:
        return {
            "funding_rate_annualized_pct": self.funding_rate_annualized_pct,
            "oi_change_7d_pct": self.oi_change_7d_pct,
            "long_short_ratio": self.long_short_ratio,
            "liquidations_long_24h_usd": self.liquidations_long_24h_usd,
            "liquidations_short_24h_usd": self.liquidations_short_24h_usd,
            "oi_usd": self.oi_usd,
            "funding_source": self.funding_source,
            "source_status": self.source_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MacroData:
        return cls(
            funding_rate_annualized_pct=d.get("funding_rate_annualized_pct"),
            oi_change_7d_pct=d.get("oi_change_7d_pct"),
            long_short_ratio=d.get("long_short_ratio"),
            liquidations_long_24h_usd=d.get("liquidations_long_24h_usd"),
            liquidations_short_24h_usd=d.get("liquidations_short_24h_usd"),
            oi_usd=d.get("oi_usd"),
            funding_source=d.get("funding_source"),
            # una cache serializzata prima di questi campi resta leggibile
            source_status=d.get("source_status", STATUS_UNAVAILABLE),
        )

    def has_any_value(self) -> bool:
        """Vero se almeno un fattore è stato ottenuto."""
        return any(
            v is not None
            for v in (
                self.funding_rate_annualized_pct,
                self.oi_change_7d_pct,
                self.long_short_ratio,
                self.liquidations_long_24h_usd,
                self.liquidations_short_24h_usd,
            )
        )


def _fetch_coinglass(cg, out: MacroData) -> bool:
    """Riempie da CoinGlass i campi ancora vuoti. Vero se ne ha riempito almeno uno.

    Ogni fattore ha il suo try: un tier che non copre le liquidazioni non deve
    portarsi via anche il funding.
    """
    prima = out.has_any_value()

    if out.funding_rate_annualized_pct is None:
        try:
            fr = cg.fetch_funding_rate_history(days=14)
            if not fr.empty:
                # CoinGlass restituisce una frazione (0.0001 = 0,01% per 8h):
                # il x100 qui e' quello che CoinGecko *non* vuole.
                out.funding_rate_annualized_pct = float(fr.iloc[-1]) * 3 * 365 * 100
                out.funding_source = SOURCE_COINGLASS
        except (CoinGlassError, requests.RequestException) as exc:
            _log.warning("Funding rate fetch failed: %s", exc)

    if out.oi_change_7d_pct is None:
        try:
            oi = cg.fetch_aggregated_oi_history(days=14)
            if len(oi) >= 8 and float(oi.iloc[-8]) > 0:
                out.oi_change_7d_pct = (
                    (float(oi.iloc[-1]) - float(oi.iloc[-8]))
                    / float(oi.iloc[-8]) * 100
                )
            if len(oi) and out.oi_usd is None and float(oi.iloc[-1]) > 0:
                out.oi_usd = float(oi.iloc[-1])
        except (CoinGlassError, requests.RequestException, IndexError) as exc:
            _log.warning("OI history fetch failed: %s", exc)

    if out.long_short_ratio is None:
        try:
            ls = cg.fetch_long_short_ratio(days=3)
            if not ls.empty:
                out.long_short_ratio = float(ls.iloc[-1])
        except (CoinGlassError, requests.RequestException) as exc:
            _log.warning("Long/short ratio fetch failed: %s", exc)

    if out.liquidations_long_24h_usd is None:
        try:
            liq = cg.fetch_liquidations(days=2)
            if not liq.empty:
                out.liquidations_long_24h_usd = float(liq["long_usd"].iloc[-1])
                out.liquidations_short_24h_usd = float(liq["short_usd"].iloc[-1])
        except (CoinGlassError, requests.RequestException) as exc:
            _log.warning("Liquidations fetch failed: %s", exc)

    return out.has_any_value() and not prima


def _oi_change_dallo_storico(notes_db, days: int = 7) -> float | None:
    """Variazione dell'OI dagli snapshot che accumuliamo noi.

    CoinGecko da' il livello istantaneo, non la serie: senza questo la finestra a
    7 giorni resterebbe scoperta per sempre. Se il DB non e' leggibile la
    variazione resta ``None`` — un fattore in meno, non un fetch fallito.
    """
    try:
        db = notes_db
        if db is None:
            from src.edgar.structured_notes_db import StructuredNotesDB

            db = StructuredNotesDB()
        return db.get_oi_change_pct(days)
    except Exception as exc:  # noqa: BLE001 — lo storico e' un di piu', non un requisito
        _log.warning("Storico OI non leggibile: %s", exc)
        return None


def _fetch_coingecko(gecko, notes_db, out: MacroData) -> bool:
    """Riempie funding e open interest da CoinGecko. Vero se ha riempito qualcosa.

    Non tocca long/short ratio e liquidazioni: CoinGecko non li espone, e un
    campo riempito a caso e' peggio di un campo vuoto.
    """
    funding, oi_usd, n_contratti = gecko.fetch_funding_and_oi()
    if funding is None:
        return False

    out.funding_rate_annualized_pct = funding
    out.funding_source = SOURCE_COINGECKO
    if oi_usd is not None:
        out.oi_usd = oi_usd
    if out.oi_change_7d_pct is None:
        out.oi_change_7d_pct = _oi_change_dallo_storico(notes_db)

    _log.info(
        "Macro da CoinGecko: funding %.2f%% ann su %d perpetui (long/short e "
        "liquidazioni restano scoperti)", funding, n_contratti,
    )
    return True


def fetch_macro_data(
    *,
    cg_client=None,
    gecko_client=None,
    notes_db=None,
    cache_data: dict | None = None,
) -> MacroData:
    """Fetch dati macro con CoinGlass preferito e CoinGecko di ripiego.

    Args:
        cg_client: CoinGlassClient opzionale (se None, ne crea uno nuovo).
        gecko_client: CoinGeckoClient opzionale, interrogato solo se il funding
                      manca ancora dopo CoinGlass.
        notes_db: StructuredNotesDB opzionale, per lo storico dell'open interest.
        cache_data: dict opzionale da cui leggere valori pre-esistenti
                    (es. cache_get("macro_data") nell'API).

    Returns:
        MacroData con i valori disponibili (None per quelli non fetchabili) e
        ``source_status`` che spiega il perché di eventuali None.
    """
    cache_data = cache_data or {}
    out = MacroData.from_dict(cache_data)
    cg = cg_client or CoinGlassClient()

    da_coinglass = False
    if cg.has_api_key:
        da_coinglass = _fetch_coinglass(cg, out)
    else:
        # Senza chiave ogni chiamata fallirebbe: cinque richieste e cinque warning
        # per niente, a ogni giro di /api/signals. Meglio saltarle e ripiegare.
        _log.info(
            "COINGLASS_API_KEY non configurata: uso CoinGecko per funding e open "
            "interest. Long/short ratio e liquidazioni restano scoperti."
        )

    # CoinGecko entra in gioco solo se il funding manca: e' quello il fattore che
    # porta, e chiedere una risposta da 8 MB per un dato che abbiamo gia' sarebbe
    # solo banda sprecata.
    da_coingecko = False
    if out.funding_rate_annualized_pct is None:
        try:
            da_coingecko = _fetch_coingecko(
                gecko_client or CoinGeckoClient(), notes_db, out
            )
        except Exception as exc:  # noqa: BLE001 — il ripiego non deve poter rompere il fetch
            _log.warning("Ripiego CoinGecko fallito: %s", exc)

    if da_coinglass:
        out.source_status = STATUS_OK
    elif da_coingecko:
        out.source_status = STATUS_PARTIAL_COINGECKO
    elif out.has_any_value():
        # Solo valori dalla cache: conserva lo stato con cui erano stati scritti,
        # altrimenti un ripiego riletto dalla cache si promuoverebbe da solo a `ok`.
        stato = cache_data.get("source_status")
        out.source_status = (
            stato if stato in (STATUS_OK, STATUS_PARTIAL_COINGECKO) else STATUS_OK
        )
    elif not cg.has_api_key:
        out.source_status = STATUS_NO_API_KEY
    else:
        out.source_status = STATUS_UNAVAILABLE

    return out
