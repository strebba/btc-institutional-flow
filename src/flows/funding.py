"""Conversione del funding rate: la convenzione unica per tutte le fonti.

Esiste per una ragione sola: questa conversione era duplicata in cinque punti e
in quattro era sbagliata. Un funding rate si annualizza su tre finestre da 8 ore
al giorno per 365 giorni. Il fattore ``×100`` che si vede in giro serve **solo**
se la fonte restituisce una frazione — e nessuna delle due che usiamo lo fa.

Misurato sulle API vere, stessa metrica OI-weighted nello stesso istante:

===========  ==========  =============================
fonte        valore 8h   convenzione
===========  ==========  =============================
CoinGecko    0,007499    percentuale (documentata)
CoinGlass    0,005477    percentuale (verificata)
===========  ==========  =============================

Il rapporto fra le due è 0,73×. Se le convenzioni fossero diverse sarebbe ~100×.

Sbagliarlo non dà un numero un po' storto: dà 599% invece di 6%, e lo scorer del
funding legge "estremo: flush imminente" dove il mercato dice "caldo ma compresso
dal pin gamma". Per questo la conversione sta qui e non nei chiamanti.
"""
from __future__ import annotations

#: Tre finestre di funding al giorno, da 8 ore ciascuna.
FUNDING_WINDOWS_PER_DAY = 3
DAYS_PER_YEAR = 365


def annualize_funding_pct(rate_pct_8h: float) -> float:
    """Annualizza un funding rate già espresso in punti percentuali per 8 ore.

    Args:
        rate_pct_8h: il tasso come lo restituiscono CoinGlass e CoinGecko —
            ``0.01`` significa *lo 0,01% per 8 ore*, non l'1%.

    Returns:
        Il tasso annualizzato in punti percentuali: ``0.01`` → ``10.95``.
    """
    return rate_pct_8h * FUNDING_WINDOWS_PER_DAY * DAYS_PER_YEAR
