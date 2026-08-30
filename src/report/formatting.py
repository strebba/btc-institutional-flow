"""Formattazione numerica italiana per il Desk Note.

Le card mostrano un numero solo, quindi quel numero deve essere leggibile senza
sforzo: separatore migliaia col punto, decimali con la virgola, segno esplicito
dove il segno e' l'informazione.
"""
from __future__ import annotations

MINUS = "−"  # segno meno tipografico, non il trattino ASCII


def _it(value: float, decimals: int = 0) -> str:
    """Formatta con punto per le migliaia e virgola per i decimali."""
    s = f"{abs(value):,.{decimals}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _signed(value: float, body: str, *, force_sign: bool) -> str:
    if value < 0:
        return f"{MINUS}{body}"
    return f"+{body}" if force_sign else body


def price(value: float | None) -> str:
    """Prezzo in dollari senza decimali: 77703.9 -> '77.704'."""
    if value is None:
        return "n/d"
    return _it(round(value), 0)


def usd_millions(value: float | None, *, force_sign: bool = True) -> str:
    """USD in milioni/miliardi con una cifra decimale: 171930539 -> '+171,9M'.

    Sopra il miliardo passa a 'B' perche' '63.365M' non si legge a colpo d'occhio.
    """
    if value is None:
        return "n/d"
    av = abs(value)
    unit, scaled = ("B", av / 1e9) if av >= 1e9 else ("M", av / 1e6)
    # un decimale a zero è rumore su un numero che deve leggersi a colpo d'occhio:
    # "+445M", non "+445,0M"
    decimals = 0 if round(scaled, 1) == round(scaled) else 1
    return _signed(value, f"{_it(scaled, decimals)}{unit}", force_sign=force_sign)


def percent(value: float | None, *, decimals: int = 1, force_sign: bool = True) -> str:
    """Percentuale gia' espressa in punti percentuali: -1.06 -> '−1,1%'."""
    if value is None:
        return "n/d"
    return _signed(value, f"{_it(abs(value), decimals)}%", force_sign=force_sign)


def count(value: float | None) -> str:
    """Conteggio intero con separatore migliaia: 14999 -> '14.999'."""
    if value is None:
        return "n/d"
    return _it(round(value), 0)


def ratio(value: float | None, *, decimals: int = 2) -> str:
    """Rapporto adimensionale: 0.4535 -> '0,45'."""
    if value is None:
        return "n/d"
    return _it(value, decimals)
