"""Charm e vanna in forma chiusa, dalla chain di opzioni Deribit.

Sono le due greche del secondo ordine che spiegano i flussi di hedging che il GEX
da solo non vede:

- **charm** (∂Δ/∂t) è la deriva di delta che scade da sola col tempo. È l'unico
  flusso che arriva su un calendario invece che in reazione al prezzo: un dealer
  che non fa nulla vede comunque il proprio delta cambiare, e deve ribilanciare.
- **vanna** (∂Δ/∂σ) dice quanto quel delta si muove quando si muove la volatilità
  implicita. Con vanna negativa un tape di IV in discesa smette di essere
  carburante per il rialzo.

Deribit non pubblica né charm né vanna, ma pubblica delta, gamma e ``mark_iv``.
Il modulo ricostruisce d₁ e d₂ dagli stessi input e da lì ricava tutte le greche:
:func:`black_scholes_greeks` restituisce anche delta e gamma proprio perché
possano essere confrontati con quelli dichiarati da Deribit. Se la ricostruzione
riproduce i loro numeri, charm e vanna calcolate dalla stessa d₁ sono affidabili;
se non li riproduce, non lo sono — ed è un test, non un'opinione.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Le opzioni Deribit sono su BTC: nessun dividendo.
_Q = 0.0

#: Tasso privo di rischio assunto nullo. Su scadenze di giorni o settimane il
#: contributo di r a d₁ è trascurabile rispetto all'incertezza su σ, e Deribit
#: stessa quota le greche in questa convenzione.
_R = 0.0

#: Giorni di calendario in un anno — BTC tratta 365 giorni, coerente con il resto
#: del progetto (backtest, regime_analysis, correlation usano tutti sqrt(365)).
DAYS_PER_YEAR = 365.0

#: Sotto questa vita residua le greche del secondo ordine esplodono: d₂/(2T)
#: diverge e il numero smette di significare qualcosa. Le opzioni in scadenza
#: oggi vengono escluse invece di produrre un charm enorme e privo di senso.
_MIN_T_YEARS = 1.0 / (DAYS_PER_YEAR * 24.0)  # un'ora

#: IV non positiva significa dato mancante, non volatilità zero.
_MIN_SIGMA = 1e-6


@dataclass(frozen=True)
class OptionGreeks:
    """Greche di una singola opzione, per unità di sottostante.

    ``delta`` e ``gamma`` sono inclusi non perché servano al calcolo — Deribit li
    fornisce già — ma perché sono il metro con cui si verifica che d₁ sia stata
    ricostruita bene.
    """

    d1: float
    d2: float
    delta: float
    gamma: float
    vega: float
    #: ∂Δ/∂σ, per punto di volatilità (σ espressa in frazione, non in %).
    vanna: float
    #: ∂Δ/∂t espressa **per giorno**: è l'unità in cui il flusso ha senso su una card.
    charm_per_day: float


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def time_to_expiry_years(expiration_timestamp_ms: float, now_ms: float) -> float:
    """Vita residua in anni da due timestamp in millisecondi."""
    return max(0.0, (expiration_timestamp_ms - now_ms) / 1000.0) / (DAYS_PER_YEAR * 86_400.0)


def black_scholes_greeks(
    *,
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    option_type: str,
) -> OptionGreeks | None:
    """Greche Black-Scholes di una singola opzione, dividendo nullo.

    Args:
        spot: prezzo del sottostante.
        strike: strike dell'opzione.
        t_years: vita residua in anni.
        sigma: volatilità implicita in frazione (0.65 = 65%), non in percentuale.
        option_type: "call" o "put".

    Returns:
        OptionGreeks, oppure None se gli input non permettono un calcolo sensato
        (scadenza troppo vicina, IV assente, prezzi non positivi). Restituire
        None invece di uno zero è deliberato: uno zero si somma e sporca il
        totale, un None si conta come dato mancante.
    """
    tipo = (option_type or "").lower()
    if tipo not in ("call", "put"):
        return None
    if spot <= 0 or strike <= 0 or sigma < _MIN_SIGMA or t_years < _MIN_T_YEARS:
        return None

    sqrt_t = math.sqrt(t_years)
    sig_sqrt_t = sigma * sqrt_t

    d1 = (math.log(spot / strike) + (_R - _Q + 0.5 * sigma * sigma) * t_years) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    pdf_d1 = _norm_pdf(d1)

    # Con q=0 il fattore di sconto sul dividendo è 1.
    delta = _norm_cdf(d1) if tipo == "call" else _norm_cdf(d1) - 1.0
    gamma = pdf_d1 / (spot * sig_sqrt_t)
    vega = spot * pdf_d1 * sqrt_t

    # vanna = ∂²V/∂S∂σ = −φ(d₁)·d₂/σ. Uguale per call e put: la put-call parity
    # differisce per una costante, che derivata sparisce.
    vanna = -pdf_d1 * d2 / sigma

    # charm = ∂Δ/∂t. Con q=0 e r=0 si riduce a φ(d₁)·d₂/(2T), e vale identica per
    # call e put perché Δ_put = Δ_call − 1 e la costante non dipende dal tempo.
    # Il segno è quello di d₂: sopra lo strike il delta di una call sale col
    # passare del tempo, sotto scende.
    charm_per_year = pdf_d1 * d2 / (2.0 * t_years)

    return OptionGreeks(
        d1=d1,
        d2=d2,
        delta=delta,
        gamma=gamma,
        vega=vega,
        vanna=vanna,
        charm_per_day=charm_per_year / DAYS_PER_YEAR,
    )


def dealer_sign(option_type: str) -> float:
    """Segno della posizione del dealer, nella convenzione già usata dal GEX.

    Deve restare allineato a ``GexCalculator._option_gex``: call +1, put −1, con
    il dealer assunto short gamma. Se le due convenzioni divergessero, il charm e
    il GEX pubblicati sulla stessa card racconterebbero storie opposte.
    """
    return 1.0 if (option_type or "").lower() == "call" else -1.0


def option_charm_usd_day(
    greeks: OptionGreeks,
    *,
    open_interest: float,
    spot: float,
    contract_size: float = 1.0,
    option_type: str = "call",
) -> float:
    """Charm di una posizione in dollari al giorno.

    È la scalatura che trasforma una greca in una frase: non "charm 0,0003" ma
    "+7,7 milioni di acquisto già in calendario".
    """
    return (
        greeks.charm_per_day
        * open_interest
        * contract_size
        * spot
        * dealer_sign(option_type)
    )


def option_vanna_usd(
    greeks: OptionGreeks,
    *,
    open_interest: float,
    spot: float,
    contract_size: float = 1.0,
    option_type: str = "call",
) -> float:
    """Vanna di una posizione in dollari per punto di volatilità.

    Il fattore 0.01 normalizza per un movimento di **un punto** di IV (da 65% a
    66%), che è l'unità in cui la si legge, invece che per il raddoppio di σ.
    """
    return (
        greeks.vanna
        * 0.01
        * open_interest
        * contract_size
        * spot
        * dealer_sign(option_type)
    )
