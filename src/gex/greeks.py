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


# ─── Aggregazione sulla chain ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CharmProjectionDay:
    """Charm netto atteso per un giorno futuro."""

    days_ahead: int
    charm_usd_day: float
    #: Strumenti ancora vivi quel giorno: il crollo dopo una scadenza si legge qui.
    live_instruments: int


@dataclass
class ChainGreeks:
    """Charm e vanna aggregati su tutta la chain."""

    spot: float
    total_charm_usd_day: float
    total_vanna_usd: float
    #: {strike: charm in USD/giorno}, per trovare lo strike magnete.
    charm_by_strike: dict[float, float]
    #: Proiezione a N giorni con decadimento di T e caduta degli scaduti.
    projection: list[CharmProjectionDay]
    n_instruments: int
    n_skipped: int

    @property
    def magnet_strike(self) -> float | None:
        """Lo strike che tira di più: massimo |charm|."""
        if not self.charm_by_strike:
            return None
        return max(self.charm_by_strike, key=lambda k: abs(self.charm_by_strike[k]))


def aggregate_chain_greeks(
    options: list[dict],
    *,
    spot: float,
    now_ms: float,
    contract_size: float = 1.0,
    projection_days: int = 10,
) -> ChainGreeks | None:
    """Calcola charm e vanna su tutta la chain e proietta il charm in avanti.

    La proiezione ricalcola il charm per ogni giorno futuro facendo decadere la
    vita residua e lasciando cadere gli strumenti scaduti: è così che si vede la
    marea giornaliera e il salto che segue una scadenza.

    Args:
        options: chain da ``DeribitClient.fetch_all_options``.
        spot: prezzo spot corrente.
        now_ms: istante di riferimento in millisecondi.
        contract_size: dimensione del contratto (da settings.yaml).
        projection_days: quanti giorni proiettare in avanti.

    Returns:
        ChainGreeks, oppure None se nessuno strumento è calcolabile — senza
        nulla da sommare non si restituisce uno zero che sembrerebbe un dato.
    """
    if not options or spot <= 0:
        return None

    charm_by_strike: dict[float, float] = {}
    tot_charm = tot_vanna = 0.0
    validi: list[tuple[dict, float]] = []   # (opzione, vita residua in anni)
    saltati = 0

    for o in options:
        t = time_to_expiry_years(o.get("expiration_timestamp", 0), now_ms)
        g = black_scholes_greeks(
            spot=o.get("underlying_price") or spot,
            strike=o.get("strike", 0.0),
            t_years=t,
            sigma=(o.get("mark_iv") or 0.0) / 100.0,   # Deribit quota la IV in %
            option_type=o.get("option_type", ""),
        )
        oi = o.get("open_interest") or 0.0
        if g is None or oi <= 0:
            saltati += 1
            continue

        tipo = o.get("option_type", "")
        c = option_charm_usd_day(
            g, open_interest=oi, spot=spot, contract_size=contract_size, option_type=tipo
        )
        v = option_vanna_usd(
            g, open_interest=oi, spot=spot, contract_size=contract_size, option_type=tipo
        )
        tot_charm += c
        tot_vanna += v
        strike = o["strike"]
        charm_by_strike[strike] = charm_by_strike.get(strike, 0.0) + c
        validi.append((o, t))

    if not validi:
        return None

    projection: list[CharmProjectionDay] = []
    for giorno in range(projection_days):
        offset = giorno / DAYS_PER_YEAR
        somma = 0.0
        vivi = 0
        for o, t in validi:
            residua = t - offset
            g = black_scholes_greeks(
                spot=o.get("underlying_price") or spot,
                strike=o.get("strike", 0.0),
                t_years=residua,
                sigma=(o.get("mark_iv") or 0.0) / 100.0,
                option_type=o.get("option_type", ""),
            )
            if g is None:   # scaduto entro quel giorno: esce dal calcolo
                continue
            vivi += 1
            somma += option_charm_usd_day(
                g,
                open_interest=o.get("open_interest") or 0.0,
                spot=spot,
                contract_size=contract_size,
                option_type=o.get("option_type", ""),
            )
        projection.append(
            CharmProjectionDay(days_ahead=giorno, charm_usd_day=somma, live_instruments=vivi)
        )

    return ChainGreeks(
        spot=spot,
        total_charm_usd_day=tot_charm,
        total_vanna_usd=tot_vanna,
        charm_by_strike=charm_by_strike,
        projection=projection,
        n_instruments=len(validi),
        n_skipped=saltati,
    )
