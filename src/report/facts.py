"""Estrazione dei fatti candidati per il Desk Note.

Ogni estrattore legge il payload di un endpoint esistente e restituisce un
:class:`Fact` gia' scritto in italiano, oppure ``None`` se i dati che gli
servono non ci sono. Nessun estrattore inventa: se ``notional_usd`` e' nullo
parla di conteggi, non di dollari.

La salienza 0-1 e' cio' che decide quale fatto prende il titolo di giornata.
La regola comune e' che urgenza = vicinanza a una soglia: un knock-in all'1%
conta piu' di uno al 15%, un GEX al 95esimo percentile conta piu' di uno al 55esimo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from src.report import formatting as fmt

# ─── Modello ──────────────────────────────────────────────────────────────────

#: Segno del fatto, mappato sul colore della card (verde compra, ambra vende/rompe).
SIGN_POSITIVE = "positive"
SIGN_NEGATIVE = "negative"
SIGN_NEUTRAL = "neutral"

#: Sotto queste grandezze un numero non e' un fatto ma rumore — e quando la
#: fonte a monte e' giu' arriva comunque zero, indistinguibile da un vero zero.
#: Pubblicare "+0M di ETF in tre giorni" sarebbe un fatto falso col marchio sopra.
_MIN_FLOW_USD_M = 25.0
_MIN_GEX_USD_M = 5.0
_MIN_CHARM_USD_M = 5.0
_MIN_VANNA_USD_M = 5.0

#: Sotto questo funding annualizzato non c'e' niente da raccontare: e' il
#: rendimento di un titolo di stato, non una posizione affollata.
_MIN_FUNDING_ANN_PCT = 4.0

#: Sotto questo nozionale la card parla di conteggi invece che di dollari: un
#: numero piccolo in cifra tonda impressiona meno di "quattro barriere all'1%".
_MIN_NOTIONAL_CARD_USD_M = 20.0


@dataclass
class Fact:
    """Un fatto candidato a diventare una card.

    Attributes:
        key: identificatore stabile, usato per deduplicare fra un'edizione e l'altra.
        topic: famiglia di appartenenza (gex|barrier|flows|signal), serve a
            garantire varieta' nella selezione finale.
        salience: 0-1, quanto questo fatto merita il titolo oggi.
        headline: la tesi, mai il nome della metrica.
        body: uno o due paragrafi che portano al numero.
        hero_value: il numero grande, gia' formattato.
        hero_caption: etichetta sotto il numero, su una o due righe.
        sign: SIGN_POSITIVE | SIGN_NEGATIVE | SIGN_NEUTRAL.
        takeaway: versione da una riga, usata nella card di copertina.
    """

    key: str
    topic: str
    salience: float
    headline: str
    body: list[str]
    hero_value: str
    hero_caption: str
    sign: str = SIGN_NEUTRAL
    takeaway: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# ─── Helper di salienza ───────────────────────────────────────────────────────


def _proximity(distance_pct: float | None, scale: float) -> float:
    """Salienza decrescente con la distanza da una soglia.

    ``scale`` e' la distanza in punti percentuali a cui la salienza scende a 1/e.
    """
    if distance_pct is None:
        return 0.0
    return math.exp(-abs(distance_pct) / max(scale, 1e-9))


def _extremity(percentile: float | None) -> float:
    """Quanto un percentile 0-100 e' lontano dalla mediana: 50 -> 0, 0 o 100 -> 1."""
    if percentile is None:
        return 0.0
    return min(1.0, abs(percentile - 50.0) / 50.0)


def _fra_giorni(days: int | float | None) -> str:
    """Distanza temporale in italiano leggibile: 0 -> 'oggi', 1 -> 'fra 1 giorno'.

    "fra 0 giorni" e "fra 1 giorni" sono due modi diversi di far sembrare il
    report generato da una macchina.
    """
    if days is None:
        return "n/d"
    n = int(days)
    if n <= 0:
        return "oggi"
    return "fra 1 giorno" if n == 1 else f"fra {fmt.count(n)} giorni"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _num(d: dict | None, *keys: str) -> float | None:
    """Legge una chiave annidata restituendo None invece di sollevare."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, (int, float)) else None


# ─── Fatti GEX ────────────────────────────────────────────────────────────────


def fact_gex_asymmetry(gex: dict) -> Fact | None:
    """Come e' distribuita la gamma sopra e sotto lo spot.

    E' il fatto piu' forte quando il GEX totale e' positivo ma sotto lo spot e'
    negativo: il titolo dice "positivo", la realta' sotto i piedi dice il contrario.
    """
    spot = _num(gex, "snapshot", "spot_price")
    profile = (gex or {}).get("strike_profile")
    total = _num(gex, "snapshot", "total_net_gex")
    if not spot or not profile or total is None:
        return None

    below = sum(r["net_gex_m"] for r in profile if r.get("strike", 0) < spot) * 1e6
    above = sum(r["net_gex_m"] for r in profile if r.get("strike", 0) >= spot) * 1e6

    if abs(below) + abs(above) < _MIN_GEX_USD_M * 1e6:
        return None  # profilo piatto: non c'e' niente da raccontare

    contraddizione = total > 0 > below
    salience = 0.88 if contraddizione else _clamp01(
        abs(below - above) / max(abs(below) + abs(above), 1.0) * 0.6
    )

    if contraddizione:
        headline = "La gamma positiva è tutta sopra di te"
        chiusura = (
            "Finché resti sotto, i dealer non ti stabilizzano: ti seguono."
        )
    else:
        headline = "Dove sta davvero la gamma"
        chiusura = "È il lato dove i dealer assorbono i movimenti."

    return Fact(
        key="gex_asymmetry",
        topic="gex",
        salience=salience,
        headline=headline,
        body=[
            f"Il GEX netto è {fmt.usd_millions(total)}, ma è distribuito male.",
            (
                f"Sopra lo spot ci sono {fmt.usd_millions(above)} di gamma. "
                f"Sotto ce ne sono {fmt.usd_millions(below)}. {chiusura}"
            ),
        ],
        hero_value=fmt.usd_millions(below),
        hero_caption="GEX netto sotto lo spot\nper movimento dell'1%",
        sign=SIGN_NEGATIVE if below < 0 else SIGN_POSITIVE,
        takeaway=f"Sotto lo spot la gamma netta è {fmt.usd_millions(below)}.",
        meta={"gex_below": below, "gex_above": above},
    )


def fact_gex_flip(gex: dict) -> Fact | None:
    """Distanza fra spot e gamma flip — la soglia che separa i due regimi."""
    spot = _num(gex, "snapshot", "spot_price")
    flip = _num(gex, "snapshot", "gamma_flip_price")
    total = _num(gex, "snapshot", "total_net_gex")
    pct = _num(gex, "regime", "gex_percentile")
    if not spot or not flip:
        return None

    dist = (spot - flip) / flip * 100.0
    sotto = dist < 0
    # vicino al flip = fatto urgente; il boost tiene conto di quanto e' estremo il percentile
    salience = _clamp01(0.75 * _proximity(dist, 4.0) + 0.25 * _extremity(pct))

    if sotto:
        headline = f"Il flip è {fmt.percent(abs(dist), force_sign=False)} sopra di te"
        conseguenza = (
            "Sotto quella soglia l'hedging dei dealer amplifica i movimenti "
            "invece di assorbirli."
        )
    else:
        headline = f"Sei {fmt.percent(abs(dist), force_sign=False)} sopra il flip"
        conseguenza = (
            "Sopra quella soglia i dealer comprano i ribassi e vendono i rialzi: "
            "è il regime che comprime la volatilità."
        )

    percentile_txt = (
        f" — il {fmt.count(pct)}° percentile degli ultimi 90 giorni" if pct is not None else ""
    )
    return Fact(
        key="gex_flip",
        topic="gex",
        salience=salience,
        headline=headline,
        body=[
            f"Il gamma flip è a {fmt.price(flip)}, lo spot a {fmt.price(spot)}.",
            f"Il GEX totale è {fmt.usd_millions(total)}{percentile_txt}. {conseguenza}",
        ],
        hero_value=fmt.price(flip),
        hero_caption=f"Gamma flip\n{fmt.percent(dist)} dallo spot",
        sign=SIGN_NEGATIVE if sotto else SIGN_POSITIVE,
        takeaway=(
            f"GEX {fmt.usd_millions(total)} ma lo spot sta "
            f"{fmt.percent(abs(dist), force_sign=False)} sotto la soglia a {fmt.price(flip)}."
            if sotto
            else f"GEX {fmt.usd_millions(total)} e spot sopra la soglia a {fmt.price(flip)}."
        ),
        meta={"flip": flip, "distance_pct": dist},
    )


def fact_gex_walls(gex: dict) -> Fact | None:
    """Il corridoio fra put wall e call wall, con il max pain come terzo riferimento."""
    spot = _num(gex, "snapshot", "spot_price")
    put_wall = _num(gex, "snapshot", "put_wall")
    call_wall = _num(gex, "snapshot", "call_wall")
    max_pain = _num(gex, "snapshot", "max_pain")
    pcr = _num(gex, "options_metrics", "put_call_ratio")
    profile = (gex or {}).get("strike_profile") or []
    if not spot or not put_wall or not call_wall:
        return None

    d_call = (call_wall - spot) / spot * 100.0
    d_put = (put_wall - spot) / spot * 100.0
    # il corridoio diventa notizia quando lo spot ne tocca un bordo
    salience = _clamp01(0.55 + 0.45 * max(_proximity(d_call, 3.0), _proximity(d_put, 3.0)))

    riga_call = next((r for r in profile if r.get("strike") == call_wall), None)
    dettaglio = ""
    if riga_call:
        dettaglio = (
            f" su {fmt.count(riga_call.get('call_oi'))} call contro "
            f"{fmt.count(riga_call.get('put_oi'))} put"
        )
    gex_call_wall = (riga_call or {}).get("net_gex_m")

    terza = []
    if max_pain:
        terza.append(f"Max pain a {fmt.price(max_pain)}")
    if pcr is not None:
        lettura = "nessuno sta comprando protezione" if pcr < 0.7 else "la protezione è cara"
        terza.append(f"put/call ratio {fmt.ratio(pcr)}: {lettura}")

    return Fact(
        key="gex_walls",
        topic="gex",
        salience=salience,
        headline=f"{fmt.price(call_wall)} è il tetto, {fmt.price(put_wall)} è la rete",
        body=[
            (
                f"Il call wall a {fmt.price(call_wall)} concentra "
                f"{fmt.usd_millions((gex_call_wall or 0) * 1e6)} di gamma{dettaglio}: "
                f"è il livello dove i dealer vendono di più contro il rialzo."
            ),
            f"Sotto, il put wall a {fmt.price(put_wall)} è l'unico supporto costruito, "
            f"{fmt.percent(d_put)} da qui. " + (". ".join(terza) + "." if terza else ""),
        ],
        hero_value=fmt.usd_millions((gex_call_wall or 0) * 1e6),
        hero_caption=f"Strike {fmt.price(call_wall)} — il più pesante\ndi tutta la board",
        sign=SIGN_NEUTRAL,
        takeaway=(
            f"Corridoio {fmt.price(put_wall)}–{fmt.price(call_wall)}, "
            f"max pain a {fmt.price(max_pain)}."
            if max_pain
            else f"Corridoio {fmt.price(put_wall)}–{fmt.price(call_wall)}."
        ),
        meta={"put_wall": put_wall, "call_wall": call_wall, "max_pain": max_pain},
    )


# ─── Fatti barriere (il differenziatore) ──────────────────────────────────────

_TIPO_IT = {
    "knock_in": "knock-in",
    "knock_out": "knock-out",
    "autocall": "autocall",
    "buffer": "buffer",
}


def fact_barrier_nearest(barriers: dict) -> Fact | None:
    """La barriera di nota strutturata piu' vicina allo spot.

    E' il fatto che nessun feed di opzioni puo' produrre: sono livelli depositati
    alla SEC, dove una banca deve muoversi per contratto e non per convenienza.
    """
    spot = (barriers or {}).get("spot_price")
    rows = (barriers or {}).get("barriers") or []
    if not spot or not rows:
        return None

    priced = [b for b in rows if (b.get("level_price_btc") or 0) > 0]
    if not priced:
        return None

    nearest = min(priced, key=lambda b: abs(b["level_price_btc"] - spot))
    dist = (nearest["level_price_btc"] - spot) / spot * 100.0
    salience = _clamp01(0.95 * _proximity(dist, 3.0))

    # quante barriere condividono la zona stretta intorno allo spot
    vicine = [b for b in priced if abs(b["level_price_btc"] - spot) / spot * 100.0 <= 2.0]
    sotto_2 = [b for b in vicine if b["level_price_btc"] < spot]

    entro_20 = [b for b in priced if abs(b["level_price_btc"] - spot) / spot * 100.0 <= 20.0]
    per_issuer: dict[str, int] = {}
    for b in entro_20:
        iss = b.get("issuer") or "n/d"
        per_issuer[iss] = per_issuer.get(iss, 0) + 1
    top_issuer = max(per_issuer.items(), key=lambda kv: kv[1]) if per_issuer else None

    # Il nozionale, dove c'è. Sommato per nota e non per barriera: una nota con
    # tre livelli non vale il triplo, e contarla tre volte gonfierebbe il numero
    # più visibile della card.
    def _notional_note(righe: list[dict]) -> tuple[float, int]:
        per_nota = {
            b.get("note_id"): b["notional_usd"]
            for b in righe
            if b.get("note_id") is not None and b.get("notional_usd")
        }
        return sum(per_nota.values()), len(per_nota)

    notional_vicine, n_note_vicine = _notional_note(vicine)
    notional_20, _ = _notional_note(entro_20)

    meta_info = (barriers or {}).get("meta") or {}
    totale = meta_info.get("total_active") or (barriers or {}).get("count") or len(rows)
    tipo = _TIPO_IT.get(nearest.get("barrier_type", ""), nearest.get("barrier_type", "barriera"))

    # Il titolo dice dollari solo quando i dollari ci sono davvero: e' la
    # differenza fra "quattro barriere" e "x milioni di note che si attivano".
    ha_soldi = notional_vicine >= _MIN_NOTIONAL_CARD_USD_M * 1e6 and len(vicine) >= 2
    if ha_soldi:
        dove = "sotto di te" if len(sotto_2) >= len(vicine) / 2 else "da qui"
        headline = (
            f"{fmt.usd_millions(notional_vicine, force_sign=False)} di note "
            f"si attivano entro il 2% {dove}"
        )
    elif len(sotto_2) >= 2:
        headline = (
            f"{fmt.count(len(sotto_2))} barriere bancarie a meno del 2% sotto di te"
        )
    else:
        headline = f"Un {tipo} bancario a {fmt.percent(dist)} da qui"

    seconda = (
        f"La più vicina è un {tipo} {nearest.get('issuer', 'n/d')} a "
        f"{fmt.price(nearest['level_price_btc'])}, {fmt.percent(dist)} dallo spot. "
        f"Entro il 20% ce ne sono {fmt.count(len(entro_20))}"
    )
    if top_issuer:
        seconda += f", e {top_issuer[0]} da sola ne ha {fmt.count(top_issuer[1])}"
    seconda += "."
    if notional_20 >= _MIN_NOTIONAL_CARD_USD_M * 1e6:
        seconda += (
            f" In tutto valgono {fmt.usd_millions(notional_20, force_sign=False)} "
            f"di nozionale depositato."
        )

    return Fact(
        key="barrier_nearest",
        topic="barrier",
        salience=salience,
        headline=headline,
        body=[
            (
                "Le banche hanno venduto note strutturate su IBIT con le barriere "
                f"depositate alla SEC. Ne sono attive {fmt.count(totale)}, "
                f"di cui {fmt.count(len(priced))} con un prezzo."
            ),
            seconda,
        ],
        hero_value=(
            fmt.usd_millions(notional_vicine, force_sign=False)
            if ha_soldi
            else fmt.price(nearest["level_price_btc"])
        ),
        hero_caption=(
            f"Nozionale che si attiva entro il 2%\n"
            f"su {fmt.count(n_note_vicine)} note depositate alla SEC"
            if ha_soldi
            else f"{tipo.capitalize()} più vicino · {nearest.get('issuer', 'n/d')}\n"
                 f"{fmt.percent(dist)} dallo spot"
        ),
        sign=SIGN_NEGATIVE if dist < 0 else SIGN_POSITIVE,
        takeaway=(
            f"{fmt.usd_millions(notional_vicine, force_sign=False)} di note bancarie "
            f"si attivano entro il 2%."
            if ha_soldi
            else f"{fmt.count(len(sotto_2))} barriere bancarie entro il 2%: la più vicina "
            f"{nearest.get('issuer', 'n/d')} a {fmt.price(nearest['level_price_btc'])}."
            if len(sotto_2) >= 2
            else f"Barriera {nearest.get('issuer', 'n/d')} a "
            f"{fmt.price(nearest['level_price_btc'])}, {fmt.percent(dist)} da qui."
        ),
        meta={
            "distance_pct": dist,
            "n_within_2pct": len(vicine),
            "n_within_20pct": len(entro_20),
            "notional_within_2pct_usd": notional_vicine or None,
            "notional_within_20pct_usd": notional_20 or None,
        },
    )


# ─── Fatti flussi ETF ─────────────────────────────────────────────────────────


def fact_flows_3d(flows: dict, signals: dict) -> Fact | None:
    """Il flusso IBIT a 3 giorni, letto insieme al punteggio del pilastro."""
    flow_3d = _num(signals, "inputs", "ibit_flow_3d_usd_m")
    summary = (flows or {}).get("summary") or {}
    # senza riepilogo non c'e' modo di corroborare il numero, e sotto la soglia
    # di materialita' e' quasi sempre "non lo sappiamo" travestito da zero
    if flow_3d is None or not summary or abs(flow_3d) < _MIN_FLOW_USD_M:
        return None
    flow_usd = flow_3d * 1e6

    pillar = next(
        (p for p in (signals or {}).get("pillars", []) if p.get("name") == "etf_flows"), None
    )
    score = (pillar or {}).get("score")
    ibit_net = _num(summary, "ibit", "net_flow_usd_b")
    corr = summary.get("full_period_corr_ibit_btc_next1d")
    giorni = _num(summary, "ibit", "days_with_data")

    # salienza proporzionale alla grandezza, senza pavimento: un flusso appena
    # sopra la soglia non deve competere con una barriera all'1%
    salience = _clamp01(0.90 * min(1.0, abs(flow_3d) / 600.0))

    seconda = []
    if ibit_net is not None:
        seconda.append(
            f"IBIT ha assorbito {fmt.usd_millions(ibit_net * 1e9)} netti da inizio vita"
        )
    gbtc = _num(summary, "by_ticker", "GBTC", "net_flow_usd_b")
    if gbtc is not None:
        seconda.append(f"mentre GBTC ne perdeva {fmt.usd_millions(abs(gbtc) * 1e9, force_sign=False)}")
    riga2 = (", ".join(seconda) + ". ") if seconda else ""
    if corr is not None and giorni:
        riga2 += (
            f"Ma la correlazione fra flusso di oggi e prezzo di domani è "
            f"{fmt.ratio(corr)} su {fmt.count(giorni)} giorni: non è un timer, è una marea."
        )

    prima = f"Il flusso netto IBIT degli ultimi 3 giorni è {fmt.usd_millions(flow_usd)}."
    if score is not None:
        prima += f" Il pilastro flussi segna {fmt.count(score)} su 100."

    return Fact(
        key="flows_3d",
        topic="flows",
        salience=salience,
        headline=f"{fmt.usd_millions(flow_usd)} di ETF in tre giorni",
        body=[prima, riga2 or "È il pilastro con più storico alle spalle."],
        hero_value=fmt.usd_millions(flow_usd),
        hero_caption="Flusso netto IBIT\nultimi 3 giorni",
        sign=SIGN_POSITIVE if flow_usd >= 0 else SIGN_NEGATIVE,
        takeaway=f"{fmt.usd_millions(flow_usd)} di flusso ETF in tre giorni.",
        meta={"flow_3d_usd": flow_usd},
    )


def fact_flows_rotation(flows: dict) -> Fact | None:
    """La rotazione strutturale GBTC -> IBIT, il fatto che non scade mai."""
    summary = (flows or {}).get("summary") or {}
    ibit = _num(summary, "ibit", "net_flow_usd_b")
    gbtc = _num(summary, "by_ticker", "GBTC", "net_flow_usd_b")
    if ibit is None or gbtc is None:
        return None

    by_ticker = summary.get("by_ticker") or {}
    positivi = sum(1 for v in by_ticker.values() if (v or {}).get("net_flow_usd_b", 0) > 0)

    return Fact(
        key="flows_rotation",
        topic="flows",
        # fatto strutturale: sempre vero, quindi salienza bassa e costante
        salience=0.42,
        headline="Il travaso che nessuno racconta",
        body=[
            (
                f"IBIT ha raccolto {fmt.usd_millions(ibit * 1e9)} netti. "
                f"GBTC ne ha persi {fmt.usd_millions(abs(gbtc) * 1e9, force_sign=False)}."
            ),
            (
                f"Non è denaro nuovo che entra nel bitcoin: è denaro che cambia veicolo, "
                f"e paga meno commissioni per farlo. Su {fmt.count(len(by_ticker))} ETF "
                f"censiti, {fmt.count(positivi)} sono in raccolta netta positiva."
            ),
        ],
        hero_value=fmt.usd_millions(abs(gbtc) * 1e9, force_sign=False),
        hero_caption="Uscite nette da GBTC\ndall'inizio",
        sign=SIGN_NEUTRAL,
        takeaway=f"GBTC {fmt.usd_millions(gbtc * 1e9)} contro IBIT {fmt.usd_millions(ibit * 1e9)}.",
        meta={"ibit_b": ibit, "gbtc_b": gbtc},
    )


# ─── Fatto segnale (accountability) ───────────────────────────────────────────


def fact_signal_scoreboard(signals: dict, forecast: dict | None = None) -> Fact | None:
    """Il punteggio composito e le previsioni aperte.

    Pubblicabile solo se i pilastri hanno copertura sufficiente: un composito
    retto da pilastri mezzi vuoti non va in pagina.
    """
    score = (signals or {}).get("score")
    label = (signals or {}).get("signal")
    pillars = (signals or {}).get("pillars") or []
    if score is None or not pillars:
        return None

    parti = [
        f"{p['name'].replace('etf_flows', 'flussi ETF').replace('barrier', 'barriere')} "
        f"{fmt.count(p['score'])}"
        for p in pillars
        if p.get("score") is not None
    ]
    aperte = (forecast or {}).get("open")

    # piu' il punteggio e' vicino a una soglia operativa, piu' e' notizia
    salience = _clamp01(0.45 + 0.45 * max(_proximity(score - 65, 8.0), _proximity(score - 40, 8.0)))

    seconda = (
        "Ogni chiamata resta aperta e viene marcata a mercato, con data, prezzo "
        "di riferimento e orizzonte scritti prima."
    )
    if aperte:
        seconda = (
            f"{fmt.count(aperte)} previsioni sono in verifica adesso, con data, "
            f"prezzo di riferimento e orizzonte scritti prima."
        )

    return Fact(
        key="signal_scoreboard",
        topic="signal",
        salience=salience,
        headline=f"Il punteggio di oggi è {fmt.count(score)}. E lo verifichiamo.",
        body=[
            (
                f"Quattro pilastri, un numero: {', '.join(parti)}. "
                f"Sotto 40 è risk-off, sopra 65 è long."
            ),
            seconda,
        ],
        hero_value=f"{fmt.count(score)} / 100",
        hero_caption=f"Segnale composito\n{(label or '').capitalize()}",
        sign=SIGN_POSITIVE if score >= 65 else SIGN_NEGATIVE if score < 40 else SIGN_NEUTRAL,
        takeaway=f"Segnale composito {fmt.count(score)} su 100, {label}.",
        meta={"score": score, "signal": label},
    )


# ─── Registro ─────────────────────────────────────────────────────────────────


def extract_all(
    *,
    gex: dict | None = None,
    barriers: dict | None = None,
    flows: dict | None = None,
    signals: dict | None = None,
    forecast: dict | None = None,
    macro: dict | None = None,
) -> list[Fact]:
    """Esegue tutti gli estrattori, scartando quelli senza dati.

    Un estrattore che solleva non deve far cadere l'edizione: viene saltato.
    """
    import logging

    log = logging.getLogger(__name__)
    candidati = [
        ("gex_asymmetry", lambda: fact_gex_asymmetry(gex or {})),
        ("gex_flip", lambda: fact_gex_flip(gex or {})),
        ("gex_walls", lambda: fact_gex_walls(gex or {})),
        ("barrier_nearest", lambda: fact_barrier_nearest(barriers or {})),
        ("flows_3d", lambda: fact_flows_3d(flows or {}, signals or {})),
        ("flows_rotation", lambda: fact_flows_rotation(flows or {})),
        ("charm_tide", lambda: fact_charm_tide(gex or {})),
        ("vanna_sign", lambda: fact_vanna_sign(gex or {})),
        ("funding_cost", lambda: fact_funding_cost(macro or {})),
        ("signal_scoreboard", lambda: fact_signal_scoreboard(signals or {}, forecast)),
    ]

    out: list[Fact] = []
    for nome, fn in candidati:
        try:
            f = fn()
        except Exception as exc:  # noqa: BLE001 — un estrattore rotto costa una
            # card, non l'edizione: qualunque errore su un singolo fatto viene
            # loggato e saltato, perché i payload upstream cambiano forma senza
            # preavviso e un report a cinque card è meglio di un 500.
            log.warning("Estrattore %s fallito: %s", nome, exc)
            continue
        if f is not None:
            out.append(f)
    return out


# ─── Fatti charm e vanna ──────────────────────────────────────────────────────


def fact_charm_tide(gex: dict) -> Fact | None:
    """Il flusso di hedging che arriva su un calendario, non in reazione al prezzo.

    E' il fatto piu' insolito della serie: tutte le altre card raccontano cosa
    succede *se* il prezzo si muove, questa racconta cosa succede comunque.
    """
    charm = (gex or {}).get("charm")
    if not charm:
        return None

    oggi = charm.get("total_charm_usd_day")
    proj = charm.get("projection") or []
    magnete = charm.get("magnet_strike")
    if oggi is None or abs(oggi) < _MIN_CHARM_USD_M * 1e6:
        return None

    # Il picco della marea, e quando arriva. Se cade oggi coincide col titolo:
    # ripeterlo non aggiunge niente, quindi si dice solo che il colmo e' adesso.
    picco = max(proj, key=lambda p: abs(p.get("charm_usd_day") or 0), default=None)
    totale = sum(abs(p.get("charm_usd_day") or 0) for p in proj)

    # il crollo post-expiry: dove il numero di strumenti vivi scende di piu'
    salto = None
    for prima, dopo in pairwise(proj):
        persi = (prima.get("live_instruments") or 0) - (dopo.get("live_instruments") or 0)
        if persi > 0 and (salto is None or persi > salto[1]):
            salto = (dopo.get("days_ahead"), persi)

    compra = oggi > 0
    salience = _clamp01(0.35 + 0.5 * min(1.0, abs(oggi) / 60e6))

    seconda = (
        f"Nei prossimi {fmt.count(len(proj))} giorni il calendario muove in tutto "
        f"{fmt.usd_millions(totale, force_sign=False)}"
    )
    if picco is not None:
        if int(picco.get("days_ahead") or 0) <= 0:
            seconda += ", e il massimo è oggi"
        else:
            seconda += (
                f", col massimo di {fmt.usd_millions(picco['charm_usd_day'])} "
                f"{_fra_giorni(picco['days_ahead'])}"
            )
    if salto:
        seconda += (
            f". Poi la scadenza {_fra_giorni(salto[0])} spegne "
            f"{fmt.count(salto[1])} strumenti e il flusso cala"
        )
    seconda += "."

    return Fact(
        key="charm_tide",
        topic="charm",
        salience=salience,
        headline=(
            f"{fmt.usd_millions(oggi)} al giorno già in calendario"
            if compra
            else f"{fmt.usd_millions(oggi)} al giorno di vendita programmata"
        ),
        body=[
            (
                "Il charm è la deriva di copertura che il tempo programma da solo: "
                "l'unico flusso che i dealer devono eseguire anche se il prezzo non "
                "si muove."
            ),
            seconda,
        ],
        hero_value=fmt.usd_millions(oggi),
        hero_caption=(
            "Copertura già programmata\nper oggi"
            + (f" · magnete {fmt.price(magnete)}" if magnete else "")
        ),
        sign=SIGN_POSITIVE if compra else SIGN_NEGATIVE,
        takeaway=f"{fmt.usd_millions(oggi)} al giorno di copertura già in calendario.",
        meta={"charm_usd_day": oggi, "magnet_strike": magnete},
    )


def fact_vanna_sign(gex: dict) -> Fact | None:
    """Cosa fa il delta dei dealer quando si muove la volatilità implicita.

    Con vanna negativa una IV in discesa smette di essere carburante per il
    rialzo: e' la lettura che spiega i rialzi che si spengono da soli.
    """
    charm = (gex or {}).get("charm")
    if not charm:
        return None

    vanna = charm.get("total_vanna_usd_per_iv_pt")
    if vanna is None or abs(vanna) < _MIN_VANNA_USD_M * 1e6:
        return None

    positiva = vanna > 0
    salience = _clamp01(0.30 + 0.45 * min(1.0, abs(vanna) / 60e6))

    if positiva:
        conseguenza = (
            "Con vanna positiva una volatilità implicita in discesa costringe i "
            "dealer a comprare: il tape calmo diventa carburante per il rialzo."
        )
    else:
        conseguenza = (
            "Con vanna negativa una volatilità implicita in discesa non aiuta più "
            "il rialzo — anzi lo toglie. È così che un rialzo si spegne da solo."
        )

    return Fact(
        key="vanna_sign",
        topic="charm",
        salience=salience,
        headline=(
            "La volatilità che scende ora compra"
            if positiva
            else "La volatilità che scende non spinge più"
        ),
        body=[
            (
                "La vanna dice quanto si muove la copertura dei dealer quando si "
                "muove la volatilità implicita, non il prezzo."
            ),
            (
                f"Oggi vale {fmt.usd_millions(vanna)} per ogni punto di implicita. "
                f"{conseguenza}"
            ),
        ],
        hero_value=fmt.usd_millions(vanna),
        hero_caption="Vanna netta\nper punto di volatilità implicita",
        sign=SIGN_POSITIVE if positiva else SIGN_NEGATIVE,
        takeaway=(
            f"Vanna {fmt.usd_millions(vanna)} per punto di implicita: "
            + ("la vol in calo compra." if positiva else "la vol in calo non spinge più.")
        ),
        meta={"vanna_usd": vanna},
    )


# ─── Fatto macro ──────────────────────────────────────────────────────────────


def fact_funding_cost(macro: dict) -> Fact | None:
    """Quanto costa restare long, in percentuale annua.

    E' l'unico fatto della serie che non viene dalle opzioni, e forse il piu'
    leggibile di tutti: chiunque abbia pagato un interesse capisce cosa vuol
    dire pagare il 12% l'anno per tenere aperta una posizione. Il segno dice
    da che parte sta l'affollamento — i long pagano quando sono troppi.
    """
    funding = (macro or {}).get("funding_rate_annualized_pct")
    if not isinstance(funding, (int, float)):
        return None
    if abs(funding) < _MIN_FUNDING_ANN_PCT:
        return None

    oi = (macro or {}).get("futures_oi_usd") or (macro or {}).get("oi_usd")
    long_pagano = funding > 0

    # 0,50 a soglia, 1,0 oltre il 60% annuo: sopra quella soglia il funding
    # smette di essere un costo e diventa il fatto principale della giornata.
    salience = _clamp01(0.35 + 0.55 * min(1.0, abs(funding) / 60.0))

    chi_paga = "I long pagano gli short" if long_pagano else "Gli short pagano i long"
    verso = (
        "la leva è tutta da una parte, e quella parte è quella lunga"
        if long_pagano
        else "a pagare sono gli short: il ribasso è affollato, non il rialzo"
    )

    corpo = [
        (
            f"{chi_paga} {fmt.percent(abs(funding), decimals=1, force_sign=False)} l'anno per tenere "
            f"aperta la posizione sui perpetui. È il prezzo dell'affollamento: "
            f"{verso}."
        )
    ]
    if isinstance(oi, (int, float)) and oi > 0:
        # senza la scala il 12% non dice quanto pesa: su 66 miliardi sono
        # miliardi di dollari l'anno che cambiano mano fra le due parti
        annuo = abs(funding) / 100.0 * oi
        corpo.append(
            f"Su {fmt.usd_millions(oi, force_sign=False)} di posizioni aperte fa circa "
            f"{fmt.usd_millions(annuo, force_sign=False)} l'anno che passano da una parte all'altra."
        )

    return Fact(
        key="funding_cost",
        topic="macro",
        salience=salience,
        headline=(
            f"Stare long costa {fmt.percent(abs(funding), decimals=1, force_sign=False)} l'anno"
            if long_pagano
            else f"Stare short costa {fmt.percent(abs(funding), decimals=1, force_sign=False)} l'anno"
        ),
        body=corpo,
        hero_value=fmt.percent(funding, decimals=1),
        hero_caption="Funding annualizzato\nsui perpetui BTC",
        sign=SIGN_NEGATIVE if long_pagano else SIGN_POSITIVE,
        takeaway=(
            f"{chi_paga} {fmt.percent(abs(funding), decimals=1, force_sign=False)} "
            f"l'anno per restare esposti."
        ),
        meta={"funding_ann_pct": funding, "oi_usd": oi},
    )
