"""Assemblaggio del Desk Note: dai fatti candidati alle sei card.

La parte difficile del report non è renderizzare le immagini: è decidere quale
dei fatti di oggi merita il titolo. Quella decisione vive qui, in un posto solo,
così la pagina web, il carosello PNG e il recap Telegram restano tre renderer
dello stesso JSON.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.report import formatting as fmt
from src.report.facts import Fact, extract_all

#: Numero di card della serie, copertina inclusa.
N_CARDS = 6

#: Massimo di card per famiglia, così la serie non diventa sei modi di dire GEX.
MAX_PER_TOPIC = 2

#: Sotto questa salienza un fatto non entra in edizione nemmeno se c'è posto.
MIN_SALIENCE = 0.20


@dataclass
class Card:
    """Una card pubblicabile."""

    index: int
    total: int
    eyebrow: str
    headline: str
    body: list[str]
    hero_value: str | None = None
    hero_caption: str | None = None
    sign: str = "neutral"
    kind: str = "fact"          # "cover" | "fact"
    takeaways: list[str] = field(default_factory=list)
    source_key: str = ""


@dataclass
class DeskNote:
    """L'edizione completa."""

    generated_at: str
    tape: str
    cards: list[Card]
    facts_considered: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "tape": self.tape,
            "cards": [asdict(c) for c in self.cards],
            "facts_considered": self.facts_considered,
            "warnings": self.warnings,
        }


# ─── Selezione ────────────────────────────────────────────────────────────────


def select_facts(facts: list[Fact], *, limit: int = N_CARDS - 1) -> list[Fact]:
    """Ordina per salienza e impone varietà di famiglia.

    Prima passata: al massimo MAX_PER_TOPIC fatti per topic, così una giornata
    tutta di GEX non produce cinque card di GEX. Seconda passata: se restano
    buchi li riempie con i migliori scartati, perché una card vuota è peggio di
    una card ripetitiva.
    """
    ordinati = sorted(
        (f for f in facts if f.salience >= MIN_SALIENCE),
        key=lambda f: f.salience,
        reverse=True,
    )

    scelti: list[Fact] = []
    per_topic: dict[str, int] = {}
    scartati: list[Fact] = []

    for f in ordinati:
        if len(scelti) >= limit:
            break
        if per_topic.get(f.topic, 0) >= MAX_PER_TOPIC:
            scartati.append(f)
            continue
        scelti.append(f)
        per_topic[f.topic] = per_topic.get(f.topic, 0) + 1

    for f in scartati:
        if len(scelti) >= limit:
            break
        scelti.append(f)

    return scelti


# ─── Composizione ─────────────────────────────────────────────────────────────


def _build_tape(gex: dict | None, generated_at: datetime) -> str:
    """La striscia di contesto ripetuta su ogni card, come la status bar di un terminale."""
    pezzi = []
    spot = ((gex or {}).get("snapshot") or {}).get("spot_price")
    total = ((gex or {}).get("snapshot") or {}).get("total_net_gex")
    flip = ((gex or {}).get("snapshot") or {}).get("gamma_flip_price")
    if spot:
        pezzi.append(f"BTC {fmt.price(spot)}")
    if total is not None:
        pezzi.append(f"GEX {fmt.usd_millions(total)}")
    if flip:
        pezzi.append(f"FLIP {fmt.price(flip)}")
    pezzi.append(generated_at.strftime("%d/%m %H:%M UTC"))
    return " · ".join(pezzi)


def _cover_headline(gex: dict | None, principale: Fact | None) -> str:
    """La tesi di copertina.

    Non ripete il titolo della card 2 — sarebbe la stessa frase due volte di
    seguito. Preferisce inquadrare il prezzo dentro il corridoio put wall /
    call wall, che è il contesto sotto cui stanno tutte le altre card, e
    ripiega sul fatto più saliente solo se il corridoio non è calcolabile.
    """
    snap = (gex or {}).get("snapshot") or {}
    spot, put_wall, call_wall = snap.get("spot_price"), snap.get("put_wall"), snap.get("call_wall")
    if spot and put_wall and call_wall and call_wall > put_wall:
        ampiezza = (call_wall - put_wall) / spot * 100.0
        return (
            f"{fmt.price(spot)} dollari, dentro una scatola larga il "
            f"{fmt.percent(ampiezza, decimals=0, force_sign=False)}"
        )
    return principale.headline if principale else "Il quadro di oggi"


def _build_cover(scelti: list[Fact], eyebrow: str, gex: dict | None) -> Card:
    """La copertina riassume in una tesi i tre fatti più salienti."""
    top = scelti[:3]
    principale = top[0] if top else None
    headline = _cover_headline(gex, principale)

    return Card(
        index=1,
        total=N_CARDS,
        eyebrow=eyebrow,
        headline=headline,
        body=[],
        kind="cover",
        takeaways=[f.takeaway or f.headline for f in top],
        sign=principale.sign if principale else "neutral",
        source_key="cover",
    )


def build_desk_note(
    *,
    gex: dict | None = None,
    barriers: dict | None = None,
    flows: dict | None = None,
    signals: dict | None = None,
    forecast: dict | None = None,
    macro: dict | None = None,
    generated_at: datetime | None = None,
    eyebrow: str = "Wagmi Lab · Desk",
) -> DeskNote:
    """Costruisce l'edizione a partire dai payload degli endpoint esistenti.

    Args:
        gex: payload di ``/api/gex`` (chiave ``data``).
        barriers: payload di ``/api/barriers``.
        flows: payload di ``/api/flows``.
        signals: payload di ``/api/signals``.
        forecast: payload di ``/api/forecast/status``, opzionale.
        macro: payload di ``/api/macro``, opzionale — serve solo a spiegare nei
            warning *perche'* il pilastro macro e' scoperto.
        generated_at: istante dell'edizione, per test riproducibili.
        eyebrow: occhiello ripetuto su tutte le card.

    Returns:
        DeskNote con al più N_CARDS card. Se i dati bastano solo per due fatti,
        l'edizione ne ha tre invece di riempire con frasi vuote.
    """
    ts = generated_at or datetime.now(timezone.utc)

    facts = extract_all(
        gex=gex, barriers=barriers, flows=flows, signals=signals,
        forecast=forecast, macro=macro,
    )
    scelti = select_facts(facts)

    cards: list[Card] = []
    if scelti:
        cards.append(_build_cover(scelti, eyebrow, gex))

    for i, f in enumerate(scelti, start=2):
        cards.append(
            Card(
                index=i,
                total=len(scelti) + 1,
                eyebrow=eyebrow,
                headline=f.headline,
                body=f.body,
                hero_value=f.hero_value,
                hero_caption=f.hero_caption,
                sign=f.sign,
                kind="fact",
                source_key=f.key,
            )
        )

    # il totale reale può essere < N_CARDS: allineo l'indicatore di pagina
    for c in cards:
        c.total = len(cards)

    return DeskNote(
        generated_at=ts.isoformat(),
        tape=_build_tape(gex, ts),
        cards=cards,
        facts_considered=len(facts),
        warnings=_collect_warnings(signals, macro),
    )


def _collect_warnings(
    signals: dict | None, macro: dict | None = None
) -> list[str]:
    """Avvisi che devono impedire la pubblicazione di una card, non decorarla.

    Un avviso serve solo se dice cosa fare: "pilastro macro senza dati" manda a
    indagare, "manca la chiave CoinGlass" si risolve in due minuti. Quando la
    causa è nota la si nomina, invece di elencare i sintomi.
    """
    out: list[str] = []

    stato_macro = (macro or {}).get("source_status")
    if stato_macro == "no_api_key":
        out.append(
            "pilastro macro spento: COINGLASS_API_KEY non è configurata — "
            "impostala fra gli envs dell'app (Console DO → Settings → App Spec)"
        )
    elif stato_macro == "unavailable":
        out.append("pilastro macro senza dati: CoinGlass non risponde")
    elif stato_macro == "partial_coingecko":
        # Non e' spento: funding e open interest ci sono, e il funding da solo
        # pesa 0,30 del pilastro. Dire "spento" manderebbe a cercare una chiave
        # che ora serve per completare, non per accendere.
        out.append(
            "pilastro macro a metà: funding e open interest arrivano da CoinGecko, "
            "long/short ratio e liquidazioni restano scoperti (servirebbe CoinGlass)"
        )

    pillars = (signals or {}).get("pillars") or []
    for p in pillars:
        # la causa del macro è già stata nominata sopra: non ripeterla per sintomi
        if p.get("name") == "macro" and stato_macro in (
            "no_api_key", "unavailable", "partial_coingecko"
        ):
            continue
        comps = p.get("components") or {}
        vuoti = [k for k, v in comps.items() if v is None]
        if vuoti and len(vuoti) == len(comps):
            out.append(f"pilastro {p.get('name')} senza dati: {', '.join(vuoti)}")
        elif len(vuoti) >= 4:
            out.append(
                f"pilastro {p.get('name')} con {len(vuoti)} fattori su "
                f"{len(comps)} mancanti: {', '.join(vuoti)}"
            )
    return out
