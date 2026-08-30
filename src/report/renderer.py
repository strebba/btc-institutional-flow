"""Rendering HTML del Desk Note.

Un renderer solo per due destinazioni: la pagina web servita da nginx e il
carosello PNG esportato con Playwright. Le card hanno sempre la geometria
nativa 1080x1350 (4:5, il formato di LinkedIn e X); la pagina web le rimpicciolisce
con una trasformazione CSS, l'export la disattiva e fotografa l'elemento a
grandezza naturale.

L'identita' visiva e' quella gia' dichiarata in .streamlit/config.toml — nero
puro e verde Wagmi — cosi' le card e la dashboard sembrano la stessa cosa.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

from src.report.narrative import Card, DeskNote

#: Cartella dei woff2 incorporati (vedi fonts/README.md per licenza e motivazione).
_FONTS_DIR = Path(__file__).parent / "fonts"

#: Geometria nativa di una card, in pixel. 4:5 e' il formato dei caroselli.
CARD_W = 1080
CARD_H = 1350

#: Palette Wagmi Lab, allineata a .streamlit/config.toml.
_NERO = "#000000"
_NEON = "#00FF9D"
_BIANCO = "#FFFFFF"
_AMBRA = "#FFB020"
_CORPO = "#B4BDC6"
_DIM = "#69737E"
_RIGA = "#23282D"

_FONT_SANS = (
    "'IBM Plex Sans','Helvetica Neue',Helvetica,Arial,'Segoe UI',Roboto,sans-serif"
)
_FONT_MONO = "'IBM Plex Mono','SF Mono',Menlo,Consolas,'Liberation Mono',monospace"


@lru_cache(maxsize=1)
def _font_faces() -> str:
    """Regole @font-face con i woff2 incorporati come data URI.

    Le card diventano PNG col marchio sopra: la tipografia non puo' dipendere
    dal fatto che il container raggiunga o meno fonts.gstatic.com. Se un file
    manca si degrada in silenzio sul fallback di sistema — meglio una card col
    font sbagliato che nessuna card.
    """
    facce = [
        ("IBM Plex Sans", "IBMPlexSans-var-latin.woff2", "100 700"),
        ("IBM Plex Mono", "IBMPlexMono-400-latin.woff2", "400"),
        ("IBM Plex Mono", "IBMPlexMono-500-latin.woff2", "500"),
        ("IBM Plex Mono", "IBMPlexMono-600-latin.woff2", "600"),
    ]
    out = []
    for famiglia, nome, peso in facce:
        path = _FONTS_DIR / nome
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        out.append(
            f"@font-face{{font-family:'{famiglia}';font-style:normal;"
            f"font-weight:{peso};font-display:block;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "".join(out)


def _hero_color(sign: str) -> str:
    """Il colore del numero grande e' il segno: verde compra, ambra vende o rompe."""
    return _AMBRA if sign == "negative" else _NEON


def _css(export: bool) -> str:
    # in export ogni card sta su una pagina sua a grandezza naturale; sul web
    # scalano dentro la griglia responsive
    layout = (
        f"""
  body {{ background:{_NERO}; padding:0; }}
  .page {{ display:block; }}
  .slot {{ width:{CARD_W}px; height:{CARD_H}px; }}
  .card {{ transform:none; }}
"""
        if export
        else f"""
  body {{ background:#0b0d0f; padding:32px 20px 80px; }}
  .page {{
    display:grid; gap:28px; margin:0 auto; max-width:1400px;
    grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  }}
  .slot {{ width:100%; aspect-ratio:{CARD_W}/{CARD_H}; position:relative; overflow:hidden;
           border-radius:3px; box-shadow:0 16px 40px -18px rgba(0,0,0,.9); }}
  .card {{ position:absolute; top:0; left:0; transform-origin:top left;
           transform:scale(calc(100cqw / {CARD_W})); }}
  .slot {{ container-type:inline-size; }}
  header.masthead {{
    max-width:1400px; margin:0 auto 34px; padding:0 4px;
    color:{_BIANCO}; font-family:{_FONT_SANS};
  }}
  header.masthead h1 {{ font-size:26px; font-weight:600; letter-spacing:-.02em; margin:0 0 8px; }}
  header.masthead .tape {{
    font-family:{_FONT_MONO}; font-size:12.5px; letter-spacing:.08em; color:{_DIM};
  }}
  header.masthead .warn {{
    margin-top:16px; padding:11px 15px; border:1px solid {_AMBRA};
    color:{_AMBRA}; font-family:{_FONT_MONO}; font-size:12px; line-height:1.6;
    letter-spacing:.03em; max-width:900px;
  }}
"""
    )

    return f"""
  *{{box-sizing:border-box;}}
  html,body{{margin:0;}}
  body{{-webkit-font-smoothing:antialiased;}}
{layout}
  .card {{
    width:{CARD_W}px; height:{CARD_H}px; background:{_NERO}; color:{_BIANCO};
    font-family:{_FONT_SANS}; font-size:26px; line-height:1.5;
    padding:76px 76px 60px; display:flex; flex-direction:column; overflow:hidden;
  }}
  .tape {{
    font-family:{_FONT_MONO}; font-size:.72em; letter-spacing:.06em; color:{_DIM};
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:1.5em;
    font-variant-numeric:tabular-nums;
  }}
  .eyebrow {{
    font-family:{_FONT_MONO}; font-size:.76em; font-weight:600; letter-spacing:.2em;
    text-transform:uppercase; color:{_NEON};
  }}
  .hr {{ height:1px; background:{_RIGA}; margin:.85em 0 1.5em; }}
  h2 {{
    font-size:2.05em; line-height:1.06; font-weight:600; letter-spacing:-.028em;
    margin:0; text-wrap:balance;
  }}
  .cover h2 {{ font-size:2.5em; }}
  .body {{ margin-top:1.5em; flex:1; min-height:0; }}
  .body p {{ margin:0 0 .85em; color:{_CORPO}; }}
  .body p:last-child {{ margin-bottom:0; }}
  .body b {{ color:{_BIANCO}; font-weight:600; }}
  .takeaways {{
    margin-top:auto; display:flex; flex-direction:column;
    border-top:1px solid {_RIGA};
  }}
  .takeaway {{
    display:grid; grid-template-columns:2.4em 1fr; gap:.9em; align-items:baseline;
    padding:1.15em 0; border-bottom:1px solid {_RIGA};
  }}
  .takeaway:last-child {{ border-bottom:none; }}
  .takeaway .n {{
    font-family:{_FONT_MONO}; font-size:.82em; font-weight:600; color:{_NEON};
    letter-spacing:.06em;
  }}
  .takeaway .t {{ color:{_CORPO}; }}
  .hero {{ margin-top:1.4em; }}
  .hero .n {{
    font-family:{_FONT_MONO}; font-weight:600; font-size:3.15em; line-height:.95;
    letter-spacing:-.045em; font-variant-numeric:tabular-nums; display:block;
  }}
  .hero .c {{
    font-family:{_FONT_MONO}; font-size:.78em; font-weight:500; letter-spacing:.13em;
    text-transform:uppercase; color:{_DIM}; line-height:1.45; margin-top:.7em; display:block;
  }}
  footer {{
    display:flex; justify-content:space-between; align-items:center; margin-top:2.6em;
    font-family:{_FONT_MONO}; font-size:.78em; letter-spacing:.1em; color:{_DIM};
  }}
  footer .mark {{ color:{_BIANCO}; font-weight:600; letter-spacing:.14em; }}
  .rail {{ height:3px; background:{_RIGA}; margin-top:.9em; position:relative; }}
  .rail i {{ position:absolute; top:0; bottom:0; left:0; background:{_NEON}; display:block; }}
"""


def _render_card(card: Card, tape: str) -> str:
    """Una card. La copertina porta i takeaway numerati, le altre il numero grande."""
    testa = (
        f'<div class="tape">{escape(tape)}</div>'
        f'<div class="eyebrow">{escape(card.eyebrow)}</div>'
        f'<div class="hr"></div>'
        f"<h2>{escape(card.headline)}</h2>"
    )

    if card.kind == "cover":
        voci = "".join(
            f'<div class="takeaway"><span class="n">{i:02d}</span>'
            f'<span class="t">{escape(t)}</span></div>'
            for i, t in enumerate(card.takeaways, start=1)
        )
        centro = f'<div class="takeaways">{voci}</div>'
    else:
        paragrafi = "".join(f"<p>{escape(p)}</p>" for p in card.body)
        centro = f'<div class="body">{paragrafi}</div>'
        if card.hero_value:
            didascalia = escape(card.hero_caption or "").replace("\n", "<br>")
            centro += (
                f'<div class="hero">'
                f'<span class="n" style="color:{_hero_color(card.sign)}">'
                f"{escape(card.hero_value)}</span>"
                f'<span class="c">{didascalia}</span>'
                f"</div>"
            )

    avanzamento = card.index / card.total * 100 if card.total else 0
    piede = (
        f'<footer><span class="mark">WAGMI LAB</span>'
        f"<span>{card.index:02d} / {card.total:02d}</span></footer>"
        f'<div class="rail"><i style="width:{avanzamento:.4g}%"></i></div>'
    )

    classe = "card cover" if card.kind == "cover" else "card"
    return f'<div class="slot"><div class="{classe}">{testa}{centro}{piede}</div></div>'


def render_html(note: DeskNote, *, export: bool = False, title: str = "Wagmi Desk Note") -> str:
    """Compone la pagina completa.

    Args:
        note: l'edizione da renderizzare.
        export: se True disattiva la riduzione responsive e mette ogni card a
            grandezza naturale, cosi' Playwright la fotografa a 1080x1350 esatti.
        title: titolo della pagina.

    Returns:
        Il documento HTML come stringa, completamente autoconsistente: font
        incorporati, nessuna richiesta di rete.
    """
    cards = "".join(_render_card(c, note.tape) for c in note.cards)

    intestazione = ""
    if not export:
        avvisi = ""
        if note.warnings:
            righe = "<br>".join(escape(w) for w in note.warnings)
            avvisi = (
                f'<div class="warn">Dati incompleti — queste card non vanno '
                f"pubblicate finché non è risolto:<br>{righe}</div>"
            )
        intestazione = (
            f'<header class="masthead"><h1>{escape(title)}</h1>'
            f'<div class="tape">{escape(note.tape)}</div>{avvisi}</header>'
        )

    vuoto = (
        ""
        if note.cards
        else '<header class="masthead"><h1>Nessuna edizione</h1>'
        '<div class="tape">Dati insufficienti per comporre le card.</div></header>'
    )

    return (
        "<!doctype html><html lang=\"it\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{_font_faces()}{_css(export)}</style></head><body>"
        f'{intestazione}{vuoto}<div class="page">{cards}</div>'
        "</body></html>"
    )
