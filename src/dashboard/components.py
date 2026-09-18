"""Componenti visivi custom (stile Desk Note) per la dashboard.

Solo HTML/CSS proprio via `st.html` — non tocca i widget nativi Streamlit.
`app.py` inietta gli stili una volta per run; i componenti usano classi
prefissate `wx-` per non collidere con Streamlit.

La tipografia replica il Desk Note: IBM Plex Mono per numeri/etichette mono,
IBM Plex Sans per il testo. Stessi colori: nero `#000`, neon `#00FF9D`,
ambra `#FFB020`, rosso `#FF0033`.
"""
from __future__ import annotations

from html import escape

import streamlit as st

STYLE = """<style>
.wx-tape {
  font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:.06em;
  color:#8b949e; text-transform:uppercase; font-variant-numeric:tabular-nums;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin:2px 0 14px;
}
.wx-hero {
  background:#0d0d0d; border:1px solid #23282d; border-radius:10px;
  padding:22px 26px; height:100%;
}
.wx-hero .wx-eyebrow {
  font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
  letter-spacing:.2em; text-transform:uppercase; color:#8b949e;
}
.wx-hero .wx-n {
  font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:54px;
  line-height:1; letter-spacing:-.04em; font-variant-numeric:tabular-nums;
  margin-top:12px; color:#fff;
}
.wx-hero .wx-n small { font-size:.4em; color:#8b949e; letter-spacing:0; }
.wx-hero .wx-verdict {
  font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600;
  letter-spacing:.14em; text-transform:uppercase; margin-top:10px; color:#8b949e;
}
.wx-hero.long .wx-n, .wx-hero.long .wx-verdict { color:#00FF9D; }
.wx-hero.risk .wx-n, .wx-hero.risk .wx-verdict { color:#FF0033; }
.wx-hero.mixed .wx-n, .wx-hero.mixed .wx-verdict { color:#FFB020; }

.wx-pillars { display:flex; gap:20px; flex-wrap:wrap; }
.wx-pillar { flex:1 1 0; min-width:120px; }
.wx-pillar-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.wx-pillar-name {
  font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.06em;
}
.wx-pillar-score {
  font-family:'IBM Plex Mono',monospace; font-size:16px; font-weight:600;
  font-variant-numeric:tabular-nums; color:#8b949e;
}
.wx-pillar-score.long { color:#00FF9D; }
.wx-pillar-score.risk { color:#FF0033; }
.wx-pillar-score.mixed { color:#FFB020; }
.wx-bar { height:4px; background:#23282d; border-radius:2px; margin:8px 0 6px; overflow:hidden; }
.wx-bar i { display:block; height:100%; }
.wx-bar i.long { background:#00FF9D; }
.wx-bar i.risk { background:#FF0033; }
.wx-bar i.mixed { background:#FFB020; }
.wx-pillar-meta { font-size:11px; color:#8b949e; line-height:1.45; }
</style>"""

_PILLAR_LABELS = {
    "gex": "GEX",
    "barrier": "Barrier",
    "etf_flows": "ETF Flows",
    "macro": "Macro",
}


def inject_style() -> None:
    """Inietta il CSS dei componenti. Chiamare una volta per run (in app.py)."""
    st.html(STYLE)


def tape(text: str) -> None:
    """Riga di stato mono uppercase (stile `tape` del Desk Note)."""
    st.html(f'<div class="wx-tape">{escape(text)}</div>')


def eyebrow(text: str) -> None:
    """Label di sezione mono uppercase (stile `eyebrow` del Desk Note)."""
    st.html(f'<div class="wx-eyebrow">{escape(text)}</div>')


def _signal_class(signal: str) -> str:
    if signal == "LONG":
        return "long"
    if signal == "RISK_OFF":
        return "risk"
    return "mixed"


def hero(score: float, signal: str, caption: str = "") -> None:
    """Numero grande del segnale composito + verdetto, colorato per segno."""
    cls = _signal_class(signal)
    cap = f'<div class="wx-tape" style="margin:12px 0 0">{escape(caption)}</div>' if caption else ""
    st.html(
        f'<div class="wx-hero {cls}">'
        f'<div class="wx-eyebrow">Segnale composito</div>'
        f'<div class="wx-n">{score:.0f}<small>/100</small></div>'
        f'<div class="wx-verdict">{escape(signal)}</div>'
        f"{cap}</div>"
    )


def pillar_bars(pillars: list[dict]) -> None:
    """Quattro mini-bar (score + peso + lettura), una per pilastro."""
    order = ["gex", "barrier", "etf_flows", "macro"]
    by_name = {p["name"]: p for p in pillars}
    rows = []
    for name in order:
        p = by_name.get(name)
        if p is None:
            continue
        score = p.get("score")
        if score is None:
            cls, pct, val = "", 0.0, "n/d"
        else:
            cls = "long" if score >= 65 else "risk" if score < 40 else "mixed"
            pct = max(0.0, min(100.0, float(score)))
            val = f"{score:.0f}"
        rows.append(
            f'<div class="wx-pillar">'
            f'<div class="wx-pillar-head">'
            f'<span class="wx-pillar-name">{escape(_PILLAR_LABELS.get(name, name))}</span>'
            f'<span class="wx-pillar-score {cls}">{val}</span></div>'
            f'<div class="wx-bar"><i class="{cls}" style="width:{pct:.0f}%"></i></div>'
            f'<div class="wx-pillar-meta">peso {(p.get("weight") or 0) * 100:.0f}% · '
            f'{escape(p.get("reason") or "—")}</div></div>'
        )
    st.html(f'<div class="wx-pillars">{"".join(rows)}</div>')
