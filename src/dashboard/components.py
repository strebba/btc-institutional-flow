"""Componenti visivi custom per la dashboard.

Solo HTML/CSS proprio via `st.html` — non tocca i widget nativi Streamlit.
`app.py` inietta gli stili una volta per run; i componenti usano classi
prefissate `wx-` per non collidere con Streamlit.

Tipografia: IBM Plex Mono per numeri/etichette mono, IBM Plex Sans per il
testo. Colori Wagmi Lab: nero `#000`, neon `#00FF9D`, rosso `#FF0033`.
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
.wx-eyebrow {
  font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
  letter-spacing:.2em; text-transform:uppercase; color:#8b949e; margin-bottom:6px;
}
</style>"""


def inject_style() -> None:
    """Inietta il CSS dei componenti. Chiamare una volta per run (in app.py)."""
    st.html(STYLE)


def tape(text: str) -> None:
    """Riga di stato mono uppercase."""
    st.html(f'<div class="wx-tape">{escape(text)}</div>')


def eyebrow(text: str) -> None:
    """Label di sezione mono uppercase."""
    st.html(f'<div class="wx-eyebrow">{escape(text)}</div>')
