"""Genera un indicatore TradingView (Pine Script v6) dai livelli GEX correnti.

Pine Script non può fare chiamate HTTP, quindi un indicatore non può leggere
`/api/gex` in tempo reale: `build_pine_indicator()` congela uno snapshot GEX
in un file `.pine` con gamma flip, put/call wall e max pain come livelli
fissi. Per aggiornarli bisogna rigenerare e re-incollare nel Pine Editor.

Usata sia da `scripts/export_pine_indicator.py` (CLI) sia dal tasto "Genera
indicatore TradingView" nella tab GEX della dashboard, sulla stessa forma di
snapshot prodotta da `GexCalculator.gex_to_dict()` (+ chiave `regime` stringa).
"""
from __future__ import annotations

_TEMPLATE = """\
//@version=6
indicator("WAGMI Lab — BTC GEX Levels", shorttitle="GEX Levels", overlay=true, max_lines_count=20, max_labels_count=20)

// Generato {timestamp} — livelli statici, non si aggiornano da soli.
// Regime al calcolo: {regime_label} | Net GEX: {net_gex_m}M USD | Spot: {spot_price}
// Per aggiornare: rigenera e incolla di nuovo qui, sovrascrivendo tutto.

showFlip    = input.bool(true, "Gamma Flip")
showWalls   = input.bool(true, "Put / Call Wall")
showMaxPain = input.bool(true, "Max Pain")

{level_assignments}

{hlines}

var label lblInfo = na
{level_label_vars}
if barstate.islast
    label.delete(lblInfo)
    infoText = "GEX @ {timestamp}\\nRegime: {regime_label}\\nNet GEX: {net_gex_m}M USD"
    lblInfo := label.new(bar_index, high, infoText, xloc=xloc.bar_index, yloc=yloc.abovebar, style=label.style_label_down, color=color.new(color.gray, 20), textcolor=color.white, size=size.small)
{level_labels}
"""

# nome, variabile pine, campo sorgente in snapshot, colore, flag di toggle
_LEVELS = [
    ("Gamma Flip", "gammaFlip", "gamma_flip_price", "color.orange", "showFlip"),
    ("Put Wall", "putWall", "put_wall", "color.green", "showWalls"),
    ("Call Wall", "callWall", "call_wall", "color.red", "showWalls"),
    ("Max Pain", "maxPain", "max_pain", "color.blue", "showMaxPain"),
]


def build_pine_indicator(snapshot: dict) -> str:
    """Costruisce il sorgente Pine da uno snapshot GEX.

    Args:
        snapshot: dict con i campi di ``GexCalculator.gex_to_dict()``
            (``gamma_flip_price``, ``put_wall``, ``call_wall``, ``max_pain``,
            ``total_net_gex_m``, ``spot_price``, ``timestamp``) più una
            chiave ``regime`` stringa (es. ``"positive_gamma"``).

    Returns:
        Sorgente Pine Script v6 pronto da incollare nel Pine Editor.
    """
    regime = snapshot.get("regime") or "sconosciuto"

    assignments = []
    hlines = []
    label_vars = []
    labels = []
    for name, var, field, color, flag in _LEVELS:
        value = snapshot.get(field)
        if value is None:
            assignments.append(f"{var} = na  // {name}: non disponibile alla generazione")
            continue
        assignments.append(f"{var} = {value:.2f}")
        hlines.append(
            f'hline({flag} ? {var} : na, title="{name}", color={color}, '
            f'linestyle=hline.style_dashed, linewidth=1)'
        )
        lbl_name = f"lbl{var[0].upper()}{var[1:]}"
        label_vars.append(f"var label {lbl_name} = na")
        labels.append(
            f'    if {flag} and not na({var})\n'
            f'        label.delete({lbl_name})\n'
            f'        {lbl_name} := label.new(bar_index + 3, {var}, "{name} " + str.tostring({var}, format.mintick), '
            f'xloc=xloc.bar_index, style=label.style_label_left, color={color}, '
            f'textcolor=color.white, size=size.small)'
        )

    return _TEMPLATE.format(
        timestamp=snapshot.get("timestamp", "?"),
        regime_label=regime,
        net_gex_m=snapshot.get("total_net_gex_m", "?"),
        spot_price=snapshot.get("spot_price", "?"),
        level_assignments="\n".join(assignments),
        hlines="\n".join(hlines),
        level_label_vars="\n".join(label_vars),
        level_labels="\n".join(labels),
    )
