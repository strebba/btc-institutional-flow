"""Esporta i livelli GEX in un indicatore Pine Script v6 per TradingView.

Pine Script gira nella sandbox di TradingView e **non può chiamare API esterne**:
l'unico modo di portare il GEX di Deribit su un grafico BTC è generare uno script
con i livelli già incorporati e rigenerarlo quando i dati cambiano.

Lo script prodotto contiene:
  * i livelli dello snapshot corrente — gamma flip, put wall, call wall, max pain;
  * il profilo call/put GEX per strike (istogramma orizzontale a destra del prezzo);
  * la storia dei livelli (step line) da ``gex_snapshots``, se disponibile;
  * un pannello di riepilogo (regime live, net GEX, PCR, distanze, età dei dati);
  * alert su attraversamento del gamma flip e prossimità ai wall.

Il modulo è puro: non fa rete né I/O. Chi lo usa (router ``/api/gex/pine``,
``scripts/export_pine.py``) gli passa snapshot e storia già pronti.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from src.gex.models import GexByStrike, GexSnapshot

# Limiti di TradingView: max 500 box per indicatore → 2 box per strike (call+put).
MAX_PROFILE_STRIKES = 24
# Oltre ~60 argomenti per `array.from` conviene spezzare in chunk concatenati.
_ARRAY_CHUNK = 60

__all__ = [
    "MAX_PROFILE_STRIKES",
    "build_pine_script",
    "normalize_history",
    "select_profile_strikes",
]


# ─── Formattazione letterali Pine ─────────────────────────────────────────────


def _num(value: Any, default: float = 0.0) -> float:
    """Converte un valore in float, mappando None/NaN/non numerici su default."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):  # NaN / Inf
        return default
    return out


def _f(value: Any, decimals: int = 6, default: float = 0.0) -> str:
    """Letterale float Pine: sempre con punto decimale, senza zeri superflui."""
    text = f"{round(_num(value, default), decimals):.{decimals}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _pine_str(value: Any) -> str:
    """Letterale stringa Pine con escaping di quote e newline."""
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + text.replace("\n", " ").replace("\r", " ") + '"'


def _array_literal(name: str, pine_type: str, values: Sequence[str]) -> tuple[str, list[str]]:
    """Genera la dichiarazione di un array Pine, spezzata in chunk se lunga.

    Args:
        name: nome della variabile Pine.
        pine_type: "float" o "int".
        values: letterali già formattati.

    Returns:
        tuple: (riga di dichiarazione, righe `array.concat` da eseguire una volta).
    """
    if not values:
        return f"var array<{pine_type}> {name} = array.new<{pine_type}>()", []

    head = values[:_ARRAY_CHUNK]
    decl = f"var array<{pine_type}> {name} = array.from({', '.join(head)})"
    chunks = [
        f"    {name} := array.concat({name}, array.from({', '.join(values[i:i + _ARRAY_CHUNK])}))"
        for i in range(_ARRAY_CHUNK, len(values), _ARRAY_CHUNK)
    ]
    return decl, chunks


# ─── Selezione dati ───────────────────────────────────────────────────────────


def select_profile_strikes(
    gex_by_strike: Sequence[GexByStrike],
    spot_price: float,
    max_strikes: int = MAX_PROFILE_STRIKES,
    range_pct: float = 0.15,
) -> list[GexByStrike]:
    """Seleziona gli strike più rilevanti attorno allo spot per il profilo.

    Filtra gli strike entro ``range_pct`` dallo spot, tiene i ``max_strikes`` con
    il maggiore |GEX| (call o put) e li riordina per strike crescente.

    Args:
        gex_by_strike: profilo completo dello snapshot.
        spot_price: spot BTC al momento del calcolo.
        max_strikes: numero massimo di strike da tenere.
        range_pct: ampiezza della finestra attorno allo spot (0.15 = ±15%).

    Returns:
        list[GexByStrike] ordinata per strike crescente (vuota se non c'è nulla).
    """
    if not gex_by_strike or spot_price <= 0 or max_strikes <= 0:
        return []

    in_range = [
        g for g in gex_by_strike
        if g.strike > 0 and abs(g.strike - spot_price) / spot_price <= range_pct
    ]
    if not in_range:  # finestra vuota → ripiega sugli strike più vicini allo spot
        in_range = sorted(gex_by_strike, key=lambda g: abs(g.strike - spot_price))[:max_strikes]

    ranked = sorted(
        in_range,
        key=lambda g: max(abs(_num(g.call_gex)), abs(_num(g.put_gex))),
        reverse=True,
    )[:max_strikes]
    return sorted(ranked, key=lambda g: g.strike)


def _strike_step(strikes: Sequence[float], spot_price: float) -> float:
    """Passo tipico fra strike consecutivi (mediana), con fallback sul 2% dello spot."""
    diffs = [b - a for a, b in zip(strikes, strikes[1:]) if b > a]
    if diffs:
        return float(statistics.median(diffs))
    return max(_num(spot_price) * 0.02, 1.0)


def _to_epoch_ms(value: Any) -> Optional[int]:
    """Converte datetime/date/str ISO/Timestamp in epoch ms UTC (None se impossibile)."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    to_pydatetime = getattr(value, "to_pydatetime", None)  # pd.Timestamp
    if callable(to_pydatetime):
        value = to_pydatetime()
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    isoformat = getattr(value, "isoformat", None)  # datetime.date
    if callable(isoformat):
        return _to_epoch_ms(f"{isoformat()}T00:00:00+00:00")
    return None


def normalize_history(history: Any, limit: int = 365) -> list[tuple[int, float, float, float]]:
    """Normalizza la storia dei livelli in tuple (epoch_ms, flip, put_wall, call_wall).

    Accetta il DataFrame di ``GexDB.get_walls_series()``, una lista di ``GexSnapshot``
    o una lista di mapping con le stesse chiavi. Le righe senza timestamp valido o
    con tutti i livelli mancanti vengono scartate; l'output è ordinato per tempo.

    Args:
        history: DataFrame / sequenza di snapshot / sequenza di mapping.
        limit: numero massimo di punti (tiene i più recenti).

    Returns:
        list[tuple[int, float, float, float]]: livelli a 0.0 quando mancanti.
    """
    if history is None:
        return []

    rows: list[tuple[Any, Any, Any, Any]] = []
    if hasattr(history, "itertuples") and hasattr(history, "columns"):  # pandas DataFrame
        cols = set(history.columns)
        for idx, row in zip(history.index, history.to_dict("records")):
            rows.append((
                idx,
                row.get("gamma_flip_price") if "gamma_flip_price" in cols else None,
                row.get("put_wall") if "put_wall" in cols else None,
                row.get("call_wall") if "call_wall" in cols else None,
            ))
    else:
        for item in history:
            if isinstance(item, Mapping):
                rows.append((
                    item.get("timestamp") or item.get("date"),
                    item.get("gamma_flip_price"),
                    item.get("put_wall"),
                    item.get("call_wall"),
                ))
            else:
                rows.append((
                    getattr(item, "timestamp", None),
                    getattr(item, "gamma_flip_price", None),
                    getattr(item, "put_wall", None),
                    getattr(item, "call_wall", None),
                ))

    out: list[tuple[int, float, float, float]] = []
    for ts, flip, put_wall, call_wall in rows:
        epoch_ms = _to_epoch_ms(ts)
        if epoch_ms is None:
            continue
        levels = (_num(flip), _num(put_wall), _num(call_wall))
        if not any(v > 0 for v in levels):
            continue
        out.append((epoch_ms, *levels))

    out.sort(key=lambda r: r[0])
    return out[-limit:] if limit and limit > 0 else out


# ─── Generazione script ───────────────────────────────────────────────────────


_TEMPLATE = '''//@version=6
// ═════════════════════════════════════════════════════════════════════════════
//  BTC GEX — Gamma Flip, Put/Call Walls & profilo GEX per strike
//  Generato da btc-institutional-flow · src/gex/pine_exporter.py
//
//  Snapshot     : {generated_at}
//  Spot BTC     : {spot_hint} USD
//  Regime GEX   : {regime}   ·   Net GEX: {net_gex_hint} M$
//  Strike       : {n_strikes} nel profilo   ·   Storico: {n_history} punti
//
//  ⚠ Pine non può interrogare API: i livelli qui sotto sono uno SNAPSHOT.
//    Per aggiornarli rigenera lo script e ri-incollalo nel Pine Editor:
//      make export-pine                 → exports/btc_gex_tradingview.pine
//      curl {api_hint}   → stesso contenuto in text/plain
//
//  GEX = gamma × OI × contract_size × spot² × 0.01   (call +, put −), fonte Deribit.
//  Prezzi in USD/BTC: su grafici IBIT imposta la "Scala prezzi" nelle impostazioni.
// ═════════════════════════════════════════════════════════════════════════════
indicator({title}, "GEX", overlay = true, max_boxes_count = 500, max_lines_count = 500, max_labels_count = 500)

// ───────────────────────────── DATI AUTO-GENERATI ────────────────────────────
GEN_TS      = {generated_at_str}
GEN_MS      = {generated_ms}
SNAP_SPOT   = {spot}
FLIP        = {flip}
PUT_WALL    = {put_wall}
CALL_WALL   = {call_wall}
MAX_PAIN    = {max_pain}
NET_GEX_M   = {net_gex_m}
PCR         = {pcr}
GEX_PCTL    = {gex_pctl}
REGIME      = {regime_str}
STRIKE_STEP = {strike_step}

{arr_strikes}
{arr_call_gex}
{arr_put_gex}
{arr_hist_t}
{arr_hist_flip}
{arr_hist_pw}
{arr_hist_cw}
{chunk_block}
// ─────────────────────────────────── INPUT ───────────────────────────────────
gL = "Livelli"
gP = "Profilo GEX per strike"
gH = "Storico livelli"
gA = "Alert"
gV = "Aspetto"

scaleF   = input.float(1.0, {scale_title}, minval = 0.0000001, group = gL, tooltip = "I livelli sono in USD/BTC. Su un grafico IBIT usa il ratio IBIT/BTC come fattore di scala.")
showFlip = input.bool(true, "Gamma flip", group = gL)
showWall = input.bool(true, "Put wall / Call wall", group = gL)
showPain = input.bool(true, "Max pain", group = gL)
levelLen = input.int(150, "Lunghezza livelli (barre)", minval = 5, maxval = 1000, group = gL)
rightPad = input.int(25, "Estensione a destra (barre)", minval = 0, maxval = 200, group = gL)

showProf = input.bool(true, "Mostra profilo call/put GEX", group = gP)
profW    = input.int(60, "Larghezza massima barre", minval = 5, maxval = 150, group = gP)
profGap  = input.int(10, "Distacco dal prezzo (barre)", minval = 0, maxval = 100, group = gP)
// rightPad + profGap + profW ≤ 450: TradingView non disegna oltre 500 barre nel futuro.

showHist = input.bool(true, "Livelli storici (step line)", group = gH)

wallTol  = input.float(0.5, "Tolleranza wall (%)", minval = 0.0, step = 0.1, group = gA)
staleH   = input.int(36, "Soglia dati stantii (ore)", minval = 1, group = gA)

showTbl  = input.bool(true, "Pannello riepilogo", group = gV)
showBg   = input.bool(true, "Sfondo per regime gamma", group = gV)
cCall    = input.color(#26a69a, "Call / gamma positivo", group = gV)
cPut     = input.color(#ef5350, "Put / gamma negativo", group = gV)
cFlip    = input.color(#f0b90b, "Gamma flip", group = gV)
cPain    = input.color(#9598a1, "Max pain", group = gV)

// ────────────────────────────────── LIVELLI ──────────────────────────────────
f_lvl(v) => v > 0 ? v * scaleF : na

flip = f_lvl(FLIP)
pw   = f_lvl(PUT_WALL)
cw   = f_lvl(CALL_WALL)
mp   = f_lvl(MAX_PAIN)

// Regime "live": prezzo sopra il flip → dealer long gamma (hedging che comprime la
// volatilità); sotto il flip → short gamma (hedging pro-ciclico che la amplifica).
inLongGamma = not na(flip) and close > flip
regimeTxt   = na(flip) ? "n/d" : inLongGamma ? "LONG GAMMA (vol compressa)" : "SHORT GAMMA (vol amplificata)"

bgcolor(showBg and not na(flip) ? (inLongGamma ? color.new(cCall, 93) : color.new(cPut, 93)) : na, title = "Regime gamma")

// ──────────────────────── STORICO LIVELLI (step line) ────────────────────────
histN = array.size(HIST_T)

var int hIdx = 0
if histN > 1
    while hIdx < histN - 1 and array.get(HIST_T, hIdx + 1) <= time
        hIdx += 1

float hFlip = na
float hPw   = na
float hCw   = na
if showHist and histN > 0
    if time >= array.get(HIST_T, 0)
        hFlip := f_lvl(array.get(HIST_FLIP, hIdx))
        hPw   := f_lvl(array.get(HIST_PW, hIdx))
        hCw   := f_lvl(array.get(HIST_CW, hIdx))

plot(hFlip, "Gamma flip storico", color = color.new(cFlip, 20), style = plot.style_stepline, linewidth = 1)
plot(hPw, "Put wall storico", color = color.new(cPut, 40), style = plot.style_stepline, linewidth = 1)
plot(hCw, "Call wall storico", color = color.new(cCall, 40), style = plot.style_stepline, linewidth = 1)

// Solo Data Window / alert: i livelli dello snapshot come serie costanti.
plot(flip, "Gamma flip (snapshot)", color = cFlip, display = display.data_window)
plot(pw, "Put wall (snapshot)", color = cPut, display = display.data_window)
plot(cw, "Call wall (snapshot)", color = cCall, display = display.data_window)

// ─────────────────── LIVELLI CORRENTI: linea + etichetta ─────────────────────
f_level(ln, lb, price, col, txt, sty, on) =>
    line.delete(ln)
    label.delete(lb)
    line  newLine  = na
    label newLabel = na
    if on and not na(price)
        newLine := line.new(math.max(0, bar_index - levelLen), price, bar_index + rightPad, price, xloc = xloc.bar_index, color = col, style = sty, width = 2)
        newLabel := label.new(bar_index + rightPad, price, txt + "  " + str.tostring(price, format.mintick), xloc = xloc.bar_index, style = label.style_label_left, color = color.new(col, 80), textcolor = col, size = size.small)
    [newLine, newLabel]

var line  lnFlip = na
var label lbFlip = na
var line  lnPw   = na
var label lbPw   = na
var line  lnCw   = na
var label lbCw   = na
var line  lnMp   = na
var label lbMp   = na

if barstate.islast
    [l1, b1] = f_level(lnFlip, lbFlip, flip, cFlip, "Gamma flip", line.style_dashed, showFlip)
    lnFlip := l1
    lbFlip := b1
    [l2, b2] = f_level(lnPw, lbPw, pw, cPut, "Put wall", line.style_solid, showWall)
    lnPw := l2
    lbPw := b2
    [l3, b3] = f_level(lnCw, lbCw, cw, cCall, "Call wall", line.style_solid, showWall)
    lnCw := l3
    lbCw := b3
    [l4, b4] = f_level(lnMp, lbMp, mp, cPain, "Max pain", line.style_dotted, showPain)
    lnMp := l4
    lbMp := b4

// ─────────────────── PROFILO GEX PER STRIKE (call vs put) ────────────────────
var array<box> profBoxes = array.new<box>()

if barstate.islast
    for b in profBoxes
        box.delete(b)
    array.clear(profBoxes)

    nStrikes = array.size(STRIKES)
    if showProf and nStrikes > 0
        maxAbs = 0.0
        for i = 0 to nStrikes - 1
            maxAbs := math.max(maxAbs, math.abs(array.get(CALL_GEX, i)))
            maxAbs := math.max(maxAbs, math.abs(array.get(PUT_GEX, i)))
        if maxAbs > 0
            x0 = bar_index + rightPad + profGap
            h  = STRIKE_STEP * scaleF * 0.36
            for i = 0 to nStrikes - 1
                k  = array.get(STRIKES, i) * scaleF
                wc = int(math.round(profW * math.abs(array.get(CALL_GEX, i)) / maxAbs))
                wp = int(math.round(profW * math.abs(array.get(PUT_GEX, i)) / maxAbs))
                if wc > 0
                    array.push(profBoxes, box.new(x0, k + h, x0 + wc, k, xloc = xloc.bar_index, border_color = color.new(cCall, 40), bgcolor = color.new(cCall, 60)))
                if wp > 0
                    array.push(profBoxes, box.new(x0, k, x0 + wp, k - h, xloc = xloc.bar_index, border_color = color.new(cPut, 40), bgcolor = color.new(cPut, 60)))

// ───────────────────────────── PANNELLO RIEPILOGO ────────────────────────────
f_dist(lvl) =>
    if na(lvl) or close == 0
        "n/d"
    else
        delta = (lvl - close) / close * 100
        str.tostring(lvl, format.mintick) + "   " + (delta >= 0 ? "+" : "") + str.tostring(delta, "#.##") + "%"

ageH    = (timenow - GEN_MS) / 3600000.0
isStale = ageH > staleH

var table tbl = table.new(position.top_right, 2, 10, border_width = 1, frame_width = 1, frame_color = color.new(color.gray, 50))
if showTbl and barstate.islast
    table.cell(tbl, 0, 0, "BTC GEX — snapshot", text_color = color.white, bgcolor = color.new(color.gray, 20), text_size = size.small)
    table.cell(tbl, 1, 0, GEN_TS, text_color = isStale ? cPut : color.new(color.white, 20), bgcolor = color.new(color.gray, 20), text_size = size.small)
    table.cell(tbl, 0, 1, "Regime live", text_size = size.small)
    table.cell(tbl, 1, 1, regimeTxt, text_color = inLongGamma ? cCall : cPut, text_size = size.small)
    table.cell(tbl, 0, 2, "Regime snapshot", text_size = size.small)
    table.cell(tbl, 1, 2, REGIME, text_size = size.small)
    table.cell(tbl, 0, 3, "Net GEX", text_size = size.small)
    table.cell(tbl, 1, 3, str.tostring(NET_GEX_M, "#.##") + " M$", text_color = NET_GEX_M >= 0 ? cCall : cPut, text_size = size.small)
    table.cell(tbl, 0, 4, "Gamma flip", text_size = size.small)
    table.cell(tbl, 1, 4, f_dist(flip), text_color = cFlip, text_size = size.small)
    table.cell(tbl, 0, 5, "Call wall", text_size = size.small)
    table.cell(tbl, 1, 5, f_dist(cw), text_color = cCall, text_size = size.small)
    table.cell(tbl, 0, 6, "Put wall", text_size = size.small)
    table.cell(tbl, 1, 6, f_dist(pw), text_color = cPut, text_size = size.small)
    table.cell(tbl, 0, 7, "Max pain", text_size = size.small)
    table.cell(tbl, 1, 7, f_dist(mp), text_color = cPain, text_size = size.small)
    table.cell(tbl, 0, 8, "Put/Call OI", text_size = size.small)
    table.cell(tbl, 1, 8, str.tostring(PCR, "#.###") + "   (pctl GEX " + str.tostring(GEX_PCTL, "#") + ")", text_size = size.small)
    table.cell(tbl, 0, 9, "Età dati", text_size = size.small)
    table.cell(tbl, 1, 9, str.tostring(ageH, "#.#") + " h" + (isStale ? "  ⚠ rigenera" : ""), text_color = isStale ? cPut : color.gray, text_size = size.small)

// ─────────────────────────────────── ALERT ───────────────────────────────────
crossUp = not na(flip) and ta.crossover(close, flip)
crossDn = not na(flip) and ta.crossunder(close, flip)
nearCw  = not na(cw) and math.abs(close - cw) / cw * 100 <= wallTol
nearPw  = not na(pw) and math.abs(close - pw) / pw * 100 <= wallTol
prevCw  = na(nearCw[1]) ? false : nearCw[1]
prevPw  = na(nearPw[1]) ? false : nearPw[1]
hitCw   = nearCw and not prevCw
hitPw   = nearPw and not prevPw

alertcondition(crossUp, "Gamma flip ↑", "BTC sopra il gamma flip → dealer long gamma, volatilità compressa")
alertcondition(crossDn, "Gamma flip ↓", "BTC sotto il gamma flip → dealer short gamma, volatilità amplificata")
alertcondition(hitCw, "Call wall raggiunto", "BTC in prossimità del call wall (resistenza meccanica)")
alertcondition(hitPw, "Put wall raggiunto", "BTC in prossimità del put wall (supporto meccanico)")

if barstate.isconfirmed
    if crossUp
        alert("BTC ha superato il gamma flip " + str.tostring(flip, format.mintick), alert.freq_once_per_bar_close)
    if crossDn
        alert("BTC è sceso sotto il gamma flip " + str.tostring(flip, format.mintick), alert.freq_once_per_bar_close)
    if hitCw
        alert("BTC in prossimità del call wall " + str.tostring(cw, format.mintick), alert.freq_once_per_bar_close)
    if hitPw
        alert("BTC in prossimità del put wall " + str.tostring(pw, format.mintick), alert.freq_once_per_bar_close)
'''


def build_pine_script(
    snapshot: GexSnapshot,
    *,
    regime: str = "n/d",
    gex_percentile: Optional[float] = None,
    history: Any = None,
    max_strikes: int = MAX_PROFILE_STRIKES,
    range_pct: float = 0.15,
    history_limit: int = 365,
    ibit_ratio: Optional[float] = None,
    title: str = "BTC GEX — Walls & Gamma Flip",
    api_hint: str = "https://btc-institutional-flow-tpw9m.ondigitalocean.app/api/gex/pine",
) -> str:
    """Genera lo script Pine v6 con i livelli GEX dello snapshot.

    Args:
        snapshot: snapshot GEX da cui prendere livelli e profilo per strike.
        regime: etichetta di regime ('positive_gamma' | 'negative_gamma' | 'neutral').
        gex_percentile: percentile storico del GEX (0-100), opzionale.
        history: storia dei livelli (DataFrame di GexDB.get_walls_series() o
            sequenza di GexSnapshot/mapping); None per ometterla.
        max_strikes: strike massimi nel profilo (limite box di TradingView).
        range_pct: finestra attorno allo spot per il profilo (0.15 = ±15%).
        history_limit: numero massimo di punti storici da incorporare.
        ibit_ratio: ratio IBIT/BTC, suggerito nel titolo del fattore di scala.
        title: titolo dell'indicatore su TradingView.
        api_hint: URL mostrato nell'header per rigenerare lo script.

    Returns:
        str: sorgente Pine Script v6 pronto da incollare nel Pine Editor.
    """
    spot = _num(snapshot.spot_price)
    profile = select_profile_strikes(snapshot.gex_by_strike, spot, max_strikes, range_pct)
    strikes = [_num(g.strike) for g in profile]
    hist = normalize_history(history, limit=history_limit)

    arr_strikes, chunks_k = _array_literal("STRIKES", "float", [_f(s, 2) for s in strikes])
    # GEX in milioni di USD: valori raw (1e8+) sono illeggibili nel sorgente.
    arr_call, chunks_c = _array_literal(
        "CALL_GEX", "float", [_f(_num(g.call_gex) / 1e6, 4) for g in profile]
    )
    arr_put, chunks_p = _array_literal(
        "PUT_GEX", "float", [_f(_num(g.put_gex) / 1e6, 4) for g in profile]
    )
    arr_ht, chunks_t = _array_literal("HIST_T", "int", [str(r[0]) for r in hist])
    arr_hf, chunks_hf = _array_literal("HIST_FLIP", "float", [_f(r[1], 2) for r in hist])
    arr_hp, chunks_hp = _array_literal("HIST_PW", "float", [_f(r[2], 2) for r in hist])
    arr_hc, chunks_hc = _array_literal("HIST_CW", "float", [_f(r[3], 2) for r in hist])

    all_chunks = chunks_k + chunks_c + chunks_p + chunks_t + chunks_hf + chunks_hp + chunks_hc
    chunk_block = ""
    if all_chunks:
        chunk_block = "\n// Chunk aggiuntivi degli array (limite argomenti di array.from)\nif barstate.isfirst\n" \
            + "\n".join(all_chunks) + "\n"

    generated = snapshot.timestamp or datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)

    ratio_hint = f"IBIT ≈ {ibit_ratio:.6f}" if ibit_ratio and ibit_ratio > 0 else "IBIT: usa il ratio IBIT/BTC"
    net_gex_m = _num(snapshot.total_net_gex) / 1e6

    return _TEMPLATE.format(
        title=_pine_str(title),
        generated_at=generated.strftime("%Y-%m-%d %H:%M UTC"),
        generated_at_str=_pine_str(generated.strftime("%Y-%m-%d %H:%M UTC")),
        generated_ms=int(generated.timestamp() * 1000),
        spot=_f(spot, 2),
        spot_hint=f"{spot:,.0f}",
        flip=_f(snapshot.gamma_flip_price, 2),
        put_wall=_f(snapshot.put_wall, 2),
        call_wall=_f(snapshot.call_wall, 2),
        max_pain=_f(snapshot.max_pain, 2),
        net_gex_m=_f(net_gex_m, 4),
        net_gex_hint=f"{net_gex_m:,.1f}",
        pcr=_f(snapshot.put_call_ratio, 4),
        gex_pctl=_f(gex_percentile, 2),
        regime=regime or "n/d",
        regime_str=_pine_str(regime or "n/d"),
        strike_step=_f(_strike_step(strikes, spot), 2),
        n_strikes=len(profile),
        n_history=len(hist),
        api_hint=api_hint,
        scale_title=_pine_str(f"Scala prezzi (BTC = 1, {ratio_hint})"),
        arr_strikes=arr_strikes,
        arr_call_gex=arr_call,
        arr_put_gex=arr_put,
        arr_hist_t=arr_ht,
        arr_hist_flip=arr_hf,
        arr_hist_pw=arr_hp,
        arr_hist_cw=arr_hc,
        chunk_block=chunk_block,
    )
