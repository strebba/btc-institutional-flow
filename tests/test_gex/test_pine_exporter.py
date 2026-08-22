"""Test per pine_exporter.py — generazione dell'indicatore TradingView (Pine v6)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.gex.models import GexByStrike, GexSnapshot
from src.gex.pine_exporter import (
    MAX_PROFILE_STRIKES,
    build_pine_script,
    normalize_history,
    select_profile_strikes,
)

_TS = datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)


def _strikes(spot: float = 100_000.0, n: int = 30, step: float = 1_000.0) -> list[GexByStrike]:
    """Profilo sintetico centrato sullo spot: call GEX positivo, put GEX negativo."""
    start = spot - (n // 2) * step
    out = []
    for i in range(n):
        strike = start + i * step
        call = (i + 1) * 1e6
        put = -(n - i) * 1e6
        out.append(
            GexByStrike(strike=strike, call_gex=call, put_gex=put, net_gex=call + put,
                        call_oi=10.0, put_oi=12.0)
        )
    return out


def _snapshot(**overrides) -> GexSnapshot:
    defaults = dict(
        timestamp=_TS,
        spot_price=100_000.0,
        total_net_gex=4.5e8,
        gamma_flip_price=98_500.0,
        put_wall=92_000.0,
        call_wall=108_000.0,
        max_pain=99_000.0,
        gex_by_strike=_strikes(),
        total_call_oi=1_000.0,
        total_put_oi=1_200.0,
    )
    defaults.update(overrides)
    return GexSnapshot(**defaults)


class TestSelectProfileStrikes:
    def test_filtra_finestra_e_ordina_per_strike(self):
        sel = select_profile_strikes(_strikes(), 100_000.0, max_strikes=50, range_pct=0.05)
        assert sel, "la finestra ±5% deve contenere strike"
        assert all(abs(g.strike - 100_000.0) / 100_000.0 <= 0.05 for g in sel)
        assert [g.strike for g in sel] == sorted(g.strike for g in sel)

    def test_rispetta_max_strikes_tenendo_i_gex_maggiori(self):
        sel = select_profile_strikes(_strikes(n=60), 100_000.0, max_strikes=8, range_pct=1.0)
        assert len(sel) == 8
        scartati = {g.strike for g in _strikes(n=60)} - {g.strike for g in sel}
        peggiore_tenuto = min(max(abs(g.call_gex), abs(g.put_gex)) for g in sel)
        migliore_scartato = max(
            max(abs(g.call_gex), abs(g.put_gex)) for g in _strikes(n=60) if g.strike in scartati
        )
        assert peggiore_tenuto >= migliore_scartato

    def test_finestra_vuota_ripiega_sugli_strike_piu_vicini(self):
        # spot lontanissimo dagli strike → nessuno entra nella finestra ±15%
        sel = select_profile_strikes(_strikes(), 1_000_000.0, max_strikes=5, range_pct=0.15)
        assert len(sel) == 5

    def test_input_degeneri(self):
        assert select_profile_strikes([], 100_000.0) == []
        assert select_profile_strikes(_strikes(), 0.0) == []
        assert select_profile_strikes(_strikes(), 100_000.0, max_strikes=0) == []


class TestNormalizeHistory:
    def test_dataframe_da_get_walls_series(self):
        idx = pd.to_datetime(["2026-08-01", "2026-08-02"])
        df = pd.DataFrame(
            {"put_wall": [90_000.0, 91_000.0],
             "call_wall": [110_000.0, 111_000.0],
             "gamma_flip_price": [99_000.0, 99_500.0]},
            index=idx,
        )
        out = normalize_history(df)
        assert len(out) == 2
        assert out[0][1] == 99_000.0 and out[0][2] == 90_000.0 and out[0][3] == 110_000.0
        assert out[0][0] < out[1][0]

    def test_lista_di_snapshot_e_di_mapping(self):
        snaps = [_snapshot(timestamp=_TS + timedelta(days=i)) for i in range(3)]
        assert len(normalize_history(snaps)) == 3
        mappings = [{"timestamp": "2026-08-01T00:00:00+00:00", "gamma_flip_price": 99_000.0,
                     "put_wall": None, "call_wall": None}]
        assert normalize_history(mappings) == [(_ms("2026-08-01T00:00:00+00:00"), 99_000.0, 0.0, 0.0)]

    def test_scarta_righe_senza_timestamp_o_senza_livelli(self):
        rows = [
            {"timestamp": None, "gamma_flip_price": 99_000.0},
            {"timestamp": "non-una-data", "gamma_flip_price": 99_000.0},
            {"timestamp": "2026-08-01", "gamma_flip_price": None, "put_wall": None, "call_wall": None},
            {"timestamp": "2026-08-02", "gamma_flip_price": 99_000.0},
        ]
        out = normalize_history(rows)
        assert len(out) == 1 and out[0][1] == 99_000.0

    def test_ordina_e_applica_il_limite_tenendo_i_recenti(self):
        rows = [{"timestamp": _TS + timedelta(days=i), "gamma_flip_price": float(i + 1)}
                for i in reversed(range(10))]
        out = normalize_history(rows, limit=3)
        assert [r[1] for r in out] == [8.0, 9.0, 10.0]

    def test_none_e_vuoto(self):
        assert normalize_history(None) == []
        assert normalize_history([]) == []


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


class TestBuildPineScript:
    def test_header_e_dichiarazione_indicator(self):
        src = build_pine_script(_snapshot())
        assert src.startswith("//@version=6")
        assert 'indicator("BTC GEX — Walls & Gamma Flip"' in src
        assert "overlay = true" in src

    def test_nessun_placeholder_non_risolto(self):
        src = build_pine_script(_snapshot(), regime="positive_gamma", gex_percentile=70.0)
        assert not re.search(r"\{[a-z_]+\}", src), "template con placeholder non sostituiti"

    def test_livelli_incorporati(self):
        src = build_pine_script(_snapshot(), regime="negative_gamma", gex_percentile=12.5)
        assert "FLIP        = 98500.0" in src
        assert "PUT_WALL    = 92000.0" in src
        assert "CALL_WALL   = 108000.0" in src
        assert "MAX_PAIN    = 99000.0" in src
        assert "NET_GEX_M   = 450.0" in src          # 4.5e8 USD → M$
        assert 'REGIME      = "negative_gamma"' in src
        assert "GEX_PCTL    = 12.5" in src

    def test_livelli_mancanti_diventano_zero_neutralizzato_da_f_lvl(self):
        src = build_pine_script(
            _snapshot(gamma_flip_price=None, put_wall=None, call_wall=None, max_pain=None)
        )
        assert "FLIP        = 0.0" in src
        # f_lvl mappa i valori <= 0 su na → nessuna linea disegnata
        assert "f_lvl(v) => v > 0 ? v * scaleF : na" in src

    def test_profilo_call_put_per_strike(self):
        src = build_pine_script(_snapshot(), max_strikes=6, range_pct=1.0)
        strikes = _pine_array(src, "STRIKES")
        calls = _pine_array(src, "CALL_GEX")
        puts = _pine_array(src, "PUT_GEX")
        assert len(strikes) == len(calls) == len(puts) == 6
        assert all(float(v) >= 0 for v in calls), "call GEX esportato in M$ non negativi"
        assert all(float(v) <= 0 for v in puts), "put GEX esportato in M$ non positivi"

    def test_profilo_vuoto_usa_array_new(self):
        src = build_pine_script(_snapshot(gex_by_strike=[]))
        assert "var array<float> STRIKES = array.new<float>()" in src
        assert "if barstate.isfirst" not in src

    def test_storico_incorporato_come_array(self):
        rows = [{"timestamp": _TS + timedelta(days=i), "gamma_flip_price": 99_000.0 + i,
                 "put_wall": 92_000.0, "call_wall": 108_000.0} for i in range(5)]
        src = build_pine_script(_snapshot(), history=rows)
        assert len(_pine_array(src, "HIST_T")) == 5
        assert len(_pine_array(src, "HIST_FLIP")) == 5
        assert "Storico: 5 punti" in src

    def test_storico_lungo_spezzato_in_chunk_concatenati(self):
        rows = [{"timestamp": _TS + timedelta(days=i), "gamma_flip_price": 99_000.0 + i}
                for i in range(150)]
        src = build_pine_script(_snapshot(), history=rows)
        assert "if barstate.isfirst" in src
        assert "HIST_T := array.concat(HIST_T, array.from(" in src
        # 150 punti = 60 nella dichiarazione + 2 chunk (60 + 30)
        assert src.count("HIST_FLIP := array.concat(") == 2

    def test_history_limit_tronca_i_punti(self):
        rows = [{"timestamp": _TS + timedelta(days=i), "gamma_flip_price": 99_000.0 + i}
                for i in range(50)]
        src = build_pine_script(_snapshot(), history=rows, history_limit=10)
        assert len(_pine_array(src, "HIST_T")) == 10

    def test_timestamp_naive_trattato_come_utc(self):
        src = build_pine_script(_snapshot(timestamp=datetime(2026, 8, 22, 10, 0)))
        assert f"GEN_MS      = {int(_TS.timestamp() * 1000)}" in src

    def test_titolo_con_virgolette_viene_escapato(self):
        src = build_pine_script(_snapshot(), title='GEX "live"')
        assert 'indicator("GEX \\"live\\""' in src

    def test_ibit_ratio_suggerito_nel_titolo_dell_input_scala(self):
        src = build_pine_script(_snapshot(), ibit_ratio=0.000567)
        assert "IBIT ≈ 0.000567" in src
        assert "IBIT: usa il ratio IBIT/BTC" in build_pine_script(_snapshot(), ibit_ratio=None)

    def test_alert_e_pannello_presenti(self):
        src = build_pine_script(_snapshot())
        for atteso in ('alertcondition(crossUp', 'alertcondition(crossDn',
                       'alertcondition(hitCw', 'alertcondition(hitPw', 'table.new('):
            assert atteso in src

    @pytest.mark.parametrize("max_strikes", [1, 5, MAX_PROFILE_STRIKES])
    def test_box_entro_il_limite_di_tradingview(self, max_strikes):
        # 2 box per strike (call + put) devono stare nei 500 box di TradingView
        src = build_pine_script(_snapshot(), max_strikes=max_strikes, range_pct=1.0)
        assert len(_pine_array(src, "STRIKES")) * 2 <= 500
        assert "max_boxes_count = 500" in src

    def test_drawing_entro_i_limiti_di_tradingview(self):
        """x dei drawing: mai negativa a sinistra, mai oltre 500 barre nel futuro."""
        src = build_pine_script(_snapshot())
        assert "line.new(math.max(0, bar_index - levelLen)" in src
        maxvals = {
            nome: int(re.search(rf"{nome}\s*= input\.int\(.*?maxval = (\d+)", src).group(1))
            for nome in ("rightPad", "profW", "profGap")
        }
        assert sum(maxvals.values()) <= 450, f"offset massimi oltre il limite di 500 barre: {maxvals}"

    def test_bool_na_gestito_esplicitamente(self):
        """Pine v6 usa logica a tre valori: nearCw[1] è na sulla prima barra."""
        src = build_pine_script(_snapshot())
        assert "prevCw  = na(nearCw[1]) ? false : nearCw[1]" in src
        assert "hitCw   = nearCw and not prevCw" in src

    def test_parentesi_bilanciate_su_ogni_riga(self):
        src = build_pine_script(_snapshot(), history=[
            {"timestamp": _TS, "gamma_flip_price": 99_000.0}
        ])
        for i, line in enumerate(src.splitlines(), start=1):
            code = line.split("//")[0] if not line.lstrip().startswith("//") else ""
            if '"' in code:      # le stringhe possono contenere parentesi non bilanciate
                continue
            assert code.count("(") == code.count(")"), f"parentesi sbilanciate a riga {i}: {line}"


def _pine_array(src: str, name: str) -> list[str]:
    """Estrae i valori della dichiarazione `var array<...> NAME = array.from(...)`."""
    match = re.search(rf"var array<\w+> {name} = array\.from\((.*?)\)\n", src)
    if not match:
        assert f"var array<float> {name} = array.new<float>()" in src or \
               f"var array<int> {name} = array.new<int>()" in src
        return []
    return [v.strip() for v in match.group(1).split(",")]
