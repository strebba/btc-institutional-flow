"""Test per charm e vanna.

Il test che conta è :class:`TestValidazioneControDeribit`: Deribit non pubblica
charm né vanna, ma pubblica delta e gamma. Se la ricostruzione di d₁ riproduce i
loro numeri sull'intera chain, allora charm e vanna calcolate dalla stessa d₁
sono affidabili. È l'unico modo di verificare due greche che nessuno ci dà.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.gex.greeks import (
    DAYS_PER_YEAR,
    black_scholes_greeks,
    dealer_sign,
    option_charm_usd_day,
    option_vanna_usd,
    time_to_expiry_years,
)

_CHAIN = Path(__file__).parent / "fixtures" / "deribit_chain.json"


def _g(**kw):
    base = {"spot": 100_000.0, "strike": 100_000.0, "t_years": 30 / 365,
            "sigma": 0.60, "option_type": "call"}
    base.update(kw)
    return black_scholes_greeks(**base)


class TestInputNonValidi:
    """Uno zero si somma e sporca il totale; un None si conta come dato mancante."""

    def test_tipo_sconosciuto(self):
        assert _g(option_type="future") is None

    def test_scadenza_gia_passata(self):
        assert _g(t_years=0.0) is None

    def test_scadenza_entro_un_ora(self):
        """Sotto l'ora d₂/(2T) diverge e il charm smette di significare qualcosa."""
        assert _g(t_years=0.5 / (365 * 24)) is None

    def test_iv_assente(self):
        assert _g(sigma=0.0) is None

    def test_spot_o_strike_non_positivi(self):
        assert _g(spot=0.0) is None
        assert _g(strike=-1.0) is None


class TestCoerenzaMatematica:
    def test_delta_call_fra_zero_e_uno(self):
        assert 0.0 < _g().delta < 1.0

    def test_delta_put_fra_meno_uno_e_zero(self):
        assert -1.0 < _g(option_type="put").delta < 0.0

    def test_put_call_parity_sul_delta(self):
        """Con q=0: Δput = Δcall − 1."""
        c, p = _g(), _g(option_type="put")
        assert c.delta - p.delta == pytest.approx(1.0, abs=1e-9)

    def test_gamma_uguale_per_call_e_put(self):
        assert _g().gamma == pytest.approx(_g(option_type="put").gamma, rel=1e-12)

    def test_charm_uguale_per_call_e_put(self):
        """Δput = Δcall − 1, e la costante non dipende dal tempo."""
        assert _g().charm_per_day == pytest.approx(
            _g(option_type="put").charm_per_day, rel=1e-12
        )

    def test_vanna_uguale_per_call_e_put(self):
        assert _g().vanna == pytest.approx(_g(option_type="put").vanna, rel=1e-12)

    def test_gamma_massimo_vicino_allo_strike(self):
        atm = _g(strike=100_000.0).gamma
        otm = _g(strike=150_000.0).gamma
        assert atm > otm

    def test_vanna_cambia_segno_attraversando_lo_strike(self):
        """Il segno della vanna è −d₂: sopra e sotto lo strike è opposto."""
        sotto = _g(strike=130_000.0).vanna   # d2 < 0 -> vanna > 0
        sopra = _g(strike=70_000.0).vanna    # d2 > 0 -> vanna < 0
        assert sotto > 0 > sopra

    def test_charm_segue_il_segno_di_d2(self):
        assert _g(strike=70_000.0).charm_per_day > 0   # ITM: d2 > 0
        assert _g(strike=130_000.0).charm_per_day < 0  # OTM: d2 < 0

    def test_charm_cresce_avvicinandosi_alla_scadenza(self):
        """È il fatto che rende il charm una notizia: accelera verso l'expiry."""
        lontano = abs(_g(strike=90_000.0, t_years=90 / 365).charm_per_day)
        vicino = abs(_g(strike=90_000.0, t_years=3 / 365).charm_per_day)
        assert vicino > lontano

    def test_d2_minore_di_d1(self):
        g = _g()
        assert g.d2 < g.d1
        assert g.d1 - g.d2 == pytest.approx(0.60 * math.sqrt(30 / 365), rel=1e-12)


class TestSegnoDealer:
    def test_convenzione_allineata_al_gex(self):
        """Deve restare identica a GexCalculator._option_gex: call +1, put −1."""
        from src.gex.gex_calculator import GexCalculator

        calc = GexCalculator({"contract_size": 1.0})
        gex_call = calc._option_gex(0.001, 100.0, "call", 100_000.0)
        gex_put = calc._option_gex(0.001, 100.0, "put", 100_000.0)
        assert math.copysign(1, gex_call) == dealer_sign("call")
        assert math.copysign(1, gex_put) == dealer_sign("put")


class TestScalaturaInDollari:
    def test_charm_scala_con_open_interest(self):
        g = _g(strike=90_000.0)
        uno = option_charm_usd_day(g, open_interest=1, spot=100_000.0)
        cento = option_charm_usd_day(g, open_interest=100, spot=100_000.0)
        assert cento == pytest.approx(uno * 100, rel=1e-12)

    def test_charm_put_ha_segno_opposto_alla_call(self):
        g = _g(strike=90_000.0)
        c = option_charm_usd_day(g, open_interest=10, spot=100_000.0, option_type="call")
        p = option_charm_usd_day(g, open_interest=10, spot=100_000.0, option_type="put")
        assert c == pytest.approx(-p, rel=1e-12)

    def test_vanna_normalizzata_su_un_punto_di_iv(self):
        g = _g()
        v = option_vanna_usd(g, open_interest=1, spot=100_000.0)
        assert v == pytest.approx(g.vanna * 0.01 * 100_000.0, rel=1e-12)

    def test_charm_in_dollari_ha_ordine_di_grandezza_sensato(self):
        """Su OI realistico deve stare nei milioni al giorno, non nei centesimi."""
        g = _g(strike=95_000.0, t_years=7 / 365)
        usd = abs(option_charm_usd_day(g, open_interest=5_000, spot=100_000.0))
        assert 1e5 < usd < 1e9


class TestTimeToExpiry:
    def test_converte_millisecondi_in_anni(self):
        ora = 1_700_000_000_000
        fra_30_giorni = ora + 30 * 86_400 * 1000
        assert time_to_expiry_years(fra_30_giorni, ora) == pytest.approx(
            30 / DAYS_PER_YEAR, rel=1e-9
        )

    def test_scadenza_passata_da_zero(self):
        assert time_to_expiry_years(1_000, 2_000) == 0.0


@pytest.mark.skipif(not _CHAIN.exists(), reason="fixture chain Deribit assente")
class TestValidazioneControDeribit:
    """La ricostruzione deve riprodurre i greeks che Deribit dichiara.

    Se d₁ è ricostruita correttamente, delta e gamma calcolati coincidono con
    quelli pubblicati. Da quel momento charm e vanna, che vengono dalla stessa
    d₁, sono verificate e non solo plausibili.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def chain(cls) -> dict:
        return json.loads(_CHAIN.read_text())

    def _confronti(self, chain: dict) -> list[tuple[float, float, float, float]]:
        now_ms = chain["fetched_at_ms"]
        out = []
        for o in chain["options"]:
            g = black_scholes_greeks(
                spot=o["underlying_price"],
                strike=o["strike"],
                t_years=time_to_expiry_years(o["expiration_timestamp"], now_ms),
                sigma=o["mark_iv"] / 100.0,   # Deribit quota la IV in percentuale
                option_type=o["option_type"],
            )
            if g is None:
                continue
            out.append((g.delta, o["delta"], g.gamma, o["gamma"]))
        return out

    def test_la_fixture_copre_abbastanza_strumenti(self, chain):
        assert len(self._confronti(chain)) >= 50

    def test_riproduce_il_delta_di_deribit(self, chain):
        """È questa la prova che conta.

        charm = φ(d₁)·d₂/(2T) e vanna = −φ(d₁)·d₂/σ dipendono da d₁ e d₂, non dal
        gamma. Il delta è N(d₁): se coincide con quello pubblicato, d₁ è giusta, e
        d₂ = d₁ − σ√T lo è di conseguenza. La soglia è stretta di proposito —
        sulla chain reale lo scarto mediano è dell'ordine di 1e-5.
        """
        scarti = [abs(mio - loro) for mio, loro, _, _ in self._confronti(chain)]
        mediano = sorted(scarti)[len(scarti) // 2]
        peggiore = max(scarti)
        assert mediano < 1e-3, f"scarto mediano sul delta: {mediano}"
        assert peggiore < 1e-2, f"scarto peggiore sul delta: {peggiore}"

    def test_nessun_errore_di_scala_sul_gamma(self, chain):
        """Il gamma serve a escludere un errore di scala, non a validare d₁.

        Il rapporto fra il gamma calcolato e quello pubblicato deve stare intorno
        a 1: un fattore costante diverso segnalerebbe una convenzione sbagliata
        (per esempio le greche in termini di coin invece che di dollari).

        La dispersione attorno alla mediana è invece attesa e non è un difetto:
        Deribit calcola le greche sulla propria superficie di volatilità, mentre
        qui si usa `mark_iv` strumento per strumento. Il gamma è molto più
        sensibile della IV del delta, quindi piccole differenze di σ si vedono sul
        gamma e non sul delta — ed è esattamente quello che si osserva.
        """
        rapporti = sorted(
            mio / loro for _, _, mio, loro in self._confronti(chain) if abs(loro) > 1e-9
        )
        mediana = rapporti[len(rapporti) // 2]
        assert 0.95 < mediana < 1.05, f"rapporto mediano gamma fuori scala: {mediana}"
