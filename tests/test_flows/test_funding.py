"""Test sulla convenzione del funding rate, che e' la trappola di questo dominio.

Il funding si annualizza moltiplicando per tre finestre da 8 ore per 365 giorni.
Il fattore in piu' che rovina tutto e' il ×100: si applica solo se la fonte
restituisce una **frazione**, e nessuna delle due fonti che usiamo lo fa.

Misurato sulle API vere, stessa metrica OI-weighted nello stesso istante:

    CoinGecko  0,007499   (percentuale, documentata)
    CoinGlass  0,005477
    rapporto   0,73×

Se le convenzioni fossero diverse il rapporto sarebbe ~100×, non 0,73. Sono la
stessa unita': percentuale per 8 ore. Il ×100 che il codice applicava al ramo
CoinGlass produceva 599,73% invece di 6,00%, e lo scorer del funding leggeva
"estremo: flush imminente" (0,35) dove il mercato dava "caldo ma compresso dal
pin gamma" (0,60).
"""
from __future__ import annotations

import pytest

from src.flows.funding import annualize_funding_pct


class TestConvenzione:
    def test_il_valore_reale_di_coinglass_fa_sei_percento(self):
        assert annualize_funding_pct(0.005477) == pytest.approx(6.00, abs=0.01)

    def test_il_valore_reale_di_coingecko_fa_otto_percento(self):
        assert annualize_funding_pct(0.007499) == pytest.approx(8.21, abs=0.01)

    def test_le_due_fonti_restano_confrontabili(self):
        """Stessa metrica, stesso istante: devono restare vicine dopo la conversione.

        E' questo il test che avrebbe fatto cadere subito il ×100: con quello, i
        due numeri distavano due ordini di grandezza pur misurando la stessa cosa.
        """
        glass = annualize_funding_pct(0.005477)
        gecko = annualize_funding_pct(0.007499)
        assert 0.3 < glass / gecko < 3.0

    def test_non_applica_il_fattore_cento(self):
        assert annualize_funding_pct(0.005477) < 100

    def test_lo_zero_virgola_zero_uno_percento_fa_dieci_e_mezzo(self):
        assert annualize_funding_pct(0.01) == pytest.approx(10.95, abs=0.01)

    def test_negativo_resta_negativo(self):
        assert annualize_funding_pct(-0.005) < 0

    def test_zero_resta_zero(self):
        assert annualize_funding_pct(0.0) == 0.0


class TestNonRegressione:
    """Il funding annualizzato di BTC vive in una banda stretta.

    Fuori da questa banda non c'e' un mercato estremo: c'e' una conversione
    sbagliata. Il pilastro macro non deve poter pubblicare 599% senza che
    qualcosa si accorga.
    """

    @pytest.mark.parametrize("rate_8h_pct", [0.001, 0.005477, 0.007499, 0.01, 0.05])
    def test_i_valori_plausibili_restano_in_banda(self, rate_8h_pct):
        assert -200 < annualize_funding_pct(rate_8h_pct) < 200

    def test_la_vecchia_formula_sarebbe_stata_fuori_banda(self):
        """Documenta il bug: il ×100 portava un mercato calmo a 599%."""
        vecchia = 0.005477 * 3 * 365 * 100
        assert vecchia > 200, "se questo non e' fuori banda il test non prova nulla"
        assert annualize_funding_pct(0.005477) < 200
