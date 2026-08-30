"""Test per la formattazione numerica italiana del Desk Note."""
from __future__ import annotations

from src.report.formatting import MINUS, count, percent, price, ratio, usd_millions


class TestPrice:
    def test_separatore_migliaia_col_punto(self):
        assert price(77703.9) == "77.704"

    def test_arrotonda_all_intero(self):
        assert price(77_000.4) == "77.000"

    def test_none_diventa_nd(self):
        assert price(None) == "n/d"


class TestUsdMillions:
    def test_milioni_con_segno_e_virgola(self):
        assert usd_millions(171_930_539.8) == "+171,9M"

    def test_negativo_usa_il_meno_tipografico(self):
        out = usd_millions(-37_900_000)
        assert out == f"{MINUS}37,9M"
        assert "-" not in out  # mai il trattino ASCII

    def test_sopra_il_miliardo_passa_a_B(self):
        assert usd_millions(63_364_700_000) == "+63,4B"

    def test_decimale_a_zero_viene_omesso(self):
        """'+445M' si legge a colpo d'occhio, '+445,0M' no."""
        assert usd_millions(445_000_000) == "+445M"

    def test_segno_disattivabile(self):
        assert usd_millions(27_600_000_000, force_sign=False) == "27,6B"

    def test_none_diventa_nd(self):
        assert usd_millions(None) == "n/d"


class TestPercent:
    def test_negativo_con_una_cifra(self):
        assert percent(-1.06) == f"{MINUS}1,1%"

    def test_positivo_ha_segno_esplicito(self):
        assert percent(5.53) == "+5,5%"

    def test_zero_decimali(self):
        assert percent(9.01, decimals=0, force_sign=False) == "9%"


class TestCount:
    def test_migliaia(self):
        assert count(14999) == "14.999"

    def test_unita(self):
        assert count(4) == "4"


class TestRatio:
    def test_due_decimali_con_virgola(self):
        assert ratio(0.4535170782) == "0,45"
