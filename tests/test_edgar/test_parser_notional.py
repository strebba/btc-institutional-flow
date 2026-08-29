"""Test per l'estrazione del nozionale dai 424B2.

I frammenti HTML qui sotto riproducono le tre forme trovate scaricando i filing
reali dalla SEC, fixture nel repo invece che richieste di rete:

- JPMorgan spezza le parole in <span> per lo styling. Con
  ``get_text(separator=" ")`` il testo esce come "n otional fina ncing co st" e
  nessuna regex puo' matchare: e' la causa del 5% di copertura su 111 barriere.
- Goldman scrive "aggregate face amount" e "$X in the aggregate", mai
  "principal amount".
- Citigroup scrive "Aggregate stated principal amount".

La tabella commissioni ("Total $X") e' l'ancora comune a tutti gli emittenti.
"""
from __future__ import annotations

import pytest

from src.edgar.parser import (
    ProspectusParser,
    _extract_barrier_levels,
    _parse_notional,
)


@pytest.fixture
def parser() -> ProspectusParser:
    return ProspectusParser()


# ─── Frammenti reali ──────────────────────────────────────────────────────────

# JPMorgan: ogni parola spezzata da span, come nei filing veri
HTML_JPM_SHREDDED = """
<html><body>
<p><span>Pri</span><span>cing su</span><span>pplement</span> <span>dated Ju</span><span>ne 3, 2024</span></p>
<p><span>JPMorgan Chase Financial Company LLC</span></p>
<p><span>Struct</span><span>ured Inv</span><span>estments</span></p>
<table>
  <tr><td>Total</td><td>$1,135,000</td><td>$48,237</td></tr>
</table>
<p><span>Inve</span><span>stors sho</span><span>uld be will</span><span>ing to acc</span><span>ept the risk</span></p>
</body></html>
"""

HTML_GOLDMAN = """
<html><body>
<p>Goldman Sachs Group, Inc.</p>
<p>We will sell $11,803,000 in the aggregate on the original issue date; the
aggregate face amount may be increased if the company, at its sole option,
decides to sell an additional amount.</p>
<p>you will receive a coupon for each $1,000 face amount of your notes equal to $10</p>
</body></html>
"""

HTML_CITI = """
<html><body>
<p>Citigroup Global Markets Holdings Inc.</p>
<p>Aggregate stated principal amount: $11,803,000</p>
<p>Stated principal amount: $1,000 per security</p>
</body></html>
"""


class TestEstrazioneTesto:
    """Il testo deve uscire leggibile anche quando i tag spezzano le parole."""

    def test_ricompone_le_parole_spezzate(self, parser):
        testo = parser._extract_relevant_text(HTML_JPM_SHREDDED)
        assert "Pricing supplement" in testo
        assert "Structured Investments" in testo
        assert "willing to accept" in testo

    def test_non_incolla_celle_di_tabella_adiacenti(self, parser):
        """Il separatore fra blocchi va mantenuto: 'Total' e '$1,135,000' sono
        due celle, non una parola sola."""
        testo = parser._extract_relevant_text(HTML_JPM_SHREDDED)
        assert "Total$1,135,000" not in testo
        assert "Total $1,135,000" in testo

    def test_non_incolla_paragrafi_adiacenti(self, parser):
        testo = parser._extract_relevant_text(
            "<html><body><p>primo</p><p>secondo</p></body></html>"
        )
        assert "primosecondo" not in testo


class TestParseNotional:
    def test_tabella_commissioni(self):
        """L'ancora piu' affidabile: struttura standard del 424B2."""
        assert _parse_notional("Total $1,135,000 $48,237") == 1_135_000.0

    def test_tabella_commissioni_coi_due_punti(self):
        assert _parse_notional("Total: $11,803,000") == 11_803_000.0

    def test_aggregate_stated_principal_amount(self):
        assert _parse_notional("Aggregate stated principal amount: $11,803,000") == 11_803_000.0

    def test_aggregate_face_amount(self):
        assert _parse_notional("The aggregate face amount is $5,000,000 for this offering") == 5_000_000.0

    def test_in_the_aggregate(self):
        assert _parse_notional("We will sell $593,000 in the aggregate on the issue date") == 593_000.0

    def test_ignora_la_denominazione_per_nota(self):
        """La trappola principale: '$1,000 face amount' e' il taglio, non il totale."""
        assert _parse_notional("a coupon for each $1,000 face amount of your notes") is None

    def test_ignora_importi_sotto_soglia(self):
        assert _parse_notional("Total: $500") is None

    def test_senza_nulla_torna_none(self):
        assert _parse_notional("nessun importo qui") is None


class TestEndToEndPerEmittente:
    """Il numero deve arrivare in fondo, non solo passare la regex."""

    def test_jpmorgan_dal_documento_spezzato(self, parser):
        testo = parser._extract_relevant_text(HTML_JPM_SHREDDED)
        assert _parse_notional(testo) == 1_135_000.0

    def test_goldman(self, parser):
        testo = parser._extract_relevant_text(HTML_GOLDMAN)
        assert _parse_notional(testo) == 11_803_000.0

    def test_citigroup(self, parser):
        testo = parser._extract_relevant_text(HTML_CITI)
        assert _parse_notional(testo) == 11_803_000.0

    def test_morgan_stanley_non_regredisce(self, parser):
        """Gruppo di controllo: la forma gia' coperta deve continuare a funzionare."""
        html = (
            "<html><body><p>Morgan Stanley Finance LLC</p>"
            "<p>$3,250,000 aggregate principal amount of notes</p></body></html>"
        )
        assert _parse_notional(parser._extract_relevant_text(html)) == 3_250_000.0


class TestLivelliImplausibili:
    """Le percentuali di prosa non sono barriere.

    Con il testo risanato emergono formule che prima erano illeggibili, e alcune
    contengono percentuali che il parser scambiava per livelli: "you will lose 1%
    of the principal for every 1% that the Final Value declines" produceva un
    knock_in all'1%, e "50% per annum deduction" uno al 50% su una nota che nel
    documento non nomina mai la parola "barrier".
    """

    def test_scarta_il_rapporto_di_perdita(self, parser):
        html = (
            "<html><body><p>JPMorgan Chase Financial Company LLC</p>"
            "<p>If the Final Value of any Index is less than its Barrier Amount, you "
            "will lose 1% of the principal amount of your notes for every 1% that the "
            "Final Value declines below the Initial Value.</p></body></html>"
        )
        livelli = _extract_barrier_levels(parser._extract_relevant_text(html), 50.0)
        assert all(b.level_pct >= 10.0 for b in livelli)

    def test_tiene_i_knock_in_profondi_legittimi(self, parser):
        """15% e 25% sono barriere vere sulle note lunghe di Goldman: non vanno tagliate."""
        html = (
            "<html><body><p>Goldman Sachs</p>"
            "<p>The knock-in barrier level is 15% of the Initial Value.</p></body></html>"
        )
        livelli = _extract_barrier_levels(parser._extract_relevant_text(html), 50.0)
        assert any(abs(b.level_pct - 15.0) < 1e-9 for b in livelli)

    def test_tiene_i_buffer_tipici(self, parser):
        html = (
            "<html><body><p>Morgan Stanley</p>"
            "<p>The buffer level is 80% of the Initial Value.</p></body></html>"
        )
        livelli = _extract_barrier_levels(parser._extract_relevant_text(html), 50.0)
        assert any(abs(b.level_pct - 80.0) < 1e-9 for b in livelli)
