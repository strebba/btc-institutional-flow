"""Test per l'assemblaggio del Desk Note."""
from __future__ import annotations

from datetime import datetime, timezone

from src.report.facts import Fact
from src.report.narrative import (
    MAX_PER_TOPIC,
    N_CARDS,
    build_desk_note,
    select_facts,
)

_TS = datetime(2026, 8, 29, 14, 37, tzinfo=timezone.utc)


def _fact(key: str, topic: str, salience: float) -> Fact:
    return Fact(
        key=key, topic=topic, salience=salience, headline=f"titolo {key}",
        body=["corpo"], hero_value="1M", hero_caption="etichetta",
    )


class TestSelectFacts:
    def test_ordina_per_salienza(self):
        facts = [_fact("a", "gex", 0.3), _fact("b", "flows", 0.9), _fact("c", "signal", 0.6)]
        assert [f.key for f in select_facts(facts)] == ["b", "c", "a"]

    def test_limita_le_card_per_famiglia(self):
        """Con varietà a sufficienza il cap per topic tiene.

        Cinque fatti di GEX molto salienti non devono monopolizzare l'edizione
        se ci sono altre famiglie disponibili a riempire i posti.
        """
        facts = [_fact(f"g{i}", "gex", 0.9 - i * 0.01) for i in range(5)]
        facts += [
            _fact("f1", "flows", 0.5), _fact("f2", "flows", 0.45),
            _fact("s1", "signal", 0.4),
        ]
        scelti = select_facts(facts, limit=5)
        assert sum(1 for f in scelti if f.topic == "gex") == MAX_PER_TOPIC
        assert len(scelti) == 5

    def test_il_cap_cede_solo_quando_manca_varieta(self):
        """Meglio una card ripetitiva che una card vuota: il backfill può superare il cap.

        È il rovescio del test precedente — con una sola famiglia disponibile
        l'edizione si riempie comunque invece di uscire dimezzata.
        """
        facts = [_fact(f"g{i}", "gex", 0.9 - i * 0.01) for i in range(5)]
        scelti = select_facts(facts, limit=4)
        assert len(scelti) == 4
        assert sum(1 for f in scelti if f.topic == "gex") > MAX_PER_TOPIC

    def test_scarta_i_fatti_sotto_la_soglia(self):
        facts = [_fact("alto", "gex", 0.8), _fact("basso", "flows", 0.05)]
        assert [f.key for f in select_facts(facts)] == ["alto"]

    def test_lista_vuota_non_esplode(self):
        assert select_facts([]) == []


class TestBuildDeskNote:
    def test_produce_sei_card_con_dati_completi(
        self, gex_payload, barriers_payload, flows_payload, signals_payload
    ):
        note = build_desk_note(
            gex=gex_payload, barriers=barriers_payload, flows=flows_payload,
            signals=signals_payload, generated_at=_TS,
        )
        assert len(note.cards) == N_CARDS
        assert note.cards[0].kind == "cover"
        assert all(c.kind == "fact" for c in note.cards[1:])

    def test_la_copertina_non_ripete_il_titolo_della_seconda_card(
        self, gex_payload, barriers_payload, flows_payload, signals_payload
    ):
        note = build_desk_note(
            gex=gex_payload, barriers=barriers_payload, flows=flows_payload,
            signals=signals_payload, generated_at=_TS,
        )
        assert note.cards[0].headline != note.cards[1].headline

    def test_la_copertina_porta_i_tre_takeaway(
        self, gex_payload, barriers_payload, flows_payload, signals_payload
    ):
        note = build_desk_note(
            gex=gex_payload, barriers=barriers_payload, flows=flows_payload,
            signals=signals_payload, generated_at=_TS,
        )
        assert len(note.cards[0].takeaways) == 3
        assert all(t for t in note.cards[0].takeaways)

    def test_la_tape_riporta_spot_gex_e_flip(self, gex_payload):
        note = build_desk_note(gex=gex_payload, generated_at=_TS)
        assert "BTC 77.704" in note.tape
        assert "GEX +171,9M" in note.tape
        assert "FLIP 79.783" in note.tape

    def test_indice_di_pagina_coerente_col_totale_reale(self, gex_payload):
        """Con pochi dati escono meno card, ma '2 / 3' non deve diventare '2 / 6'."""
        note = build_desk_note(gex=gex_payload, generated_at=_TS)
        assert len(note.cards) < N_CARDS
        assert all(c.total == len(note.cards) for c in note.cards)
        assert [c.index for c in note.cards] == list(range(1, len(note.cards) + 1))

    def test_senza_nessun_dato_non_produce_card(self):
        note = build_desk_note(generated_at=_TS)
        assert note.cards == []
        assert note.facts_considered == 0

    def test_segnala_il_pilastro_macro_scoperto(self, signals_payload, gex_payload):
        """Il warning è ciò che tiene ferma la card 06 quando CoinGlass è giù."""
        note = build_desk_note(gex=gex_payload, signals=signals_payload, generated_at=_TS)
        assert any("macro" in w for w in note.warnings)
        assert any("funding" in w for w in note.warnings)

    def test_nessun_warning_se_i_pilastri_sono_coperti(self, gex_payload):
        pieni = {
            "score": 60.0, "signal": "CAUTION",
            "pillars": [{"name": "macro", "score": 60.0,
                         "components": {"funding": 0.5, "oi_change": 0.5,
                                        "long_short": 0.5, "put_call": 0.5,
                                        "liquidations": 0.5}}],
        }
        note = build_desk_note(gex=gex_payload, signals=pieni, generated_at=_TS)
        assert note.warnings == []

    def test_to_dict_e_serializzabile(
        self, gex_payload, barriers_payload, flows_payload, signals_payload
    ):
        import json

        note = build_desk_note(
            gex=gex_payload, barriers=barriers_payload, flows=flows_payload,
            signals=signals_payload, generated_at=_TS,
        )
        d = note.to_dict()
        assert json.loads(json.dumps(d))["cards"][0]["kind"] == "cover"
        assert d["generated_at"] == _TS.isoformat()


class TestWarningMacroAzionabile:
    """Un avviso serve solo se dice cosa fare.

    "pilastro macro senza dati" manda a indagare; "manca la chiave CoinGlass" si
    risolve in due minuti. Quando la causa e' nota va nominata, non lasciata
    dedurre dall'elenco dei sintomi.
    """

    def test_chiave_mancante_nomina_la_variabile(self, gex_payload, signals_payload):
        note = build_desk_note(
            gex=gex_payload, signals=signals_payload,
            macro={"source_status": "no_api_key"}, generated_at=_TS,
        )
        avviso = " ".join(note.warnings)
        assert "COINGLASS_API_KEY" in avviso
        assert "App Spec" in avviso

    def test_api_giu_e_diverso_da_chiave_mancante(self, gex_payload, signals_payload):
        note = build_desk_note(
            gex=gex_payload, signals=signals_payload,
            macro={"source_status": "unavailable"}, generated_at=_TS,
        )
        avviso = " ".join(note.warnings)
        assert "non risponde" in avviso
        assert "COINGLASS_API_KEY" not in avviso

    def test_non_ripete_i_sintomi_dopo_aver_nominato_la_causa(
        self, gex_payload, signals_payload
    ):
        """Una riga sulla causa, non due: quella e poi l'elenco dei fattori vuoti."""
        note = build_desk_note(
            gex=gex_payload, signals=signals_payload,
            macro={"source_status": "no_api_key"}, generated_at=_TS,
        )
        assert len([w for w in note.warnings if "macro" in w]) == 1

    def test_senza_payload_macro_resta_il_comportamento_di_prima(
        self, gex_payload, signals_payload
    ):
        note = build_desk_note(gex=gex_payload, signals=signals_payload, generated_at=_TS)
        assert any("macro" in w and "funding" in w for w in note.warnings)

    def test_macro_ok_non_produce_avvisi_sulla_fonte(self, gex_payload):
        pieni = {
            "score": 60.0, "signal": "CAUTION",
            "pillars": [{"name": "macro", "score": 60.0,
                         "components": {"funding": 0.5, "oi_change": 0.5,
                                        "long_short": 0.5, "put_call": 0.5,
                                        "liquidations": 0.5}}],
        }
        note = build_desk_note(
            gex=gex_payload, signals=pieni, macro={"source_status": "ok"}, generated_at=_TS
        )
        assert note.warnings == []
