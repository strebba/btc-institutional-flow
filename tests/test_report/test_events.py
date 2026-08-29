"""Test per il rilevamento eventi del Desk Note."""
from __future__ import annotations

import pytest

from src.report.events import (
    PUBLISH_THRESHOLD,
    ReportStateDB,
    detect_events,
    should_publish,
    snapshot_state,
)


def _stato(**kw) -> dict:
    base = {
        "spot": 77_704.0,
        "regime": "positive_gamma",
        "gamma_flip": 79_783.0,
        "put_wall": 75_000.0,
        "call_wall": 82_000.0,
        "signal_score": 56.5,
        "flow_3d_usd_m": 445.0,
        "barrier_levels": [76_897.0, 79_348.0],
    }
    base.update(kw)
    return base


class TestSnapshotState:
    def test_estrae_le_grandezze_che_servono(self, gex_payload, barriers_payload, signals_payload):
        s = snapshot_state(gex=gex_payload, barriers=barriers_payload, signals=signals_payload)
        assert s["spot"] == 77_703.9
        assert s["regime"] == "positive_gamma"
        assert s["gamma_flip"] == 79_783.46
        assert s["signal_score"] == 49.4
        assert s["flow_3d_usd_m"] == 445.0

    def test_i_livelli_barriera_sono_ordinati(self, barriers_payload):
        s = snapshot_state(barriers=barriers_payload)
        assert s["barrier_levels"] == sorted(s["barrier_levels"])

    def test_scarta_i_livelli_non_numerici(self, barriers_payload):
        sporco = {
            **barriers_payload,
            "barriers": barriers_payload["barriers"] + [{"level_price_btc": None}],
        }
        assert None not in snapshot_state(barriers=sporco)["barrier_levels"]

    def test_payload_vuoti_non_esplodono(self):
        s = snapshot_state()
        assert s["spot"] is None and s["barrier_levels"] == []


class TestPrimoGiro:
    def test_senza_fotografia_precedente_nessun_evento(self):
        """La prima esecuzione stabilisce la linea di base, non dichiara eventi."""
        assert detect_events(_stato(), None) == []
        assert detect_events(_stato(), {}) == []


class TestRegime:
    def test_ribaltamento_di_regime_e_l_evento_piu_grave(self):
        ev = detect_events(_stato(regime="negative_gamma"), _stato())
        assert ev[0].key == "gamma_regime_flip"
        assert ev[0].severity >= 0.9
        assert "amplificare" in ev[0].detail

    def test_regime_invariato_non_produce_evento(self):
        assert not [e for e in detect_events(_stato(), _stato()) if e.key == "gamma_regime_flip"]


class TestAttraversamenti:
    def test_spot_che_scende_sotto_il_flip(self):
        ev = detect_events(_stato(spot=79_000.0), _stato(spot=80_500.0))
        flip = next(e for e in ev if e.key == "gamma_flip_crossed")
        assert flip.meta["direction"] == "down"

    def test_spot_che_sale_sopra_il_flip(self):
        ev = detect_events(_stato(spot=80_500.0), _stato(spot=79_000.0))
        flip = next(e for e in ev if e.key == "gamma_flip_crossed")
        assert flip.meta["direction"] == "up"

    def test_call_wall_superato(self):
        ev = detect_events(_stato(spot=83_000.0), _stato(spot=81_000.0))
        assert any(e.key == "call_wall_crossed" for e in ev)

    def test_put_wall_perso(self):
        ev = detect_events(_stato(spot=74_000.0), _stato(spot=76_000.0))
        muro = next(e for e in ev if e.key == "put_wall_crossed")
        assert muro.meta["direction"] == "down"

    def test_movimento_dentro_il_corridoio_non_attraversa_nulla(self):
        ev = detect_events(_stato(spot=77_900.0), _stato(spot=77_500.0))
        assert not [e for e in ev if e.key.endswith("_crossed")]


class TestBarriere:
    def test_barriera_attraversata(self):
        ev = detect_events(_stato(spot=76_000.0), _stato(spot=77_500.0))
        b = next(e for e in ev if e.key == "barrier_breached")
        assert b.meta["levels"] == [76_897.0]

    def test_piu_barriere_insieme_alzano_la_severita(self):
        una = detect_events(_stato(spot=76_000.0), _stato(spot=77_500.0))
        molte = detect_events(_stato(spot=74_000.0), _stato(spot=80_000.0))
        s_una = next(e.severity for e in una if e.key == "barrier_breached")
        s_molte = next(e.severity for e in molte if e.key == "barrier_breached")
        assert s_molte > s_una

    def test_severita_resta_nel_range(self):
        molti = _stato(barrier_levels=[76_000.0 + i * 100 for i in range(20)])
        ev = detect_events(_stato(spot=70_000.0), {**molti, "spot": 85_000.0})
        assert all(0.0 <= e.severity <= 1.0 for e in ev)


class TestSegnale:
    def test_attraversamento_della_soglia_long(self):
        ev = detect_events(_stato(signal_score=70.0), _stato(signal_score=60.0))
        s = next(e for e in ev if e.key == "signal_crossed_65")
        assert s.meta["direction"] == "up"

    def test_caduta_in_risk_off(self):
        ev = detect_events(_stato(signal_score=35.0), _stato(signal_score=45.0))
        assert any(e.key == "signal_crossed_40" for e in ev)

    def test_movimento_dentro_la_fascia_caution_non_e_evento(self):
        ev = detect_events(_stato(signal_score=55.0), _stato(signal_score=45.0))
        assert not [e for e in ev if e.key.startswith("signal_crossed")]


class TestFlussi:
    def test_picco_di_flusso_sopra_soglia(self):
        ev = detect_events(_stato(flow_3d_usd_m=620.0), _stato(flow_3d_usd_m=200.0))
        assert any(e.key == "flow_spike" for e in ev)

    def test_non_ripete_l_evento_se_era_gia_sopra_soglia(self):
        """Un flusso che resta alto è uno stato, non una notizia nuova ogni giro."""
        ev = detect_events(_stato(flow_3d_usd_m=650.0), _stato(flow_3d_usd_m=600.0))
        assert not [e for e in ev if e.key == "flow_spike"]

    def test_anche_le_uscite_sono_un_picco(self):
        ev = detect_events(_stato(flow_3d_usd_m=-700.0), _stato(flow_3d_usd_m=-100.0))
        assert any(e.key == "flow_spike" for e in ev)


class TestOrdinamentoEPubblicazione:
    def test_eventi_ordinati_per_gravita(self):
        ev = detect_events(
            _stato(spot=74_000.0, regime="negative_gamma", signal_score=35.0),
            _stato(spot=80_500.0, signal_score=45.0),
        )
        assert [e.severity for e in ev] == sorted((e.severity for e in ev), reverse=True)

    def test_un_evento_grave_fa_uscire_l_edizione(self):
        ev = detect_events(_stato(regime="negative_gamma"), _stato())
        assert should_publish(ev)

    def test_giornata_piatta_non_pubblica(self):
        assert not should_publish(detect_events(_stato(spot=77_750.0), _stato(spot=77_700.0)))

    def test_nessun_evento_non_pubblica(self):
        assert not should_publish([])

    def test_soglia_di_pubblicazione_e_configurabile(self):
        ev = detect_events(_stato(flow_3d_usd_m=620.0), _stato(flow_3d_usd_m=200.0))
        assert should_publish(ev, threshold=0.6)
        assert not should_publish(ev, threshold=0.9)
        assert PUBLISH_THRESHOLD < 0.9


class TestReportStateDB:
    @pytest.fixture
    def db(self, tmp_path) -> ReportStateDB:
        return ReportStateDB(db_path=tmp_path / "runtime.db")

    def test_senza_niente_salvato_torna_none(self, db):
        assert db.load() is None

    def test_salva_e_rilegge(self, db):
        db.save(_stato())
        assert db.load()["spot"] == 77_704.0

    def test_il_salvataggio_sovrascrive(self, db):
        db.save(_stato())
        db.save(_stato(spot=90_000.0))
        assert db.load()["spot"] == 90_000.0

    def test_chiavi_diverse_sono_indipendenti(self, db):
        db.save(_stato(), key="a")
        db.save(_stato(spot=1.0), key="b")
        assert db.load("a")["spot"] == 77_704.0
        assert db.load("b")["spot"] == 1.0

    def test_riga_corrotta_non_blocca_l_edizione(self, db, tmp_path):
        import sqlite3

        db.save(_stato())
        conn = sqlite3.connect(tmp_path / "runtime.db")
        conn.execute("UPDATE report_state SET payload = ? WHERE key = ?", ("{non-json", "desk_note"))
        conn.commit()
        conn.close()
        assert db.load() is None  # riparte dalla linea di base invece di sollevare

    def test_giro_completo_salva_confronta_rileva(self, db):
        """Il ciclo reale: salvo, arriva un nuovo stato, rilevo cosa è cambiato."""
        db.save(_stato())
        nuovo = _stato(spot=74_000.0, regime="negative_gamma")
        eventi = detect_events(nuovo, db.load())
        assert should_publish(eventi)
        db.save(nuovo)
        assert detect_events(nuovo, db.load()) == []  # niente di nuovo al giro dopo
