"""Test unitari per StructuredNotesDB."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from src.edgar.models import BarrierLevel, StructuredNote
from src.edgar.structured_notes_db import StructuredNotesDB


@pytest.fixture
def db(tmp_path: Path) -> StructuredNotesDB:
    """DB temporaneo per i test."""
    return StructuredNotesDB(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_note() -> StructuredNote:
    return StructuredNote(
        filing_url="https://example.com/test.htm",
        issuer="JPMorgan",
        issue_date=date(2024, 6, 1),
        maturity_date=date(2025, 6, 1),
        notional_usd=10_000_000.0,
        product_type="autocallable",
        initial_level=45.0,
        autocall_trigger_pct=100.0,
        knockin_barrier_pct=70.0,
        barriers=[
            BarrierLevel(barrier_type="knock_in", level_pct=70.0, level_price_ibit=31.5),
            BarrierLevel(barrier_type="autocall",  level_pct=100.0, level_price_ibit=45.0),
        ],
    )


class TestUpsertNote:
    def test_insert_new(self, db, sample_note):
        note_id = db.upsert_note(sample_note)
        assert note_id > 0

    def test_idempotent(self, db, sample_note):
        id1 = db.upsert_note(sample_note)
        id2 = db.upsert_note(sample_note)  # stesso url → update
        assert id1 == id2

    def test_barriers_saved(self, db, sample_note):
        db.upsert_note(sample_note)
        retrieved = db.get_all_notes()
        assert len(retrieved) == 1
        assert len(retrieved[0].barriers) == 2


class TestGetAllNotes:
    def test_empty_db(self, db):
        assert db.get_all_notes() == []

    def test_returns_notes(self, db, sample_note):
        db.upsert_note(sample_note)
        notes = db.get_all_notes()
        assert len(notes) == 1
        n = notes[0]
        assert n.issuer == "JPMorgan"
        assert n.notional_usd == 10_000_000.0
        assert n.initial_level == 45.0


class TestActiveBarriers:
    def test_active(self, db, sample_note):
        db.upsert_note(sample_note)
        active = db.get_active_barriers()
        assert len(active) == 2
        assert all(b["status"] == "active" for b in active)


class TestComputeBtcPrices:
    def test_conversion(self, db, sample_note):
        db.upsert_note(sample_note)
        ratio = 0.001  # 1 IBIT ≈ 0.001 BTC
        db.compute_btc_prices(ibit_btc_ratio=ratio)
        notes = db.get_all_notes()
        btc_prices = [b.level_price_btc for b in notes[0].barriers if b.level_price_ibit]
        assert all(p is not None for p in btc_prices)
        # level_price_ibit = 31.5 → btc_price = 31.5 / 0.001 = 31500
        assert any(abs(p - 31_500) < 1 for p in btc_prices)


class TestUpdateBarrierStatus:
    def test_knockin_triggered(self, db, sample_note):
        db.upsert_note(sample_note)
        # IBIT a 30 < 31.5 → knock_in triggered
        counts = db.update_barrier_statuses(current_ibit_price=30.0)
        assert counts["triggered"] >= 1

    def test_no_trigger(self, db, sample_note):
        db.upsert_note(sample_note)
        # IBIT a 40: knock_in level=31.5 (40>31.5 → no), autocall level=45 (40<45 → no)
        counts = db.update_barrier_statuses(current_ibit_price=40.0)
        assert counts["triggered"] == 0


class TestPreliminaryFlag:
    def _prelim_note(self) -> StructuredNote:
        return StructuredNote(
            filing_url="https://example.com/prelim.htm",
            issuer="JPMorgan",
            issue_date=date(2026, 2, 2),
            product_type="autocallable",
            is_preliminary=True,
            barriers=[BarrierLevel(barrier_type="knock_in", level_pct=70.0)],
        )

    def test_schema_version_current(self, db):
        with db._conn() as conn:
            ver = conn.execute("PRAGMA user_version").fetchone()[0]
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(notes)")}
        assert ver >= 2
        assert "is_preliminary" in cols

    def test_roundtrip_flag(self, db):
        db.upsert_note(self._prelim_note())
        n = db.get_all_notes()[0]
        assert n.is_preliminary is True

    def test_default_false(self, db, sample_note):
        db.upsert_note(sample_note)
        assert db.get_all_notes()[0].is_preliminary is False

    def test_active_barriers_exclude_preliminary(self, db, sample_note):
        # Una nota finale (2 barriere) + una preliminare (1 barriera).
        db.upsert_note(sample_note)
        db.upsert_note(self._prelim_note())
        active = db.get_active_barriers()
        # Solo le 2 barriere della nota finale devono comparire.
        assert len(active) == 2
        assert all(b["filing_url"] == sample_note.filing_url for b in active)


class TestUnderlyingFilter:
    """Le funzioni in prezzi IBIT devono ignorare le note su altri ETF."""

    def _fbtc_note(self) -> StructuredNote:
        return StructuredNote(
            filing_url="https://example.com/fbtc.htm",
            issuer="Goldman Sachs",
            issue_date=date(2026, 1, 15),
            product_type="autocallable",
            underlying="FBTC",
            initial_level=80.0,
            barriers=[BarrierLevel(barrier_type="knock_in", level_pct=70.0,
                                   level_price_ibit=56.0)],
        )

    def test_underlying_roundtrip(self, db):
        db.upsert_note(self._fbtc_note())
        assert db.get_all_notes()[0].underlying == "FBTC"

    def test_active_barriers_exclude_non_ibit(self, db, sample_note):
        db.upsert_note(sample_note)
        db.upsert_note(self._fbtc_note())
        active = db.get_active_barriers()
        assert len(active) == 2
        assert all(b["underlying"] == "IBIT" for b in active)

    def test_active_barriers_explicit_underlying(self, db, sample_note):
        db.upsert_note(sample_note)
        db.upsert_note(self._fbtc_note())
        active = db.get_active_barriers(underlying="FBTC")
        assert len(active) == 1
        assert active[0]["underlying"] == "FBTC"

    def test_compute_btc_prices_skips_non_ibit(self, db, sample_note):
        # Il ratio IBIT/BTC non si applica ai prezzi FBTC: level_price_btc
        # delle note FBTC deve restare NULL.
        db.upsert_note(sample_note)
        db.upsert_note(self._fbtc_note())
        db.compute_btc_prices(ibit_btc_ratio=0.001)
        notes = {n.filing_url: n for n in db.get_all_notes()}
        fbtc_prices = [b.level_price_btc for b in notes[self._fbtc_note().filing_url].barriers]
        ibit_prices = [b.level_price_btc for b in notes[sample_note.filing_url].barriers]
        assert all(p is None for p in fbtc_prices)
        assert all(p is not None for p in ibit_prices)

    def test_update_statuses_skips_non_ibit(self, db, sample_note):
        # IBIT a 30: la knock_in FBTC (56.0, in prezzi FBTC) non va confrontata
        # col prezzo IBIT e deve restare active.
        db.upsert_note(sample_note)
        db.upsert_note(self._fbtc_note())
        db.update_barrier_statuses(current_ibit_price=30.0)
        fbtc_barriers = db.get_active_barriers(underlying="FBTC")
        assert len(fbtc_barriers) == 1
        assert fbtc_barriers[0]["status"] == "active"


class TestCheckpoint:
    def test_checkpoint_persists_to_main_db(self, tmp_path, sample_note):
        # Scrive, fa checkpoint, poi rimuove i sidecar -wal/-shm: i dati devono
        # restare nel file .db principale (il caso del DB versionato in git).
        path = tmp_path / "cp.db"
        db = StructuredNotesDB(db_path=path)
        db.upsert_note(sample_note)
        db.checkpoint()
        for sidecar in (f"{path}-wal", f"{path}-shm"):
            p = Path(sidecar)
            if p.exists():
                p.unlink()
        db2 = StructuredNotesDB(db_path=path)
        assert len(db2.get_all_notes()) == 1


class TestSummary:
    def test_summary(self, db, sample_note):
        db.upsert_note(sample_note)
        s = db.summary()
        assert s["total_notes"] == 1
        assert s["total_barriers"] == 2
        assert s["active_barriers"] == 2
        assert s["total_notional_usd"] == 10_000_000.0
        assert "autocallable" in s["by_product_type"]
        assert "JPMorgan" in s["by_issuer"]


class TestRefreshRuns:
    """Un mercato tranquillo non deve far sembrare rotta una pipeline sana.

    Il workflow gira due volte a settimana; se in quella finestra nessuna banca
    deposita note IBIT, MAX(created_at) non si muove. Registrare l'esito del run
    separa "il refresh non gira" da "il refresh gira e non trova niente".
    """

    def test_senza_run_registrati_torna_none(self, db):
        assert db.get_last_refresh_run() is None

    def test_registra_e_rilegge(self, db):
        db.record_refresh_run(filings_seen=12, notes_written=3, ok=True)
        run = db.get_last_refresh_run()
        assert run["filings_seen"] == 12
        assert run["notes_written"] == 3
        assert run["ok"] is True
        assert run["run_at"]

    def test_registra_anche_un_run_a_vuoto(self, db):
        """Il caso che ha generato il falso allarme: zero filing, run riuscito."""
        db.record_refresh_run(filings_seen=0, notes_written=0, ok=True)
        run = db.get_last_refresh_run()
        assert run["ok"] is True
        assert run["notes_written"] == 0

    def test_registra_un_fallimento(self, db):
        db.record_refresh_run(filings_seen=0, notes_written=0, ok=False)
        assert db.get_last_refresh_run()["ok"] is False

    def test_tiene_il_piu_recente(self, db):
        db.record_refresh_run(filings_seen=1, notes_written=1, ok=True)
        db.record_refresh_run(filings_seen=9, notes_written=0, ok=True)
        assert db.get_last_refresh_run()["filings_seen"] == 9

    def test_conserva_lo_storico(self, db):
        for i in range(3):
            db.record_refresh_run(filings_seen=i, notes_written=0, ok=True)
        with db._conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM refresh_runs").fetchone()[0] == 3


class TestEdgarStatsDistingueRefreshDaScrittura:
    def test_espone_entrambe_le_date(self, db, sample_note):
        db.upsert_note(sample_note)
        db.record_refresh_run(filings_seen=0, notes_written=0, ok=True)
        stats = db.get_edgar_stats()
        assert stats["last_refresh_at"] is not None
        assert stats["last_note_written_at"] is not None

    def test_senza_refresh_registrati_la_data_e_none(self, db, sample_note):
        """Retrocompatibile: un DB che precede la tabella non deve esplodere."""
        db.upsert_note(sample_note)
        stats = db.get_edgar_stats()
        assert stats["last_refresh_at"] is None
        assert stats["last_note_written_at"] is not None

    def test_last_update_resta_per_compatibilita(self, db, sample_note):
        db.upsert_note(sample_note)
        stats = db.get_edgar_stats()
        assert stats["last_update"] == stats["last_note_written_at"]


class TestMacroSnapshots:
    """Storico dell'open interest, per la variazione a 7 giorni.

    Sta nel DB versionato e non in runtime.db perche' il filesystem di DO e'
    effimero: uno storico li' sparirebbe a ogni redeploy e la finestra a sette
    giorni non maturerebbe mai.
    """

    def test_senza_storico_la_variazione_e_none(self, db):
        assert db.get_oi_change_pct(days=7) is None

    def test_con_un_solo_punto_la_variazione_e_none(self, db):
        """Un punto non fa una variazione: meglio None che un numero inventato."""
        db.record_macro_snapshot(funding_ann_pct=12.3, oi_usd=66e9, n_contracts=138)
        assert db.get_oi_change_pct(days=7) is None

    def test_registra_e_rilegge(self, db):
        db.record_macro_snapshot(funding_ann_pct=12.3, oi_usd=66e9, n_contracts=138)
        ultimo = db.get_last_macro_snapshot()
        assert ultimo["funding_ann_pct"] == 12.3
        assert ultimo["oi_usd"] == 66e9
        assert ultimo["n_contracts"] == 138

    def test_un_solo_snapshot_al_giorno(self, db):
        """Due giri nello stesso giorno aggiornano la riga invece di duplicarla."""
        db.record_macro_snapshot(funding_ann_pct=10.0, oi_usd=60e9, n_contracts=100)
        db.record_macro_snapshot(funding_ann_pct=12.0, oi_usd=66e9, n_contracts=138)
        with db._conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM macro_snapshots").fetchone()[0] == 1
        assert db.get_last_macro_snapshot()["oi_usd"] == 66e9

    def test_calcola_la_variazione_su_due_punti_distanti(self, db):
        from datetime import date, timedelta

        vecchio = (date.today() - timedelta(days=7)).isoformat()
        db.record_macro_snapshot(funding_ann_pct=10.0, oi_usd=60e9, n_contracts=100,
                                 snapshot_date=vecchio)
        db.record_macro_snapshot(funding_ann_pct=12.0, oi_usd=66e9, n_contracts=138)
        # da 60B a 66B = +10%
        assert db.get_oi_change_pct(days=7) == pytest.approx(10.0, abs=0.01)

    def test_ignora_i_punti_troppo_recenti(self, db):
        """Con soli due giorni di storico la finestra a sette non e' ancora matura."""
        from datetime import date, timedelta

        db.record_macro_snapshot(funding_ann_pct=10.0, oi_usd=60e9, n_contracts=100,
                                 snapshot_date=(date.today() - timedelta(days=2)).isoformat())
        db.record_macro_snapshot(funding_ann_pct=12.0, oi_usd=66e9, n_contracts=138)
        assert db.get_oi_change_pct(days=7) is None

    def test_tollera_una_finestra_approssimata(self, db):
        """Il workflow gira una volta al giorno ma puo' saltare: 6 o 9 giorni vanno bene."""
        from datetime import date, timedelta

        db.record_macro_snapshot(funding_ann_pct=10.0, oi_usd=50e9, n_contracts=100,
                                 snapshot_date=(date.today() - timedelta(days=9)).isoformat())
        db.record_macro_snapshot(funding_ann_pct=12.0, oi_usd=55e9, n_contracts=138)
        assert db.get_oi_change_pct(days=7) == pytest.approx(10.0, abs=0.01)

    def test_oi_zero_non_divide_per_zero(self, db):
        from datetime import date, timedelta

        db.record_macro_snapshot(funding_ann_pct=10.0, oi_usd=0.0, n_contracts=0,
                                 snapshot_date=(date.today() - timedelta(days=7)).isoformat())
        db.record_macro_snapshot(funding_ann_pct=12.0, oi_usd=66e9, n_contracts=138)
        assert db.get_oi_change_pct(days=7) is None
