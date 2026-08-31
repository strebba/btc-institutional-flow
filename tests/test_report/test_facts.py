"""Test per gli estrattori di fatti del Desk Note."""
from __future__ import annotations

from src.report.facts import (
    SIGN_NEGATIVE,
    SIGN_POSITIVE,
    extract_all,
    fact_barrier_nearest,
    fact_charm_tide,
    fact_flows_3d,
    fact_flows_rotation,
    fact_funding_cost,
    fact_gex_asymmetry,
    fact_gex_flip,
    fact_gex_walls,
    fact_signal_scoreboard,
    fact_vanna_sign,
)

_TUTTI = [
    lambda: fact_gex_asymmetry({}),
    lambda: fact_gex_flip({}),
    lambda: fact_gex_walls({}),
    lambda: fact_barrier_nearest({}),
    lambda: fact_flows_3d({}, {}),
    lambda: fact_flows_rotation({}),
    lambda: fact_signal_scoreboard({}),
]


class TestSenzaDati:
    def test_ogni_estrattore_torna_none_su_payload_vuoto(self):
        """Nessun estrattore deve inventare quando i dati non ci sono."""
        assert [fn() for fn in _TUTTI] == [None] * len(_TUTTI)

    def test_extract_all_su_niente_torna_lista_vuota(self):
        assert extract_all() == []


class TestGexAsymmetry:
    def test_riconosce_la_contraddizione_e_la_premia(self, gex_payload):
        """GEX totale positivo ma negativo sotto lo spot: è il fatto del giorno."""
        f = fact_gex_asymmetry(gex_payload)
        assert f.salience > 0.8
        assert f.sign == SIGN_NEGATIVE
        assert f.meta["gex_below"] < 0 < f.meta["gex_above"]

    def test_hero_e_la_gamma_sotto_lo_spot(self, gex_payload):
        f = fact_gex_asymmetry(gex_payload)
        assert f.hero_value == "−21,7M"  # -11.11 + -10.57 milioni


class TestGexFlip:
    def test_spot_sotto_il_flip_e_negativo(self, gex_payload):
        f = fact_gex_flip(gex_payload)
        assert f.sign == SIGN_NEGATIVE
        assert f.meta["distance_pct"] < 0

    def test_piu_vicino_al_flip_piu_e_saliente(self, gex_payload):
        vicino = dict(gex_payload)
        vicino["snapshot"] = {**gex_payload["snapshot"], "gamma_flip_price": 77_800.0}
        lontano = dict(gex_payload)
        lontano["snapshot"] = {**gex_payload["snapshot"], "gamma_flip_price": 110_000.0}
        assert fact_gex_flip(vicino).salience > fact_gex_flip(lontano).salience

    def test_spot_sopra_il_flip_e_positivo(self, gex_payload):
        sopra = dict(gex_payload)
        sopra["snapshot"] = {**gex_payload["snapshot"], "gamma_flip_price": 70_000.0}
        assert fact_gex_flip(sopra).sign == SIGN_POSITIVE


class TestGexWalls:
    def test_titolo_nomina_entrambi_i_muri(self, gex_payload):
        f = fact_gex_walls(gex_payload)
        assert "82.000" in f.headline and "75.000" in f.headline

    def test_include_lo_sbilanciamento_di_open_interest(self, gex_payload):
        f = fact_gex_walls(gex_payload)
        assert "14.999" in f.body[0] and "499" in f.body[0]


class TestBarrierNearest:
    def test_trova_la_barriera_piu_vicina(self, barriers_payload):
        f = fact_barrier_nearest(barriers_payload)
        assert "76.897" in f.hero_value
        assert "JPMorgan" in f.hero_caption

    def test_barriera_sotto_lo_spot_e_negativa(self, barriers_payload):
        assert fact_barrier_nearest(barriers_payload).sign == SIGN_NEGATIVE

    def test_non_cita_dollari_se_il_notional_e_nullo(self, barriers_payload):
        """notional_usd è None in produzione: il testo deve parlare di conteggi."""
        f = fact_barrier_nearest(barriers_payload)
        testo = " ".join(f.body) + f.headline
        assert "notional" not in testo.lower()
        assert "293" in testo  # usa il totale attivo, non un valore inventato

    def test_barriera_lontana_e_meno_saliente(self, barriers_payload):
        lontana = {
            **barriers_payload,
            "barriers": [{"barrier_type": "knock_in", "level_price_btc": 40_000.0,
                          "issuer": "JPMorgan"}],
        }
        assert fact_barrier_nearest(lontana).salience < fact_barrier_nearest(
            barriers_payload
        ).salience


class TestFlussoNonMaterialeNonEUnFatto:
    """Un flusso a zero significa quasi sempre 'non lo sappiamo', non 'zero'.

    Quando Farside e yfinance sono irraggiungibili /api/signals restituisce
    ibit_flow_3d_usd_m = 0.0, indistinguibile da un flusso realmente nullo.
    Pubblicare '+0M di ETF in tre giorni' sarebbe un fatto falso col marchio sopra.
    """

    def test_flusso_zero_non_produce_card(self, flows_payload, signals_payload):
        zero = {**signals_payload, "inputs": {"ibit_flow_3d_usd_m": 0.0}}
        assert fact_flows_3d(flows_payload, zero) is None

    def test_flusso_trascurabile_non_produce_card(self, flows_payload, signals_payload):
        briciole = {**signals_payload, "inputs": {"ibit_flow_3d_usd_m": 3.0}}
        assert fact_flows_3d(flows_payload, briciole) is None

    def test_senza_storico_flussi_non_produce_card(self, signals_payload):
        """Senza il riepilogo non c'è modo di corroborare il numero."""
        assert fact_flows_3d({}, signals_payload) is None

    def test_flusso_grande_e_piu_saliente_di_uno_piccolo(self, flows_payload, signals_payload):
        piccolo = {**signals_payload, "inputs": {"ibit_flow_3d_usd_m": 60.0}}
        grande = {**signals_payload, "inputs": {"ibit_flow_3d_usd_m": 700.0}}
        assert (
            fact_flows_3d(flows_payload, grande).salience
            > fact_flows_3d(flows_payload, piccolo).salience
        )


class TestGexNonMaterialeNonEUnFatto:
    def test_profilo_piatto_non_produce_card(self, gex_payload):
        """Con gamma trascurabile su entrambi i lati non c'è niente da dire."""
        piatto = {**gex_payload, "strike_profile": [
            {"strike": 75_000.0, "net_gex_m": 0.0, "call_oi": 0, "put_oi": 0},
            {"strike": 82_000.0, "net_gex_m": 0.0, "call_oi": 0, "put_oi": 0},
        ]}
        assert fact_gex_asymmetry(piatto) is None


class TestFlows:
    def test_flusso_positivo_e_positivo(self, flows_payload, signals_payload):
        f = fact_flows_3d(flows_payload, signals_payload)
        assert f.sign == SIGN_POSITIVE
        assert f.hero_value == "+445M"

    def test_flusso_negativo_ribalta_il_segno(self, flows_payload, signals_payload):
        uscite = {**signals_payload, "inputs": {"ibit_flow_3d_usd_m": -300.0}}
        assert fact_flows_3d(flows_payload, uscite).sign == SIGN_NEGATIVE

    def test_rotazione_confronta_ibit_e_gbtc(self, flows_payload):
        f = fact_flows_rotation(flows_payload)
        assert "63,4B" in f.body[0] and "27,6B" in f.body[0]


class TestSignalScoreboard:
    def test_traduce_i_nomi_dei_pilastri(self, signals_payload):
        f = fact_signal_scoreboard(signals_payload)
        assert "flussi ETF" in f.body[0] and "barriere" in f.body[0]
        assert "etf_flows" not in f.body[0]

    def test_cita_le_previsioni_aperte_quando_ci_sono(self, signals_payload):
        f = fact_signal_scoreboard(signals_payload, {"open": 12})
        assert "12 previsioni" in f.body[1]


class TestExtractAll:
    def test_produce_tutti_i_fatti_disponibili(
        self, gex_payload, barriers_payload, flows_payload, signals_payload
    ):
        facts = extract_all(
            gex=gex_payload, barriers=barriers_payload,
            flows=flows_payload, signals=signals_payload,
        )
        chiavi = {f.key for f in facts}
        assert chiavi == {
            "gex_asymmetry", "gex_flip", "gex_walls", "barrier_nearest",
            "flows_3d", "flows_rotation", "signal_scoreboard",
        }

    def test_un_estrattore_rotto_non_blocca_gli_altri(self, gex_payload, monkeypatch):
        """Un dato malformato deve costare una card, non l'edizione."""
        rotto = {**gex_payload, "strike_profile": [{"strike": "non-un-numero"}]}
        facts = extract_all(gex=rotto)
        assert any(f.key == "gex_walls" for f in facts)

    def test_ogni_fatto_ha_salienza_nel_range(
        self, gex_payload, barriers_payload, flows_payload, signals_payload
    ):
        facts = extract_all(
            gex=gex_payload, barriers=barriers_payload,
            flows=flows_payload, signals=signals_payload,
        )
        assert all(0.0 <= f.salience <= 1.0 for f in facts)


_CHARM = {
    "charm": {
        "total_charm_usd_day": -67_900_000.0,
        "total_vanna_usd_per_iv_pt": 71_000_000.0,
        "magnet_strike": 82_000.0,
        "projection": [
            {"days_ahead": 0, "charm_usd_day": -67_900_000.0, "live_instruments": 854},
            {"days_ahead": 1, "charm_usd_day": -56_800_000.0, "live_instruments": 817},
            {"days_ahead": 2, "charm_usd_day": -31_000_000.0, "live_instruments": 654},
        ],
    }
}


class TestCharmTide:
    def test_senza_blocco_charm_torna_none(self):
        assert fact_charm_tide({}) is None
        assert fact_charm_tide({"charm": None}) is None

    def test_charm_trascurabile_non_e_un_fatto(self):
        piccolo = {"charm": {**_CHARM["charm"], "total_charm_usd_day": 1_000_000.0}}
        assert fact_charm_tide(piccolo) is None

    def test_charm_negativo_e_vendita_programmata(self):
        f = fact_charm_tide(_CHARM)
        assert f.sign == SIGN_NEGATIVE
        assert "vendita programmata" in f.headline

    def test_charm_positivo_e_acquisto_in_calendario(self):
        pos = {"charm": {**_CHARM["charm"], "total_charm_usd_day": 40_000_000.0}}
        f = fact_charm_tide(pos)
        assert f.sign == SIGN_POSITIVE
        assert "calendario" in f.headline

    def test_nomina_il_salto_post_expiry(self):
        """È il dettaglio che rende il charm una notizia con una scadenza."""
        f = fact_charm_tide(_CHARM)
        assert "scadenza fra 2 giorni" in f.body[1]
        assert "163" in f.body[1]  # 817 - 654 strumenti spenti

    def test_cita_lo_strike_magnete(self):
        assert "82.000" in fact_charm_tide(_CHARM).hero_caption

    def test_piu_grande_e_piu_saliente(self):
        piccolo = {"charm": {**_CHARM["charm"], "total_charm_usd_day": 8_000_000.0}}
        assert fact_charm_tide(_CHARM).salience > fact_charm_tide(piccolo).salience


class TestVannaSign:
    def test_senza_blocco_charm_torna_none(self):
        assert fact_vanna_sign({}) is None

    def test_vanna_trascurabile_non_e_un_fatto(self):
        piccola = {"charm": {**_CHARM["charm"], "total_vanna_usd_per_iv_pt": 500_000.0}}
        assert fact_vanna_sign(piccola) is None

    def test_vanna_positiva_la_vol_in_calo_compra(self):
        f = fact_vanna_sign(_CHARM)
        assert f.sign == SIGN_POSITIVE
        assert "compra" in f.headline

    def test_vanna_negativa_toglie_carburante(self):
        neg = {"charm": {**_CHARM["charm"], "total_vanna_usd_per_iv_pt": -71_000_000.0}}
        f = fact_vanna_sign(neg)
        assert f.sign == SIGN_NEGATIVE
        assert "spegne da solo" in " ".join(f.body)


class TestCharmNelRegistro:
    def test_extract_all_include_i_due_nuovi(self, gex_payload):
        arricchito = {**gex_payload, **_CHARM}
        chiavi = {f.key for f in extract_all(gex=arricchito)}
        assert "charm_tide" in chiavi
        assert "vanna_sign" in chiavi

    def test_condividono_la_famiglia_per_non_monopolizzare(self, gex_payload):
        """Stesso topic: il cap per famiglia impedisce un'edizione tutta di greche."""
        assert fact_charm_tide(_CHARM).topic == fact_vanna_sign(_CHARM).topic


class TestBarriereConNozionale:
    """Dopo il re-parse l'88% delle barriere attive ha un nozionale.

    È la differenza fra "quattro barriere entro il 2%" e "x milioni di note che
    si attivano entro il 2%" — il motivo per cui il re-parse valeva la pena.
    """

    @staticmethod
    def _payload(notional: float | None, n_note: int = 3) -> dict:
        return {
            "spot_price": 77_722.68,
            "meta": {"total_active": 293},
            "barriers": [
                {
                    "note_id": i,
                    "barrier_type": "knock_in",
                    "level_price_btc": 77_000.0 - i * 50,
                    "issuer": "JPMorgan",
                    "notional_usd": notional,
                }
                for i in range(n_note)
            ],
        }

    def test_il_titolo_dice_i_dollari_quando_ci_sono(self):
        f = fact_barrier_nearest(self._payload(50_000_000.0))
        assert "150M" in f.headline
        assert "si attivano" in f.headline
        assert f.hero_value == "150M"

    def test_ripiega_sui_conteggi_sotto_soglia(self):
        """Un numero piccolo in cifra tonda impressiona meno di "tre barriere"."""
        f = fact_barrier_nearest(self._payload(1_000_000.0))
        assert "barriere bancarie" in f.headline
        assert "M" not in f.hero_value

    def test_ripiega_sui_conteggi_senza_nozionale(self):
        f = fact_barrier_nearest(self._payload(None))
        assert "barriere bancarie" in f.headline

    def test_non_conta_la_stessa_nota_piu_volte(self):
        """Una nota con tre livelli non vale il triplo: si somma per nota."""
        tre_livelli = {
            "spot_price": 77_722.68,
            "meta": {"total_active": 293},
            "barriers": [
                {"note_id": 1, "barrier_type": "knock_in", "level_price_btc": lvl,
                 "issuer": "JPMorgan", "notional_usd": 100_000_000.0}
                for lvl in (77_000.0, 76_900.0, 76_800.0)
            ],
        }
        f = fact_barrier_nearest(tre_livelli)
        assert f.meta["notional_within_2pct_usd"] == 100_000_000.0

    def test_il_nozionale_finisce_nel_meta(self):
        f = fact_barrier_nearest(self._payload(50_000_000.0))
        assert f.meta["notional_within_2pct_usd"] == 150_000_000.0


class TestFundingPerpetui:
    """Il costo di restare long, che e' il fatto che CoinGecko ha reso disponibile.

    E' l'unico numero della serie che non viene dalle opzioni: dice quanto pagano
    i long agli short ogni anno per tenere la posizione aperta, ed e' leggibile
    da chiunque abbia mai pagato un interesse.
    """

    _MACRO = {
        "source_status": "partial_coingecko",
        "funding_source": "coingecko",
        "funding_rate_annualized_pct": 12.33,
        "futures_oi_usd": 66_238_576_882.0,
    }

    def test_senza_macro_non_produce_niente(self):
        assert fact_funding_cost({}) is None

    def test_senza_funding_non_produce_niente(self):
        assert fact_funding_cost({"futures_oi_usd": 6.6e10}) is None

    def test_funding_positivo_dice_chi_paga(self):
        f = fact_funding_cost(self._MACRO)
        assert f is not None
        assert "12,3" in f.hero_value
        assert "long" in " ".join(f.body).lower()

    def test_funding_negativo_ribalta_il_verso(self):
        f = fact_funding_cost({**self._MACRO, "funding_rate_annualized_pct": -8.0})
        assert f is not None
        corpo = " ".join(f.body).lower()
        assert "short" in corpo and "pagano" in corpo

    def test_un_funding_piatto_non_e_un_fatto(self):
        """Sotto la soglia non c'e' niente da raccontare: e' il costo di un BTP."""
        assert fact_funding_cost({**self._MACRO, "funding_rate_annualized_pct": 1.2}) is None

    def test_l_open_interest_da_la_scala(self):
        f = fact_funding_cost(self._MACRO)
        assert "66" in " ".join(f.body), "senza l'OI il 12% non dice quanto pesa"

    def test_un_funding_estremo_e_piu_saliente(self):
        tiepido = fact_funding_cost(self._MACRO)
        rovente = fact_funding_cost({**self._MACRO, "funding_rate_annualized_pct": 60.0})
        assert rovente.salience > tiepido.salience

    def test_ha_una_famiglia_sua(self):
        """Non e' ne' gex ne' flows: senza un topic proprio il cap per famiglia
        lo farebbe competere con card che non gli somigliano."""
        f = fact_funding_cost(self._MACRO)
        assert f.topic == "macro"

    def test_entra_nel_registro_degli_estrattori(self, gex_payload):
        chiavi = {f.key for f in extract_all(gex=gex_payload, macro=self._MACRO)}
        assert "funding_cost" in chiavi

    def test_il_registro_regge_un_macro_assente(self, gex_payload):
        """Il macro e' opzionale: chi chiamava extract_all senza non deve rompersi."""
        extract_all(gex=gex_payload)
