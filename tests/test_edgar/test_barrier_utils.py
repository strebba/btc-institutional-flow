"""Unit test per src/edgar/barrier_utils.py.

Confluenza barriere↔GEX: clustering, segno direzionale, rilevamento confluenza,
scoring. Funzioni pure — nessun I/O, nessun mock necessario (tranne la config,
che ha un default sensato).
"""
from __future__ import annotations

from src.edgar.barrier_utils import (
    BarrierCluster,
    barrier_confluence_scores,
    barrier_sign,
    compute_confluence,
    detect_clusters,
    get_proximity_pct,
)

SPOT = 85_000.0


# ─── barrier_sign ─────────────────────────────────────────────────────────────


class TestBarrierSign:
    def test_autocall_above_is_bullish(self):
        assert barrier_sign("autocall", 90_000, SPOT) == "bullish"

    def test_knockout_above_is_bullish(self):
        assert barrier_sign("knock_out", 90_000, SPOT) == "bullish"

    def test_knockin_below_is_bearish(self):
        assert barrier_sign("knock_in", 80_000, SPOT) == "bearish"

    def test_buffer_below_is_bearish(self):
        assert barrier_sign("buffer", 80_000, SPOT) == "bearish"

    def test_atypical_knockin_above_is_neutral(self):
        assert barrier_sign("knock_in", 90_000, SPOT) == "neutral"

    def test_invalid_levels_are_neutral(self):
        assert barrier_sign("knock_in", 0, SPOT) == "neutral"
        assert barrier_sign("knock_in", 80_000, 0) == "neutral"


# ─── detect_clusters ──────────────────────────────────────────────────────────


class TestDetectClusters:
    def test_empty_returns_empty(self):
        assert detect_clusters([], SPOT) == []

    def test_nearby_barriers_merge_into_one_cluster(self):
        barriers = [
            {"level_price_btc": 80_000, "barrier_type": "knock_in", "notional_usd": 100e6},
            {"level_price_btc": 80_200, "barrier_type": "knock_in", "notional_usd": 50e6},
        ]
        clusters = detect_clusters(barriers, SPOT, proximity_pct=2.0)
        assert len(clusters) == 1
        cl = clusters[0]
        assert isinstance(cl, BarrierCluster)
        assert cl.n_barriers == 2
        assert cl.total_notional_usd == 150e6
        assert cl.dominant_type == "knock_in"
        assert cl.sign == "bearish"
        assert 80_000 <= cl.mean_price_btc <= 80_200

    def test_distant_barriers_stay_separate(self):
        barriers = [
            {"level_price_btc": 80_000, "barrier_type": "knock_in"},
            {"level_price_btc": 95_000, "barrier_type": "autocall"},
        ]
        clusters = detect_clusters(barriers, SPOT, proximity_pct=2.0)
        assert len(clusters) == 2

    def test_invalid_levels_filtered(self):
        barriers = [
            {"level_price_btc": 0, "barrier_type": "knock_in"},
            {"level_price_btc": None, "barrier_type": "buffer"},
            {"level_price_btc": 80_000, "barrier_type": "knock_in"},
        ]
        clusters = detect_clusters(barriers, SPOT)
        assert len(clusters) == 1

    def test_clusters_sorted_descending_by_price(self):
        barriers = [
            {"level_price_btc": 80_000, "barrier_type": "knock_in"},
            {"level_price_btc": 95_000, "barrier_type": "autocall"},
        ]
        clusters = detect_clusters(barriers, SPOT, proximity_pct=2.0)
        assert clusters[0].mean_price_btc > clusters[1].mean_price_btc

    def test_spot_zero_yields_neutral_sign(self):
        barriers = [{"level_price_btc": 80_000, "barrier_type": "knock_in"}]
        clusters = detect_clusters(barriers, 0.0)
        assert clusters[0].sign == "neutral"


# ─── compute_confluence ───────────────────────────────────────────────────────


def _bearish_cluster(price=80_000.0, notional=100e6):
    return BarrierCluster(
        barriers=[], mean_price_btc=price, total_notional_usd=notional,
        dominant_type="knock_in", sign="bearish", n_barriers=1,
        distance_to_spot_pct=-5.0,
    )


def _bullish_cluster(price=95_000.0, notional=100e6):
    return BarrierCluster(
        barriers=[], mean_price_btc=price, total_notional_usd=notional,
        dominant_type="autocall", sign="bullish", n_barriers=1,
        distance_to_spot_pct=11.0,
    )


class TestComputeConfluence:
    def test_empty_clusters_returns_empty(self):
        assert compute_confluence([], 80_000, 95_000, 85_000) == []

    def test_bearish_cluster_on_put_wall_is_reinforced(self):
        conf = compute_confluence([_bearish_cluster(80_000)], put_wall=80_050,
                                  call_wall=95_000, gamma_flip=85_000)
        assert len(conf) == 1
        assert conf[0]["confluence_type"] == "bearish_reinforced"
        assert conf[0]["gex_level_name"] == "put_wall"

    def test_bullish_cluster_on_call_wall_is_reinforced(self):
        conf = compute_confluence([_bullish_cluster(95_000)], put_wall=80_000,
                                  call_wall=95_100, gamma_flip=85_000)
        types = {c["confluence_type"] for c in conf}
        assert "bullish_reinforced" in types

    def test_bearish_cluster_on_call_wall_is_mixed(self):
        conf = compute_confluence([_bearish_cluster(95_000)], put_wall=80_000,
                                  call_wall=95_050, gamma_flip=10_000)
        assert conf and all(c["confluence_type"] == "mixed" for c in conf)

    def test_far_cluster_yields_no_confluence(self):
        conf = compute_confluence([_bearish_cluster(80_000)], put_wall=70_000,
                                  call_wall=95_000, gamma_flip=88_000)
        assert conf == []

    def test_none_walls_skipped(self):
        conf = compute_confluence([_bearish_cluster(80_000)], put_wall=None,
                                  call_wall=0, gamma_flip=None)
        assert conf == []


# ─── barrier_confluence_scores ────────────────────────────────────────────────


class TestConfluenceScores:
    def test_empty_is_zero(self):
        assert barrier_confluence_scores([]) == (0.0, 0.0)

    def test_bearish_reinforced_high_notional_scores_bearish(self):
        conf = [{"confluence_type": "bearish_reinforced",
                 "cluster_notional_usd": 200e6, "cluster_n_barriers": 1}]
        bear, bull = barrier_confluence_scores(conf)
        assert bear >= 0.99
        assert bull == 0.0

    def test_bullish_reinforced_scores_bullish(self):
        conf = [{"confluence_type": "bullish_reinforced",
                 "cluster_notional_usd": 200e6, "cluster_n_barriers": 1}]
        bear, bull = barrier_confluence_scores(conf)
        assert bull >= 0.99
        assert bear == 0.0

    def test_scores_capped_at_one(self):
        conf = [{"confluence_type": "bearish_reinforced",
                 "cluster_notional_usd": 999e6, "cluster_n_barriers": 5}]
        bear, _ = barrier_confluence_scores(conf)
        assert bear <= 1.0


# ─── get_proximity_pct ────────────────────────────────────────────────────────


def test_get_proximity_pct_reads_settings():
    pct = get_proximity_pct()
    assert isinstance(pct, float)
    assert pct > 0
