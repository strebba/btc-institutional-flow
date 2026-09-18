"""Test per i componenti della dashboard Streamlit."""

from __future__ import annotations


class TestBarrierMapTab:
    def test_module_importable(self):
        from src.dashboard.tabs import barrier_map
        assert barrier_map is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.barrier_map import _tab_barrier_map
        assert callable(_tab_barrier_map)


class TestGexTab:
    def test_module_importable(self):
        from src.dashboard.tabs import gex
        assert gex is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.gex import _tab_gex
        assert callable(_tab_gex)


class TestFlowsTab:
    def test_module_importable(self):
        from src.dashboard.tabs import flows
        assert flows is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.flows import _tab_flows
        assert callable(_tab_flows)


class TestSignalsTab:
    def test_module_importable(self):
        from src.dashboard.tabs import signals
        assert signals is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.signals import _tab_signals
        assert callable(_tab_signals)


class TestEdgarTab:
    def test_module_importable(self):
        from src.dashboard.tabs import edgar
        assert edgar is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.edgar import _tab_edgar_monitor
        assert callable(_tab_edgar_monitor)


class TestValidationTab:
    def test_module_importable(self):
        from src.dashboard.tabs import validation
        assert validation is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.validation import _tab_validation
        assert callable(_tab_validation)


class TestPanoramicaTab:
    def test_module_importable(self):
        from src.dashboard.tabs import panoramica
        assert panoramica is not None

    def test_tab_function_exists(self):
        from src.dashboard.tabs.panoramica import _tab_panoramica
        assert callable(_tab_panoramica)


class TestComponents:
    def test_module_importable(self):
        from src.dashboard import components
        assert components is not None

    def test_component_functions_exist(self):
        from src.dashboard.components import inject_style, tape, hero, pillar_bars
        assert callable(inject_style)
        assert callable(tape)
        assert callable(hero)
        assert callable(pillar_bars)
