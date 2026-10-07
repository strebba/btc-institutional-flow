"""Pagine della dashboard e loro disponibilità per asset.

Ogni pagina dichiara la feature di cui ha bisogno (vedi ``AssetSpec.features``):
con ETH, che in fase 1 ha solo i dati, spariscono Barrier Map ed EDGAR invece
di mostrare pagine vuote o numeri BTC sotto un'etichetta ETH.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assets import AssetSpec

_PAGES_DIR = Path(__file__).resolve().parent / "app_pages"


@dataclass(frozen=True)
class PageSpec:
    file: str
    title: str
    icon: str
    feature: str
    default: bool = False

    @property
    def path(self) -> str:
        return str(_PAGES_DIR / self.file)


PAGES: tuple[PageSpec, ...] = (
    PageSpec("panoramica.py", "Panoramica", ":material/dashboard:", "gex", default=True),
    PageSpec("gex.py", "GEX", ":material/candlestick_chart:", "gex"),
    PageSpec("flows.py", "ETF Flows", ":material/water:", "flows"),
    PageSpec("barrier_map.py", "Barrier Map", ":material/sell:", "barriers"),
    PageSpec("edgar.py", "EDGAR", ":material/search:", "edgar"),
)


def visible_pages(spec: AssetSpec) -> list[PageSpec]:
    """Pagine con dati per l'asset, nell'ordine della navigazione."""
    return [p for p in PAGES if spec.has(p.feature)]
