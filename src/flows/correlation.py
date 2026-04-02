"""Calcolo della correlazione rolling tra flussi ETF e rendimenti BTC.

Unisce i dati Farside (flussi) con i prezzi yfinance (BTC, IBIT) e calcola
correlazioni rolling a 30, 60, 90 giorni con Plotly per la visualizzazione.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.config import get_settings, setup_logging
from src.flows.models import AggregateFlows, MergedRecord

_log = setup_logging("flows.correlation")


class FlowCorrelation:
    """Analisi di correlazione flussi ETF ↔ rendimenti BTC.

    Args:
        cfg: configurazione analytics.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        settings = get_settings()
        self._cfg = cfg or settings["analytics"]

    # ──────────────────────────────────────────────────────────────────────────
    # Data preparation
    # ──────────────────────────────────────────────────────────────────────────

    def merge(
        self,
        flows: list[AggregateFlows],
        prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Unisce flussi aggregati e prezzi in un unico DataFrame.

        Args:
            flows: lista di AggregateFlows (da FarsideScraper.aggregate).
            prices_df: DataFrame da PriceFetcher.get_all_prices().

        Returns:
            pd.DataFrame: con colonne ibit_flow, total_flow, btc_close,
                btc_return, ibit_close, ibit_btc_ratio, btc_vol_7d,
                e colonne per-ticker (es. fbtc_flow, gbtс_flow, ...).
        """
        if not flows:
            _log.warning("Lista flussi vuota — merge impossibile")
            return pd.DataFrame()

        # Costruisci il DataFrame con tutti i flussi per-ticker
        flow_records: list[dict] = []
        for f in flows:
            row: dict = {
                "date": pd.Timestamp(f.date),
                "ibit_flow": f.ibit_flow_usd,
                "total_flow": f.total_flow_usd,
            }
            # Aggiungi flussi per ogni ETF disponibile
            for ticker, val in f.flows_by_ticker.items():
                col_name = f"{ticker.lower()}_flow"
                row[col_name] = val
            flow_records.append(row)

        flow_df = pd.DataFrame(flow_records).set_index("date")

        # Se ci sono duplicati (stessa data da fonti diverse), aggrega con somma
        flow_df = flow_df.groupby(level=0).sum()

        merged = flow_df.join(prices_df, how="outer")
        merged.sort_index(inplace=True)

        # Flussi BTC nei prossimi N giorni (per analisi di predittività)
        merged["btc_return_next1d"] = merged["btc_return"].shift(-1)
        merged["btc_return_next2d"] = merged["btc_return"].shift(-2)
        # Flussi rolling 3 giorni
        merged["ibit_flow_3d"] = merged["ibit_flow"].rolling(3, min_periods=1).sum()
        merged["total_flow_3d"] = merged["total_flow"].rolling(3, min_periods=1).sum()

        _log.info(
            "Merged: %d righe, %s → %s, colonne: %s",
            len(merged),
            merged.index.min().date(),
            merged.index.max().date(),
            list(merged.columns),
        )
        # Report gap
        bday_diff = merged.index.to_series().diff().dt.days
        gaps = bday_diff[bday_diff > 3]
        if not gaps.empty:
            _log.warning("Gap nel dataset: %s", gaps.to_dict())

        return merged

    # ──────────────────────────────────────────────────────────────────────────
    # Correlazione rolling
    # ──────────────────────────────────────────────────────────────────────────

    def rolling_correlations(
        self,
        merged: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Calcola rolling correlation per diverse finestre temporali.

        Per ogni finestra calcola:
          - corr(ibit_flow, btc_return_next1d)
          - corr(total_flow, btc_return_next1d)
          - corr(ibit_flow, btc_vol_7d)

        Args:
            merged: DataFrame dal metodo merge().
            windows: finestre in giorni (default da settings.yaml).

        Returns:
            dict[str, pd.DataFrame]: chiave = "30d"/"60d"/"90d",
                valore = DataFrame con le correlazioni.
        """
        windows = windows or self._cfg.get("correlation_windows", [30, 60, 90])
        results: dict[str, pd.DataFrame] = {}

        pairs = [
            ("ibit_flow", "btc_return_next1d", "ibit_flow_vs_btc_return"),
            ("total_flow", "btc_return_next1d", "total_flow_vs_btc_return"),
            ("ibit_flow", "btc_vol_7d", "ibit_flow_vs_btc_vol"),
        ]

        for w in windows:
            corr_df = pd.DataFrame(index=merged.index)
            for x_col, y_col, label in pairs:
                if x_col not in merged.columns or y_col not in merged.columns:
                    _log.warning("Colonna mancante: %s o %s", x_col, y_col)
                    continue
                corr_df[label] = (
                    merged[[x_col, y_col]]
                    .dropna()
                    .rolling(w, min_periods=max(10, w // 3))
                    .corr()
                    .unstack()[y_col][x_col]
                    .reindex(merged.index)
                )
            results[f"{w}d"] = corr_df
            _log.info("Correlazione rolling %dd calcolata", w)

        return results

    def summary_stats(self, merged: pd.DataFrame) -> dict:
        """Calcola statistiche descrittive sui flussi e sui rendimenti.

        Args:
            merged: DataFrame da merge().

        Returns:
            dict: metriche aggregate.
        """
        stats: dict = {}

        if "ibit_flow" in merged.columns:
            ibit = merged["ibit_flow"].dropna()
            stats["ibit"] = {
                "total_inflow_usd_b": ibit[ibit > 0].sum() / 1e9,
                "total_outflow_usd_b": ibit[ibit < 0].sum() / 1e9,
                "net_flow_usd_b": ibit.sum() / 1e9,
                "max_daily_inflow": ibit.max(),
                "max_daily_outflow": ibit.min(),
                "days_with_data": len(ibit),
            }

        if "total_flow" in merged.columns:
            total = merged["total_flow"].dropna()
            stats["total_etf"] = {
                "net_flow_usd_b": total.sum() / 1e9,
                "avg_daily_usd_m": total.mean() / 1e6,
            }

        if "btc_return" in merged.columns:
            ret = merged["btc_return"].dropna()
            stats["btc"] = {
                "annualized_vol": ret.std() * (252**0.5),
                "sharpe_approx": ret.mean() / ret.std() * (252**0.5) if ret.std() > 0 else 0,
                "total_return": float(np.exp(ret.sum()) - 1),
            }

        # Correlazione punto (tutta la serie)
        if "ibit_flow" in merged.columns and "btc_return_next1d" in merged.columns:
            valid = merged[["ibit_flow", "btc_return_next1d"]].dropna()
            if len(valid) > 5:
                corr = valid.corr().iloc[0, 1]
                stats["full_period_corr_ibit_btc_next1d"] = round(float(corr), 4)

        # Per-ticker stats (all ETFs)
        etf_flow_cols = [
            c
            for c in merged.columns
            if c.endswith("_flow") and c not in ("ibit_flow", "total_flow")
        ]
        if etf_flow_cols:
            etf_stats: dict[str, dict] = {}
            for col in etf_flow_cols:
                ticker = col.replace("_flow", "").upper()
                series = merged[col].dropna()
                if not series.empty:
                    etf_stats[ticker] = {
                        "net_flow_usd_b": round(series.sum() / 1e9, 3),
                        "avg_daily_usd_m": round(series.mean() / 1e6, 2),
                        "max_inflow_usd_m": round(series.max() / 1e6, 2),
                        "max_outflow_usd_m": round(series.min() / 1e6, 2),
                    }
            if etf_stats:
                stats["by_ticker"] = etf_stats

        return stats

    def flow_concentration(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Calcola la concentrazione dei flussi per ogni ETF.

        Per ogni giorno, calcola:
          - % del totale per ogni ETF (valore assoluto)
          - Herfindahl-Hirschman Index (HHI) per concentrazione
          - Numero di ETF con inflow > 0

        Args:
            merged: DataFrame da merge().

        Returns:
            pd.DataFrame con colonne: ibit_pct, fbtc_pct, ..., hhi, n_inflow_etfs.
        """
        etf_flow_cols = [c for c in merged.columns if c.endswith("_flow") and c != "total_flow"]
        if not etf_flow_cols:
            return pd.DataFrame()

        etf_flows = merged[etf_flow_cols].fillna(0)
        abs_total = etf_flows.abs().sum(axis=1)
        abs_total = abs_total.replace(0, np.nan)

        result = pd.DataFrame(index=merged.index)
        for col in etf_flow_cols:
            ticker = col.replace("_flow", "")
            result[f"{ticker}_pct"] = (etf_flows[col].abs() / abs_total * 100).round(2)

        # HHI: somma dei quadrati delle quote (0-10000, dove 10000 = monopolio)
        shares = etf_flows.abs().div(abs_total, axis=0)
        result["hhi"] = (shares**2).sum(axis=1).round(4)

        # Numero di ETF con inflow positivo
        result["n_inflow_etfs"] = (etf_flows > 0).sum(axis=1)

        return result.dropna(how="all")

    def flow_divergence(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Rileva divergenze tra flussi degli ETF maggiori.

        Una divergenza si verifica quando ETF principali hanno flussi
        di segno opposto (es. IBIT inflow + GBTC outflow).

        Args:
            merged: DataFrame da merge().

        Returns:
            pd.DataFrame con colonne:
              - divergence_flag: True se c'è divergenza significativa
              - divergence_magnitude: valore assoluto della differenza netta
              - inflow_etfs: lista ticker con inflow
              - outflow_etfs: lista ticker con outflow
        """
        major_tickers = ["ibit_flow", "fbtc_flow", "gbtc_flow", "bitb_flow", "arkb_flow"]
        available = [c for c in major_tickers if c in merged.columns]
        if len(available) < 2:
            return pd.DataFrame()

        etf_flows = merged[available].fillna(0)
        result = pd.DataFrame(index=merged.index)

        # Divergenza: almeno un ETF con inflow e uno con outflow significativo
        has_inflow = (etf_flows > 50e6).any(axis=1)  # > $50M inflow
        has_outflow = (etf_flows < -50e6).any(axis=1)  # > $50M outflow
        result["divergence_flag"] = has_inflow & has_outflow

        # Magnitudine: differenza tra il max inflow e il max outflow
        result["divergence_magnitude_usd_m"] = (
            (etf_flows.max(axis=1) - etf_flows.min(axis=1)) / 1e6
        ).round(2)

        # Ticker con inflow/outflow
        result["inflow_etfs"] = etf_flows.apply(
            lambda row: [c.replace("_flow", "").upper() for c in available if row[c] > 50e6], axis=1
        )
        result["outflow_etfs"] = etf_flows.apply(
            lambda row: [c.replace("_flow", "").upper() for c in available if row[c] < -50e6],
            axis=1,
        )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Conversione in MergedRecord list
    # ──────────────────────────────────────────────────────────────────────────

    def to_merged_records(self, merged: pd.DataFrame) -> list[MergedRecord]:
        """Converte il DataFrame in lista di MergedRecord dataclass.

        Args:
            merged: DataFrame da merge().

        Returns:
            list[MergedRecord].
        """
        result: list[MergedRecord] = []
        for idx, row in merged.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            result.append(
                MergedRecord(
                    date=d,
                    ibit_flow=float(row["ibit_flow"]) if pd.notna(row.get("ibit_flow")) else None,
                    total_flow=float(row["total_flow"])
                    if pd.notna(row.get("total_flow"))
                    else None,
                    btc_close=float(row["btc_close"]) if pd.notna(row.get("btc_close")) else None,
                    btc_return=float(row["btc_return"])
                    if pd.notna(row.get("btc_return"))
                    else None,
                    ibit_close=float(row["ibit_close"])
                    if pd.notna(row.get("ibit_close"))
                    else None,
                    ibit_btc_ratio=float(row["ibit_btc_ratio"])
                    if pd.notna(row.get("ibit_btc_ratio"))
                    else None,
                    btc_realized_vol_7d=float(row["btc_vol_7d"])
                    if pd.notna(row.get("btc_vol_7d"))
                    else None,
                )
            )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Plotly visualization
    # ──────────────────────────────────────────────────────────────────────────

    def plot_flows(self, merged: pd.DataFrame, window: int = 30):
        """Genera grafico Plotly: flussi IBIT giornalieri + rolling correlation.

        Args:
            merged: DataFrame da merge().
            window: finestra per la rolling correlation (default 30).

        Returns:
            plotly.graph_objects.Figure.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            _log.error("plotly non installato")
            return None

        theme = get_settings()["dashboard"]["theme"]

        corr_cols = self.rolling_correlations(merged, windows=[window])
        corr_df = corr_cols.get(f"{window}d", pd.DataFrame())

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "IBIT Net Flows (USD)",
                "BTC Price (USD)",
                f"Rolling {window}d Correlation: IBIT Flows → BTC Next-Day Return",
            ],
            vertical_spacing=0.07,
        )

        # Flussi IBIT
        if "ibit_flow" in merged.columns:
            colors = merged["ibit_flow"].apply(
                lambda v: theme["positive"] if (v or 0) >= 0 else theme["negative"]
            )
            fig.add_trace(
                go.Bar(
                    x=merged.index,
                    y=merged["ibit_flow"] / 1e6,
                    marker_color=colors,
                    name="IBIT Flow ($M)",
                ),
                row=1,
                col=1,
            )

        # Prezzo BTC
        if "btc_close" in merged.columns:
            fig.add_trace(
                go.Scatter(
                    x=merged.index,
                    y=merged["btc_close"],
                    line=dict(color=theme["neutral"], width=1.5),
                    name="BTC Price",
                ),
                row=2,
                col=1,
            )

        # Rolling correlation
        if not corr_df.empty and "ibit_flow_vs_btc_return" in corr_df.columns:
            corr_series = corr_df["ibit_flow_vs_btc_return"]
            fig.add_trace(
                go.Scatter(
                    x=corr_series.index,
                    y=corr_series,
                    line=dict(color=theme["positive"], width=1.5),
                    name=f"Corr {window}d",
                ),
                row=3,
                col=1,
            )
            # Linea zero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text"]),
            showlegend=True,
            height=800,
            title_text="IBIT ETF Flows & BTC Correlation",
        )
        fig.update_xaxes(gridcolor=theme["grid"])
        fig.update_yaxes(gridcolor=theme["grid"])

        return fig
