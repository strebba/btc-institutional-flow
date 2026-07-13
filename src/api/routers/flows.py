"""ETF Flows endpoint."""
from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.cache import cache_get, cache_set
from src.api.helpers import ok, http_error

router = APIRouter(prefix="/api/flows", tags=["flows"])


@router.get("")
def get_flows() -> JSONResponse:
    cached = cache_get("flows")
    if cached is not None:
        return cached

    try:
        from src.flows.scraper import FarsideScraper
        from src.flows.price_fetcher import PriceFetcher
        from src.flows.correlation import FlowCorrelation
        from src.analytics.granger import GrangerAnalysis

        scraper = FarsideScraper()
        raw_flows = scraper.fetch()
        agg_flows = scraper.aggregate(raw_flows)
        df_pivot = scraper.to_dataframe(raw_flows)

        fetcher = PriceFetcher()
        prices = fetcher.get_all_prices()

        corr_eng = FlowCorrelation()
        merged = corr_eng.merge(agg_flows, prices)

        if merged.empty:
            raise ValueError("Merge flussi/prezzi vuoto")

        stats = corr_eng.summary_stats(merged)
        roll_corrs = corr_eng.rolling_correlations(merged, windows=[30, 60, 90])

        granger_eng = GrangerAnalysis()
        granger_raw = granger_eng.run(merged)
        granger_out: dict[str, list] = {}
        for direction, results in granger_raw.items():
            granger_out[direction] = [
                {"lag": r.lag, "f_stat": round(r.f_stat, 4),
                 "p_value": round(r.p_value, 6), "significant": r.significant}
                for r in results
            ]

        btc_prices: dict[str, float] = {}
        btc_vols: dict[str, float] = {}
        ibit_btc_vals: dict[str, float] = {}
        total_flow_vals: dict[str, float] = {}
        if not merged.empty:
            for col, target in [
                ("btc_close", btc_prices), ("btc_vol_7d", btc_vols),
                ("ibit_btc_ratio", ibit_btc_vals), ("total_flow", total_flow_vals),
            ]:
                if col in merged.columns:
                    for idx, val in merged[col].dropna().items():
                        target[str(idx.date())] = float(val)

        all_etf_tickers = [tk for tk in df_pivot.columns if tk.lower() not in ("total", "date")]
        ticker_series: dict[str, dict[str, float]] = {}
        for tk in all_etf_tickers:
            ticker_series[tk] = {str(d.date()): float(v) for d, v in df_pivot[tk].dropna().tail(365).items()}

        primary_series = ticker_series.get("IBIT", {})
        if not primary_series:
            for tk in all_etf_tickers:
                if ticker_series.get(tk):
                    primary_series = ticker_series[tk]
                    break

        all_dates = sorted(set(primary_series) | set(total_flow_vals), reverse=False)[-365:]
        history: list[dict] = []
        primary_ticker = "IBIT" if "IBIT" in ticker_series else (all_etf_tickers[0] if all_etf_tickers else None)
        for d in all_dates:
            row: dict = {"date": d}
            if primary_ticker:
                row[f"{primary_ticker.lower()}_flow_usd"] = ticker_series.get(primary_ticker, {}).get(d)
            row["total_flow_usd"] = total_flow_vals.get(d)
            row["btc_close"] = btc_prices.get(d)
            row["btc_vol_7d"] = btc_vols.get(d)
            row["ibit_btc_ratio"] = ibit_btc_vals.get(d)
            for tk in all_etf_tickers:
                if tk != primary_ticker:
                    row[f"{tk.lower()}_flow_usd"] = ticker_series.get(tk, {}).get(d)
            history.append(row)

        corr_latest: dict[str, dict] = {}
        for window_key, corr_df in roll_corrs.items():
            last = corr_df.dropna(how="all")
            if not last.empty:
                row = last.iloc[-1].to_dict()
                corr_latest[window_key] = {k: round(float(v), 4) if v == v else None for k, v in row.items()}

        source_counts: dict[str, int] = {}
        for f in raw_flows:
            source_counts[f.source] = source_counts.get(f.source, 0) + 1
        dominant_source = max(source_counts, key=source_counts.get) if source_counts else "unknown"
        is_estimate = dominant_source.startswith("yfinance")
        flow_quality = {
            "dominant_source": dominant_source,
            "source_breakdown": source_counts,
            "quality_label": "low_estimate" if is_estimate else "ok",
            "is_estimate": is_estimate,
        }

        response = ok({
            "summary": stats,
            "history": history,
            "rolling_correlations_latest": corr_latest,
            "granger": granger_out,
            "data_quality": flow_quality,
        })
        cache_set("flows", response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Flows error: {exc}")
