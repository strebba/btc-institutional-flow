"""ETF Flows endpoint."""
from __future__ import annotations

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.cache import asset_key, cache_get, cache_set
from src.api.helpers import AssetParam, ok, http_error

router = APIRouter(prefix="/api/flows", tags=["flows"])


@router.get("")
def get_flows(asset: AssetParam = "btc") -> JSONResponse:
    from src.assets import get_asset

    spec = get_asset(asset)
    cache_key = asset_key("flows", spec.key)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from src.flows.scraper import FarsideScraper
        from src.flows.correlation import FlowCorrelation
        from src.analytics.granger import GrangerAnalysis
        from src.api.data_pipeline import get_flow_context

        flow_ctx = get_flow_context(asset=spec.key)
        raw_flows = flow_ctx["raw"]
        merged = flow_ctx["merged_df"]
        df_pivot = FarsideScraper(asset=spec.key).to_dataframe(raw_flows)

        corr_eng = FlowCorrelation()

        if merged.empty:
            raise ValueError("Merge flussi/prezzi vuoto")

        stats = corr_eng.summary_stats(merged, asset=spec.key)
        roll_corrs = corr_eng.rolling_correlations(merged, windows=[30, 60, 90], asset=spec.key)

        granger_eng = GrangerAnalysis()
        granger_raw = granger_eng.run(
            merged, flow_col=spec.lead_flow_col, return_col=spec.return_col
        )
        granger_out: dict[str, list] = {}
        for direction, results in granger_raw.items():
            granger_out[direction] = [
                {"lag": r.lag, "f_stat": round(r.f_stat, 4),
                 "p_value": round(r.p_value, 6), "significant": r.significant}
                for r in results
            ]

        # Nomi di colonna dallo spec: per BTC restano btc_close, btc_vol_7d,
        # ibit_btc_ratio (contratto PTF-Dashboard); per ETH eth_close, ...
        ratio_col = f"{spec.lead_prefix}_{spec.prefix}_ratio"
        price_by_date: dict[str, float] = {}
        vol_by_date: dict[str, float] = {}
        ratio_by_date: dict[str, float] = {}
        total_flow_vals: dict[str, float] = {}
        if not merged.empty:
            for col, target in [
                (spec.price_col, price_by_date), (spec.vol_col, vol_by_date),
                (ratio_col, ratio_by_date), ("total_flow", total_flow_vals),
            ]:
                if col in merged.columns:
                    for idx, val in merged[col].dropna().items():
                        target[str(idx.date())] = float(val)

        all_etf_tickers = [tk for tk in df_pivot.columns if tk.lower() not in ("total", "date")]
        ticker_series: dict[str, dict[str, float]] = {}
        for tk in all_etf_tickers:
            ticker_series[tk] = {str(d.date()): float(v) for d, v in df_pivot[tk].dropna().tail(365).items()}

        primary_series = ticker_series.get(spec.lead_etf, {})
        if not primary_series:
            for tk in all_etf_tickers:
                if ticker_series.get(tk):
                    primary_series = ticker_series[tk]
                    break

        all_dates = sorted(set(primary_series) | set(total_flow_vals), reverse=False)[-365:]
        history: list[dict] = []
        primary_ticker = (
            spec.lead_etf if spec.lead_etf in ticker_series
            else (all_etf_tickers[0] if all_etf_tickers else None)
        )
        for d in all_dates:
            row: dict = {"date": d}
            if primary_ticker:
                row[f"{primary_ticker.lower()}_flow_usd"] = ticker_series.get(primary_ticker, {}).get(d)
            row["total_flow_usd"] = total_flow_vals.get(d)
            row[spec.price_col] = price_by_date.get(d)
            row[spec.vol_col] = vol_by_date.get(d)
            row[ratio_col] = ratio_by_date.get(d)
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
            "asset": spec.key,
            "summary": stats,
            "history": history,
            "rolling_correlations_latest": corr_latest,
            "granger": granger_out,
            "data_quality": flow_quality,
        })
        cache_set(cache_key, response)
        return response

    except Exception as exc:
        traceback.print_exc()
        raise http_error(f"Flows error: {exc}")
