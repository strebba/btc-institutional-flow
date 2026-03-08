# ibit-gamma-tracker — Project Memory

## Architecture
- **Language**: Python 3.9+, Streamlit dashboard
- **DB**: SQLite at `data/structured_notes.db` (notes + barrier_levels + prices tables)
- **Packages**: `src/edgar`, `src/flows`, `src/gex`, `src/analytics`, `src/dashboard`
- **Config**: `config/settings.yaml` (loaded via `src/config.get_settings()`)
- **Tests**: 232 tests in `tests/`, run with `.venv/bin/pytest tests/ -v`
- **Launch**: `streamlit run src/dashboard/app.py`

## Key File Paths
- Dashboard: `src/dashboard/app.py`
- Charts: `src/dashboard/charts.py`
- GEX calc: `src/gex/gex_calculator.py`
- GEX client: `src/gex/deribit_client.py`
- Flows scraper: `src/flows/scraper.py`
- Price fetcher: `src/flows/price_fetcher.py`
- EDGAR DB: `src/edgar/structured_notes_db.py`

## Bugs Fixed (session 2026-03-08)
1. **`gex_to_dict()` key mismatch**: Added `total_net_gex` (raw USD) and `n_instruments`
   alongside existing `total_net_gex_m` / `num_strikes` in `gex_calculator.py`
2. **`fetch_and_store` missing method**: `app.py` called non-existent `PriceFetcher.fetch_and_store()`.
   Fixed to use `pf.fetch("BTC-USD")` + `pf.fetch("IBIT")` directly.
3. **Deribit timeout**: Changed from 15s to 30s in `deribit_client.py`

## Dashboard Tab Structure (current)
1. 🎯 Barrier Map — visual chart of BTC barrier levels + contextual alerts
2. 📊 GEX — gamma exposure profile, regime box, regime analysis
3. 💰 ETF Flows — IBIT flows, KPI, alert boxes, Granger expander
4. 🚦 Segnali — composite signal (GEX+Flows+Barriers) + backtest results
5. 🔍 EDGAR Monitor — structured notes KPI, filings table, event study

## Important Patterns
- `gex_to_dict()` returns both `total_net_gex` (raw USD) and `total_net_gex_m` (millions)
- `PriceFetcher` has no `fetch_and_store()` — use `fetch(ticker)` directly
- All Streamlit cache functions use `ttl=_REFRESH` (900s default)
- Composite signal logic is in `_composite_signal()` in `app.py`
- `barrier_map()` chart function added to `charts.py`
