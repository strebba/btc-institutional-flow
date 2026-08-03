# Modulo 5 — Dashboard Streamlit

## Avvio

```bash
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

## Struttura

```
src/dashboard/
├── app.py           # Orchestrazione Streamlit (tab, sidebar, cache, header)
├── data_loader.py   # Funzioni @st.cache_data condivise (GEX, flows, barriers, backtest)
├── charts.py        # Funzioni Plotly pure: DataFrame/dict → go.Figure
├── header.py        # KPI sempre visibili + banner regime
├── sidebar.py       # Filtri e refresh manuale
├── tabs/            # 6 moduli: barrier_map, gex, flows, signals, edgar, validation
└── static/style.css # CSS custom (tema Wagmi Lab)
```

## Tab e contenuto

### Header — KPI sempre visibili

```
₿ ibit-gamma-tracker
─────────────────────────────────────────────────────────
BTC Spot: $84,200 │ GEX: +41.5M$ │ Put Wall: $75k (-11%) │ Call Wall: $90k (+7%) │ BTC Return: -1.2%
┌─────────────────────────────┐
│ Regime: POSITIVE GAMMA      │
└─────────────────────────────┘
⚠️ NEAR CALL_WALL: spot entro 2% dal call wall $75,000
```

### Tab 1 — Barrier Map (🎯)

Mappa visuale dei barrier level delle note strutturate IBIT sul prezzo BTC:
- Barriere knock-in (rosse), autocall (verdi), buffer (blu)
- Linea spot corrente e distanza % dalla barriera più vicina
- Alert contestuale: < 3% (rosso), 3-8% (giallo), > 8% (verde)

### Tab 2 — GEX (📊)

**Sinistra (2/3 larghezza):** Grafico a barre del profilo GEX per strike
- Barre verdi = GEX positivo (stabilizzante)
- Barre rosse = GEX negativo (destabilizzante)
- Linea tratteggiata = prezzo spot corrente

**Destra (1/3 larghezza):** Livelli chiave — Call Wall (verde), Gamma Flip (blu),
Spot (bianco), Put Wall (rosso)

**Sotto:** Metriche Gamma Flip, Max Pain, Put/Call OI ratio, N strumenti +
**Regime Analysis** (bar chart return/vol/Sharpe per regime, Welch t-test).

### Tab 3 — ETF Flows (💰)

Tre pannelli sincronizzati sull'asse X:
1. **IBIT Flows** — barre verdi/rosse in M$
2. **BTC Price** — linea continua
3. **Correlazione rolling 30d** — tra flussi IBIT e rendimenti BTC

Riepilogo ultimi 30 giorni + expander **Granger Causality** (heatmap p-values
direzione × lag, con interpretazione testuale).

### Tab 4 — Segnali (🚦)

Segnale composito a **4 pilastri** (vedi `docs/ANALYTICS.md` §5), calcolato dalla stessa
`CompositeSignal` esposta da `/api/signals` (unica fonte di verità):

- **Gauge top-level** del punteggio composito 0-100 + banner regime
  (🟢 LONG / 🟡 CAUTION / 🔴 RISK_OFF)
- **4 sotto-gauge** dei pilastri (GEX, Barrier, ETF Flows, Macro) con peso effettivo
- **Tabella leggibile** con score, peso e "lettura" testuale di ciascun pilastro
- Expander "Come viene calcolato il segnale" (spiegazione dei 4 pilastri)
- **Backtest** della strategia a 4 pilastri vs Buy & Hold BTC: equity curve, Sharpe,
  max drawdown, win rate

> Il pilastro **Macro** richiede CoinGlass: se la chiave non è configurata in locale,
> appare "n/d" e i pesi si riscalano sugli altri pilastri.

### Tab 5 — EDGAR Monitor (🔍)

- KPI note strutturate + tabella filing con tipo, emittente, prodotto, livello %, prezzo
  IBIT, prezzo BTC, scadenza, status
- Event Study CAR con confidence interval (quando ci sono eventi)
- Drill-down per singola nota disponibile anche via API: `/api/notes/by-url`

## Cache e performance

Tutti i dati sono cachati con `@st.cache_data(ttl=900)` (15 minuti, configurabile):

```python
@st.cache_data(ttl=_REFRESH, show_spinner=False)
def load_gex() -> tuple[dict, list[dict]]:
    ...
```

**Refresh manuale:** bottone nella sidebar svuota tutti i cache e ricarica.

Il GEX richiede ~2 minuti per il fetch di 948 opzioni — appare uno spinner dedicato.

## `charts.py` — Funzioni disponibili

Tutte le funzioni sono pure (nessun effetto collaterale) e restituiscono `go.Figure`:

| Funzione | Input | Output |
|----------|-------|--------|
| `barrier_map(barriers, spot_price)` | list[dict], float | Mappa barrier level vs spot |
| `gex_profile(gex_by_strike, spot)` | list[dict], float | Bar chart GEX per strike |
| `gex_walls(snapshot_dict)` | dict | Livelli chiave (put wall, call wall, flip) |
| `flows_chart(merged_df)` | DataFrame | 3 pannelli: flows, BTC, correlazione |
| `granger_heatmap(granger_df)` | DataFrame | Heatmap p-values Granger |
| `regime_bars(regime_result)` | RegimeComparisonResult | Bar chart comparativo regimi |
| `composite_gauge(score, signal)` | float, str | Gauge top-level del segnale composito |
| `pillar_gauges(pillars)` | list[dict] | 4 sotto-gauge dei pilastri |
| `backtest_equity(results)` | dict[str, BacktestMetrics] | Equity curve + daily returns |
| `event_study_car(event_results)` | list[EventStudyResult] | CAR ± CI per tipo barriera |

**Uso standalone (fuori Streamlit):**
```python
from src.dashboard.charts import flows_chart
fig = flows_chart(merged_df)
fig.show()          # browser
fig.write_html("output/flows.html")
fig.write_image("output/flows.png")
```

## Tema

Configurato in `config/settings.yaml`:

```yaml
dashboard:
  refresh_interval_s: 900
  theme:
    background: "#1a1a2e"   # sfondo scuro navy
    text:       "#ffffff"   # testo bianco
    grid:       "#2a2a3e"   # griglia
    positive:   "#00ff88"   # verde neon
    negative:   "#ff4444"   # rosso
    neutral:    "#4488ff"   # blu
```

## Deployment

### Locale (sviluppo)
```bash
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

### Produzione — DO App Platform (container unico)

La dashboard NON si deploya da sola: gira nello stesso container del backend FastAPI,
esposta pubblicamente da **nginx** che fa da reverse proxy.

```
supervisord (container DO, http_port 8080)
├── nginx :8080      → pubblico: /api/* → FastAPI, /* → Streamlit
├── uvicorn :8000    → FastAPI (solo loopback)
└── streamlit :8501  → dashboard (solo loopback)
```

- URL pubblico: `https://btc-institutional-flow-tpw9m.ondigitalocean.app/`
- Config processi: `supervisord.conf` · proxy: `nginx.conf` · spec: `.do/app.yaml`
- Replica locale completa: `docker compose up -d --build` → http://localhost:8080
- Config Streamlit: `.streamlit/config.toml` (tema Wagmi Lab, headless)

### Embed in iframe (sito Wix wagmi-lab.com)

nginx rimuove `X-Frame-Options` (hardcoded da Streamlit, non disattivabile via config)
e imposta `Content-Security-Policy: frame-ancestors https://www.wagmi-lab.com`.

```html
<iframe src="https://btc-institutional-flow-tpw9m.ondigitalocean.app/?embed=true"
        style="width:100%; height:800px; border:none;"
        title="BTC Institutional Flow"></iframe>
```

`?embed=true` nasconde la toolbar Streamlit e riduce il padding.
