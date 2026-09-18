# Modulo 5 — Dashboard Streamlit

## Avvio

```bash
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

## Struttura

```
src/dashboard/
├── app.py           # Orchestratore: carica GEX/flussi/barriere in session_state,
│                    # st.navigation(position="top") → 7 pagine, header, caption
├── app_pages/       # 7 thin wrapper st.Page (panoramica default, signals, gex,
│                    # flows, barrier_map, edgar, validation)
├── tabs/            # Contenuto delle pagine (funzioni _tab_*)
├── data_loader.py   # Funzioni @st.cache_data condivise (GEX, flows, barriers, backtest)
├── charts.py        # Funzioni Plotly pure: DataFrame/dict → go.Figure
├── components.py    # Design system (tape, eyebrow, hero, pillar_bars) in st.html
├── header.py        # KPI sempre visibili + banner regime
├── sidebar.py       # Filtri e refresh manuale
└── static/          # Font IBM Plex self-hosted (woff2)
```

La navigazione è **lazy**: solo la pagina attiva viene eseguita. Prima `st.tabs` era
eager e ogni load calcolava backtest, walk-forward, factor decomposition, sensitivity,
IC, Granger ed event study tutti insieme.

## Header — KPI sempre visibili

```
₿ ibit-gamma-tracker
─────────────────────────────────────────────────────────
BTC Spot: $84,200 │ GEX: +41.5M$ │ Put Wall: $75k (-11%) │ Call Wall: $90k (+7%) │ BTC Return: -1.2%
┌─────────────────────────────┐
│ Regime: POSITIVE GAMMA      │
└─────────────────────────────┘
⚠️ NEAR CALL_WALL: spot entro 2% dal call wall $75,000
```

## Pagine e contenuto

### Panoramica (default)

Answer-first: `tape` di stato → `hero` del CompositeSignal (numero grande colorato)
+ `pillar_bars` → posizionamento GEX (`gex_walls`) → flussi/derivati → callout
"prossimo trigger".

### Segnali

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

### GEX

**Sinistra (2/3 larghezza):** Grafico a barre del profilo GEX per strike
- Barre verdi = GEX positivo (stabilizzante)
- Barre rosse = GEX negativo (destabilizzante)
- Linea tratteggiata = prezzo spot corrente

**Destra (1/3 larghezza):** Livelli chiave — Call Wall (verde), Gamma Flip (blu),
Spot (bianco), Put Wall (rosso)

**Sotto:** Metriche Gamma Flip, Max Pain, Put/Call OI ratio, N strumenti +
**Regime Analysis** (bar chart return/vol/Sharpe per regime, Welch t-test).

### ETF Flows

Tre pannelli sincronizzati sull'asse X:
1. **IBIT Flows** — barre verdi/rosse in M$
2. **BTC Price** — linea continua
3. **Correlazione rolling 30d** — tra flussi IBIT e rendimenti BTC

Riepilogo ultimi 30 giorni + expander **Granger Causality** (heatmap p-values
direzione × lag, con interpretazione testuale).

### Barrier Map

Mappa visuale dei barrier level delle note strutturate IBIT sul prezzo BTC:
- Barriere knock-in (rosse), autocall (verdi), buffer (blu)
- Linea spot corrente e distanza % dalla barriera più vicina
- Alert contestuale: < 3% (rosso), 3-8% (giallo), > 8% (verde)
- Confluenza barriere↔GEX (`barrier_gex_confluence_chart`)

### EDGAR

- KPI note strutturate + tabella filing con tipo, emittente, prodotto, livello %, prezzo
  IBIT, prezzo BTC, scadenza, status
- Event Study CAR con confidence interval (quando ci sono eventi)
- Drill-down per singola nota disponibile anche via API: `/api/notes/by-url`

### Validation

- **Information Coefficient** (Spearman IC, rolling 60gg, t-stat, IR, null model,
  alpha decay)
- **Walk-Forward** validation rolling train→test
- **Factor Decomposition** OLS alpha/beta
- **Parameter Sensitivity** ±20%

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
| `barrier_gex_confluence_chart(...)` | barriers, snapshot | Confluenza barriere ↔ GEX |
| `flows_chart(merged_df)` | DataFrame | 3 pannelli: flows, BTC, correlazione |
| `flows_stacked_chart(merged_df, etf_tickers)` | DataFrame, list | Flussi multi-ETF impilati |
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

## `components.py` — design system

Componenti HTML in `st.html` con classi `wx-` (stile Desk Note), senza toccare i widget
nativi Streamlit:

| Funzione | Scopo |
|----------|-------|
| `inject_style()` | Inietta il CSS una volta in `app.py` |
| `tape(text)` | Riga di stato monospace (ticker tape) |
| `eyebrow(text)` | Label neon maiuscola sopra un numero |
| `hero(score, signal, caption)` | Numero grande colorato per il CompositeSignal |
| `pillar_bars(pillars)` | Barre dei 4 pilastri |

## Tema

Il tema dell'app è **nativo Streamlit** in `.streamlit/config.toml` (nero `#000` +
neon `#00FF9D`), con font **IBM Plex self-hosted** in `src/dashboard/static/`
(serviti via `server.enableStaticServing=true` + `[[theme.fontFaces]]`).
Nessuna dipendenza da fonts.gstatic.com.

I colori dei grafici Plotly restano in `config/settings.yaml → dashboard.theme`
(allineati al tema nativo):

```yaml
dashboard:
  refresh_interval_s: 900
  theme:
    background: "#000000"   # sfondo nero
    text:       "#FFFFFF"   # testo bianco
    grid:       "#2a2a3e"   # griglia
    positive:   "#00FF9D"   # verde Wagmi Lab
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
