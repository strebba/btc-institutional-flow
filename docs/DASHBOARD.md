# Modulo 5 — Dashboard Streamlit

## Avvio

```bash
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

## Struttura

```
src/dashboard/
├── app.py      # Orchestrazione Streamlit (5 tab, sidebar, cache, header)
└── charts.py   # Funzioni Plotly pure: DataFrame/dict → go.Figure
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

### Tab 1 — GEX (📊)

**Sinistra (2/3 larghezza):** Grafico a barre del profilo GEX per strike
- Barre verdi = GEX positivo (stabilizzante)
- Barre rosse = GEX negativo (destabilizzante)
- Linea tratteggiata = prezzo spot corrente

**Destra (1/3 larghezza):** Livelli chiave
- Call Wall (verde)
- Gamma Flip (blu)
- Spot (bianco)
- Put Wall (rosso)

**Sotto:** Metriche Gamma Flip, Max Pain, Put/Call OI ratio, N strumenti

### Tab 2 — ETF Flows (💸)

Tre pannelli sincronizzati sull'asse X:
1. **IBIT Flows** — barre verdi/rosse in M$
2. **BTC Price** — linea continua
3. **Correlazione rolling 30d** — tra flussi IBIT e rendimenti BTC

Riepilogo ultimi 30 giorni: flusso totale, giorni inflow, rendimento BTC.

### Tab 3 — Analytics (🔬)

**Granger Causality:**
- Heatmap p-values (verde = significativo p<0.05, rosso = non significativo)
- Assi: direzione (flows→returns, returns→flows) × lag (1-10 giorni)
- Expander con interpretazione testuale completa

**Regime Analysis:**
- Bar chart comparativo: return medio, volatilità, Sharpe per regime
- p-value del Welch t-test + etichetta "SIGNIFICATIVO" se p<0.05
- Correlazione media GEX ↔ BTC Vol

### Tab 4 — Backtest (📈)

- Tabella performance comparativa (GEX+Flows Strategy vs Buy & Hold BTC)
- Metriche: Delta Sharpe, giorni long, N trades
- Equity curve + rendimenti giornalieri (2 pannelli sincronizzati)
- Info box quando la strategia è flat (GEX storico non ancora accumulato)

### Tab 5 — EDGAR Barriers (🏛️)

- Tabella barriere attive con: tipo, emittente, prodotto, livello %, prezzo IBIT, prezzo BTC, scadenza, status
- Event Study CAR con confidence interval (quando ci sono eventi)
- Messaggio informativo quando nessun evento viene trovato

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
| `gex_profile(gex_by_strike, spot)` | list[dict], float | Bar chart GEX per strike |
| `gex_walls(snapshot_dict)` | dict | Livelli chiave (put wall, call wall, flip) |
| `flows_chart(merged_df)` | DataFrame | 3 pannelli: flows, BTC, correlazione |
| `granger_heatmap(granger_df)` | DataFrame | Heatmap p-values Granger |
| `regime_bars(regime_result)` | RegimeComparisonResult | Bar chart comparativo regimi |
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
```

### Con opzioni custom
```bash
streamlit run src/dashboard/app.py \
  --server.port 8080 \
  --server.headless true \
  --browser.gatherUsageStats false
```

### Docker (esempio)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e "."
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.headless", "true"]
```

### Streamlit Community Cloud
1. Push su GitHub
2. Vai su [share.streamlit.io](https://share.streamlit.io)
3. Seleziona repo, branch `main`, file `src/dashboard/app.py`
4. Aggiungi eventuali secrets in `Settings > Secrets`
