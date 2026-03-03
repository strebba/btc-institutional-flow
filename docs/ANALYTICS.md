# Modulo 4 — Statistical Analysis

## Panoramica

Quattro test statistici per validare (o confutare) la teoria di Hayes sul legame tra dealer hedging, flussi ETF e prezzo BTC.

---

## 1. Granger Causality (`granger.py`)

**Domanda:** *I flussi IBIT precedono i rendimenti BTC? O è il contrario?*

### Metodologia

Il test di Granger verifica se aggiungere i valori passati della serie X migliora la previsione di Y rispetto a un modello che usa solo i valori passati di Y.

- **H1:** `flows → returns` — i flussi IBIT predicono i rendimenti BTC
- **H2:** `returns → flows` — i rendimenti BTC predicono i flussi IBIT

Implementazione: `statsmodels.tsa.stattools.grangercausalitytests`, F-test (SSR), lag 1–10.

**Preprocessing:**
1. Normalizzazione flussi in miliardi (stessa scala dei rendimenti %)
2. Test ADF (Augmented Dickey-Fuller) per stazionarietà
3. Differenziazione se non stazionaria

### Risultati con dati reali

```
Direzione: flows→returns
  ✓ SIGNIFICATIVO: p=0.028 al lag 5 (F=2.57)
                   p=0.029 al lag 6 (F=2.40)
                   p=0.049 al lag 7 (F=2.06)
  → I flussi IBIT predicono i rendimenti BTC con anticipo di 5-7 giorni

Direzione: returns→flows
  ✗ Non significativo (p > 0.05 per tutti i lag 1-10)
  → I prezzi BTC non predicono i flussi IBIT
```

**Interpretazione operativa:** La causalità è unidirezionale. I forti afflussi verso IBIT tendono ad anticipare rialzi di BTC di circa una settimana lavorativa. Non è un feedback loop: i prezzi non "attirano" flussi in modo statisticamente rilevante.

### API

```python
from src.analytics.granger import GrangerAnalysis

analyzer = GrangerAnalysis()
results  = analyzer.run(merged_df, max_lags=10)
# results = {
#   "flows→returns": [GrangerResult(lag=1, p=0.09, ...), ...],
#   "returns→flows": [GrangerResult(lag=1, p=0.21, ...), ...],
# }

print(analyzer.interpret(results))   # testo narrativo
df = analyzer.to_dataframe(results)  # DataFrame per export
```

---

## 2. Event Study (`event_study.py`)

**Domanda:** *Il prezzo BTC si comporta in modo anomalo quando si avvicina a un barrier level?*

### Metodologia

Analisi CAR (Cumulative Abnormal Returns) intorno alle date in cui BTC è entro ±2% di un barrier level.

```
AR_t  = R_t − rolling_mean(R, 30)     # rendimento anormale giornaliero
CAR_t = Σ AR_τ  per τ da -window a t  # cumulativo [-5, +5] giorni
```

Test: t-test a un campione su `H0: CAR = 0` al giorno +window.

**Confidence interval 95%:**
```
CI = CAR_mean ± 1.96 × (std / √n)
```

### Interpretazione dei risultati

| CAR significativamente > 0 | I prezzi salgono anomalamente vicino al barrier | Supporto meccanico (dealer copre delta) |
|---|---|---|
| CAR significativamente < 0 | I prezzi scendono anomalamente vicino al barrier | Pressione di vendita meccanica |
| CAR ≈ 0 (non significativo) | Nessun effetto rilevabile | Barrier non ancora testato o dati insufficienti |

### Limitazione attuale

Con i dati del DB attuale (10 barriere, tutte con `level_price_btc = None` o prezzi non ancora raggiunti dal BTC nell'ultimo anno), l'event study restituisce 0 eventi. Il modulo funziona correttamente e produrrà risultati man mano che:
1. Il parser EDGAR estrae più prezzi BTC assoluti
2. Il BTC si avvicina ai livelli barrier storici

### API

```python
from src.analytics.event_study import EventStudy

study = EventStudy()

# Da barriere EDGAR
results = study.run(barriers, btc_prices_df)

# Da livelli arbitrari (round numbers, livelli tecnici)
result = study.run_on_price_levels([80000, 90000, 100000], "round_numbers", btc_prices_df)

for r in results:
    print(f"{r.barrier_type}: n={r.n_events}, CAR={r.car_mean:.4f}, p={r.p_value:.4f}")
```

---

## 3. Regime Analysis (`regime_analysis.py`)

**Domanda:** *BTC performa meglio in positive gamma o negative gamma?*

### Metodologia

1. Classifica ogni giorno come `positive_gamma`, `negative_gamma` o `neutral` in base al GEX
2. Calcola statistiche condizionali per ciascun regime
3. Welch t-test (varianze diseguali) sulla differenza di rendimenti medi

```python
t_stat, p_value = scipy.stats.ttest_ind(
    pos_gamma_returns,
    neg_gamma_returns,
    equal_var=False,   # Welch
)
```

**Metriche per regime:**
- `mean_return` — rendimento medio giornaliero nel regime
- `std_return` — deviazione standard
- `mean_vol` — volatilità realizzata media (7d rolling)
- `sharpe` — `mean_return / std_return × √252`
- `cum_return` — rendimento cumulativo nel regime

**Correlazione rolling GEX ↔ volatilità:**
```python
rolling_corr = df[["_gex", "btc_vol_7d"]].rolling(30).corr()
```
Se negativa: più alto il GEX, più bassa la volatilità → conferma la teoria.

### Limitazione attuale

Il sistema ha solo lo snapshot GEX odierno (un singolo punto). Per un'analisi regime completa servono almeno 30-60 snapshot storici giornalieri. Il DB accumulerà questi dati nel tempo con esecuzioni periodiche di `run_gex.py`.

### API

```python
from src.analytics.regime_analysis import RegimeAnalysis

analyzer = RegimeAnalysis()

# Costruisci GEX series da snapshot storici
gex_series = analyzer.build_gex_series(snapshots_list)

# Analisi
result = analyzer.analyze(merged_df, gex_series)

print(result.interpretation)
print(f"p-value: {result.p_value:.4f}")
print(f"Significativo: {result.significant}")
```

---

## 4. Backtest (`backtest.py`)

**Domanda:** *Una strategia basata su GEX + flussi ETF batte il buy-and-hold?*

### Regole di trading

```
LONG   se: GEX > 0
          AND ibit_flow_3d > +100M$
          AND nessuna barriera attiva entro ±5% dal prezzo corrente

SHORT  se: GEX < 0
          AND ibit_flow_3d < -200M$

FLAT   altrimenti
```

**Lag di 1 giorno:** i segnali sono applicati il giorno successivo alla loro generazione per evitare look-ahead bias.

```python
signals_lagged = signals.shift(1).fillna(0)
strategy_returns = btc_returns × signals_lagged
```

### Metriche calcolate

| Metrica | Formula |
|---------|---------|
| Total Return | `equity_curve[-1] - 1` |
| Annualized Return | `(1 + total_ret)^(252/n_days) - 1` |
| Sharpe Ratio | `annualized_return / (std_daily × √252)` |
| Max Drawdown | `min((equity - cummax(equity)) / cummax(equity))` |
| Win Rate | `n_positive_days / n_total_days` |
| Profit Factor | `sum(positive_returns) / abs(sum(negative_returns))` |
| N Trades | `count(signal_changes)` |

### Interpretazione risultati

| Indicatore | Ottimo | Buono | Insufficiente |
|-----------|--------|-------|---------------|
| Delta Sharpe vs B&H | > +1.0 | +0.3 a +1.0 | < 0 |
| Max Drawdown | > -15% | -15% a -30% | < -30% |
| Win Rate | > 55% | 50-55% | < 50% |
| Profit Factor | > 2.0 | 1.5-2.0 | < 1.5 |

### Nota sul backtest attuale

La strategia è attualmente **flat su tutto il periodo** perché:
1. Non ci sono dati GEX storici — solo lo snapshot odierno
2. Le stime di flusso da yfinance (fallback) sono approssimative

**Con dati storici reali** (GEX accumulato + flussi Farside reali) la strategia produrrà segnali long/short. Il confronto con il Buy & Hold BTC (-28% nell'ultimo anno con max drawdown -52%) diventerà significativo.

### API

```python
from src.analytics.backtest import Backtest

bt = Backtest()

# Esegui backtest
results = bt.run(merged_df, gex_series=gex_series, active_barriers=barriers)

# Tabella comparativa
table = bt.summary_table(results)
print(table)

# Equity curve Plotly
fig = bt.plot(results)
fig.show()
```

---

## Script CLI

```bash
# Tutti gli analytics
python scripts/run_analytics.py

# Singoli moduli
python scripts/run_analytics.py --granger
python scripts/run_analytics.py --regime
python scripts/run_analytics.py --events
python scripts/run_analytics.py --backtest
```
