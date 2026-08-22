# Indicatore TradingView — GEX, put/call wall e gamma flip

Porta i livelli GEX calcolati da questo repo (Deribit → `src/gex/`) direttamente sul grafico
prezzo di BTC su TradingView, come indicatore Pine Script v6 in overlay.

## Perché lo script va *generato* e non semplicemente installato

Pine Script gira nella sandbox di TradingView: **non può fare chiamate HTTP**, non può leggere
un JSON né interrogare `/api/gex`. L'unico modo di avere il GEX su TradingView senza essere un
data provider ufficiale è **incorporare i livelli nel sorgente** e rigenerare lo script quando
i dati cambiano.

Da qui la struttura: `src/gex/pine_exporter.py` prende uno `GexSnapshot` e restituisce il
sorgente Pine con i valori già dentro. Due modi per ottenerlo:

```bash
make export-pine                       # → exports/btc_gex_tradingview.pine (fetch live Deribit)
python3 scripts/export_pine.py --from-db   # ultimo snapshot dal DB, nessuna rete
python3 scripts/export_pine.py --stdout    # sorgente su stdout

curl -s http://localhost:8000/api/gex/pine                 # stesso contenuto, text/plain
curl -s "http://localhost:8000/api/gex/pine?download=1" -O # come file .pine
```

## Installazione su TradingView

1. Apri il grafico di BTC (`BINANCE:BTCUSDT`, `COINBASE:BTCUSD`, `INDEX:BTCUSD`…).
2. **Pine Editor** (in basso) → *Open* → *New indicator*.
3. Incolla tutto il contenuto di `exports/btc_gex_tradingview.pine` sostituendo il template.
4. *Save* → dai un nome → **Add to chart**.
5. Per aggiornare i livelli: rigenera, incolla di nuovo, *Save*. L'indicatore resta sul grafico
   con le stesse impostazioni.

Su un grafico **IBIT** i livelli vanno riscalati: nelle impostazioni dell'indicatore imposta
*Scala prezzi* al ratio IBIT/BTC (il valore corrente è suggerito nel titolo dell'input, es.
`0.000566`).

## Cosa disegna

| Elemento | Significato |
|----------|-------------|
| **Gamma flip** (tratteggiata, gialla) | prezzo dove il GEX cumulativo cambia segno: sopra = dealer long gamma, sotto = short gamma |
| **Call wall** (verde) | strike con il massimo GEX positivo → resistenza meccanica |
| **Put wall** (rossa) | strike con il massimo \|GEX\| negativo → supporto meccanico |
| **Max pain** (punteggiata, grigia) | strike che minimizza il payoff totale delle opzioni |
| **Profilo call/put** (istogramma a destra) | per ogni strike: barra verde = call GEX, barra rossa = put GEX, larghezza ∝ \|GEX\| |
| **Step line storiche** | evoluzione giorno per giorno di flip/put wall/call wall da `gex_snapshots` |
| **Sfondo** | verde = prezzo sopra il flip (vol compressa), rosso = sotto (vol amplificata) |
| **Pannello in alto a destra** | regime live, regime dello snapshot, net GEX, distanze % dai livelli, put/call ratio, percentile GEX, età dei dati |

Le linee orizzontali sono lo **snapshot corrente**; le step line mostrano invece dove stavano
gli stessi livelli nei giorni passati (utile per vedere se un wall regge o si sposta).

## Lettura operativa

- **Prezzo sopra il gamma flip** → i dealer sono long gamma: l'hedging vende forza e compra
  debolezza, la volatilità realizzata tende a comprimersi e il prezzo a rimanere nel range
  put wall ↔ call wall.
- **Prezzo sotto il gamma flip** → dealer short gamma: l'hedging è pro-ciclico e amplifica i
  movimenti. È il regime in cui le rotture dei livelli tendono a estendersi.
- **Test del call wall** dall'alto verso il basso o del put wall dal basso: sono i punti dove il
  flusso di hedging è più denso, quindi dove il prezzo tende a rallentare.
- Il **percentile GEX** nel pannello contestualizza il net GEX rispetto agli ultimi 90 snapshot.

## Alert

Lo script espone quattro `alertcondition` (menu *Crea alert* → condizione = l'indicatore) e le
stesse condizioni via `alert()`:

- gamma flip attraversato al rialzo / al ribasso;
- prezzo entrato nella tolleranza (default 0.5%) del call wall o del put wall.

## Limiti da tenere presenti

- **I livelli invecchiano.** Il pannello mostra l'età dei dati e segnala `⚠ rigenera` oltre la
  soglia impostata (default 36 h). Il GEX si muove con OI ed espirazioni: rigenera almeno
  quotidianamente, idealmente dopo l'expiry delle 08:00 UTC.
- Le step line storiche esistono solo se `cron_gex.py` ha accumulato snapshot nel DB.
- Il profilo è limitato a 24 strike attorno allo spot (2 box per strike, limite di 500 box per
  indicatore su TradingView) e ai soli strike entro ±15% dallo spot.
- I livelli sono in **USD/BTC**: su qualsiasi altro sottostante serve il fattore di scala.
- Deribit copre la maggior parte dell'open interest sulle opzioni BTC ma non tutto: i wall sono
  una proxy del posizionamento aggregato, non la fotografia completa del mercato.

## Riferimenti nel codice

| File | Ruolo |
|------|-------|
| `src/gex/pine_exporter.py` | generazione del sorgente Pine (puro, senza I/O) |
| `src/api/routers/gex.py` → `GET /api/gex/pine` | stesso script servito come `text/plain` |
| `scripts/export_pine.py` | CLI, scrive `exports/btc_gex_tradingview.pine` |
| `tests/test_gex/test_pine_exporter.py` | test dell'exporter |

Per la teoria del GEX e il calcolo dei livelli vedi [`docs/GEX.md`](GEX.md).
