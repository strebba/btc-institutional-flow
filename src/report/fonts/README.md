# Font incorporati nel Desk Note

Sottoinsiemi **latin** di [IBM Plex](https://github.com/IBM/plex), scaricati da
Google Fonts e incorporati nella pagina come data URI da `src/report/renderer.py`.

| File | Uso |
|------|-----|
| `IBMPlexSans-var-latin.woff2` | Titoli e corpo (variabile, pesi 400–600) |
| `IBMPlexMono-400-latin.woff2` | Striscia di contesto e didascalie |
| `IBMPlexMono-500-latin.woff2` | Didascalie sotto il numero grande |
| `IBMPlexMono-600-latin.woff2` | Occhiello, numeri grandi, indice di pagina |

## Perché sono nel repo e non caricati da CDN

Le card diventano PNG pubblicati con il marchio sopra: la tipografia non può
cambiare a seconda che il container abbia o meno rete verso `fonts.gstatic.com`.
Incorporandoli, la stessa edizione rende identica in locale, in CI e su DO — e
la pagina resta autoconsistente, senza richieste esterne.

Il sottoinsieme latin copre gli accenti italiani (`è à ù ì ò`) e il segno meno
tipografico `−` (U+2212) usato da `src/report/formatting.py`.

## Licenza

IBM Plex è distribuito con [SIL Open Font License 1.1](https://github.com/IBM/plex/blob/master/LICENSE.txt),
che permette la ridistribuzione anche incorporata. Copyright © IBM Corp.
