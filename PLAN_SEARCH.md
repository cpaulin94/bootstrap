# Piano di lavoro — campionamento dello spazio di ricerca + bug overlay "crimson star"

Data: 2026-08-25. Autore del piano: Opus. Implementatore: Sonnet.
Baseline: `uv run pytest` deve passare prima e dopo. Ogni numero qui sotto è stato
misurato su questo repo con i dati reali; il comando di riproduzione è incluso.

Due interventi **indipendenti**, implementabili e mergiabili separatamente:

- **Parte A** — nuovo campionatore dello spazio dei portafogli (`engine/search.py`).
- **Parte B** — bug: la stella cremisi non è confrontabile con la nuvola (`gui.py`).

La Parte B è un **bug di correttezza che invalida ciò che l'utente vede a schermo** e va
fatta per prima. La Parte A è un miglioramento di copertura.

---

# PARTE B — la stella cremisi cade fuori dalla nuvola

## B0. Sintomo riportato

Un portafoglio in library mostra come overlay (stella cremisi) `annualised_return_p50 ≈ 8.2%`.
Lanciando una search in un intorno strettissimo di quello stesso portafoglio (±0.1% per peso),
la nuvola risultante è centrata su `p50 ≈ 7.3%`, con `annualised_return_p1` molto diverso.
Ci si aspetterebbe la nuvola centrata **sulla** stella.

## B1. Diagnosi — causa dominante: disallineamento del `sim_seed`

Riproduzione: `uv run python` sullo script in §B6.1. Portafoglio `Very_interesting`,
spazio di ricerca `±0.1%` attorno ai suoi pesi, 400 portafogli, `n_sim=1000` (il default
di `cfg.N_SIMULATIONS`), `sim_seed=123`:

```
NUVOLA (400 portafogli entro ±0.1%, sim_seed=123)
  annualised_return_p50  media=0.0739  sd=0.00003
  annualised_return_p1   media=-0.0211 sd=0.0001
  volatility_10y         media=0.0405  sd=0.00002

STELLA con lo STESSO seed della nuvola (percorso corretto)
  p50=0.0739 (+0.2 sd dal centro)   p1=-0.0211 (-0.6 sd)   vol=0.0406 (+0.9 sd)   ✔

STELLA con seed DIVERSO (cache stantia / ramo run_bootstrap con random_seed=None)
  seed=7    : p50=0.0743 ( +9.7 sd)  p1=-0.0270 (-51.2 sd)  vol=0.0418 (+31.1 sd)
  seed=99   : p50=0.0747 (+18.3 sd)  p1=-0.0263 (-45.1 sd)  vol=0.0413 (+18.1 sd)
  seed=2024 : p50=0.0766 (+56.2 sd)  p1=-0.0304 (-80.8 sd)  vol=0.0425 (+48.0 sd)
```

Il meccanismo, e perché l'effetto è così violento:

1. Dal fix C3 dell'audit, **tutti** i candidati di una run condividono un unico `sim_seed`
   (common random numbers). Questo è corretto e va mantenuto: elimina il rumore Monte Carlo
   *relativo* fra candidati.
2. Conseguenza: in un intorno strettissimo la nuvola diventa **estremamente compatta**
   (sd di `p50` = 0.00003, cioè 0.003pp). Non c'è più rumore che la allarghi.
3. Ma il seed condiviso introduce un offset **di modo comune** su tutta la nuvola: quel
   singolo sorteggio di blocchi storici è più o meno fortunato. Misurato sullo stesso
   portafoglio al variare del solo `sim_seed`, con `n_sim=1000`:

   | metrica | sd fra seed | spread (25 seed) |
   |---|---|---|
   | `annualised_return_p50` | 0.0018 | **0.66 pp** |
   | `annualised_return_p1`  | 0.0040 | **1.96 pp** |
   | `volatility_10y`        | 0.0007 | 0.32 pp |

4. Quindi una stella calcolata con un seed diverso dalla nuvola cade a **decine di deviazioni
   standard** dal centro della nuvola. È esattamente il sintomo: "come se le due metriche
   fossero calcolate con criteri diversi". E `p1` è la metrica più colpita (spread 1.96pp),
   che è precisamente ciò che l'utente segnala come "molto diverso".

## B2. Diagnosi — il difetto strutturale che permette il disallineamento

`_compute_overlay_worker` ([gui.py:1173](gui.py:1173)) ha due rami:

- **ramo corretto** (`use_preloaded=True`, [gui.py:1192](gui.py:1192)): usa `self._last_data`
  e `rng=np.random.default_rng(self._last_sim_seed)` → stessa finestra dati, stesso sorteggio
  della nuvola. Verificato ✔ (+0.2 sd sopra).
- **ramo di fallback** ([gui.py:1228](gui.py:1228)): `run_bootstrap(..., random_seed=None)`
  → **seed casuale** e ricaricamento dati sui soli ticker del portafoglio.

Il fallback scatta quando `self._last_data is None` (nessuna run ancora fatta) o quando
`self._last_run_params != params`.

Il difetto è nella **chiave di validità della cache**. In `_recompute_selected_overlays`
([gui.py:1250](gui.py:1250)) la cache è considerata valida se:

```python
if cached and cached.get("params") == params:
    continue          # <-- non ricalcola
```

e `params` contiene **solo** ([gui.py:1315](gui.py:1315) e [gui.py:1156](gui.py:1156)):
`n_sim, horizon_years, block_size, date_start, date_end, independent`.

Non contiene:

- **`sim_seed`** — quindi una stella calcolata con seed casuale resta valida per sempre;
- **l'insieme di ticker dello spazio di ricerca** (`sorted_tickers`) — che determina la
  finestra storica per intersezione di date;
- **se sia stata calcolata dal ramo preloaded o dal fallback**.

Sequenza che produce il bug, tutta con parametri GUI immutati:

1. L'utente spunta l'overlay **prima** di lanciare una run. `_current_overlay_params()`
   costruisce il dict dai campi GUI; `_last_data is None` → ramo di fallback →
   stella con seed casuale e finestra dati dei soli suoi ticker.
2. L'utente lancia la search con gli stessi parametri GUI. `_last_run_params` viene
   costruito **dalle stesse variabili GUI**, quindi risulta `==` al dict cachato.
3. A fine run `_recompute_selected_overlays` ([gui.py:1547](gui.py:1547)) considera la
   cache valida e **non ricalcola**. La stella stantia viene disegnata sopra una nuvola
   calcolata con un altro seed e un'altra finestra storica.

## B3. Causa secondaria (minore, ma reale): finestra storica

`preload_returns` interseca le date di **tutti** i ticker dello spazio di ricerca, mentre
`run_bootstrap` interseca solo quelli del portafoglio. Su `Very_interesting` (12 ticker) vs
`search.csv` (15 ticker): 305 mesi → 292 mesi, 13 mesi in meno (BTOP50, WRDA, XDWS).
Effetto misurato a parità di seed: `p50 −0.07pp`, `p1 +0.40pp`, `max_dd_depth_p2 −1.48pp`.

Piccolo su `p50` ma **non** su `p1` e sul drawdown. Va corretto insieme a B2, non separatamente.

## B4. Interventi richiesti

### B4.1 — Estendere la chiave di cache dell'overlay (fix principale)

Introdurre una chiave esplicita che includa tutto ciò che cambia il risultato. Metterla in
una **funzione pura a livello di modulo**, non in un metodo della classe Tk, così è testabile
senza avviare Tkinter:

```python
# in bootstrap_gui/assets.py (o un nuovo bootstrap_gui/overlay.py)
def overlay_key(params: dict, sorted_tickers: list[str] | None,
                sim_seed: int | None) -> tuple:
    """Identità completa del contesto in cui un overlay è stato calcolato.

    Due overlay sono confrontabili con la stessa nuvola solo se questa chiave
    coincide: i parametri di simulazione NON bastano, perché finestra storica
    (derivata dall'insieme di ticker) e sorteggio Monte-Carlo (sim_seed) cambiano
    le metriche di piu' di quanto le cambi la differenza fra i portafogli vicini.
    """
    return (
        tuple(sorted(params.items())),
        tuple(sorted_tickers) if sorted_tickers else None,
        sim_seed,
    )
```

- Salvare `cache_entry["key"] = overlay_key(...)` in `_compute_overlay_worker`.
- In `_recompute_selected_overlays` e nel loop di disegno del grafico
  ([gui.py:1856](gui.py:1856)) confrontare `key`, non `params`.
- `_current_overlay_params` resta com'è; la chiave la si costruisce a parte da
  `self._last_sorted_tickers` e `self._last_sim_seed`.

Effetto: dopo ogni run, ogni stella calcolata in un contesto diverso viene **ricalcolata**.

### B4.2 — Non disegnare mai una stella non confrontabile

Se l'overlay è stato calcolato dal ramo di fallback (nessuna run disponibile), è comunque
utile all'utente ma **non è sovrapponibile a una nuvola**. Marcarlo:

- salvare `cache_entry["comparable"] = use_preloaded`;
- se `comparable=False`, disegnare la stella **vuota** (`marker_symbol="star-open"`,
  colore attenuato) e aggiungere in coda all'hover la riga:
  `⚠ calcolato su finestra dati e sorteggio diversi dalla nuvola — non confrontabile`;
- appena una run termina, la chiave B4.1 la invalida e viene ricalcolata piena.

Non rimuovere il ramo di fallback: serve quando l'utente vuole solo vedere dove sta un
portafoglio prima di lanciare la ricerca.

### B4.3 — Rendere il `sim_seed` visibile

Attualmente `sim_seed` è invisibile in GUI ma determina l'offset di modo comune dell'intera
nuvola. Aggiungere in Space Explorer:

- il valore effettivo di `sim_seed` nel pannello di log a inizio run (`_log(...)`), e
- una riga nel titolo/sottotitolo del grafico: `sim_seed=<n> · n_sim=<n>`.

Serve anche a rendere intelligibile il punto B5.

### B4.4 — Avviso su `n_sim` basso

Con `n_sim=1000` (default attuale di `cfg.N_SIMULATIONS`) l'incertezza di modo comune su
`annualised_return_p1` è ±2pp (tabella §B1). Il **livello assoluto** dell'intera nuvola non
è affidabile a quella risoluzione, anche se i confronti *fra* portafogli lo sono.

Aggiungere in `_start_run` un warning nel log quando `n_sim < 5000`:
`⚠ n_sim=1000: il livello assoluto delle metriche ha ±0.7pp (p50) / ±2pp (p1) di rumore di
sorteggio. I confronti fra portafogli restano validi (common random numbers).`

**Non** cambiare il default di `cfg.N_SIMULATIONS` in questo intervento: è una decisione di
costo/precisione che spetta all'utente (5x tempo di run). Segnalarla, basta.

## B5. Nota concettuale da preservare nel codice

Il common-random-numbers è **corretto** e non va rimosso. Ma va documentato con precisione,
perché la sua conseguenza è controintuitiva e ha generato questo bug:

> Il `sim_seed` condiviso azzera il rumore Monte-Carlo *relativo* fra candidati (ciò che
> serve per un fronte di Pareto onesto) ma **non** quello *assoluto*: l'intera nuvola trasla
> in blocco con la fortuna di quel singolo sorteggio. Confrontare fra loro punti di nuvole
> con seed diversi — o una stella con una nuvola — è quindi privo di significato.

Aggiungere questo paragrafo al docstring di `run_multi_streaming` (che già descrive il CRN)
e a `CLAUDE.md` sotto "Conventions".

## B6. Test richiesti (Parte B)

Tutti headless, nessun Tkinter.

1. `test_overlay_key_distinguishes_seed` — stessa `params`, `sim_seed` diverso → chiavi diverse.
2. `test_overlay_key_distinguishes_ticker_set` — stessa `params`, `sorted_tickers` diverso →
   chiavi diverse. Questo è il test di regressione del bug.
3. `test_overlay_key_stable_under_dict_order` — `params` con chiavi in ordine diverso →
   stessa chiave.
4. `test_star_matches_cloud_center_with_shared_seed` (regressione end-to-end, in
   `tests/test_runner.py`): spazio ±0.1% attorno a un portafoglio noto, ~100 candidati,
   `n_sim=500`, `sim_seed` fissato; `run_bootstrap_preloaded` sullo stesso `sim_seed` e sugli
   stessi dati da `on_data_ready` deve cadere entro **3 sd** dalla media della nuvola su
   `annualised_return_p50`, `annualised_return_p1`, `volatility_10y`.
5. `test_star_with_different_seed_is_far` — lo stesso ma con seed diverso deve cadere a
   **> 5 sd**. Documenta il perché del fix e fallisce se qualcuno rimuove il CRN.

### B6.1 Script di riproduzione (già verificato)

Salvato in scratchpad durante la diagnosi; ricrearlo in `tests/` come da §B6.4-5. Usa
`run_multi_streaming(..., on_data_ready=...)` per catturare `(sorted_tickers, matrice, sim_seed)`
e valutare la stella esattamente sugli stessi dati.

---

# PARTE A — campionamento dello spazio di ricerca

## A0. Problema

`sample_random_portfolios` ([engine/search.py:36](engine/search.py:36)) estrae uniforme nel
box e poi **normalizza**. Normalizzare è mediare: per il TCL la varianza della somma scala
come 1/k, quindi la nuvola collassa sul baricentro tanto più quanto più asset ci sono.
Misurato su `search.csv` reale (k=15, `lo=0`, `hi=0.30`, 20 000 campioni):

| k | max weight mediano | min weight mediano | raggio esplorato (frazione del massimo) |
|---|---|---|---|
| 5 | 0.318 | 0.064 | 0.237 |
| 10 | 0.182 | 0.014 | 0.184 |
| **15** | **0.125** | **0.006** | **0.150** |

Con `hi=0.30`, su 20 000 estrazioni il portafoglio più concentrato arriva a 0.263 e il
99° percentile si ferma a 0.181. Non si sta campionando il politopo: si campiona una pallina
attorno all'equipesato. È il difetto **M4** dell'audit, dove il tentativo di correggerlo con
Dirichlet fu correttamente rifiutato (acceptance ~0% sui box stretti reali). Questo piano lo
risolve senza reintrodurre quel problema.

Nota: il difetto morde solo con box **larghi** (il caso reale di `search.csv`). Con bound in
bande di 1-2pp il sampler attuale è di fatto corretto, perché lì il politopo *è* una pallina.

## A1. Guadagno atteso (misurato)

A parità di budget — 1500 portafogli, stessi dati, stesso `sim_seed`, `n_sim=300`:

| sampler | range CAGR_p50 | range vol 10y | miglior CAGR_p1 | punti Pareto |
|---|---|---|---|---|
| attuale | 1.91 pp | 2.19 pp | −2.23% | 15 |
| CDHR (uniforme esatto) | 2.74 pp | 3.13 pp | −1.83% | 20 |
| sparse (asset spenti) | 3.36 pp | 4.21 pp | **+1.62%** | 26 |
| misto | 3.53 pp | 3.90 pp | +0.60% | 21 |

Il dato che conta: **oggi il sampler non trova mai un portafoglio con CAGR al 1° percentile
positivo**. Esistono, il campionamento sparse li trova. Costo computazionale identico.

## A2. `sample_cdhr_portfolios` — uniforme esatto su simplesso ∩ box

Hit-and-run per coordinate (CDHR). Le direzioni `e_i − e_j` generano l'inviluppo affine del
simplesso, quindi la catena è irriducibile sul politopo e la sua distribuzione stazionaria è
**uniforme**. Nessun rigetto: si campiona direttamente l'intervallo feasible.

```python
def sample_cdhr_portfolios(space, n, rng, n_steps=None):
    k = len(space)
    lo = np.array([s["lo"] for s in space]); hi = np.array([s["hi"] for s in space])
    # infeasible -> stesso contratto di sample_random_portfolios: array vuoto + log.error
    if lo.sum() > 1 + 1e-9 or hi.sum() < 1 - 1e-9:
        return np.empty((0, k))
    if k == 1:
        return np.ones((n, 1))
    if n_steps is None:
        n_steps = max(200, 20 * k)

    r = 1.0 - lo.sum(); span = hi - lo
    W = np.tile(lo + r * span / span.sum(), (n, 1))   # punto interno feasible
    rows = np.arange(n)
    for _ in range(n_steps):
        i = rng.integers(0, k, n)
        j = (i + 1 + rng.integers(0, k - 1, n)) % k   # j != i, uniforme
        wi = W[rows, i]; wj = W[rows, j]
        t_lo = np.maximum(lo[i] - wi, wj - hi[j])
        t_hi = np.minimum(hi[i] - wi, wj - lo[j])
        t = t_lo + (t_hi - t_lo) * rng.random(n)
        W[rows, i] = wi + t
        W[rows, j] = wj - t
    return W
```

Note per l'implementazione:

- Il punto di partenza `lo + r*(hi-lo)/sum(hi-lo)` è **sempre** feasible quando lo spazio lo è
  (`r ≤ sum(hi-lo)` garantisce `w_i ≤ hi_i`). Non serve un LP.
- `n` catene indipendenti dallo stesso punto, `n_steps` passi ciascuna, tutto vettorizzato →
  rispetta la convenzione "vettorizza su n_sim, mai loop Python sui campioni". Il loop è sui
  passi (limitato, ~300), non sui campioni.
- Misurato: 20 000 campioni × k=15 in **0.7 s**, accettazione **100%**.
- Sul box stretto che uccise Dirichlet (bande 1-2pp): Dirichlet accetta lo 0.0067%,
  CDHR il 100%, con bound e `sum=1` verificati. **Questo chiude M4.**
- Errore di arrotondamento: dopo molti passi `sum` può derivare di ~1e-16. Ri-normalizzare
  **non** va bene (rompe i bound). Usare invece l'aggiornamento simmetrico sopra
  (`w_j -= t` con lo stesso `t`), che conserva la somma esattamente in floating point fino
  all'ultimo bit tranne cancellazioni trascurabili; assertare `atol=1e-9` nei test.

## A3. `sample_sparse_portfolios` — spegnere asset e cercare nelle sotto-configurazioni

```python
def sample_sparse_portfolios(space, n, rng, size_range=None):
```

1. **Eleggibili allo spegnimento**: solo gli asset con `lo == 0`. Un asset con `lo > 0` è
   obbligatorio e va **sempre** incluso nel supporto. Questo va rispettato o si generano
   portafogli che violano i bound.
2. **Cardinalità minima**: il supporto `S` deve poter arrivare a somma 1, cioè
   `sum(hi[S]) >= 1`. Con `hi = 0.30` uniforme servono almeno `ceil(1/0.30) = 4` asset attivi.
   Calcolarlo, non assumerlo: ordinare gli `hi` decrescenti e prendere il minimo prefisso che
   somma ≥ 1. **Bug da evitare**: una prima versione di questo test estraeva supporti di
   dimensione 3 e produceva pesi fino a 0.85, violando `hi=0.30`.
3. **Distribuzione della cardinalità**: uniforme su `[m_min, k]`. Non favorire i supporti
   piccoli: è la varietà di cardinalità a dare copertura.
4. **Pesi sul supporto**: richiamare il **core CDHR ristretto a `S`** (stessi `lo`/`hi` ma
   solo sulle colonne di `S`), non un Dirichlet con rigetto. Riusa il codice, è esatto, e
   gestisce nativamente il caso `lo > 0`.
5. Scrivere il risultato in una matrice `(n, k)` di zeri, riempiendo le colonne di `S`.

Raggruppare le estrazioni **per cardinalità** (`m` uguale → una sola chiamata CDHR
vettorizzata su tutte le righe con quella `m`), altrimenti si ricade in un loop Python per
campione — misurato 6 s per 20 000 campioni nella versione ingenua, contro <1 s raggruppata.

## A4. `sample_vertex_portfolios` — i veri spigoli del politopo

Per ogni riga, estrarre un obiettivo lineare casuale `c ~ N(0, I)` e risolvere
`max c·w` su simplesso ∩ box. La soluzione è esatta e greedy:

```python
w = lo.copy(); r = 1 - lo.sum()
for i in np.argsort(-c):
    add = min(hi[i] - lo[i], r); w[i] += add; r -= add
```

Dà i **vertici** del politopo — i portafogli estremi, dove tipicamente vive il fronte di
Pareto. Vettorizzabile con `argsort` per righe + `cumsum` + `clip`; non serve un loop Python
per riga.

## A5. `sample_mixed_portfolios` — il metodo da esporre all'utente

Composizione di default: **40% CDHR, 40% sparse, 20% vertici**. Costanti a livello di modulo
(`MIXED_WEIGHTS = (0.4, 0.4, 0.2)`), non numeri magici sparsi. Concatenare e **mescolare** le
righe con `rng.permutation`, altrimenti i tre blocchi arrivano in ordine ai worker e i
progressi a schermo mostrano tre regimi consecutivi anziché una nuvola che cresce uniforme.

## A6. Cablaggio

- `engine/search.py`: le tre nuove funzioni + `sample_mixed_portfolios`. Stessa firma di
  `sample_random_portfolios` (`space, n, rng -> (n, k)`) e **stesso contratto sui casi
  infeasible** (array vuoto, `log.error`, mai `np.empty` non inizializzato — vedi il commento
  esistente a [engine/search.py:104](engine/search.py:104)).
- `engine/runner.py:390` — aggiungere il ramo `elif method == "mixed":`. Nient'altro cambia.
- `engine/config.py:66` — `SEARCH_METHOD` può diventare `"mixed"`; documentare i valori
  ammessi (`"random" | "mixed" | "grid"`) nel commento.
- `gui.py:683` — terzo `ttk.Radiobutton`: `text="Mixed Search (consigliato)"`, `value="mixed"`.
  Portare il default di `self.search_mode` a `"mixed"`.
- **Non rimuovere** `"random"`: serve a riprodurre le run vecchie.

## A7. Test richiesti (Parte A) — `tests/test_search.py`

Parametrizzare 1-3 su tutti e quattro i sampler.

1. `sum(axis=1) == 1` a `atol=1e-9`.
2. Bound rispettati: `lo - 1e-9 <= w <= hi + 1e-9`.
3. Determinismo: stesso seed → stesso array (`assert_array_equal`).
4. **Regressione M4**: su uno spazio a bande strette (`lo=0.29, hi=0.31` e simili, k=15)
   CDHR restituisce `n` righe piene e valide. È il caso che fece revertare Dirichlet.
5. Spazio infeasible (`sum(lo) > 1`, e `sum(hi) < 1`): array vuoto, nessuna eccezione,
   nessuna riga di memoria non inizializzata.
6. **Uniformità di CDHR**: su k=3 con box larghi, confrontare la marginale del primo peso
   contro un campionamento per rigetto di riferimento (fattibile a k=3) con un test KS;
   `p > 0.01`. È il test che dimostra che CDHR fa ciò che dichiara.
7. **Regressione di copertura** (il motivo di tutto l'intervento): su k=15, `lo=0`, `hi=0.30`,
   `sample_mixed_portfolios` deve produrre `percentile(max_weight, 99) > 0.25`, dove
   `sample_random_portfolios` sullo stesso spazio dà 0.181. Asserire un numero, non "è meglio".
8. **Sparse — cardinalità**: nessuna riga con meno di `m_min` asset attivi; nessun asset con
   `lo > 0` mai spento. Test diretto sul bug descritto in §A3.2.
9. `sample_mixed_portfolios` con `n` piccolo (1, 2, 5) non deve crollare per via degli
   arrotondamenti delle quote 40/40/20.

## A8. Cosa NON fare in questo intervento

- **Non** toccare `sample_grid_portfolios`.
- **Non** rimuovere né modificare `sample_random_portfolios`: resta il metodo `"random"`.
- **Non** introdurre dipendenze nuove. Tutto è numpy.
- **Non** implementare il raffinamento a due fasi (giro esplorativo → perturbazione locale
  attorno al fronte di Pareto, stile NSGA-II). È il passo successivo giusto — in 15 dimensioni
  nessun campionamento "riempie" lo spazio, e con budget 100k copri quanto 2 punti per asse —
  ma tocca `run_multi_streaming` e il ciclo della GUI, ed è un intervento a parte.

## A9. Nota su `type_entropy`

`cfg.PARETO_METRICS` include `type_entropy` in massimizzazione. I portafogli sparse hanno
entropia bassa per costruzione: **non** domineranno il fronte, lo **estenderanno** verso il
lato "concentrato". È il comportamento voluto — non è un segno che il sampler sbagli, e non
va "corretto" bilanciando l'entropia dentro il sampler.

---

# Ordine di lavoro consigliato

1. Parte B (bug di correttezza) → test B6 → `uv run pytest`.
2. Parte A (§A2 CDHR → §A3 sparse → §A4 vertici → §A5 misto) → test A7 → `uv run pytest`.
3. Aggiornare `CLAUDE.md`: la sezione "Conventions" con la nota §B5, e la descrizione di
   `engine/search.py`. Aggiornare la riga **M4** nella tabella di stato di `AUDIT.md`
   ("Tried, reverted") con l'esito di questo lavoro.
