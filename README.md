# Bootstrap Portfolio Simulator

A Monte-Carlo bootstrap framework for evaluating and optimising multi-asset portfolios.
Simulate thousands of possible futures, compute risk/return metrics, and find Pareto-optimal allocations.

---

## Repository structure

```
bootstrap/
│
├── engine/                        # Python package — all core logic lives here
│   ├── __init__.py                #   public API re-exports
│   ├── config.py                  #   single source of truth for ALL settings
│   ├── data.py                    #   portfolio CSV & return-data loading
│   ├── simulation.py              #   vectorised Monte-Carlo bootstrap
│   ├── metrics.py                 #   weight-based (Shannon entropy) + simulation-based metrics
│   ├── pareto.py                  #   N-dimensional Pareto frontier
│   ├── search.py                  #   random / grid portfolio samplers
│   └── runner.py                  #   single + multi bootstrap orchestrators
│
├── scripts/                       # Thin CLI entry points
│   ├── preprocess.py              #   raw Curvo data → standardised CSVs
│   ├── run_single.py              #   evaluate one portfolio
│   └── run_multi.py               #   mass search (random / grid)
│
├── notebooks/                     # Interactive analysis
│   ├── single_bootstrap.ipynb     #   single portfolio plots
│   └── multi_bootstrap.ipynb      #   search results, Pareto, current-portfolio overlay
│
├── data/
│   ├── TER_table.csv              # Annual TER per ticker
│   ├── raw_curvo/                 # Raw CSV exports from Curvo
│   └── standard/                  # Preprocessed monthly returns (auto-generated)
│
├── results/                       # Multi-bootstrap output (auto-generated)
│   └── multi_bootstrap_results.csv
│
├── sample_portfolio.csv           # Example portfolio definition
├── search.csv                     # Search space (tickers + weight ranges)
├── requirements.txt
└── README.md
```

### Design principles

- **`engine/` is a plain Python package** — no CLI, no I/O side-effects, no prints
  in the hot path.  It can be imported by notebooks, CLI scripts, or a future
  Tkinter / web GUI without changes.
- **One config file** — `engine/config.py` merges the old `config.py` + `run_config.py`.
  Override values there or pass them as function arguments.
- **Scripts are thin wrappers** — they parse CLI args and call `engine` functions.
- **Notebooks are consumers** — they import from `engine` and focus on visualisation.

---

## Quick start

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Preprocess raw data

```bash
python scripts/preprocess.py
```

Reads `data/raw_curvo/*.csv`, adjusts for TER, writes `data/standard/*.csv`.

### 2. Run a single bootstrap

```bash
python scripts/run_single.py sample_portfolio.csv
```

Or use **`notebooks/single_bootstrap.ipynb`** for interactive analysis.

### 3. Run a multi-bootstrap search

```bash
# Random search — 5 000 portfolios
python scripts/run_multi.py --random 5000

# Grid search — step size 0.10
python scripts/run_multi.py --grid 0.10

# Override simulation depth and parallelism
python scripts/run_multi.py --random 10000 --sims 500 --horizon 20 --jobs 8
```

Results → `results/multi_bootstrap_results.csv`.

### 4. Visualise results

Open **`notebooks/multi_bootstrap.ipynb`** to:
- 2D scatter of any two metrics (with hover: full weights + all metrics)
- Pareto frontier overlay
- Current-portfolio comparison (crimson star)
- Metric distributions

---

## Configuration

All settings live in **`engine/config.py`**:

| Section | Key parameters |
|---|---|
| **Paths** | `STANDARD_DIR`, `RESULTS_FILE`, … |
| **Simulation** | `N_SIMULATIONS` (1 000), `HORIZON_YEARS` (10), `USE_AFTER_TER_RETURNS` (True) |
| **Metrics** | `RETURN_PERCENTILES` [1,5,50,95,99], `VOLATILITY_WINDOWS` [1,3,5,10], `BAD_PERCENTILE` (2) |
| **Search** | `SEARCH_METHOD` ("random"), `N_PORTFOLIOS` (10 000), `GRID_STEP` (0.05) |
| **Pareto** | `PARETO_METRICS` — list of {name, direction} objectives |
| **Parallelism** | `N_JOBS` (-1 = all cores) |

---

## Metrics

### Shannon entropy (normalised)

A weight-based metric (no bootstrap needed).  Measures portfolio diversification:

$$H_{\text{norm}} = \frac{-\sum w_i \ln w_i}{\ln N} \in [0, 1]$$

- 0 = single asset
- 1 = equal-weight across all N assets

Included automatically in `compute_metrics()` when weights are provided,
and in every row of multi-bootstrap results.

### Annualised return percentiles

Full-horizon annualised return $(V_{\text{final}})^{1/\text{years}} - 1$ per simulation.
Report P1, P5, P50, P95, P99 (configurable).

### Volatility at N-year windows

Standard deviation of N-year annualised returns **across simulations**.
Decreases with longer windows (time diversification).

### Max drawdown depth / length / MDA

Reported at the **bad percentile** — e.g. `BAD_PERCENTILE = 2` means the
98th percentile of per-simulation max drawdowns (worst 2%).

- **Depth**: deepest peak-to-trough loss
- **Length**: longest contiguous below-peak period (months)
- **MDA**: worst "damage dose" in equivalent months at 100% drawdown

---

## Pareto frontier

Identifies portfolios where no other is better on **all** objectives simultaneously.
Configure in `engine/config.py`:

```python
PARETO_METRICS = [
    {"name": "annualised_return_p50",  "direction": "maximize"},
    {"name": "annualised_return_p1",   "direction": "maximize"},
    {"name": "volatility_10y",         "direction": "minimize"},
    {"name": "max_dd_depth_p2",        "direction": "minimize"},
    # {"name": "shannon_entropy",      "direction": "maximize"},
]
```

---

## Programmatic usage (for Tkinter / GUI)

The `engine` package is designed for direct import:

```python
from engine import run_bootstrap, load_portfolio_csv, shannon_entropy, compute_pareto
from engine import config as cfg

# single portfolio
portfolio = load_portfolio_csv("sample_portfolio.csv")
metrics = run_bootstrap(portfolio, n_sim=1000)
print(metrics["annualised_return_p50"], metrics["shannon_entropy"])

# multi-bootstrap
from engine import run_multi_bootstrap
csv_path = run_multi_bootstrap(method="random", n_portfolios=500, n_sim=200)

# fast repeated calls with preloaded data
from engine.data import preload_returns
from engine.runner import run_bootstrap_preloaded
tickers_sorted, ret_matrix = preload_returns(["MWEQ", "GOVH", "SP5A"])
result = run_bootstrap_preloaded(np.array([0.5, 0.3, 0.2]), ret_matrix)
```

---

## Performance notes

- Data loaded once via `preload_returns()`, shared across workers
- Bootstrap fully vectorised (numpy matmul + cumprod)
- Multiprocessing distributes portfolios across all CPU cores
- Typical: **50–250 portfolios/second** (1 000 sims × 10y, 8 assets)

---

## Adding new assets

1. Export CSV from Curvo → `data/raw_curvo/{TICKER}.csv`
2. Add ticker + annual TER to `data/TER_table.csv`
3. Run `python scripts/preprocess.py`
4. Add ticker to `search.csv` with weight range

---

## Requirements

- Python ≥ 3.10
- numpy, pandas, plotly
