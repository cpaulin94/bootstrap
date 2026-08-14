# Bootstrap Portfolio Simulator

A Monte-Carlo bootstrap framework for evaluating and optimising multi-asset portfolios, plus a
Tkinter desktop GUI. Simulate thousands of possible futures, compute risk/return metrics, find
Pareto-optimal allocations, and run a full life-strategy (accumulate → hold → withdraw)
projection with taxes and inflation.

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
│   ├── runner.py                  #   single + multi bootstrap orchestrators
│   └── lifecycle.py               #   life-strategy simulator (accumulate/hold/withdraw + taxes)
│
├── gui.py                         # Tkinter desktop application (4 sections, see below)
├── main.py                        # Entry point — `uv run main.py`
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
├── tests/                         # pytest suite for engine/
│
├── data/
│   ├── TER_table.csv              # Annual TER per ticker + asset TYPE
│   ├── raw_curvo/                 # Raw CSV exports from Curvo
│   └── standard/                  # Preprocessed monthly returns (auto-generated)
│
├── results/                       # Multi-bootstrap output (auto-generated, gitignored)
│   └── multi_bootstrap_results.csv
│
├── search.csv                     # Search space (tickers + weight ranges)
├── sample_portfolio.csv           # Example portfolio definition
├── .portfolio_library.json        # Saved portfolios (GUI persistence)
├── .life_plans.json               # Saved life-strategy plans (GUI persistence)
└── README.md
```

### Design principles

- **`engine/` is a plain Python package** — no CLI, no I/O side-effects, no prints
  in the hot path. It can be imported by notebooks, CLI scripts, or the GUI without changes.
- **One config file** — `engine/config.py` holds every default. Override values there or pass
  them as function arguments.
- **Scripts are thin wrappers** — they parse CLI args and call `engine` functions.
- **The GUI contains no numerical logic** — every computation lives in `engine/`.

---

## Quick start

This project is managed with [uv](https://docs.astral.sh/uv/).

### 0. Install dependencies

```bash
uv sync
```

### 1. Launch the GUI

```bash
uv run main.py
```

### 2. Preprocess raw data

```bash
uv run scripts/preprocess.py
```

Reads `data/raw_curvo/*.csv`, adjusts for TER, writes `data/standard/*.csv`.

### 3. Run a single bootstrap (CLI)

```bash
uv run scripts/run_single.py sample_portfolio.csv
```

### 4. Run a multi-bootstrap search (CLI)

```bash
# Random search — 5 000 portfolios
uv run scripts/run_multi.py --random 5000

# Grid search — step size 0.10
uv run scripts/run_multi.py --grid 0.10

# Override simulation depth and parallelism
uv run scripts/run_multi.py --random 10000 --sims 500 --horizon 20 --jobs 8
```

Results → `results/multi_bootstrap_results.csv`.

### 5. Run the test suite

```bash
uv run pytest
```

---

## The GUI (`uv run main.py`)

Four sections:

1. **Portfolio Builder** — pick assets, set weights, inspect the correlation heatmap, save to
   the portfolio library.
2. **Space Explorer** — random/grid search over a weight simplex, Pareto frontier, interactive
   Plotly scatter (click a point to add it to the library).
3. **Single Bootstrap** — compare portfolios (fan chart, return distribution, percentile bars,
   volatility) or sweep block-size sensitivity.
4. **Life Strategy** — see below.

---

## Life Strategy Simulator

Answers: *"If I invest like this, contribute like this, and later withdraw like this, how does
it end?"*

You pick **one portfolio** from the library and define a **sequence of phases**:

- **Accumulate** — a monthly contribution (with an optional annual *real* growth rate, e.g. career
  progression), for N years.
- **Hold** — stay invested, no cash flow.
- **Withdraw** — a monthly **net** (after-tax) income taken out, for N years, in one of four
  styles (see below).

Plus optional **one-off lump sums** at any point in time (positive = inflow such as an
inheritance, negative = outflow such as a house purchase).

All amounts are entered in **today's euros**. Internally the simulator inflates every cash flow
to nominal terms and, on display, can deflate results back to today's euros — this is purely a
display choice and does not re-run the simulation.

### Sampling

Uses a **monthly block bootstrap** (`engine.simulation.sample_monthly_returns`): historical
returns are resampled in overlapping 12-month blocks (configurable) to preserve within-year
autocorrelation, while keeping every individual month available for applying cash flows.

### Capital-gains tax — average-cost method

Each simulation tracks two scalars per path: `V` (market value) and `B` (cost basis of capital
not yet sold). The latent-gain fraction is `u = max(0, 1 - B/V)`.

- **Contribution** `c`: `V += c`, `B += c` (untaxed, basis grows with what's paid in).
- **Net withdrawal** `w_net`: solve the gross-up `g = w_net / (1 - rate·u)` for the gross amount
  to sell, so that after tax the investor pockets exactly `w_net`. Then
  `B -= g·(B/V)` (pro-rata basis reduction, computed **before** updating `V`), `V -= g`, and
  `tax = rate·g·u`.
- **Exhaustion**: if `g > V`, sell everything (`g = V`), the investor receives
  `V·(1 - rate·u)` — less than requested — and the shortfall is recorded. `V` floors at 0 and
  stays there until a future contribution or lump sum.
- **Loss** (`V < B`, so `u = 0`): no tax on withdrawal, `g = w_net`.
- **Rebirth**: a lump sum after exhaustion resets `V = B = amount` — a fresh cost basis, no
  carried-over gain.

Loss carry-forward against future gains is **not** modelled (a documented simplification).

### Withdrawal styles — one of four per phase, re-evaluated every month

A WITHDRAW phase's monthly NET target is computed one of four ways
(`engine.lifecycle.WithdrawalStyle`). The first is deterministic — identical for every
simulation. The other three are **path-dependent**: the target itself differs simulation to
simulation, because it depends on how that particular simulated market behaved.

- **Fixed** — the target is `monthly_amount` (today's €), indexed to inflation exactly like an
  ACCUMULATE contribution. This is the only style available before this feature existed.
- **Drawdown-curtailed** — normally withdraws the FULL target, but cuts to a lower REDUCED floor
  in any month where that simulation's drawdown from its own historical peak is at or beyond a
  custom trigger. Re-evaluated every month (not committed for a year), and the cut is binary (no
  partial tapering). Drawdown uses the same peak-to-trough convention as
  `engine.metrics._drawdown_series`: `dd = (running_max - V) / running_max`, where `running_max`
  is that simulation's highest portfolio value so far. The cut applies when `dd >= threshold`.
  Both the full and reduced amounts grow together under the phase's real-growth rate.
- **% of portfolio** — the NET target each month is `withdrawal_pct_per_month`% of that
  simulation's *current* portfolio value (before that month's withdrawal). The target scales
  automatically with the market — no separate inflation indexing applies, since it is already
  relative to a nominal quantity.
- **% of portfolio (ramp)** — same mechanic as "% of portfolio", except the rate itself moves in
  a straight line across the phase, from `withdrawal_pct_per_month` (first month) to
  `withdrawal_pct_end_per_month` (last month) — e.g. "0.05% of the portfolio in month 1, ramping
  up to 0.45% by month 240" over a 20-year phase. Either end can be the larger one — ramping down
  is just as valid as ramping up. The per-month rate is precomputed as
  `np.linspace(start, end, n_months)` in `engine.lifecycle._phase_schedule`; from there on it is
  resolved exactly like the constant-rate style (same internal style id — see
  `_WITHDRAWAL_STYLE_ID`), since by the time the month loop runs there is nothing left to
  distinguish "this month's rate" from a flat rate.

All four styles pass through the same tax gross-up (`apply_withdrawal`) — a drawdown-curtailed
or percentage target is just a different way of arriving at the `w_net` figure that function
grosses up; the tax mechanics don't change.

### Inflation

- Monthly factor: `infl_m = (1 + inflation_pct/100) ** (1/12)`.
- A cash flow entered as `today_amount` becomes, at month `m`,
  `today_amount * infl_m ** m` in nominal terms.
- Career-progression growth on contributions compounds **on top of** inflation, in annual steps:
  `contribution(m) = base_today * (1 + real_growth_pct/100) ** floor(months_into_phase/12)
  * infl_m ** m`.
- The historical return series is used as-is (nominal, as recorded); only cash flows are
  indexed to expected inflation, and only the *result* is deflated for display. This is the
  standard convention and is a deliberate simplification, not an oversight.

### Charts

Two panels, sharing an x-axis in years:

- **Wealth cloud + confidence bands** (top, `LineCollection` of up to 300 sampled paths, plus
  P1/P10/P50/P90/P99 bands, phase boundary markers, and lump-sum annotations). Governed by the
  "today's €" toggle — shows either nominal or real, one at a time.
- **Monthly cash flow** (bottom) — always shows **nominal and today's-€ together**, regardless of
  the toggle above, with a hatched wedge filled between the two: that wedge *is* inflation's
  effect on the cash flow, made visible rather than left to be inferred. For a constant
  today's-€ target (e.g. €2,000/month), the today's-€ line is flat while the nominal line climbs
  — the wedge widens over time. Withdrawals are drawn as a P10–P90 band around the nominal
  target (flat for a Fixed-style phase, genuinely spread for the path-dependent styles — that
  spread is the point: it shows how much a drawdown-curtailed or percentage strategy actually
  swings across simulations) plus the median *realised* withdrawal, which diverges from the
  target once a simulation's portfolio runs out of money.
- **Interactive cursor**: hover or click-lock a year to read exact wealth percentiles, this
  month's contribution/withdrawal (target and actual, in both currencies), cumulative totals to
  date, and the fraction of simulations already ruined at that point.

---

## Metrics

### Shannon entropy (normalised)

A weight-based metric (no bootstrap needed). Measures portfolio diversification:

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

## Programmatic usage

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

# life-strategy simulation
from engine.data import load_all_returns
from engine.lifecycle import LifePlan, Phase, PhaseKind, LumpSum, simulate_life_strategy

weights, returns = load_all_returns({"MWEQ": 0.5, "GOVH": 0.3, "GOLD": 0.2})
plan = LifePlan(
    portfolio_name="my-plan",
    initial_capital=100_000,
    tax_rate_pct=26.0,
    inflation_pct=2.0,
    phases=[
        Phase(PhaseKind.ACCUMULATE, years=20, monthly_amount=1_000, real_growth_pct=3.0),
        Phase(PhaseKind.WITHDRAW, years=10, monthly_amount=1_500),
    ],
    lump_sums=[LumpSum(at_year=20, amount=150_000, label="inheritance")],
)
result = simulate_life_strategy(plan, weights, returns)
print(result.summary())
```

---

## Performance notes

- Data loaded once via `preload_returns()`, shared across workers
- Bootstrap fully vectorised (numpy matmul + cumprod)
- Multiprocessing distributes portfolios across all CPU cores
- Typical: **50–250 portfolios/second** (1 000 sims × 10y, 8 assets)
- Life-strategy: 5 000 sims × 30 years typically completes end-to-end in under 2 seconds

---

## Adding new assets

1. Export CSV from Curvo → `data/raw_curvo/{TICKER}.csv`
2. Add ticker + annual TER to `data/TER_table.csv`
3. Run `uv run scripts/preprocess.py`
4. Add ticker to `search.csv` with weight range

---

## Requirements

- Python ≥ 3.13
- numpy, pandas, matplotlib, plotly
- pytest (dev)
