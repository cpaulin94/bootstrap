"""
engine.config — Single source of truth for all settings.

Centralises simulation parameters, search configuration, Pareto objectives,
and path definitions.  Import this module from anywhere:

    from engine import config as cfg
"""

from __future__ import annotations

import os

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_DIR     = os.path.join(BASE_DIR, "data")
STANDARD_DIR  = os.path.join(DATA_DIR, "standard")
RAW_DIR       = os.path.join(DATA_DIR, "raw_curvo")
RAW_MSCI_DIR  = os.path.join(DATA_DIR, "raw_msci")
TER_FILE      = os.path.join(DATA_DIR, "TER_table.csv")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "multi_bootstrap_results.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation parameters
# ═══════════════════════════════════════════════════════════════════════════════

N_SIMULATIONS: int      = 50_000    # Monte-Carlo paths per portfolio
HORIZON_YEARS: int      = 10        # investment horizon in years
MONTHS_PER_YEAR: int    = 12
HORIZON_MONTHS: int     = HORIZON_YEARS * MONTHS_PER_YEAR

BLOCK_SIZE: int         = 12       # block-bootstrap block length in months
                                    # 1 = classic iid bootstrap (no autocorrelation)
                                    # 12 = sample year-long consecutive blocks

USE_AFTER_TER_RETURNS: bool = True  # True → month_return_after_TER
RANDOM_SEED: int | None     = None  # set to int for reproducibility

# Date-range cutoff for the historical returns used in the bootstrap.
# Format: "YYYY-MM" strings, or None to use the full available history.
# Useful to avoid overfitting to specific market regimes (e.g. COVID, GFC).
DATE_START: str | None = None       # e.g. "2002-03"
DATE_END:   str | None = None       # e.g. "2019-05"

# ═══════════════════════════════════════════════════════════════════════════════
# Metric parameters
# ═══════════════════════════════════════════════════════════════════════════════

RETURN_PERCENTILES: list[int] = [1, 5, 50, 95, 99]
VOLATILITY_WINDOWS: list[int] = [1, 3, 5, 10]
BAD_PERCENTILE: float         = 2   # worst-case fraction for drawdown metrics

# ═══════════════════════════════════════════════════════════════════════════════
# Search settings  (multi-bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_CSV: str    = os.path.join(BASE_DIR, "search.csv")
SEARCH_METHOD: str = "random"       # "random" | "mixed" (recommended, see
                                     # engine.search.sample_mixed_portfolios) | "grid"
N_PORTFOLIOS: int  = 10_000         # random-search count (ignored for grid)
GRID_STEP: float   = 0.05           # grid-search increment (ignored for random)

METRICS_TO_SAVE: list[str] | None = None  # None = save everything

# ═══════════════════════════════════════════════════════════════════════════════
# Pareto frontier objectives
# ═══════════════════════════════════════════════════════════════════════════════

_bp = BAD_PERCENTILE
PARETO_METRICS: list[dict] = [
    {"name": "annualised_return_p50",              "direction": "maximize"},
    {"name": "annualised_return_p1",               "direction": "maximize"},
    {"name": f"volatility_{VOLATILITY_WINDOWS[-1]}y", "direction": "minimize"},
    {"name": f"max_dd_depth_p{_bp}",               "direction": "minimize"},
    {"name": "effective_n_types",           "direction": "maximize"},
    
]

# ═══════════════════════════════════════════════════════════════════════════════
# Parallelism
# ═══════════════════════════════════════════════════════════════════════════════

N_JOBS: int = -1  # -1 = all CPU cores

# ═══════════════════════════════════════════════════════════════════════════════
# Life Strategy Simulator defaults
# ═══════════════════════════════════════════════════════════════════════════════

LIFE_TAX_RATE_PCT: float = 26.0        # capital-gains tax on realised gains
LIFE_INITIAL_CAPITAL: float = 100_000.0
LIFE_INFLATION_PCT: float = 2.0
LIFE_N_SIM: int = 5_000
LIFE_BLOCK_MONTHS: int = 12
