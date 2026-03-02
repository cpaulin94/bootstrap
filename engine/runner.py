"""
engine.runner — Bootstrap orchestrators (single + multi).

Public API
----------
    run_bootstrap(portfolio, ...) -> dict
        Single portfolio: load data → simulate → compute all metrics.

    run_bootstrap_preloaded(weights, ret_matrix, ...) -> dict
        Fast path for repeated calls with pre-loaded data.

    run_multi_bootstrap(...) -> str
        Mass search: generate portfolios → parallel bootstrap → save CSV.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np

from engine import config as cfg
from engine.data import load_all_returns, preload_returns
from engine.metrics import compute_metrics
from engine.search import (
    load_search_space,
    sample_grid_portfolios,
    sample_random_portfolios,
)
from engine.simulation import simulate


# ═══════════════════════════════════════════════════════════════════════════════
# Single-portfolio bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def run_bootstrap_preloaded(
    weights: np.ndarray,
    ret_matrix: np.ndarray,
    *,
    n_sim: int           = cfg.N_SIMULATIONS,
    horizon_years: int   = cfg.HORIZON_YEARS,
    block_size: int      = cfg.BLOCK_SIZE,
    rng: Optional[np.random.Generator] = None,
    percentiles: list[int] = cfg.RETURN_PERCENTILES,
    vol_windows: list[int] = cfg.VOLATILITY_WINDOWS,
    bad_pct: float         = cfg.BAD_PERCENTILE,
) -> dict:
    """Run bootstrap with pre-loaded data.  Skips CSV I/O."""
    if rng is None:
        rng = np.random.default_rng()
    horizon_months = horizon_years * cfg.MONTHS_PER_YEAR
    paths = simulate(weights, ret_matrix, n_sim, horizon_months, rng,
                     block_size=block_size)
    return compute_metrics(
        paths, horizon_years, horizon_months,
        block_size=block_size,
        percentiles=percentiles, vol_windows=vol_windows, bad_pct=bad_pct,
        weights=weights,
    )


def run_bootstrap(
    portfolio: dict[str, float],
    *,
    n_sim: int             = cfg.N_SIMULATIONS,
    horizon_years: int     = cfg.HORIZON_YEARS,
    block_size: int        = cfg.BLOCK_SIZE,
    use_after_ter: bool    = cfg.USE_AFTER_TER_RETURNS,
    random_seed: Optional[int] = cfg.RANDOM_SEED,
) -> dict:
    """Full pipeline: load data → simulate → compute metrics → return dict."""
    horizon_months = horizon_years * cfg.MONTHS_PER_YEAR
    rng = np.random.default_rng(random_seed)

    weights, ret_matrix = load_all_returns(portfolio, use_after_ter)
    paths = simulate(weights, ret_matrix, n_sim, horizon_months, rng,
                     block_size=block_size)
    metrics = compute_metrics(paths, horizon_years, horizon_months,
                              block_size=block_size, weights=weights)

    # attach metadata
    metrics["n_simulations"]     = n_sim
    metrics["horizon_years"]     = horizon_years
    metrics["block_size"]        = block_size
    metrics["use_after_ter"]     = use_after_ter
    metrics["n_months_history"]  = ret_matrix.shape[0]

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-portfolio bootstrap  (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level globals for multiprocessing workers
_W_RET_MATRIX: np.ndarray | None = None
_W_PARAMS: dict = {}


def _init_worker(ret_matrix: np.ndarray, params: dict) -> None:
    """Initialiser called once per worker process."""
    global _W_RET_MATRIX, _W_PARAMS
    _W_RET_MATRIX = ret_matrix
    _W_PARAMS = params


def _eval_portfolio(weights: np.ndarray) -> dict:
    """Evaluate a single weight vector (called in worker process)."""
    rng = np.random.default_rng()
    return run_bootstrap_preloaded(
        weights,
        _W_RET_MATRIX,
        n_sim=_W_PARAMS["n_sim"],
        horizon_years=_W_PARAMS["horizon_years"],
        block_size=_W_PARAMS["block_size"],
        rng=rng,
        percentiles=_W_PARAMS["percentiles"],
        vol_windows=_W_PARAMS["vol_windows"],
        bad_pct=_W_PARAMS["bad_pct"],
    )


def run_multi_bootstrap(
    search_csv: str         = cfg.SEARCH_CSV,
    method: str             = cfg.SEARCH_METHOD,
    n_portfolios: int       = cfg.N_PORTFOLIOS,
    grid_step: float        = cfg.GRID_STEP,
    n_sim: int              = cfg.N_SIMULATIONS,
    horizon_years: int      = cfg.HORIZON_YEARS,
    block_size: int         = cfg.BLOCK_SIZE,
    use_after_ter: bool     = cfg.USE_AFTER_TER_RETURNS,
    n_jobs: int             = cfg.N_JOBS,
    output_path: str        = cfg.RESULTS_FILE,
    metrics_to_save: list[str] | None = cfg.METRICS_TO_SAVE,
) -> str:
    """Run the full multi-bootstrap pipeline. Returns the output CSV path."""

    # ── 1. search space ───────────────────────────────────────────────────
    space = load_search_space(search_csv)
    tickers = [s["ticker"] for s in space]
    print(f"Search space: {len(tickers)} tickers  {tickers}")

    # ── 2. pre-load return data ───────────────────────────────────────────
    sorted_tickers, ret_matrix = preload_returns(tickers, use_after_ter)
    print(f"Return matrix: {ret_matrix.shape[0]} months × {ret_matrix.shape[1]} assets")

    # ── 3. generate candidate portfolios ──────────────────────────────────
    rng = np.random.default_rng()
    if method == "random":
        portfolios = sample_random_portfolios(space, n_portfolios, rng)
        print(f"Sampled {len(portfolios)} random portfolios")
    elif method == "grid":
        portfolios = sample_grid_portfolios(space, grid_step)
        print(f"Generated {len(portfolios)} grid portfolios (step={grid_step})")
    else:
        raise ValueError(f"Unknown search method: {method!r}")

    if len(portfolios) == 0:
        print("No valid portfolios found. Check search space constraints.")
        sys.exit(1)

    # reorder columns to match sorted_tickers
    ticker_order = [tickers.index(t) for t in sorted_tickers]
    portfolios = portfolios[:, ticker_order]

    # ── 4. run bootstraps in parallel ─────────────────────────────────────
    n_workers = cpu_count() if n_jobs == -1 else n_jobs
    params = dict(
        n_sim=n_sim,
        horizon_years=horizon_years,
        block_size=block_size,
        percentiles=cfg.RETURN_PERCENTILES,
        vol_windows=cfg.VOLATILITY_WINDOWS,
        bad_pct=cfg.BAD_PERCENTILE,
    )

    print(f"Running {len(portfolios)} bootstraps "
          f"({n_sim} sims × {horizon_years}y, block={block_size}m) "
          f"on {n_workers} cores...")
    t0 = time.perf_counter()

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(ret_matrix, params),
    ) as pool:
        all_metrics = pool.map(_eval_portfolio, portfolios, chunksize=16)

    elapsed = time.perf_counter() - t0
    rate = len(portfolios) / elapsed
    print(f"Done in {elapsed:.1f}s  ({rate:.0f} portfolios/s)")

    # ── 5. assemble and save results ──────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metric_keys = sorted(all_metrics[0].keys())
    if metrics_to_save is not None:
        metric_keys = [k for k in metric_keys if k in metrics_to_save]

    header = [f"w_{t}" for t in sorted_tickers] + metric_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for weights, mdict in zip(portfolios, all_metrics):
            row = [round(w, 6) for w in weights] + [mdict.get(k, "") for k in metric_keys]
            writer.writerow(row)

    print(f"Results saved → {output_path}  ({len(portfolios)} rows × {len(header)} cols)")
    return output_path
