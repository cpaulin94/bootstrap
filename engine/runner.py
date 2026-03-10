"""
engine.runner — Bootstrap orchestrators (single + multi).

Public API
----------
    run_bootstrap(portfolio, ...) -> dict
        Single portfolio: load data → simulate → compute all metrics.

    run_bootstrap_preloaded(weights, ret_matrix, ...) -> dict
        Fast path for repeated calls with pre-loaded data.

    run_multi_streaming(...) -> tuple
        Callback-driven multi-bootstrap with cancellation support.

    run_multi_bootstrap(...) -> str
        Mass search: generate portfolios → parallel bootstrap → save CSV.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from multiprocessing import Pool, cpu_count
from typing import Callable, Optional

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
    date_start: Optional[str]  = cfg.DATE_START,
    date_end: Optional[str]    = cfg.DATE_END,
) -> dict:
    """Full pipeline: load data → simulate → compute metrics → return dict."""
    horizon_months = horizon_years * cfg.MONTHS_PER_YEAR
    rng = np.random.default_rng(random_seed)

    weights, ret_matrix = load_all_returns(
        portfolio, use_after_ter,
        date_start=date_start, date_end=date_end,
    )
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
    metrics["date_start"]        = date_start or ""
    metrics["date_end"]          = date_end or ""

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-portfolio bootstrap  (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level globals for multiprocessing workers
_W_RET_MATRIX: np.ndarray | None = None
_W_PARAMS: dict = {}


def _init_worker(ret_matrix: np.ndarray, params: dict) -> None:
    """Initialiser called once per worker process."""
    import warnings
    warnings.filterwarnings("ignore")          # suppress BLAS RuntimeWarnings
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


def _eval_portfolio_with_weights(weights: np.ndarray) -> dict:
    """Like _eval_portfolio but attaches the weight vector to the result dict.

    Defined here (not in gui.py) so that multiprocessing 'spawn' workers
    only need to import this lightweight module, not the heavy GUI module.
    """
    m = _eval_portfolio(weights)
    m["_weights"] = weights.tolist()
    return m


def run_multi_streaming(
    space: list[dict],
    *,
    method: str             = cfg.SEARCH_METHOD,
    n_portfolios: int       = cfg.N_PORTFOLIOS,
    grid_step: float        = cfg.GRID_STEP,
    n_sim: int              = cfg.N_SIMULATIONS,
    horizon_years: int      = cfg.HORIZON_YEARS,
    block_size: int         = cfg.BLOCK_SIZE,
    use_after_ter: bool     = cfg.USE_AFTER_TER_RETURNS,
    n_jobs: int             = cfg.N_JOBS,
    date_start: Optional[str] = cfg.DATE_START,
    date_end: Optional[str]   = cfg.DATE_END,
    seed: Optional[int]     = None,
    flush_every: int        = 500,
    return_weights: bool    = False,
    on_start: Optional[Callable[[list[str], int], None]] = None,
    on_batch: Optional[Callable[[list[dict], int, int, float], None]] = None,
    on_done: Optional[Callable[[int, float, float], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> tuple[list[str], list[dict]]:
    """Streaming multi-bootstrap — the shared engine for CLI and GUI.

    Parameters
    ----------
    space : list[dict]
        Search space, each dict has keys: ticker, lo, hi.
    on_start(sorted_tickers, total) :
        Called once after data is loaded and portfolios are generated.
    on_batch(results, n_done, total, speed) :
        Called every *flush_every* completed portfolios.
    on_done(n_done, elapsed, avg_speed) :
        Called when all portfolios are evaluated.
    on_error(message) :
        Called on failure.
    stop_event :
        Set this ``threading.Event`` to request early termination.
    return_weights :
        If True, each result dict includes ``_weights`` (list[float]).

    Returns
    -------
    (sorted_tickers, all_results) — the ordered ticker list and list of
    metric dicts.  Empty on error / early stop.
    """
    try:
        # ── 1. pre-load return data ───────────────────────────────────────
        tickers = [s["ticker"] for s in space]
        sorted_tickers, ret_matrix = preload_returns(
            tickers, use_after_ter,
            date_start=date_start, date_end=date_end,
        )

        # ── 2. generate candidate portfolios ──────────────────────────────
        rng = np.random.default_rng(seed)
        if method == "random":
            portfolios = sample_random_portfolios(space, n_portfolios, rng)
        elif method == "grid":
            portfolios = sample_grid_portfolios(space, grid_step)
        else:
            raise ValueError(f"Unknown search method: {method!r}")

        if len(portfolios) == 0:
            if on_error:
                on_error("No valid portfolios generated.")
            return sorted_tickers, []

        # reorder columns to match sorted_tickers
        ticker_order = [tickers.index(t) for t in sorted_tickers]
        portfolios = portfolios[:, ticker_order]
        total = len(portfolios)

        if on_start:
            on_start(sorted_tickers, total)

        # ── 3. run bootstraps in parallel ─────────────────────────────────
        n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        params = dict(
            n_sim=n_sim,
            horizon_years=horizon_years,
            block_size=block_size,
            percentiles=cfg.RETURN_PERCENTILES,
            vol_windows=cfg.VOLATILITY_WINDOWS,
            bad_pct=cfg.BAD_PERCENTILE,
        )

        eval_fn = _eval_portfolio_with_weights if return_weights else _eval_portfolio
        chunksize = max(1, min(64, total // (n_workers * 4)))

        all_results: list[dict] = []
        t_start = time.perf_counter()
        batch_buf: list[dict] = []
        batch_t0 = time.perf_counter()

        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(ret_matrix, params),
        ) as pool:
            for m in pool.imap_unordered(eval_fn, portfolios, chunksize=chunksize):
                if stop_event and stop_event.is_set():
                    pool.terminate()
                    break

                all_results.append(m)
                batch_buf.append(m)

                if len(batch_buf) >= flush_every or len(all_results) == total:
                    batch_elapsed = time.perf_counter() - batch_t0
                    speed = len(batch_buf) / max(batch_elapsed, 1e-6)
                    if on_batch:
                        on_batch(list(batch_buf), len(all_results), total, speed)
                    batch_buf = []
                    batch_t0 = time.perf_counter()

        elapsed = time.perf_counter() - t_start
        avg_speed = len(all_results) / max(elapsed, 1e-6)
        if on_done:
            on_done(len(all_results), elapsed, avg_speed)

        return sorted_tickers, all_results

    except Exception as exc:
        if on_error:
            on_error(str(exc))
        return [], []


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
    date_start: Optional[str] = cfg.DATE_START,
    date_end: Optional[str]   = cfg.DATE_END,
) -> str:
    """Run the full multi-bootstrap pipeline. Returns the output CSV path.

    This is a thin CLI wrapper around :func:`run_multi_streaming` that adds
    console logging and CSV output.
    """
    space = load_search_space(search_csv)
    tickers = [s["ticker"] for s in space]
    n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
    print(f"Search space: {len(tickers)} tickers  {tickers}")

    # Store portfolios for CSV output (closure variable)
    _portfolios_ref: list[np.ndarray] = []

    def _on_start(sorted_tickers, total):
        print(f"Return matrix loaded.  {total} portfolios to evaluate.")
        print(f"Running {total} bootstraps "
              f"({n_sim} sims × {horizon_years}y, block={block_size}m) "
              f"on {n_workers} cores...")

    def _on_done(n_done, elapsed, avg_speed):
        print(f"Done in {elapsed:.1f}s  ({avg_speed:.0f} portfolios/s)")

    def _on_error(msg):
        print(f"ERROR: {msg}", file=sys.stderr)

    sorted_tickers, all_metrics = run_multi_streaming(
        space,
        method=method,
        n_portfolios=n_portfolios,
        grid_step=grid_step,
        n_sim=n_sim,
        horizon_years=horizon_years,
        block_size=block_size,
        use_after_ter=use_after_ter,
        n_jobs=n_jobs,
        date_start=date_start,
        date_end=date_end,
        flush_every=max(1, n_portfolios),  # single batch → same as pool.map
        return_weights=True,               # need weights for CSV
        on_start=_on_start,
        on_done=_on_done,
        on_error=_on_error,
    )

    if not all_metrics:
        print("No results produced.")
        sys.exit(1)

    # ── assemble and save results ─────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metric_keys = sorted(k for k in all_metrics[0].keys() if not k.startswith("_"))
    if metrics_to_save is not None:
        metric_keys = [k for k in metric_keys if k in metrics_to_save]

    header = [f"w_{t}" for t in sorted_tickers] + metric_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for mdict in all_metrics:
            weights = mdict.get("_weights", [])
            row = [round(w, 6) for w in weights] + [mdict.get(k, "") for k in metric_keys]
            writer.writerow(row)

    print(f"Results saved → {output_path}  ({len(all_metrics)} rows × {len(header)} cols)")
    return output_path
