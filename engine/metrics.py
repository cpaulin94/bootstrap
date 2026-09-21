"""
engine.metrics — Portfolio and simulation metrics.

Contains both:
- **Weight-based metrics** (no bootstrap needed): Shannon entropy
- **Simulation-based metrics** (operate on paths matrix): returns, volatility, drawdowns
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from engine import config as cfg

log = logging.getLogger("bootstrap.metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# Weight-based metrics  (no simulation required)
# ═══════════════════════════════════════════════════════════════════════════════

def _shannon_nats(w: np.ndarray) -> float:
    """Raw Shannon entropy in nats: ``H = -Σ wᵢ·ln(wᵢ)`` over ``wᵢ > 0``.

    Zero weights contribute nothing (``0·ln 0 → 0``), which is what makes
    every metric built on this independent of how many zero-weight slots
    the weight vector happens to carry.
    """
    w = np.asarray(w, dtype=np.float64)
    nz = w[w > 0]
    if len(nz) == 0:
        return 0.0
    return -float(np.sum(nz * np.log(nz)))


def effective_n_assets(weights: np.ndarray) -> float:
    """Effective number of assets held: ``exp(H)`` (the perplexity of the
    weight vector).

    1.0 for a single-asset portfolio, exactly *k* for *k* equally-weighted
    assets, and between the two for anything concentrated — e.g.
    ``[0.9, 0.05, 0.03, 0.02]`` is "effectively 1.4 assets".

    This deliberately replaces the earlier ``shannon_entropy``, which
    divided ``H`` by ``ln(len(weights))`` — the length of the CONTAINER,
    not a property of the portfolio. That made the same holdings score
    differently depending on which tool computed them: a 73/27 two-asset
    portfolio scored 0.84 from the Single Bootstrap section (weight vector
    of length 2) and 0.22 from a Space Explorer run (length 15,
    zero-padded), and an equally-weighted 5-asset portfolio read as a
    perfect 1.00 in one place and 0.59 in the other. ``exp(H)`` is a
    monotone transform of ``H``, so rankings WITHIN any single run are
    unchanged — only the cross-tool disagreement goes away.
    """
    return round(float(np.exp(_shannon_nats(weights))), 6)


def effective_n_types(weights: np.ndarray, tickers: list[str]) -> float:
    """Effective number of asset-class TYPES held: ``exp(H)`` over
    type-aggregated weights (types from ``TER_table.csv``).

    Weights are summed by type first, so two equally-weighted STOCK
    tickers count as one type, not two. Tickers missing from the type map
    are grouped under ``"UNKNOWN"``.

    Same container-independence as :func:`effective_n_assets` — see its
    docstring for what this replaced and why.
    """
    from engine.data import load_type_map

    type_map = load_type_map()
    w = np.asarray(weights, dtype=np.float64)

    type_weights: dict[str, float] = {}
    for i, ticker in enumerate(tickers):
        t = type_map.get(ticker.upper(), "UNKNOWN")
        type_weights[t] = type_weights.get(t, 0.0) + w[i]

    tw = np.array(list(type_weights.values()), dtype=np.float64)
    return round(float(np.exp(_shannon_nats(tw))), 6)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation-based metrics  (operate on (n_sim, T+1) paths matrix)
# ═══════════════════════════════════════════════════════════════════════════════

def _annualised_returns(paths: np.ndarray, horizon_years: int) -> np.ndarray:
    """Annualised total return for each simulation."""
    final = paths[:, -1]
    return final ** (1.0 / horizon_years) - 1.0


def _annualised_return_percentiles(
    paths: np.ndarray,
    horizon_years: int,
    percentiles: list[int],
) -> dict[str, float]:
    ann = _annualised_returns(paths, horizon_years)
    pcts = np.percentile(ann, percentiles)
    return {f"annualised_return_p{p}": round(float(v), 6)
            for p, v in zip(percentiles, pcts)}


def _volatility_at_windows(
    paths: np.ndarray,
    windows_years: list[int],
    horizon_months: int,
    block_size: int = 1,
) -> dict[str, float]:
    """Std-dev of N-year annualised returns across simulations.

    When *block_size* > 1 the path has one column per block, so column
    indices are converted via ``n_yr * 12 // block_size``.  Windows that
    don't fall exactly on a block boundary are skipped.
    """
    result: dict[str, float] = {}
    n_steps = paths.shape[1] - 1          # total path steps (excl. t=0)
    for n_yr in windows_years:
        m = n_yr * 12                     # window in months
        if m > horizon_months:
            continue
        if block_size > 1:
            if m % block_size != 0:
                continue                  # skip non-aligned windows
            step_idx = m // block_size
        else:
            step_idx = m
        if step_idx > n_steps:
            continue
        total_ret = paths[:, step_idx] / paths[:, 0]
        with np.errstate(invalid="ignore"):
            ann_ret = total_ret ** (1.0 / n_yr) - 1.0
        valid = np.isfinite(ann_ret)
        if valid.sum() == 0:
            continue
        vol = float(np.std(ann_ret[valid]))
        result[f"volatility_{n_yr}y"] = round(vol, 6)
    return result


def _drawdown_series(paths: np.ndarray) -> np.ndarray:
    """Element-wise drawdown from running peak.  Shape same as *paths*."""
    running_max = np.maximum.accumulate(paths, axis=1)
    dd = (running_max - paths) / running_max
    return dd


def _max_drawdown_depth(dd: np.ndarray, bad_pct: float) -> float:
    """Deepest drawdown at the bad percentile across sims.

    bad_pct = fraction of worst outcomes (e.g. 2 → worst 2%).
    Reports the (100 − bad_pct)-th percentile of per-sim max-DD.
    """
    worst_per_sim = dd.max(axis=1)
    return round(float(np.percentile(worst_per_sim, 100 - bad_pct)), 6)


def _last_reset_index(in_dd: np.ndarray) -> np.ndarray:
    """For each column *t*, the index of the most recent ``False`` at or
    before *t* (or ``-1`` if none exists yet).

    Vectorised "reset" trick: replacing ``True`` entries with ``-1`` and
    ``False`` entries with their own column index, then taking a running
    max, produces exactly this — because a running max over indices only
    ever advances at a ``False`` (reset) position.
    """
    n_sim, T = in_dd.shape
    idx = np.arange(T)[None, :]
    reset_idx = np.where(in_dd, -1, idx)
    return np.maximum.accumulate(reset_idx, axis=1)


def _max_drawdown_length(paths: np.ndarray, bad_pct: float, block_size: int = 1) -> int:
    """Longest drawdown (months) at the bad percentile.

    When *block_size* > 1, each path step spans *block_size* months, so
    the raw step count is multiplied by *block_size* to report months.

    Vectorised: for a run ending at column *t*, its length is
    ``t - last_reset[t]`` where *last_reset* is the last ``False`` column
    at or before *t* (see :func:`_last_reset_index`).
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    in_dd = paths < running_max

    T = in_dd.shape[1]
    idx = np.arange(T)[None, :]
    last_reset = _last_reset_index(in_dd)
    run_length = np.where(in_dd, idx - last_reset, 0)
    max_lengths = run_length.max(axis=1)

    steps = int(np.percentile(max_lengths, 100 - bad_pct))
    return steps * block_size


def _max_drawdown_area(
    dd: np.ndarray, paths: np.ndarray, bad_pct: float, block_size: int = 1,
) -> float:
    """MDA at the bad percentile (equiv. months at 100% drawdown).

    When *block_size* > 1 each step spans *block_size* months, so the
    raw area is multiplied by *block_size* to remain in month-units.

    Vectorised: the sum of *dd* over the run ending at column *t* is a
    "cumulative sum that resets at each False" — computed as the global
    cumsum at *t* minus the global cumsum at *last_reset[t]* (padded with
    a leading zero so ``last_reset == -1`` maps cleanly to "no prior sum").
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    in_dd = paths < running_max

    n_sim = in_dd.shape[0]
    last_reset = _last_reset_index(in_dd)

    cumsum = np.cumsum(dd, axis=1)
    padded = np.concatenate([np.zeros((n_sim, 1), dtype=cumsum.dtype), cumsum], axis=1)
    start_vals = np.take_along_axis(padded, last_reset + 1, axis=1)

    run_sum = np.where(in_dd, cumsum - start_vals, 0.0)
    mda_per_sim = run_sum.max(axis=1)

    raw = float(np.percentile(mda_per_sim, 100 - bad_pct))
    return round(raw * block_size, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# Combined metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    paths: np.ndarray,
    horizon_years: int     = cfg.HORIZON_YEARS,
    horizon_months: int    = cfg.HORIZON_MONTHS,
    block_size: int        = cfg.BLOCK_SIZE,
    percentiles: list[int] = cfg.RETURN_PERCENTILES,
    vol_windows: list[int] = cfg.VOLATILITY_WINDOWS,
    bad_pct: float         = cfg.BAD_PERCENTILE,
    weights: Optional[np.ndarray] = None,
    tickers: Optional[list[str]] = None,
) -> dict:
    """Compute all metrics on the simulated paths matrix.

    When *block_size* > 1 the paths have one column per block (not per
    month).  Volatility windows and drawdown durations are converted
    back to calendar-month units automatically.

    A *block_size* that doesn't evenly divide *horizon_months* truncates
    the final partial block (see :func:`engine.simulation.simulate`), so
    ``paths`` may cover fewer months than the nominal *horizon_years* asks
    for. Annualised-return percentiles are computed against the months
    ACTUALLY simulated (``paths.shape[1] - 1`` steps, each *block_size*
    months), not the nominal horizon — using the nominal horizon here would
    silently understate/overstate the annualised return by conflating
    "fewer months were simulated" with "the portfolio performed worse".

    If *weights* is provided, ``effective_n_assets`` is included (no
    bootstrap needed). If *tickers* is also provided, ``effective_n_types``
    (diversification across asset-class types) is included as well.
    """
    metrics: dict = {}

    # ── weight-based metrics ──────────────────────────────────────────────
    if weights is not None:
        metrics["effective_n_assets"] = effective_n_assets(weights)
        if tickers is not None:
            metrics["effective_n_types"] = effective_n_types(weights, tickers)

    # ── simulation-based metrics ──────────────────────────────────────────
    n_steps = paths.shape[1] - 1
    effective_months = n_steps * max(block_size, 1)
    if effective_months <= 0:
        raise ValueError(f"paths has {n_steps} simulated steps — nothing to compute metrics on.")
    if effective_months != horizon_months:
        log.warning(
            "[METRICS] block_size=%d doesn't evenly divide horizon_months=%d — "
            "only %d months were actually simulated; annualising against that "
            "instead of the nominal horizon.", block_size, horizon_months, effective_months,
        )
    effective_years = effective_months / 12.0

    metrics.update(_annualised_return_percentiles(paths, effective_years, percentiles))
    metrics.update(_volatility_at_windows(paths, vol_windows, horizon_months,
                                          block_size=block_size))

    dd = _drawdown_series(paths)
    bp = str(bad_pct)
    metrics[f"max_dd_depth_p{bp}"] = _max_drawdown_depth(dd, bad_pct)
    metrics[f"max_dd_length_months_p{bp}"] = _max_drawdown_length(
        paths, bad_pct, block_size=block_size)
    metrics[f"mda_months_p{bp}"] = _max_drawdown_area(
        dd, paths, bad_pct, block_size=block_size)

    return metrics
