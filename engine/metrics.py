"""
engine.metrics — Portfolio and simulation metrics.

Contains both:
- **Weight-based metrics** (no bootstrap needed): Shannon entropy
- **Simulation-based metrics** (operate on paths matrix): returns, volatility, drawdowns
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from engine import config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Weight-based metrics  (no simulation required)
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(weights: np.ndarray) -> float:
    """Normalised Shannon entropy of a weight vector.

    H = -Σ wᵢ·ln(wᵢ)  (for wᵢ > 0)
    H_norm = H / ln(N)  ∈ [0, 1]  where N = total number of assets (including zeros)

    Returns 0.0 for a single-asset portfolio, 1.0 for equal weights.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)                    # total assets, including zeros
    if n <= 1:
        return 0.0
    w_nonzero = w[w > 0]
    h = -float(np.sum(w_nonzero * np.log(w_nonzero)))
    return round(h / np.log(n), 6)


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


def _max_drawdown_length(paths: np.ndarray, bad_pct: float, block_size: int = 1) -> int:
    """Longest drawdown (months) at the bad percentile.

    When *block_size* > 1, each path step spans *block_size* months, so
    the raw step count is multiplied by *block_size* to report months.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    in_dd = paths < running_max

    n_sim = in_dd.shape[0]
    max_lengths = np.zeros(n_sim, dtype=np.int64)

    for i in range(n_sim):
        row = in_dd[i]
        length = 0
        best = 0
        for v in row:
            if v:
                length += 1
                if length > best:
                    best = length
            else:
                length = 0
        max_lengths[i] = best

    steps = int(np.percentile(max_lengths, 100 - bad_pct))
    return steps * block_size


def _max_drawdown_area(
    dd: np.ndarray, paths: np.ndarray, bad_pct: float, block_size: int = 1,
) -> float:
    """MDA at the bad percentile (equiv. months at 100% drawdown).

    When *block_size* > 1 each step spans *block_size* months, so the
    raw area is multiplied by *block_size* to remain in month-units.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    in_dd = paths < running_max

    n_sim = in_dd.shape[0]
    mda_per_sim = np.zeros(n_sim, dtype=np.float64)

    for i in range(n_sim):
        row_dd = dd[i]
        row_in = in_dd[i]
        best_area = 0.0
        current_area = 0.0
        for t in range(len(row_in)):
            if row_in[t]:
                current_area += row_dd[t]
            else:
                if current_area > best_area:
                    best_area = current_area
                current_area = 0.0
        if current_area > best_area:
            best_area = current_area
        mda_per_sim[i] = best_area

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
) -> dict:
    """Compute all metrics on the simulated paths matrix.

    When *block_size* > 1 the paths have one column per block (not per
    month).  Volatility windows and drawdown durations are converted
    back to calendar-month units automatically.

    If *weights* is provided, Shannon entropy is included (no bootstrap needed).
    """
    metrics: dict = {}

    # ── weight-based metrics ──────────────────────────────────────────────
    if weights is not None:
        metrics["shannon_entropy"] = shannon_entropy(weights)

    # ── simulation-based metrics ──────────────────────────────────────────
    metrics.update(_annualised_return_percentiles(paths, horizon_years, percentiles))
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
