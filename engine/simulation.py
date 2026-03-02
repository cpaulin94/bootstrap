"""
engine.simulation — Vectorised Monte-Carlo bootstrap.
"""

from __future__ import annotations

import numpy as np


def _precompute_block_gross(
    weights: np.ndarray,
    returns: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Pre-compute rolling block gross returns for the portfolio.

    For each valid starting month *i*, the block gross return is::

        prod(1 + portfolio_return[i : i + block_size])

    Cross-asset correlation within each month is preserved because we
    combine assets first (via *weights*), then aggregate across months.

    Returns
    -------
    block_gross : (n_blocks_available,)
        One entry per overlapping block in the historical data.
    """
    port_monthly = returns @ weights             # (T,)
    with np.errstate(all="ignore"):
        gross_monthly = 1.0 + port_monthly       # (T,)
    T = len(gross_monthly)
    n_blocks = T - block_size + 1

    # Efficient rolling product via cumulative product + division
    cum = np.cumprod(gross_monthly)               # (T,)
    block_gross = np.empty(n_blocks, dtype=np.float64)
    block_gross[0] = cum[block_size - 1]
    if n_blocks > 1:
        block_gross[1:] = cum[block_size:] / cum[:n_blocks - 1]

    return block_gross


def simulate(
    weights: np.ndarray,
    returns: np.ndarray,
    n_sim: int,
    horizon_months: int,
    rng: np.random.Generator,
    block_size: int = 1,
) -> np.ndarray:
    """Run the bootstrap and return portfolio cumulative price paths.

    Parameters
    ----------
    block_size : int
        Number of consecutive months aggregated into one block.
        ``1`` = classic iid monthly bootstrap.
        ``> 1`` = block bootstrap.  Pre-computes rolling N-month gross
        returns for the portfolio, then samples from those — dramatically
        faster because the path has ``horizon / block_size`` steps instead
        of ``horizon`` steps.

    Returns
    -------
    paths : ndarray
        * block_size == 1 → shape ``(n_sim, horizon_months + 1)``
        * block_size  > 1 → shape ``(n_sim, n_blocks + 1)``
          where ``n_blocks = horizon_months // block_size``
        Price paths starting at 1.0.
    """
    n_months_avail = returns.shape[0]

    if block_size <= 1:
        # ── classic iid monthly bootstrap ─────────────────────────────
        idx = rng.integers(0, n_months_avail, size=(n_sim, horizon_months))
        sampled = returns[idx]                    # (S, H, A)
        with np.errstate(all="ignore"):
            port_ret = sampled @ weights          # (S, H)
        gross = 1.0 + port_ret
    else:
        # ── block bootstrap ───────────────────────────────────────────
        # 1) Pre-compute all overlapping block gross returns (done once)
        block_gross_all = _precompute_block_gross(weights, returns, block_size)
        n_avail = len(block_gross_all)

        # 2) Sample n_blocks per simulation
        n_blocks = horizon_months // block_size
        idx = rng.integers(0, n_avail, size=(n_sim, n_blocks))
        gross = block_gross_all[idx]              # (S, n_blocks)

    # ── cumulative price path ─────────────────────────────────────────
    with np.errstate(over="ignore"):
        cum = np.cumprod(gross, axis=1)

    ones = np.ones((n_sim, 1), dtype=np.float64)
    paths = np.hstack([ones, cum])
    return paths
