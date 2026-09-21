"""
engine.simulation — Vectorised Monte-Carlo bootstrap.
"""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("bootstrap.simulation")


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
    # Apple Accelerate BLAS on aarch64 emits spurious divide-by-zero /
    # overflow / invalid-value warnings during SIMD-vectorised matmul
    # even when all inputs are finite and the results are correct.
    # Verified: row-by-row matmul produces 0 warnings; only the batched
    # BLAS path triggers them.  Safe to suppress here.
    with np.errstate(all="ignore"):
        port_monthly = returns @ weights         # (T,)
    gross_monthly = 1.0 + port_monthly           # (T,)
    T = len(gross_monthly)
    n_blocks = T - block_size + 1

    # Log-space rolling product avoids cumulative-product overflow.
    # Instead of  cum = cumprod(gross)  which can overflow to inf for
    # long series, we compute  cum_log = cumsum(log(gross))  and only
    # exponentiate the block-sized differences.  This stays finite as
    # long as no single gross return is <= 0  (i.e. monthly loss < 100%).
    log_gross = np.log(gross_monthly)             # (T,)
    cum_log = np.cumsum(log_gross)                # (T,)
    block_gross = np.empty(n_blocks, dtype=np.float64)
    block_gross[0] = np.exp(cum_log[block_size - 1])
    if n_blocks > 1:
        block_gross[1:] = np.exp(cum_log[block_size:] - cum_log[:n_blocks - 1])

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
    log.debug("[SIM] simulate: weights=%s  returns=%s  n_sim=%d  "
              "horizon_months=%d  block_size=%d  n_months_avail=%d",
              weights.shape, returns.shape, n_sim, horizon_months,
              block_size, n_months_avail)
    t0 = time.perf_counter()

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

        # 2) Sample n_blocks per simulation. Floor division: a block_size
        # that doesn't evenly divide horizon_months truncates the final
        # partial block (see the module-level metrics-side compensation in
        # compute_metrics, which annualises by the months ACTUALLY
        # simulated here, not the nominal horizon). block_size larger than
        # the whole horizon would floor to 0 simulated months — a
        # degenerate "portfolio" whose every metric is silently 0 — so
        # that case is rejected outright instead.
        n_blocks = horizon_months // block_size
        if n_blocks < 1:
            raise ValueError(
                f"block_size={block_size} exceeds horizon_months={horizon_months} — "
                f"produces zero simulated steps. Reduce block_size or increase the horizon."
            )
        idx = rng.integers(0, n_avail, size=(n_sim, n_blocks))
        gross = block_gross_all[idx]              # (S, n_blocks)

    # ── cumulative price path ─────────────────────────────────────────
    with np.errstate(over="ignore"):
        cum = np.cumprod(gross, axis=1)

    ones = np.ones((n_sim, 1), dtype=np.float64)
    paths = np.hstack([ones, cum])
    elapsed = time.perf_counter() - t0
    log.debug("[SIM] Simulation done: paths=%s  elapsed=%.4fs", paths.shape, elapsed)
    return paths


def sample_monthly_returns(
    weights: np.ndarray,
    returns: np.ndarray,
    n_sim: int,
    horizon_months: int,
    rng: np.random.Generator,
    block_months: int = 12,
) -> np.ndarray:
    """Block-bootstrap MONTHLY portfolio returns, preserving within-block autocorrelation.

    Unlike :func:`simulate` with ``block_size > 1`` — which collapses each
    sampled block into a single gross return, producing one path column per
    block — this keeps every individual month inside the sampled block, so
    monthly cash flows (contributions, withdrawals) can be applied to the
    simulated path. Used by :mod:`engine.lifecycle`.

    Parameters
    ----------
    weights : (n_assets,)
    returns : (n_months_avail, n_assets) historical monthly returns
    n_sim : number of Monte-Carlo paths
    horizon_months : investment horizon in months
    rng : numpy random generator
    block_months : block length in months. Blocks are overlapping — any
        historical month may be the start of a sampled block — which
        maximises sampling diversity versus only aligning to calendar years.

    Returns
    -------
    monthly : (n_sim, horizon_months) array of simple monthly portfolio
        returns. If ``horizon_months`` is not a multiple of *block_months*,
        the final partial block is truncated to length.
    """
    n_months_avail = returns.shape[0]
    n_starts = n_months_avail - block_months + 1
    if n_starts < 1:
        raise ValueError(
            f"Only {n_months_avail} months of history available, but "
            f"block_months={block_months} requires at least {block_months}. "
            f"Reduce block_months or use a portfolio with a longer history."
        )

    if n_sim * horizon_months > 30_000_000:
        raise ValueError(
            f"n_sim × horizon_months = {n_sim * horizon_months:,} exceeds the "
            f"30,000,000 memory guard. Reduce n_sim (currently {n_sim}) or "
            f"the horizon (currently {horizon_months} months)."
        )

    log.debug(
        "[SAMPLE_MONTHLY] n_sim=%d horizon_months=%d block_months=%d "
        "n_months_avail=%d n_starts=%d",
        n_sim, horizon_months, block_months, n_months_avail, n_starts,
    )
    t0 = time.perf_counter()

    with np.errstate(all="ignore"):
        port_monthly = returns @ weights  # (T,)

    n_blocks = -(-horizon_months // block_months)  # ceil division
    starts = rng.integers(0, n_starts, size=(n_sim, n_blocks))  # (S, B)
    offsets = np.arange(block_months)  # (block_months,)
    idx = starts[:, :, None] + offsets[None, None, :]  # (S, B, block_months)

    monthly = port_monthly[idx].reshape(n_sim, n_blocks * block_months)[:, :horizon_months]

    elapsed = time.perf_counter() - t0
    log.debug("[SAMPLE_MONTHLY] done: monthly=%s elapsed=%.4fs", monthly.shape, elapsed)
    return monthly


def simulate_independent(
    weights: np.ndarray,
    returns_list: list[np.ndarray],
    n_sim: int,
    horizon_months: int,
    rng: np.random.Generator,
    block_size: int = 1,
) -> np.ndarray:
    """Bootstrap simulation with **independent** per-asset resampling.

    Unlike :func:`simulate`, each asset's returns are resampled from its
    own history independently, breaking all inter-asset correlations.
    This is the "aggressive cross-validation" mode.

    Parameters
    ----------
    weights : (n_assets,)
    returns_list : list of 1-D arrays, one per asset (lengths may differ)
    n_sim : number of Monte-Carlo paths
    horizon_months : investment horizon in months
    rng : numpy random generator
    block_size : block length (months).  ``1`` = iid, ``> 1`` = block bootstrap.

    Returns
    -------
    paths : ndarray, same shape convention as :func:`simulate`.
    """
    n_assets = len(weights)
    t0 = time.perf_counter()
    log.debug("[SIM_IND] simulate_independent: n_assets=%d  n_sim=%d  "
              "horizon_months=%d  block_size=%d",
              n_assets, n_sim, horizon_months, block_size)

    if block_size <= 1:
        # ── iid monthly: sample each asset independently ──────────────
        # Build portfolio monthly returns by sampling each asset separately
        port_ret = np.zeros((n_sim, horizon_months), dtype=np.float64)
        for a in range(n_assets):
            T_a = len(returns_list[a])
            idx = rng.integers(0, T_a, size=(n_sim, horizon_months))
            sampled = returns_list[a][idx]             # (n_sim, horizon_months)
            port_ret += weights[a] * sampled
        gross = 1.0 + port_ret
    else:
        # ── block bootstrap: per-asset independent block sampling ─────
        n_blocks = horizon_months // block_size
        if n_blocks < 1:
            raise ValueError(
                f"block_size={block_size} exceeds horizon_months={horizon_months} — "
                f"produces zero simulated steps. Reduce block_size or increase the horizon."
            )
        # Pre-compute rolling block gross returns for each asset
        block_gross_per_asset = []
        for a in range(n_assets):
            ret_a = returns_list[a]
            gross_a = 1.0 + ret_a
            T_a = len(gross_a)
            n_avail = T_a - block_size + 1
            if n_avail < 1:
                raise ValueError(
                    f"Asset {a}: only {T_a} months of history, "
                    f"need at least {block_size} for block_size={block_size}"
                )
            log_gross = np.log(gross_a)
            cum_log = np.cumsum(log_gross)
            bg = np.empty(n_avail, dtype=np.float64)
            bg[0] = np.exp(cum_log[block_size - 1])
            if n_avail > 1:
                bg[1:] = np.exp(cum_log[block_size:] - cum_log[:n_avail - 1])
            block_gross_per_asset.append(bg)

        # For each sim, sample blocks independently per asset, then combine
        # Each block_gross is the cum return of that asset over the block.
        # portfolio block_gross = sum_a( w_a * (block_gross_a - 1) ) + 1
        # which equals the weighted portfolio return over the block, +1.
        port_block_gross = np.ones((n_sim, n_blocks), dtype=np.float64)
        for a in range(n_assets):
            n_avail = len(block_gross_per_asset[a])
            idx = rng.integers(0, n_avail, size=(n_sim, n_blocks))
            # block_gross_per_asset[a] is the gross return for asset a
            # portfolio block return += w_a * (asset_block_gross - 1)
            port_block_gross += weights[a] * (block_gross_per_asset[a][idx] - 1.0)
        gross = port_block_gross

    # ── cumulative price path ─────────────────────────────────────────
    with np.errstate(over="ignore"):
        cum = np.cumprod(gross, axis=1)

    ones = np.ones((n_sim, 1), dtype=np.float64)
    paths = np.hstack([ones, cum])
    elapsed = time.perf_counter() - t0
    log.debug("[SIM_IND] Simulation done: paths=%s  elapsed=%.4fs", paths.shape, elapsed)
    return paths
