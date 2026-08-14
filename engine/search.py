"""
engine.search — Portfolio-space samplers and search-space loading.
"""

from __future__ import annotations

import csv
import logging
import time

import numpy as np

log = logging.getLogger("bootstrap.search")


def load_search_space(path: str) -> list[dict]:
    """Parse search.csv → list of ``{ticker, lo, hi}``."""
    space: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            space.append({
                "ticker": row[0].strip().upper(),
                "lo":     float(row[1].strip()),
                "hi":     float(row[2].strip()),
            })
    return space


def sample_random_portfolios(
    space: list[dict],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *n* weight vectors uniformly on the constrained simplex.

    Returns (n, k) array of weights that sum to 1.
    """
    k = len(space)
    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])

    log.info("[RANDOM] Sampling %d portfolios from %d-asset simplex", n, k)
    log.info("[RANDOM] Bounds: lo=%s  hi=%s", lo, hi)
    log.info("[RANDOM] Sum of lo=%.4f  sum of hi=%.4f  (need room for sum=1.0)",
             lo.sum(), hi.sum())
    if lo.sum() > 1.0 + 1e-9:
        log.error("[RANDOM] ⚠️ Sum of lower bounds (%.4f) > 1.0 — "
                  "impossible to satisfy simplex constraint!", lo.sum())
    if hi.sum() < 1.0 - 1e-9:
        log.error("[RANDOM] ⚠️ Sum of upper bounds (%.4f) < 1.0 — "
                  "impossible to satisfy simplex constraint!", hi.sum())

    t0 = time.perf_counter()
    results = np.empty((n, k), dtype=np.float64)
    filled = 0
    batch = max(n * 2, 10_000)
    total_attempts = 0
    rejection_rounds = 0

    while filled < n:
        raw = rng.uniform(lo, hi, size=(batch, k))
        row_sums = raw.sum(axis=1, keepdims=True)
        normed = raw / row_sums

        valid = np.all((normed >= lo) & (normed <= hi), axis=1)
        good = normed[valid]
        total_attempts += batch
        rejection_rounds += 1

        take = min(len(good), n - filled)
        if take > 0:
            results[filled : filled + take] = good[:take]
            filled += take

        if rejection_rounds % 10 == 0:
            acceptance = filled / total_attempts * 100
            log.info("[RANDOM] Progress: %d/%d filled after %d attempts "
                     "(%.1f%% acceptance, round %d)",
                     filled, n, total_attempts, acceptance, rejection_rounds)
        if rejection_rounds > 200 and filled == 0:
            log.error("[RANDOM] 0 valid portfolios after %d attempts — "
                      "search space is likely infeasible!", total_attempts)
            break

    elapsed = time.perf_counter() - t0
    acceptance = filled / max(total_attempts, 1) * 100
    log.info("[RANDOM] Done: %d portfolios in %.3fs  "
             "(%d attempts, %.1f%% acceptance rate, %d rounds)",
             filled, elapsed, total_attempts, acceptance, rejection_rounds)

    return results


def sample_grid_portfolios(
    space: list[dict],
    step: float,
) -> np.ndarray:
    """Enumerate all weight combos on the simplex with given step size."""
    k = len(space)
    lo = [s["lo"] for s in space]
    hi = [s["hi"] for s in space]
    log.info("[GRID] Enumerating grid portfolios: %d assets, step=%.4f", k, step)
    log.info("[GRID] Bounds: lo=%s  hi=%s", lo, hi)
    t0 = time.perf_counter()

    results: list[list[float]] = []

    def _recurse(idx: int, current: list[float], remaining: float) -> None:
        if idx == k - 1:
            w = round(remaining, 10)
            if lo[idx] - 1e-9 <= w <= hi[idx] + 1e-9:
                results.append(current + [w])
            return

        future_min = sum(lo[idx + 1 :])
        future_max = sum(hi[idx + 1 :])

        w_lo = max(lo[idx], remaining - future_max)
        w_hi = min(hi[idx], remaining - future_min)

        if w_lo > w_hi + 1e-9:
            return

        w = w_lo
        while w <= w_hi + 1e-9:
            _recurse(idx + 1, current + [round(w, 10)], remaining - round(w, 10))
            w += step

    _recurse(0, [], 1.0)
    elapsed = time.perf_counter() - t0
    log.info("[GRID] Done: %d portfolios enumerated in %.3fs", len(results), elapsed)
    return np.array(results, dtype=np.float64) if results else np.empty((0, k))
