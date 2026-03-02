"""
engine.search — Portfolio-space samplers and search-space loading.
"""

from __future__ import annotations

import csv

import numpy as np


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

    results = np.empty((n, k), dtype=np.float64)
    filled = 0
    batch = max(n * 2, 10_000)

    while filled < n:
        raw = rng.uniform(lo, hi, size=(batch, k))
        row_sums = raw.sum(axis=1, keepdims=True)
        normed = raw / row_sums

        valid = np.all((normed >= lo) & (normed <= hi), axis=1)
        good = normed[valid]

        take = min(len(good), n - filled)
        if take > 0:
            results[filled : filled + take] = good[:take]
            filled += take

    return results


def sample_grid_portfolios(
    space: list[dict],
    step: float,
) -> np.ndarray:
    """Enumerate all weight combos on the simplex with given step size."""
    k = len(space)
    lo = [s["lo"] for s in space]
    hi = [s["hi"] for s in space]

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
    return np.array(results, dtype=np.float64) if results else np.empty((0, k))
