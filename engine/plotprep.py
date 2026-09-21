"""
engine.plotprep — turn a run's result dicts into plot-ready numeric arrays,
and thin a dense point cloud down to something a browser can actually draw.

The Space Explorer produces up to a few hundred thousand result dicts. Two
things about that used to make the interactive chart unusable:

  * every point carried its own pre-rendered HTML hover string and a
    per-point ``{ticker: weight}`` dict, so the figure serialised to
    ~1.4 kB *per point* (a 100k-portfolio run wrote a 137 MB HTML file);
  * every point was actually handed to the browser, even though a 1400 x
    850 px plot has ~1.2M pixels and the marker size is 5 px — most of
    those points land on a pixel that is already painted.

``results_to_arrays`` fixes the first (one numeric matrix, one shared
hover template, base64-serialised by plotly instead of megabytes of JSON
strings) and ``thin_scatter`` fixes the second.

Both are pure numpy and know nothing about plotly or Tkinter.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("bootstrap.plotprep")


def metric_keys(results: list[dict]) -> list[str]:
    """Sorted public, numeric metric names present in *results*.

    Keys starting with ``_`` are internals (e.g. ``_weights``) and keys
    whose value is not a real number (booleans included) can't go in a
    float matrix, so both are left out.
    """
    if not results:
        return []
    names: set[str] = set()
    for r in results:
        names.update(r.keys())

    keys = []
    for name in sorted(names):
        if name.startswith("_"):
            continue
        for r in results:
            if name in r:
                v = r[name]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    keys.append(name)
                break
    return keys


def results_to_arrays(
    results: list[dict],
    n_assets: int,
    keys: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Vectorise *results* into ``(keys, values, weights)``.

    ``values`` is (n_results, len(keys)) float64, ``np.nan`` where a
    result simply doesn't carry that metric (a volatility window longer
    than the run's horizon, say) — never 0, which would read as "scored
    zero on that axis".

    ``weights`` is (n_results, n_assets) float64, ``np.nan`` for results
    that were produced without ``return_weights``.
    """
    keys = metric_keys(results) if keys is None else list(keys)
    n = len(results)

    values = np.full((n, len(keys)), np.nan, dtype=np.float64)
    nan = float("nan")
    for j, key in enumerate(keys):
        values[:, j] = [r.get(key, nan) for r in results]

    weights = np.full((n, n_assets), np.nan, dtype=np.float64)
    for i, r in enumerate(results):
        w = r.get("_weights")
        if w is not None and len(w) == n_assets:
            weights[i] = w

    return keys, values, weights


def thin_scatter(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    *,
    aspect: float = 1400 / 850,
    seed: int = 0,
) -> np.ndarray:
    """Indices of a subset of (x, y) that *looks* like the full cloud.

    Returns every index when there are already few enough points.

    The cloud is binned onto a grid sized so it holds about *max_points*
    cells, and one point is kept per occupied cell. That alone preserves
    the cloud's silhouette — no occupied region can disappear — but it
    flattens the density shading, so whatever budget is left over is
    filled with a uniform random sample of the discarded points, which
    keeps dense regions visibly darker than sparse ones. The four axis
    extremes are always kept so thinning never changes the plot range.

    *seed* keeps the choice reproducible: re-plotting the same cloud on
    the same axes gives the same picture rather than reshuffling under
    the user on every refresh.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.intp)

    gx = max(1, int(np.sqrt(max_points * aspect)))
    gy = max(1, int(max_points // gx))

    def _bin(v: np.ndarray, g: int) -> np.ndarray:
        lo, hi = float(np.min(v)), float(np.max(v))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return np.zeros(v.size, dtype=np.int64)
        return np.clip(((v - lo) / span * g).astype(np.int64), 0, g - 1)

    cells = _bin(x, gx) * gy + _bin(y, gy)
    _, first = np.unique(cells, return_index=True)

    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    keep[[int(np.argmin(x)), int(np.argmax(x)),
          int(np.argmin(y)), int(np.argmax(y))]] = True

    budget = max_points - int(keep.sum())
    if budget > 0:
        rest = np.flatnonzero(~keep)
        if rest.size:
            rng = np.random.default_rng(seed)
            extra = rng.choice(rest, size=min(budget, rest.size), replace=False)
            keep[extra] = True

    idx = np.flatnonzero(keep)
    log.info("[PLOTPREP] Thinned %d points → %d (grid %dx%d, budget %d)",
             n, idx.size, gx, gy, max_points)
    return idx
