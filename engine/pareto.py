"""
engine.pareto — N-dimensional Pareto frontier utilities.
"""

from __future__ import annotations

import numpy as np


def pareto_frontier_indices(costs: np.ndarray) -> np.ndarray:
    """Return row-indices of Pareto-optimal (non-dominated) points.

    All objectives in *costs* are assumed to be **minimised**.
    Negate maximisation objectives before calling this function.

    Ties are kept: rows whose cost vector is identical to a frontier row
    are all returned, not just the first one.

    Implementation note: this used to be a per-row scan that rebuilt a
    fancy-indexed copy of the whole cost matrix on every iteration —
    O(n²) work *and* O(n²) memory traffic, which at the 100k points the
    Space Explorer routinely produces meant minutes of wall clock inside
    a chart refresh. The sweep below discards every point dominated by
    the current one before moving on, so the candidate set collapses to
    (roughly) the frontier after a handful of passes.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2:
        raise ValueError("costs must be a 2-D (n_points, n_objectives) array")
    if costs.shape[0] == 0:
        return np.empty(0, dtype=np.intp)

    # Collapse exact duplicates first. The sweep treats "equal in every
    # objective" as dominated (otherwise no point could ever eliminate its
    # own copies), so deciding on the unique rows and mapping back is what
    # preserves the tie semantics above. np.unique also returns the rows
    # lexicographically sorted, which is the order the sweep likes best.
    uniq, inverse = np.unique(costs, axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)

    idx = np.arange(uniq.shape[0], dtype=np.intp)
    pts = uniq
    i = 0
    while i < pts.shape[0]:
        # Keep the points that beat pts[i] on at least one objective; the
        # rest are dominated by it (or equal to it) and can never return.
        keep = np.any(pts < pts[i], axis=1)
        keep[i] = True
        if keep.all():
            i += 1
        else:
            n_before = int(np.count_nonzero(keep[:i]))
            idx = idx[keep]
            pts = pts[keep]
            i = n_before + 1

    efficient = np.zeros(uniq.shape[0], dtype=bool)
    efficient[idx] = True
    return np.flatnonzero(efficient[inverse])


def compute_pareto(
    data: np.ndarray,
    directions: list[str],
) -> np.ndarray:
    """Convenience wrapper accepting mixed maximize / minimize objectives.

    Parameters
    ----------
    data       : (n_points, n_objectives)
    directions : list of ``"maximize"`` or ``"minimize"`` per objective

    Returns
    -------
    Indices of Pareto-optimal rows in the original *data* array.
    """
    costs = np.array(data, dtype=float, copy=True)
    for j, d in enumerate(directions):
        if d == "maximize":
            costs[:, j] = -costs[:, j]
    return pareto_frontier_indices(costs)


def crowding_distance(costs: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance: how isolated each point is from its
    neighbours along each objective, summed across objectives.

    All objectives assumed **minimised**, same convention as
    :func:`pareto_frontier_indices` — negate maximisation objectives first.

    Boundary points (the best and worst on any single objective) get
    ``+inf`` — always maximally interesting to keep. Interior points get
    the sum, over objectives, of the normalised gap between their two
    neighbours on that objective (0 = surrounded on all sides, large =
    sitting in a sparse stretch of the frontier).

    Used to bias which frontier points an evolutionary search spends its
    next generation's budget refining around: sampling PROPORIONALLY to
    crowding distance pushes new candidates toward under-explored/extreme
    regions of the frontier instead of piling more samples where it's
    already dense — the frontier grows outward instead of just thicker.

    An objective with zero range (every point identical on it) is skipped
    for that objective — it carries no information about spread and would
    otherwise divide by zero.
    """
    costs = np.asarray(costs, dtype=float)
    n, m = costs.shape
    if n == 0:
        return np.empty(0, dtype=float)
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n, dtype=float)
    for j in range(m):
        order = np.argsort(costs[:, j], kind="stable")
        vals = costs[order, j]
        span = vals[-1] - vals[0]
        if span <= 1e-12:
            continue  # no spread on this objective: it contributes nothing,
                      # including no spurious +inf to two arbitrary tied points
        d = np.zeros(n, dtype=float)
        d[0] = np.inf
        d[-1] = np.inf
        d[1:-1] = (vals[2:] - vals[:-2]) / span
        dist[order] += d
    return dist
