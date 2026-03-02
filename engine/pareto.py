"""
engine.pareto — N-dimensional Pareto frontier utilities.
"""

from __future__ import annotations

import numpy as np


def pareto_frontier_indices(costs: np.ndarray) -> np.ndarray:
    """Return row-indices of Pareto-optimal (non-dominated) points.

    All objectives in *costs* are assumed to be **minimised**.
    Negate maximisation objectives before calling this function.
    """
    n = costs.shape[0]
    order = np.argsort(costs[:, 0])
    costs = costs[order]

    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        rest = is_efficient.copy()
        rest[i] = False
        if not rest.any():
            break
        remaining = costs[rest]
        leq = np.all(remaining <= costs[i], axis=1)
        strictly = np.any(remaining < costs[i], axis=1)
        if np.any(leq & strictly):
            is_efficient[i] = False

    return order[is_efficient]


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
    costs = data.copy()
    for j, d in enumerate(directions):
        if d == "maximize":
            costs[:, j] = -costs[:, j]
    return pareto_frontier_indices(costs)
