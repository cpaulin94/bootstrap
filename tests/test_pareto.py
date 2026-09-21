"""
tests.test_pareto — closed-form checks on the Pareto frontier sweep.

Every expected value here is worked out by hand, not read back from the
implementation.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from engine.pareto import compute_pareto, pareto_frontier_indices


def test_empty_input_returns_empty():
    out = pareto_frontier_indices(np.empty((0, 2)))
    assert out.shape == (0,)


def test_single_point_is_its_own_frontier():
    assert pareto_frontier_indices(np.array([[3.0, 7.0]])).tolist() == [0]


def test_hand_computed_2d_frontier():
    # Minimising both. (1,5) and (4,2) are non-dominated; (5,6) is
    # dominated by both, (2,6) is dominated by (1,5).
    costs = np.array([
        [1.0, 5.0],   # 0 — frontier
        [5.0, 6.0],   # 1 — dominated by 0 and 2
        [4.0, 2.0],   # 2 — frontier
        [2.0, 6.0],   # 3 — dominated by 0
    ])
    assert pareto_frontier_indices(costs).tolist() == [0, 2]


def test_weak_domination_is_domination():
    # (2,3) is >= (1,3) on the first objective and equal on the second,
    # so it is dominated even though it ties one objective.
    costs = np.array([[1.0, 3.0], [2.0, 3.0]])
    assert pareto_frontier_indices(costs).tolist() == [0]


def test_exact_duplicates_are_all_kept():
    # Two identical optimal rows: neither dominates the other, so both
    # belong on the frontier.
    costs = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    assert pareto_frontier_indices(costs).tolist() == [0, 1]


def test_all_points_on_frontier_when_perfectly_traded_off():
    costs = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    assert pareto_frontier_indices(costs).tolist() == [0, 1, 2, 3]


def test_compute_pareto_handles_mixed_directions():
    # Maximise the first column, minimise the second.
    #   A: (10, 1) — best on both → frontier
    #   B: ( 5, 1) — A has more return at the same risk → dominated
    #   C: ( 4, 0) — lowest risk of all → frontier
    data = np.array([[10.0, 1.0], [5.0, 1.0], [4.0, 0.0]])
    idx = compute_pareto(data, ["maximize", "minimize"])
    assert sorted(idx.tolist()) == [0, 2]


def test_compute_pareto_does_not_mutate_its_input():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    before = data.copy()
    compute_pareto(data, ["maximize", "minimize"])
    assert np.array_equal(data, before)


def test_five_objective_frontier_matches_brute_force():
    rng = np.random.default_rng(7)
    costs = rng.integers(0, 6, size=(300, 5)).astype(float)

    n = costs.shape[0]
    expected = []
    for i in range(n):
        leq = np.all(costs <= costs[i], axis=1)
        less = np.any(costs < costs[i], axis=1)
        if not np.any(leq & less):
            expected.append(i)

    assert pareto_frontier_indices(costs).tolist() == expected


@pytest.mark.parametrize("n", [50_000])
def test_large_cloud_is_fast(n):
    """The Space Explorer routinely feeds this 100k points on every chart
    refresh; the old O(n²) sweep took minutes there."""
    rng = np.random.default_rng(0)
    costs = rng.random((n, 5))
    t0 = time.perf_counter()
    idx = pareto_frontier_indices(costs)
    elapsed = time.perf_counter() - t0
    assert idx.size > 0
    assert elapsed < 5.0, f"frontier of {n} points took {elapsed:.1f}s"


# ═══════════════════════════════════════════════════════════════════════
# crowding_distance — used by the evolutionary search to bias which
# frontier points get refined around (denser regions sampled less).
# ═══════════════════════════════════════════════════════════════════════

from engine.pareto import crowding_distance


def test_crowding_distance_boundary_points_are_infinite():
    costs = np.array([[0.0], [5.0], [10.0]])
    d = crowding_distance(costs)
    assert d[0] == np.inf
    assert d[2] == np.inf
    # middle point: (10-0)/(10-0) = 1.0, single objective
    assert d[1] == pytest.approx(1.0)


def test_crowding_distance_hand_computed_two_objectives():
    # Three points on the diagonal: for EACH objective, sorted order is
    # the same (0,1,2), boundaries get inf, middle gets (2-0)/(2-0)=1.0.
    # Summed across both objectives, the middle point gets 1.0+1.0=2.0.
    costs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    d = crowding_distance(costs)
    assert d[0] == np.inf
    assert d[2] == np.inf
    assert d[1] == pytest.approx(2.0)


def test_crowding_distance_zero_and_two_points_are_all_infinite():
    assert crowding_distance(np.empty((0, 2))).shape == (0,)
    d1 = crowding_distance(np.array([[1.0, 2.0]]))
    assert d1.tolist() == [np.inf]
    d2 = crowding_distance(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert d2.tolist() == [np.inf, np.inf]


def test_crowding_distance_constant_objective_contributes_nothing():
    """Regression: an objective where every point has the IDENTICAL value
    carries no spread information at all. It must not hand out a spurious
    +inf to two arbitrarily-ordered "boundary" points on that objective —
    only a genuinely varying objective should ever contribute to the
    distance."""
    # obj 0 varies (0,1,2,3); obj 1 is constant (5,5,5,5) for everyone.
    costs = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    d = crowding_distance(costs)
    d_obj0_only = crowding_distance(costs[:, :1])
    np.testing.assert_allclose(d, d_obj0_only)


def test_crowding_distance_denser_region_scores_lower():
    """The actual property the evolutionary search relies on: a point
    sandwiched tightly between close neighbours must score lower than one
    sitting in an isolated stretch of the same frontier."""
    # obj 0 values: 0, 1, 1.01, 1.02, 10 — index 2 and 3 are tightly
    # packed; index 1 sits in a much larger gap on one side.
    costs = np.array([[0.0], [1.0], [1.01], [1.02], [10.0]])
    d = crowding_distance(costs)
    # index 2 (surrounded by 1.0 and 1.02, a tiny span) must score lower
    # than index 1 (surrounded by 0.0 and 1.01, a much larger span).
    assert d[2] < d[1]
