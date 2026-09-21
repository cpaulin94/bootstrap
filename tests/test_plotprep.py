"""
tests.test_plotprep — array building and point-cloud thinning.
"""

from __future__ import annotations

import numpy as np

from engine.plotprep import metric_keys, results_to_arrays, thin_scatter


# ── metric_keys / results_to_arrays ───────────────────────────────────────

def test_metric_keys_are_sorted_public_and_numeric():
    results = [
        {"beta": 1.0, "alpha": 2.0, "_weights": [0.5, 0.5], "label": "x"},
        {"beta": 3.0, "gamma": 4.0},
    ]
    assert metric_keys(results) == ["alpha", "beta", "gamma"]


def test_metric_keys_of_empty_results_is_empty():
    assert metric_keys([]) == []


def test_missing_metric_becomes_nan_not_zero():
    # A metric absent from a result must NOT read as "scored 0.0" — the
    # chart drops those points instead of plotting them at the origin.
    results = [{"a": 1.0, "b": 2.0}, {"a": 3.0}]
    keys, values, _ = results_to_arrays(results, n_assets=0)
    assert keys == ["a", "b"]
    assert values[0].tolist() == [1.0, 2.0]
    assert values[1, 0] == 3.0
    assert np.isnan(values[1, 1])


def test_weights_matrix_matches_result_order():
    results = [
        {"a": 1.0, "_weights": [0.2, 0.8]},
        {"a": 2.0, "_weights": [0.6, 0.4]},
    ]
    _, _, weights = results_to_arrays(results, n_assets=2)
    assert weights.tolist() == [[0.2, 0.8], [0.6, 0.4]]


def test_result_without_weights_is_nan_filled():
    _, _, weights = results_to_arrays([{"a": 1.0}], n_assets=3)
    assert weights.shape == (1, 3)
    assert np.all(np.isnan(weights))


def test_explicit_key_list_is_honoured():
    results = [{"a": 1.0, "b": 2.0}]
    keys, values, _ = results_to_arrays(results, n_assets=0, keys=["b"])
    assert keys == ["b"]
    assert values.tolist() == [[2.0]]


# ── thin_scatter ──────────────────────────────────────────────────────────

def test_no_thinning_below_the_budget():
    x = np.arange(10.0)
    idx = thin_scatter(x, x, max_points=100)
    assert idx.tolist() == list(range(10))


def test_thinning_respects_the_budget():
    rng = np.random.default_rng(1)
    x, y = rng.random(50_000), rng.random(50_000)
    idx = thin_scatter(x, y, max_points=5_000)
    assert idx.size <= 5_000
    assert idx.size > 4_000        # the budget should be nearly filled
    assert np.all(np.diff(idx) > 0)   # sorted, no duplicates


def test_axis_extremes_always_survive():
    rng = np.random.default_rng(2)
    x, y = rng.random(20_000), rng.random(20_000)
    x[123], y[456] = -5.0, 9.0     # lone outliers, easy to thin away
    idx = thin_scatter(x, y, max_points=1_000)
    assert 123 in idx.tolist()
    assert 456 in idx.tolist()
    assert x[idx].min() == x.min() and x[idx].max() == x.max()
    assert y[idx].min() == y.min() and y[idx].max() == y.max()


def test_sparse_region_is_not_swallowed_by_a_dense_one():
    # 30k points crammed into a corner plus 5 lonely points far away:
    # a plain random sample would likely drop the 5, the grid pass can't.
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.random(30_000) * 0.01, np.linspace(0.5, 1.0, 5)])
    y = np.concatenate([rng.random(30_000) * 0.01, np.linspace(0.5, 1.0, 5)])
    idx = set(thin_scatter(x, y, max_points=500).tolist())
    assert {30_000, 30_001, 30_002, 30_003, 30_004} <= idx


def test_thinning_is_reproducible():
    rng = np.random.default_rng(4)
    x, y = rng.random(30_000), rng.random(30_000)
    a = thin_scatter(x, y, max_points=2_000)
    b = thin_scatter(x, y, max_points=2_000)
    assert np.array_equal(a, b)


def test_constant_axis_does_not_crash():
    x = np.full(5_000, 0.3)
    y = np.linspace(0.0, 1.0, 5_000)
    idx = thin_scatter(x, y, max_points=200)
    assert 0 < idx.size <= 200
