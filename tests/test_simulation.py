"""Tests for engine.simulation — the vectorised Monte-Carlo bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from engine.simulation import simulate, simulate_independent, sample_monthly_returns


def _fake_returns(seed: int, n_months: int = 240, n_assets: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.004, 0.03, size=(n_months, n_assets))


def test_simulate_shape_block_size_one():
    returns = _fake_returns(0)
    weights = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(42)
    paths = simulate(weights, returns, n_sim=100, horizon_months=60, rng=rng, block_size=1)
    assert paths.shape == (100, 61)
    assert np.all(paths[:, 0] == 1.0)


def test_simulate_shape_block_size_greater_than_one():
    returns = _fake_returns(0)
    weights = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(42)
    horizon_months = 60
    block_size = 6
    paths = simulate(weights, returns, n_sim=100, horizon_months=horizon_months,
                      rng=rng, block_size=block_size)
    n_blocks = horizon_months // block_size
    assert paths.shape == (100, n_blocks + 1)
    assert np.all(paths[:, 0] == 1.0)


def test_simulate_paths_are_positive_and_finite():
    returns = _fake_returns(1)
    weights = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(7)
    paths = simulate(weights, returns, n_sim=50, horizon_months=36, rng=rng, block_size=1)
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0)


def test_simulate_determinism_same_seed():
    returns = _fake_returns(2)
    weights = np.array([0.4, 0.4, 0.2])
    paths_a = simulate(weights, returns, n_sim=200, horizon_months=48,
                        rng=np.random.default_rng(123), block_size=3)
    paths_b = simulate(weights, returns, n_sim=200, horizon_months=48,
                        rng=np.random.default_rng(123), block_size=3)
    np.testing.assert_array_equal(paths_a, paths_b)


def test_simulate_different_seeds_diverge():
    returns = _fake_returns(3)
    weights = np.array([0.4, 0.4, 0.2])
    paths_a = simulate(weights, returns, n_sim=200, horizon_months=48,
                        rng=np.random.default_rng(1), block_size=1)
    paths_b = simulate(weights, returns, n_sim=200, horizon_months=48,
                        rng=np.random.default_rng(2), block_size=1)
    assert not np.array_equal(paths_a, paths_b)


def test_simulate_single_month_history_matches_closed_form():
    # With one month of history at return r and block_size=1, every sampled
    # month must equal r, so the path is deterministic: V0 * (1+r)**H.
    r = 0.01
    returns = np.array([[r]])
    weights = np.array([1.0])
    rng = np.random.default_rng(0)
    horizon = 12
    paths = simulate(weights, returns, n_sim=10, horizon_months=horizon, rng=rng, block_size=1)
    expected_final = (1.0 + r) ** horizon
    assert np.allclose(paths[:, -1], expected_final)


def test_sample_monthly_returns_shape_and_range():
    weights = np.array([0.5, 0.3, 0.2])
    returns = _fake_returns(0, n_months=48)
    rng = np.random.default_rng(1)
    monthly = sample_monthly_returns(weights, returns, n_sim=64, horizon_months=30,
                                      rng=rng, block_months=12)
    assert monthly.shape == (64, 30)
    port_ret = returns @ weights
    assert monthly.min() >= port_ret.min() - 1e-9
    assert monthly.max() <= port_ret.max() + 1e-9


def test_sample_monthly_returns_preserves_within_block_order():
    # A single block equal to the whole 12-month history: every sampled
    # block must reproduce those 12 months in the SAME order (that's what
    # "preserving autocorrelation" means, as opposed to iid monthly shuffle).
    weights = np.array([1.0])
    returns = np.arange(12, dtype=np.float64).reshape(12, 1) / 100.0  # 0.00, 0.01, ..., 0.11
    rng = np.random.default_rng(2)
    monthly = sample_monthly_returns(weights, returns, n_sim=5, horizon_months=12,
                                      rng=rng, block_months=12)
    expected_block = returns[:, 0]
    for row in monthly:
        np.testing.assert_allclose(row, expected_block)


def test_sample_monthly_returns_determinism():
    weights = np.array([0.5, 0.5])
    returns = _fake_returns(3, n_months=36, n_assets=2)
    a = sample_monthly_returns(weights, returns, n_sim=50, horizon_months=24,
                                rng=np.random.default_rng(9), block_months=6)
    b = sample_monthly_returns(weights, returns, n_sim=50, horizon_months=24,
                                rng=np.random.default_rng(9), block_months=6)
    np.testing.assert_array_equal(a, b)


def test_sample_monthly_returns_insufficient_history_raises():
    weights = np.array([1.0])
    returns = np.zeros((5, 1))
    with pytest.raises(ValueError):
        sample_monthly_returns(weights, returns, n_sim=10, horizon_months=12,
                                rng=np.random.default_rng(0), block_months=12)


def test_simulate_independent_shape():
    returns_list = [_fake_returns(i, n_months=100 + i * 10, n_assets=1)[:, 0] for i in range(3)]
    weights = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(9)
    paths = simulate_independent(weights, returns_list, n_sim=40, horizon_months=24,
                                  rng=rng, block_size=1)
    assert paths.shape == (40, 25)
    assert np.all(paths[:, 0] == 1.0)
    assert np.all(np.isfinite(paths))
