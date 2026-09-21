"""Tests for engine.metrics — entropy, percentiles, volatility, drawdown."""

from __future__ import annotations

import numpy as np
import pytest

from engine.metrics import (
    _max_drawdown_area,
    _max_drawdown_depth,
    _max_drawdown_length,
    _drawdown_series,
    compute_metrics,
    effective_n_assets,
)


# ── Shannon entropy ──────────────────────────────────────────────────────────

def test_effective_n_assets_single_asset_is_one():
    assert effective_n_assets(np.array([1.0])) == pytest.approx(1.0)


def test_effective_n_assets_equal_weights_is_the_asset_count():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert effective_n_assets(w) == pytest.approx(len(w), abs=1e-6)


def test_effective_n_assets_concentrated_below_equal():
    equal = effective_n_assets(np.array([0.25, 0.25, 0.25, 0.25]))
    concentrated = effective_n_assets(np.array([0.9, 0.05, 0.03, 0.02]))
    assert 0.0 <= concentrated < equal


def test_effective_n_assets_ignores_zero_weights():
    # Zero-weight entries must not produce NaN (0 * log(0) undefined).
    w = np.array([0.5, 0.5, 0.0, 0.0])
    val = effective_n_assets(w)
    assert np.isfinite(val)


def test_effective_n_assets_is_independent_of_zero_padding():
    """The metric must describe the PORTFOLIO, not the weight vector it
    happens to be encoded in. The old normalised shannon_entropy divided
    by ln(len(weights)), so the same holdings scored 0.84 from the Single
    Bootstrap section (2-long vector) and 0.22 from a Space Explorer run
    (15-long zero-padded vector) — the same class of cross-tool
    disagreement as evaluating them on different date windows."""
    held = np.array([0.73, 0.27])
    for pad in (0, 3, 13):
        padded = np.concatenate([held, np.zeros(pad)])
        assert effective_n_assets(padded) == pytest.approx(
            effective_n_assets(held), abs=1e-9
        )


def test_effective_n_assets_still_ranks_diversification_correctly():
    """exp(H) is a monotone transform of H, so within one run (where the
    old denominator was constant) every ranking is unchanged — an
    equal-weight 5-asset portfolio must still beat an equal-weight
    3-asset one, which the ln(n_held)-normalised alternative would not."""
    eq3 = np.concatenate([np.ones(3) / 3, np.zeros(12)])
    eq5 = np.concatenate([np.ones(5) / 5, np.zeros(10)])
    assert effective_n_assets(eq5) > effective_n_assets(eq3)
    assert effective_n_assets(eq3) == pytest.approx(3.0, abs=1e-6)
    assert effective_n_assets(eq5) == pytest.approx(5.0, abs=1e-6)


# ── Drawdown: naive reference vs. engine implementation ─────────────────────
#
# These naive loops mirror the ORIGINAL engine.metrics implementation
# (pre-vectorisation) and serve as an independent ground truth so that any
# future optimisation of _max_drawdown_length / _max_drawdown_area can be
# verified for exact equivalence.

def _naive_max_drawdown_length(paths: np.ndarray, bad_pct: float, block_size: int = 1) -> int:
    running_max = np.maximum.accumulate(paths, axis=1)
    in_dd = paths < running_max
    n_sim = in_dd.shape[0]
    max_lengths = np.zeros(n_sim, dtype=np.int64)
    for i in range(n_sim):
        row = in_dd[i]
        length = 0
        best = 0
        for v in row:
            if v:
                length += 1
                best = max(best, length)
            else:
                length = 0
        max_lengths[i] = best
    steps = int(np.percentile(max_lengths, 100 - bad_pct))
    return steps * block_size


def _naive_max_drawdown_area(dd: np.ndarray, paths: np.ndarray, bad_pct: float,
                              block_size: int = 1) -> float:
    running_max = np.maximum.accumulate(paths, axis=1)
    in_dd = paths < running_max
    n_sim = in_dd.shape[0]
    mda_per_sim = np.zeros(n_sim, dtype=np.float64)
    for i in range(n_sim):
        row_dd = dd[i]
        row_in = in_dd[i]
        best_area = 0.0
        current_area = 0.0
        for t in range(len(row_in)):
            if row_in[t]:
                current_area += row_dd[t]
            else:
                best_area = max(best_area, current_area)
                current_area = 0.0
        best_area = max(best_area, current_area)
        mda_per_sim[i] = best_area
    raw = float(np.percentile(mda_per_sim, 100 - bad_pct))
    return round(raw * block_size, 6)


def _random_paths(seed: int, n_sim: int = 50, n_steps: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    monthly = rng.normal(0.004, 0.03, size=(n_sim, n_steps))
    gross = 1.0 + monthly
    cum = np.cumprod(gross, axis=1)
    ones = np.ones((n_sim, 1))
    return np.hstack([ones, cum])


@pytest.mark.parametrize("seed", range(20))
def test_max_drawdown_length_matches_naive_reference(seed):
    paths = _random_paths(seed)
    engine_val = _max_drawdown_length(paths, bad_pct=2, block_size=1)
    naive_val = _naive_max_drawdown_length(paths, bad_pct=2, block_size=1)
    assert engine_val == naive_val


@pytest.mark.parametrize("seed", range(20))
def test_max_drawdown_area_matches_naive_reference(seed):
    paths = _random_paths(seed)
    dd = _drawdown_series(paths)
    engine_val = _max_drawdown_area(dd, paths, bad_pct=2, block_size=1)
    naive_val = _naive_max_drawdown_area(dd, paths, bad_pct=2, block_size=1)
    assert engine_val == pytest.approx(naive_val, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 5, 11])
@pytest.mark.parametrize("block_size", [1, 3, 6])
def test_max_drawdown_length_matches_naive_reference_block_size(seed, block_size):
    paths = _random_paths(seed)
    engine_val = _max_drawdown_length(paths, bad_pct=2, block_size=block_size)
    naive_val = _naive_max_drawdown_length(paths, bad_pct=2, block_size=block_size)
    assert engine_val == naive_val


@pytest.mark.parametrize("seed", [0, 5, 11])
@pytest.mark.parametrize("block_size", [1, 3, 6])
def test_max_drawdown_area_matches_naive_reference_block_size(seed, block_size):
    paths = _random_paths(seed)
    dd = _drawdown_series(paths)
    engine_val = _max_drawdown_area(dd, paths, bad_pct=2, block_size=block_size)
    naive_val = _naive_max_drawdown_area(dd, paths, bad_pct=2, block_size=block_size)
    assert engine_val == pytest.approx(naive_val, abs=1e-6)


def test_max_drawdown_depth_is_between_zero_and_one():
    paths = _random_paths(seed=1)
    dd = _drawdown_series(paths)
    depth = _max_drawdown_depth(dd, bad_pct=2)
    assert 0.0 <= depth <= 1.0


# ── compute_metrics: shape / keys ────────────────────────────────────────────

def test_compute_metrics_contains_expected_keys():
    paths = _random_paths(seed=2, n_sim=200, n_steps=120)
    weights = np.array([0.6, 0.4])
    metrics = compute_metrics(
        paths,
        horizon_years=10,
        horizon_months=120,
        block_size=1,
        percentiles=[1, 50, 99],
        vol_windows=[1, 5, 10],
        bad_pct=2,
        weights=weights,
        tickers=["MWEQ", "GOVH"],
    )
    for key in (
        "annualised_return_p1", "annualised_return_p50", "annualised_return_p99",
        "volatility_1y", "volatility_5y", "volatility_10y",
        "max_dd_depth_p2", "max_dd_length_months_p2", "mda_months_p2",
        "effective_n_assets", "effective_n_types",
    ):
        assert key in metrics, f"missing key: {key}"


# ── compute_metrics: block_size truncation annualises against actual months ──

def test_compute_metrics_annualises_against_actual_months_simulated():
    """A block_size that doesn't evenly divide horizon_months truncates the
    final partial block (see engine.simulation.simulate). compute_metrics
    must annualise against the months ACTUALLY simulated
    (paths.shape[1] - 1) * block_size, not the nominal horizon_months —
    otherwise identical per-month growth reports a different annualised
    return purely because of the block_size, which is a bootstrap-sampling
    knob and shouldn't change the answer for a deterministic path.
    """
    n_sim = 10
    monthly_growth = 1.01               # deterministic +1%/month
    # 9 blocks of 13 months = 117 months actually simulated (nominal
    # horizon is 120 months, so block_size=13 doesn't divide it evenly).
    n_steps = 9
    gross = np.full((n_sim, n_steps), monthly_growth ** 13)
    cum = np.cumprod(gross, axis=1)
    paths = np.hstack([np.ones((n_sim, 1)), cum])

    metrics = compute_metrics(
        paths, horizon_years=10, horizon_months=120, block_size=13,
        percentiles=[50], vol_windows=[], bad_pct=2,
    )
    expected_ann = monthly_growth ** 12 - 1.0
    assert metrics["annualised_return_p50"] == pytest.approx(expected_ann, abs=1e-6)
    # Sanity: using the WRONG (nominal) horizon would have given a
    # different, incorrect number for the same deterministic path.
    wrong_ann = paths[0, -1] ** (1.0 / 10) - 1.0
    assert metrics["annualised_return_p50"] != pytest.approx(wrong_ann, abs=1e-6)


def test_compute_metrics_raises_on_zero_simulated_steps():
    paths = np.ones((5, 1))   # only the t=0 column — nothing was simulated
    with pytest.raises(ValueError):
        compute_metrics(paths, horizon_years=10, horizon_months=120, block_size=1,
                        percentiles=[50], vol_windows=[], bad_pct=2)
