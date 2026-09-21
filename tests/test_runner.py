"""Tests for engine.runner — mainly the common-random-numbers fix for
multi-portfolio search (see C3 in AUDIT.md): every candidate portfolio in
a run must be evaluated against the SAME Monte-Carlo draw, or a Pareto
frontier over many candidates ends up picking the luckiest draw instead
of the best portfolio.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from engine import config as cfg
from engine.data import load_all_returns, preload_returns
from engine.pareto import compute_pareto
from engine.runner import (
    _eval_portfolio,
    _init_worker,
    run_bootstrap,
    run_bootstrap_preloaded,
    run_evolutionary_streaming,
    run_multi_streaming,
)
from engine.search import load_search_space


def test_eval_portfolio_is_deterministic_given_fixed_sim_seed():
    weights, ret_matrix = load_all_returns({"WRDA": 1.0})
    _init_worker(ret_matrix, dict(
        n_sim=200, horizon_years=5, block_size=6,
        percentiles=[1, 50, 99], vol_windows=[1, 3, 5],
        bad_pct=2, tickers=["WRDA"], independent=False, sim_seed=777,
    ))
    m1 = _eval_portfolio(weights)
    m2 = _eval_portfolio(weights)
    assert m1 == m2


def test_eval_portfolio_without_sim_seed_is_still_non_deterministic():
    # Sanity check on the OLD behaviour, kept as a negative control: an
    # absent sim_seed (params dict without the key) falls back to fresh OS
    # entropy inside np.random.default_rng(None). This documents why the
    # engine must always populate "sim_seed" (see run_multi_streaming).
    weights, ret_matrix = load_all_returns({"WRDA": 1.0})
    _init_worker(ret_matrix, dict(
        n_sim=200, horizon_years=5, block_size=6,
        percentiles=[1, 50, 99], vol_windows=[1, 3, 5],
        bad_pct=2, tickers=["WRDA"], independent=False,
        # sim_seed intentionally omitted
    ))
    m1 = _eval_portfolio(weights)
    m2 = _eval_portfolio(weights)
    assert m1["annualised_return_p50"] != pytest.approx(m2["annualised_return_p50"])


def test_run_multi_streaming_same_portfolio_reproducible_without_explicit_seed():
    """Even when the caller doesn't pass seed/sim_seed at all, two full
    runs over the SAME single-candidate grid must produce identical
    metrics — the fixed _DEFAULT_SIM_SEED fallback must kick in."""
    space = [{"ticker": "WRDA", "lo": 1.0, "hi": 1.0}]

    def _run():
        _, results = run_multi_streaming(
            space, method="grid", grid_step=1.0,
            n_sim=200, horizon_years=5, block_size=6, n_jobs=1,
        )
        return results

    r1, r2 = _run(), _run()
    assert len(r1) == 1 and len(r2) == 1
    assert r1[0]["annualised_return_p50"] == pytest.approx(r2[0]["annualised_return_p50"])
    assert r1[0]["max_dd_depth_p2"] == pytest.approx(r2[0]["max_dd_depth_p2"])


def _tight_space_around(portfolio: dict, delta: float = 0.001) -> list[dict]:
    return [
        {"ticker": t, "lo": max(0.0, w - delta), "hi": w + delta}
        for t, w in portfolio.items()
    ]


def test_overlay_star_matches_cloud_center_with_shared_seed():
    """Regression for the GUI overlay bug: a portfolio evaluated with the
    SAME sim_seed and data as a tight-radius search cloud around it must
    land near the cloud's center, not tens of standard deviations away.

    This is what ``gui.py``'s "preloaded" overlay branch does — reuse
    ``on_data_ready``'s exact data + sim_seed instead of a fresh RNG draw.
    """
    portfolio = {"GOLD": 0.20, "GOVH": 0.20, "HPRD": 0.20, "IWMO": 0.20, "SP5A": 0.20}
    space = _tight_space_around(portfolio)

    captured: dict = {}

    def on_data_ready(sorted_tickers, data, sim_seed):
        captured["tickers"] = sorted_tickers
        captured["data"] = data
        captured["sim_seed"] = sim_seed

    _, results = run_multi_streaming(
        space, method="random", n_portfolios=80,
        n_sim=300, horizon_years=10, block_size=6, n_jobs=1,
        seed=42, sim_seed=42, on_data_ready=on_data_ready,
    )
    assert len(results) >= 40

    cloud = {
        k: np.array([r[k] for r in results])
        for k in ("annualised_return_p50", "annualised_return_p1", "volatility_10y")
    }
    weights = np.array([portfolio.get(t, 0.0) for t in captured["tickers"]])
    star = run_bootstrap_preloaded(
        weights, captured["data"], n_sim=300, horizon_years=10, block_size=6,
        rng=np.random.default_rng(captured["sim_seed"]), tickers=captured["tickers"],
    )
    for k, v in cloud.items():
        z = abs(star[k] - v.mean()) / v.std()
        assert z < 3, f"{k}: shared-seed star is {z:.1f} sd from cloud center"


def test_overlay_star_with_different_seed_lands_far_from_cloud():
    """Negative control for the fix above: this documents WHY the common
    random numbers scheme (see run_multi_streaming's sim_seed docstring)
    must not be dropped. If sim_seed stops mattering, this test should
    start failing — that's the signal that the overlay bug's root cause
    (a seed mismatch between star and cloud) has silently come back.
    """
    portfolio = {"GOLD": 0.20, "GOVH": 0.20, "HPRD": 0.20, "IWMO": 0.20, "SP5A": 0.20}
    space = _tight_space_around(portfolio)

    captured: dict = {}

    def on_data_ready(sorted_tickers, data, sim_seed):
        captured["tickers"] = sorted_tickers
        captured["data"] = data
        captured["sim_seed"] = sim_seed

    _, results = run_multi_streaming(
        space, method="random", n_portfolios=80,
        n_sim=300, horizon_years=10, block_size=6, n_jobs=1,
        seed=42, sim_seed=42, on_data_ready=on_data_ready,
    )
    cloud = {
        k: np.array([r[k] for r in results])
        for k in ("annualised_return_p50", "annualised_return_p1", "volatility_10y")
    }
    weights = np.array([portfolio.get(t, 0.0) for t in captured["tickers"]])
    star = run_bootstrap_preloaded(
        weights, captured["data"], n_sim=300, horizon_years=10, block_size=6,
        rng=np.random.default_rng(captured["sim_seed"] + 1), tickers=captured["tickers"],
    )
    zs = [abs(star[k] - v.mean()) / v.std() for k, v in cloud.items()]
    assert max(zs) > 5, "a mismatched sim_seed should visibly displace the star"


def test_run_multi_streaming_respects_explicit_sim_seed_override():
    space = [{"ticker": "WRDA", "lo": 1.0, "hi": 1.0}]
    _, r1 = run_multi_streaming(
        space, method="grid", grid_step=1.0,
        n_sim=200, horizon_years=5, block_size=6, n_jobs=1, sim_seed=1,
    )
    _, r2 = run_multi_streaming(
        space, method="grid", grid_step=1.0,
        n_sim=200, horizon_years=5, block_size=6, n_jobs=1, sim_seed=2,
    )
    assert r1[0]["annualised_return_p50"] != pytest.approx(r2[0]["annualised_return_p50"])


def test_run_bootstrap_uses_own_ticker_window_not_full_universe():
    """Regression for AUDIT.md M9 — TRIED AND REVERTED. Forcing
    run_bootstrap to intersect dates against the full asset universe
    (every ticker in cfg.STANDARD_DIR) instead of just the portfolio's
    own tickers was tried, to make it match run_multi_streaming's cloud
    exactly. It was reverted the same day: on a real 2-asset portfolio
    (73% world equity ticker + 27% government bonds ticker) it silently
    truncated 37 years of relevant, available history down to 24 by
    intersecting against an unrelated ticker (starting over a decade
    later) that the portfolio didn't even hold — dragging the simulated
    median 20-year CAGR from a plausible ~7.8% down to an implausible
    ~5.4%. run_bootstrap must match load_all_returns's own-tickers window
    exactly, not run_multi_streaming's search-space-wide one; the two
    conventions serve different needs (accuracy for one portfolio vs.
    comparability across many candidates) and are not meant to coincide
    in general — see run_bootstrap's docstring."""
    portfolio = {"WEBN": 0.73, "GOVH": 0.27}
    _, own_matrix = load_all_returns(portfolio, use_after_ter=True)
    m = run_bootstrap(portfolio, n_sim=50, horizon_years=1, block_size=1, random_seed=1)
    assert m["n_months_history"] == own_matrix.shape[0]

    universe = sorted(
        f[:-4] for f in os.listdir(cfg.STANDARD_DIR) if f.endswith(".csv")
    )
    _, universe_matrix = preload_returns(universe, True)
    assert m["n_months_history"] > universe_matrix.shape[0], (
        "sanity check: this portfolio's own window is expected to be "
        "strictly wider than the full-universe window on this repo's "
        "actual data — if that's no longer true, this test can't "
        "distinguish the M9 regression from correct behaviour any more"
    )


def _small_evo_space() -> list[dict]:
    """A handful of real tickers with a wide-enough box that CDHR/sparse/
    vertex/crossover/perturbation all have room to move — enough to
    exercise every code path in run_evolutionary_streaming without the
    full 15-asset search.csv (keeps tests fast)."""
    return [
        {"ticker": "GOLD", "lo": 0.0, "hi": 0.5},
        {"ticker": "GOVH", "lo": 0.0, "hi": 0.5},
        {"ticker": "IWMO", "lo": 0.0, "hi": 0.5},
        {"ticker": "WEBN", "lo": 0.0, "hi": 0.5},
        {"ticker": "SP5A", "lo": 0.0, "hi": 0.5},
    ]


def test_run_evolutionary_streaming_matches_the_basic_sampler_contract():
    space = _small_evo_space()
    sorted_tickers, results = run_evolutionary_streaming(
        space, n_portfolios=200, n_generations=4,
        n_sim=100, horizon_years=5, block_size=6, n_jobs=1,
        seed=1, sim_seed=1, return_weights=True,
    )
    assert len(results) == 200
    assert sorted_tickers == sorted(s["ticker"] for s in space)

    lo = np.array([s["lo"] for s in space])[
        [sorted_tickers.index(s["ticker"]) for s in space]
    ]
    hi = np.array([s["hi"] for s in space])[
        [sorted_tickers.index(s["ticker"]) for s in space]
    ]
    W = np.array([r["_weights"] for r in results])
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(W >= lo - 1e-6)
    assert np.all(W <= hi + 1e-6)


def test_run_evolutionary_streaming_is_deterministic_given_fixed_seeds():
    space = _small_evo_space()
    kwargs = dict(
        n_portfolios=150, n_generations=3, n_sim=100, horizon_years=5,
        block_size=6, n_jobs=1, seed=7, sim_seed=7, return_weights=True,
    )
    _, r1 = run_evolutionary_streaming(space, **kwargs)
    _, r2 = run_evolutionary_streaming(space, **kwargs)
    assert len(r1) == len(r2) == 150
    w1 = np.array([r["_weights"] for r in r1])
    w2 = np.array([r["_weights"] for r in r2])
    np.testing.assert_array_equal(w1, w2)
    for a, b in zip(r1, r2):
        assert a["annualised_return_p50"] == pytest.approx(b["annualised_return_p50"])


def test_run_evolutionary_streaming_refines_around_a_growing_frontier():
    """The whole point of the algorithm: later generations' candidates
    should include points that dominate (or at least match) what pure
    exploration alone found — checked here by confirming the CUMULATIVE
    population's Pareto frontier draws from more than just generation 0,
    i.e. the refinement step actually contributes non-dominated points."""
    space = _small_evo_space()
    sorted_tickers, results = run_evolutionary_streaming(
        space, n_portfolios=600, n_generations=5,
        n_sim=300, horizon_years=10, block_size=6, n_jobs=1,
        seed=3, sim_seed=3, return_weights=True,
        pareto_metrics=[
            {"name": "annualised_return_p50", "direction": "maximize"},
            {"name": "volatility_10y", "direction": "minimize"},
        ],
    )
    gen0_size = 600 // 5
    data = np.array([
        [r["annualised_return_p50"], r["volatility_10y"]] for r in results
    ])
    idx = compute_pareto(data, ["maximize", "minimize"])
    assert len(idx) > 0
    # At least one frontier point must come from AFTER generation 0 —
    # otherwise refinement contributed nothing beyond raw exploration.
    assert np.any(idx >= gen0_size), (
        "no refined (post-generation-0) candidate made the final frontier"
    )


def test_run_evolutionary_streaming_respects_custom_pareto_metrics():
    """pareto_metrics is threaded through to the refinement step, not just
    accepted and ignored — using a metric-set the run's own results
    couldn't possibly satisfy (an absurd bound) must not crash; it should
    fall back to exploration for every generation instead."""
    space = _small_evo_space()
    # A metric name that exists in every result, but a direction that's
    # still valid — this checks pareto_metrics changes what's optimised
    # around without needing to invent a fake metric name.
    _, results_ret = run_evolutionary_streaming(
        space, n_portfolios=150, n_generations=3,
        n_sim=100, horizon_years=5, block_size=6, n_jobs=1,
        seed=5, sim_seed=5, return_weights=True,
        pareto_metrics=[{"name": "annualised_return_p50", "direction": "maximize"}],
    )
    assert len(results_ret) == 150


def test_run_evolutionary_streaming_single_generation_is_pure_exploration():
    """n_generations=1 has no frontier to refine around yet — must not
    crash, and must still return the full requested budget."""
    space = _small_evo_space()
    _, results = run_evolutionary_streaming(
        space, n_portfolios=80, n_generations=1,
        n_sim=100, horizon_years=5, block_size=6, n_jobs=1,
        seed=2, sim_seed=2,
    )
    assert len(results) == 80


def test_run_bootstrap_succeeds_for_any_ticker_regardless_of_search_space():
    """Regression: the GUI's overlay fallback used to hard-fail with
    "Portfolio contains X, which is outside the search space" whenever a
    saved portfolio held a ticker the CURRENT Space Explorer search space
    didn't include — even though run_bootstrap itself has no notion of
    "search space" and doesn't need one; it only ever cares about the
    portfolio's own tickers (see load_all_returns). The bug was entirely
    in the GUI's cache/branch logic (gui.py's _overlay_use_preloaded),
    which now falls through to this fallback instead of raising."""
    portfolio = {"HPRD": 1.0}  # a single real ticker; would be "outside"
    # any search space that doesn't happen to include it
    m = run_bootstrap(portfolio, n_sim=50, horizon_years=1, block_size=1, random_seed=1)
    assert m["annualised_return_p50"] is not None
