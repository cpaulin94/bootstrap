"""Tests for engine.search — portfolio-space samplers on the simplex."""

from __future__ import annotations

import numpy as np
import pytest

from engine.search import (
    sample_random_portfolios,
    sample_cdhr_portfolios,
    sample_sparse_portfolios,
    sample_vertex_portfolios,
    sample_mixed_portfolios,
    crossover_portfolios,
    local_perturb_portfolios,
)

ALL_SAMPLERS = [
    sample_random_portfolios,
    sample_cdhr_portfolios,
    sample_sparse_portfolios,
    sample_vertex_portfolios,
    sample_mixed_portfolios,
]


def _space(k: int, lo: float = 0.0, hi: float = 1.0) -> list[dict]:
    return [{"ticker": f"T{i}", "lo": lo, "hi": hi} for i in range(k)]


def test_sample_random_portfolios_rows_sum_to_one():
    rng = np.random.default_rng(0)
    w = sample_random_portfolios(_space(5), 200, rng)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_sample_random_portfolios_respects_bounds():
    space = [
        {"ticker": "A", "lo": 0.1, "hi": 0.5},
        {"ticker": "B", "lo": 0.0, "hi": 0.3},
        {"ticker": "C", "lo": 0.2, "hi": 0.6},
    ]
    rng = np.random.default_rng(1)
    w = sample_random_portfolios(space, 300, rng)
    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])
    assert np.all(w >= lo - 1e-9)
    assert np.all(w <= hi + 1e-9)


def test_sample_random_portfolios_fills_narrow_per_asset_bounds():
    """Real search spaces for this tool routinely constrain every asset to
    a narrow 1-2 percentage-point band (e.g. lo=0.29/hi=0.31). A uniform-
    on-the-full-simplex sampler (Dirichlet(1,...,1) + rejection — tried and
    reverted, see the function docstring) has a near-zero acceptance rate
    against a box this narrow and fails to fill the request at all; this
    must still reliably fill it."""
    space = [
        {"ticker": "A", "lo": 0.0, "hi": 0.01},
        {"ticker": "B", "lo": 0.0, "hi": 0.01},
        {"ticker": "C", "lo": 0.09, "hi": 0.11},
        {"ticker": "D", "lo": 0.19, "hi": 0.21},
        {"ticker": "E", "lo": 0.0, "hi": 0.01},
        {"ticker": "F", "lo": 0.29, "hi": 0.31},
        {"ticker": "G", "lo": 0.29, "hi": 0.31},
        {"ticker": "H", "lo": 0.0, "hi": 0.01},
        {"ticker": "I", "lo": 0.0, "hi": 0.01},
        {"ticker": "J", "lo": 0.0, "hi": 0.01},
        {"ticker": "K", "lo": 0.0, "hi": 0.01},
        {"ticker": "L", "lo": 0.09, "hi": 0.11},
        {"ticker": "M", "lo": 0.0, "hi": 0.01},
        {"ticker": "N", "lo": 0.0, "hi": 0.01},
        {"ticker": "O", "lo": 0.0, "hi": 0.01},
    ]
    rng = np.random.default_rng(5)
    w = sample_random_portfolios(space, 500, rng)
    assert len(w) == 500
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_sample_random_portfolios_infeasible_bounds_returns_empty_gracefully():
    # sum(lo) > 1.0 -> impossible to satisfy; must not hang, must return
    # without raising (the caller checks len(result) == 0).
    space = [{"ticker": "A", "lo": 0.7, "hi": 0.9}, {"ticker": "B", "lo": 0.7, "hi": 0.9}]
    rng = np.random.default_rng(4)
    w = sample_random_portfolios(space, 100, rng)
    assert len(w) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Shared contract across all five samplers
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLER_IDS = [fn.__name__ for fn in ALL_SAMPLERS]


def _real_search_space() -> list[dict]:
    """The shape of this tool's actual search.csv: 15 assets, lo=0, hi=0.30."""
    return [{"ticker": f"A{i}", "lo": 0.0, "hi": 0.30} for i in range(15)]


def _tight_band_space() -> list[dict]:
    """Every bound within a 1-2 percentage-point band — the real search
    space that made Dirichlet(1,...,1)+rejection's acceptance rate ~0%
    (AUDIT.md M4). Mixes lo=0 (free) and lo>0 (mandatory) assets."""
    return [
        {"ticker": "A", "lo": 0.0, "hi": 0.01},
        {"ticker": "B", "lo": 0.0, "hi": 0.01},
        {"ticker": "C", "lo": 0.09, "hi": 0.11},
        {"ticker": "D", "lo": 0.19, "hi": 0.21},
        {"ticker": "E", "lo": 0.0, "hi": 0.01},
        {"ticker": "F", "lo": 0.29, "hi": 0.31},
        {"ticker": "G", "lo": 0.29, "hi": 0.31},
        {"ticker": "H", "lo": 0.0, "hi": 0.01},
        {"ticker": "I", "lo": 0.0, "hi": 0.01},
        {"ticker": "J", "lo": 0.0, "hi": 0.01},
        {"ticker": "K", "lo": 0.0, "hi": 0.01},
        {"ticker": "L", "lo": 0.09, "hi": 0.11},
        {"ticker": "M", "lo": 0.0, "hi": 0.01},
        {"ticker": "N", "lo": 0.0, "hi": 0.01},
        {"ticker": "O", "lo": 0.0, "hi": 0.01},
    ]


@pytest.mark.parametrize("sampler", ALL_SAMPLERS, ids=_SAMPLER_IDS)
def test_sampler_rows_sum_to_one(sampler):
    rng = np.random.default_rng(0)
    w = sampler(_space(5), 200, rng)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("sampler", ALL_SAMPLERS, ids=_SAMPLER_IDS)
def test_sampler_respects_bounds(sampler):
    space = [
        {"ticker": "A", "lo": 0.1, "hi": 0.5},
        {"ticker": "B", "lo": 0.0, "hi": 0.3},
        {"ticker": "C", "lo": 0.2, "hi": 0.6},
    ]
    rng = np.random.default_rng(1)
    w = sampler(space, 300, rng)
    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])
    assert np.all(w >= lo - 1e-8)
    assert np.all(w <= hi + 1e-8)


@pytest.mark.parametrize("sampler", ALL_SAMPLERS, ids=_SAMPLER_IDS)
def test_sampler_is_deterministic_given_fixed_seed(sampler):
    space = _real_search_space()
    w1 = sampler(space, 100, np.random.default_rng(7))
    w2 = sampler(space, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(w1, w2)


@pytest.mark.parametrize("sampler", ALL_SAMPLERS, ids=_SAMPLER_IDS)
def test_sampler_infeasible_bounds_returns_empty_gracefully(sampler):
    # sum(lo) > 1.0: impossible to satisfy. Must not hang or raise.
    space = [{"ticker": "A", "lo": 0.7, "hi": 0.9}, {"ticker": "B", "lo": 0.7, "hi": 0.9}]
    w = sampler(space, 100, np.random.default_rng(4))
    assert len(w) == 0
    assert w.shape[1] == 2


@pytest.mark.parametrize(
    "sampler", [sample_cdhr_portfolios, sample_sparse_portfolios,
                sample_vertex_portfolios, sample_mixed_portfolios],
    ids=["cdhr", "sparse", "vertex", "mixed"],
)
def test_sampler_fills_narrow_per_asset_bounds(sampler):
    """M4 regression: on the tight-band space that killed Dirichlet's
    acceptance rate, every non-rejection sampler must still fill the
    request exactly (they have no rejection step at all)."""
    w = sampler(_tight_band_space(), 500, np.random.default_rng(5))
    assert len(w) == 500
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)


def test_cdhr_is_uniform_on_the_unconstrained_simplex():
    """With lo=0/hi=1 on k=3 assets the box never binds, so CDHR should
    sample uniformly over the full 2-simplex. Each marginal weight is then
    Beta(1, k-1) = Beta(1, 2): mean=1/3, var=1/18. Checked via moments
    (no scipy dependency in this project) against their CLT standard
    error at n=50,000 — this is the test that would catch CDHR silently
    drifting off-uniform (e.g. a sign error in the move interval)."""
    space = [{"ticker": f"T{i}", "lo": 0.0, "hi": 1.0} for i in range(3)]
    w = sample_cdhr_portfolios(space, 50_000, np.random.default_rng(2))
    w0 = w[:, 0]
    expected_mean, expected_var = 1 / 3, 1 / 18
    se = np.sqrt(expected_var / len(w0))
    assert abs(w0.mean() - expected_mean) < 6 * se
    assert abs(w0.var() - expected_var) < 0.15 * expected_var


def test_mixed_sampler_covers_more_of_the_polytope_than_random():
    """The whole point of this work: sample_random_portfolios concentrates
    mass near the box centroid (worse as k grows — see its docstring), so
    on a real 15-asset, hi=0.30 space it essentially never produces a
    concentrated portfolio. Measured: p99(max_weight) = 0.181 for
    sample_random_portfolios on this space; sample_mixed_portfolios must
    clear a substantially higher floor."""
    space = _real_search_space()
    w_random = sample_random_portfolios(space, 20_000, np.random.default_rng(0))
    w_mixed = sample_mixed_portfolios(space, 20_000, np.random.default_rng(0))
    p99_random = np.percentile(w_random.max(axis=1), 99)
    p99_mixed = np.percentile(w_mixed.max(axis=1), 99)
    assert p99_random < 0.20, f"baseline assumption changed: p99={p99_random:.3f}"
    assert p99_mixed > 0.25, f"mixed sampler coverage regressed: p99={p99_mixed:.3f}"


def test_sparse_never_drops_below_minimum_active_count():
    """On a uniform hi=0.30, 15-asset space, no support smaller than
    ceil(1/0.30)=4 can reach sum(w)=1 within bounds — sampling below that
    floor would either violate hi caps or fail to sum to 1."""
    space = _real_search_space()
    w = sample_sparse_portfolios(space, 3000, np.random.default_rng(3))
    active_counts = (w > 1e-9).sum(axis=1)
    assert active_counts.min() >= 4


def test_sparse_never_switches_off_a_mandatory_asset():
    """Assets with lo > 0 can never legally be 0 — sample_sparse_portfolios
    must always keep them active, unlike the lo=0 (eligible-to-switch-off)
    assets."""
    space = _tight_band_space()
    w = sample_sparse_portfolios(space, 2000, np.random.default_rng(6))
    mandatory = [i for i, s in enumerate(space) if s["lo"] > 0]
    for i in mandatory:
        assert np.all(w[:, i] > 0), f"mandatory asset {space[i]['ticker']} was switched off"


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_mixed_sampler_handles_small_n_without_rounding_crash(n):
    """The 40/40/20 split is computed with round(); small n must still add
    up to exactly n and never crash on a zero-sized sub-sample."""
    w = sample_mixed_portfolios(_real_search_space(), n, np.random.default_rng(0))
    assert w.shape == (n, 15)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)


def _heterogeneous_cap_space() -> list[dict]:
    """Caps that differ wildly between assets — the shape that exposed the
    sparse sampler's per-row feasibility bug. Uniform caps can't: there,
    every support of a given size has the same cap sum."""
    return [
        {"ticker": "E", "lo": 0.0, "hi": 0.60},
        {"ticker": "B", "lo": 0.0, "hi": 0.50},
        {"ticker": "G", "lo": 0.0, "hi": 0.15},
        {"ticker": "R", "lo": 0.0, "hi": 0.10},
        {"ticker": "C", "lo": 0.0, "hi": 0.10},
        {"ticker": "X", "lo": 0.0, "hi": 0.05},
    ]


def _mixed_mandatory_space() -> list[dict]:
    """Heterogeneous caps AND some lo>0 (non-switchable) assets."""
    return [
        {"ticker": "M1", "lo": 0.10, "hi": 0.30},
        {"ticker": "M2", "lo": 0.05, "hi": 0.20},
        {"ticker": "F1", "lo": 0.00, "hi": 0.40},
        {"ticker": "F2", "lo": 0.00, "hi": 0.10},
        {"ticker": "F3", "lo": 0.00, "hi": 0.10},
    ]


@pytest.mark.parametrize(
    "space_fn",
    [_heterogeneous_cap_space, _mixed_mandatory_space],
    ids=["heterogeneous_caps", "heterogeneous_caps_with_mandatory"],
)
@pytest.mark.parametrize(
    "sampler", [sample_sparse_portfolios, sample_mixed_portfolios],
    ids=["sparse", "mixed"],
)
def test_sparse_respects_bounds_with_heterogeneous_caps(sampler, space_fn):
    """Regression: sample_sparse_portfolios picked each row's active
    subset at random but sized it from the LARGEST caps (sorted
    descending). A row whose support came from the small-cap end then had
    sum(hi[S]) < 1 — it literally cannot reach sum(w) == 1 within bounds —
    and _cdhr_core's starting point overshot the caps, with the walk
    (which only preserves the sum) never repairing it. Measured before the
    fix: 948/2000 rows out of bounds on the heterogeneous space, 0 on a
    uniform-cap space, which is why the original tests missed it."""
    space = space_fn()
    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])
    w = sampler(space, 2000, np.random.default_rng(0))

    assert len(w) == 2000
    assert np.all(w <= hi + 1e-8), (
        f"{int((w > hi + 1e-8).any(axis=1).sum())} rows exceed their upper bounds"
    )
    assert np.all(w >= lo - 1e-8)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)


def test_sparse_support_always_has_enough_cap_headroom():
    """Every row's ACTIVE support must be able to reach sum(w)==1, i.e.
    sum(hi) over the active assets >= 1. This is the invariant the fix
    enforces; asserting it directly (rather than only its symptom) means
    the test still catches a regression that happens to stay in bounds by
    luck on a particular seed."""
    space = _heterogeneous_cap_space()
    hi = np.array([s["hi"] for s in space])
    w = sample_sparse_portfolios(space, 2000, np.random.default_rng(1))
    active = w > 1e-9
    cap_headroom = (active * hi).sum(axis=1)
    assert np.all(cap_headroom >= 1.0 - 1e-9)


# ═══════════════════════════════════════════════════════════════════════
# Evolutionary refinement operators
# ═══════════════════════════════════════════════════════════════════════

def test_crossover_children_are_feasible():
    """A convex combination of two feasible points is feasible for ANY
    box+simplex region — the whole reason crossover needs no repair
    step. Checked on a heterogeneous-cap space specifically, since that's
    where an incorrect crossover (e.g. naive per-asset averaging that
    somehow drifted) would be most likely to show a violation."""
    space = _heterogeneous_cap_space()
    rng = np.random.default_rng(0)
    parents_a = sample_cdhr_portfolios(space, 500, rng)
    parents_b = sample_cdhr_portfolios(space, 500, rng)
    children = crossover_portfolios(parents_a, parents_b, rng)

    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])
    assert children.shape == parents_a.shape
    np.testing.assert_allclose(children.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(children >= lo - 1e-8)
    assert np.all(children <= hi + 1e-8)


def test_crossover_of_a_point_with_itself_is_that_point():
    space = _real_search_space()
    rng = np.random.default_rng(1)
    p = sample_cdhr_portfolios(space, 100, rng)
    children = crossover_portfolios(p, p, rng)
    np.testing.assert_allclose(children, p, atol=1e-10)


def test_crossover_rejects_mismatched_shapes():
    a = np.ones((5, 3)) / 3
    b = np.ones((4, 3)) / 3
    with pytest.raises(ValueError):
        crossover_portfolios(a, b, np.random.default_rng(0))


def test_local_perturb_stays_feasible():
    space = _heterogeneous_cap_space()
    rng = np.random.default_rng(2)
    seeds = sample_cdhr_portfolios(space, 500, rng)
    perturbed = local_perturb_portfolios(seeds, space, rng, n_steps=10)

    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])
    np.testing.assert_allclose(perturbed.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(perturbed >= lo - 1e-8)
    assert np.all(perturbed <= hi + 1e-8)


def test_local_perturb_fewer_steps_stays_closer_to_the_seed():
    """This is the whole point of exposing n_steps: it's the "how far to
    push" knob for an evolutionary refinement pass. A handful of steps
    should land, on average, closer to the seed than a long walk that's
    had time to wander toward the polytope's uniform distribution."""
    space = _real_search_space()
    seeds = sample_cdhr_portfolios(space, 300, np.random.default_rng(3))

    near = local_perturb_portfolios(seeds, space, np.random.default_rng(4), n_steps=2)
    far = local_perturb_portfolios(seeds, space, np.random.default_rng(4), n_steps=300)

    dist_near = np.linalg.norm(near - seeds, axis=1).mean()
    dist_far = np.linalg.norm(far - seeds, axis=1).mean()
    assert dist_near < dist_far


def test_local_perturb_zero_steps_returns_the_seed_unchanged():
    space = _real_search_space()
    seeds = sample_cdhr_portfolios(space, 50, np.random.default_rng(5))
    out = local_perturb_portfolios(seeds, space, np.random.default_rng(6), n_steps=0)
    np.testing.assert_allclose(out, seeds, atol=1e-12)
