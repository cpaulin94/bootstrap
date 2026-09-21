"""
engine.search — Portfolio-space samplers and search-space loading.

Four samplers, all with the same ``(space, n, rng) -> (n, k)`` contract
(weights sum to 1, bounds respected, empty array on an infeasible space):

  - ``sample_random_portfolios`` — box-uniform-then-normalise. Fast and
    simple, but concentrates mass near the box's centroid, worse as the
    number of assets grows (see its docstring for the mechanism).
  - ``sample_cdhr_portfolios`` — exact uniform sampling via
    coordinate-direction hit-and-run, no rejection at any box width.
  - ``sample_sparse_portfolios`` — random-cardinality active subsets, for
    the "concentrated in a few assets" corners the other samplers miss.
  - ``sample_vertex_portfolios`` — the polytope's extreme points, along
    random linear objectives.
  - ``sample_mixed_portfolios`` — a 40/40/20 blend of CDHR/sparse/vertex;
    the recommended default (see engine.config.SEARCH_METHOD).
"""

from __future__ import annotations

import csv
import logging
import time

import numpy as np

log = logging.getLogger("bootstrap.search")


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
    """Sample *n* weight vectors from the constrained simplex.

    Draws each weight independently and uniformly from its own ``[lo, hi]``
    box, then normalises the row to sum to 1, rejecting rows that fall
    outside the bounds after normalising.

    This is NOT uniform on the constrained simplex: normalising a
    box-uniform vector concentrates mass near the box's centroid (an
    average-of-uniforms effect), so concentrated portfolios — one asset
    near its ``hi`` cap, the rest near 0 — are under-sampled relative to a
    true uniform distribution over the constrained region.

    A ``Dirichlet(1, ..., 1)`` (uniform on the FULL simplex) + rejection
    was tried as a fix and reverted: it's only uniform over the WHOLE
    simplex, and rejecting down to a narrow per-asset box — e.g. every
    bound within a 1-2 percentage-point band, a realistic search-space
    shape for this tool — has a near-zero acceptance rate (measured: ~0%
    after 2,000,000 draws on a real 15-asset space with such bounds,
    where this box-uniform method accepts about 1%). Reliably filling
    typical narrow search spaces matters more than exact uniformity here.

    Returns (n, k) array of weights that sum to 1.
    """
    k = len(space)
    lo = np.array([s["lo"] for s in space])
    hi = np.array([s["hi"] for s in space])

    log.info("[RANDOM] Sampling %d portfolios from %d-asset simplex", n, k)
    log.info("[RANDOM] Bounds: lo=%s  hi=%s", lo, hi)
    log.info("[RANDOM] Sum of lo=%.4f  sum of hi=%.4f  (need room for sum=1.0)",
             lo.sum(), hi.sum())
    if lo.sum() > 1.0 + 1e-9:
        log.error("[RANDOM] ⚠️ Sum of lower bounds (%.4f) > 1.0 — "
                  "impossible to satisfy simplex constraint!", lo.sum())
    if hi.sum() < 1.0 - 1e-9:
        log.error("[RANDOM] ⚠️ Sum of upper bounds (%.4f) < 1.0 — "
                  "impossible to satisfy simplex constraint!", hi.sum())

    t0 = time.perf_counter()
    results = np.empty((n, k), dtype=np.float64)
    filled = 0
    batch = max(n * 2, 10_000)
    total_attempts = 0
    rejection_rounds = 0

    while filled < n:
        raw = rng.uniform(lo, hi, size=(batch, k))
        row_sums = raw.sum(axis=1, keepdims=True)
        normed = raw / row_sums

        valid = np.all((normed >= lo) & (normed <= hi), axis=1)
        good = normed[valid]
        total_attempts += batch
        rejection_rounds += 1

        take = min(len(good), n - filled)
        if take > 0:
            results[filled : filled + take] = good[:take]
            filled += take

        if rejection_rounds % 10 == 0:
            acceptance = filled / total_attempts * 100
            log.info("[RANDOM] Progress: %d/%d filled after %d attempts "
                     "(%.1f%% acceptance, round %d)",
                     filled, n, total_attempts, acceptance, rejection_rounds)
        if rejection_rounds > 200 and filled == 0:
            log.error("[RANDOM] 0 valid portfolios after %d attempts — "
                      "search space is likely infeasible!", total_attempts)
            break

    elapsed = time.perf_counter() - t0
    acceptance = filled / max(total_attempts, 1) * 100
    log.info("[RANDOM] Done: %d portfolios in %.3fs  "
             "(%d attempts, %.1f%% acceptance rate, %d rounds)",
             filled, elapsed, total_attempts, acceptance, rejection_rounds)

    # On the infeasible-bounds bail-out above, `filled` can be < n — the
    # tail of `results` past that point was never written (uninitialised
    # memory from np.empty). Truncating is a no-op on the normal path
    # (filled == n there) and turns the failure case into a clean empty/
    # short array instead of returning garbage rows.
    return results[:filled]


def _cdhr_core(
    lo: np.ndarray,
    hi: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_steps: int | None = None,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Coordinate-direction hit-and-run walk on ``{w: lo<=w<=hi, sum(w)=1}``.

    *lo*/*hi* may be ``(k,)`` (one shared box, tiled across all *n* rows —
    the plain-CDHR case) or ``(n, k)`` (a DIFFERENT box per row — what
    ``sample_sparse_portfolios`` needs, one row per randomly-chosen active
    subset). Both are handled by the same code via broadcasting.

    Any column with ``lo[c] == hi[c]`` (a "pinned" coordinate — a fixed
    asset, or an asset switched off for that row via ``lo=hi=0``) is
    never moved away from its fixed value: whenever the walk picks a pair
    (i, j) where either index is pinned for that row, the feasible move
    interval collapses to exactly ``[0, 0]`` (the pinned side's own bound
    already forces ``t=0``), so passing a mix of free and pinned columns
    in one call is safe without any special-casing.

    Moves along ``e_i - e_j`` directions preserve ``sum(w)`` exactly and
    span the simplex's affine hull, so the chain is irreducible on the
    polytope and its stationary distribution is uniform — with, unlike
    ``sample_random_portfolios``, a 100% acceptance rate regardless of how
    narrow the box is (see AUDIT.md M4: this is what closes that finding).

    *start*, if given, is an ``(n, k)`` array of feasible starting points
    used INSTEAD of the box's own centroid — every step of the walk stays
    exactly feasible regardless of where it starts (it's still a sequence
    of ``e_i - e_j`` moves within the same bounds), so this is a
    correctness-free way to do a SHORT local walk anchored near existing
    points (see ``local_perturb_portfolios``) rather than a fresh
    from-scratch sample. Few steps from a real start stay close to it; many
    steps converge to the same box-centroid-independent uniform
    distribution as the default start.
    """
    lo = np.atleast_2d(np.asarray(lo, dtype=np.float64))
    hi = np.atleast_2d(np.asarray(hi, dtype=np.float64))
    k = lo.shape[-1]
    if n_steps is None:
        n_steps = max(200, 20 * k)

    if start is not None:
        W = np.array(start, dtype=np.float64, copy=True)
        if W.shape != (n, k):
            raise ValueError(f"start must have shape ({n}, {k}), got {W.shape}")
    else:
        r = 1.0 - lo.sum(axis=1, keepdims=True)
        span = hi - lo
        span_sum = span.sum(axis=1, keepdims=True)
        span_sum_safe = np.where(span_sum < 1e-12, 1.0, span_sum)
        w0 = lo + r * span / span_sum_safe  # feasible interior point per row
        W = np.broadcast_to(w0, (n, k)).copy()

    if k < 2:
        return W

    lo_b = np.broadcast_to(lo, (n, k))
    hi_b = np.broadcast_to(hi, (n, k))
    rows = np.arange(n)
    for _ in range(n_steps):
        i = rng.integers(0, k, n)
        j = (i + 1 + rng.integers(0, k - 1, n)) % k  # j != i, uniform
        wi = W[rows, i]
        wj = W[rows, j]
        lo_i = lo_b[rows, i]
        hi_i = hi_b[rows, i]
        lo_j = lo_b[rows, j]
        hi_j = hi_b[rows, j]
        t_lo = np.maximum(lo_i - wi, wj - hi_j)
        t_hi = np.minimum(hi_i - wi, wj - lo_j)
        # t_hi < t_lo means this row's box is infeasible (its caps can't
        # reach sum == 1). Callers are responsible for not producing such
        # rows — see sample_sparse_portfolios' per-row feasibility floor —
        # but guard anyway: sampling the reversed interval would move
        # AWAY from the feasible set and silently emit out-of-bounds
        # weights, which is exactly how that bug stayed invisible. Freezing
        # the row instead makes a caller-side mistake show up as a
        # detectably stuck row rather than as plausible-looking bad data.
        span_t = np.maximum(t_hi - t_lo, 0.0)
        t = np.where(t_hi >= t_lo, t_lo + span_t * rng.random(n), 0.0)
        W[rows, i] = wi + t
        W[rows, j] = wj - t
    return W


def _check_feasible(lo: np.ndarray, hi: np.ndarray, tag: str) -> bool:
    if lo.sum() > 1.0 + 1e-9:
        log.error("[%s] Sum of lower bounds (%.4f) > 1.0 — infeasible!", tag, lo.sum())
        return False
    if hi.sum() < 1.0 - 1e-9:
        log.error("[%s] Sum of upper bounds (%.4f) < 1.0 — infeasible!", tag, hi.sum())
        return False
    return True


def sample_cdhr_portfolios(
    space: list[dict],
    n: int,
    rng: np.random.Generator,
    n_steps: int | None = None,
) -> np.ndarray:
    """Uniform sampling of the constrained simplex via coordinate-direction
    hit-and-run (CDHR) — see :func:`_cdhr_core`.

    Unlike :func:`sample_random_portfolios`, this is EXACT uniform sampling
    over ``{w: lo<=w<=hi, sum(w)=1}`` with a 100% acceptance rate, no
    rejection, regardless of box width. ``sample_random_portfolios``
    concentrates mass near the box's centroid — worse the more assets
    there are (it's an average-of-uniforms effect, so variance shrinks as
    1/k) — and that bias is what this fixes: measured on a real 15-asset,
    ``hi=0.30`` search space over 20,000 draws, the 99th percentile of the
    largest weight in a portfolio is 0.181 with
    ``sample_random_portfolios`` vs 0.295 here.

    Returns (n, k) array of weights that sum to 1, or an empty array if
    the bounds are infeasible (same contract as sample_random_portfolios).
    """
    k = len(space)
    lo = np.array([s["lo"] for s in space], dtype=np.float64)
    hi = np.array([s["hi"] for s in space], dtype=np.float64)

    log.info("[CDHR] Sampling %d portfolios from %d-asset simplex (uniform, no rejection)", n, k)
    log.info("[CDHR] Bounds: lo=%s  hi=%s", lo, hi)
    if not _check_feasible(lo, hi, "CDHR"):
        return np.empty((0, k))

    t0 = time.perf_counter()
    W = _cdhr_core(lo, hi, n, rng, n_steps=n_steps)
    log.info("[CDHR] Done: %d portfolios in %.3fs", n, time.perf_counter() - t0)
    return W


def sample_sparse_portfolios(
    space: list[dict],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample portfolios with a randomly-sized, randomly-chosen SUBSET of
    assets active — the rest pinned at exactly zero — with CDHR-uniform
    weights over the active subset.

    Every asset with ``lo > 0`` is mandatory (it can never legally sit at
    0) and is active in every row. Assets with ``lo == 0`` are eligible to
    be switched off; each row independently draws how many of them stay
    active, uniformly between the minimum needed for feasibility and all
    of them.

    Neither ``sample_random_portfolios`` nor ``sample_cdhr_portfolios``
    ever tests "concentrated in a handful of assets" corners of the
    search space — both spread weight across every asset by construction
    — which is exactly where interesting risk/return tradeoffs tend to
    live. Measured on a real 15-asset space at equal budget (1500
    candidates, common sim_seed), this sampler found a portfolio with
    1st-percentile CAGR of +1.6%; neither of the other two found ANY
    portfolio with positive p1 CAGR.

    Feasibility floor: a support is only usable if the sum of its ``hi``
    caps is >= 1 (enough headroom to reach 100% allocation). The minimum
    support size is every mandatory asset plus the fewest free assets
    (highest ``hi`` first) needed to close that gap — for a uniform
    ``hi=0.30`` space that floor is ``ceil(1/0.30) = 4`` assets. Sampling
    below it would silently produce rows that can't reach ``sum=1``
    within bounds.

    Returns (n, k) array of weights that sum to 1, or an empty array if
    the bounds are infeasible.
    """
    k = len(space)
    lo = np.array([s["lo"] for s in space], dtype=np.float64)
    hi = np.array([s["hi"] for s in space], dtype=np.float64)

    log.info("[SPARSE] Sampling %d portfolios, random active subset of %d assets", n, k)
    if not _check_feasible(lo, hi, "SPARSE"):
        return np.empty((0, k))

    mandatory_idx = np.flatnonzero(lo > 1e-12)
    free_idx = np.flatnonzero(lo <= 1e-12)
    n_free = len(free_idx)

    mandatory_hi_sum = hi[mandatory_idx].sum()
    remaining_needed = 1.0 - mandatory_hi_sum
    if remaining_needed <= 1e-12:
        m_free_min = 0
    else:
        free_hi_sorted = np.sort(hi[free_idx])[::-1]
        cum = np.cumsum(free_hi_sorted)
        m_free_min = int(np.searchsorted(cum, remaining_needed - 1e-9)) + 1
        if m_free_min > n_free:
            log.error(
                "[SPARSE] No feasible support size — even all %d free assets "
                "can't reach sum(hi)>=1 alongside the %d mandatory assets.",
                n_free, len(mandatory_idx),
            )
            return np.empty((0, k))

    log.info("[SPARSE] %d mandatory assets, %d free assets, min free active=%d",
             len(mandatory_idx), n_free, m_free_min)

    t0 = time.perf_counter()
    row_lo = np.zeros((n, k), dtype=np.float64)
    row_hi = np.zeros((n, k), dtype=np.float64)
    row_lo[:, mandatory_idx] = lo[mandatory_idx]
    row_hi[:, mandatory_idx] = hi[mandatory_idx]

    if n_free > 0:
        m_free_per_row = rng.integers(m_free_min, n_free + 1, size=n)
        # Vectorised "choose m distinct free assets per row": rank each
        # free asset by a per-row random priority, then keep the assets
        # whose rank falls below that row's chosen count. Avoids a Python
        # loop over rows calling rng.choice n times (measured: ~6s for
        # 20,000 samples the naive way vs <1s vectorised).
        priorities = rng.random((n, n_free))
        order = np.argsort(priorities, axis=1)
        ranks = np.empty_like(order)
        row_idx = np.arange(n)[:, None]
        ranks[row_idx, order] = np.arange(n_free)[None, :]

        # PER-ROW feasibility. `m_free_min` above is the smallest support
        # size that CAN work — it's computed from the largest caps, sorted
        # descending. But each row picks its support at RANDOM, so a
        # row-sized support drawn from the small-cap end can have
        # sum(hi[S]) < 1 and be unable to reach sum(w) == 1 within bounds
        # at all. Left unguarded, `_cdhr_core`'s starting point
        # `lo + r*span/span_sum` then overshoots the caps and the walk —
        # which only ever preserves the sum, never repairs a violation —
        # emits out-of-bounds rows for the rest of the chain. Measured on
        # a 6-asset space with caps [.6,.5,.15,.10,.10,.05]: 1482/2000
        # rows violated their upper bounds; uniform caps (the shape the
        # tests happened to cover) hid it completely, since there every
        # support of a given size has the same cap sum.
        #
        # Fix: walk the row's own priority order and extend the support to
        # the shortest prefix that reaches sum(hi[S]) >= 1, then take
        # whichever is larger — the drawn cardinality or that floor. The
        # full free set is always feasible (guaranteed by _check_feasible
        # above), so `argmax` always finds a valid prefix.
        hi_ordered = hi[free_idx][order]
        cum_hi = np.cumsum(hi_ordered, axis=1) + mandatory_hi_sum
        min_prefix = np.argmax(cum_hi >= 1.0 - 1e-9, axis=1) + 1
        m_free_per_row = np.maximum(m_free_per_row, min_prefix)

        active_free = ranks < m_free_per_row[:, None]
        row_hi[:, free_idx] = np.where(active_free, hi[free_idx][None, :], 0.0)

    W = _cdhr_core(row_lo, row_hi, n, rng)
    log.info("[SPARSE] Done: %d portfolios in %.3fs", n, time.perf_counter() - t0)
    return W


def sample_vertex_portfolios(
    space: list[dict],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample VERTICES of the constrained simplex.

    For each row, draws a random direction ``c ~ N(0, I)`` and solves
    ``max c.w`` s.t. ``lo<=w<=hi, sum(w)=1`` via the standard greedy
    water-filling algorithm for a simplex-constrained LP: give as much
    extra weight as possible (up to each asset's ``hi`` cap) to the
    highest-``c`` assets first, in order, until the budget above the
    ``lo`` floor is exhausted. The exact solution is always a vertex of
    the polytope — where a linear Pareto frontier typically sits, and a
    region ``sample_random_portfolios``/``sample_cdhr_portfolios`` only
    reach in the limit of infinitely many draws.

    Returns (n, k) array of weights that sum to 1, or an empty array if
    the bounds are infeasible.
    """
    k = len(space)
    lo = np.array([s["lo"] for s in space], dtype=np.float64)
    hi = np.array([s["hi"] for s in space], dtype=np.float64)

    log.info("[VERTEX] Sampling %d vertex portfolios from %d-asset simplex", n, k)
    if not _check_feasible(lo, hi, "VERTEX"):
        return np.empty((0, k))

    t0 = time.perf_counter()
    c = rng.standard_normal((n, k))
    order = np.argsort(-c, axis=1)  # highest-c asset first, per row

    lo_b = np.broadcast_to(lo, (n, k))
    hi_b = np.broadcast_to(hi, (n, k))
    lo_ordered = np.take_along_axis(lo_b, order, axis=1)
    hi_ordered = np.take_along_axis(hi_b, order, axis=1)

    room = 1.0 - lo.sum()  # slack above the lower-bound floor, shared by all rows
    extra_cap = hi_ordered - lo_ordered
    prev_cum = np.cumsum(extra_cap, axis=1) - extra_cap  # exclusive cumsum
    add = np.clip(room - prev_cum, 0.0, extra_cap)
    w_ordered = lo_ordered + add

    W = np.empty((n, k), dtype=np.float64)
    np.put_along_axis(W, order, w_ordered, axis=1)
    log.info("[VERTEX] Done: %d portfolios in %.3fs", n, time.perf_counter() - t0)
    return W


MIXED_WEIGHTS = (0.4, 0.4, 0.2)  # cdhr, sparse, vertex


def sample_mixed_portfolios(
    space: list[dict],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Default recommended sampler: a blend of :func:`sample_cdhr_portfolios`
    (40%, uniform coverage of the whole polytope),
    :func:`sample_sparse_portfolios` (40%, concentrated/few-asset corners)
    and :func:`sample_vertex_portfolios` (20%, the polytope's extreme
    points). See each function's docstring for what it fixes and why
    ``sample_random_portfolios`` under-covers all three regions as
    dimensionality grows (AUDIT.md M4).

    Rows from the three sub-samples are shuffled together so a caller
    watching progress (e.g. the GUI's batch callback) sees one cloud
    growing, not three back-to-back regimes.

    Returns (n, k) array of weights that sum to 1, or an empty array if
    the bounds are infeasible.
    """
    k = len(space)
    n_cdhr = round(n * MIXED_WEIGHTS[0])
    n_sparse = round(n * MIXED_WEIGHTS[1])
    n_vertex = n - n_cdhr - n_sparse  # remainder absorbs rounding, total stays == n

    log.info("[MIXED] Sampling %d portfolios: %d CDHR + %d sparse + %d vertex",
             n, n_cdhr, n_sparse, n_vertex)

    parts = []
    if n_cdhr > 0:
        parts.append(sample_cdhr_portfolios(space, n_cdhr, rng))
    if n_sparse > 0:
        parts.append(sample_sparse_portfolios(space, n_sparse, rng))
    if n_vertex > 0:
        parts.append(sample_vertex_portfolios(space, n_vertex, rng))

    if not parts or any(len(p) == 0 for p in parts):
        # Infeasible space — the failing sub-sampler already logged why.
        return np.empty((0, k))

    W = np.vstack(parts)
    rng.shuffle(W, axis=0)
    return W


# ═══════════════════════════════════════════════════════════════════════════
# Evolutionary refinement operators — used by
# engine.runner.run_evolutionary_streaming to push a Pareto frontier
# outward across generations instead of only ever sampling it once.
# ═══════════════════════════════════════════════════════════════════════════

def crossover_portfolios(
    parents_a: np.ndarray,
    parents_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Blend-crossover: each child is a random convex combination of the
    matching rows of *parents_a* and *parents_b*.

    ``{w: lo<=w<=hi, sum(w)=1}`` is the intersection of a box and a
    hyperplane — both convex — so ANY convex combination of two feasible
    portfolios is itself feasible. That makes this crossover exact with
    zero repair step: no clipping, no renormalising, no risk of a child
    landing outside bounds regardless of how different its two parents
    are, unlike a naive per-asset average-then-clip scheme.

    Hybridising two Pareto-frontier portfolios this way explores the
    segment BETWEEN them — a real portfolio that trades off their
    strengths — rather than only ever perturbing one point at a time.

    Returns an ``(n, k)`` array; *parents_a* and *parents_b* must already
    have matching shape (the caller picks which frontier rows to pair).
    """
    if parents_a.shape != parents_b.shape:
        raise ValueError(
            f"parents_a {parents_a.shape} and parents_b {parents_b.shape} "
            f"must have the same shape"
        )
    n = parents_a.shape[0]
    alpha = rng.random((n, 1))
    return alpha * parents_a + (1.0 - alpha) * parents_b


def local_perturb_portfolios(
    seeds: np.ndarray,
    space: list[dict],
    rng: np.random.Generator,
    n_steps: int = 20,
) -> np.ndarray:
    """A short CDHR walk starting from each row of *seeds*, instead of
    from the box's centroid — a feasibility-preserving local perturbation.

    This is the "push a little further" step of an evolutionary
    refinement pass: few steps keep the result close to its seed, more
    steps let it wander further, and enough steps converge to the same
    box-centroid-independent uniform sample :func:`sample_cdhr_portfolios`
    produces from scratch. Every intermediate state of the walk is
    EXACTLY feasible by construction (see :func:`_cdhr_core`), so there is
    nothing to repair even after a single step — unlike, say, adding
    Gaussian noise to a weight vector and clipping it back into bounds,
    which needs a separate (and non-trivial, for a box+simplex
    intersection) projection step to stay feasible.

    *seeds* must already be feasible for *space* — typically Pareto
    frontier points from a previous generation.
    """
    k = len(space)
    lo = np.array([s["lo"] for s in space], dtype=np.float64)
    hi = np.array([s["hi"] for s in space], dtype=np.float64)
    n = seeds.shape[0]
    return _cdhr_core(lo, hi, n, rng, n_steps=max(0, n_steps), start=seeds)


def sample_grid_portfolios(
    space: list[dict],
    step: float,
) -> np.ndarray:
    """Enumerate all weight combos on the simplex with given step size."""
    k = len(space)
    lo = [s["lo"] for s in space]
    hi = [s["hi"] for s in space]
    log.info("[GRID] Enumerating grid portfolios: %d assets, step=%.4f", k, step)
    log.info("[GRID] Bounds: lo=%s  hi=%s", lo, hi)
    t0 = time.perf_counter()

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
    elapsed = time.perf_counter() - t0
    log.info("[GRID] Done: %d portfolios enumerated in %.3fs", len(results), elapsed)
    return np.array(results, dtype=np.float64) if results else np.empty((0, k))
