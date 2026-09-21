# Mathematical & methodological audit — `bootstrap`

Date: 2026-08-21. Scope: `engine/`, `bootstrap_gui/`, `gui.py`, `scripts/preprocess.py`, `data/`.
Baseline: all 133 tests pass (`uv run pytest` → `133 passed`). Every finding below was
reproduced numerically on this repo's real data; the reproduction command is included so the
implementer can verify before and after.

Findings are ranked by how much they distort a conclusion a user would actually draw.
Nothing here is speculative "code smell" — items I checked and found **correct** are listed
in §7 so nobody re-does that work or "fixes" them by accident.

---

## Implementation status (2026-08-21, same-day follow-up)

All items below were implemented, each with new regression tests reproducing the original
bug and asserting the fix (154/154 tests passing, up from the 133 baseline above).

**2026-08-25 independent re-audit (Opus):** three further defects found by verifying the
numerical core against from-scratch reference implementations and by stress-testing the new
samplers on search-space shapes the tests didn't cover. See rows M10–M12. Verified CORRECT and
left alone: `_precompute_block_gross` (exact match to a naive rolling-product loop at every
block size), annualised-return exponent convention, all three drawdown metrics (exact match to
naive per-simulation loops), block-size/horizon truncation accounting, and the data layer
(no gaps, duplicates, NaNs, or returns <= -100%; every ticker intersection contiguous, which
block bootstrapping depends on; TER correctly applied). The iid bootstrap reproduces the
historical geometric CAGR to 4 decimal places (0.0783 vs 0.0783); block bootstrap shows the
standard, expected moving-block edge-effect bias (-0.07pp at block=6, -0.15pp at block=12).

**2026-08-25 follow-up:** M4 (previously reverted) was fixed; M8 (newly found) was fixed; M9
(newly found) was tried and reverted the same day, same pattern as M4 — see their rows below
and `PLAN_SEARCH.md` for the full design. 229/229 tests passing, up from the 154 above.

**2026-08-26:** M10–M12 fixed (see the re-audit note above); the evolutionary search capability
added (§5b). 249/249 tests passing.

| Item | Status |
|---|---|
| C1 (row-tail alignment) | **Fixed** — `engine/data.py`: `align_on_common_dates` intersects calendar months instead of truncating by row position. |
| C2 (one-month return-date shift) | **Fixed** — `scripts/preprocess.py` regenerated; `data/standard/*.csv` rewritten with correct labels, including a new BTOP50 VAMI parser. |
| C3 (independent RNG per candidate) | **Fixed** — `engine/runner.py`: every candidate portfolio in a search now shares `sim_seed` (common random numbers). |
| H1 (block_size truncation) | **Fixed, minimum-patch variant** — `compute_metrics` annualises against months actually simulated; `simulate()`/`simulate_independent()` raise instead of silently returning all-zero metrics when `block_size > horizon_months`. The "preferred" full rewrite (routing `simulate()` through `sample_monthly_returns` for monthly resolution regardless of block_size) was **not** done — it would remove the block-bootstrap's speed advantage for the 10k-portfolio search, a real regression the audit itself flagged as a reason to prefer the minimum patch. M3 (block-endpoint resolution understating drawdowns) is therefore still open, by the same reasoning. |
| H2 (shortfall_months_median) | **Fixed** — compares each simulation to its own requested target. |
| H3 (DRAWDOWN_CURTAILED balance vs market) | **Fixed** — a cash-flow-neutral `market_unit` index now drives the trigger. |
| M1 (missing metrics → silent 0) | **Fixed** — `gui.py`: chart drops points missing the axis metric instead of defaulting to 0; Pareto uses `NaN` (not 0) for a missing objective so `isfinite` actually excludes it; axis dropdowns repopulate from the metrics a run actually produced. |
| M2 (overlay data mismatch) | **Fixed** — bundled with C3: `run_multi_streaming` gained `on_data_ready`, and Space Explorer overlays now reuse the run's exact preloaded data + `sim_seed`. |
| M4 (non-uniform simplex sampling) | **Fixed (2026-08-25), superseding the earlier revert below.** `engine/search.py` gained `sample_cdhr_portfolios` — coordinate-direction hit-and-run, EXACT uniform sampling over `{w: lo<=w<=hi, sum(w)=1}` with **no rejection step at all**, so it doesn't reproduce Dirichlet's failure mode (measured on the narrow-band space that broke Dirichlet: 100% fill, bounds and `sum=1` verified, vs Dirichlet's ~0.0067% acceptance). Also added `sample_sparse_portfolios` (random-cardinality active subsets — the "concentrated in a few assets" corners neither the old sampler nor CDHR ever reach) and `sample_vertex_portfolios` (the polytope's extreme points along random linear objectives), blended into a new `"mixed"` search method (`sample_mixed_portfolios`, 40/40/20 CDHR/sparse/vertex) now offered as the GUI's default Search Mode radio option. `sample_random_portfolios` (`"random"` method) is unchanged and still available for reproducing old runs. Coverage measured on the real 15-asset, `hi=0.30` search space: `sample_random_portfolios`'s 99th-percentile largest weight is 0.181 over 20,000 draws; `sample_mixed_portfolios`'s is >0.25. At equal budget (1500 candidates, shared `sim_seed`), `sample_mixed_portfolios` found a portfolio with positive 1st-percentile CAGR; neither `sample_random_portfolios` nor CDHR alone found any. 32 new collected test cases in `tests/test_search.py` (parametrised across all 5 samplers) cover the shared contract (`sum=1`, bounds, determinism, graceful infeasible-space handling), the M4 narrow-band regression specifically, CDHR's uniformity (moment check against the exact Beta(1,k-1) marginal), the coverage claim above, and sparse's cardinality-floor / never-switches-off-a-mandatory-asset invariants. See `PLAN_SEARCH.md` for the full design rationale. — Earlier attempt, superseded: Dirichlet(1,...,1) + rejection replaced box-uniform-then-normalise, and all tests/synthetic checks passed. Live use surfaced the actual failure mode: real search spaces for this tool routinely constrain every asset to a narrow 1-2 point band (e.g. `lo=0.29/hi=0.31`); Dirichlet sampling is uniform over the FULL simplex, so its acceptance rate against a box that narrow is ~0% (measured: 0/5000 after 2,010,000 draws on the user's actual 15-asset space), where box-uniform-then-normalise accepts ~1%. Reverted to box-uniform-then-normalise at the time. The independent uninitialised-memory bug found alongside this (`sample_random_portfolios` returning `np.empty` rows unfilled after an infeasible-bounds bail-out) is still fixed — `return results[:filled]` — and is unrelated to which sampler is used. |
| M8 (Space Explorer overlay marker uncomparable to its cloud) | **Fixed (2026-08-25).** An overlay ("crimson star") could be cached from a `sim_seed`/data window that no longer matched the cloud on screen — e.g. computed via the seedless fallback path before any run existed, then left stale after a run completed with matching `n_sim`/`horizon`/`block_size` GUI params (the only fields the old cache key checked). Measured impact: at `n_sim=1000`, the same portfolio's metrics move by up to 0.66pp (`annualised_return_p50`) / 1.96pp (`annualised_return_p1`) across `sim_seed` alone — enough to place a mismatched overlay tens of standard deviations from a tight search cloud around the identical portfolio (reproduced: seed mismatch measured at 20-80 sd off-center; matched seed lands at <1 sd). Fixed via `bootstrap_gui.assets.overlay_key(params, sorted_tickers, sim_seed)`, a pure function used everywhere the GUI decides whether a cached overlay is still valid; overlays computed via the fallback branch (no matching run loaded) are now drawn as an open, dimmed marker with a hover warning instead of looking equivalent to a real one. `sim_seed` and an `n_sim<5000` noise warning are now surfaced in the run log and chart title. 6 new tests in `tests/test_assets.py`, plus 2 end-to-end regression tests in `tests/test_runner.py` reproducing the star-vs-cloud distance numerically. See `PLAN_SEARCH.md` Part B. |
| M9 (Single Bootstrap / overlay-fallback / CLI disagree with the search cloud even at matched seed) | **Tried, reverted same day (2026-08-25) — real regression caught in production, same pattern as M4.** Root cause identified correctly: `run_bootstrap` (backs Single Bootstrap, the overlay fallback, and `scripts/run_single.py`) intersects dates across only the portfolio's OWN tickers (`load_all_returns`), while `run_multi_streaming`'s cloud always intersects the FULL search-space ticker set — a strictly wider set, hence a narrower-or-equal window. At `n_sim=50,000` (MC noise <0.03pp) a real 5-asset portfolio showed a ~1pp gap between Single Bootstrap and a Space Explorer search, traced to a 38-month date-window difference — confirmed to be a pure data-input mismatch, not a math error. **The fix tried — force `run_bootstrap` onto the FULL asset universe's window (`list_available_tickers()`/`load_returns_on_full_universe`, unconditionally, regardless of what the portfolio actually holds) — was wrong**, caught by the user within the hour on a second real portfolio: a 73% world-equity / 27% government-bond mix (`WEBN`/`GOVH`, 37.4 years of genuinely overlapping history, 7.83% geometric CAGR) got silently truncated to the 24.3-year window forced by `XDWS` — a ticker starting in 1999 that this portfolio doesn't even hold, just because it's also in the 15-asset universe. Simulated median 20-year CAGR dropped from a correct, plausible **7.82%** to an implausible **5.40%** — an even bigger, more misleading error than the one being fixed, and on a much more mundane portfolio (any 2-asset combination where one ticker in the wider universe starts late enough). **Reverted**: `run_bootstrap`, `CompareSubMode`, `SweepSubMode` all use `load_all_returns` (portfolio's own tickers) again; `list_available_tickers`/`load_returns_on_full_universe` were removed entirely (dead code with no remaining caller, and a proven footgun — better removed than left for someone to accidentally re-wire in without this context). **The lesson, stated once so it doesn't get re-learned by another attempt**: `run_multi_streaming`'s search-space-wide window and `run_bootstrap`'s own-tickers window serve genuinely different, non-reconcilable goals — cross-CANDIDATE comparability within one search (needs one shared window, non-negotiable given the sparse sampler deliberately varies each candidate's active tickers — see M4) vs. maximum-accuracy evaluation of ONE specific, named portfolio (needs its own tickers' full history). They will coincide only when a portfolio's own tickers happen to equal the search space's — forcing them to coincide in general trades a correct, explicable disagreement for a silently wrong shared number. The Space Explorer overlay's `comparable` flag (see the M8 entry above) is the right way to surface this: match the cloud exactly when the portfolio's tickers are covered by the loaded search space, and clearly mark (open/dimmed star, hover warning) — not silently degrade — when they aren't. `shannon_entropy`/`type_entropy` are correspondingly back to being computed over each tool's own weight-vector length; they were never audited as needing to match across tools, and no user-facing report has raised that they don't. Test suite: the tests asserting `run_bootstrap` matches the full-universe cloud convention were removed; `tests/test_runner.py::test_run_bootstrap_uses_own_ticker_window_not_full_universe` now locks in the CORRECT (own-tickers) behaviour using the exact `WEBN`/`GOVH` case that caught this, asserting `n_months_history` matches `load_all_returns` and is strictly wider than the full-universe window — so a future re-attempt at forcing universe-wide anchoring fails this test immediately instead of shipping. One fix from this finding remains valid and unrelated to the window choice: the overlay's hard `ValueError` when a portfolio held a ticker outside the current search space (`"Portfolio 'X' contains TICKER, which is outside the search space"`) was removable independently of M9's window decision — `load_all_returns` never had any "search space" restriction to begin with, so that hard-fail was purely a leftover check in `gui.py`'s `_compute_overlay_worker` that had no reason to exist even before this whole M9 detour. `_overlay_use_preloaded`/`_current_overlay_key` (`gui.py`) still take the portfolio and require its tickers to be covered by the loaded search space before using the preloaded (comparable) path; when they aren't, the worker falls through to the `run_bootstrap` fallback (now correctly own-tickers-anchored, comparable=False, open/dimmed marker) instead of raising — kept, tested by `test_run_bootstrap_succeeds_for_any_ticker_regardless_of_search_space`. |
| M10 (sparse sampler emits out-of-bounds weights on heterogeneous caps) | **Fixed (2026-08-25, found in the Opus re-audit).** `sample_sparse_portfolios` chose each row's active subset at RANDOM but sized it from `m_free_min`, computed off the caps sorted DESCENDING. A row whose support happened to come from the small-cap end then had `sum(hi[S]) < 1` — it cannot reach `sum(w) == 1` within bounds at all — so `_cdhr_core`'s starting point `lo + r*span/span_sum` overshot the caps, and the walk (which only ever preserves the sum, never repairs a violation) left it out of bounds for the whole chain. Worse, the move interval `[t_lo, t_hi]` inverts on such a row, so sampling it moved the weights FURTHER out. Measured: 1482/2000 rows violated their upper bounds on a 6-asset space with caps `[.6,.5,.15,.10,.10,.05]`; 318/2000 with mandatory (`lo>0`) assets present; **0** on a uniform-cap space — which is the only shape the original tests covered, so this shipped invisible. The repo's current `search.csv` uses uniform `hi=0.30` throughout and is therefore NOT affected, but any realistic per-asset cap scheme ("max 60% equity, max 15% gold") silently produced invalid portfolios that were then scored and Pareto-ranked as if real. Fixed in `engine/search.py`: the support is now extended along the row's own random priority order to the shortest prefix reaching `sum(hi[S]) >= 1`, then `max(drawn_cardinality, that_floor)` is used — preserving which assets are chosen at random while guaranteeing per-row feasibility. `_cdhr_core` additionally freezes a row whose interval is inverted rather than sampling the reversed interval, so a future caller-side mistake surfaces as a detectably stuck row instead of plausible-looking bad data. 5 new parametrised tests in `tests/test_search.py` covering heterogeneous caps with and without mandatory assets, plus a direct assertion of the `sum(hi[active]) >= 1` invariant (so a regression that stays in bounds by luck on one seed is still caught). |
| M11 (Multi-Portfolio Comparison ranks portfolios on DIFFERENT historical windows) | **Fixed (2026-08-25, found in the Opus re-audit).** The single worst defect found: `CompareSubMode._worker` called `load_all_returns(portfolio, ...)` inside its per-portfolio loop, so each selected portfolio was loaded on the intersection of its OWN tickers. Portfolios plotted on one fan chart and ranked in one summary table were therefore measured over different, non-overlapping historical periods — and since a portfolio built from long-history assets gets a longer, differently-composed window, the tool systematically rewarded holding assets with more data rather than better allocation. This is not a rounding-scale effect: comparing a 2-asset portfolio (449-month own window) against a 12-asset one (305 months), the ranking **reverses** — 7.90% vs 7.70% on their own windows (2-asset wins) versus 6.09% vs 7.70% on the shared 305-month window (12-asset wins decisively), a 1.8pp swing produced purely by the window mismatch. Fixed: `engine/data.py` gained `load_portfolios_on_common_window(portfolios, ...)`, which intersects calendar months across the UNION of every selected portfolio's tickers and returns one matrix plus zero-padded weights per portfolio; `CompareSubMode` now loads once through it (the historical-overlay path too, so the drawn history lines also span identical months), and the summary table states the shared window and how many tickers set it. Independent-resampling mode is exempt and unchanged — it deliberately never aligns assets to a common calendar. This is the correct scope for the idea M9 got wrong: a shared window is required when portfolios are RANKED against each other, and is wrong to force on a single portfolio being evaluated on its own terms (`run_bootstrap`/`load_all_returns` keep maximising that portfolio's own history — the 73/27 case that exposed M9 still reports its full 37.4 years). 2 new tests in `tests/test_data.py`, including that the single-portfolio case degenerates exactly to `load_all_returns` rather than narrowing further. |
| M12 (entropy metrics depend on the weight-vector length, not the portfolio) | **Fixed (2026-08-25, found in the Opus re-audit).** `shannon_entropy` normalised `H` by `ln(len(weights))` and `type_entropy` by `ln(n_types_in_the_ticker_list)` — the size of the CONTAINER, not a property of the holdings. The same 73/27 portfolio therefore scored **0.84** from the Single Bootstrap section (2-long weight vector) and **0.22** from a Space Explorer run (15-long, zero-padded) — a 4x cross-tool disagreement, the same class of defect as M8/M9/M11 but on the diversification axis, and `type_entropy` is a `cfg.PARETO_METRICS` objective. An equally-weighted 5-asset portfolio read as a perfect 1.00 in one tool and 0.59 in the other. Fixed by replacing both with the encoding-invariant effective-number (perplexity) form `exp(H)`: `effective_n_assets` and `effective_n_types` in `engine/metrics.py`, updated in `cfg.PARETO_METRICS`, `bootstrap_gui.assets.build_metric_list`, `engine/__init__` exports and the README. `exp(H)` is a MONOTONE transform of `H`, so every ranking within a single run — where the old denominator was constant — is bit-for-bit unchanged; only the cross-tool disagreement disappears. It is also directly interpretable ("effectively 1.79 assets", "effectively 7.73 types") and correctly keeps ranking a 15-asset equal-weight portfolio above a 5-asset one (a `ln(n_held)` normalisation, the other encoding-invariant option, would have scored both 1.00 and destroyed that ordering). Renamed rather than redefined in place so results CSVs written before today keep their old column name and are not silently reinterpreted. 3 new tests in `tests/test_metrics.py` (zero-padding invariance, diversification ordering preserved, equal-weight exactness). |
| M5 (lump-sum shortfall discarded) | **Fixed** — `LifeSimResult` gained `lump_outflow_requested`/`lump_outflow_received`; `summary()` reports `probability_of_lump_sum_shortfall`; the Life Strategy panel shows a warning line when it's > 0. |
| M6 (returns cache keyed on name only) | **Not a bug — verified false positive.** `bootstrap_gui/sections/lifecycle.py:522` already registers `self.library.on_change(self._returns_cache.clear)` in `__init__`, so ANY library edit (including re-saving a portfolio's weights under the same name) clears the whole cache. Confirmed empirically (edit → cache size drops to 0 → next read reflects the new weights). My original audit pass missed this registration because I read a narrower window of the file. Left untouched. |
| M7 (dead date_start/date_end fields) | **Fixed** — wired through: new "Historical Data Window" UI fields, `plan_to_dict`/`plan_from_dict` serialisation, `load_all_returns(..., date_start=, date_end=)`, and the returns cache key now includes them too (needed for correctness of the new feature, independent of M6). |
| L3 (historical overlay off-by-one + stretched axis) | **Fixed** — `gui.py`: inclusive date-range end is now one month earlier (was `horizon*12+1` months instead of `horizon*12`); `hist_years` now uses `np.arange(len(hist))/12.0` instead of a `linspace` that stretches spacing by a `N/(N-1)` factor. |
| L4 (fan chart mislabels x-axis for block_size>1) | **Fixed** — `gui.py`: `_plot_fan_chart` now uses `np.arange(n_steps) * block_size / 12.0`, matching each path column to the months it actually represents, instead of assuming the columns span the full nominal horizon evenly. |
| L5 (misleading block_months validation message) | **Fixed** — reworded; the full horizon is always simulated (`sample_monthly_returns` uses ceil + trims the excess), only the final *sampled historical block* is used in part. |
| L7 (silent zero-withdrawal at tax_rate_pct ≥ 100) | **Fixed, narrowly** — Life Strategy's Run button now hard-blocks on `tax_rate_pct` outside `[0, 100)` specifically (the one case that degrades silently to a "successful" run with fabricated zeros); other `validate()` warnings remain non-blocking by design. |
| L8 (`int()` vs `round()` in `_max_drawdown_length`) | **Declined.** Not implemented: the existing "naive reference" tests already encode `int()` truncation as ground truth, so this isn't an objectively-wrong convention (unlike C2) — changing it would require rewriting the independent reference tests that exist specifically to catch this class of regression, for a cosmetic one-step difference with no clear "more correct" answer. |
| L1, L2, L6, L9 | Not implemented — design/convention decisions, not bugs (L6's naming concern was addressed for free by C3's `sim_seed` addition). Left as documented caveats, per the audit's own framing. |
| §5 methodological caveats | Unchanged — these are properties of the method, not bugs, and remain true after the above fixes. |

---

## 1. CRITICAL

### C1 — `load_all_returns` / `preload_returns` align assets by row tail, not by calendar date

**Where:** `engine/data.py:222` and `engine/data.py:270`

```python
ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers])
```

Each asset is truncated to the **last** `min_len` rows. This is only correct if every asset
ends on the same month. They do not:

| ticker | last month | | ticker | last month |
|---|---|---|---|---|
| BTOP50 | 2026-05 | | GOVH | 2025-04 |
| XDWS | 2026-01 | | HPRD | 2023-03 |
| EXUS/GOLD/LGAP/WRDA/IWVL/MVOL/SP5A | 2025-12 | | IWMO/MWEQ/PRAM/WEBN | 2025-07 |

So row *i* of column A and row *i* of column B are **different calendar months** — up to 33
months apart. Every cross-asset relationship is destroyed: the matrix is used by
`simulate()` (`returns @ weights`) and by `sample_monthly_returns()`, i.e. by the Single
Bootstrap, the Space Explorer search, the Block-Size sweep **and** the Life Strategy Simulator.

**Reproduction (two assets):**

```bash
uv run python -c "
import numpy as np
from engine.data import load_all_returns, load_returns
w,R = load_all_returns({'WRDA':0.5,'HPRD':0.5})
print('corr as the engine sees it:', np.corrcoef(R.T)[0,1])
d1,r1=load_returns('HPRD',True); d2,r2=load_returns('WRDA',True)
m1,m2=dict(zip(d1,r1)),dict(zip(d2,r2)); c=sorted(set(d1)&set(d2))
print('corr date-aligned      :', np.corrcoef([m1[k] for k in c],[m2[k] for k in c])[0,1])"
```

→ `0.0197` vs `0.7496`.

**Reproduction on the user's own saved portfolio** (`Selected Portfolio`, 12 assets):

| quantity | current (tail-aligned) | correct (date-aligned) |
|---|---|---|
| mean pairwise \|corr\| | 0.190 | 0.454 |
| portfolio monthly σ | 2.02 % | 2.90 % |
| `volatility_10y` | 0.0309 | 0.0427 |
| `max_dd_depth_p2` | **26.0 %** | **44.5 %** |
| `annualised_return_p1` | **+1.26 %** | **−2.91 %** |

The tool currently understates the worst-2 % drawdown of this portfolio by 18 percentage
points and turns a −2.9 %/yr 1st-percentile outcome into a +1.3 %/yr one.

This is made worse by the fact that the **correlation heatmap in Portfolio Builder is
correct** (`gui.py:361-372` calls `compute_date_intersection` then `apply_date_filter`, so
all series cover the same window). The user sees true correlations in the UI while the
engine simulates scrambled ones.

**Patch.** Align on the date intersection inside `engine/data.py`. Add a helper and use it
from both `load_all_returns` and `preload_returns`:

```python
def align_on_common_dates(
    per_ticker: dict[str, tuple[list[tuple[int, int]], np.ndarray]],
    tickers: list[str],
) -> np.ndarray:
    """Column-stack returns on the intersection of the tickers' month sets.

    Aligning by row tail (the old behaviour) silently pairs different calendar
    months whenever two assets end on different dates, which destroys the
    cross-asset correlation the whole point of the bootstrap depends on.
    """
    common = set(per_ticker[tickers[0]][0])
    for t in tickers[1:]:
        common &= set(per_ticker[t][0])
    if not common:
        raise ValueError(
            "No overlapping months across " + ", ".join(tickers) +
            " — these assets cannot be simulated together."
        )
    months = sorted(common)
    cols = []
    for t in tickers:
        dates, arr = per_ticker[t]
        lookup = dict(zip(dates, arr))
        cols.append(np.array([lookup[m] for m in months], dtype=np.float64))
    return np.column_stack(cols), months
```

Keep the existing `apply_date_filter` step before this (user bounds still apply), then log
the resulting window (`months[0] … months[-1]`) instead of just the row count. Leave
`load_independent_returns` / `preload_independent_returns` untouched — not aligning is
deliberate there.

Also add a regression test: build two synthetic tickers with different end dates and assert
that `load_all_returns` returns the intersection window and that a known correlation is
preserved. Note `tests/test_data.py:56` is currently named
`test_load_all_returns_aligns_to_shortest_history` and asserts nothing about alignment —
rename it and make it assert the intersection.

**Warn the user:** after this fix, results change materially (histories get shorter — 326 →
292 months for `Selected Portfolio`) and every risk number gets worse. That is the point.

---

### C2 — Every return series is labelled one month too early

**Where:** `scripts/preprocess.py:104-113`

```python
for i, (date_str, price) in enumerate(rows):
    if i < n - 1:
        next_price = rows[i + 1][1]
        gross_ret  = next_price / price - 1.0
        ...
        out_rows.append({"month_year": date_str, ...})   # <- date of the EARLIER price
```

The return from price(M) to price(M+1) is month **M+1**'s return, but it is written under
month **M**. Confirmed against known events:

```bash
grep "^09/2008\|^10/2008" data/standard/WRDA.csv
# 09/2008,168860.687478,-0.09139009,...   <- price ratio 10/2008 ÷ 09/2008 = October's return
```

Same shift visible everywhere: LGAP's −43.98 % (Black Monday, **October** 1987) is stored as
`1987-09`; HPRD's −22.7 % (COVID, **March** 2020) is stored as `2020-02`; the 2008 crash
month appears as `2008-09`.

**Impact.** The bootstrap *distribution* is unaffected (same returns, same order), so this
does not change any un-filtered simulation. It does break:

- every `date_start` / `date_end` filter — it selects a window shifted one month earlier.
  A user excluding COVID with `2020-03 … 2020-05` does **not** exclude the crash.
- the "History: X → Y" strings from `compute_date_intersection`, and the asset table.
- `data/standard/BTOP50.csv` too — its source is a Year×Month return table, and January
  1987's +11.0 % was written under `12/1986`, so the same convention applies (consistently,
  at least).

**Patch.** In `_write_standard`, attribute each return to the *later* month and leave the
**first** row's return empty instead of the last:

```python
for i, (date_str, price) in enumerate(rows):
    if i == 0:
        out_rows.append({"month_year": date_str, "price_eur": round(price, 6),
                         "month_return": "", "month_return_after_TER": ""})
        continue
    prev_price = rows[i - 1][1]
    gross_ret = price / prev_price - 1.0
    net_ret   = (1.0 + gross_ret) / monthly_ter_factor - 1.0
    out_rows.append({"month_year": date_str, "price_eur": round(price, 6),
                     "month_return": round(gross_ret, 8),
                     "month_return_after_TER": round(net_ret, 8)})
```

Then regenerate `data/standard/` (`uv run python scripts/preprocess.py`). BTOP50 has no
raw-source parser in `preprocess.py` — it must be re-derived from
`data/BTOP50_Index_historical_data 3.csv` with the same convention, or shifted by one row by
hand; do not leave it on the old convention while the rest moves.

C1 and C2 are independent: because the shift is uniform across all series, the C1
intersection fix is correct either way.

---

### C3 — The portfolio search gives every candidate an independent random draw

**Where:** `engine/runner.py:216`

```python
def _eval_portfolio(weights: np.ndarray) -> dict:
    rng = np.random.default_rng()      # fresh OS entropy, per portfolio
```

`run_multi_streaming(seed=...)` only seeds the *portfolio sampler*, never the simulations.
So the 10 000 points in the Space Explorer cloud each carry an independent Monte-Carlo error,
and the Pareto frontier — which maximises `annualised_return_p1` and minimises
`max_dd_depth_p2` — selects the **luckiest draws**, not the best portfolios. Classic
winner's curse.

**Reproduction — same portfolio, 30 evaluations, n_sim=1000 (the default):**

| metric | sd across repeats | min→max range |
|---|---|---|
| `annualised_return_p1` | 0.75 pp | 3.5 pp |
| `annualised_return_p50` | 0.22 pp | 0.95 pp |
| `max_dd_depth_p2` | 1.81 pp | 7.2 pp |

That noise is the same order as the real dispersion across candidate portfolios.

**Fix — common random numbers.** Because `simulate()` draws `rng.integers(0, n_avail,
size=(n_sim, n_blocks))` where `n_avail` depends only on the (shared, preloaded) return
matrix, a fixed seed produces **identical block indices for every portfolio**. Differences
then reflect weights only.

```python
# runner.py
def _eval_portfolio(weights: np.ndarray) -> dict:
    rng = np.random.default_rng(_W_PARAMS["sim_seed"])   # same for every portfolio
```

Add `sim_seed` to the `params` dict built in `run_multi_streaming` (default it from the
existing `seed` argument, falling back to a fixed constant like 12345 rather than `None`).
Same treatment for the un-seeded overlay path (see M2).

**Measured benefit** — sd of the *difference* in `annualised_return_p1` between two
neighbouring portfolios (w = 60/40 vs 65/35), 20 repeats:

- independent seeds: **sd = 0.01082**
- common random numbers: **sd = 0.00125** (8.7× tighter)

Also document in the README that with n_sim=1000 the frontier is still noise-limited, and
consider raising the default `N_SIMULATIONS` for search runs.

---

## 2. HIGH

### H1 — A `block_size` that doesn't divide the horizon silently shortens it, but metrics still annualise by the full horizon

**Where:** `engine/simulation.py:107` (`n_blocks = horizon_months // block_size`, floor) and
`engine/metrics.py:76` (`final ** (1.0 / horizon_years)`).

With `horizon_years=10` (120 months):

| block_size | months actually simulated | `annualised_return_p50` (WRDA) |
|---|---|---|
| 1 | 120 | 10.27 % |
| 6 | 120 | 10.23 % |
| **7** | **119** | 10.15 % |
| **9** | **117** | 9.84 % |
| 12 | 120 | 10.29 % |
| **36** | **108** | **9.27 %** |

The ~1 pp drop at block_size=36 is pure truncation, not a market effect. The **Block-Size
Sensitivity** panel (`gui.py:2391`) sweeps `range(bs_min, bs_max+1)`, i.e. every integer,
so it plots this artifact as if it were a real sensitivity — the exact chart a user consults
to choose a block size.

Worse: `block_size > horizon_months` gives `n_blocks == 0`, `paths` of shape `(n_sim, 1)`,
and **all metrics silently equal 0** (a 0 % return / 0 drawdown "portfolio"):

```bash
uv run python -c "
import numpy as np; from engine.data import load_all_returns
from engine.runner import run_bootstrap_preloaded
w,R=load_all_returns({'WRDA':1.0})
print(run_bootstrap_preloaded(w,R,n_sim=500,horizon_years=3,block_size=60,rng=np.random.default_rng(1)))"
```

**Patch (preferred):** make `simulate()` cover the full horizon at monthly resolution by
reusing `sample_monthly_returns` — the machinery already exists and `engine.lifecycle`
already relies on it. This fixes H1 and M3 at once, and removes the whole
`block_size`-column-index bookkeeping from `engine/metrics.py`
(`_volatility_at_windows`, `_max_drawdown_length`, `_max_drawdown_area` would no longer need
a `block_size` argument, and no volatility window would ever be skipped).

```python
def simulate(weights, returns, n_sim, horizon_months, rng, block_size=1):
    monthly = sample_monthly_returns(weights, returns, n_sim, horizon_months, rng,
                                     block_months=max(1, block_size))
    with np.errstate(over="ignore"):
        cum = np.cumprod(1.0 + monthly, axis=1)
    return np.hstack([np.ones((n_sim, 1)), cum])
```

This changes `paths.shape` from `(n_sim, n_blocks+1)` to `(n_sim, horizon_months+1)` for
`block_size > 1` — update `tests/test_simulation.py:25`
(`test_simulate_shape_block_size_greater_than_one`), the `block_size=` call sites in
`compute_metrics`, and `_plot_fan_chart` (see L4). It is a deliberate behaviour change: say
so in the commit message and in the README.

**Minimum patch if the shape change is judged too invasive:** annualise by the months
actually simulated (`n_blocks * block_size / 12`) rather than `horizon_years`, and raise a
clear `ValueError` when `block_size > horizon_months`.

---

### H2 — `LifeSimResult.summary()["shortfall_months_median"]` is wrong for the path-dependent withdrawal styles

**Where:** `engine/lifecycle.py:756-760`

```python
shortfall_months = np.sum(
    (self.planned_withdrawals[None, :] > 0)
    & (self.realised_withdrawals < self.planned_withdrawals[None, :] - 1e-9),
    axis=1,
)
```

`planned_withdrawals` is the **median across simulations** of what was requested. For FIXED
withdrawals that equals each path's own request, so the number is right. For
`PERCENTAGE_OF_PORTFOLIO`, `PERCENTAGE_RAMP` and `DRAWDOWN_CURTAILED` the request is
path-dependent, so every below-median path is counted as "shortfall" even when it received
exactly 100 % of what it asked for.

**Reproduction** (a `% of portfolio` plan where no path is ever short):

```bash
uv run python -c "
import numpy as np
from engine.lifecycle import *
from engine.lifecycle import _simulate_from_monthly_returns
plan=LifePlan(portfolio_name='x',initial_capital=1_000_000,tax_rate_pct=26,inflation_pct=0.0,
  adjust_for_inflation=False,
  phases=[Phase(kind=PhaseKind.WITHDRAW,years=10,
                withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
                withdrawal_pct_per_month=0.3)])
r=_simulate_from_monthly_returns(plan, np.random.default_rng(1).normal(0.005,0.04,(2000,120)))
print('reported:', r.summary()['shortfall_months_median'])
print('truth   :', np.median(np.sum((r.withdrawal_requested>0)&
      (r.realised_withdrawals<r.withdrawal_requested-1e-9),axis=1)))"
```

→ reported **59** shortfall months out of 120; truth **0**.

**Patch:** compare each simulation against its own request.

```python
shortfall_months = np.sum(
    (self.withdrawal_requested > 0)
    & (self.realised_withdrawals < self.withdrawal_requested - 1e-9),
    axis=1,
)
```

Add a test asserting 0 shortfall months for a `% of portfolio` plan that never exhausts, and
keep an existing-style FIXED test to show the number is unchanged there.

`summary()` is not currently rendered by the GUI — this is a public-API bug, not a
visible-number bug. Fix it anyway; it is the obvious thing to surface next.

---

### H3 — `DRAWDOWN_CURTAILED` measures a drawdown that the withdrawals themselves create

**Where:** `engine/lifecycle.py:_resolve_withdrawal_target` (style 2) and the `running_max`
updates in `_simulate_from_monthly_returns:833-851`.

`running_max` tracks the running peak of the **euro balance `V`**, and `V` is reduced by
every withdrawal and increased by every contribution. So `dd = (running_max - V)/running_max`
mixes "the market fell" with "I have been spending" and "I stopped contributing".

**Reproduction — a market that rises +0.40 % every single month (zero market drawdown, ever):**

```bash
uv run python -c "
import numpy as np
from engine.lifecycle import *
from engine.lifecycle import _simulate_from_monthly_returns
plan=LifePlan(portfolio_name='x',initial_capital=1_000_000,tax_rate_pct=26,inflation_pct=0.0,
  adjust_for_inflation=False,n_sim=1,
  phases=[Phase(kind=PhaseKind.WITHDRAW,years=20,monthly_amount=4000,
                withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
                reduced_monthly_amount=2500,drawdown_threshold_pct=5.0)])
w=_simulate_from_monthly_returns(plan,np.full((1,240),0.004)).withdrawal_requested[0]
print('months curtailed:',int((w<4000).sum()),' first cut at month',int(np.argmax(w<4000))+1)"
```

→ **45 of 240 months curtailed**, first cut at month 148, in a market that never once went
down. A user setting up "cut my spending when markets crash" gets "cut my spending once I
have spent 5 % of my peak balance", which is a different — and much more frequently
triggered — rule. It also means an ACCUMULATE phase preceding the WITHDRAW phase sets a peak
that the withdrawal phase can never regain.

**Patch:** track a cash-flow-neutral unit index alongside `V` and take the drawdown from
*that*.

```python
# in _simulate_from_monthly_returns, next to V/B:
unit = np.ones(n_sim, dtype=np.float64)          # market factor only
unit_max = np.ones(n_sim, dtype=np.float64)
...
for i in range(horizon_months):
    V = np.maximum(V * (1.0 + monthly[:, i]), 0.0)
    unit = unit * (1.0 + monthly[:, i])
    unit_max = np.maximum(unit_max, unit)
    # lump sums / contributions: do NOT touch unit / unit_max
    ...
    w = _resolve_withdrawal_target(style_i, unit, unit_max, ...)   # style 2 only
```

`_resolve_withdrawal_target` keeps its signature — style 2 receives `(unit, unit_max)` while
style 3 (`% of portfolio`) must keep receiving the real `V`. Cleanest is to pass both and let
the function pick; add a short comment saying why the two styles read different series.

Then update the README section "Drawdown-curtailed" and the docstring in
`engine/lifecycle.py:64-76`, which currently promise `engine.metrics._drawdown_series`
semantics (a *market* drawdown) while delivering a balance drawdown. Existing tests
`test_resolve_withdrawal_target_drawdown_curtailed_*` test the pure function and stay valid;
`test_drawdown_curtailment_switches_target_month_by_month` (`tests/test_lifecycle.py:420`)
will need its expected values recomputed.

If the current behaviour is wanted as an option, keep it as a separate style
(`BALANCE_DRAWDOWN_CURTAILED`) rather than silently redefining the existing one — but the
default should be the market-drawdown version.

---

## 3. MEDIUM

### M1 — Missing metrics silently become `0` in the Space Explorer chart and the Pareto frontier

**Where:** `gui.py:1461-1462` (`r.get(x_key, 0)`), `gui.py:1512` (`r.get(n, 0)` for Pareto),
`gui.py:2139` (`metrics.get(..., 0)` for the bar chart).

`_volatility_at_windows` (`engine/metrics.py:104-113`) **skips** any window that exceeds the
horizon or isn't a multiple of `block_size`. So:

- horizon 5 y → `volatility_10y` absent from every result. It is a `cfg.PARETO_METRICS`
  objective *and* a dropdown option, and every point silently becomes 0.0 — the objective
  drops out of the frontier without a word, and the scatter draws a column of zeros.
- `block_size = 7` → **all four** volatility windows absent.

```bash
uv run python -c "
import numpy as np; from engine.data import load_all_returns
from engine.runner import run_bootstrap_preloaded
w,R=load_all_returns({'WRDA':1.0})
m=run_bootstrap_preloaded(w,R,n_sim=500,horizon_years=5,block_size=6,rng=np.random.default_rng(1))
print('volatility_10y present?', 'volatility_10y' in m)"
```

**Patch:** never substitute 0 for a missing metric.

- `_regenerate_chart`: drop points where `x_key`/`y_key` is missing and show a status message
  naming the metric and the count.
- Pareto: build the objective list from the metrics actually present; if a
  `cfg.PARETO_METRICS` entry is missing from *all* results, skip that objective and log it
  visibly in the run log rather than feeding zeros to `compute_pareto`.
- Populate the metric dropdown from the keys actually produced by the current run
  (`sorted(self.all_results[0])`) instead of the static `build_metric_list()`.

Note H1's preferred patch also removes the `block_size` alignment cause; the
horizon-too-short cause remains, so M1 still needs fixing on its own.

### M2 — Space Explorer overlays are computed on a different historical window (and a different RNG) than the cloud

**Where:** `gui.py:1092-1102` (`_compute_overlay_worker`).

The cloud comes from `preload_returns(all 15 search-space tickers)` → the matrix is truncated
to the shortest of *all* of them (326 months today). The overlay calls
`run_bootstrap(portfolio)`, which reloads only *that portfolio's* tickers → a possibly much
longer window (e.g. WRDA+GOVH → 484 months), plus `random_seed=None`. The overlay marker is
therefore not comparable with the cloud it is plotted on top of — different data, different
draw.

**Patch:** compute overlays from the same preloaded matrix and window as the run (keep the
matrix from `run_multi_streaming` on the section, project the overlay's weights onto
`sorted_tickers` with zeros for absent assets, and call `run_bootstrap_preloaded`), with the
same `sim_seed` as C3. If the overlay contains a ticker outside the search space, refuse with
a clear message rather than silently using a different dataset.

### M3 — Block-endpoint sampling understates drawdowns

**Where:** `engine/simulation.py:simulate` (block path keeps only block endpoints) → all of
`engine/metrics.py`'s drawdown functions.

Isolating the resolution effect (same sampled monthly returns, measured monthly vs at block
endpoints only):

| block | drawdown at monthly resolution | at block endpoints (current) |
|---|---|---|
| 3 | 52.95 % | 51.82 % |
| 6 | 55.17 % | 52.24 % |
| 12 | 59.76 % | **55.31 %** |

3–4.5 pp understated, and `max_dd_length_months_p*` / `mda_months_p*` are quantised to
multiples of `block_size` (they are literally `steps * block_size`). Fixed for free by H1's
preferred patch.

### M4 — `sample_random_portfolios` is not uniform on the simplex

**Where:** `engine/search.py:37` — docstring says "uniformly on the constrained simplex",
implementation draws uniform in the box then normalises. Normalising a box-uniform vector
concentrates mass near the centroid; concentrated portfolios (one asset at its 0.3 cap, the
rest near 0) are heavily under-sampled. With 15 assets and `hi=0.3` the search barely visits
the boundary of its own search space.

**Patch:** either correct the docstring, or (better) sample `Dirichlet(alpha=1)` over the
k assets and reject rows violating `lo`/`hi` — same rejection loop, genuinely uniform:

```python
raw = rng.dirichlet(np.ones(k), size=batch)
valid = np.all((raw >= lo) & (raw <= hi), axis=1)
```

For tight `hi` bounds the acceptance rate will drop; the existing progress logging and the
"0 valid portfolios" bail-out already handle that. Mention the change in the README, because
the shape of the point cloud will visibly change.

### M5 — A negative lump sum that the portfolio can't fund is silently shrunk

**Where:** `engine/lifecycle.py:838-841`

```python
V, B, tax, _net = _apply_inflow_or_lump(V, B, lump_amount, tax_rate)
```

`_net` is discarded. A "buy a house, −€300 000" lump sum at a moment when only €120 000 is
available becomes a €120 000 purchase — no error, no flag, and the result object has no field
that records it. The user sees a plan that "worked".

**Patch:** add `lump_sum_received: np.ndarray  # (n_sim, horizon_months)` and
`lump_sum_requested` to `LifeSimResult`, fill them in the month loop, and expose a
`lump_sum_shortfall_paths` fraction in `summary()` / the readout panel. Minimum viable: a
per-simulation boolean `lump_shortfall` plus a warning line in the Life Strategy status bar.

### M6 — Life Strategy caches return data under the portfolio *name* only

**Where:** `bootstrap_gui/sections/lifecycle.py:1032-1040`

```python
key = (portfolio_name,)
cached = self._returns_cache.get(key)
```

The cache stores `(weights, returns)`. Edit a portfolio's weights in Portfolio Builder, come
back to Life Strategy in the same session, re-run → the **old weights** and old matrix are
used. Silent wrong answer.

**Patch:** key on the weights themselves, e.g.
`key = (portfolio_name, tuple(sorted(weights_dict.items())))`, or subscribe to
`PortfolioLibrary.on_change` and clear the cache.

### M7 — `LifePlan.date_start` / `date_end` are dead fields

**Where:** declared at `engine/lifecycle.py:165-166`; never read anywhere (`grep` confirms:
only the declaration). `plan_to_dict` doesn't serialise them either, so a hand-edited
`.life_plans.json` containing them is silently ignored while looking like it works.

**Patch:** either wire them through — `load_all_returns(weights, cfg.USE_AFTER_TER_RETURNS,
date_start=plan.date_start, date_end=plan.date_end)` in `_get_returns_cached`, add them to
`plan_to_dict`/`plan_from_dict` and to the cache key — or delete the two fields. Wiring them
is the better call: the Life Strategy panel is the only section without a date filter.

---

## 4. LOW / polish

- **L1 — Moving-block edge under-weighting.** `sample_monthly_returns` uses non-circular
  overlapping blocks (`n_starts = T - b + 1`), so the first and last `b-1` months are
  systematically under-sampled. With T=292: at `b=12` the extreme months get 8.7 % of uniform
  weight and 3.9 % of the sampling mass is distorted; at `b=24`, 4.5 % and 8.6 %. A circular
  block bootstrap (wrap indices modulo T) removes it in three lines: `idx = (starts[:,:,None]
  + offsets) % T` with `starts` drawn from `[0, T)`. Worth doing, and worth a README note
  either way.
- **L2 — Real-growth raise lands one month early.** `engine/lifecycle.py:434`:
  `growth = (1 + g/100) ** floor(phase_elapsed / 12)` with `phase_elapsed` 1-indexed, so the
  first raise applies in month **12** of the phase rather than month 13. Defensible as a
  convention, but it is currently undocumented and frozen into
  `tests/test_lifecycle.py:188`. Decide, then document in the `Phase.real_growth_pct` comment.
- **L3 — Historical-overlay window is one month too long, and stretched.** `gui.py:1953-1955`
  builds `hist_end = start_year + horizon` with an *inclusive* filter → `horizon*12 + 1`
  months against a `horizon*12`-month simulation. Use `end = start + horizon years − 1 month`.
  Separately, `gui.py:2098`: `hist_years = np.linspace(0, len(hist)/12, len(hist))` puts the
  last point at `(months+1)/12`; it should be `np.arange(len(hist)) / 12.0`.
- **L4 — Fan chart stretches truncated block paths to the full horizon.** `gui.py:2083`:
  `years = np.linspace(0, horizon, n_steps)` labels `n_blocks` blocks as if they spanned
  `horizon` years. Disappears with H1's preferred patch; otherwise use
  `np.arange(n_steps) * block_size / 12.0`.
- **L5 — `validate()`'s block-divisibility warning is misleading.** `engine/lifecycle.py:253`
  warns "the final block will be truncated"; `sample_monthly_returns` uses `ceil` and truncates
  the *last* block to length, so the full horizon **is** covered and nothing is lost. The real
  (mild) effect is that the final partial block always samples the *beginning* of a historical
  block. Reword, or drop the warning.
- **L6 — `run_multi_streaming(seed=...)` doesn't make a run reproducible** (it only seeds the
  portfolio sampler). Rename it `portfolio_seed` and add `sim_seed` (see C3), or document.
- **L7 — Life Strategy only hard-blocks on "no phases".** `bootstrap_gui/sections/lifecycle.py:1049`
  — every other `validate()` problem is a warning the user can run past. Mostly harmless, but
  a `tax_rate_pct >= 100` makes the gross-up denominator `1 - rate*u` go negative once the
  latent-gain fraction exceeds `1/rate`, and withdrawals silently become **zero** instead of
  erroring. Block at least the out-of-range tax rate, n_sim and block_months.
- **L8 — `_max_drawdown_length` truncates the percentile** (`int(np.percentile(...))` instead
  of rounding). One-step bias, cosmetic.
- **L9 — Memory.** `_simulate_from_monthly_returns` allocates six `(n_sim, horizon_months)`
  float64 arrays. At the `_MAX_SIM_MONTHS = 30_000_000` guard that is ≈1.4 GB. Either lower
  the guard or count arrays, not sim-months.

---

## 5. Methodological caveats to document (not bugs — do not "fix" these in code)

1. **No parameter uncertainty.** The bootstrap resamples the historical returns, so every
   projection is conditioned on "the historical mean is the true mean". With ~292 months and a
   2.9 % monthly σ, the standard error on the mean is ≈0.17 %/month ≈ **2 %/year** — wider than
   the gap between most of the portfolios the search compares. The P1–P99 fan does *not*
   contain that uncertainty. Say so, prominently, next to the fan chart.
2. **Returns are nominal; only cash flows are inflated.** Correct and documented in
   `engine/lifecycle.py`, but it means "real" outputs assume the historical *nominal* return
   distribution recurs alongside the user's assumed forward inflation. A 1979-2025 nominal
   equity series carries 1970s-80s inflation inside it.
3. **`simulate` and `simulate_independent` use different rebalancing conventions for
   `block_size > 1`.** `_precompute_block_gross` compounds the monthly *portfolio* return
   (monthly rebalancing); `simulate_independent` combines per-asset block gross returns as
   `1 + Σ wₐ(gₐ − 1)` (buy-and-hold within the block). So switching "break correlations" on
   changes two things at once. Document, or make them consistent.
4. **TER on BTOP50.** `data/TER_table.csv` applies 0.50 %/yr to BTOP50, a managed-futures
   index that is already reported net of manager fees. Verify whether that is intentional
   double-counting of costs.
5. **Data provenance is unverifiable from the repo.** Every `data/raw_curvo/*.csv` carries the
   same copy-pasted header ("UBS ETF (IE) MSCI Emerging Markets…") regardless of ticker. The
   data underneath are clearly different series, but nothing in the repo records what each
   ticker actually is. Add a provenance column to `TER_table.csv` or a `data/README.md`.
6. **`volatility_Ny` is the cross-sectional dispersion of N-year annualised returns**, not
   return volatility in the usual sense. The chart title and axis label are accurate; the
   metric *name* is the misleading part. Consider `dispersion_Ny`.

---

## 5b. New capability (2026-08-26): evolutionary search

`engine.runner.run_evolutionary_streaming` — a multi-generation refinement
search that spends most of each generation's budget on crossover/local
perturbation around the Pareto frontier accumulated so far, biased by
NSGA-II crowding distance, instead of one flat exploration batch. See
`README.md`'s "Evolutionary search" section for the algorithm and the
measured frontier-density / extreme-value trade-off, and `PLAN_SEARCH.md`
for design notes. Selectable in the GUI as a fourth Search Mode; the
Pareto Objectives panel it shares with the chart's frontier line is also
new (see the "Pareto frontier" section of `README.md`).

One bug caught and fixed during development, before ever shipping (so not
a numbered `M` finding — those are for defects that reached a working
build): `engine.pareto.crowding_distance`'s per-objective loop assigned
`+inf` to the first/last points in SORTED order unconditionally, even for
an objective with zero spread (every point identical on it, e.g. every
candidate landing on the same `volatility_10y` due to a data quirk) —
handing two arbitrarily-tied points a spurious "most isolated" credit
they hadn't earned. Fixed to skip a zero-span objective's contribution
entirely, matching the function's own docstring. Caught by
`tests/test_pareto.py::test_crowding_distance_constant_objective_contributes_nothing`
before this ever ran against real search results.

`engine.search.local_perturb_portfolios` had a related n_steps=0 bug: the
"take a short walk from a frontier point" wrapper floored `n_steps` at 1
(`max(1, n_steps)`), so passing 0 — a legitimate "no perturbation at all"
request, e.g. the tail end of a cooling schedule — still moved every
point by one CDHR step instead of returning it unchanged. `_cdhr_core`
itself already handles `n_steps=0` correctly (its step loop is a no-op);
the wrapper's floor was fixed to `max(0, n_steps)`. Caught by
`tests/test_search.py::test_local_perturb_zero_steps_returns_the_seed_unchanged`.

---

## 6. Suggested order of work

1. **C1** (alignment) — everything downstream is wrong until this lands.
2. **C2** (date labels) + regenerate `data/standard/`, BTOP50 included.
3. **C3** (common random numbers) + **M2** (overlay comparability).
4. **H1** via the `sample_monthly_returns` refactor of `simulate()` — closes **M3** and the
   `block_size` half of **M1** too.
5. **H2**, **H3**, **M5**, **M6**, **M7** (Life Strategy correctness).
6. **M1** remainder, **M4**, then the L-items.
7. Re-run the full suite after each step; C1/C2/H1/H3 all change expected values, so update the
   affected tests deliberately (listed inline above) rather than loosening tolerances.

---

## 7. Checked and found correct — do not change

- `apply_withdrawal` gross-up, pro-rata basis reduction and exhaustion cap
  (`net = g - tax` collapses both branches exactly). Verified algebraically and by test.
- Inflation indexing: `_inflation_index`, `_phase_schedule`'s `infl_m ** elapsed_months`,
  `_lump_sum_schedule`'s `infl_m ** elapsed_target` and the `inflation_index[1:H+1]` deflator
  are mutually consistent. Deflation commutes with the percentile (positive per-column scalar).
- `cash_flow_bands` / `cash_flow_percentiles_at` compute `withdrawal_gross` per simulation
  *before* taking percentiles. This is correct and the docstring explains why — do not
  "simplify" it into a sum of bands.
- `pareto_frontier_indices` is correct despite mutating `is_efficient` mid-loop: dominance is
  transitive, and truly non-dominated points are never removed, so any dominated point is
  always caught by a survivor. Ties (identical rows) are both retained, correctly.
- `_last_reset_index` / `_max_drawdown_length` / `_max_drawdown_area` vectorised reset trick —
  already covered by reference-implementation tests at several seeds and block sizes.
- `_precompute_block_gross` log-space rolling product — safe; no asset has a month ≤ −100 %
  (worst is LGAP −43.98 %).
- `PERCENTAGE_RAMP`'s `np.linspace(start, end, hi-lo)` hits the stated endpoints exactly.
- Ruin bookkeeping: `ruin_month` records the first month with `V <= 0`; `median_ruin_year =
  (median(ruin_month)+1)/12` and the GUI's `ruined < month` "ruined by now" test are both
  right for the 0-indexed convention.
- Dead-path rebirth in `_apply_inflow_or_lump` resets the cost basis, which is the consistent
  choice given `LOSS_CARRYFORWARD_SUPPORTED = False`.
- Portfolio weights: the Builder enforces sum = 100 % on save; all 84 saved portfolios sum to
  1.0; `clean_portfolio` only drops zero weights.
- Per-ticker CSVs have **no internal month gaps** (verified for all 15), so contiguity inside a
  bootstrap block is genuine.
- `PortfolioLibrary` / `LifePlanLibrary` atomic writes (`.tmp` + `os.replace`).
