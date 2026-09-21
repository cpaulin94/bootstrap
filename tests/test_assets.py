"""Tests for bootstrap_gui.assets — currently just overlay_key(), the
identity an overlay must match to be considered comparable to the cloud
it's drawn on top of (see the Space Explorer "crimson star" bug: an
overlay computed against a different ticker set or sim_seed than the
cloud can land tens of standard deviations away from it — see
tests/test_runner.py's test_overlay_star_with_different_seed_lands_far_from_cloud
for the numeric reproduction).
"""

from __future__ import annotations

from bootstrap_gui.assets import overlay_key


def _params(**overrides) -> dict:
    base = {
        "n_sim": 1000,
        "horizon_years": 10,
        "block_size": 6,
        "date_start": None,
        "date_end": None,
        "independent": False,
    }
    base.update(overrides)
    return base


def test_overlay_key_distinguishes_sim_seed():
    a = overlay_key(_params(), ["GOLD", "GOVH"], sim_seed=1)
    b = overlay_key(_params(), ["GOLD", "GOVH"], sim_seed=2)
    assert a != b


def test_overlay_key_distinguishes_ticker_set():
    """Regression test for the actual bug: same simulation params, but a
    different search-space ticker set (which changes the historical date
    window via preload_returns' date intersection) must produce a
    different key, so a stale overlay gets recomputed instead of drawn as
    if it were comparable."""
    a = overlay_key(_params(), ["GOLD", "GOVH", "HPRD"], sim_seed=42)
    b = overlay_key(_params(), ["GOLD", "GOVH"], sim_seed=42)
    assert a != b


def test_overlay_key_stable_under_dict_key_order():
    p1 = {"n_sim": 1000, "horizon_years": 10, "block_size": 6,
          "date_start": None, "date_end": None, "independent": False}
    p2 = {"independent": False, "block_size": 6, "date_start": None,
          "n_sim": 1000, "date_end": None, "horizon_years": 10}
    assert overlay_key(p1, ["GOLD"], 42) == overlay_key(p2, ["GOLD"], 42)


def test_overlay_key_stable_under_ticker_list_order():
    a = overlay_key(_params(), ["GOLD", "GOVH"], sim_seed=42)
    b = overlay_key(_params(), ["GOVH", "GOLD"], sim_seed=42)
    assert a == b


def test_overlay_key_matches_for_identical_context():
    a = overlay_key(_params(), ["GOLD", "GOVH"], sim_seed=42)
    b = overlay_key(_params(n_sim=1000), ["GOLD", "GOVH"], sim_seed=42)
    assert a == b


def test_overlay_key_none_tickers_distinct_from_any_real_run():
    """The fallback-branch marker (sorted_tickers=None, sim_seed=None) must
    never accidentally collide with a real run's key."""
    fallback = overlay_key(_params(), None, None)
    real_run = overlay_key(_params(), ["GOLD"], 42)
    assert fallback != real_run


# ═══════════════════════════════════════════════════════════════════════
# default_pareto_direction — seeds the Pareto Objectives checkboxes'
# default maximize/minimize, and must cover every metric name
# build_metric_list can actually produce.
# ═══════════════════════════════════════════════════════════════════════

from bootstrap_gui.assets import build_metric_list, default_pareto_direction


def test_default_pareto_direction_covers_every_metric_build_metric_list_produces():
    """If a new metric is ever added to build_metric_list without teaching
    default_pareto_direction its convention, the Pareto Objectives panel
    would silently default an unrecognised metric to 'minimize' — wrong
    for anything return- or diversification-like. This fails loudly
    instead."""
    known_prefixes = ("annualised_return", "volatility_", "max_dd_", "mda_",
                      "effective_n_")
    for name in build_metric_list():
        assert name.startswith(known_prefixes), (
            f"{name!r} doesn't match any prefix default_pareto_direction "
            f"recognises — its default direction needs deciding explicitly"
        )


def test_default_pareto_direction_returns_are_maximized():
    assert default_pareto_direction("annualised_return_p50") == "maximize"
    assert default_pareto_direction("annualised_return_p1") == "maximize"


def test_default_pareto_direction_diversification_is_maximized():
    assert default_pareto_direction("effective_n_assets") == "maximize"
    assert default_pareto_direction("effective_n_types") == "maximize"


def test_default_pareto_direction_risk_metrics_are_minimized():
    assert default_pareto_direction("volatility_10y") == "minimize"
    assert default_pareto_direction("max_dd_depth_p2") == "minimize"
    assert default_pareto_direction("max_dd_length_months_p2") == "minimize"
    assert default_pareto_direction("mda_months_p2") == "minimize"
