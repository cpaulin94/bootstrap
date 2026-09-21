"""Closed-form tests for engine.lifecycle — the Life Strategy Simulator.

Every test here has a known answer computed by hand (or via an independent
formula) in the test itself — these are not just "does it run" smoke tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.lifecycle import (
    LifePlan,
    LifeSimResult,
    LumpSum,
    Phase,
    PhaseKind,
    WithdrawalStyle,
    apply_withdrawal,
    simulate_life_strategy,
    _phase_schedule,
    _resolve_withdrawal_target,
    _simulate_from_monthly_returns,
)


def _zero_return_history(n_months: int = 24, n_assets: int = 1) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(n_assets) / n_assets
    returns = np.zeros((n_months, n_assets))
    return weights, returns


# ── 1. Zero return, accumulate only ─────────────────────────────────────────

def test_zero_return_accumulate_only_matches_closed_form():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=100_000.0,
        tax_rate_pct=26.0,
        inflation_pct=0.0,
        adjust_for_inflation=True,
        phases=[Phase(PhaseKind.ACCUMULATE, years=10, monthly_amount=1_000.0)],
        n_sim=200,
        block_months=12,
        seed=1,
    )
    result = simulate_life_strategy(plan, weights, returns)

    expected_final = 100_000.0 + 120 * 1_000.0  # 220,000
    assert np.allclose(result.values[:, -1], expected_final)
    assert np.all(result.taxes_paid == 0.0)
    assert np.all(result.ruin_month == -1)


# ── 2. Deterministic constant return ────────────────────────────────────────

def test_deterministic_constant_return_matches_closed_form():
    r = 0.01
    weights = np.array([1.0])
    returns = np.array([[r]])  # single month of history, block_months=1 -> always samples r
    v0 = 50_000.0
    horizon_years = 5
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=v0,
        inflation_pct=0.0,
        phases=[Phase(PhaseKind.HOLD, years=horizon_years)],
        n_sim=30,
        block_months=1,
        seed=2,
    )
    result = simulate_life_strategy(plan, weights, returns)

    expected_final = v0 * (1.0 + r) ** (horizon_years * 12)
    assert np.allclose(result.values[:, -1], expected_final)
    # every path identical (single-month history -> no actual randomness)
    assert np.allclose(result.values, result.values[0])


# ── 3. Tax gross-up correctness ─────────────────────────────────────────────

def test_apply_withdrawal_gross_up_correctness():
    V = np.array([200_000.0])
    B = np.array([100_000.0])
    rate = 0.26
    w_net = 10_000.0

    V_new, B_new, tax_paid, net_received = apply_withdrawal(V, B, w_net, rate)

    u = 0.5
    expected_g = w_net / (1 - rate * u)
    expected_tax = rate * expected_g * u

    assert net_received[0] == pytest.approx(10_000.0, abs=1e-6)
    assert tax_paid[0] == pytest.approx(expected_tax, abs=1e-6)
    assert V_new[0] == pytest.approx(200_000.0 - expected_g, abs=1e-6)
    assert B_new[0] == pytest.approx(100_000.0 - expected_g * (100_000.0 / 200_000.0), abs=1e-6)


# ── 4. No tax on a loss ──────────────────────────────────────────────────────

def test_apply_withdrawal_no_tax_on_loss():
    V = np.array([80_000.0])
    B = np.array([100_000.0])
    V_new, B_new, tax_paid, net_received = apply_withdrawal(V, B, 10_000.0, 0.26)

    assert tax_paid[0] == pytest.approx(0.0, abs=1e-9)
    assert net_received[0] == pytest.approx(10_000.0, abs=1e-6)
    assert V_new[0] == pytest.approx(70_000.0, abs=1e-6)


# ── 5. Exhaustion and floor at zero ─────────────────────────────────────────

def test_exhaustion_floors_at_zero_and_stays():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=1_000.0,
        tax_rate_pct=26.0,
        inflation_pct=0.0,
        phases=[Phase(PhaseKind.WITHDRAW, years=3 / 12, monthly_amount=5_000.0)],
        n_sim=50,
        block_months=1,
        seed=3,
    )
    result = simulate_life_strategy(plan, weights, returns)

    assert result.values.shape == (50, 4)
    assert np.all(result.values >= 0.0)
    assert np.all(result.ruin_month == 0)
    # month 0: only 1,000 was available against a 5,000 request
    assert np.allclose(result.realised_withdrawals[:, 0], 1_000.0)
    # subsequent months: nothing left to give
    assert np.allclose(result.realised_withdrawals[:, 1], 0.0)
    assert np.allclose(result.realised_withdrawals[:, 2], 0.0)
    assert np.allclose(result.values[:, 1:], 0.0)


# ── 6. Rebirth from a lump sum, no tax on the immediate withdrawal ─────────

def test_lump_sum_rebirth_no_tax_on_immediate_withdrawal():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=0.0,
        tax_rate_pct=26.0,
        inflation_pct=0.0,
        phases=[Phase(PhaseKind.WITHDRAW, years=2 / 12, monthly_amount=1_000.0)],
        lump_sums=[LumpSum(at_year=0.0, amount=50_000.0, label="inheritance")],
        n_sim=20,
        block_months=1,
        seed=4,
    )
    result = simulate_life_strategy(plan, weights, returns)

    assert np.all(result.ruin_month == -1)  # never observed at zero at a recorded checkpoint
    assert np.allclose(result.taxes_paid[:, 0], 0.0)  # fresh basis == value -> u=0
    assert np.allclose(result.values[:, 1], 49_000.0)  # 50,000 - 1,000 withdrawal
    assert np.allclose(result.values[:, 2], 48_000.0)


# ── 6b. M5: negative lump sum exceeding the portfolio is tracked, not silent ─

def test_lump_outflow_shortfall_is_tracked_when_portfolio_cannot_cover_it():
    """A negative lump sum (e.g. a house purchase) is grossed-up and capped
    at whatever's available, exactly like a withdrawal. Before this fix,
    the shortfall (net_received < requested) was computed by
    _apply_inflow_or_lump but discarded by the caller — the plan looked
    like it "worked" even though the investor got less than they asked
    for."""
    plan = LifePlan(
        portfolio_name="x", initial_capital=120_000.0, tax_rate_pct=0.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=3,
        phases=[Phase(kind=PhaseKind.HOLD, years=5)],
        lump_sums=[LumpSum(at_year=1.0, amount=-300_000.0, label="house")],
    )
    monthly = np.zeros((3, 60))
    result = _simulate_from_monthly_returns(plan, monthly)

    i = 11  # month 12 = at_year 1.0
    assert result.lump_outflow_requested[i] == pytest.approx(300_000.0)
    np.testing.assert_allclose(result.lump_outflow_received[:, i], 120_000.0)
    assert result.values[0, i + 1] == pytest.approx(0.0)
    assert result.summary()["probability_of_lump_sum_shortfall"] == pytest.approx(1.0)


def test_lump_outflow_within_means_has_no_shortfall():
    plan = LifePlan(
        portfolio_name="x", initial_capital=500_000.0, tax_rate_pct=0.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=3,
        phases=[Phase(kind=PhaseKind.HOLD, years=5)],
        lump_sums=[LumpSum(at_year=1.0, amount=-50_000.0, label="car")],
    )
    monthly = np.zeros((3, 60))
    result = _simulate_from_monthly_returns(plan, monthly)

    i = 11
    assert result.lump_outflow_requested[i] == pytest.approx(50_000.0)
    np.testing.assert_allclose(result.lump_outflow_received[:, i], 50_000.0)
    assert result.summary()["probability_of_lump_sum_shortfall"] == pytest.approx(0.0)


def test_lump_inflow_never_flagged_as_outflow_shortfall():
    # Positive lump sums (inflows) always succeed exactly by construction —
    # only negative ones (outflows) are tracked at all.
    plan = LifePlan(
        portfolio_name="x", initial_capital=0.0, tax_rate_pct=0.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=3,
        phases=[Phase(kind=PhaseKind.HOLD, years=5)],
        lump_sums=[LumpSum(at_year=1.0, amount=50_000.0, label="inheritance")],
    )
    monthly = np.zeros((3, 60))
    result = _simulate_from_monthly_returns(plan, monthly)
    assert np.all(result.lump_outflow_requested == 0.0)
    assert result.summary()["probability_of_lump_sum_shortfall"] == pytest.approx(0.0)


# ── 7. Inflation indexing of contributions ──────────────────────────────────

def test_inflation_indexing_matches_geometric_series():
    infl_pct = 2.0
    infl_m = (1.0 + infl_pct / 100.0) ** (1.0 / 12.0)
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=0.0,
        inflation_pct=infl_pct,
        adjust_for_inflation=True,
        phases=[Phase(PhaseKind.ACCUMULATE, years=10, monthly_amount=1_000.0)],
    )
    contribution, *_rest = _phase_schedule(plan, plan.horizon_months, infl_m)

    expected_last = 1_000.0 * infl_m ** 120
    assert contribution[-1] == pytest.approx(expected_last, rel=1e-9)
    assert expected_last == pytest.approx(1_000.0 * 1.02 ** 10, rel=1e-6)

    # closed-form geometric series: sum_{k=1}^{120} 1000 * infl_m^k
    expected_total = 1_000.0 * infl_m * (infl_m ** 120 - 1.0) / (infl_m - 1.0)
    assert contribution.sum() == pytest.approx(expected_total, rel=1e-9)


# ── 8. Real growth composes with inflation ──────────────────────────────────

def test_real_growth_composes_with_inflation():
    infl_pct = 2.0
    infl_m = (1.0 + infl_pct / 100.0) ** (1.0 / 12.0)
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=0.0,
        inflation_pct=infl_pct,
        adjust_for_inflation=True,
        phases=[Phase(PhaseKind.ACCUMULATE, years=5, monthly_amount=1_000.0, real_growth_pct=3.0)],
    )
    contribution, *_rest = _phase_schedule(plan, plan.horizon_months, infl_m)

    expected_month_24 = 1_000.0 * 1.03 ** 2 * 1.02 ** 2
    assert contribution[23] == pytest.approx(expected_month_24, rel=1e-9)


# ── 9. Real percentiles commute with deflation ──────────────────────────────

def test_percentiles_real_commutes_with_deflation():
    weights, returns = _zero_return_history(n_months=48, n_assets=2)
    rng = np.random.default_rng(0)
    returns = rng.normal(0.004, 0.03, size=(48, 2))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=100_000.0,
        inflation_pct=3.0,
        adjust_for_inflation=True,
        phases=[
            Phase(PhaseKind.ACCUMULATE, years=5, monthly_amount=500.0),
            Phase(PhaseKind.WITHDRAW, years=3, monthly_amount=800.0),
        ],
        n_sim=300,
        block_months=6,
        seed=5,
    )
    result = simulate_life_strategy(plan, weights, returns)

    got = result.percentiles((1, 10, 50, 90, 99), real=True)
    expected = np.percentile(
        result.values / result.inflation_index[None, :], (1, 10, 50, 90, 99), axis=0
    )
    np.testing.assert_allclose(got, expected, atol=1e-9)


# ── 10. Determinism ──────────────────────────────────────────────────────────

def test_determinism_same_seed_different_seed():
    rng = np.random.default_rng(42)
    weights = np.array([0.6, 0.4])
    returns = rng.normal(0.005, 0.04, size=(60, 2))

    def make_plan(seed):
        return LifePlan(
            portfolio_name="test",
            initial_capital=100_000.0,
            phases=[Phase(PhaseKind.ACCUMULATE, years=4, monthly_amount=500.0)],
            n_sim=100,
            block_months=6,
            seed=seed,
        )

    result_a = simulate_life_strategy(make_plan(123), weights, returns)
    result_b = simulate_life_strategy(make_plan(123), weights, returns)
    np.testing.assert_array_equal(result_a.values, result_b.values)

    result_c = simulate_life_strategy(make_plan(999), weights, returns)
    assert not np.array_equal(result_a.values, result_c.values)


# ── 11. Shape and invariants ─────────────────────────────────────────────────

def test_shape_and_invariants_three_phase_plan():
    rng = np.random.default_rng(7)
    weights = np.array([1.0])
    returns = rng.normal(0.004, 0.03, size=(80, 1))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=100_000.0,
        phases=[
            Phase(PhaseKind.ACCUMULATE, years=3, monthly_amount=1_000.0),
            Phase(PhaseKind.HOLD, years=2),
            Phase(PhaseKind.WITHDRAW, years=1, monthly_amount=1_500.0),
        ],
        n_sim=150,
        block_months=6,
        seed=11,
    )
    result = simulate_life_strategy(plan, weights, returns)

    h = plan.horizon_months
    assert result.values.shape == (150, h + 1)
    assert np.all(result.values >= 0.0)
    assert np.all(result.values[:, 0] == 100_000.0)
    both_nonzero = (result.planned_contributions > 0) & (result.planned_withdrawals > 0)
    assert not np.any(both_nonzero)


# ── 12. validate() catches malformed plans ──────────────────────────────────

def test_validate_empty_phases():
    plan = LifePlan(portfolio_name="p", phases=[])
    problems = plan.validate()
    assert any("at least one phase" in p for p in problems)


def test_validate_negative_phase_duration():
    plan = LifePlan(portfolio_name="p", phases=[Phase(PhaseKind.HOLD, years=-1)])
    problems = plan.validate()
    assert any("duration must be > 0" in p for p in problems)


def test_validate_horizon_too_short():
    plan = LifePlan(portfolio_name="p", phases=[Phase(PhaseKind.HOLD, years=0.1)])
    problems = plan.validate()
    assert any("at least 1 year" in p for p in problems)


def test_validate_horizon_too_long():
    plan = LifePlan(portfolio_name="p", phases=[Phase(PhaseKind.HOLD, years=90)])
    problems = plan.validate()
    assert any("80-year cap" in p for p in problems)


def test_validate_negative_initial_capital():
    plan = LifePlan(portfolio_name="p", initial_capital=-1, phases=[Phase(PhaseKind.HOLD, years=5)])
    problems = plan.validate()
    assert any("Initial capital" in p for p in problems)


def test_validate_tax_rate_out_of_range():
    plan = LifePlan(portfolio_name="p", tax_rate_pct=150, phases=[Phase(PhaseKind.HOLD, years=5)])
    problems = plan.validate()
    assert any("Tax rate" in p for p in problems)


def test_validate_lump_sum_outside_horizon():
    plan = LifePlan(
        portfolio_name="p",
        phases=[Phase(PhaseKind.HOLD, years=5)],
        lump_sums=[LumpSum(at_year=25, amount=1000)],
    )
    problems = plan.validate()
    assert any("outside the" in p and "horizon" in p for p in problems)


def test_validate_degenerate_withdraw_first_zero_capital():
    plan = LifePlan(
        portfolio_name="p",
        initial_capital=0,
        phases=[Phase(PhaseKind.WITHDRAW, years=5, monthly_amount=100)],
    )
    problems = plan.validate()
    assert any("degenerate" in p for p in problems)


def test_validate_valid_plan_has_no_hard_problems_beyond_expected():
    plan = LifePlan(
        portfolio_name="p",
        initial_capital=100_000,
        phases=[
            Phase(PhaseKind.ACCUMULATE, years=20, monthly_amount=1_000, real_growth_pct=3.0),
            Phase(PhaseKind.WITHDRAW, years=10, monthly_amount=1_500),
        ],
        lump_sums=[LumpSum(at_year=20, amount=150_000, label="inheritance")],
        block_months=12,
    )
    problems = plan.validate()
    assert problems == []


# ═══════════════════════════════════════════════════════════════════════════════
# Path-dependent withdrawal strategies: drawdown-curtailed and percentage-of-portfolio
# ═══════════════════════════════════════════════════════════════════════════════

# ── _resolve_withdrawal_target: unit tests per style ────────────────────────

def test_resolve_withdrawal_target_none_style_is_zero():
    V = np.array([1000.0, 2000.0])
    out = _resolve_withdrawal_target(0, V, V.copy(), 4000.0, 1500.0, 0.05, 0.003)
    assert np.array_equal(out, np.zeros(2))


def test_resolve_withdrawal_target_fixed_is_constant_regardless_of_v():
    V = np.array([1000.0, 999_999.0])
    running_max = np.array([1000.0, 1_500_000.0])
    out = _resolve_withdrawal_target(1, V, running_max, 4000.0, 1500.0, 0.05, 0.003)
    assert np.array_equal(out, np.array([4000.0, 4000.0]))


def test_resolve_withdrawal_target_drawdown_curtailed_above_threshold_is_full():
    # dd = (1000 - 960) / 1000 = 0.04 < 0.05 threshold -> full amount
    V = np.array([960.0])
    running_max = np.array([1000.0])
    out = _resolve_withdrawal_target(2, V, running_max, 4000.0, 1500.0, 0.05, 0.0)
    assert out[0] == pytest.approx(4000.0)


def test_resolve_withdrawal_target_drawdown_curtailed_at_threshold_is_reduced():
    # dd = (1000 - 950) / 1000 = 0.05 exactly == threshold -> cut applies (>=)
    V = np.array([950.0])
    running_max = np.array([1000.0])
    out = _resolve_withdrawal_target(2, V, running_max, 4000.0, 1500.0, 0.05, 0.0)
    assert out[0] == pytest.approx(1500.0)


def test_resolve_withdrawal_target_drawdown_curtailed_below_threshold_is_reduced():
    # dd = (1000 - 800) / 1000 = 0.20 > 0.05 -> cut applies
    V = np.array([800.0])
    running_max = np.array([1000.0])
    out = _resolve_withdrawal_target(2, V, running_max, 4000.0, 1500.0, 0.05, 0.0)
    assert out[0] == pytest.approx(1500.0)


def test_resolve_withdrawal_target_drawdown_curtailed_dead_path_is_zero_drawdown():
    # running_max == 0 (never had a positive value) -> dd reported as 0 -> full
    # amount requested (apply_withdrawal will still floor the actual sale at 0
    # since V is also 0 there; this function only decides the TARGET).
    V = np.array([0.0])
    running_max = np.array([0.0])
    out = _resolve_withdrawal_target(2, V, running_max, 4000.0, 1500.0, 0.05, 0.0)
    assert out[0] == pytest.approx(4000.0)


def test_resolve_withdrawal_target_percentage_scales_with_v():
    V = np.array([200_000.0, 100_000.0, 0.0])
    running_max = V.copy()
    out = _resolve_withdrawal_target(3, V, running_max, 0.0, 0.0, 0.0, 0.003)
    np.testing.assert_allclose(out, [600.0, 300.0, 0.0])


# ── Deterministic end-to-end drawdown-curtailment scenario ──────────────────

def test_drawdown_curtailment_switches_target_month_by_month():
    # Single sim, hand-picked monthly returns so the drawdown path is exact:
    #   m0: +0%   -> V=1,000,000, dd=0%     -> FULL (4000)
    #   m1: -10%  -> V=  896,400, dd=10.36% -> REDUCED (1500)
    #   m2: +0%   -> V=  894,900, dd=10.51% -> REDUCED (1500)
    #   m3: +20%  -> V=1,072,080, dd=0% (new peak) -> FULL (4000)
    monthly = np.array([[0.0, -0.10, 0.0, 0.20]])  # shape (n_sim=1, H=4)
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=1_000_000.0,
        tax_rate_pct=0.0,
        inflation_pct=0.0,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=4 / 12, monthly_amount=4_000.0,
            withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
            drawdown_threshold_pct=5.0, reduced_monthly_amount=1_500.0,
        )],
        n_sim=1, block_months=1, seed=1,
    )
    result = _simulate_from_monthly_returns(plan, monthly)

    expected_targets = [4000.0, 1500.0, 1500.0, 4000.0]
    np.testing.assert_allclose(result.withdrawal_requested[0], expected_targets)

    expected_v = [996_000.0, 894_900.0, 893_400.0, 1_068_080.0]
    np.testing.assert_allclose(result.values[0, 1:], expected_v, rtol=1e-9)
    # zero tax rate -> requested == realised exactly, no shortfall
    np.testing.assert_allclose(result.realised_withdrawals[0], expected_targets)


# ── H3: DRAWDOWN_CURTAILED reads a market-only signal, not the balance ──────

def test_drawdown_curtailed_never_triggers_in_a_monotonically_rising_market():
    """The trigger is a MARKET drawdown. A market that goes up every single
    month has zero market drawdown, ever — so the curtailment must never
    fire, no matter how much has been withdrawn (spending alone must not
    look like a market crash)."""
    plan = LifePlan(
        portfolio_name="x", initial_capital=1_000_000.0, tax_rate_pct=26.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=1,
        phases=[Phase(
            kind=PhaseKind.WITHDRAW, years=20, monthly_amount=4_000.0,
            withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
            reduced_monthly_amount=2_500.0, drawdown_threshold_pct=5.0,
        )],
    )
    monthly = np.full((1, 240), 0.004)   # +0.4%/month, every month, forever
    result = _simulate_from_monthly_returns(plan, monthly)
    assert np.all(result.withdrawal_requested[0] == pytest.approx(4_000.0))


def test_drawdown_curtailed_ignores_peak_inflated_by_a_prior_accumulate_phase():
    """A balance-based trigger would carry the ACCUMULATE phase's peak
    (inflated by contributions) into the WITHDRAW phase, so a flat market
    there — no market drawdown at all — could still read as a deep
    "drawdown" purely because contributions stopped. The market-only index
    resets its own reference at 1.0 when the simulation starts, so it must
    stay at dd=0 (full target) through a flat market regardless of what an
    earlier phase's balance did."""
    plan = LifePlan(
        portfolio_name="x", initial_capital=10_000.0, tax_rate_pct=0.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=1,
        phases=[
            Phase(kind=PhaseKind.ACCUMULATE, years=5, monthly_amount=2_000.0),
            Phase(
                kind=PhaseKind.WITHDRAW, years=5, monthly_amount=500.0,
                withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
                reduced_monthly_amount=100.0, drawdown_threshold_pct=5.0,
            ),
        ],
    )
    # Flat market throughout: 0% every month. The ACCUMULATE phase still
    # builds a large balance (peak) purely from contributions; the
    # WITHDRAW phase then only ever spends it down — a balance-based
    # trigger would misread that steady decline as a market drawdown.
    monthly = np.zeros((1, plan.horizon_months))
    result = _simulate_from_monthly_returns(plan, monthly)
    withdraw_targets = result.withdrawal_requested[0, 60:]   # the WITHDRAW phase's months
    assert np.all(withdraw_targets == pytest.approx(500.0))


# ── Percentage-of-portfolio: exact target, gross-up passthrough ─────────────

def test_percentage_of_portfolio_target_matches_v_before_withdrawal():
    rng = np.random.default_rng(21)
    weights = np.array([1.0])
    returns = rng.normal(0.004, 0.03, size=(60, 1))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=200_000.0,
        tax_rate_pct=0.0,  # zero tax -> requested == realised, no gross-up complexity
        inflation_pct=0.0,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=2, monthly_amount=0.0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
            withdrawal_pct_per_month=0.3,
        )],
        n_sim=40, block_months=6, seed=8,
    )
    result = simulate_life_strategy(plan, weights, returns)

    # V "before withdrawal" that month = V after (this month's withdrawal was
    # the only cash flow) + what was withdrawn that month.
    v_before = result.values[:, 1:] + result.realised_withdrawals
    expected = 0.003 * v_before
    np.testing.assert_allclose(result.withdrawal_requested, expected, rtol=1e-9)
    # zero tax -> realised == requested exactly (no exhaustion expected either,
    # since a percentage of a positive balance never fully drains it)
    np.testing.assert_allclose(result.realised_withdrawals, result.withdrawal_requested, rtol=1e-9)


# ── FIXED-style planned_withdrawals collapses to the deterministic scalar ───

def test_fixed_style_planned_withdrawals_matches_scalar_target():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=1_000_000.0,
        tax_rate_pct=26.0,
        inflation_pct=0.0,
        phases=[Phase(PhaseKind.WITHDRAW, years=1, monthly_amount=2_000.0)],
        n_sim=25, block_months=1, seed=9,
    )
    result = simulate_life_strategy(plan, weights, returns)
    # every sim sees the identical target every month (FIXED, no market variance in target)
    assert np.all(result.withdrawal_requested == 2_000.0)
    assert np.allclose(result.planned_withdrawals, 2_000.0)


# ── Regression: summary() off-by-one deflator bug ───────────────────────────

def test_summary_real_total_contributed_matches_closed_form():
    # Pure ACCUMULATE, no growth: nominal contribution at month i is
    # 1000 * infl_m**(i+1); dividing by the SAME factor (the correct
    # deflator) must leave exactly 1000 every month, so the real total is
    # exactly n_months * 1000 -- regardless of the inflation rate.
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=0.0,
        inflation_pct=5.0,
        adjust_for_inflation=True,
        phases=[Phase(PhaseKind.ACCUMULATE, years=10, monthly_amount=1_000.0)],
        n_sim=10, block_months=1, seed=3,
    )
    result = simulate_life_strategy(plan, weights, returns)
    summary_real = result.summary(real=True)
    assert summary_real["total_contributed"] == pytest.approx(120_000.0, rel=1e-9)

    summary_nominal = result.summary(real=False)
    expected_nominal = float(result.planned_contributions.sum())
    assert summary_nominal["total_contributed"] == pytest.approx(expected_nominal, rel=1e-9)


# ── shortfall_months_median: must compare each sim to its OWN target ────────

def test_shortfall_months_zero_for_percentage_style_that_never_exhausts():
    """PERCENTAGE_OF_PORTFOLIO always withdraws exactly what it asks for
    (the target itself is a fraction of current V, so it can never exceed
    V) — every simulation's realised withdrawal exactly equals its own
    requested target, so the true shortfall count is 0 for every sim.
    Comparing against the CROSS-SIMULATION MEDIAN target instead of each
    sim's own target used to flag every below-median path as a false
    shortfall."""
    plan = LifePlan(
        portfolio_name="x", initial_capital=1_000_000, tax_rate_pct=26.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=500, seed=1,
        phases=[Phase(kind=PhaseKind.WITHDRAW, years=10,
                      withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
                      withdrawal_pct_per_month=0.3)],
    )
    rng = np.random.default_rng(1)
    monthly = rng.normal(0.005, 0.04, size=(plan.n_sim, plan.horizon_months))
    result = _simulate_from_monthly_returns(plan, monthly)

    # Ground truth: shortfall only exists where a sim received less than
    # ITS OWN request (which, for this style, never happens).
    truth = np.median(np.sum(
        (result.withdrawal_requested > 0)
        & (result.realised_withdrawals < result.withdrawal_requested - 1e-9),
        axis=1,
    ))
    assert truth == 0.0
    assert result.summary()["shortfall_months_median"] == pytest.approx(0.0)


def test_shortfall_months_matches_per_simulation_ground_truth_fixed_style():
    """For a FIXED-style withdrawal every sim's target equals the plan-wide
    median exactly, so this must be unaffected by the H2 fix."""
    plan = LifePlan(
        portfolio_name="x", initial_capital=50_000, tax_rate_pct=26.0,
        inflation_pct=0.0, adjust_for_inflation=False, n_sim=200, seed=2,
        phases=[Phase(kind=PhaseKind.WITHDRAW, years=10, monthly_amount=2000)],
    )
    rng = np.random.default_rng(2)
    monthly = rng.normal(-0.01, 0.05, size=(plan.n_sim, plan.horizon_months))
    result = _simulate_from_monthly_returns(plan, monthly)

    truth = np.median(np.sum(
        (result.withdrawal_requested > 0)
        & (result.realised_withdrawals < result.withdrawal_requested - 1e-9),
        axis=1,
    ))
    assert result.summary()["shortfall_months_median"] == pytest.approx(truth)


# ── cash_flow_at / cash_flow_bands consistency ───────────────────────────────

def test_cash_flow_at_cumulative_matches_manual_sum():
    weights, returns = _zero_return_history(n_months=48, n_assets=1)
    rng = np.random.default_rng(4)
    returns = rng.normal(0.003, 0.02, size=(48, 1))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=100_000.0,
        inflation_pct=2.0,
        phases=[
            Phase(PhaseKind.ACCUMULATE, years=2, monthly_amount=500.0),
            Phase(PhaseKind.WITHDRAW, years=1, monthly_amount=800.0),
        ],
        n_sim=60, block_months=6, seed=6,
    )
    result = simulate_life_strategy(plan, weights, returns)
    H = result.planned_contributions.shape[0]

    snapshot = result.cash_flow_at(H)
    assert snapshot["cumulative_contributed_nominal"] == pytest.approx(
        float(result.planned_contributions.sum()), rel=1e-9
    )
    median_withdrawn = float(np.median(result.realised_withdrawals.sum(axis=1)))
    assert snapshot["cumulative_withdrawn_nominal"] == pytest.approx(median_withdrawn, rel=1e-9)

    # mid-plan snapshot: per-month values match the raw arrays at that index
    mid = H // 2
    snap_mid = result.cash_flow_at(mid)
    assert snap_mid["contribution_nominal"] == pytest.approx(
        float(result.planned_contributions[mid - 1]), rel=1e-9
    )


def test_cash_flow_bands_shapes_and_median_matches_at():
    weights, returns = _zero_return_history(n_months=36, n_assets=1)
    rng = np.random.default_rng(5)
    returns = rng.normal(0.003, 0.02, size=(36, 1))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=50_000.0,
        inflation_pct=2.0,
        phases=[Phase(PhaseKind.WITHDRAW, years=2, monthly_amount=500.0)],
        n_sim=80, block_months=6, seed=7,
    )
    result = simulate_life_strategy(plan, weights, returns)
    bands = result.cash_flow_bands((10, 50, 90))
    H = result.planned_contributions.shape[0]

    for key in ("contribution_nominal", "withdrawal_requested_nominal",
                "withdrawal_realised_nominal", "tax_nominal"):
        assert bands[key].shape == (3, H)

    # P50 band of withdrawal_realised should equal the per-month median
    at_month_1 = result.cash_flow_at(1)
    assert bands["withdrawal_realised_nominal"][1, 0] == pytest.approx(
        at_month_1["withdrawal_realised_nominal"], rel=1e-9
    )


def test_cash_flow_at_zero_month_is_all_zero():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        phases=[Phase(PhaseKind.ACCUMULATE, years=1, monthly_amount=1_000.0)],
        n_sim=10, block_months=1, seed=2,
    )
    result = simulate_life_strategy(plan, weights, returns)
    snap = result.cash_flow_at(0)
    assert snap["inflation_factor"] == 1.0
    assert snap["contribution_nominal"] == 0.0
    assert snap["cumulative_contributed_nominal"] == 0.0


# ── validate(): new withdrawal-style rules ───────────────────────────────────

def test_validate_drawdown_curtailed_reduced_exceeds_full():
    plan = LifePlan(
        portfolio_name="p",
        phases=[Phase(
            PhaseKind.WITHDRAW, years=5, monthly_amount=1000,
            withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
            reduced_monthly_amount=2000,  # > full amount
        )],
    )
    problems = plan.validate()
    assert any("reduced amount" in p for p in problems)


def test_validate_drawdown_curtailed_bad_threshold():
    plan = LifePlan(
        portfolio_name="p",
        phases=[Phase(
            PhaseKind.WITHDRAW, years=5, monthly_amount=1000,
            withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
            reduced_monthly_amount=500, drawdown_threshold_pct=0.0,
        )],
    )
    problems = plan.validate()
    assert any("drawdown trigger" in p for p in problems)


def test_validate_percentage_pct_out_of_range():
    plan = LifePlan(
        portfolio_name="p",
        phases=[Phase(
            PhaseKind.WITHDRAW, years=5, monthly_amount=0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
            withdrawal_pct_per_month=0.0,
        )],
    )
    problems = plan.validate()
    assert any("%/month" in p for p in problems)


def test_validate_valid_drawdown_curtailed_plan_has_no_problems():
    plan = LifePlan(
        portfolio_name="p",
        initial_capital=100_000,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=10, monthly_amount=4000,
            withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
            reduced_monthly_amount=1500, drawdown_threshold_pct=5.0,
        )],
        block_months=12,
    )
    assert plan.validate() == []


def test_validate_valid_percentage_plan_has_no_problems():
    plan = LifePlan(
        portfolio_name="p",
        initial_capital=100_000,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=10, monthly_amount=0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
            withdrawal_pct_per_month=0.3,
        )],
        block_months=12,
    )
    assert plan.validate() == []


# ── cash_flow_percentiles_at ─────────────────────────────────────────────────

def test_cash_flow_percentiles_at_p50_matches_cash_flow_at_median():
    rng = np.random.default_rng(30)
    weights = np.array([1.0])
    returns = rng.normal(0.004, 0.03, size=(40, 1))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=200_000.0,
        inflation_pct=2.0,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=2, monthly_amount=4000,
            withdrawal_style=WithdrawalStyle.DRAWDOWN_CURTAILED,
            reduced_monthly_amount=1500, drawdown_threshold_pct=5.0,
        )],
        n_sim=50, block_months=6, seed=15,
    )
    result = simulate_life_strategy(plan, weights, returns)
    month = 10

    at = result.cash_flow_at(month)
    spread = result.cash_flow_percentiles_at(month, (10, 50, 90))

    p50_idx = list(spread["pcts"]).index(50)
    assert spread["withdrawal_requested_nominal"][p50_idx] == pytest.approx(
        at["withdrawal_requested_nominal"], rel=1e-9
    )
    assert spread["withdrawal_realised_nominal"][p50_idx] == pytest.approx(
        at["withdrawal_realised_nominal"], rel=1e-9
    )
    # a path-dependent style should show real spread (P90 >= P10) at some month
    assert spread["withdrawal_requested_nominal"][-1] >= spread["withdrawal_requested_nominal"][0]


def test_cash_flow_percentiles_at_fixed_style_degenerate_spread():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=1_000_000.0,
        phases=[Phase(PhaseKind.WITHDRAW, years=1, monthly_amount=2_000.0)],
        n_sim=20, block_months=1, seed=16,
    )
    result = simulate_life_strategy(plan, weights, returns)
    spread = result.cash_flow_percentiles_at(5, (10, 50, 90))
    # FIXED style: every simulation asks for the identical amount -> flat spread
    assert np.allclose(spread["withdrawal_requested_nominal"], spread["withdrawal_requested_nominal"][0])


def test_cash_flow_percentiles_at_zero_month_is_zero():
    weights, returns = _zero_return_history()
    plan = LifePlan(
        portfolio_name="test",
        phases=[Phase(PhaseKind.ACCUMULATE, years=1, monthly_amount=1_000.0)],
        n_sim=10, block_months=1, seed=17,
    )
    result = simulate_life_strategy(plan, weights, returns)
    spread = result.cash_flow_percentiles_at(0)
    assert np.all(spread["withdrawal_requested_nominal"] == 0.0)


# ── withdrawal_tax must not double-count lump-sum tax ────────────────────────

def test_withdrawal_tax_excludes_coincident_lump_sum_tax():
    """A lump-sum OUTFLOW landing the same month as a phase withdrawal must
    not inflate withdrawal_tax — that field is specifically the tax on the
    withdrawal, so `taxes_paid` (the plan-wide total) can exceed it, but
    `withdrawal_tax` itself must be identical to a plan with no lump sum at
    all, for every month the lump sum doesn't touch."""
    weights = np.array([1.0])
    returns = np.array([[0.01]])  # single month of history -> deterministic path
    common = dict(
        portfolio_name="test", initial_capital=200_000.0, tax_rate_pct=26.0, inflation_pct=0.0,
        phases=[Phase(PhaseKind.WITHDRAW, years=3 / 12, monthly_amount=1_000.0)],
        n_sim=5, block_months=1, seed=1,
    )
    plan_with_lump = LifePlan(**common, lump_sums=[LumpSum(at_year=2 / 12, amount=-5_000.0, label="expense")])
    plan_no_lump = LifePlan(**common)

    result_with_lump = simulate_life_strategy(plan_with_lump, weights, returns)
    result_no_lump = simulate_life_strategy(plan_no_lump, weights, returns)

    # Lump sum lands at elapsed_target=2 -> loop index i=1. Month 0 is untouched
    # by it in either plan, so everything about the withdrawal that month must
    # match exactly.
    np.testing.assert_allclose(
        result_with_lump.withdrawal_tax[:, 0], result_no_lump.withdrawal_tax[:, 0]
    )
    np.testing.assert_allclose(
        result_with_lump.realised_withdrawals[:, 0], result_no_lump.realised_withdrawals[:, 0]
    )

    # Month 1 (where the lump sum DOES land): the plan-wide total tax must be
    # strictly greater than the withdrawal-specific tax (the lump sum's own
    # tax is folded into taxes_paid but must not leak into withdrawal_tax).
    assert np.all(result_with_lump.taxes_paid[:, 1] > result_with_lump.withdrawal_tax[:, 1])
    assert np.all(result_with_lump.withdrawal_tax[:, 1] >= 0)


def test_taxes_paid_equals_withdrawal_tax_when_no_lump_sums():
    """With no lump sums, every taxable event in a month IS the withdrawal —
    so the plan-wide total (`taxes_paid`) must equal the withdrawal-specific
    figure (`withdrawal_tax`) exactly, every month. This is what makes
    `taxes_paid` a strict superset once lump sums are introduced (see
    ``test_withdrawal_tax_excludes_coincident_lump_sum_tax``), not a
    separately-computed, possibly-inconsistent number."""
    rng = np.random.default_rng(42)
    weights = np.array([0.5, 0.5])
    returns = rng.normal(0.006, 0.035, size=(60, 2))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=150_000.0,
        tax_rate_pct=26.0,
        inflation_pct=2.0,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=3,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
            withdrawal_pct_per_month=0.5,
        )],
        n_sim=80, block_months=6, seed=8,
    )
    result = simulate_life_strategy(plan, weights, returns)
    np.testing.assert_array_equal(result.taxes_paid, result.withdrawal_tax)


def test_cash_flow_percentiles_at_gross_matches_combined_array_not_separate_sums():
    """withdrawal_gross percentiles must come from percentile(net + tax) per
    simulation, NOT percentile(net) + percentile(tax) — those differ for
    path-dependent styles because percentiles don't distribute over sums."""
    rng = np.random.default_rng(11)
    weights = np.array([1.0])
    returns = rng.normal(0.005, 0.04, size=(50, 1))
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=300_000.0,
        tax_rate_pct=26.0,
        inflation_pct=2.0,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=4,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO,
            withdrawal_pct_per_month=0.4,
        )],
        n_sim=200, block_months=6, seed=21,
    )
    result = simulate_life_strategy(plan, weights, returns)
    month = 20
    i = month - 1
    pcts = (10, 50, 90)

    spread = result.cash_flow_percentiles_at(month, pcts)
    expected_gross = np.percentile(
        result.realised_withdrawals[:, i] + result.withdrawal_tax[:, i], pcts
    )
    np.testing.assert_allclose(spread["withdrawal_gross_nominal"], expected_gross)


# ── PERCENTAGE_RAMP withdrawal style ─────────────────────────────────────────

def test_percentage_ramp_schedule_is_a_straight_line():
    """The per-month rate must be exactly linspace(start, end, n_months) —
    not the withdrawal amount (which also depends on V), just the rate."""
    plan = LifePlan(
        portfolio_name="test",
        phases=[Phase(
            PhaseKind.WITHDRAW, years=1, monthly_amount=0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_RAMP,
            withdrawal_pct_per_month=5.0, withdrawal_pct_end_per_month=45.0,
        )],
    )
    _, _, _, _, _, withdrawal_pct, _ = _phase_schedule(plan, plan.horizon_months, infl_m=1.0)
    expected = np.linspace(0.05, 0.45, 12)
    np.testing.assert_allclose(withdrawal_pct, expected)
    assert withdrawal_pct[0] == pytest.approx(0.05)
    assert withdrawal_pct[-1] == pytest.approx(0.45)


def test_percentage_ramp_matches_closed_form_compounding():
    """Zero-return history + zero inflation -> u stays 0 forever (V and B
    shrink in lockstep), so the whole path is a pure closed-form product:
    V_m = V_0 * prod_{k<m} (1 - rate_k), no tax, no gross-up needed."""
    weights, returns = _zero_return_history()
    v0 = 1_000_000.0
    start_pct, end_pct = 1.0, 3.0
    n_months = 24
    plan = LifePlan(
        portfolio_name="test",
        initial_capital=v0,
        inflation_pct=0.0,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=n_months / 12, monthly_amount=0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_RAMP,
            withdrawal_pct_per_month=start_pct, withdrawal_pct_end_per_month=end_pct,
        )],
        n_sim=10, block_months=1, seed=19,
    )
    result = simulate_life_strategy(plan, weights, returns)

    rates = np.linspace(start_pct / 100.0, end_pct / 100.0, n_months)
    expected_values = np.empty(n_months + 1)
    expected_values[0] = v0
    for m in range(n_months):
        expected_values[m + 1] = expected_values[m] * (1.0 - rates[m])

    for sim_row in result.values:
        np.testing.assert_allclose(sim_row, expected_values, rtol=1e-9)
    assert np.all(result.taxes_paid == 0.0)  # u stays 0 throughout -> no gain, no tax


def test_percentage_ramp_with_equal_start_end_matches_flat_percentage():
    """A ramp from X% to X% must behave identically to the constant-rate style."""
    rng = np.random.default_rng(5)
    weights = np.array([1.0])
    returns = rng.normal(0.005, 0.03, size=(48, 1))
    common = dict(
        portfolio_name="test", initial_capital=250_000.0, tax_rate_pct=26.0, inflation_pct=2.0,
        n_sim=60, block_months=6, seed=31,
    )
    plan_flat = LifePlan(**common, phases=[Phase(
        PhaseKind.WITHDRAW, years=3, monthly_amount=0,
        withdrawal_style=WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO, withdrawal_pct_per_month=0.4,
    )])
    plan_ramp = LifePlan(**common, phases=[Phase(
        PhaseKind.WITHDRAW, years=3, monthly_amount=0,
        withdrawal_style=WithdrawalStyle.PERCENTAGE_RAMP,
        withdrawal_pct_per_month=0.4, withdrawal_pct_end_per_month=0.4,
    )])
    result_flat = simulate_life_strategy(plan_flat, weights, returns)
    result_ramp = simulate_life_strategy(plan_ramp, weights, returns)
    np.testing.assert_array_equal(result_flat.values, result_ramp.values)


def test_validate_percentage_ramp_out_of_range():
    plan = LifePlan(
        portfolio_name="p",
        phases=[Phase(
            PhaseKind.WITHDRAW, years=5, monthly_amount=0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_RAMP,
            withdrawal_pct_per_month=0.0, withdrawal_pct_end_per_month=25.0,
        )],
    )
    problems = plan.validate()
    assert any("starting withdrawal" in p for p in problems)
    assert any("ending withdrawal" in p for p in problems)


def test_validate_percentage_ramp_valid_plan_has_no_problems():
    plan = LifePlan(
        portfolio_name="p",
        initial_capital=200_000,
        phases=[Phase(
            PhaseKind.WITHDRAW, years=20, monthly_amount=0,
            withdrawal_style=WithdrawalStyle.PERCENTAGE_RAMP,
            withdrawal_pct_per_month=0.05, withdrawal_pct_end_per_month=0.45,
        )],
        block_months=12,
    )
    assert plan.validate() == []
