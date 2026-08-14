"""
engine.lifecycle — Life Strategy Simulator.

Simulates a full life plan on a single portfolio: an initial lump-sum
investment, a sequence of phases (accumulate / hold / withdraw), and
optional one-off cash flows (lump sums), under a Monte-Carlo block
bootstrap of the portfolio's historical monthly returns.

Design notes (also surfaced in the GUI):

- All amounts are entered by the caller in TODAY's euros. Internally the
  simulator inflates every cash flow to nominal terms (see
  ``LifePlan.inflation_pct``) and only deflates back to today's euros for
  reporting (:meth:`LifeSimResult.percentiles`, :meth:`LifeSimResult.summary`,
  :meth:`LifeSimResult.cash_flow_bands`, :meth:`LifeSimResult.cash_flow_at`).
  The historical return series itself is used as recorded (nominal) —
  only cash flows are indexed to expected inflation, not the return
  distribution. This is the standard convention, not an oversight.
- Capital-gains tax uses the average-cost method: each simulated path
  tracks a market value ``V`` and a cost basis ``B`` (capital paid in and
  not yet sold). A withdrawal is grossed-up so the investor receives
  exactly the requested NET amount after tax on the realised gain
  fraction. See :func:`apply_withdrawal`.
- A WITHDRAW phase's NET monthly target can be computed three ways
  (:class:`WithdrawalStyle`): a FIXED today's-€ amount, an amount that is
  CURTAILED to a lower floor whenever the path is in a deep-enough
  drawdown from its own historical high, or a PERCENTAGE of the current
  portfolio value. The first is deterministic (identical across every
  simulation); the other two are *path-dependent* — the target itself
  differs simulation to simulation. See :func:`_resolve_withdrawal_target`.
- Loss carry-forward against future gains is NOT modelled — a documented
  simplification (``LOSS_CARRYFORWARD_SUPPORTED = False``).
- Within a simulated month: (1) the market moves, (2) one-off lump sums
  are applied, (3) the phase's contribution/withdrawal is applied,
  (4) the end-of-month state is recorded. All four steps are vectorised
  over the ``n_sim`` simulations; only the month loop itself is a Python
  loop (bounded by the horizon, typically well under 1000 iterations).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from engine.simulation import sample_monthly_returns

log = logging.getLogger("bootstrap.lifecycle")

LOSS_CARRYFORWARD_SUPPORTED = False


# ═══════════════════════════════════════════════════════════════════════════════
# Domain model
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseKind(str, Enum):
    ACCUMULATE = "accumulate"   # monthly contribution into the portfolio
    HOLD = "hold"                # invested, no cash flow
    WITHDRAW = "withdraw"        # monthly NET (after-tax) income taken out


class WithdrawalStyle(str, Enum):
    """How a WITHDRAW phase's net monthly target is computed. Ignored for
    ACCUMULATE / HOLD phases."""
    FIXED = "fixed"
    # Withdraw `monthly_amount` normally; cut to `reduced_monthly_amount`
    # in any month where the path's drawdown from its own historical peak
    # is at or beyond `drawdown_threshold_pct`. Re-evaluated every month.
    DRAWDOWN_CURTAILED = "drawdown_curtailed"
    # Withdraw `withdrawal_pct_per_month` % of the CURRENT portfolio value,
    # net (post-tax) — the target itself scales with the portfolio.
    PERCENTAGE_OF_PORTFOLIO = "percentage"
    # Same mechanic as PERCENTAGE_OF_PORTFOLIO, but the rate itself moves in
    # a straight line from `withdrawal_pct_per_month` (at the first month of
    # the phase) to `withdrawal_pct_end_per_month` (at the last), e.g. "0.05%
    # of the portfolio in month 1, ramping up to 0.45% by month 240" for a
    # 20-year phase — a common way to model a spend rate that rises (or
    # falls) deliberately over retirement rather than staying flat.
    PERCENTAGE_RAMP = "percentage_ramp"


# Internal integer encoding used by the vectorised month loop —
# 0 means "no withdrawal this month" (ACCUMULATE / HOLD months). RAMP shares
# id 3 with PERCENTAGE_OF_PORTFOLIO: by the time the month loop runs, both
# have already been reduced to "this month's rate is X%" in `_phase_schedule`
# (constant for one, linearly interpolated for the other) — `_resolve_
# withdrawal_target` only ever sees that resolved per-month rate, so there is
# nothing left to distinguish between them at that point.
_WITHDRAWAL_STYLE_ID: dict[WithdrawalStyle, int] = {
    WithdrawalStyle.FIXED: 1,
    WithdrawalStyle.DRAWDOWN_CURTAILED: 2,
    WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO: 3,
    WithdrawalStyle.PERCENTAGE_RAMP: 3,
}


@dataclass(frozen=True)
class Phase:
    """One contiguous segment of the life plan.

    All monetary amounts are expressed in TODAY's euros; the simulator
    inflates them to nominal terms internally (see `LifePlan.inflation_pct`).

    ``withdrawal_style`` and its style-specific fields are only interpreted
    when ``kind == PhaseKind.WITHDRAW``:

    - FIXED (default): only ``monthly_amount`` / ``real_growth_pct`` matter,
      exactly as before this field existed.
    - DRAWDOWN_CURTAILED: ``monthly_amount`` is the FULL target, cut to
      ``reduced_monthly_amount`` whenever the path's drawdown from its own
      running peak is >= ``drawdown_threshold_pct``. Both amounts grow
      together under ``real_growth_pct``.
    - PERCENTAGE_OF_PORTFOLIO: ``monthly_amount`` / ``reduced_monthly_amount``
      / ``drawdown_threshold_pct`` are ignored; the net target is
      ``withdrawal_pct_per_month`` % of that simulation's CURRENT portfolio
      value each month.
    - PERCENTAGE_RAMP: same as PERCENTAGE_OF_PORTFOLIO, except the rate
      itself moves in a straight line across the phase, from
      ``withdrawal_pct_per_month`` (first month) to
      ``withdrawal_pct_end_per_month`` (last month). Either can be the
      larger one — a ramp DOWN is just as valid as a ramp up.
    """
    kind: PhaseKind
    years: float                       # duration, > 0, may be fractional
    monthly_amount: float = 0.0        # today's €/month. 0 for HOLD.
    real_growth_pct: float = 0.0       # annual REAL growth of monthly_amount
                                        #   (e.g. 3.0 = career progression on top of inflation)
    label: str = ""                    # free-text, shown in the UI list
    withdrawal_style: WithdrawalStyle = WithdrawalStyle.FIXED
    drawdown_threshold_pct: float = 5.0     # trigger: drawdown >= this % (e.g. 5.0 = -5%)
    reduced_monthly_amount: float = 0.0     # today's €, DRAWDOWN_CURTAILED floor
    withdrawal_pct_per_month: float = 0.3   # % of current portfolio — PERCENTAGE_OF_PORTFOLIO's
                                             #   rate, or PERCENTAGE_RAMP's rate at the phase's START
    withdrawal_pct_end_per_month: float = 0.3   # % of current portfolio at the phase's END,
                                                 #   PERCENTAGE_RAMP only


@dataclass(frozen=True)
class LumpSum:
    """A one-off cash flow at a given time from the start of the plan."""
    at_year: float                     # >= 0, may be fractional (0.5 = month 6)
    amount: float                      # today's €. POSITIVE = inflow (inheritance),
                                        #   NEGATIVE = outflow (house purchase), taxed like a sale
    label: str = ""


@dataclass
class LifePlan:
    portfolio_name: str
    initial_capital: float = 100_000.0     # today's €, invested at t=0
    tax_rate_pct: float = 26.0             # capital-gains tax, on REALISED gains only
    inflation_pct: float = 2.0             # annual; ignored when adjust_for_inflation is False
    adjust_for_inflation: bool = True
    phases: list[Phase] = field(default_factory=list)
    lump_sums: list[LumpSum] = field(default_factory=list)

    # simulation knobs
    n_sim: int = 5_000
    block_months: int = 12                 # bootstrap block length
    seed: Optional[int] = 42
    date_start: Optional[str] = None       # "YYYY-MM" filter on the historical window
    date_end: Optional[str] = None

    @property
    def horizon_years(self) -> float:
        return sum(p.years for p in self.phases)

    @property
    def horizon_months(self) -> int:
        return int(round(self.horizon_years * 12))

    @property
    def effective_inflation_pct(self) -> float:
        return self.inflation_pct if self.adjust_for_inflation else 0.0

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list = plan is valid.

        Distinguishes hard errors (plan cannot be simulated) from soft
        warnings (plan can run, but the result may not mean what the user
        expects) only by wording — callers that need to block a Run should
        treat every non-empty return as "fix these first"; the GUI may
        choose to only hard-block on the errors it can identify by prefix
        (this function itself does not draw that line, since what should
        block a Run is a UI policy decision, not an engine one).
        """
        problems: list[str] = []

        if not self.phases:
            problems.append("The plan needs at least one phase.")
        for i, p in enumerate(self.phases):
            if p.years <= 0:
                problems.append(f"Phase {i + 1} ({p.kind.value}): duration must be > 0 years.")
            if p.monthly_amount < 0:
                problems.append(
                    f"Phase {i + 1} ({p.kind.value}): monthly amount must be >= 0 "
                    f"(the sign is determined by the phase type, not the number)."
                )
            if p.kind == PhaseKind.WITHDRAW:
                if p.withdrawal_style == WithdrawalStyle.DRAWDOWN_CURTAILED:
                    if p.reduced_monthly_amount < 0:
                        problems.append(f"Phase {i + 1}: reduced amount must be >= 0.")
                    elif p.reduced_monthly_amount > p.monthly_amount:
                        problems.append(
                            f"Phase {i + 1}: reduced amount must be <= the full amount "
                            f"(it's a floor for bad markets, not a raise)."
                        )
                    if not (0.0 < p.drawdown_threshold_pct < 100.0):
                        problems.append(f"Phase {i + 1}: drawdown trigger must be between 0 and 100%.")
                elif p.withdrawal_style == WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO:
                    if not (0.0 < p.withdrawal_pct_per_month <= 20.0):
                        problems.append(f"Phase {i + 1}: withdrawal %/month must be between 0 and 20%.")
                elif p.withdrawal_style == WithdrawalStyle.PERCENTAGE_RAMP:
                    if not (0.0 < p.withdrawal_pct_per_month <= 20.0):
                        problems.append(
                            f"Phase {i + 1}: starting withdrawal %/month must be between 0 and 20%."
                        )
                    if not (0.0 < p.withdrawal_pct_end_per_month <= 20.0):
                        problems.append(
                            f"Phase {i + 1}: ending withdrawal %/month must be between 0 and 20%."
                        )

        horizon_years = self.horizon_years
        if self.phases and horizon_years < 1:
            problems.append(f"Total horizon ({horizon_years:.2f}y) must be at least 1 year.")
        if horizon_years > 80:
            problems.append(f"Total horizon ({horizon_years:.2f}y) exceeds the 80-year cap.")

        if self.initial_capital < 0:
            problems.append("Initial capital must be >= 0.")
        if not (0.0 <= self.tax_rate_pct < 100.0):
            problems.append("Tax rate must be in [0, 100).")
        if not (-5.0 <= self.inflation_pct <= 20.0):
            problems.append("Inflation must be in [-5, 20] %/year.")
        if not (100 <= self.n_sim <= 100_000):
            problems.append("n_sim must be between 100 and 100,000.")
        if not (1 <= self.block_months <= 60):
            problems.append("block_months must be between 1 and 60.")

        for i, ls in enumerate(self.lump_sums):
            if ls.at_year < 0:
                problems.append(f"Lump sum {i + 1} ({ls.label or '?'}): at_year must be >= 0.")
            elif self.phases and ls.at_year > horizon_years + 1e-9:
                problems.append(
                    f"Lump sum {i + 1} ({ls.label or '?'}) at year {ls.at_year:g} falls "
                    f"outside the {horizon_years:g}-year horizon and will be ignored."
                )

        if self.phases and self.horizon_months > 0 and self.horizon_months % self.block_months != 0:
            problems.append(
                f"block_months={self.block_months} does not evenly divide the "
                f"{self.horizon_months}-month horizon; the final block will be truncated."
            )

        if self.phases and self.phases[0].kind == PhaseKind.WITHDRAW and self.initial_capital <= 0:
            problems.append(
                "The plan starts in a Withdraw phase with zero initial capital — "
                "this is a degenerate plan (nothing to withdraw from at the start)."
            )

        return problems


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised capital-gains-aware withdrawal (average-cost method)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_withdrawal(
    V: np.ndarray, B: np.ndarray, w_net, tax_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sell just enough to hand the investor exactly *w_net* after tax.

    Average-cost method: latent-gain fraction ``u = clip(1 - B/V, 0, 1)``.
    Gross-up: ``g = w_net / (1 - tax_rate * u)``, capped at the available
    value ``V`` (exhaustion — the investor receives less than requested).
    The basis reduction is pro-rata (``B -= g * B/V``) and is computed
    against the PRE-withdrawal ``V``.

    Net received always equals ``g - tax``, which collapses the "normal"
    and "exhausted" cases into one formula: when ``g < V`` this equals
    ``w_net`` exactly by construction of the gross-up; when capped at
    ``V`` it equals ``V * (1 - tax_rate * u)`` — the correct shortfall.

    Parameters
    ----------
    V, B : (n,) arrays — market value / cost basis, updated in place logically
        (new arrays are returned; inputs are not mutated).
    w_net : scalar or (n,) array — requested NET withdrawal amount, >= 0.
    tax_rate : fraction, e.g. 0.26.

    Returns
    -------
    V_new, B_new, tax_paid, net_received : (n,) arrays.
    """
    n = V.shape[0]
    w_net_arr = np.broadcast_to(np.asarray(w_net, dtype=np.float64), (n,))

    alive = V > 0
    V_safe = np.where(alive, V, 1.0)
    u = np.where(alive, np.clip(1.0 - B / V_safe, 0.0, 1.0), 0.0)

    denom = 1.0 - tax_rate * u
    g = np.where(alive, w_net_arr / denom, 0.0)
    g = np.clip(g, 0.0, V)  # cap at available value; also zeroes out dead sims (V<=0)

    tax_paid = tax_rate * g * u
    net_received = g - tax_paid

    B_frac = np.where(alive, B / V_safe, 0.0)
    B_new = np.maximum(B - g * B_frac, 0.0)
    V_new = np.maximum(V - g, 0.0)

    return V_new, B_new, tax_paid, net_received


def _apply_inflow_or_lump(V: np.ndarray, B: np.ndarray, amount: float,
                           tax_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a single scalar cash flow (contribution or lump sum) to every sim.

    Positive: a contribution (or inflow lump sum). Dead paths (V<=0) are
    reborn with a fresh cost basis; alive paths just add to V and B.
    Negative: an outflow, taxed like a sale via :func:`apply_withdrawal`.
    """
    if amount >= 0:
        dead = V <= 0
        V_new = np.where(dead, amount, V + amount)
        B_new = np.where(dead, amount, B + amount)
        return V_new, B_new, np.zeros_like(V), np.full_like(V, amount)
    V_new, B_new, tax_paid, net_received = apply_withdrawal(V, B, -amount, tax_rate)
    return V_new, B_new, tax_paid, net_received


def _resolve_withdrawal_target(
    style_id: int,
    V: np.ndarray,
    running_max: np.ndarray,
    full_nominal: float,
    reduced_nominal: float,
    threshold_frac: float,
    pct_frac: float,
) -> np.ndarray:
    """Vectorised net withdrawal TARGET per simulation for one month —
    i.e. what :func:`apply_withdrawal` should be asked for, before tax
    gross-up / exhaustion are applied there.

    ``style_id`` is a single int (0=none, 1=fixed, 2=drawdown-curtailed,
    3=percentage): the style is a property of the *phase*, so within one
    month it is the same for every simulation — only the resulting target
    (for styles 2 and 3) varies simulation to simulation.

    Drawdown convention: ``dd = (running_max - V) / running_max`` — the
    same peak-to-trough definition used by
    :func:`engine.metrics._drawdown_series`. ``running_max`` is 0 only if
    the path has never had a positive value; that path is dead, dd is
    reported as 0, and the target collapses to 0 anyway once it goes
    through :func:`apply_withdrawal`. The cut is binary: the FULL amount
    applies while ``dd < threshold_frac``, the REDUCED floor applies from
    ``dd >= threshold_frac`` (the threshold month itself gets the cut).
    """
    n = V.shape[0]
    if style_id == 0:
        return np.zeros(n, dtype=np.float64)
    if style_id == 1:
        return np.full(n, float(full_nominal), dtype=np.float64)
    if style_id == 2:
        safe_peak = np.where(running_max > 0, running_max, 1.0)
        dd = np.where(running_max > 0, (running_max - V) / safe_peak, 0.0)
        return np.where(dd >= threshold_frac, float(reduced_nominal), float(full_nominal))
    if style_id == 3:
        return np.maximum(V, 0.0) * float(pct_frac)
    raise ValueError(f"Unknown withdrawal style id: {style_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic monthly schedule (contributions / withdrawals / lump sums)
# ═══════════════════════════════════════════════════════════════════════════════

def _inflation_index(horizon_months: int, inflation_pct: float) -> np.ndarray:
    """``(horizon_months + 1,)`` array; index[0] = 1.0 (t0), index[k] = infl_m**k."""
    infl_m = (1.0 + inflation_pct / 100.0) ** (1.0 / 12.0)
    return infl_m ** np.arange(horizon_months + 1, dtype=np.float64)


def _phase_schedule(
    plan: LifePlan, horizon_months: int, infl_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the deterministic (n_months,) schedules describing what SHOULD
    happen each month, before any simulated market outcome is known.

    Uses cumulative rounding of phase boundaries so the total always sums
    exactly to *horizon_months*, even if individual phase durations don't
    round to whole months cleanly.

    Returns
    -------
    contribution         : nominal €, ACCUMULATE phases (same for every sim)
    withdrawal_style      : int8, 0=none 1=fixed 2=drawdown_curtailed 3=percentage
                             (PERCENTAGE_RAMP also encodes as 3 — see _WITHDRAWAL_STYLE_ID)
    withdrawal_full       : nominal € — the FIXED target, or the DRAWDOWN_CURTAILED "full" ceiling
    withdrawal_reduced    : nominal € — the DRAWDOWN_CURTAILED floor
    withdrawal_threshold  : fraction (e.g. 0.05) — DRAWDOWN_CURTAILED trigger
    withdrawal_pct        : fraction (e.g. 0.003) — PERCENTAGE_OF_PORTFOLIO's constant rate, or
                             PERCENTAGE_RAMP's rate for THAT month (linearly interpolated across
                             the phase — this is what lets the month loop / _resolve_withdrawal_target
                             treat both styles identically, see _WITHDRAWAL_STYLE_ID)
    phase_id              : index into plan.phases, -1 outside any phase
    """
    contribution = np.zeros(horizon_months, dtype=np.float64)
    withdrawal_style = np.zeros(horizon_months, dtype=np.int8)
    withdrawal_full = np.zeros(horizon_months, dtype=np.float64)
    withdrawal_reduced = np.zeros(horizon_months, dtype=np.float64)
    withdrawal_threshold = np.zeros(horizon_months, dtype=np.float64)
    withdrawal_pct = np.zeros(horizon_months, dtype=np.float64)
    phase_id = np.full(horizon_months, -1, dtype=np.int64)

    years_cum = np.cumsum([p.years for p in plan.phases])
    end_i = np.round(years_cum * 12).astype(np.int64)
    end_i = np.clip(end_i, 0, horizon_months)
    start_i = np.concatenate(([0], end_i[:-1]))

    for p, phase in enumerate(plan.phases):
        lo, hi = int(start_i[p]), int(end_i[p])
        if hi <= lo:
            continue
        phase_id[lo:hi] = p
        if phase.kind == PhaseKind.HOLD:
            continue

        i_range = np.arange(lo, hi, dtype=np.float64)
        elapsed_months = i_range + 1.0                # global months elapsed (1-indexed)
        phase_elapsed = i_range - lo + 1.0             # months elapsed within this phase
        growth = (1.0 + phase.real_growth_pct / 100.0) ** np.floor(phase_elapsed / 12.0)
        infl = infl_m ** elapsed_months

        if phase.kind == PhaseKind.ACCUMULATE:
            contribution[lo:hi] = phase.monthly_amount * growth * infl
            continue

        # WITHDRAW
        style = _WITHDRAWAL_STYLE_ID[phase.withdrawal_style]
        withdrawal_style[lo:hi] = style
        if phase.withdrawal_style == WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO:
            withdrawal_pct[lo:hi] = phase.withdrawal_pct_per_month / 100.0
        elif phase.withdrawal_style == WithdrawalStyle.PERCENTAGE_RAMP:
            start_frac = phase.withdrawal_pct_per_month / 100.0
            end_frac = phase.withdrawal_pct_end_per_month / 100.0
            withdrawal_pct[lo:hi] = np.linspace(start_frac, end_frac, hi - lo)
        else:
            withdrawal_full[lo:hi] = phase.monthly_amount * growth * infl
            if phase.withdrawal_style == WithdrawalStyle.DRAWDOWN_CURTAILED:
                withdrawal_reduced[lo:hi] = phase.reduced_monthly_amount * growth * infl
                withdrawal_threshold[lo:hi] = phase.drawdown_threshold_pct / 100.0

    return (contribution, withdrawal_style, withdrawal_full, withdrawal_reduced,
            withdrawal_threshold, withdrawal_pct, phase_id)


def _lump_sum_schedule(
    plan: LifePlan, horizon_months: int, infl_m: float,
) -> dict[int, list[float]]:
    """Map 0-indexed month loop iteration -> list of nominal lump-sum amounts.

    A lump sum at ``at_year <= 0`` is folded into month index 0 (applied
    after that month's market return, before that month's phase flow) —
    it is not literally "before t0", which the per-month loop has no slot
    for. Lump sums beyond the horizon are dropped (``validate()`` warns
    about this; this function stays silent so it never crashes on a plan
    the caller chose to run anyway).
    """
    by_month: dict[int, list[float]] = {}
    for ls in plan.lump_sums:
        elapsed_target = max(1, int(round(ls.at_year * 12)))
        if elapsed_target > horizon_months:
            log.debug("[LIFE] lump sum '%s' at year %.2f falls outside horizon — dropped",
                      ls.label, ls.at_year)
            continue
        i = elapsed_target - 1
        nominal = ls.amount * (infl_m ** elapsed_target)
        by_month.setdefault(i, []).append(nominal)
    return by_month


# ═══════════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LifeSimResult:
    plan: LifePlan
    values: np.ndarray                     # (n_sim, horizon_months + 1), nominal €
    planned_contributions: np.ndarray      # (horizon_months,), nominal €, >= 0
    planned_withdrawals: np.ndarray        # (horizon_months,), nominal € — median of withdrawal_requested;
                                            #   for a FIXED-style phase every sim is identical so this equals
                                            #   the deterministic scalar target exactly
    withdrawal_requested: np.ndarray       # (n_sim, horizon_months), nominal € — the NET target that
                                            #   month's withdrawal strategy asked for, per sim, before
                                            #   exhaustion could reduce what was actually received
    realised_withdrawals: np.ndarray       # (n_sim, horizon_months), nominal €, net actually received
    withdrawal_tax: np.ndarray             # (n_sim, horizon_months), nominal € — tax on THIS phase's
                                            #   withdrawal specifically. realised_withdrawals + withdrawal_tax
                                            #   == the exact gross amount sold for the withdrawal, always.
    taxes_paid: np.ndarray                 # (n_sim, horizon_months), nominal € — TOTAL tax that month,
                                            #   including any one-off lump-sum sale taxed in the same month.
                                            #   >= withdrawal_tax; use withdrawal_tax when explaining a
                                            #   specific withdrawal number, taxes_paid for a plan-wide total.
    ruin_month: np.ndarray                 # (n_sim,), first month index where V==0, else -1
    inflation_index: np.ndarray            # (horizon_months + 1,), infl_m**k
    phase_id: np.ndarray                   # (horizon_months,), index into plan.phases

    def percentiles(self, pcts=(1, 10, 50, 90, 99), *, real: bool = True) -> np.ndarray:
        """``(len(pcts), horizon_months + 1)`` percentile bands.

        ``real=True`` deflates by ``inflation_index`` first. Deflation is a
        strictly positive per-column scalar, so it commutes with the
        percentile: this deflates the bands (equivalently, one could
        deflate every path first — the two give identical results).
        """
        vals = self.values
        if real:
            vals = vals / self.inflation_index[None, :]
        return np.percentile(vals, pcts, axis=0)

    def cash_flow_bands(self, pcts=(10, 50, 90)) -> dict[str, np.ndarray]:
        """Per-month cash flows in BOTH nominal and today's-€ terms, as
        percentile bands across simulations — the data behind the "flows"
        chart panel.

        Contributions (and, for a FIXED-style withdrawal phase, the
        withdrawal target) are identical across every simulation, so their
        bands collapse to a single repeated line — that's expected, not a
        bug, and lets a chart treat every series the same way regardless
        of whether the underlying quantity happens to be path-dependent.

        Returns a dict with keys ``month_years``, ``inflation_factor``,
        and, for each of ``contribution`` / ``withdrawal_requested`` (net
        target) / ``withdrawal_realised`` (net received) / ``withdrawal_tax``
        (tax on THIS withdrawal only) / ``withdrawal_gross`` (= realised +
        withdrawal_tax, exact by construction) / ``tax`` (TOTAL that month,
        including any lump sum sold the same month), a ``<name>_nominal``
        and ``<name>_real`` array of shape ``(len(pcts), horizon_months)``.

        Note ``withdrawal_gross`` is computed by summing realised + tax
        PER SIMULATION first, then taking percentiles of that combined
        array — NOT by adding the already-separate realised/tax percentile
        bands. Percentiles don't distribute over addition in general (the
        simulation at the P50 of realised isn't necessarily the one at the
        P50 of tax), so summing bands row-by-row would silently misstate
        the actual gross-sold distribution. This way it's exact.
        """
        H = self.planned_contributions.shape[0]
        month_years = (np.arange(H) + 1) / 12.0
        deflator = self.inflation_index[1:H + 1]
        deflator = np.where(deflator > 0, deflator, 1.0)

        def _bands(per_sim_or_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            arr = per_sim_or_flat if per_sim_or_flat.ndim == 2 else per_sim_or_flat[None, :]
            nominal = np.percentile(arr, pcts, axis=0)
            real = nominal / deflator[None, :]
            return nominal, real

        contribution_nominal, contribution_real = _bands(self.planned_contributions)
        withdrawal_requested_nominal, withdrawal_requested_real = _bands(self.withdrawal_requested)
        withdrawal_realised_nominal, withdrawal_realised_real = _bands(self.realised_withdrawals)
        withdrawal_tax_nominal, withdrawal_tax_real = _bands(self.withdrawal_tax)
        withdrawal_gross_nominal, withdrawal_gross_real = _bands(
            self.realised_withdrawals + self.withdrawal_tax
        )
        tax_nominal, tax_real = _bands(self.taxes_paid)

        return {
            "month_years": month_years,
            "inflation_factor": deflator,
            "contribution_nominal": contribution_nominal,
            "contribution_real": contribution_real,
            "withdrawal_requested_nominal": withdrawal_requested_nominal,
            "withdrawal_requested_real": withdrawal_requested_real,
            "withdrawal_realised_nominal": withdrawal_realised_nominal,
            "withdrawal_realised_real": withdrawal_realised_real,
            "withdrawal_tax_nominal": withdrawal_tax_nominal,
            "withdrawal_tax_real": withdrawal_tax_real,
            "withdrawal_gross_nominal": withdrawal_gross_nominal,
            "withdrawal_gross_real": withdrawal_gross_real,
            "tax_nominal": tax_nominal,
            "tax_real": tax_real,
        }

    def cash_flow_at(self, month: int) -> dict[str, float]:
        """Scalar snapshot at *month* (1 = the first simulated month; 0 or
        earlier = t0, before any flow — all zeros, inflation factor 1.0).

        Nominal + today's-€ values for that month's contribution and
        withdrawal (requested net target, net received, tax on THAT
        withdrawal, and the resulting gross sold = received + tax, exactly)
        plus the plan-wide total tax that month (which can exceed the
        withdrawal's own tax if a lump sum was also sold that month), and
        cumulative totals from month 1 through *month* in both currencies.
        Every number the GUI readout panel shows comes from here — no
        arithmetic happens on the Tkinter side.

        ``withdrawal_gross`` is the median of the PER-SIMULATION gross
        (``realised + withdrawal_tax`` computed before taking the median),
        so it is not generally equal to ``withdrawal_realised + withdrawal_tax``
        computed from the two medians separately — see the note on
        :meth:`cash_flow_bands` for why that's correct, not a bug.
        """
        H = self.planned_contributions.shape[0]
        keys = (
            "contribution", "withdrawal_requested", "withdrawal_realised",
            "withdrawal_tax", "withdrawal_gross", "tax",
        )
        if month <= 0 or H == 0:
            zeros = {f"{k}_nominal": 0.0 for k in keys} | {f"{k}_real": 0.0 for k in keys}
            cumulative = {
                f"cumulative_{k}_{c}": 0.0
                for k in ("contributed", "withdrawn", "withdrawal_tax", "tax")
                for c in ("nominal", "real")
            }
            return {"month": 0, "inflation_factor": 1.0, **zeros, **cumulative}

        month = min(month, H)
        i = month - 1
        deflator_full = self.inflation_index[1:H + 1]
        deflator_full = np.where(deflator_full > 0, deflator_full, 1.0)
        infl = float(deflator_full[i])

        gross_per_sim = self.realised_withdrawals[:, i] + self.withdrawal_tax[:, i]

        contribution_nominal = float(self.planned_contributions[i])
        withdrawal_requested_nominal = float(np.median(self.withdrawal_requested[:, i]))
        withdrawal_realised_nominal = float(np.median(self.realised_withdrawals[:, i]))
        withdrawal_tax_nominal = float(np.median(self.withdrawal_tax[:, i]))
        withdrawal_gross_nominal = float(np.median(gross_per_sim))
        tax_nominal = float(np.median(self.taxes_paid[:, i]))

        cum_deflator = deflator_full[:month]
        cum_contributed_nominal = float(self.planned_contributions[:month].sum())
        cum_withdrawn_nominal = float(np.median(self.realised_withdrawals[:, :month].sum(axis=1)))
        cum_withdrawal_tax_nominal = float(np.median(self.withdrawal_tax[:, :month].sum(axis=1)))
        cum_tax_nominal = float(np.median(self.taxes_paid[:, :month].sum(axis=1)))
        cum_contributed_real = float((self.planned_contributions[:month] / cum_deflator).sum())
        cum_withdrawn_real = float(
            np.median((self.realised_withdrawals[:, :month] / cum_deflator[None, :]).sum(axis=1))
        )
        cum_withdrawal_tax_real = float(
            np.median((self.withdrawal_tax[:, :month] / cum_deflator[None, :]).sum(axis=1))
        )
        cum_tax_real = float(
            np.median((self.taxes_paid[:, :month] / cum_deflator[None, :]).sum(axis=1))
        )

        return {
            "month": month,
            "inflation_factor": infl,
            "contribution_nominal": contribution_nominal,
            "contribution_real": contribution_nominal / infl,
            "withdrawal_requested_nominal": withdrawal_requested_nominal,
            "withdrawal_requested_real": withdrawal_requested_nominal / infl,
            "withdrawal_realised_nominal": withdrawal_realised_nominal,
            "withdrawal_realised_real": withdrawal_realised_nominal / infl,
            "withdrawal_tax_nominal": withdrawal_tax_nominal,
            "withdrawal_tax_real": withdrawal_tax_nominal / infl,
            "withdrawal_gross_nominal": withdrawal_gross_nominal,
            "withdrawal_gross_real": withdrawal_gross_nominal / infl,
            "tax_nominal": tax_nominal,
            "tax_real": tax_nominal / infl,
            "cumulative_contributed_nominal": cum_contributed_nominal,
            "cumulative_contributed_real": cum_contributed_real,
            "cumulative_withdrawn_nominal": cum_withdrawn_nominal,
            "cumulative_withdrawn_real": cum_withdrawn_real,
            "cumulative_withdrawal_tax_nominal": cum_withdrawal_tax_nominal,
            "cumulative_withdrawal_tax_real": cum_withdrawal_tax_real,
            "cumulative_tax_nominal": cum_tax_nominal,
            "cumulative_tax_real": cum_tax_real,
        }

    def cash_flow_percentiles_at(
        self, month: int, pcts=(10, 50, 90),
    ) -> dict[str, np.ndarray]:
        """Percentile SPREAD of that month's cash flows across simulations —
        the piece :meth:`cash_flow_at` deliberately collapses to a median.

        For a FIXED-style withdrawal every simulation asks for the same
        amount, so this spread is degenerate (every percentile equal). For
        the path-dependent styles (drawdown-curtailed, % of portfolio) the
        target itself varies simulation to simulation — this is where that
        becomes visible as a number instead of only as a chart band.

        Returns a dict with ``pcts`` (as given), and for each of
        ``withdrawal_requested`` (net target) / ``withdrawal_realised`` (net
        received) / ``withdrawal_tax`` (tax on THIS withdrawal only, not
        counting any lump sum sold the same month) / ``withdrawal_gross``
        (= realised + withdrawal_tax, summed PER SIMULATION before taking
        the percentile — exact, see :meth:`cash_flow_bands`) a
        ``<name>_nominal`` and ``<name>_real`` array of length ``len(pcts)``.
        Contribution is always deterministic (ACCUMULATE phases aren't
        path-dependent) so it is omitted — use ``cash_flow_at`` for it.
        """
        H = self.planned_contributions.shape[0]
        pcts_arr = np.asarray(pcts, dtype=np.float64)
        names = ("withdrawal_requested", "withdrawal_realised", "withdrawal_tax", "withdrawal_gross")
        if month <= 0 or H == 0:
            zeros = np.zeros(len(pcts))
            return {
                "pcts": pcts_arr,
                **{f"{k}_{c}": zeros for k in names for c in ("nominal", "real")},
            }

        month = min(month, H)
        i = month - 1
        deflator_full = self.inflation_index[1:H + 1]
        infl = float(deflator_full[i]) if deflator_full[i] > 0 else 1.0

        wreq_nominal = np.percentile(self.withdrawal_requested[:, i], pcts)
        wreal_nominal = np.percentile(self.realised_withdrawals[:, i], pcts)
        wtax_nominal = np.percentile(self.withdrawal_tax[:, i], pcts)
        wgross_nominal = np.percentile(
            self.realised_withdrawals[:, i] + self.withdrawal_tax[:, i], pcts
        )

        return {
            "pcts": pcts_arr,
            "withdrawal_requested_nominal": wreq_nominal,
            "withdrawal_requested_real": wreq_nominal / infl,
            "withdrawal_realised_nominal": wreal_nominal,
            "withdrawal_realised_real": wreal_nominal / infl,
            "withdrawal_tax_nominal": wtax_nominal,
            "withdrawal_tax_real": wtax_nominal / infl,
            "withdrawal_gross_nominal": wgross_nominal,
            "withdrawal_gross_real": wgross_nominal / infl,
        }

    def summary(self, *, real: bool = True) -> dict[str, float]:
        """Headline numbers: final-value percentiles, ruin probability,
        totals contributed/withdrawn/taxed (medians across simulations
        where the quantity varies by simulation)."""
        final = self.values[:, -1].copy()
        if real:
            final = final / self.inflation_index[-1]
        p10, p50, p90 = (float(x) for x in np.percentile(final, [10, 50, 90]))

        ruined = self.ruin_month >= 0
        prob_ruin = float(np.mean(ruined)) if len(ruined) else 0.0
        median_ruin_year = (
            float(np.median(self.ruin_month[ruined]) + 1) / 12.0 if ruined.any() else None
        )

        H = self.planned_contributions.shape[0]
        flow = self.cash_flow_at(H)
        suffix = "real" if real else "nominal"

        shortfall_months = np.sum(
            (self.planned_withdrawals[None, :] > 0)
            & (self.realised_withdrawals < self.planned_withdrawals[None, :] - 1e-9),
            axis=1,
        )

        return {
            "final_value_p10": p10,
            "final_value_p50": p50,
            "final_value_p90": p90,
            "probability_of_ruin": prob_ruin,
            "median_ruin_year": median_ruin_year,
            "total_contributed": flow[f"cumulative_contributed_{suffix}"],
            "total_withdrawn_median": flow[f"cumulative_withdrawn_{suffix}"],
            "total_tax_paid_median": flow[f"cumulative_tax_{suffix}"],
            "shortfall_months_median": float(np.median(shortfall_months)) if H > 0 else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_SIM_MONTHS = 30_000_000  # n_sim * horizon_months guard (mirrors sample_monthly_returns)


def _simulate_from_monthly_returns(
    plan: LifePlan,
    monthly: np.ndarray,
    *,
    progress: Optional[Callable[[int, int], bool]] = None,
) -> LifeSimResult:
    """Run the month-by-month cash-flow simulation given an ALREADY-SAMPLED
    monthly portfolio-return matrix ``(n_sim, horizon_months)``.

    Used internally by :func:`simulate_life_strategy` (which samples via
    :func:`engine.simulation.sample_monthly_returns`) and directly by tests
    that need a fully deterministic, hand-constructed return history —
    the same "test the seam below the randomness" pattern used for
    :func:`apply_withdrawal`.

    If ``monthly.shape[1]`` is shorter than ``plan.horizon_months`` (e.g. a
    test deliberately simulating only a prefix), the deterministic
    schedules are truncated to match.
    """
    n_sim, horizon_months = monthly.shape
    inflation_pct = plan.effective_inflation_pct
    infl_m = (1.0 + inflation_pct / 100.0) ** (1.0 / 12.0)

    (contribution, withdrawal_style, withdrawal_full, withdrawal_reduced,
     withdrawal_threshold, withdrawal_pct, phase_id) = _phase_schedule(
        plan, plan.horizon_months, infl_m
    )
    contribution = contribution[:horizon_months]
    withdrawal_style = withdrawal_style[:horizon_months]
    withdrawal_full = withdrawal_full[:horizon_months]
    withdrawal_reduced = withdrawal_reduced[:horizon_months]
    withdrawal_threshold = withdrawal_threshold[:horizon_months]
    withdrawal_pct = withdrawal_pct[:horizon_months]
    phase_id = phase_id[:horizon_months]

    lump_by_month = _lump_sum_schedule(plan, plan.horizon_months, infl_m)
    tax_rate = plan.tax_rate_pct / 100.0

    V = np.full(n_sim, plan.initial_capital, dtype=np.float64)
    B = np.full(n_sim, plan.initial_capital, dtype=np.float64)
    running_max = np.full(n_sim, max(plan.initial_capital, 0.0), dtype=np.float64)

    values = np.empty((n_sim, horizon_months + 1), dtype=np.float64)
    values[:, 0] = V
    withdrawal_requested = np.zeros((n_sim, horizon_months), dtype=np.float64)
    realised_withdrawals = np.zeros((n_sim, horizon_months), dtype=np.float64)
    withdrawal_tax = np.zeros((n_sim, horizon_months), dtype=np.float64)
    taxes_paid = np.zeros((n_sim, horizon_months), dtype=np.float64)
    ruin_month = np.full(n_sim, -1, dtype=np.int64)

    progress_every = max(1, horizon_months // 10)
    months_done = 0

    for i in range(horizon_months):
        V = np.maximum(V * (1.0 + monthly[:, i]), 0.0)
        running_max = np.maximum(running_max, V)

        for lump_amount in lump_by_month.get(i, ()):
            V, B, tax, _net = _apply_inflow_or_lump(V, B, lump_amount, tax_rate)
            taxes_paid[:, i] += tax
            running_max = np.maximum(running_max, V)

        c = float(contribution[i])
        if c > 0:
            V = V + c
            B = B + c
            running_max = np.maximum(running_max, V)

        style_i = int(withdrawal_style[i])
        if style_i != 0:
            w = _resolve_withdrawal_target(
                style_i, V, running_max,
                float(withdrawal_full[i]), float(withdrawal_reduced[i]),
                float(withdrawal_threshold[i]), float(withdrawal_pct[i]),
            )
            withdrawal_requested[:, i] = w
            V, B, tax, net = apply_withdrawal(V, B, w, tax_rate)
            realised_withdrawals[:, i] = net
            withdrawal_tax[:, i] = tax          # this withdrawal's tax ONLY —
                                                 # net + withdrawal_tax == gross sold, exactly, always
            taxes_paid[:, i] += tax             # running total, may also include a lump sum's tax above

        values[:, i + 1] = V
        newly_ruined = (V <= 0) & (ruin_month == -1)
        if newly_ruined.any():
            ruin_month[newly_ruined] = i

        months_done = i + 1
        if progress is not None and (months_done % progress_every == 0 or months_done == horizon_months):
            if progress(months_done, horizon_months) is False:
                log.info("[LIFE] Cancelled at month %d/%d", months_done, horizon_months)
                break

    if months_done < horizon_months:
        values = values[:, : months_done + 1]
        withdrawal_requested = withdrawal_requested[:, :months_done]
        realised_withdrawals = realised_withdrawals[:, :months_done]
        withdrawal_tax = withdrawal_tax[:, :months_done]
        taxes_paid = taxes_paid[:, :months_done]
        contribution = contribution[:months_done]
        phase_id = phase_id[:months_done]
        horizon_months = months_done

    inflation_index = _inflation_index(horizon_months, inflation_pct)
    planned_withdrawals = (
        np.median(withdrawal_requested, axis=0) if horizon_months > 0
        else np.zeros(0, dtype=np.float64)
    )

    log.info("[LIFE] Done: final value median=%.2f  probability_of_ruin=%.3f",
             float(np.median(values[:, -1])), float(np.mean(ruin_month >= 0)))

    return LifeSimResult(
        plan=plan,
        values=values,
        planned_contributions=contribution,
        planned_withdrawals=planned_withdrawals,
        withdrawal_requested=withdrawal_requested,
        realised_withdrawals=realised_withdrawals,
        withdrawal_tax=withdrawal_tax,
        taxes_paid=taxes_paid,
        ruin_month=ruin_month,
        inflation_index=inflation_index,
        phase_id=phase_id,
    )


def simulate_life_strategy(
    plan: LifePlan,
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    progress: Optional[Callable[[int, int], bool]] = None,
) -> LifeSimResult:
    """Run the full life-strategy Monte-Carlo.

    Parameters
    ----------
    plan : validated LifePlan (call ``plan.validate()`` first; this function
        does not re-validate, so a malformed plan may raise or produce
        nonsensical output).
    weights, returns : from ``engine.data.load_all_returns(portfolio, ...)``.
    rng : optional pre-seeded generator; defaults to ``np.random.default_rng(plan.seed)``.
    progress : optional callback invoked every ~10% of the month loop with
        ``(done, total)``; return ``False`` to cancel (the GUI passes a
        closure over a ``threading.Event``). On cancellation the function
        returns a partial result truncated to the months completed so far
        — callers that don't want partial results should check the
        returned ``values.shape[1] - 1`` against ``plan.horizon_months``.
    """
    horizon_months = plan.horizon_months
    if horizon_months <= 0:
        raise ValueError("Plan has zero or negative horizon — check plan.phases.")

    n_sim = plan.n_sim
    if n_sim * horizon_months > _MAX_SIM_MONTHS:
        raise ValueError(
            f"n_sim x horizon_months = {n_sim * horizon_months:,} exceeds the "
            f"{_MAX_SIM_MONTHS:,} memory guard. Reduce n_sim (currently {n_sim}) "
            f"or the horizon (currently {horizon_months} months)."
        )

    rng = rng if rng is not None else np.random.default_rng(plan.seed)

    log.info("[LIFE] Simulating '%s': n_sim=%d horizon_months=%d block_months=%d "
             "tax_rate=%.1f%% inflation=%.2f%%",
             plan.portfolio_name, n_sim, horizon_months, plan.block_months,
             plan.tax_rate_pct, plan.effective_inflation_pct)

    monthly = sample_monthly_returns(
        weights, returns, n_sim, horizon_months, rng, block_months=plan.block_months,
    )
    return _simulate_from_monthly_returns(plan, monthly, progress=progress)
