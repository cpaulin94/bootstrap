"""bootstrap_gui.sections.lifecycle — Life Strategy Simulator panel.

Lets the user pick one portfolio from the library, build a sequence of
accumulate/hold/withdraw phases plus one-off lump sums, then run a
Monte-Carlo block-bootstrap projection (engine.lifecycle) and explore it:
a semi-transparent path cloud with confidence bands, a monthly cash-flow
chart, and an interactive year cursor.

The panel contains no simulation math — everything numeric lives in
engine/lifecycle.py. This module is UI wiring + matplotlib rendering only.
"""

from __future__ import annotations

import csv
import logging
import os
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from engine import config as cfg
from engine.data import load_all_returns
from engine.lifecycle import (
    LifePlan, LifeSimResult, LumpSum, Phase, PhaseKind, WithdrawalStyle, simulate_life_strategy,
)

from bootstrap_gui import fmt, theme
from bootstrap_gui.assets import compute_date_intersection
from bootstrap_gui.library import LifePlanLibrary, PortfolioLibrary
from bootstrap_gui.runner_mixin import BackgroundJobMixin
from bootstrap_gui.widgets import ErrorLabel, FieldLabel, NumericEntry, ScrollableFrame, StyledButton

log = logging.getLogger("bootstrap.lifecycle_ui")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# Plan <-> dict serialisation (for LifePlanLibrary / JSON persistence)
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_to_dict(p: Phase) -> dict:
    return {
        "kind": p.kind.value, "years": p.years, "monthly_amount": p.monthly_amount,
        "real_growth_pct": p.real_growth_pct, "label": p.label,
        "withdrawal_style": p.withdrawal_style.value,
        "drawdown_threshold_pct": p.drawdown_threshold_pct,
        "reduced_monthly_amount": p.reduced_monthly_amount,
        "withdrawal_pct_per_month": p.withdrawal_pct_per_month,
        "withdrawal_pct_end_per_month": p.withdrawal_pct_end_per_month,
    }


def _phase_from_dict(d: dict) -> Phase:
    return Phase(
        kind=PhaseKind(d["kind"]), years=float(d["years"]),
        monthly_amount=float(d.get("monthly_amount", 0.0)),
        real_growth_pct=float(d.get("real_growth_pct", 0.0)),
        label=d.get("label", ""),
        withdrawal_style=WithdrawalStyle(d.get("withdrawal_style", WithdrawalStyle.FIXED.value)),
        drawdown_threshold_pct=float(d.get("drawdown_threshold_pct", 5.0)),
        reduced_monthly_amount=float(d.get("reduced_monthly_amount", 0.0)),
        withdrawal_pct_per_month=float(d.get("withdrawal_pct_per_month", 0.3)),
        withdrawal_pct_end_per_month=float(d.get("withdrawal_pct_end_per_month", 0.3)),
    )


def _lump_to_dict(l: LumpSum) -> dict:
    return {"at_year": l.at_year, "amount": l.amount, "label": l.label}


def _lump_from_dict(d: dict) -> LumpSum:
    return LumpSum(at_year=float(d["at_year"]), amount=float(d["amount"]), label=d.get("label", ""))


def plan_to_dict(plan: LifePlan) -> dict:
    return {
        "portfolio_name": plan.portfolio_name,
        "initial_capital": plan.initial_capital,
        "tax_rate_pct": plan.tax_rate_pct,
        "inflation_pct": plan.inflation_pct,
        "adjust_for_inflation": plan.adjust_for_inflation,
        "phases": [_phase_to_dict(p) for p in plan.phases],
        "lump_sums": [_lump_to_dict(l) for l in plan.lump_sums],
        "n_sim": plan.n_sim,
        "block_months": plan.block_months,
        "seed": plan.seed,
        "date_start": plan.date_start,
        "date_end": plan.date_end,
    }


def plan_from_dict(d: dict) -> LifePlan:
    return LifePlan(
        portfolio_name=d.get("portfolio_name", ""),
        initial_capital=float(d.get("initial_capital", cfg.LIFE_INITIAL_CAPITAL)),
        tax_rate_pct=float(d.get("tax_rate_pct", cfg.LIFE_TAX_RATE_PCT)),
        inflation_pct=float(d.get("inflation_pct", cfg.LIFE_INFLATION_PCT)),
        adjust_for_inflation=bool(d.get("adjust_for_inflation", True)),
        phases=[_phase_from_dict(p) for p in d.get("phases", [])],
        lump_sums=[_lump_from_dict(l) for l in d.get("lump_sums", [])],
        n_sim=int(d.get("n_sim", cfg.LIFE_N_SIM)),
        block_months=int(d.get("block_months", cfg.LIFE_BLOCK_MONTHS)),
        seed=d.get("seed", 42),
        date_start=d.get("date_start") or None,
        date_end=d.get("date_end") or None,
    )


def _plan_summary_text(plan: LifePlan) -> str:
    if not plan.phases:
        return "No phases defined yet — add at least one below."
    parts = [f"Start with {fmt.money(plan.initial_capital)}."]
    for p in plan.phases:
        if p.kind == PhaseKind.ACCUMULATE:
            growth = f" (+{p.real_growth_pct:g}%/yr real)" if p.real_growth_pct else ""
            parts.append(f"Contribute {fmt.money(p.monthly_amount)}/month for {p.years:g} years{growth}.")
        elif p.kind == PhaseKind.WITHDRAW:
            growth = f" (+{p.real_growth_pct:g}%/yr real)" if p.real_growth_pct else ""
            if p.withdrawal_style == WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO:
                parts.append(
                    f"Withdraw {p.withdrawal_pct_per_month:g}%/month of portfolio value "
                    f"(net) for {p.years:g} years."
                )
            elif p.withdrawal_style == WithdrawalStyle.PERCENTAGE_RAMP:
                parts.append(
                    f"Withdraw a %/month of portfolio value (net) that ramps from "
                    f"{p.withdrawal_pct_per_month:g}% to {p.withdrawal_pct_end_per_month:g}% "
                    f"over {p.years:g} years."
                )
            elif p.withdrawal_style == WithdrawalStyle.DRAWDOWN_CURTAILED:
                parts.append(
                    f"Withdraw {fmt.money(p.monthly_amount)}/month net{growth} for {p.years:g} years "
                    f"(cut to {fmt.money(p.reduced_monthly_amount)}/month if drawdown "
                    f"≥ {p.drawdown_threshold_pct:g}%)."
                )
            else:
                parts.append(
                    f"Withdraw {fmt.money(p.monthly_amount)}/month net for {p.years:g} years{growth}."
                )
        else:
            parts.append(f"Stay invested, no cash flow, for {p.years:g} years.")
    for ls in plan.lump_sums:
        sign = "+" if ls.amount >= 0 else "-"
        label = f" ({ls.label})" if ls.label else ""
        parts.append(f"One-off: {sign}{fmt.money(abs(ls.amount))} at year {ls.at_year:g}{label}.")
    parts.append(f"Horizon: {plan.horizon_years:g} years.")
    return " ".join(parts)


def _compact_money(x: float, _pos=None) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}€{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{sign}€{x / 1_000:.0f}k"
    return f"{sign}€{x:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Add / Edit dialogs
# ═══════════════════════════════════════════════════════════════════════════════

_STYLE_DISPLAY = {
    WithdrawalStyle.FIXED: "Fixed",
    WithdrawalStyle.DRAWDOWN_CURTAILED: "Drawdown-curtailed",
    WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO: "% of portfolio",
    WithdrawalStyle.PERCENTAGE_RAMP: "% of portfolio (ramp)",
}


class PhaseDialog(tk.Toplevel):
    """Modal add/edit dialog for one Phase. Result is in ``self.result`` after close.

    For a WITHDRAW phase, a style selector shows/hides the fields relevant
    to that style — Fixed (amount + growth), Drawdown-curtailed (full +
    reduced + trigger), or % of portfolio (just the rate).
    """

    def __init__(self, parent, phase: Optional[Phase] = None):
        super().__init__(parent)
        self.title("Edit Phase" if phase else "Add Phase")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self.result: Optional[Phase] = None
        self.transient(parent)

        self.kind_var = tk.StringVar(value=(phase.kind.value if phase else PhaseKind.ACCUMULATE.value))
        self.years_var = tk.StringVar(value=f"{phase.years:g}" if phase else "10")
        self.style_var = tk.StringVar(
            value=(phase.withdrawal_style.value if phase else WithdrawalStyle.FIXED.value)
        )
        self.amount_var = tk.StringVar(value=f"{phase.monthly_amount:g}" if phase else "1000")
        self.growth_var = tk.StringVar(value=f"{phase.real_growth_pct:g}" if phase else "0")
        self.reduced_var = tk.StringVar(
            value=f"{phase.reduced_monthly_amount:g}" if phase and phase.reduced_monthly_amount else "500"
        )
        self.threshold_var = tk.StringVar(
            value=f"{phase.drawdown_threshold_pct:g}" if phase else "5"
        )
        self.pct_var = tk.StringVar(
            value=f"{phase.withdrawal_pct_per_month:g}" if phase and phase.withdrawal_pct_per_month else "0.3"
        )
        self.pct_end_var = tk.StringVar(
            value=f"{phase.withdrawal_pct_end_per_month:g}"
            if phase and phase.withdrawal_pct_end_per_month else "0.3"
        )
        self.label_var = tk.StringVar(value=phase.label if phase else "")

        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=theme.SPACE_L, pady=theme.SPACE_L)

        row = 0
        FieldLabel(body, text="Type", bg=theme.BG).grid(row=row, column=0, sticky="w", pady=3)
        kind_row = tk.Frame(body, bg=theme.BG)
        kind_row.grid(row=row, column=1, sticky="w", pady=3)
        for kind in PhaseKind:
            ttk.Radiobutton(
                kind_row, text=kind.value.capitalize(), variable=self.kind_var,
                value=kind.value, command=self._on_kind_change,
            ).pack(side="left", padx=(0, 10))
        row += 1

        FieldLabel(body, text="Duration (years)", bg=theme.BG).grid(row=row, column=0, sticky="w", pady=3)
        NumericEntry(body, textvariable=self.years_var, width=14).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        # Withdrawal-style selector — only shown when kind == WITHDRAW.
        self.style_label = FieldLabel(body, text="Withdrawal style", bg=theme.BG)
        self.style_row_frame = tk.Frame(body, bg=theme.BG)
        for style in WithdrawalStyle:
            ttk.Radiobutton(
                self.style_row_frame, text=_STYLE_DISPLAY[style], variable=self.style_var,
                value=style.value, command=self._on_style_change,
            ).pack(side="left", padx=(0, 8))
        self.style_grid_row = row
        row += 1

        self.amount_label = FieldLabel(body, text="Monthly amount (today's €)", bg=theme.BG)
        self.amount_entry = NumericEntry(body, textvariable=self.amount_var, width=14)
        self.amount_grid_row = row
        row += 1

        self.growth_label = FieldLabel(body, text="Real growth %/yr (on top of inflation)", bg=theme.BG)
        self.growth_entry = NumericEntry(body, textvariable=self.growth_var, width=14)
        self.growth_grid_row = row
        row += 1

        self.reduced_label = FieldLabel(
            body, text="Reduced amount if in drawdown (today's €)", bg=theme.BG
        )
        self.reduced_entry = NumericEntry(body, textvariable=self.reduced_var, width=14)
        self.reduced_grid_row = row
        row += 1

        self.threshold_label = FieldLabel(body, text="Drawdown trigger % (e.g. 5 = -5%)", bg=theme.BG)
        self.threshold_entry = NumericEntry(body, textvariable=self.threshold_var, width=14)
        self.threshold_grid_row = row
        row += 1

        self.pct_label = FieldLabel(body, text="% of portfolio per month (net)", bg=theme.BG)
        self.pct_entry = NumericEntry(body, textvariable=self.pct_var, width=14)
        self.pct_grid_row = row
        row += 1

        self.pct_end_label = FieldLabel(body, text="% of portfolio per month, at the END (net)", bg=theme.BG)
        self.pct_end_entry = NumericEntry(body, textvariable=self.pct_end_var, width=14)
        self.pct_end_grid_row = row
        row += 1

        FieldLabel(body, text="Label (optional)", bg=theme.BG).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.label_var, width=22).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        self.error_label = ErrorLabel(body, bg=theme.BG)
        self.error_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row += 1

        btn_row = tk.Frame(body, bg=theme.BG)
        btn_row.grid(row=row, column=0, columnspan=2, sticky="e", pady=(theme.SPACE_M, 0))
        StyledButton(btn_row, text="Cancel", command=self._cancel).pack(side="right", padx=(4, 0))
        StyledButton(btn_row, text="OK", command=self._ok, style="Accent.TButton").pack(side="right")

        self._on_kind_change()
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()
        self.focus_set()
        self.wait_window(self)

    @staticmethod
    def _show_row(label, widget, row) -> None:
        label.grid(row=row, column=0, sticky="w", pady=3)
        widget.grid(row=row, column=1, sticky="w", pady=3)

    @staticmethod
    def _hide_row(label, widget) -> None:
        label.grid_remove()
        widget.grid_remove()

    def _on_kind_change(self) -> None:
        if self.kind_var.get() == PhaseKind.WITHDRAW.value:
            self.style_label.grid(row=self.style_grid_row, column=0, sticky="w", pady=3)
            self.style_row_frame.grid(row=self.style_grid_row, column=1, sticky="w", pady=3)
        else:
            self.style_label.grid_remove()
            self.style_row_frame.grid_remove()
        self._on_style_change()

    def _on_style_change(self) -> None:
        kind = self.kind_var.get()
        style = self.style_var.get()
        is_withdraw = kind == PhaseKind.WITHDRAW.value
        is_percentage = is_withdraw and style == WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO.value
        is_ramp = is_withdraw and style == WithdrawalStyle.PERCENTAGE_RAMP.value
        is_drawdown_cut = is_withdraw and style == WithdrawalStyle.DRAWDOWN_CURTAILED.value

        if kind == PhaseKind.HOLD.value:
            self._hide_row(self.amount_label, self.amount_entry)
            self._hide_row(self.growth_label, self.growth_entry)
            self._hide_row(self.reduced_label, self.reduced_entry)
            self._hide_row(self.threshold_label, self.threshold_entry)
            self._hide_row(self.pct_label, self.pct_entry)
            self._hide_row(self.pct_end_label, self.pct_end_entry)
            return

        if is_percentage or is_ramp:
            self._hide_row(self.amount_label, self.amount_entry)
            self._hide_row(self.growth_label, self.growth_entry)
            self._hide_row(self.reduced_label, self.reduced_entry)
            self._hide_row(self.threshold_label, self.threshold_entry)
            self.pct_label.config(
                text="% of portfolio per month, at the START (net)" if is_ramp
                else "% of portfolio per month (net)"
            )
            self._show_row(self.pct_label, self.pct_entry, self.pct_grid_row)
            if is_ramp:
                self._show_row(self.pct_end_label, self.pct_end_entry, self.pct_end_grid_row)
            else:
                self._hide_row(self.pct_end_label, self.pct_end_entry)
            return

        self._hide_row(self.pct_label, self.pct_entry)
        self._hide_row(self.pct_end_label, self.pct_end_entry)
        self.amount_label.config(
            text="Full amount (today's €)" if is_drawdown_cut else "Monthly amount (today's €)"
        )
        self._show_row(self.amount_label, self.amount_entry, self.amount_grid_row)
        self._show_row(self.growth_label, self.growth_entry, self.growth_grid_row)

        if is_drawdown_cut:
            self._show_row(self.reduced_label, self.reduced_entry, self.reduced_grid_row)
            self._show_row(self.threshold_label, self.threshold_entry, self.threshold_grid_row)
        else:
            self._hide_row(self.reduced_label, self.reduced_entry)
            self._hide_row(self.threshold_label, self.threshold_entry)

    def _ok(self) -> None:
        try:
            years = float(self.years_var.get())
        except ValueError:
            self.error_label.show("Enter a valid duration.")
            return
        if years <= 0:
            self.error_label.show("Duration must be > 0 years.")
            return

        kind = PhaseKind(self.kind_var.get())
        style = WithdrawalStyle(self.style_var.get())
        is_percentage = kind == PhaseKind.WITHDRAW and style == WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO
        is_ramp = kind == PhaseKind.WITHDRAW and style == WithdrawalStyle.PERCENTAGE_RAMP
        is_drawdown_cut = kind == PhaseKind.WITHDRAW and style == WithdrawalStyle.DRAWDOWN_CURTAILED

        amount = growth = reduced = 0.0
        threshold = 5.0
        pct = pct_end = 0.3
        if kind != PhaseKind.HOLD:
            try:
                if is_percentage:
                    pct = float(self.pct_var.get())
                elif is_ramp:
                    pct = float(self.pct_var.get())
                    pct_end = float(self.pct_end_var.get())
                else:
                    amount = float(self.amount_var.get() or 0.0)
                    growth = float(self.growth_var.get() or 0.0)
                    if is_drawdown_cut:
                        reduced = float(self.reduced_var.get() or 0.0)
                        threshold = float(self.threshold_var.get() or 0.0)
            except ValueError:
                self.error_label.show("Enter valid numbers.")
                return

        if amount < 0:
            self.error_label.show("Monthly amount must be >= 0.")
            return
        if is_drawdown_cut:
            if reduced < 0 or reduced > amount:
                self.error_label.show("Reduced amount must be between 0 and the full amount.")
                return
            if not (0.0 < threshold < 100.0):
                self.error_label.show("Drawdown trigger must be between 0 and 100%.")
                return
        if is_percentage and not (0.0 < pct <= 20.0):
            self.error_label.show("Withdrawal %/month must be between 0 and 20%.")
            return
        if is_ramp:
            if not (0.0 < pct <= 20.0):
                self.error_label.show("Starting %/month must be between 0 and 20%.")
                return
            if not (0.0 < pct_end <= 20.0):
                self.error_label.show("Ending %/month must be between 0 and 20%.")
                return

        self.result = Phase(
            kind=kind, years=years, monthly_amount=amount, real_growth_pct=growth,
            label=self.label_var.get().strip(),
            withdrawal_style=style if kind == PhaseKind.WITHDRAW else WithdrawalStyle.FIXED,
            drawdown_threshold_pct=threshold,
            reduced_monthly_amount=reduced,
            withdrawal_pct_per_month=pct,
            withdrawal_pct_end_per_month=pct_end,
        )
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class LumpSumDialog(tk.Toplevel):
    """Modal add/edit dialog for one LumpSum."""

    def __init__(self, parent, lump: Optional[LumpSum] = None):
        super().__init__(parent)
        self.title("Edit One-off Event" if lump else "Add One-off Event")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self.result: Optional[LumpSum] = None
        self.transient(parent)

        self.year_var = tk.StringVar(value=f"{lump.at_year:g}" if lump else "10")
        self.amount_var = tk.StringVar(value=f"{lump.amount:g}" if lump else "50000")
        self.label_var = tk.StringVar(value=lump.label if lump else "")

        body = tk.Frame(self, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=theme.SPACE_L, pady=theme.SPACE_L)

        FieldLabel(body, text="At year (from plan start)", bg=theme.BG).grid(
            row=0, column=0, sticky="w", pady=3
        )
        NumericEntry(body, textvariable=self.year_var, width=14).grid(row=0, column=1, sticky="w", pady=3)

        FieldLabel(body, text="Amount (today's €, + inflow / - outflow)", bg=theme.BG).grid(
            row=1, column=0, sticky="w", pady=3
        )
        NumericEntry(body, textvariable=self.amount_var, width=14).grid(row=1, column=1, sticky="w", pady=3)

        FieldLabel(body, text="Label (optional)", bg=theme.BG).grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(body, textvariable=self.label_var, width=22).grid(row=2, column=1, sticky="w", pady=3)

        self.error_label = ErrorLabel(body, bg=theme.BG)
        self.error_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        btn_row = tk.Frame(body, bg=theme.BG)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="e", pady=(theme.SPACE_M, 0))
        StyledButton(btn_row, text="Cancel", command=self._cancel).pack(side="right", padx=(4, 0))
        StyledButton(btn_row, text="OK", command=self._ok, style="Accent.TButton").pack(side="right")

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()
        self.focus_set()
        self.wait_window(self)

    def _ok(self) -> None:
        try:
            year = float(self.year_var.get())
            amount = float(self.amount_var.get())
        except ValueError:
            self.error_label.show("Enter valid numbers.")
            return
        if year < 0:
            self.error_label.show("Year must be >= 0.")
            return
        self.result = LumpSum(at_year=year, amount=amount, label=self.label_var.get().strip())
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# Main panel
# ═══════════════════════════════════════════════════════════════════════════════

class LifeStrategySection(tk.Frame, BackgroundJobMixin):

    def __init__(self, parent, library: PortfolioLibrary, **kw):
        super().__init__(parent, bg=theme.BG, **kw)
        self._bg_init()
        self.library = library
        self.plan_library = LifePlanLibrary(BASE_DIR, on_error=self._on_plan_library_error)

        self._phases: list[Phase] = []
        self._lump_sums: list[LumpSum] = []
        self._returns_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._last_result: Optional[LifeSimResult] = None
        self._cursor_locked = False
        self._cursor_month = 0
        self._destroyed = False

        self._build_ui()
        self.library.on_change(self._refresh_portfolio_combo)
        self.library.on_change(self._returns_cache.clear)
        self.plan_library.on_change(self._refresh_plan_combo)

    def destroy(self) -> None:
        self._destroyed = True
        self.mark_bg_destroyed()
        super().destroy()

    def _on_plan_library_error(self, msg: str) -> None:
        log.error("[LIFE_UI] %s", msg)
        if hasattr(self, "status_label"):
            self.status_label.config(text=msg)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        setup_outer = tk.Frame(paned, bg=theme.BG, width=340)
        chart_outer = tk.Frame(paned, bg=theme.BG)
        readout_outer = tk.Frame(paned, bg=theme.BG, width=250)
        paned.add(setup_outer, weight=0)
        paned.add(chart_outer, weight=1)
        paned.add(readout_outer, weight=0)

        self._build_setup_panel(setup_outer)
        self._build_chart_panel(chart_outer)
        self._build_readout_panel(readout_outer)

    def _section(self, parent, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=theme.BG)
        frame.pack(fill="x", padx=theme.SPACE_M, pady=(theme.SPACE_M, 0))
        ttk.Label(frame, text=title.upper(), style="Heading.TLabel",
                  font=(theme.FONT_FAMILY[0], 9, "bold")).pack(fill="x", pady=(0, 2))
        tk.Frame(frame, bg=theme.BORDER, height=1).pack(fill="x", pady=(0, 4))
        return frame

    def _build_setup_panel(self, parent: tk.Frame) -> None:
        scroll = ScrollableFrame(parent, bg=theme.BG)
        scroll.pack(fill="both", expand=True)
        inner = scroll.inner

        # ── Portfolio + core params ──────────────────────────────────────
        core = self._section(inner, "Plan Setup")
        FieldLabel(core, text="Portfolio", bg=theme.BG).pack(anchor="w")
        self.portfolio_var = tk.StringVar(value="")
        self.portfolio_combo = ttk.Combobox(core, textvariable=self.portfolio_var, state="readonly", width=28)
        self.portfolio_combo.pack(anchor="w", fill="x", pady=(0, 2))
        self.portfolio_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_portfolio_info())
        self.portfolio_info_label = ttk.Label(core, text="—", style="Secondary.TLabel", wraplength=300)
        self.portfolio_info_label.pack(anchor="w", fill="x", pady=(0, 6))
        self._refresh_portfolio_combo()

        self.initial_capital_var = tk.StringVar(value=f"{cfg.LIFE_INITIAL_CAPITAL:g}")
        self.tax_rate_var = tk.StringVar(value=f"{cfg.LIFE_TAX_RATE_PCT:g}")
        self.inflation_var = tk.StringVar(value=f"{cfg.LIFE_INFLATION_PCT:g}")
        for label, var in [
            ("Initial capital (today's €)", self.initial_capital_var),
            ("Capital-gains tax %", self.tax_rate_var),
            ("Inflation %/yr", self.inflation_var),
        ]:
            FieldLabel(core, text=label, bg=theme.BG).pack(anchor="w")
            NumericEntry(core, textvariable=var, width=16).pack(anchor="w", pady=(0, 4))

        self.adjust_inflation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            core, text="Adjust cash flows for inflation", variable=self.adjust_inflation_var,
            command=self._on_adjust_inflation_toggle,
        ).pack(anchor="w")
        self.show_real_var = tk.BooleanVar(value=True)
        self.show_real_check = ttk.Checkbutton(
            core, text="Wealth chart in today's € (real)", variable=self.show_real_var,
            command=self._on_show_real_toggle,
        )
        self.show_real_check.pack(anchor="w", pady=(0, 4))
        ttk.Label(
            core, style="Secondary.TLabel", wraplength=300,
            text="Amounts above are entered in today's euros. The simulator inflates "
                 "them internally; the toggle only affects the wealth chart above — the "
                 "cash-flow panel always shows nominal AND today's-€ together, so you "
                 "can see inflation's effect directly.",
        ).pack(anchor="w", fill="x", pady=(0, 4))

        # ── Sequence ──────────────────────────────────────────────────────
        seq = self._section(inner, "Sequence of Phases")
        cols = ("num", "type", "years", "amount", "growth", "ends")
        self.phase_tree = ttk.Treeview(seq, columns=cols, show="headings", height=5, selectmode="browse")
        headers = {"num": "#", "type": "Type", "years": "Yrs", "amount": "€/mo",
                   "growth": "Growth", "ends": "Ends yr"}
        widths = {"num": 20, "type": 100, "years": 35, "amount": 90, "growth": 45, "ends": 50}
        for c in cols:
            self.phase_tree.heading(c, text=headers[c])
            self.phase_tree.column(c, width=widths[c], anchor="center")
        self.phase_tree.pack(fill="x")
        self.phase_tree.bind("<Double-1>", lambda _e: self._edit_phase())

        phase_btn_row = tk.Frame(seq, bg=theme.BG)
        phase_btn_row.pack(fill="x", pady=2)
        StyledButton(phase_btn_row, text="+ Add", command=self._add_phase).pack(side="left")
        StyledButton(phase_btn_row, text="Edit", command=self._edit_phase).pack(side="left", padx=2)
        StyledButton(phase_btn_row, text="Remove", command=self._remove_phase).pack(side="left")
        StyledButton(phase_btn_row, text="↑", command=lambda: self._move_phase(-1)).pack(side="left", padx=(6, 0))
        StyledButton(phase_btn_row, text="↓", command=lambda: self._move_phase(1)).pack(side="left")

        # ── One-off events ────────────────────────────────────────────────
        lumps = self._section(inner, "One-off Events")
        lcols = ("year", "amount", "label")
        self.lump_tree = ttk.Treeview(lumps, columns=lcols, show="headings", height=3, selectmode="browse")
        for c, h, w in [("year", "At Yr", 45), ("amount", "Amount", 80), ("label", "Label", 100)]:
            self.lump_tree.heading(c, text=h)
            self.lump_tree.column(c, width=w, anchor="center")
        self.lump_tree.pack(fill="x")
        self.lump_tree.bind("<Double-1>", lambda _e: self._edit_lump())

        lump_btn_row = tk.Frame(lumps, bg=theme.BG)
        lump_btn_row.pack(fill="x", pady=2)
        StyledButton(lump_btn_row, text="+ Add", command=self._add_lump).pack(side="left")
        StyledButton(lump_btn_row, text="Edit", command=self._edit_lump).pack(side="left", padx=2)
        StyledButton(lump_btn_row, text="Remove", command=self._remove_lump).pack(side="left")

        # ── Plan summary (read-only, natural language) ─────────────────────
        summary_frame = self._section(inner, "Plan Summary")
        self.summary_label = ttk.Label(
            summary_frame, text=_plan_summary_text(LifePlan(portfolio_name="")),
            style="Secondary.TLabel", wraplength=300, justify="left",
        )
        self.summary_label.pack(anchor="w", fill="x")
        self.warnings_label = ttk.Label(
            summary_frame, text="", style="Error.TLabel", wraplength=300, justify="left",
        )
        self.warnings_label.pack(anchor="w", fill="x", pady=(4, 0))

        # ── Historical date window ───────────────────────────────────────────
        date_frame = self._section(inner, "Historical Data Window")
        self.date_start_var = tk.StringVar(value="")
        self.date_end_var = tk.StringVar(value="")
        FieldLabel(date_frame, text="Start (YYYY-MM)", bg=theme.BG).pack(anchor="w")
        ttk.Entry(date_frame, textvariable=self.date_start_var,
                  font=(theme.FONT_FAMILY[0], 10), width=12).pack(anchor="w", pady=(0, 4))
        FieldLabel(date_frame, text="End (YYYY-MM)", bg=theme.BG).pack(anchor="w")
        ttk.Entry(date_frame, textvariable=self.date_end_var,
                  font=(theme.FONT_FAMILY[0], 10), width=12).pack(anchor="w")

        # ── Simulation params + Run ─────────────────────────────────────────
        sim = self._section(inner, "Simulation")
        self.n_sim_var = tk.StringVar(value=str(cfg.LIFE_N_SIM))
        self.block_var = tk.StringVar(value=str(cfg.LIFE_BLOCK_MONTHS))
        self.seed_var = tk.StringVar(value="42")
        for label, var in [
            ("N Simulations", self.n_sim_var),
            ("Block Size (months)", self.block_var),
            ("Seed", self.seed_var),
        ]:
            FieldLabel(sim, text=label, bg=theme.BG).pack(anchor="w")
            NumericEntry(sim, textvariable=var, width=16).pack(anchor="w", pady=(0, 4))

        self.error_label = ErrorLabel(sim, bg=theme.BG)
        self.error_label.pack(fill="x")
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(sim, variable=self.progress_var, maximum=100).pack(fill="x", pady=2)
        self.status_label = ttk.Label(sim, text="Ready.", style="Secondary.TLabel")
        self.status_label.pack(anchor="w", fill="x")

        run_row = tk.Frame(sim, bg=theme.BG)
        run_row.pack(fill="x", pady=(4, 0))
        self.run_btn = StyledButton(run_row, text="Run", command=self._run, style="Accent.TButton")
        self.run_btn.pack(side="left")
        self.stop_btn = StyledButton(run_row, text="Stop", command=self._stop)
        self.stop_btn.pack(side="left", padx=(4, 0))
        self.stop_btn.config(state="disabled")

        # ── Saved plans ───────────────────────────────────────────────────
        plans = self._section(inner, "Saved Plans")
        self.plan_var = tk.StringVar(value="")
        self.plan_combo = ttk.Combobox(plans, textvariable=self.plan_var, state="readonly", width=28)
        self.plan_combo.pack(fill="x", pady=(0, 2))
        self._refresh_plan_combo()

        plan_name_row = tk.Frame(plans, bg=theme.BG)
        plan_name_row.pack(fill="x", pady=2)
        self.plan_name_var = tk.StringVar(value="My Plan")
        ttk.Entry(plan_name_row, textvariable=self.plan_name_var, width=16).pack(side="left", padx=(0, 4))
        StyledButton(plan_name_row, text="Save", command=self._save_plan).pack(side="left")

        plan_btn_row = tk.Frame(plans, bg=theme.BG)
        plan_btn_row.pack(fill="x", pady=2)
        StyledButton(plan_btn_row, text="Load", command=self._load_plan).pack(side="left")
        StyledButton(plan_btn_row, text="Delete", command=self._delete_plan).pack(side="left", padx=4)

    def _build_chart_panel(self, parent: tk.Frame) -> None:
        toolbar = tk.Frame(parent, bg=theme.BG)
        toolbar.pack(fill="x", padx=theme.SPACE_M, pady=(theme.SPACE_M, 0))
        self.log_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Log scale (both charts)", variable=self.log_scale_var,
                        command=self._rerender_if_available).pack(side="left")
        self.cursor_state_label = ttk.Label(toolbar, text="Cursor: hover to inspect, click to lock",
                                             style="Secondary.TLabel")
        self.cursor_state_label.pack(side="left", padx=(theme.SPACE_M, 0))
        StyledButton(toolbar, text="Export Bands (CSV)", command=self._export_bands).pack(side="right")

        self.chart_frame = tk.Frame(parent, bg=theme.BG)
        self.chart_frame.pack(fill="both", expand=True, padx=theme.SPACE_M, pady=theme.SPACE_M)
        self._draw_empty_chart()

    def _flow_readout_row(self, parent: tk.Frame, label_text: str) -> tuple[ttk.Label, ttk.Label]:
        """A compact 'label / today's-€ value / nominal value' row for the
        cash-flow readout. Today's € leads, large — that's the number a
        purchasing-power-constant reader actually cares about; nominal
        trails, small — the future euro-count, useful but secondary.

        Returns ``(nominal_lbl, real_lbl)`` for backwards-compatible call
        sites — the swap is purely visual (size, order, colour), done here.
        """
        row = tk.Frame(parent, bg=theme.BG)
        row.pack(fill="x", pady=(3, 0))
        ttk.Label(row, text=label_text, style="Field.TLabel").pack(anchor="w")
        vals_row = tk.Frame(row, bg=theme.BG)
        vals_row.pack(anchor="w")
        real_lbl = ttk.Label(vals_row, text="—", font=(theme.FONT_MONO[0], 11, "bold"))
        real_lbl.pack(side="left")
        nominal_lbl = ttk.Label(vals_row, text="", style="Secondary.TLabel", font=(theme.FONT_MONO[0], 8))
        nominal_lbl.pack(side="left", padx=(6, 0))
        return nominal_lbl, real_lbl

    def _flow_percentile_table(
        self, parent: tk.Frame, label_text: str, pcts: tuple[int, ...] = (90, 50, 10),
    ) -> dict[int, tuple[ttk.Label, ttk.Label]]:
        """P90/P50/P10 (by default) breakdown for a cash flow that can be
        path-dependent — for a Fixed-style withdrawal the three rows will be
        identical; for drawdown-curtailed / % of portfolio they show the
        actual spread across simulations at this month, not just the median.

        Today's € leads, large; nominal trails, small — see ``_flow_readout_row``.
        Returns ``(nominal_lbl, real_lbl)`` per percentile for backwards
        compatibility with existing call sites.
        """
        ttk.Label(parent, text=label_text, style="Field.TLabel").pack(anchor="w", pady=(3, 0))
        labels: dict[int, tuple[ttk.Label, ttk.Label]] = {}
        for pct in pcts:
            row = tk.Frame(parent, bg=theme.BG)
            row.pack(fill="x")
            ttk.Label(row, text=f"P{pct}", style="Field.TLabel", width=4).pack(side="left")
            real_lbl = ttk.Label(row, text="—", font=(theme.FONT_MONO[0], 10, "bold"))
            real_lbl.pack(side="left")
            nominal_lbl = ttk.Label(row, text="", style="Secondary.TLabel", font=(theme.FONT_MONO[0], 8))
            nominal_lbl.pack(side="left", padx=(6, 0))
            labels[pct] = (nominal_lbl, real_lbl)
        return labels

    def _build_readout_panel(self, parent: tk.Frame) -> None:
        frame = self._section(parent, "Readout")
        self.readout_year_label = ttk.Label(frame, text="—", font=(theme.FONT_FAMILY[0], 12, "bold"))
        self.readout_year_label.pack(anchor="w", pady=(0, 4))

        self.readout_pct_labels: dict[str, ttk.Label] = {}
        for pct in ("P99", "P90", "P50", "P10", "P1"):
            row = tk.Frame(frame, bg=theme.BG)
            row.pack(fill="x")
            ttk.Label(row, text=pct, style="Field.TLabel", width=5).pack(side="left")
            lbl = ttk.Label(row, text="—", font=(theme.FONT_MONO[0], 10))
            lbl.pack(side="left")
            self.readout_pct_labels[pct] = lbl

        tk.Frame(frame, bg=theme.BORDER, height=1).pack(fill="x", pady=6)

        self.readout_phase_label = ttk.Label(frame, text="Phase: —", style="Secondary.TLabel",
                                              wraplength=200)
        self.readout_phase_label.pack(anchor="w", pady=(0, 4))

        ttk.Label(frame, text="CASH FLOW  (today's € · nominal)", style="Field.TLabel").pack(anchor="w")
        self.readout_contrib_nom, self.readout_contrib_real = self._flow_readout_row(frame, "Contribution")
        # Contribution is always deterministic (same for every sim); everything
        # below is NOT for drawdown-curtailed / % of portfolio, so each gets a
        # percentile table instead of a single median row — that spread is
        # the whole point of those two strategies. Net + tax = gross EXACTLY
        # per simulation (engine.lifecycle.apply_withdrawal), but percentiles
        # of the three don't add row-by-row — see the caption below.
        self.readout_wreq_pct = self._flow_percentile_table(frame, "Target (net requested)")
        self.readout_wreal_pct = self._flow_percentile_table(frame, "Received (net, after tax)")
        self.readout_wtax_pct = self._flow_percentile_table(frame, "Tax (on this withdrawal)")
        self.readout_wgross_pct = self._flow_percentile_table(frame, "Gross sold (= received + tax)")
        ttk.Label(
            frame, style="Secondary.TLabel", wraplength=200, font=(theme.FONT_MONO[0], 7),
            text="Each row's P10/50/90 is its own distribution — e.g. P50 tax "
                 "usually isn't P50 gross − P50 received (different simulations "
                 "rank differently on each). Net + tax = gross always holds "
                 "per simulation; see Export CSV for the raw per-sim figures.",
        ).pack(anchor="w", fill="x", pady=(2, 0))
        self.readout_inflation_label = ttk.Label(frame, text="Inflation: —", style="Secondary.TLabel")
        self.readout_inflation_label.pack(anchor="w", pady=(4, 4))

        tk.Frame(frame, bg=theme.BORDER, height=1).pack(fill="x", pady=4)
        ttk.Label(frame, text="CUMULATIVE TO DATE", style="Field.TLabel").pack(anchor="w")
        self.readout_cum_contrib_nom, self.readout_cum_contrib_real = self._flow_readout_row(
            frame, "Contributed"
        )
        self.readout_cum_withdraw_nom, self.readout_cum_withdraw_real = self._flow_readout_row(
            frame, "Withdrawn (net)"
        )
        self.readout_cum_wtax_nom, self.readout_cum_wtax_real = self._flow_readout_row(
            frame, "Tax (withdrawals only)"
        )
        self.readout_cum_tax_nom, self.readout_cum_tax_real = self._flow_readout_row(
            frame, "Tax (plan total, incl. lump sums)"
        )

        tk.Frame(frame, bg=theme.BORDER, height=1).pack(fill="x", pady=6)
        self.readout_ruin_label = ttk.Label(frame, text="Ruined paths: —", style="Secondary.TLabel",
                                             wraplength=200)
        self.readout_ruin_label.pack(anchor="w")
        self.readout_lump_shortfall_label = ttk.Label(
            frame, text="", style="Error.TLabel", wraplength=200,
        )
        self.readout_lump_shortfall_label.pack(anchor="w")

    # ── Portfolio / plan combobox refresh ────────────────────────────────

    def _refresh_portfolio_combo(self) -> None:
        names = self.library.names()
        self.portfolio_combo["values"] = names
        if names and self.portfolio_var.get() not in names:
            self.portfolio_var.set(names[0])
        self._refresh_portfolio_info()

    def _refresh_portfolio_info(self) -> None:
        name = self.portfolio_var.get()
        weights = self.library.get(name) if name else {}
        if not weights:
            self.portfolio_info_label.config(text="—")
            return
        comp = " · ".join(f"{t} {w:.0%}" for t, w in sorted(weights.items()) if w > 0)
        ds, de = compute_date_intersection(list(weights.keys()))
        self.portfolio_info_label.config(text=f"{comp}\nHistory: {ds} → {de}")

    def _refresh_plan_combo(self) -> None:
        names = self.plan_library.names()
        self.plan_combo["values"] = names

    # ── Phase list management ────────────────────────────────────────────

    def _add_phase(self) -> None:
        dialog = PhaseDialog(self)
        if dialog.result is not None:
            self._phases.append(dialog.result)
            self._refresh_phase_tree()

    def _edit_phase(self) -> None:
        sel = self.phase_tree.selection()
        if not sel:
            return
        idx = self.phase_tree.index(sel[0])
        dialog = PhaseDialog(self, self._phases[idx])
        if dialog.result is not None:
            self._phases[idx] = dialog.result
            self._refresh_phase_tree()

    def _remove_phase(self) -> None:
        sel = self.phase_tree.selection()
        if not sel:
            return
        idx = self.phase_tree.index(sel[0])
        del self._phases[idx]
        self._refresh_phase_tree()

    def _move_phase(self, delta: int) -> None:
        sel = self.phase_tree.selection()
        if not sel:
            return
        idx = self.phase_tree.index(sel[0])
        new_idx = idx + delta
        if 0 <= new_idx < len(self._phases):
            self._phases[idx], self._phases[new_idx] = self._phases[new_idx], self._phases[idx]
            self._refresh_phase_tree()
            children = self.phase_tree.get_children()
            self.phase_tree.selection_set(children[new_idx])

    def _refresh_phase_tree(self) -> None:
        self.phase_tree.delete(*self.phase_tree.get_children())
        cum_years = 0.0
        for i, p in enumerate(self._phases):
            cum_years += p.years
            growth = f"+{p.real_growth_pct:g}%" if p.real_growth_pct else "—"
            if p.kind == PhaseKind.HOLD:
                type_label, amount = "Hold", "—"
            elif p.kind == PhaseKind.WITHDRAW and p.withdrawal_style == WithdrawalStyle.PERCENTAGE_OF_PORTFOLIO:
                type_label, amount, growth = "Withdraw (%)", f"{p.withdrawal_pct_per_month:g}%/mo", "—"
            elif p.kind == PhaseKind.WITHDRAW and p.withdrawal_style == WithdrawalStyle.PERCENTAGE_RAMP:
                type_label = "Withdraw (% ramp)"
                amount = f"{p.withdrawal_pct_per_month:g}% → {p.withdrawal_pct_end_per_month:g}%/mo"
                growth = "—"
            elif p.kind == PhaseKind.WITHDRAW and p.withdrawal_style == WithdrawalStyle.DRAWDOWN_CURTAILED:
                type_label = "Withdraw (dd-cut)"
                amount = f"{p.monthly_amount:g} / {p.reduced_monthly_amount:g}"
            else:
                type_label, amount = p.kind.value.capitalize(), f"{p.monthly_amount:g}"
            self.phase_tree.insert("", "end", values=(
                i + 1, type_label, f"{p.years:g}", amount, growth, f"{cum_years:g}",
            ))
        self._refresh_summary()

    # ── Lump sum list management ─────────────────────────────────────────

    def _add_lump(self) -> None:
        dialog = LumpSumDialog(self)
        if dialog.result is not None:
            self._lump_sums.append(dialog.result)
            self._refresh_lump_tree()

    def _edit_lump(self) -> None:
        sel = self.lump_tree.selection()
        if not sel:
            return
        idx = self.lump_tree.index(sel[0])
        dialog = LumpSumDialog(self, self._lump_sums[idx])
        if dialog.result is not None:
            self._lump_sums[idx] = dialog.result
            self._refresh_lump_tree()

    def _remove_lump(self) -> None:
        sel = self.lump_tree.selection()
        if not sel:
            return
        idx = self.lump_tree.index(sel[0])
        del self._lump_sums[idx]
        self._refresh_lump_tree()

    def _refresh_lump_tree(self) -> None:
        self.lump_tree.delete(*self.lump_tree.get_children())
        for ls in self._lump_sums:
            sign = "+" if ls.amount >= 0 else "-"
            self.lump_tree.insert("", "end", values=(f"{ls.at_year:g}", f"{sign}{fmt.money(abs(ls.amount))}",
                                                       ls.label))
        self._refresh_summary()

    def _on_adjust_inflation_toggle(self) -> None:
        enabled = self.adjust_inflation_var.get()
        state = "normal" if enabled else "disabled"
        self.show_real_check.configure(state=state)
        if not enabled:
            self.show_real_var.set(False)
        self._refresh_summary()

    def _on_show_real_toggle(self) -> None:
        self._rerender_if_available()

    def _refresh_summary(self) -> None:
        plan = self._build_plan_from_ui(validate_numbers=False)
        self.summary_label.config(text=_plan_summary_text(plan))

    # ── Build LifePlan from current UI state ─────────────────────────────

    def _safe_float(self, var: tk.StringVar, default: float) -> float:
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return default

    def _safe_int(self, var: tk.StringVar, default: int) -> int:
        try:
            return int(float(var.get()))
        except (ValueError, tk.TclError):
            return default

    def _build_plan_from_ui(self, *, validate_numbers: bool = True) -> LifePlan:
        seed_str = self.seed_var.get().strip() if hasattr(self, "seed_var") else "42"
        seed = int(seed_str) if seed_str.lstrip("-").isdigit() else 42
        return LifePlan(
            portfolio_name=self.portfolio_var.get() if hasattr(self, "portfolio_var") else "",
            initial_capital=self._safe_float(self.initial_capital_var, cfg.LIFE_INITIAL_CAPITAL)
            if hasattr(self, "initial_capital_var") else cfg.LIFE_INITIAL_CAPITAL,
            tax_rate_pct=self._safe_float(self.tax_rate_var, cfg.LIFE_TAX_RATE_PCT)
            if hasattr(self, "tax_rate_var") else cfg.LIFE_TAX_RATE_PCT,
            inflation_pct=self._safe_float(self.inflation_var, cfg.LIFE_INFLATION_PCT)
            if hasattr(self, "inflation_var") else cfg.LIFE_INFLATION_PCT,
            adjust_for_inflation=self.adjust_inflation_var.get() if hasattr(self, "adjust_inflation_var") else True,
            phases=list(self._phases),
            lump_sums=list(self._lump_sums),
            n_sim=self._safe_int(self.n_sim_var, cfg.LIFE_N_SIM) if hasattr(self, "n_sim_var") else cfg.LIFE_N_SIM,
            block_months=self._safe_int(self.block_var, cfg.LIFE_BLOCK_MONTHS)
            if hasattr(self, "block_var") else cfg.LIFE_BLOCK_MONTHS,
            seed=seed,
            date_start=self.date_start_var.get().strip() or None
            if hasattr(self, "date_start_var") else None,
            date_end=self.date_end_var.get().strip() or None
            if hasattr(self, "date_end_var") else None,
        )

    # ── Save / load plans ─────────────────────────────────────────────────

    def _save_plan(self) -> None:
        name = self.plan_name_var.get().strip()
        if not name:
            self.error_label.show("Enter a plan name.")
            return
        plan = self._build_plan_from_ui()
        self.plan_library.add(name, plan_to_dict(plan))
        self.status_label.config(text=f"Saved plan '{name}'.")

    def _load_plan(self) -> None:
        name = self.plan_var.get()
        d = self.plan_library.get(name)
        if not d:
            return
        plan = plan_from_dict(d)
        if plan.portfolio_name and plan.portfolio_name in self.library.names():
            self.portfolio_var.set(plan.portfolio_name)
        self.initial_capital_var.set(f"{plan.initial_capital:g}")
        self.tax_rate_var.set(f"{plan.tax_rate_pct:g}")
        self.inflation_var.set(f"{plan.inflation_pct:g}")
        self.adjust_inflation_var.set(plan.adjust_for_inflation)
        self.n_sim_var.set(str(plan.n_sim))
        self.block_var.set(str(plan.block_months))
        self.seed_var.set(str(plan.seed))
        self.date_start_var.set(plan.date_start or "")
        self.date_end_var.set(plan.date_end or "")
        self._phases = list(plan.phases)
        self._lump_sums = list(plan.lump_sums)
        self._refresh_phase_tree()
        self._refresh_lump_tree()
        self._refresh_portfolio_info()
        self.plan_name_var.set(name)
        self.status_label.config(text=f"Loaded plan '{name}'.")

    def _delete_plan(self) -> None:
        name = self.plan_var.get()
        if name:
            self.plan_library.delete(name)
            self.status_label.config(text=f"Deleted plan '{name}'.")

    # ── Run / Stop ─────────────────────────────────────────────────────────

    def _get_returns_cached(
        self, portfolio_name: str, weights_dict: dict,
        date_start: Optional[str], date_end: Optional[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        # date_start/date_end are part of the key (not just portfolio_name):
        # a plan's Historical Data Window can change without the library
        # itself changing, and that's a different data window entirely.
        key = (portfolio_name, date_start, date_end)
        cached = self._returns_cache.get(key)
        if cached is not None:
            return cached
        result = load_all_returns(
            weights_dict, cfg.USE_AFTER_TER_RETURNS,
            date_start=date_start, date_end=date_end,
        )
        self._returns_cache[key] = result
        return result

    def _run(self) -> None:
        if self.is_running:
            return
        self.error_label.clear()

        plan = self._build_plan_from_ui()
        problems = plan.validate()
        # tax_rate_pct >= 100 isn't just an unusual input: apply_withdrawal's
        # gross-up denominator (1 - tax_rate * latent_gain_fraction) goes
        # negative once the latent gain exceeds 1/tax_rate, and the
        # withdrawal silently comes back as 0 — a "successful" run with
        # every number quietly wrong, not an error. Every other soft
        # problem below degrades visibly (an engine exception, an obviously
        # off number) or is a plan the user might deliberately want to
        # explore anyway; this one doesn't, so it's blocked outright rather
        # than just listed as a warning.
        bad_tax_rate = not (0.0 <= plan.tax_rate_pct < 100.0)
        hard_block = (not plan.phases) or plan.horizon_months <= 0 or bad_tax_rate
        self.warnings_label.config(text="  •  ".join(problems) if problems else "")
        if hard_block:
            if bad_tax_rate:
                message = "Tax rate must be in [0, 100) — withdrawals go silently to zero above that."
            else:
                message = problems[0] if problems else "Invalid plan — add at least one phase."
            self.error_label.show(message)
            return

        portfolio_name = plan.portfolio_name
        weights_dict = self.library.get(portfolio_name) if portfolio_name else {}
        if not weights_dict:
            self.error_label.show("Select a portfolio with at least one asset.")
            return

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self.status_label.config(text="Running…")
        self._last_result = None

        try:
            weights, returns = self._get_returns_cached(
                portfolio_name, weights_dict, plan.date_start, plan.date_end,
            )
        except Exception as e:
            self.error_label.show(f"Could not load return data: {e}")
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            return

        def worker(*, result_queue, stop_event, plan=plan, weights=weights, returns=returns):
            def progress_cb(done, total):
                result_queue.put(("progress", done, total))
                return not stop_event.is_set()

            result = simulate_life_strategy(plan, weights, returns, progress=progress_cb)
            result_queue.put(("result", result))

        def on_message(msg):
            if msg[0] == "progress":
                done, total = msg[1], msg[2]
                self.progress_var.set(done / total * 100)
                self.status_label.config(text=f"{done}/{total} months simulated")
            elif msg[0] == "result":
                self._last_result = msg[1]

        def on_done():
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            if self._last_result is not None:
                self._render_charts(self._last_result)
                self.status_label.config(text="Simulation complete.")
                prob = self._last_result.summary().get("probability_of_lump_sum_shortfall", 0.0)
                if prob > 0:
                    self.readout_lump_shortfall_label.config(
                        text=f"⚠ {fmt.pct(prob)} of paths couldn't fully fund a "
                             f"negative lump sum (portfolio too small at the time)."
                    )
                else:
                    self.readout_lump_shortfall_label.config(text="")
            else:
                self.status_label.config(text="Simulation cancelled.")

        def on_error(exc: BaseException):
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.error_label.show(str(exc))
            self.status_label.config(text="Simulation failed.")

        self.start_job(worker, on_message=on_message, on_done=on_done, on_error=on_error)

    def _stop(self) -> None:
        self.cancel_job()
        self.status_label.config(text="Stopping…")

    # ── Charts ─────────────────────────────────────────────────────────────

    def _draw_empty_chart(self) -> None:
        for w in self.chart_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(8, 6), dpi=theme.PLOT_DPI, facecolor=theme.BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(theme.PANEL_BG)
        ax.text(0.5, 0.5, "Set up a plan and press Run to see the simulation.",
                ha="center", va="center", color=theme.TEXT_SEC, fontsize=10,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _rerender_if_available(self) -> None:
        if self._last_result is not None:
            self._render_charts(self._last_result)

    def _render_charts(self, result: LifeSimResult) -> None:
        for w in self.chart_frame.winfo_children():
            w.destroy()

        real = self.show_real_var.get() and self.adjust_inflation_var.get()
        n_months = result.values.shape[1]
        years = np.arange(n_months) / 12.0
        inflation_active = result.plan.effective_inflation_pct != 0.0

        vals = result.values / result.inflation_index[None, :] if real else result.values

        fig = Figure(figsize=(9, 8), dpi=theme.PLOT_DPI, facecolor=theme.BG, layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.12)
        ax_cloud = fig.add_subplot(gs[0, 0])
        ax_flows = fig.add_subplot(gs[1, 0], sharex=ax_cloud)
        theme.apply_style(ax_cloud, "Simulated Wealth Paths" + (" (today's €)" if real else " (nominal €)"))
        theme.apply_style(ax_flows, "Monthly Cash Flow — today's € (solid) · nominal (dashed, faint)")

        n_show = min(300, vals.shape[0])
        idx = np.random.default_rng(0).choice(vals.shape[0], size=n_show, replace=False)
        segments = [np.column_stack([years, vals[i]]) for i in idx]
        ax_cloud.add_collection(LineCollection(segments, colors=theme.TEXT, alpha=0.05, linewidths=0.5))

        pcts = (1, 10, 50, 90, 99)
        p1, p10, p50, p90, p99 = result.percentiles(pcts, real=real)
        ax_cloud.fill_between(years, p1, p99, color=theme.ACCENT, alpha=0.10, linewidth=0)
        ax_cloud.fill_between(years, p10, p90, color=theme.ACCENT, alpha=0.22, linewidth=0)
        ax_cloud.plot(years, p50, color=theme.ACCENT, linewidth=1.8)

        # phase boundaries
        cum = 0.0
        shade = False
        for p in result.plan.phases:
            if shade:
                ax_cloud.axvspan(cum, cum + p.years, color=theme.TEXT_SEC, alpha=0.03, linewidth=0)
            ax_cloud.axvline(cum, color=theme.BORDER, linewidth=0.8, linestyle="--")
            ax_flows.axvline(cum, color=theme.BORDER, linewidth=0.8, linestyle="--")
            cum += p.years
            shade = not shade
        y_top = max(float(p99.max()), 1.0)
        cum = 0.0
        for p in result.plan.phases:
            label = {"accumulate": "ACC", "hold": "HOLD", "withdraw": "WDR"}[p.kind.value]
            ax_cloud.annotate(label, xy=(cum + p.years / 2, y_top), xycoords="data",
                               ha="center", va="bottom", fontsize=7, color=theme.TEXT_SEC)
            cum += p.years

        for ls in result.plan.lump_sums:
            if 0 <= ls.at_year <= years[-1]:
                ax_cloud.annotate(
                    ("+" if ls.amount >= 0 else "-") + fmt.money(abs(ls.amount)),
                    xy=(ls.at_year, p50[min(int(round(ls.at_year * 12)), len(p50) - 1)]),
                    xytext=(0, 14), textcoords="offset points", ha="center", fontsize=7,
                    color=theme.POSITIVE if ls.amount >= 0 else theme.WARNING,
                    arrowprops=dict(arrowstyle="->", color=theme.TEXT_SEC, lw=0.7),
                )

        ax_cloud.set_xlim(years[0], years[-1])
        if self.log_scale_var.get():
            ax_cloud.set_yscale("symlog", linthresh=max(1000.0, float(np.median(p50)) * 0.01 + 1))
        # Wealth is never negative by construction (V floors at 0 on ruin) —
        # force the bottom at exactly 0 regardless of scale. Without this,
        # matplotlib's autoscale margin extends a few % past the data's
        # minimum on both sides, and with symlog that margin is free to go
        # negative (symlog is symmetric about 0), which read as "the chart
        # goes negative" even though no simulated path ever does.
        ax_cloud.set_ylim(bottom=0)
        ax_cloud.yaxis.set_major_formatter(FuncFormatter(_compact_money))

        # ── monthly flows — today's-€ is primary everywhere; nominal is a
        # thin dashed, semi-transparent reference (the mirror image of the
        # readout panel, which shows the same priority) ───────────────────
        flow = result.cash_flow_bands((1, 10, 50, 90, 99))
        month_years = flow["month_years"]
        contrib_nom = flow["contribution_nominal"][2]      # P50 row (deterministic -> flat repeat)
        contrib_real = flow["contribution_real"][2]
        wreq_nom_p1, wreq_nom_p10, wreq_nom_p50, wreq_nom_p90, wreq_nom_p99 = flow["withdrawal_requested_nominal"]
        wreq_real_p1, wreq_real_p10, wreq_real_p50, wreq_real_p90, wreq_real_p99 = flow["withdrawal_requested_real"]
        wreal_nom_p50 = flow["withdrawal_realised_nominal"][2]
        wreal_real_p50 = flow["withdrawal_realised_real"][2]

        # Contribution: today's-€ solid area is primary; nominal is a thin
        # dashed reference showing what that becomes once inflated.
        ax_flows.fill_between(month_years, 0, contrib_real, step="post",
                               color=theme.POSITIVE, alpha=0.40, label="Contribution (today's €)")
        if inflation_active:
            ax_flows.plot(month_years, contrib_nom, color=theme.POSITIVE, linewidth=1.0,
                          alpha=0.45, linestyle="--", drawstyle="steps-post",
                          label="Contribution (nominal)")

        # Withdrawal TARGET: full P1–P99 confidence bands, today's-€, primary —
        # same visual language as the wealth chart above. For a Fixed-style
        # phase every percentile collapses onto the same flat line (nothing
        # path-dependent to show); for drawdown-curtailed / % of portfolio
        # the bands are the whole point.
        ax_flows.fill_between(month_years, -wreq_real_p1, -wreq_real_p99, step="post",
                               color=theme.WARNING, alpha=0.10, linewidth=0,
                               label="Target (P1–P99, today's €)")
        ax_flows.fill_between(month_years, -wreq_real_p10, -wreq_real_p90, step="post",
                               color=theme.WARNING, alpha=0.24, linewidth=0,
                               label="Target (P10–P90, today's €)")
        ax_flows.plot(month_years, -wreq_real_p50, color=theme.WARNING, linewidth=1.8,
                      drawstyle="steps-post", label="Target (P50, today's €)")
        if inflation_active:
            ax_flows.plot(month_years, -wreq_nom_p50, color=theme.WARNING, linewidth=1.0,
                          alpha=0.45, linestyle="--", drawstyle="steps-post",
                          label="Target (P50, nominal)")

        # Withdrawal REALISED (net actually received): today's-€ median in a
        # visually distinct dash-dot, primary — diverges from the target
        # band once a path is exhausted, in the currency that matters.
        ax_flows.plot(month_years, -wreal_real_p50, color=theme.WARNING, linewidth=1.5,
                      linestyle=(0, (4, 1, 1, 1)), label="Realised (P50, today's €)")
        if inflation_active:
            ax_flows.plot(month_years, -wreal_nom_p50, color=theme.WARNING, linewidth=0.9,
                          alpha=0.4, linestyle=(0, (4, 1, 1, 1)), label="Realised (P50, nominal)")

        # Phase-boundary value annotations (nominal €/mo at start -> end of phase).
        cum = 0.0
        for p in result.plan.phases:
            if p.kind != PhaseKind.HOLD and p.years > 0 and len(month_years) > 0:
                start_i = min(int(round(cum * 12)), len(month_years) - 1)
                end_i = min(max(int(round((cum + p.years) * 12)) - 1, start_i), len(month_years) - 1)
                series = contrib_nom if p.kind == PhaseKind.ACCUMULATE else wreq_nom_p50
                sign = 1 if p.kind == PhaseKind.ACCUMULATE else -1
                y_offset = 8 if p.kind == PhaseKind.ACCUMULATE else -16
                ax_flows.annotate(
                    f"{_compact_money(series[start_i])} → {_compact_money(series[end_i])}",
                    xy=(cum + p.years / 2, sign * series[start_i]),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha="center", fontsize=6.5, color=theme.TEXT_SEC,
                )
            cum += p.years

        ax_flows.axhline(0, color=theme.TEXT, linewidth=0.5)
        if self.log_scale_var.get():
            # Flows legitimately go negative (withdrawals plotted below zero),
            # so symlog — not a forced bottom=0 — is the correct choice here;
            # linthresh keeps a linear region around zero instead of trying
            # to log-scale genuinely small/zero flows.
            flow_scale = max(
                float(np.max(contrib_nom)) if contrib_nom.size else 0.0,
                float(np.max(wreq_nom_p90)) if wreq_nom_p90.size else 0.0,
                1.0,
            )
            ax_flows.set_yscale("symlog", linthresh=max(10.0, flow_scale * 0.01))
        ax_flows.yaxis.set_major_formatter(FuncFormatter(_compact_money))
        ax_flows.set_xlabel("Years")
        ax_flows.legend(fontsize=6, ncol=2, loc="upper right", frameon=False)

        if not inflation_active:
            ax_flows.text(0.01, 0.02, "inflation off → nominal = today's €",
                          transform=ax_flows.transAxes, fontsize=7, color=theme.TEXT_SEC)
        elif np.allclose(wreal_real_p50, wreq_real_p50, atol=1.0):
            ax_flows.text(0.01, 0.02, "plan fully funded in ≥ 50% of paths",
                          transform=ax_flows.transAxes, fontsize=7, color=theme.TEXT_SEC)

        fig.text(
            0.01, 0.002,
            "solid = today's € (purchasing power) · dashed, faint = nominal € of that year — "
            "nominal(m) = today's € × (1+inflation)^(m/12) × (1+real growth)^⌊m/12⌋",
            fontsize=6.5, color=theme.TEXT_SEC, ha="left", va="bottom",
        )

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        self._wire_cursor(canvas, ax_cloud, ax_flows, result, real, years)

    # ── Interactive cursor ────────────────────────────────────────────────

    def _wire_cursor(self, canvas, ax_cloud, ax_flows, result: LifeSimResult, real: bool,
                      years: np.ndarray) -> None:
        self._cursor_locked = False
        self._cursor_result = result
        self._cursor_real = real
        self._cursor_years = years
        self._cursor_canvas = canvas
        self._cursor_line_cloud = ax_cloud.axvline(years[-1], color=theme.WARNING, linewidth=1, alpha=0.8)
        self._cursor_line_flows = ax_flows.axvline(years[-1], color=theme.WARNING, linewidth=1, alpha=0.8)
        self.cursor_state_label.config(text="Cursor: hover to inspect, click to lock")

        widget = canvas.get_tk_widget()
        widget.focus_set()

        def on_move(event):
            if self._cursor_locked or event.inaxes not in (ax_cloud, ax_flows) or event.xdata is None:
                return
            self._update_cursor(event.xdata)

        def on_click(event):
            if event.inaxes not in (ax_cloud, ax_flows) or event.xdata is None:
                return
            self._cursor_locked = not self._cursor_locked
            self.cursor_state_label.config(
                text="Cursor: LOCKED (click to unlock, ←/→ to move)" if self._cursor_locked
                else "Cursor: hover to inspect, click to lock"
            )
            if self._cursor_locked:
                self._update_cursor(event.xdata)

        def on_key(event, step_months=1):
            if not self._cursor_locked:
                return
            self._cursor_month = max(0, min(self._cursor_month + step_months, len(years) - 1))
            self._update_cursor(self._cursor_month / 12.0)

        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("button_press_event", on_click)
        widget.bind("<Left>", lambda e: on_key(e, -1))
        widget.bind("<Right>", lambda e: on_key(e, 1))
        widget.bind("<Shift-Left>", lambda e: on_key(e, -12))
        widget.bind("<Shift-Right>", lambda e: on_key(e, 12))

        self._update_cursor(years[-1])

    def _update_cursor(self, year_value: float) -> None:
        result = self._cursor_result
        n_months = result.values.shape[1]
        month = int(round(year_value * 12))
        month = max(0, min(month, n_months - 1))
        self._cursor_month = month

        vals_col = result.values[:, month]
        if self._cursor_real:
            vals_col = vals_col / result.inflation_index[month]
        p1, p10, p50, p90, p99 = np.percentile(vals_col, [1, 10, 50, 90, 99])

        self.readout_year_label.config(text=f"Year {month / 12.0:.2f}")
        for key, val in [("P99", p99), ("P90", p90), ("P50", p50), ("P10", p10), ("P1", p1)]:
            self.readout_pct_labels[key].config(text=fmt.money(val))

        # cash_flow_at() uses the SAME "months elapsed" convention as `month`
        # here (0 = t0, k = after k simulated months) — no index juggling.
        flow = result.cash_flow_at(month)

        phase_idx = int(result.phase_id[month - 1]) if 1 <= month <= len(result.phase_id) else -1
        phase = result.plan.phases[phase_idx] if 0 <= phase_idx < len(result.plan.phases) else None
        self.readout_phase_label.config(
            text=f"Phase: {phase.kind.value.capitalize()}" if phase else "Phase: start"
        )

        def _set_flow(nom_lbl: ttk.Label, real_lbl: ttk.Label, nominal_val: float, real_val: float) -> None:
            # Today's € is the headline number (large, no prefix); nominal is
            # the small annotation explaining what that becomes by then.
            real_lbl.config(text=fmt.money(real_val))
            nom_lbl.config(text=f"nominal {fmt.money(nominal_val)}")

        _set_flow(self.readout_contrib_nom, self.readout_contrib_real,
                  flow["contribution_nominal"], flow["contribution_real"])

        pcts = tuple(self.readout_wreq_pct.keys())
        spread = result.cash_flow_percentiles_at(month, pcts)
        tables = [
            (self.readout_wreq_pct, "withdrawal_requested"),
            (self.readout_wreal_pct, "withdrawal_realised"),
            (self.readout_wtax_pct, "withdrawal_tax"),
            (self.readout_wgross_pct, "withdrawal_gross"),
        ]
        for pct in pcts:
            idx = list(spread["pcts"]).index(pct)
            for table, key in tables:
                nom_lbl, real_lbl = table[pct]
                _set_flow(nom_lbl, real_lbl, spread[f"{key}_nominal"][idx], spread[f"{key}_real"][idx])

        self.readout_inflation_label.config(text=f"Inflation: ×{flow['inflation_factor']:.2f} vs today")

        _set_flow(self.readout_cum_contrib_nom, self.readout_cum_contrib_real,
                  flow["cumulative_contributed_nominal"], flow["cumulative_contributed_real"])
        _set_flow(self.readout_cum_withdraw_nom, self.readout_cum_withdraw_real,
                  flow["cumulative_withdrawn_nominal"], flow["cumulative_withdrawn_real"])
        _set_flow(self.readout_cum_wtax_nom, self.readout_cum_wtax_real,
                  flow["cumulative_withdrawal_tax_nominal"], flow["cumulative_withdrawal_tax_real"])
        _set_flow(self.readout_cum_tax_nom, self.readout_cum_tax_real,
                  flow["cumulative_tax_nominal"], flow["cumulative_tax_real"])

        ruined = result.ruin_month
        frac_ruined = float(np.mean((ruined >= 0) & (ruined < month))) if len(ruined) else 0.0
        self.readout_ruin_label.config(text=f"Ruined paths by now: {fmt.pct(frac_ruined)}")

        self._cursor_line_cloud.set_xdata([month / 12.0, month / 12.0])
        self._cursor_line_flows.set_xdata([month / 12.0, month / 12.0])
        self._cursor_canvas.draw_idle()

    # ── Export ─────────────────────────────────────────────────────────────

    def _export_bands(self) -> None:
        if self._last_result is None:
            self.error_label.show("Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="life_strategy_bands.csv",
        )
        if not path:
            return
        result = self._last_result
        n_months = result.values.shape[1]
        years = np.arange(n_months) / 12.0
        nominal_bands = result.percentiles((1, 10, 50, 90, 99), real=False)
        real_bands = result.percentiles((1, 10, 50, 90, 99), real=True)

        flow = result.cash_flow_bands((10, 50, 90))
        # Prepend a zero "month 0" column so every flow series lines up with
        # the wealth bands above (which include t0 as column 0).
        pad = lambda row: np.concatenate([[0.0], row])  # noqa: E731
        contrib_nom = pad(flow["contribution_nominal"][1])
        contrib_real = pad(flow["contribution_real"][1])
        wreq_p10_nom = pad(flow["withdrawal_requested_nominal"][0])
        wreq_p50_nom = pad(flow["withdrawal_requested_nominal"][1])
        wreq_p90_nom = pad(flow["withdrawal_requested_nominal"][2])
        wreq_p50_real = pad(flow["withdrawal_requested_real"][1])
        wreal_p50_nom = pad(flow["withdrawal_realised_nominal"][1])
        wreal_p50_real = pad(flow["withdrawal_realised_real"][1])
        wtax_p50_nom = pad(flow["withdrawal_tax_nominal"][1])
        wtax_p50_real = pad(flow["withdrawal_tax_real"][1])
        wgross_p50_nom = pad(flow["withdrawal_gross_nominal"][1])
        wgross_p50_real = pad(flow["withdrawal_gross_real"][1])

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "year", "month",
                    "p1_nominal", "p10_nominal", "p50_nominal", "p90_nominal", "p99_nominal",
                    "p1_real", "p10_real", "p50_real", "p90_real", "p99_real",
                    "contribution_nominal", "contribution_real",
                    "withdrawal_requested_p10_nominal", "withdrawal_requested_p50_nominal",
                    "withdrawal_requested_p90_nominal", "withdrawal_requested_p50_real",
                    "withdrawal_realised_p50_nominal", "withdrawal_realised_p50_real",
                    "withdrawal_tax_p50_nominal", "withdrawal_tax_p50_real",
                    "withdrawal_gross_p50_nominal", "withdrawal_gross_p50_real",
                ])
                for m in range(n_months):
                    writer.writerow([
                        f"{years[m]:.4f}", m,
                        *[f"{v:.2f}" for v in nominal_bands[:, m]],
                        *[f"{v:.2f}" for v in real_bands[:, m]],
                        f"{contrib_nom[m]:.2f}", f"{contrib_real[m]:.2f}",
                        f"{wreq_p10_nom[m]:.2f}", f"{wreq_p50_nom[m]:.2f}", f"{wreq_p90_nom[m]:.2f}",
                        f"{wreq_p50_real[m]:.2f}",
                        f"{wreal_p50_nom[m]:.2f}", f"{wreal_p50_real[m]:.2f}",
                        f"{wtax_p50_nom[m]:.2f}", f"{wtax_p50_real[m]:.2f}",
                        f"{wgross_p50_nom[m]:.2f}", f"{wgross_p50_real[m]:.2f}",
                    ])
            self.status_label.config(text=f"Exported to {os.path.basename(path)}")
        except OSError as e:
            self.error_label.show(f"Export failed: {e}")
