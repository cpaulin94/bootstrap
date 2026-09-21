"""bootstrap_gui.assets — the only place the GUI touches engine.data directly.

Small helpers shared by every section: enumerating available tickers,
computing the overlapping date range for a set of tickers, and building
the metric dropdown list from the current engine config.
"""

from __future__ import annotations

import csv
import os

from engine import config as cfg
from engine.data import parse_month_year


def get_available_assets() -> list[dict]:
    """Enumerate assets in data/standard/, read date ranges."""
    assets = []
    for fname in sorted(os.listdir(cfg.STANDARD_DIR)):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv", "")
        path = os.path.join(cfg.STANDARD_DIR, fname)
        first_date = last_date = None
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    first_date = rows[0]["month_year"].strip()
                    last_date = rows[-1]["month_year"].strip()
        except Exception:
            pass
        assets.append(
            {"ticker": ticker, "first_date": first_date or "?", "last_date": last_date or "?"}
        )
    return assets


def compute_date_intersection(tickers: list[str]) -> tuple[str, str]:
    """Compute the intersection of date ranges for selected tickers."""
    if not tickers:
        return ("", "")
    latest_start = (0, 0)
    earliest_end = (9999, 12)
    for t in tickers:
        path = os.path.join(cfg.STANDARD_DIR, f"{t}.csv")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    s = parse_month_year(rows[0]["month_year"])
                    e = parse_month_year(rows[-1]["month_year"])
                    if s > latest_start:
                        latest_start = s
                    if e < earliest_end:
                        earliest_end = e
        except Exception:
            pass
    if latest_start > earliest_end:
        return ("N/A", "N/A")
    return (
        f"{latest_start[0]:04d}-{latest_start[1]:02d}",
        f"{earliest_end[0]:04d}-{earliest_end[1]:02d}",
    )


def clean_portfolio(portfolio: dict) -> dict:
    """Remove zero-weight assets from a portfolio.

    Root cause fix for NumPy matmul warnings: zero-weight entries cause
    inf*0 or nan*0 inside BLAS routines, triggering spurious
    'divide by zero / overflow / invalid value in matmul' warnings.
    Filtering them out is mathematically equivalent and eliminates
    the warnings at the source.
    """
    return {t: w for t, w in portfolio.items() if w > 1e-9}


def overlay_key(
    params: dict,
    sorted_tickers: list[str] | None,
    sim_seed: int | None,
) -> tuple:
    """Identity of the context an overlay's metrics were computed in.

    Two overlays are only comparable with the same cloud if this key
    matches. Simulation params alone are NOT enough: the historical date
    window (derived from the FULL set of search-space tickers, via
    ``sorted_tickers``) and the Monte-Carlo draw (``sim_seed``) shift an
    overlay by far more than the difference between nearby portfolios —
    an overlay computed against a different ticker set or a different
    seed can land tens of standard deviations from a cloud it looks like
    it should sit inside.
    """
    return (
        tuple(sorted(params.items())),
        tuple(sorted(sorted_tickers)) if sorted_tickers else None,
        sim_seed,
    )


def default_pareto_direction(metric_name: str) -> str:
    """Infer whether a metric should be maximised or minimised on a Pareto
    frontier, from its naming convention (see :func:`build_metric_list`):
    returns and diversification counts are "more is better", every
    volatility/drawdown metric is "less is better".

    Used to seed the Pareto-objective checkboxes' default direction —
    every name `build_metric_list` can produce is covered, so this never
    falls through to a guess for a metric the app actually reports.
    """
    if metric_name.startswith("annualised_return") or metric_name.startswith("effective_n_"):
        return "maximize"
    return "minimize"  # volatility_*, max_dd_*, mda_*


def build_metric_list() -> list[str]:
    """Build the metric dropdown list dynamically from engine config.

    This ensures names always match what `compute_metrics` produces,
    regardless of BAD_PERCENTILE or VOLATILITY_WINDOWS settings.
    """
    bp = str(cfg.BAD_PERCENTILE)
    return (
        [f"annualised_return_p{p}" for p in cfg.RETURN_PERCENTILES]
        + [f"volatility_{w}y" for w in cfg.VOLATILITY_WINDOWS]
        + [
            f"max_dd_depth_p{bp}",
            f"max_dd_length_months_p{bp}",
            f"mda_months_p{bp}",
        ]
        + ["effective_n_assets"]
        + ["effective_n_types"]
    )
