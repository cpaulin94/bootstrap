"""
engine.data — Data loading utilities.

Functions for reading portfolio CSVs, loading preprocessed monthly returns,
and pre-loading return matrices for high-throughput bootstrap.
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Optional

import numpy as np

from engine import config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Date helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_month_year(s: str) -> tuple[int, int]:
    """Parse ``"MM/YYYY"`` → ``(year, month)``."""
    parts = s.strip().split("/")
    return int(parts[1]), int(parts[0])


def _ym_key(s: str) -> tuple[int, int]:
    """Parse ``"YYYY-MM"`` → ``(year, month)``."""
    parts = s.strip().split("-")
    return int(parts[0]), int(parts[1])


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio CSV
# ═══════════════════════════════════════════════════════════════════════════════

def load_portfolio_csv(path: str) -> dict[str, float]:
    """Load a portfolio CSV (TICKER, PCT) and return normalised weights."""
    weights: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            ticker = row[0].strip().upper()
            pct = float(row[1].strip())
            weights[ticker] = pct

    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-4):
        warnings.warn(
            f"Portfolio weights sum to {total:.6f}, not 1.0. "
            f"Normalising to keep proportions."
        )
        for t in weights:
            weights[t] /= total

    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# Monthly-return loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_returns(
    ticker: str,
    use_after_ter: bool,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Load monthly returns for *ticker* from its standard CSV.

    Returns
    -------
    dates   : list of (year, month) tuples — one per valid row
    returns : 1-D float64 array of returns (same length as *dates*)
    """
    col = "month_return_after_TER" if use_after_ter else "month_return"
    path = os.path.join(cfg.STANDARD_DIR, f"{ticker}.csv")
    dates: list[tuple[int, int]] = []
    returns: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row[col].strip()
            if val == "":
                continue
            dates.append(_parse_month_year(row["month_year"]))
            returns.append(float(val))
    return dates, np.array(returns, dtype=np.float64)


def _apply_date_filter(
    dates: list[tuple[int, int]],
    returns: np.ndarray,
    date_start: Optional[str],
    date_end: Optional[str],
) -> np.ndarray:
    """Slice *returns* to rows whose date falls within [date_start, date_end].

    Parameters
    ----------
    dates      : per-row (year, month) tuples
    returns    : 1-D array, same length as *dates*
    date_start : ``"YYYY-MM"`` inclusive lower bound, or ``None``
    date_end   : ``"YYYY-MM"`` inclusive upper bound, or ``None``
    """
    if date_start is None and date_end is None:
        return returns

    lo = _ym_key(date_start) if date_start else (0, 0)
    hi = _ym_key(date_end) if date_end else (9999, 12)

    mask = np.array([lo <= d <= hi for d in dates], dtype=bool)
    filtered = returns[mask]
    if len(filtered) == 0:
        raise ValueError(
            f"Date filter [{date_start}, {date_end}] left 0 rows — "
            f"available range is {dates[0]} … {dates[-1]}"
        )
    return filtered


def load_all_returns(
    portfolio: dict[str, float],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = cfg.DATE_START,
    date_end: Optional[str] = cfg.DATE_END,
) -> tuple[np.ndarray, np.ndarray]:
    """Load return arrays for every ticker; align to shortest history.

    Parameters
    ----------
    date_start, date_end : ``"YYYY-MM"`` bounds (inclusive) or ``None``.
        When set, only historical rows within this window are kept.

    Returns
    -------
    weights : (n_assets,) array — in alphabetical ticker order
    returns : (n_months, n_assets) array — each column is one asset
    """
    tickers = sorted(portfolio.keys())
    raw: dict[str, np.ndarray] = {}
    for t in tickers:
        dates, ret_arr = _load_returns(t, use_after_ter)
        raw[t] = _apply_date_filter(dates, ret_arr, date_start, date_end)

    min_len = min(len(v) for v in raw.values())
    ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers])
    weights = np.array([portfolio[t] for t in tickers], dtype=np.float64)
    return weights, ret_matrix


def preload_returns(
    tickers: list[str],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = cfg.DATE_START,
    date_end: Optional[str] = cfg.DATE_END,
) -> tuple[list[str], np.ndarray]:
    """Pre-load return data for a set of tickers.

    Use once, then call ``run_bootstrap_preloaded`` many times with
    different weight vectors (avoids repeated CSV I/O).

    Parameters
    ----------
    date_start, date_end : ``"YYYY-MM"`` bounds (inclusive) or ``None``.

    Returns
    -------
    sorted_tickers : list[str]   — alphabetically sorted
    ret_matrix     : (n_months, n_assets) array
    """
    tickers_sorted = sorted(tickers)
    raw: dict[str, np.ndarray] = {}
    for t in tickers_sorted:
        dates, ret_arr = _load_returns(t, use_after_ter)
        raw[t] = _apply_date_filter(dates, ret_arr, date_start, date_end)

    min_len = min(len(v) for v in raw.values())
    ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers_sorted])
    return tickers_sorted, ret_matrix
