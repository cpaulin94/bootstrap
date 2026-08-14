"""
engine.data — Data loading utilities.

Functions for reading portfolio CSVs, loading preprocessed monthly returns,
and pre-loading return matrices for high-throughput bootstrap.
"""

from __future__ import annotations

import csv
import logging
import os
import time
import warnings
from typing import Optional

import numpy as np

from engine import config as cfg

log = logging.getLogger("bootstrap.data")

# ═══════════════════════════════════════════════════════════════════════════════
# TER / TYPE table
# ═══════════════════════════════════════════════════════════════════════════════

_TYPE_MAP_CACHE: dict[str, str] | None = None


def load_type_map() -> dict[str, str]:
    """Load ticker → TYPE mapping from TER_table.csv.  Cached after first call."""
    global _TYPE_MAP_CACHE
    if _TYPE_MAP_CACHE is not None:
        return _TYPE_MAP_CACHE
    type_map: dict[str, str] = {}
    with open(cfg.TER_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        type_idx = None
        for i, col in enumerate(header):
            if col.strip().upper() == "TYPE":
                type_idx = i
                break
        if type_idx is None:
            log.warning("[LOAD_TYPE] No TYPE column in %s", cfg.TER_FILE)
            _TYPE_MAP_CACHE = {}
            return _TYPE_MAP_CACHE
        for row in reader:
            if len(row) <= type_idx:
                continue
            ticker = row[0].strip().upper()
            asset_type = row[type_idx].strip().upper()
            if ticker and asset_type:
                type_map[ticker] = asset_type
    _TYPE_MAP_CACHE = type_map
    log.info("[LOAD_TYPE] Loaded type map: %s", type_map)
    return _TYPE_MAP_CACHE


# ═══════════════════════════════════════════════════════════════════════════════
# Date helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_month_year(s: str) -> tuple[int, int]:
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

def load_returns(
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
    log.info("[LOAD] Reading %s  col=%s  file=%s", ticker, col, path)
    if not os.path.exists(path):
        log.error("[LOAD] FILE NOT FOUND: %s", path)
        raise FileNotFoundError(f"No data file for ticker {ticker}: {path}")
    t0 = time.perf_counter()
    dates: list[tuple[int, int]] = []
    returns: list[float] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        available_cols = reader.fieldnames
        if col not in (available_cols or []):
            log.error("[LOAD] Column '%s' not found in %s. Available: %s",
                      col, path, available_cols)
            raise KeyError(f"Column '{col}' missing from {path}")
        for row in reader:
            val = row[col].strip()
            if val == "":
                skipped += 1
                continue
            dates.append(parse_month_year(row["month_year"]))
            returns.append(float(val))
    elapsed = time.perf_counter() - t0
    ret_arr = np.array(returns, dtype=np.float64)
    log.info("[LOAD] %s: %d rows loaded, %d skipped (empty), "
             "date range %s → %s, elapsed %.3fs",
             ticker, len(returns), skipped,
             f"{dates[0][0]:04d}-{dates[0][1]:02d}" if dates else "?",
             f"{dates[-1][0]:04d}-{dates[-1][1]:02d}" if dates else "?",
             elapsed)
    if len(returns) == 0:
        log.error("[LOAD] %s: 0 valid rows — check CSV content!", ticker)
    return dates, ret_arr


def apply_date_filter(
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
        log.debug("[FILTER] No date filter applied, keeping all %d rows", len(returns))
        return returns

    lo = _ym_key(date_start) if date_start else (0, 0)
    hi = _ym_key(date_end) if date_end else (9999, 12)
    log.info("[FILTER] Applying date filter [%s, %s] on %d rows",
             date_start or "*", date_end or "*", len(returns))

    mask = np.array([lo <= d <= hi for d in dates], dtype=bool)
    filtered = returns[mask]
    log.info("[FILTER] After filter: %d / %d rows kept", len(filtered), len(returns))
    if len(filtered) == 0:
        log.error("[FILTER] DATE FILTER LEFT 0 ROWS! "
                  "Available range: %s … %s", dates[0], dates[-1])
        raise ValueError(
            f"Date filter [{date_start}, {date_end}] left 0 rows — "
            f"available range is {dates[0]} … {dates[-1]}"
        )
    return filtered


def load_all_returns(
    portfolio: dict[str, float],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load return arrays for every ticker; align to shortest history.

    Parameters
    ----------
    date_start, date_end : ``"YYYY-MM"`` bounds (inclusive) or ``None``.
        When set, only historical rows within this window are kept.
        Defaults are resolved from ``engine.config`` at call time (not at
        import time), so runtime changes to ``cfg.DATE_START`` / ``cfg.DATE_END``
        take effect.

    Returns
    -------
    weights : (n_assets,) array — in alphabetical ticker order
    returns : (n_months, n_assets) array — each column is one asset
    """
    if date_start is None:
        date_start = cfg.DATE_START
    if date_end is None:
        date_end = cfg.DATE_END
    tickers = sorted(portfolio.keys())
    log.info("[LOAD_ALL] Loading %d assets: %s", len(tickers), tickers)
    log.info("[LOAD_ALL] use_after_ter=%s  date_start=%s  date_end=%s",
             use_after_ter, date_start, date_end)
    t0 = time.perf_counter()
    raw: dict[str, np.ndarray] = {}
    for t in tickers:
        dates, ret_arr = load_returns(t, use_after_ter)
        raw[t] = apply_date_filter(dates, ret_arr, date_start, date_end)
        log.info("[LOAD_ALL] %s → %d months after filter", t, len(raw[t]))

    lengths = {t: len(v) for t, v in raw.items()}
    min_len = min(lengths.values())
    log.info("[LOAD_ALL] Per-asset lengths: %s", lengths)
    log.info("[LOAD_ALL] Aligning to shortest history: %d months", min_len)
    if min_len < 24:
        log.warning("[LOAD_ALL] Very short history (%d months) — results may be unreliable", min_len)
    ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers])
    weights = np.array([portfolio[t] for t in tickers], dtype=np.float64)
    elapsed = time.perf_counter() - t0
    log.info("[LOAD_ALL] Return matrix shape: %s  weights: %s  elapsed: %.3fs",
             ret_matrix.shape, weights, elapsed)
    return weights, ret_matrix


def preload_returns(
    tickers: list[str],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> tuple[list[str], np.ndarray]:
    """Pre-load return data for a set of tickers.

    Use once, then call ``run_bootstrap_preloaded`` many times with
    different weight vectors (avoids repeated CSV I/O).

    Parameters
    ----------
    date_start, date_end : ``"YYYY-MM"`` bounds (inclusive) or ``None``.
        Defaults are resolved from ``engine.config`` at call time.

    Returns
    -------
    sorted_tickers : list[str]   — alphabetically sorted
    ret_matrix     : (n_months, n_assets) array
    """
    if date_start is None:
        date_start = cfg.DATE_START
    if date_end is None:
        date_end = cfg.DATE_END
    tickers_sorted = sorted(tickers)
    log.info("[PRELOAD] Pre-loading %d assets: %s", len(tickers_sorted), tickers_sorted)
    log.info("[PRELOAD] use_after_ter=%s  date_start=%s  date_end=%s",
             use_after_ter, date_start, date_end)
    t0 = time.perf_counter()
    raw: dict[str, np.ndarray] = {}
    for t in tickers_sorted:
        dates, ret_arr = load_returns(t, use_after_ter)
        raw[t] = apply_date_filter(dates, ret_arr, date_start, date_end)
        log.info("[PRELOAD] %s → %d months after filter", t, len(raw[t]))

    lengths = {t: len(v) for t, v in raw.items()}
    min_len = min(lengths.values())
    log.info("[PRELOAD] Per-asset lengths: %s", lengths)
    log.info("[PRELOAD] Aligning to shortest history: %d months", min_len)
    if min_len < 24:
        log.warning("[PRELOAD] Very short history (%d months) — results may be unreliable", min_len)
    ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers_sorted])
    elapsed = time.perf_counter() - t0
    log.info("[PRELOAD] Return matrix shape: %s  elapsed: %.3fs",
             ret_matrix.shape, elapsed)
    return tickers_sorted, ret_matrix


def load_independent_returns(
    portfolio: dict[str, float],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load per-asset return arrays **without** aligning to a common date range.

    Each asset keeps its full available history (after optional date filter).
    Used by the independent-resampling ("break correlations") bootstrap.
    Date-bound defaults are resolved from ``engine.config`` at call time.

    Returns
    -------
    weights      : (n_assets,) array — in alphabetical ticker order
    returns_list : list of 1-D arrays, one per asset (lengths may differ)
    """
    if date_start is None:
        date_start = cfg.DATE_START
    if date_end is None:
        date_end = cfg.DATE_END
    tickers = sorted(portfolio.keys())
    log.info("[LOAD_IND] Loading %d assets independently: %s", len(tickers), tickers)
    t0 = time.perf_counter()
    returns_list: list[np.ndarray] = []
    for t in tickers:
        dates, ret_arr = load_returns(t, use_after_ter)
        filtered = apply_date_filter(dates, ret_arr, date_start, date_end)
        log.info("[LOAD_IND] %s → %d months", t, len(filtered))
        returns_list.append(filtered)
    weights = np.array([portfolio[t] for t in tickers], dtype=np.float64)
    elapsed = time.perf_counter() - t0
    log.info("[LOAD_IND] Loaded in %.3fs  per-asset lengths: %s",
             elapsed, [len(r) for r in returns_list])
    return weights, returns_list


def preload_independent_returns(
    tickers: list[str],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Pre-load per-asset returns without alignment (independent mode).

    Date-bound defaults are resolved from ``engine.config`` at call time.

    Returns
    -------
    sorted_tickers : list[str]
    returns_list   : list of 1-D arrays (lengths may differ)
    """
    if date_start is None:
        date_start = cfg.DATE_START
    if date_end is None:
        date_end = cfg.DATE_END
    tickers_sorted = sorted(tickers)
    log.info("[PRELOAD_IND] Pre-loading %d assets independently: %s",
             len(tickers_sorted), tickers_sorted)
    t0 = time.perf_counter()
    returns_list: list[np.ndarray] = []
    for t in tickers_sorted:
        dates, ret_arr = load_returns(t, use_after_ter)
        filtered = apply_date_filter(dates, ret_arr, date_start, date_end)
        log.info("[PRELOAD_IND] %s → %d months", t, len(filtered))
        returns_list.append(filtered)
    elapsed = time.perf_counter() - t0
    log.info("[PRELOAD_IND] Loaded in %.3fs  per-asset lengths: %s",
             elapsed, [len(r) for r in returns_list])
    return tickers_sorted, returns_list


# ═══════════════════════════════════════════════════════════════════════════════
# Deprecated aliases
# ═══════════════════════════════════════════════════════════════════════════════
# These private names used to be the only way to call these functions and were
# imported directly by gui.py. They are kept as aliases for backwards
# compatibility; new code should use the public names above.

_parse_month_year = parse_month_year
_load_returns = load_returns
_apply_date_filter = apply_date_filter
