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


def _date_range_mask(
    dates: list[tuple[int, int]],
    date_start: Optional[str],
    date_end: Optional[str],
) -> Optional[np.ndarray]:
    """Boolean mask of rows within [date_start, date_end], or ``None`` if
    both bounds are unset (meaning: keep everything, no mask needed)."""
    if date_start is None and date_end is None:
        return None
    lo = _ym_key(date_start) if date_start else (0, 0)
    hi = _ym_key(date_end) if date_end else (9999, 12)
    return np.array([lo <= d <= hi for d in dates], dtype=bool)


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
    mask = _date_range_mask(dates, date_start, date_end)
    if mask is None:
        log.debug("[FILTER] No date filter applied, keeping all %d rows", len(returns))
        return returns

    log.info("[FILTER] Applying date filter [%s, %s] on %d rows",
             date_start or "*", date_end or "*", len(returns))
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


def _load_and_filter(
    ticker: str,
    use_after_ter: bool,
    date_start: Optional[str],
    date_end: Optional[str],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Load one ticker and apply the date-range filter to BOTH the dates and
    the returns (``apply_date_filter`` only returns the returns array, which
    is enough for independent-resampling callers but not for
    :func:`align_on_common_dates`, which needs to know which calendar month
    each remaining row belongs to)."""
    dates, ret_arr = load_returns(ticker, use_after_ter)
    mask = _date_range_mask(dates, date_start, date_end)
    if mask is None:
        return dates, ret_arr
    filtered_dates = [d for d, m in zip(dates, mask) if m]
    filtered_ret = ret_arr[mask]
    if len(filtered_ret) == 0:
        log.error("[FILTER] DATE FILTER LEFT 0 ROWS for %s! Available range: %s … %s",
                  ticker, dates[0], dates[-1])
        raise ValueError(
            f"Date filter [{date_start}, {date_end}] left 0 rows for {ticker} — "
            f"available range is {dates[0]} … {dates[-1]}"
        )
    return filtered_dates, filtered_ret


def align_on_common_dates(
    per_ticker: dict[str, tuple[list[tuple[int, int]], np.ndarray]],
    tickers: list[str],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Column-stack returns on the INTERSECTION of the tickers' calendar months.

    Aligning by row position (the old behaviour: keep each series' last
    ``min_len`` rows) is only correct if every series ends on the same
    month. They don't — one ticker's history can end 33 months before
    another's — so a row-tail alignment silently pairs different calendar
    months across assets, destroying the cross-asset correlation the whole
    bootstrap depends on. Aligning on the literal intersection of
    (year, month) keys is the fix.

    Parameters
    ----------
    per_ticker : ticker -> (dates, returns), same length pairs, as returned
        by :func:`_load_and_filter`.
    tickers    : the (already sorted) column order to emit.

    Returns
    -------
    ret_matrix : (n_common_months, n_assets) array, one column per ticker
        in *tickers* order, rows in ascending calendar-month order.
    months     : the sorted list of (year, month) keys used — the common
        history window, for logging / display.
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


def load_all_returns(
    portfolio: dict[str, float],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load return arrays for every ticker; align on the intersection of
    their calendar months (see :func:`align_on_common_dates`).

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
    returns : (n_months, n_assets) array — each column is one asset, rows
        are the calendar months common to every ticker (ascending)
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
    raw: dict[str, tuple[list[tuple[int, int]], np.ndarray]] = {}
    for t in tickers:
        dates, ret_arr = _load_and_filter(t, use_after_ter, date_start, date_end)
        raw[t] = (dates, ret_arr)
        log.info("[LOAD_ALL] %s → %d months after filter", t, len(ret_arr))

    log.info("[LOAD_ALL] Per-asset lengths: %s", {t: len(v[1]) for t, v in raw.items()})
    ret_matrix, months = align_on_common_dates(raw, tickers)
    log.info("[LOAD_ALL] Aligned on common calendar months: %d months (%04d-%02d … %04d-%02d)",
             len(months), months[0][0], months[0][1], months[-1][0], months[-1][1])
    if len(months) < 24:
        log.warning("[LOAD_ALL] Very short common history (%d months) — results may be unreliable",
                    len(months))
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
    """Pre-load return data for a set of tickers, aligned on the
    intersection of their calendar months (see :func:`align_on_common_dates`).

    Use once, then call ``run_bootstrap_preloaded`` many times with
    different weight vectors (avoids repeated CSV I/O).

    Parameters
    ----------
    date_start, date_end : ``"YYYY-MM"`` bounds (inclusive) or ``None``.
        Defaults are resolved from ``engine.config`` at call time.

    Returns
    -------
    sorted_tickers : list[str]   — alphabetically sorted
    ret_matrix     : (n_months, n_assets) array, rows are the calendar
        months common to every ticker (ascending)
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
    raw: dict[str, tuple[list[tuple[int, int]], np.ndarray]] = {}
    for t in tickers_sorted:
        dates, ret_arr = _load_and_filter(t, use_after_ter, date_start, date_end)
        raw[t] = (dates, ret_arr)
        log.info("[PRELOAD] %s → %d months after filter", t, len(ret_arr))

    log.info("[PRELOAD] Per-asset lengths: %s", {t: len(v[1]) for t, v in raw.items()})
    ret_matrix, months = align_on_common_dates(raw, tickers_sorted)
    log.info("[PRELOAD] Aligned on common calendar months: %d months (%04d-%02d … %04d-%02d)",
             len(months), months[0][0], months[0][1], months[-1][0], months[-1][1])
    if len(months) < 24:
        log.warning("[PRELOAD] Very short common history (%d months) — results may be unreliable",
                    len(months))
    elapsed = time.perf_counter() - t0
    log.info("[PRELOAD] Return matrix shape: %s  elapsed: %.3fs",
             ret_matrix.shape, elapsed)
    return tickers_sorted, ret_matrix


def load_portfolios_on_common_window(
    portfolios: dict[str, dict[str, float]],
    use_after_ter: bool = True,
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Load several portfolios onto ONE shared historical window.

    The window is the calendar-month intersection across the UNION of
    every portfolio's tickers, so each portfolio is evaluated on exactly
    the same months as the others.

    This matters whenever portfolios are ranked against each other. Given
    their own windows, a portfolio built from long-history assets is
    measured over a different (usually longer, and differently-composed)
    period than one holding a late-starting asset — so part of any
    performance gap is the PERIOD, not the portfolio, and the tool
    silently rewards holding assets with more history. Measured on this
    repo's data, a 2-asset portfolio scored 7.90% vs a 12-asset one at
    7.70% on their own windows (449 vs 305 months), but 6.09% vs 7.70% on
    the shared 305-month window — the ranking REVERSES.

    Contrast with :func:`load_all_returns`, which is still correct for
    evaluating ONE portfolio on its own terms: there, nothing is being
    ranked, so maximising that portfolio's usable history is the right
    default. (Forcing every single-portfolio computation onto a fixed
    universe-wide window was tried and reverted — see AUDIT.md M9 — since
    it truncates history using assets the portfolio doesn't even hold.)

    Returns
    -------
    tickers    : list[str] — the union, alphabetically sorted
    ret_matrix : (n_months, n_tickers) array on the shared window
    weights    : {portfolio_name: (n_tickers,) array}, zero-padded and
        aligned to *tickers*
    """
    if date_start is None:
        date_start = cfg.DATE_START
    if date_end is None:
        date_end = cfg.DATE_END

    union = sorted({t for pf in portfolios.values() for t in pf})
    if not union:
        raise ValueError("No tickers across the given portfolios.")

    log.info("[COMMON_WINDOW] %d portfolios, %d distinct tickers: %s",
             len(portfolios), len(union), union)
    raw: dict[str, tuple[list[tuple[int, int]], np.ndarray]] = {}
    for t in union:
        raw[t] = _load_and_filter(t, use_after_ter, date_start, date_end)

    ret_matrix, months = align_on_common_dates(raw, union)
    log.info("[COMMON_WINDOW] Shared window: %d months (%04d-%02d … %04d-%02d)",
             len(months), months[0][0], months[0][1], months[-1][0], months[-1][1])
    if len(months) < 24:
        log.warning("[COMMON_WINDOW] Very short shared history (%d months) — "
                    "results may be unreliable", len(months))

    weights = {
        name: np.array([pf.get(t, 0.0) for t in union], dtype=np.float64)
        for name, pf in portfolios.items()
    }
    return union, ret_matrix, weights


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
