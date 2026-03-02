"""
engine.data — Data loading utilities.

Functions for reading portfolio CSVs, loading preprocessed monthly returns,
and pre-loading return matrices for high-throughput bootstrap.
"""

from __future__ import annotations

import csv
import os
import warnings

import numpy as np

from engine import config as cfg


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

def _load_returns(ticker: str, use_after_ter: bool) -> np.ndarray:
    """Load monthly returns for *ticker* from its standard CSV.

    Returns a 1-D float64 array of all non-empty return rows.
    """
    col = "month_return_after_TER" if use_after_ter else "month_return"
    path = os.path.join(cfg.STANDARD_DIR, f"{ticker}.csv")
    returns: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row[col].strip()
            if val == "":
                continue
            returns.append(float(val))
    return np.array(returns, dtype=np.float64)


def load_all_returns(
    portfolio: dict[str, float],
    use_after_ter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load return arrays for every ticker; align to shortest history.

    Returns
    -------
    weights : (n_assets,) array — in alphabetical ticker order
    returns : (n_months, n_assets) array — each column is one asset
    """
    tickers = sorted(portfolio.keys())
    raw = {t: _load_returns(t, use_after_ter) for t in tickers}
    min_len = min(len(v) for v in raw.values())
    ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers])
    weights = np.array([portfolio[t] for t in tickers], dtype=np.float64)
    return weights, ret_matrix


def preload_returns(
    tickers: list[str],
    use_after_ter: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Pre-load return data for a set of tickers.

    Use once, then call ``run_bootstrap_preloaded`` many times with
    different weight vectors (avoids repeated CSV I/O).

    Returns
    -------
    sorted_tickers : list[str]   — alphabetically sorted
    ret_matrix     : (n_months, n_assets) array
    """
    tickers_sorted = sorted(tickers)
    raw = {t: _load_returns(t, use_after_ter) for t in tickers_sorted}
    min_len = min(len(v) for v in raw.values())
    ret_matrix = np.column_stack([raw[t][-min_len:] for t in tickers_sorted])
    return tickers_sorted, ret_matrix
