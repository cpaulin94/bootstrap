#!/usr/bin/env python3
"""
Run a single-portfolio Monte-Carlo bootstrap.

Usage:  python scripts/run_single.py sample_portfolio.csv
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data import load_portfolio_csv
from engine.runner import run_bootstrap


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_single.py <portfolio.csv>")
        sys.exit(1)

    pf_path = sys.argv[1]
    portfolio = load_portfolio_csv(pf_path)
    print(f"Portfolio (normalised): {portfolio}")

    results = run_bootstrap(portfolio)

    print("\n── Metrics ─────────────────────────────────────────")
    for k, v in sorted(results.items()):
        print(f"  {k:42s}: {v}")


if __name__ == "__main__":
    main()
