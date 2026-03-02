#!/usr/bin/env python3
"""
Explore the portfolio weight-space via random or grid search.

Usage:
    python scripts/run_multi.py                  # defaults from engine.config
    python scripts/run_multi.py --random 5000
    python scripts/run_multi.py --grid 0.10
    python scripts/run_multi.py --random 10000 --sims 500 --jobs 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.runner import run_multi_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-bootstrap portfolio search",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", type=int, metavar="N",
                       help="Random search with N portfolios")
    group.add_argument("--grid", type=float, metavar="STEP",
                       help="Grid search with given step size")
    parser.add_argument("--sims", type=int, default=None,
                        help="Monte-Carlo simulations per portfolio")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Horizon in years")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Block-bootstrap block length in months (1=iid)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    kwargs: dict = {}
    if args.random is not None:
        kwargs["method"] = "random"
        kwargs["n_portfolios"] = args.random
    elif args.grid is not None:
        kwargs["method"] = "grid"
        kwargs["grid_step"] = args.grid
    if args.sims is not None:
        kwargs["n_sim"] = args.sims
    if args.horizon is not None:
        kwargs["horizon_years"] = args.horizon
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.jobs is not None:
        kwargs["n_jobs"] = args.jobs
    if args.output is not None:
        kwargs["output_path"] = args.output

    run_multi_bootstrap(**kwargs)


if __name__ == "__main__":
    main()
