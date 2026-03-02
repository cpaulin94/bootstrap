"""
engine — Monte-Carlo bootstrap simulator for multi-asset portfolios.

Public API
----------
    from engine.data    import load_portfolio_csv, preload_returns
    from engine.runner  import run_bootstrap, run_bootstrap_preloaded, run_multi_bootstrap
    from engine.metrics import compute_metrics, shannon_entropy
    from engine.pareto  import compute_pareto
"""

from engine.data import load_portfolio_csv, preload_returns
from engine.runner import run_bootstrap, run_bootstrap_preloaded, run_multi_bootstrap
from engine.metrics import compute_metrics, shannon_entropy
from engine.pareto import compute_pareto

__all__ = [
    "load_portfolio_csv",
    "preload_returns",
    "run_bootstrap",
    "run_bootstrap_preloaded",
    "run_multi_bootstrap",
    "compute_metrics",
    "shannon_entropy",
    "compute_pareto",
]
