"""
engine — Monte-Carlo bootstrap simulator for multi-asset portfolios.

Public API
----------
    from engine.data    import load_portfolio_csv, preload_returns
    from engine.runner  import (run_bootstrap, run_bootstrap_preloaded,
                                run_multi_bootstrap, run_multi_streaming)
    from engine.metrics import compute_metrics, shannon_entropy
    from engine.pareto  import compute_pareto
"""

from engine.data import load_portfolio_csv, preload_returns
from engine.runner import (
    run_bootstrap,
    run_bootstrap_preloaded,
    run_multi_bootstrap,
    run_multi_streaming,
)
from engine.metrics import compute_metrics, shannon_entropy
from engine.pareto import compute_pareto
from engine.lifecycle import (
    LifePlan,
    LifeSimResult,
    LumpSum,
    Phase,
    PhaseKind,
    simulate_life_strategy,
)

__all__ = [
    "load_portfolio_csv",
    "preload_returns",
    "run_bootstrap",
    "run_bootstrap_preloaded",
    "run_multi_bootstrap",
    "run_multi_streaming",
    "compute_metrics",
    "shannon_entropy",
    "compute_pareto",
    "LifePlan",
    "LifeSimResult",
    "LumpSum",
    "Phase",
    "PhaseKind",
    "simulate_life_strategy",
]
