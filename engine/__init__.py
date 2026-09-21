"""
engine — Monte-Carlo bootstrap simulator for multi-asset portfolios.

Public API
----------
    from engine.data    import load_portfolio_csv, preload_returns
    from engine.runner  import (run_bootstrap, run_bootstrap_preloaded,
                                run_multi_bootstrap, run_multi_streaming)
    from engine.metrics import compute_metrics, effective_n_assets
    from engine.pareto  import compute_pareto
    from engine.plotprep import results_to_arrays, thin_scatter
"""

from engine.data import load_portfolio_csv, preload_returns
from engine.runner import (
    run_bootstrap,
    run_bootstrap_preloaded,
    run_multi_bootstrap,
    run_multi_streaming,
)
from engine.metrics import compute_metrics, effective_n_assets
from engine.pareto import compute_pareto
from engine.plotprep import results_to_arrays, thin_scatter
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
    "effective_n_assets",
    "compute_pareto",
    "results_to_arrays",
    "thin_scatter",
    "LifePlan",
    "LifeSimResult",
    "LumpSum",
    "Phase",
    "PhaseKind",
    "simulate_life_strategy",
]
