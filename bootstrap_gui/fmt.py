"""bootstrap_gui.fmt — one place for every number → string conversion in the UI.

The old code mixed ``.1%``, ``.0%``, ``.4f`` and ``:,`` ad hoc across the
file. This module is the single source of truth so every panel reads the
same way.
"""

from __future__ import annotations


def money(x: float, *, currency: str = "€", decimals: int = 0) -> str:
    """Format e.g. 1234567.5 -> '€1.234.568' (decimals=0) or '€1.234.567,50' (decimals=2)."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    grouped = f"{x:,.{decimals}f}"
    integer_part, _, frac_part = grouped.partition(".")
    integer_part = integer_part.replace(",", ".")
    if decimals > 0:
        return f"{sign}{currency}{integer_part},{frac_part}"
    return f"{sign}{currency}{integer_part}"


def pct(x: float, decimals: int = 1) -> str:
    """Format a fraction (0.123) as '12,3%'."""
    formatted = f"{x * 100:.{decimals}f}".replace(".", ",")
    return f"{formatted}%"


def ratio(x: float, decimals: int = 2) -> str:
    """Format a plain ratio/number, e.g. Shannon entropy: 0.8734 -> '0,87'."""
    return f"{x:.{decimals}f}".replace(".", ",")
