"""Shared Matplotlib styling utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Width (inches) applied to every figure. Height may vary per plot.
FIG_WIDTH_IN = 10.0


def set_fixed_figwidth(fig: plt.Figure, width: float = FIG_WIDTH_IN) -> plt.Figure:
    """Force a consistent figure width while keeping the current height."""
    fig.set_figwidth(width)
    return fig


__all__ = ["FIG_WIDTH_IN", "set_fixed_figwidth"]
