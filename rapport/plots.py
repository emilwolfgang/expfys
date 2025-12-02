"""Entry point used inside the report notebook to create all figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expfys.plotting import (
    plot_angular_momentum_part_b,
    plot_energy_comparison,
    plot_relative_momentum_part_a,
    plot_relative_momentum_part_b,
)
from expfys.plotting.style import FIG_WIDTH_IN, set_fixed_figwidth

plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 20,
    }
)


def main() -> None:
    """Generate the exact figures used in the report."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 4))
    set_fixed_figwidth(fig)
    plot_relative_momentum_part_a(axes[0])
    axes[0].set_title("Del A - Endimensionell rörelse")
    plot_relative_momentum_part_b(axes[1])
    axes[1].set_title("Del B - Tvådimensionell rörelse")
    fig.suptitle("Förändring av rörelsemängd i rörelsens riktning", fontsize=20)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    set_fixed_figwidth(fig)
    plot_angular_momentum_part_b(ax)
    plt.show()

    fig, ax = plt.subplots()
    set_fixed_figwidth(fig)
    plot_energy_comparison(ax)
    plt.show()


if __name__ == "__main__":
    main()
