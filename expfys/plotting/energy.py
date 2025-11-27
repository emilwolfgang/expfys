"""Energy and angular momentum plots shared in the report."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.lines import Line2D

from ..materials import MATERIAL_COLOR_MAP, MaterialSeries, get_series
from ..summary_io import SummaryTable, load_tables


def _load_tables(part: str, overrides: Sequence[MaterialSeries] | None) -> list[SummaryTable]:
    series = overrides or get_series(part)
    return load_tables(series)


def plot_angular_momentum_part_b(
    ax: plt.Axes | None = None,
    series_list: Sequence[MaterialSeries] | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
    r"""Plot relative change of |L| in the :math:`\hat{n}` direction for part B."""
    tables = _load_tables("B", series_list)
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()

    all_rel = []
    last_x = None
    for table in tables:
        frame = table.frame.copy()
        if {"dt_before_s", "dt_after_s"} <= set(frame.columns):
            dt_before = frame["dt_before_s"].astype(float)
            dt_after = frame["dt_after_s"].astype(float)
            x_vals = np.minimum(dt_before, dt_after)
        else:
            x_vals = np.arange(1, len(frame) + 1)
        last_x = x_vals

        L_pre = frame["L_pre_n_kgm2_s"].astype(float).abs()
        L_post = frame["L_post_n_kgm2_s"].astype(float).abs()
        denom = L_pre.where(L_pre != 0.0, 1.0)
        rel = 100.0 * (L_post - L_pre) / denom
        all_rel.append(rel.to_numpy())

        ax.scatter(x_vals, rel, s=20, alpha=0.7, label=table.label)
        mean_val = float(rel.mean())
        std_val = float(rel.std())
        print(f"Medel relativ förändring i |L| för {table.label}: {mean_val:.2f}%")
        print(f"Standardavvikelse för {table.label}: {std_val:.2f}%")

    if all_rel:
        stacked = np.concatenate(all_rel)
        global_mean = float(stacked.mean())
        global_std = float(stacked.std())
        print(f"Globalt medelvärde (alla material): {global_mean:.2f}%")
        print(f"Global standardavvikelse (alla material): {global_std:.2f}%")
        if last_x is not None:
            ax.hlines(
                global_mean,
                np.min(last_x),
                np.max(last_x),
                colors="black",
                linestyles="--",
                linewidth=1,
            )

    if tables and {"dt_before_s", "dt_after_s"} <= set(tables[0].frame.columns):
        ax.set_xlabel("Minsta tidsfönster före/efter kollision (s)")
    else:
        ax.set_xlabel("Försök nummer")

    ax.set_ylabel(r"Relativ förändring i rörelsemängdsmoment [$\%$]")
    ax.set_title(r"Bevarande av rörelsemängdsmoment i Del B ($\hat{n}$-riktning)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, linestyle=":", linewidth=0.5)

    kontakt_handle = Line2D([], [], linestyle="None", marker="None", label="Kontaktyta:")
    mean_handle = Line2D([], [], color="black", linestyle="--", label="Globalt medelvärde")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            [kontakt_handle, *handles, mean_handle],
            ["Kontaktyta:", *labels, "Globalt medelvärde"],
            fontsize="small",
            loc="best",
        )
    return created_fig, ax


def plot_energy_comparison(
    ax: plt.Axes | None = None,
    part_a_series: Sequence[MaterialSeries] | None = None,
    part_b_series: Sequence[MaterialSeries] | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
    """Plot ``Q_E`` for both parts side by side."""
    tables_a = _load_tables("A", part_a_series)
    tables_b = _load_tables("B", part_b_series)
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()

    offsets = {"A": -0.12, "B": +0.12}
    group_map = {
        table.series.series_key: idx for idx, table in enumerate(tables_a)
    }
    for idx, table in enumerate(tables_b):
        group_map.setdefault(table.series.series_key, idx)

    def _plot_group(tables: Sequence[SummaryTable], group_key: str):
        for table in tables:
            frame = table.frame.copy()
            if {"KE_sys_pre", "KE_sys_post"} <= set(frame.columns):
                pre = frame["KE_sys_pre"].astype(float)
                post = frame["KE_sys_post"].astype(float)
            else:
                pre = frame["KE_pre_tot_J"].astype(float)
                post = frame["KE_post_tot_J"].astype(float)
            denom = pre.replace(0.0, np.nan)
            energy_frac = 100.0 * (post - pre) / denom
            energy_frac = energy_frac.fillna(0.0)

            group_idx = group_map.get(table.series.series_key, 0)
            x_pos = group_idx + offsets[group_key]
            x_vals = np.full(len(energy_frac), x_pos, dtype=float)

            ax.scatter(
                x_vals,
                energy_frac,
                s=60,
                alpha=0.8,
                marker="o",
                facecolors="none",
                edgecolors=table.color or MATERIAL_COLOR_MAP.get(table.label, "black"),
            )

            mean_val = float(energy_frac.mean())
            ax.hlines(mean_val, x_pos - 0.2, x_pos + 0.2, colors=[table.color], linestyles="--", linewidth=1)
            print(f"Medel relativ förändring i energi för {table.label} ({group_key}): {mean_val:.2f}%")

    _plot_group(tables_a, "A")
    _plot_group(tables_b, "B")

    tick_positions = [
        0 + offsets["A"],
        0 + offsets["B"],
        1 + offsets["A"],
        1 + offsets["B"],
        2 + offsets["A"],
        2 + offsets["B"],
    ]
    tick_labels = [
        "Aluminium (A)",
        "Plast (B)",
        "Kardborreband (A)",
        "Kardborreband (B)",
        "Skumgummi (A)",
        "Skumgummi (B)",
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    ax.set_xlabel("Mätserie")
    ax.set_ylabel(r"$Q_E$ [$\%$]")
    ax.set_title("Relativ förändring i kinetisk energi i Del A och Del B")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, linestyle=":", linewidth=0.5)

    point_handle = Line2D([], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor="black", linestyle="None", label="Mätvärde")
    mean_handle = Line2D([], [], color="black", linestyle="--", label="Medelvärde")
    ax.legend(handles=[point_handle, mean_handle], fontsize="small", loc="best")
    plt.subplots_adjust(bottom=0.32)
    return created_fig, ax


__all__ = [
    "plot_angular_momentum_part_b",
    "plot_energy_comparison",
]
