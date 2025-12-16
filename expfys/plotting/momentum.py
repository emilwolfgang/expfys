"""Plot helpers for relative momentum changes."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..materials import MaterialSeries, get_series
from ..summary_io import SummaryTable, infer_attempt_numbers, load_tables
from .style import percent_formatter, set_fixed_figwidth


def _load_tables(part: str, overrides: Sequence[MaterialSeries] | None) -> list[SummaryTable]:
    series = overrides or get_series(part)
    return load_tables(series)


def plot_relative_momentum_part_a(
    ax: plt.Axes | None = None,
    series_list: Sequence[MaterialSeries] | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
    r"""Scatter plot for ``Q_{p_{\hat{n}}}`` (part A)."""
    tables = _load_tables("A", series_list)
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()
        set_fixed_figwidth(created_fig)

    for table in tables:
        frame = table.frame.copy()
        attempts = infer_attempt_numbers(frame)
        rel_dp_e = 100.0 * frame["rel_dp_e"].astype(float)

        color = table.color
        label = table.label
        ax.scatter(attempts, rel_dp_e, s=20, alpha=0.7, label=label, color=color)

        mean_val = float(rel_dp_e.mean())
        ax.hlines(
            mean_val,
            attempts.min(),
            attempts.max(),
            colors=[color],
            linestyles="--",
            linewidth=1,
        )
        print(f"  medel för {table.path.name} = {mean_val:.2f}%")

    ax.set_xlabel(r"Försök nummer")
    ax.set_ylabel(r"$Q_{p_{\hat{n}}}$ [%]")
    ax.yaxis.set_major_formatter(percent_formatter())
    ax.grid(True, linestyle=":", linewidth=0.5)

    kontakt_handle = Line2D([], [], linestyle="None", marker="None", label="Kontaktyta:")
    mean_handle = Line2D([], [], color="black", linestyle="--", label="Medelvärde")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            [kontakt_handle, *handles, mean_handle],
            ["Kontaktyta:", *labels, "Medelvärde"],
            fontsize="small",
            loc="best",
        )
    return created_fig, ax


def plot_relative_momentum_part_b(
    ax: plt.Axes | None = None,
    series_list: Sequence[MaterialSeries] | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
    r"""Scatter plot for ``Q_{p_{\hat{u}}}`` vs ``Q_{p_{\hat{v}}}`` (part B)."""
    tables = _load_tables("B", series_list)
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()
        set_fixed_figwidth(created_fig)

    relx_all = []
    rely_all = []
    for table in tables:
        frame = table.frame.copy()
        dp_x = frame["dp_u_kgm_s"].astype(float)
        dp_y = frame["dp_v_kgm_s"].astype(float)
        p_pre = frame["p_pre_mag"].abs().replace(0.0, np.nan)

        rel_dp_x = 100.0 * dp_x / p_pre
        rel_dp_y = 100.0 * dp_y / p_pre

        relx_all.append(rel_dp_x.to_numpy())
        rely_all.append(rel_dp_y.to_numpy())

        color = table.color
        label = table.label
        ax.scatter(rel_dp_x, rel_dp_y, s=20, alpha=0.7, label=label, color=color)

        mean_x = float(rel_dp_x.mean())
        mean_y = float(rel_dp_y.mean())
        ax.scatter(mean_x, mean_y, s=60, color=color, edgecolor="black", zorder=5)
        print(f"  medel för {table.path.name} = ({mean_x:.2f}%, {mean_y:.2f}%)")

    if relx_all and rely_all:
        relx_all = np.concatenate(relx_all)
        rely_all = np.concatenate(rely_all)
        data_min = float(min(relx_all.min(), rely_all.min()))
        data_max = float(max(relx_all.max(), rely_all.max()))
        span = data_max - data_min
        margin = 0.05 * span if span > 0 else 1.0
        ax.set_xlim(data_min - margin, data_max + margin)
        ax.set_ylim(data_min - margin, data_max + margin)

    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel(r"$Q_{p_{\hat{u}}}$ [%]")
    ax.set_ylabel(r"$Q_{p_{\hat{v}}}$ [%]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.yaxis.set_major_formatter(percent_formatter())

    kontakt_handle = Line2D([], [], linestyle="None", marker="None", label="Kontaktyta:")
    mean_handle = Line2D([], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor="black", linestyle="None", label="Medelvärde")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            [kontakt_handle, *handles, mean_handle],
            ["Kontaktyta:", *labels, "Medelvärde"],
            fontsize="small",
            loc="best",
        )
    return created_fig, ax


__all__ = [
    "plot_relative_momentum_part_a",
    "plot_relative_momentum_part_b",
]
