#!/usr/bin/env python3
"""
QTM TSV analyzer

Usage:
  python main2.py A1(1).tsv A1(2).tsv --m1 0.2 --m2 0.2 --dist-threshold 30.0

Features:
- Robust loader: handles both tab-separated and whitespace-separated TSV exports from QTM,
  including FILE_VERSION style text exports with MARKER_NAMES and FREQUENCY.
- Flexible marker detection: supports columns like `x1,y1,z1` and names like `Marker : X` / `Marker_X`.
- Detects a collision interval between two markers based on a REQUIRED distance threshold and
  computes pre/post velocities, momentum and kinetic energy.
- Produces diagnostic plots of x(t), y(t), z(t) with fitted pre/post curves and the collision window,
  and a 1D projection along the track direction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expfys.dataio import (
    detect_time_series,
    find_markers_xyz_flexible,
    load_qmt_any,
)


# =============================================================================
# Shared data loading utilities are provided by expfys.dataio.qtm
# =============================================================================



def find_collision_time(
    df: pd.DataFrame,
    markers_xyz: Dict[str, Dict[str, str]],
    marker1: str | None,
    marker2: str | None,
    dist_threshold: float,
    max_gap_frames: int = 3,
) -> dict:
    """Find collision interval [t_p_minus, t_p_plus] where distance between two markers < dist_threshold.

    The interval is defined by contiguous frames where the distance is below the threshold,
    with short gaps (≤ max_gap_frames) filled in.
    """
    if len(markers_xyz) < 2:
        raise ValueError("Need at least two markers with X,Y,Z to compute collision time.")

    bases = list(markers_xyz.keys())
    if marker1 is None or marker2 is None:
        marker1 = bases[0]
        marker2 = bases[1]

    m1 = markers_xyz[marker1]
    m2 = markers_xyz[marker2]

    x1, y1, z1 = df[m1["X"]], df[m1["Y"]], df[m1["Z"]]
    x2, y2, z2 = df[m2["X"]], df[m2["Y"]], df[m2["Z"]]

    valid = x1.notna() & y1.notna() & z1.notna() & x2.notna() & y2.notna() & z2.notna()
    if not valid.any():
        raise ValueError("No rows with complete XYZ for both markers.")

    dx = x1[valid] - x2[valid]
    dy = y1[valid] - y2[valid]
    dz = z1[valid] - z2[valid]
    d2 = dx**2 + dy**2 + dz**2
    d = np.sqrt(d2.to_numpy())
    valid_idx = list(d2.index)

    # Discrete minimum
    idx_min = d2.idxmin()
    d_min = float(np.sqrt(float(d2.loc[idx_min])))

    t_series = detect_time_series(df)
    t_min_val = float(t_series.loc[idx_min])

    # Raw below-threshold mask
    coll_raw = d < dist_threshold
    coll = coll_raw.copy()
    n = len(coll)

    # Fill small gaps
    i = 0
    while i < n:
        if not coll[i]:
            start = i
            while i < n and not coll[i]:
                i += 1
            end = i
            gap_len = end - start
            left_true = start > 0 and coll[start - 1]
            right_true = end < n and coll[end] if end < n else False
            if left_true and right_true and gap_len <= max_gap_frames:
                coll[start:end] = True
        else:
            i += 1

    positions = np.where(coll)[0]
    if positions.size > 0:
        start_pos = int(positions[0])
        end_pos = start_pos
        while end_pos + 1 < n and coll[end_pos + 1]:
            end_pos += 1
        idx_start = valid_idx[start_pos]
        idx_end = valid_idx[end_pos]
        t_p_minus = float(t_series.loc[idx_start])
        t_p_plus = float(t_series.loc[idx_end])
        threshold_hit = True
    else:
        t_p_minus = None
        t_p_plus = None
        threshold_hit = False
        print(
            f"Warning: no distance below threshold {dist_threshold:.3f}. "
            f"Minimum distance was {d_min:.3f}."
        )

    return {
        "t_min": t_min_val,
        "idx_min": int(idx_min),
        "d_min": d_min,
        "markers": (marker1, marker2),
        "time_col": str(t_series.name) if t_series.name is not None else "time",
        "t_p_minus": t_p_minus,
        "t_p_plus": t_p_plus,
        "threshold_hit": threshold_hit,
    }


# =============================================================================
# Velocities over full time series
# =============================================================================


def _prepare_time_and_positions(df: pd.DataFrame, markers_xyz: Dict[str, Dict[str, str]]):
    """Return time array and a dict of clean (interpolated) position series for each marker."""
    t_series = detect_time_series(df).astype(float)
    if not (t_series.diff().dropna() > 0).all():
        print("Warning: time is not strictly increasing; derivative may be noisy.")

    pos = {}
    for base, comps in markers_xyz.items():
        x = df[comps["X"]].astype(float).interpolate(limit_direction="both")
        y = df[comps["Y"]].astype(float).interpolate(limit_direction="both")
        z = df[comps["Z"]].astype(float).interpolate(limit_direction="both")
        pos[base] = (x, y, z)
    return t_series, pos


def compute_velocities(df: pd.DataFrame, markers_xyz: Dict[str, Dict[str, str]] | None = None):
    """Compute velocity components vx, vy, vz for each marker over the entire time series."""
    if markers_xyz is None:
        markers_xyz = find_markers_xyz_flexible(df)
    if not markers_xyz:
        raise ValueError("No markers with X,Y,Z found.")

    t, pos = _prepare_time_and_positions(df, markers_xyz)
    tt = t.to_numpy()
    vel = {}
    for base, (x, y, z) in pos.items():
        vx = pd.Series(np.gradient(x.to_numpy(), tt), index=x.index, name=f"vX_{base}")
        vy = pd.Series(np.gradient(y.to_numpy(), tt), index=y.index, name=f"vY_{base}")
        vz = pd.Series(np.gradient(z.to_numpy(), tt), index=z.index, name=f"vZ_{base}")
        vel[base] = {"vx": vx, "vy": vy, "vz": vz}
    return t, vel


def save_velocities_tsv(out_path: Path, time: pd.Series, vel: dict):
    """Save time and velocity components per marker to a TSV file."""
    out_cols = {"time": time}
    for base in sorted(vel.keys()):
        out_cols[f"vx_{base}"] = vel[base]["vx"]
        out_cols[f"vy_{base}"] = vel[base]["vy"]
        out_cols[f"vz_{base}"] = vel[base]["vz"]
    out_df = pd.DataFrame(out_cols)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved velocities to: {out_path}")


# =============================================================================
# Windowed velocities around collision (quadratic fits)
# =============================================================================


def compute_window_velocity_avgs(
    df: pd.DataFrame,
    markers_xyz: Dict[str, Dict[str, str]],
    time: pd.Series,
    marker_pair: tuple[str, str],
    t_p_minus: float,
    t_p_plus: float,
):
    """Compute velocity components just before and just after collision.

    Uses quadratic fits to x(t), y(t) and z(t) separately for each marker in marker_pair.
    Also computes the 1D projected velocity along a track direction defined by the first marker.
    If a fit cannot be constructed, falls back to finite-difference velocities near t_p_minus / t_p_plus.
    """
    t_float = time.astype(float)
    t_vals = t_float.to_numpy()

    if not marker_pair or marker_pair[0] not in markers_xyz:
        raise ValueError("compute_window_velocity_avgs: marker_pair or first marker missing.")

    # Define a 1D track direction based on the first marker's overall motion.
    base0 = marker_pair[0]
    comps0 = markers_xyz[base0]
    x0 = df[comps0["X"]].astype(float).to_numpy()
    y0 = df[comps0["Y"]].astype(float).to_numpy()
    z0 = df[comps0["Z"]].astype(float).to_numpy()
    r_start = np.array([x0[0], y0[0], z0[0]], dtype=np.float64)
    r_end = np.array([x0[-1], y0[-1], z0[-1]], dtype=np.float64)
    d_vec = r_end - r_start
    d_norm = np.linalg.norm(d_vec)
    if d_norm == 0:
        # Fallback direction if start and end positions are identical.
        e_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        print("Warning: could not define track direction from first marker, using x-axis as fallback.")
    else:
        e_hat = d_vec / d_norm

    results: Dict[str, Dict[str, dict | None]] = {}

    for base in marker_pair:
        if base not in markers_xyz:
            continue
        comps = markers_xyz[base]
        x = df[comps["X"]].astype(float)
        y = df[comps["Y"]].astype(float)
        z = df[comps["Z"]].astype(float)

        x_arr = x.to_numpy()
        y_arr = y.to_numpy()
        z_arr = z.to_numpy()

        # Approximate component-wise velocities and speed to detect motion onset/offset
        vx_grad = np.gradient(x_arr, t_vals)
        vy_grad = np.gradient(y_arr, t_vals)
        vz_grad = np.gradient(z_arr, t_vals)
        speed = np.sqrt(vx_grad**2 + vy_grad**2 + vz_grad**2)

        if speed.size == 0 or np.all(speed == 0):
            zero_sample = {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "speed_mean": 0.0,
                "speed_vec_mean": 0.0,
                "v_parallel": 0.0,
            }
            results[base] = {"pre": zero_sample, "post": zero_sample}
            continue

        speed_max = speed.max()
        if speed_max == 0:
            zero_sample = {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "speed_mean": 0.0,
                "speed_vec_mean": 0.0,
                "v_parallel": 0.0,
            }
            results[base] = {"pre": zero_sample, "post": zero_sample}
            continue

        v_eps = 0.05 * speed_max

        pre_sample = None
        post_sample = None

        # --- Pre-collision fit ---
        pre_mask = t_vals <= t_p_minus
        pre_idx_all = np.where(pre_mask)[0]
        if pre_idx_all.size >= 3:
            moving_pre_local = np.where(speed[pre_idx_all] > v_eps)[0]
            if moving_pre_local.size > 0:
                start_idx = pre_idx_all[moving_pre_local[0]]
            else:
                start_idx = pre_idx_all[0]
            pre_indices = np.arange(start_idx, pre_idx_all[-1] + 1)
            if pre_indices.size >= 3:
                coef_x_pre = np.polyfit(t_vals[pre_indices], x_arr[pre_indices], 2)
                coef_y_pre = np.polyfit(t_vals[pre_indices], y_arr[pre_indices], 2)
                coef_z_pre = np.polyfit(t_vals[pre_indices], z_arr[pre_indices], 2)
                ax_pre, bx_pre, _ = coef_x_pre
                ay_pre, by_pre, _ = coef_y_pre
                az_pre, bz_pre, _ = coef_z_pre
                vx_pre = float(2 * ax_pre * t_p_minus + bx_pre)
                vy_pre = float(2 * ay_pre * t_p_minus + by_pre)
                vz_pre = float(2 * az_pre * t_p_minus + bz_pre)
                sp_pre = float(np.sqrt(vx_pre**2 + vy_pre**2 + vz_pre**2))
                vpar_pre = float(vx_pre * e_hat[0] + vy_pre * e_hat[1] + vz_pre * e_hat[2])
                pre_sample = {
                    "vx": vx_pre,
                    "vy": vy_pre,
                    "vz": vz_pre,
                    "speed_mean": sp_pre,
                    "speed_vec_mean": sp_pre,
                    "v_parallel": vpar_pre,
                }

        if pre_sample is None:
            idx_closest_pre = int(np.argmin(np.abs(t_vals - t_p_minus)))
            vx_pre = float(vx_grad[idx_closest_pre])
            vy_pre = float(vy_grad[idx_closest_pre])
            vz_pre = float(vz_grad[idx_closest_pre])
            sp_pre = float(np.sqrt(vx_pre**2 + vy_pre**2 + vz_pre**2))
            vpar_pre = float(vx_pre * e_hat[0] + vy_pre * e_hat[1] + vz_pre * e_hat[2])
            pre_sample = {
                "vx": vx_pre,
                "vy": vy_pre,
                "vz": vz_pre,
                "speed_mean": sp_pre,
                "speed_vec_mean": sp_pre,
                "v_parallel": vpar_pre,
            }

        # --- Post-collision fit ---
        post_mask = t_vals >= t_p_plus
        post_idx_all = np.where(post_mask)[0]
        if post_idx_all.size >= 3:
            moving_post_local = np.where(speed[post_idx_all] > v_eps)[0]
            if moving_post_local.size > 0:
                end_idx = post_idx_all[moving_post_local[-1]]
            else:
                end_idx = post_idx_all[-1]
            post_indices = np.arange(post_idx_all[0], end_idx + 1)
            if post_indices.size >= 3:
                coef_x_post = np.polyfit(t_vals[post_indices], x_arr[post_indices], 2)
                coef_y_post = np.polyfit(t_vals[post_indices], y_arr[post_indices], 2)
                coef_z_post = np.polyfit(t_vals[post_indices], z_arr[post_indices], 2)
                ax_post, bx_post, _ = coef_x_post
                ay_post, by_post, _ = coef_y_post
                az_post, bz_post, _ = coef_z_post
                vx_post = float(2 * ax_post * t_p_plus + bx_post)
                vy_post = float(2 * ay_post * t_p_plus + by_post)
                vz_post = float(2 * az_post * t_p_plus + bz_post)
                sp_post = float(np.sqrt(vx_post**2 + vy_post**2 + vz_post**2))
                vpar_post = float(vx_post * e_hat[0] + vy_post * e_hat[1] + vz_post * e_hat[2])
                post_sample = {
                    "vx": vx_post,
                    "vy": vy_post,
                    "vz": vz_post,
                    "speed_mean": sp_post,
                    "speed_vec_mean": sp_post,
                    "v_parallel": vpar_post,
                }

        if post_sample is None:
            idx_closest_post = int(np.argmin(np.abs(t_vals - t_p_plus)))
            vx_post = float(vx_grad[idx_closest_post])
            vy_post = float(vy_grad[idx_closest_post])
            vz_post = float(vz_grad[idx_closest_post])
            sp_post = float(np.sqrt(vx_post**2 + vy_post**2 + vz_post**2))
            vpar_post = float(vx_post * e_hat[0] + vy_post * e_hat[1] + vz_post * e_hat[2])
            post_sample = {
                "vx": vx_post,
                "vy": vy_post,
                "vz": vz_post,
                "speed_mean": sp_post,
                "speed_vec_mean": sp_post,
                "v_parallel": vpar_post,
            }

        results[base] = {"pre": pre_sample, "post": post_sample}

    return results


# =============================================================================
# Momentum and kinetic energy
# =============================================================================


def compute_momentum_energy(
    vel_samples: dict,
    marker_pair: tuple[str, str],
    m1_kg: float = 1.0,
    m2_kg: float = 1.0,
):
    """Compute linear momentum and translational kinetic energy before and after collision."""
    b1, b2 = marker_pair

    def stats_for_marker(base: str, phase: str, m_kg: float):
        entry = vel_samples.get(base, {}).get(phase)
        if not entry:
            return (0.0, 0.0, 0.0), 0.0
        vx_val = float(entry["vx"])
        vy_val = float(entry["vy"])
        vz_val = float(entry["vz"])
        v2_val = vx_val**2 + vy_val**2 + vz_val**2
        p_vec = (m_kg * vx_val, m_kg * vy_val, m_kg * vz_val)
        KE = 0.5 * m_kg * v2_val
        return p_vec, KE

    p1_pre, KE1_pre = stats_for_marker(b1, "pre", m1_kg)
    p2_pre, KE2_pre = stats_for_marker(b2, "pre", m2_kg)

    # Projected momentum along e_hat (1D), using v_parallel values computed earlier
    vpar1_pre = float(vel_samples[b1]["pre"].get("v_parallel", 0.0))
    vpar2_pre = float(vel_samples[b2]["pre"].get("v_parallel", 0.0))
    p1_pre_e = m1_kg * vpar1_pre
    p2_pre_e = m2_kg * vpar2_pre
    p_sys_pre_e = p1_pre_e + p2_pre_e
    p_sys_pre = tuple(p1_pre[i] + p2_pre[i] for i in range(3))
    KE_sys_pre = KE1_pre + KE2_pre

    p1_post, KE1_post = stats_for_marker(b1, "post", m1_kg)
    p2_post, KE2_post = stats_for_marker(b2, "post", m2_kg)

    # Projected post-collision momentum along e_hat (1D)
    vpar1_post = float(vel_samples[b1]["post"].get("v_parallel", 0.0))
    vpar2_post = float(vel_samples[b2]["post"].get("v_parallel", 0.0))
    p1_post_e = m1_kg * vpar1_post
    p2_post_e = m2_kg * vpar2_post
    p_sys_post_e = p1_post_e + p2_post_e
    p_sys_post = tuple(p1_post[i] + p2_post[i] for i in range(3))
    KE_sys_post = KE1_post + KE2_post

    return {
        "pre": {
            "p1": p1_pre,
            "p2": p2_pre,
            "p_sys": p_sys_pre,
            "KE1": KE1_pre,
            "KE2": KE2_pre,
            "KE_sys": KE_sys_pre,
            "p1_e": p1_pre_e,
            "p2_e": p2_pre_e,
            "p_sys_e": p_sys_pre_e,
        },
        "post": {
            "p1": p1_post,
            "p2": p2_post,
            "p_sys": p_sys_post,
            "KE1": KE1_post,
            "KE2": KE2_post,
            "KE_sys": KE_sys_post,
            "p1_e": p1_post_e,
            "p2_e": p2_post_e,
            "p_sys_e": p_sys_post_e,
        },
    }


# =============================================================================
# Plotting: positions and fits (x, y, z)
# =============================================================================


def plot_positions_xyz_with_fits(
    df: pd.DataFrame,
    markers_xyz: Dict[str, Dict[str, str]],
    time: pd.Series,
    marker_pair: tuple[str, str],
    t_p_minus: float,
    t_p_plus: float,
    title_prefix: str = "",
):
    """Plot x(t), y(t), z(t) for the marker pair with quadratic pre/post fits and collision window.

    All plot labels/titles are in Swedish (as requested).
    """
    t = time.astype(float).to_numpy()

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    ax_x, ax_y, ax_z = axes

    components = [("X", ax_x, "x-position (m)"), ("Y", ax_y, "y-position (m)"), ("Z", ax_z, "z-position (m)")]

    colors = ["C0", "C1"]  # two markers

    for j, base in enumerate(marker_pair):
        if base not in markers_xyz:
            continue
        comps = markers_xyz[base]
        x = df[comps["X"]].astype(float).to_numpy()
        y = df[comps["Y"]].astype(float).to_numpy()
        z = df[comps["Z"]].astype(float).to_numpy()

        data_dict = {"X": x, "Y": y, "Z": z}

        for comp_key, ax, ylabel in components:
            arr = data_dict[comp_key]
            # Plot raw positions
            ax.plot(
                time,
                arr,
                label=f"{base} – mätt",
                color=colors[j],
                alpha=0.7,
            )

            # Approximate speed to detect motion
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
            vz = np.gradient(z, t)
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            if speed.size == 0 or np.all(speed == 0):
                continue
            v_max = speed.max()
            if v_max == 0:
                continue
            v_eps = 0.05 * v_max

            # Pre window
            pre_mask = t <= t_p_minus
            pre_idx_all = np.where(pre_mask)[0]
            if pre_idx_all.size >= 3:
                moving_pre_local = np.where(speed[pre_idx_all] > v_eps)[0]
                if moving_pre_local.size > 0:
                    start_idx = pre_idx_all[moving_pre_local[0]]
                else:
                    start_idx = pre_idx_all[0]
                pre_indices = np.arange(start_idx, pre_idx_all[-1] + 1)
                if pre_indices.size >= 3:
                    coef_pre = np.polyfit(t[pre_indices], arr[pre_indices], 2)
                    fit_pre = np.polyval(coef_pre, t[pre_indices])
                    ax.plot(
                        time.iloc[pre_indices],
                        fit_pre,
                        linestyle="--",
                        linewidth=1.2,
                        color=colors[j],
                        label=f"{base} – för-fit ({comp_key})" if comp_key == "X" else None,
                    )

            # Post window
            post_mask = t >= t_p_plus
            post_idx_all = np.where(post_mask)[0]
            if post_idx_all.size >= 3:
                moving_post_local = np.where(speed[post_idx_all] > v_eps)[0]
                if moving_post_local.size > 0:
                    end_idx = post_idx_all[moving_post_local[-1]]
                else:
                    end_idx = post_idx_all[-1]
                post_indices = np.arange(post_idx_all[0], end_idx + 1)
                if post_indices.size >= 3:
                    coef_post = np.polyfit(t[post_indices], arr[post_indices], 2)
                    fit_post = np.polyval(coef_post, t[post_indices])
                    ax.plot(
                        time.iloc[post_indices],
                        fit_post,
                        linestyle=":",
                        linewidth=1.2,
                        color=colors[j],
                        label=f"{base} – efter-fit ({comp_key})" if comp_key == "X" else None,
                    )

            ax.set_ylabel(ylabel)

    # Collision window markers
    for ax in axes:
        ax.axvline(t_p_minus, linestyle="--", color="k", linewidth=1, label="t_p−")
        ax.axvline(t_p_plus, linestyle="--", color="k", linewidth=1, label="t_p+")

    axes[-1].set_xlabel("Tid (s)")

    title_str = f"{title_prefix} – positioner och kvadratiska anpassningar".strip(" –")
    fig.suptitle(title_str)

    # Build a combined legend (avoid duplicates)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])


def plot_1d_projection_with_fits(
    df: pd.DataFrame,
    markers_xyz: Dict[str, Dict[str, str]],
    time: pd.Series,
    marker_pair: tuple[str, str],
    t_p_minus: float,
    t_p_plus: float,
    title_prefix: str = "",
):
    """Plot 1D projected position along the track direction with quadratic pre/post fits.

    The track direction is defined by the first marker in marker_pair, based on its overall motion.
    All text in the figure is in Swedish.
    """
    t = time.astype(float).to_numpy()

    # Track direction from first marker
    base0 = marker_pair[0]
    comps0 = markers_xyz[base0]
    x0 = df[comps0["X"]].astype(float).to_numpy()
    y0 = df[comps0["Y"]].astype(float).to_numpy()
    z0 = df[comps0["Z"]].astype(float).to_numpy()
    r_start = np.array([x0[0], y0[0], z0[0]], dtype=np.float64)
    r_end = np.array([x0[-1], y0[-1], z0[-1]], dtype=np.float64)
    d_vec = r_end - r_start
    d_norm = np.linalg.norm(d_vec)
    if d_norm == 0:
        e_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        print("Warning: could not define track direction from first marker for 1D plot, using x-axis as fallback.")
    else:
        e_hat = d_vec / d_norm

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ["C0", "C1"]
    for j, base in enumerate(marker_pair):
        if base not in markers_xyz:
            continue
        comps = markers_xyz[base]
        x = df[comps["X"]].astype(float).to_numpy()
        y = df[comps["Y"]].astype(float).to_numpy()
        z = df[comps["Z"]].astype(float).to_numpy()
        r = np.stack([x, y, z], axis=1)
        s = (r - r_start) @ e_hat  # 1D projection along the track

        ax.plot(time, s, label=f"{base} – projicerad position", color=colors[j], alpha=0.7)

        # Use full 3D speed to detect motion intervals
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        vz = np.gradient(z, t)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        if speed.size == 0 or np.all(speed == 0):
            continue
        v_max = speed.max()
        if v_max == 0:
            continue
        v_eps = 0.05 * v_max

        # Pre window
        pre_mask = t <= t_p_minus
        pre_idx_all = np.where(pre_mask)[0]
        if pre_idx_all.size >= 3:
            moving_pre_local = np.where(speed[pre_idx_all] > v_eps)[0]
            if moving_pre_local.size > 0:
                start_idx = pre_idx_all[moving_pre_local[0]]
            else:
                start_idx = pre_idx_all[0]
            pre_indices = np.arange(start_idx, pre_idx_all[-1] + 1)
            if pre_indices.size >= 3:
                coef_pre = np.polyfit(t[pre_indices], s[pre_indices], 2)
                fit_pre = np.polyval(coef_pre, t[pre_indices])
                ax.plot(
                    time.iloc[pre_indices],
                    fit_pre,
                    linestyle="--",
                    linewidth=1.2,
                    color=colors[j],
                    label=f"{base} – 1D för-fit" if j == 0 else None,
                )

        # Post window
        post_mask = t >= t_p_plus
        post_idx_all = np.where(post_mask)[0]
        if post_idx_all.size >= 3:
            moving_post_local = np.where(speed[post_idx_all] > v_eps)[0]
            if moving_post_local.size > 0:
                end_idx = post_idx_all[moving_post_local[-1]]
            else:
                end_idx = post_idx_all[-1]
            post_indices = np.arange(post_idx_all[0], end_idx + 1)
            if post_indices.size >= 3:
                coef_post = np.polyfit(t[post_indices], s[post_indices], 2)
                fit_post = np.polyval(coef_post, t[post_indices])
                ax.plot(
                    time.iloc[post_indices],
                    fit_post,
                    linestyle=":",
                    linewidth=1.2,
                    color=colors[j],
                    label=f"{base} – 1D efter-fit" if j == 0 else None,
                )

    # Collision window markers
    ax.axvline(t_p_minus, linestyle="--", color="k", linewidth=1, label="t_p−")
    ax.axvline(t_p_plus, linestyle="--", color="k", linewidth=1, label="t_p+")

    ax.set_xlabel("Tid (s)")
    ax.set_ylabel("Position längs bana (m)")
    title_str = f"{title_prefix} – 1D-projektion längs banan".strip(" –")
    ax.set_title(title_str)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")

    plt.tight_layout()


# =============================================================================
# High-level pipeline
# =============================================================================


@dataclass
class Config:
    m1: float = 1.0
    m2: float = 1.0
    dist_threshold: float = 1.0
    max_gap_frames: int = 3
    save_vel: bool = True
    make_plots: bool = True
    marker1: str | None = None
    marker2: str | None = None


def analyze_file(path: Path, cfg: Config) -> dict:
    """Run the full analysis pipeline on a single file and return a structured result dict."""
    out = {
        "file": str(path),
        "markers": None,
        "collision": None,
        "vel_path": None,
        "window_avgs": None,
        "momentum_energy": None,
        "vel_prepost": None,
    }

    # 1) Load data
    df = load_qmt_any(str(path))

    # 2) Marker detection
    markers_xyz = find_markers_xyz_flexible(df)
    out["markers"] = list(markers_xyz.keys())
    if len(markers_xyz) < 2:
        raise ValueError("Need at least two markers with X,Y,Z to proceed.")

    # 3) Collision detection
    col = find_collision_time(
        df,
        markers_xyz=markers_xyz,
        marker1=cfg.marker1,
        marker2=cfg.marker2,
        dist_threshold=cfg.dist_threshold,
        max_gap_frames=cfg.max_gap_frames,
    )
    out["collision"] = col
    out["t_collision"] = col.get("t_min", None)

    # 4) Velocities over full series
    t, vel = compute_velocities(df, markers_xyz)
    if cfg.save_vel:
        out_path = path.with_suffix("")
        out_tsv = out_path.parent / f"{out_path.name}_vel.tsv"
        save_velocities_tsv(out_tsv, t, vel)
        out["vel_path"] = str(out_tsv)

    # 5) Windowed velocities, momentum & energy, plots
    if col["threshold_hit"] and col["t_p_minus"] is not None and col["t_p_plus"] is not None:
        marker_pair = tuple(col["markers"])
        wav = compute_window_velocity_avgs(
            df,
            markers_xyz,
            t,
            marker_pair,
            col["t_p_minus"],
            col["t_p_plus"],
        )
        out["window_avgs"] = wav

        me = compute_momentum_energy(wav, marker_pair, m1_kg=cfg.m1, m2_kg=cfg.m2)
        out["momentum_energy"] = me
        # Export pre/post velocities along track direction (v_parallel)
        base1, base2 = marker_pair
        out["vel_prepost"] = {
            "v1_pre": wav[base1]["pre"].get("v_parallel", 0.0),
            "v2_pre": wav[base2]["pre"].get("v_parallel", 0.0),
            "v1_post": wav[base1]["post"].get("v_parallel", 0.0),
            "v2_post": wav[base2]["post"].get("v_parallel", 0.0),
        }

        if cfg.make_plots:
            plot_positions_xyz_with_fits(
                df,
                markers_xyz,
                t,
                marker_pair,
                col["t_p_minus"],
                col["t_p_plus"],
                title_prefix=path.name,
            )
            plot_1d_projection_with_fits(
                df,
                markers_xyz,
                t,
                marker_pair,
                col["t_p_minus"],
                col["t_p_plus"],
                title_prefix=path.name,
            )
    else:
        out["window_avgs"] = None
        out["momentum_energy"] = None

    return out


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze QTM TSV: collision time, pre/post velocities, momentum & energy.\n"
            "Collision distance threshold is a REQUIRED parameter."
        )
    )
    parser.add_argument("files", nargs="+", help="Path(s) to TSV export(s) from QTM.")
    parser.add_argument("--m1", type=float, default=1.0, help="Mass of body 1 in kg (default: 1.0)")
    parser.add_argument("--m2", type=float, default=1.0, help="Mass of body 2 in kg (default: 1.0)")
    parser.add_argument(
        "--dist-threshold",
        type=float,
        required=True,
        help="Distance threshold for defining collision interval (REQUIRED, same units as positions).",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
        help="Maximum frame gap between below-threshold segments to still be merged (default: 3).",
    )
    parser.add_argument("--no-save-vel", action="store_true", help="Do not save velocities TSV.")
    parser.add_argument("--no-plots", action="store_true", help="Do not show/create plots.")
    parser.add_argument(
        "--marker1",
        type=str,
        default=None,
        help="Name of first marker to use (e.g. 'marker_1' or 'Ball'). If omitted, the first detected marker is used.",
    )
    parser.add_argument(
        "--marker2",
        type=str,
        default=None,
        help="Name of second marker. If omitted, the second detected marker is used.",
    )

    args = parser.parse_args(argv)

    cfg = Config(
        m1=args.m1,
        m2=args.m2,
        dist_threshold=args.dist_threshold,
        max_gap_frames=args.max_gap_frames,
        save_vel=not args.no_save_vel,
        make_plots=not args.no_plots,
        marker1=args.marker1,
        marker2=args.marker2,
    )

    any_ok = False

    for fname in args.files:
        path = Path(fname)
        if not path.exists():
            print(f"Warning: file not found: {path}")
            continue
        try:
            print(f"\n=== Analyzing: {path.name} ===")
            result = analyze_file(path, cfg)

            col = result["collision"]
            if col.get("threshold_hit", False):
                print(
                    f"Collision interval ({col['time_col']}): "
                    f"t_p-={col['t_p_minus']:.6f}, t_p+={col['t_p_plus']:.6f}\n"
                    f"  min distance ≈ {col['d_min']:.6f} at {col['t_min']:.6f}, index {col['idx_min']}\n"
                    f"  markers: {col['markers'][0]} vs {col['markers'][1]}"
                )
            else:
                print(
                    "No collision detected (distance never below threshold).\n"
                    f"  min distance ≈ {col['d_min']:.6f} at {col['t_min']:.6f}, index {col['idx_min']}\n"
                    f"  markers: {col['markers'][0]} vs {col['markers'][1]}"
                )

            if result["vel_path"]:
                print(f"Saved velocities: {result['vel_path']}")

            wav = result["window_avgs"]
            if wav is not None:
                for base in sorted(wav.keys()):
                    pre = wav[base]["pre"]
                    post = wav[base]["post"]

                    def fmt(v):
                        return (
                            f"vx={v['vx']:.4f}, vy={v['vy']:.4f}, vz={v['vz']:.4f}, "
                            f"v_par={v.get('v_parallel', 0.0):.4f}, "
                            f"|v|mean={v['speed_mean']:.4f}, |<v>|={v['speed_vec_mean']:.4f}"
                            if v
                            else "(no data)"
                        )

                    print(f"{base}: pre[{fmt(pre)}]  post[{fmt(post)}]")

            me = result["momentum_energy"]
            if me is not None:
                def fmt_p(p):
                    return f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})"

                def fmt_ke(ke):
                    return f"{ke:.6f}"

                print("\nMomentum (kg·m/s) and Kinetic Energy (J):")
                print(
                    f"  pre : p1={fmt_p(me['pre']['p1'])}, "
                    f"p2={fmt_p(me['pre']['p2'])}, "
                    f"p_sys={fmt_p(me['pre']['p_sys'])}, "
                    f"p_sys_e={me['pre']['p_sys_e']:.6f}, "
                    f"KE1={fmt_ke(me['pre']['KE1'])} J, "
                    f"KE2={fmt_ke(me['pre']['KE2'])} J, "
                    f"KE_sys={fmt_ke(me['pre']['KE_sys'])} J"
                )
                print(
                    f"  post: p1={fmt_p(me['post']['p1'])}, "
                    f"p2={fmt_p(me['post']['p2'])}, "
                    f"p_sys={fmt_p(me['post']['p_sys'])}, "
                    f"p_sys_e={me['post']['p_sys_e']:.6f}, "
                    f"KE1={fmt_ke(me['post']['KE1'])} J, "
                    f"KE2={fmt_ke(me['post']['KE2'])} J, "
                    f"KE_sys={fmt_ke(me['post']['KE_sys'])} J"
                )
            else:
                print("\nMomentum and kinetic energy: No collision interval, so no pre/post summary.")

            any_ok = True
        except Exception as e:
            print(f"Error analyzing {path}: {e}")

    if any_ok and cfg.make_plots:
        plt.show()
        return 0
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
