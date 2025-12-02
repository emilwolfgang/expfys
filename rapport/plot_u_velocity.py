#!/usr/bin/env python3
r"""Generate the $\hat{u}$ position/velocity figure for the report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_RC_PARAMS = {
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 20,
}
plt.rcParams.update(REPORT_RC_PARAMS)

from expfys.analysis.del_b_free import (
    compute_acceleration_series,
    compute_velocity_series,
    find_collision_interval,
    infer_dist_threshold_from_bodies,
    plot_u_and_vu,
    polyfit_pre_post,
    load_bodies,
)
from expfys.dataio import load_qtm_6d_file_version

DEFAULT_DATA = PROJECT_ROOT / "data/del_b/raw/B1(1).tsv"
DEFAULT_BODIES = PROJECT_ROOT / "data/del_b/bodies/bodiesB1.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the report figure with position and velocity in the \\hat{u} direction."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_DATA,
        help=f"QTM TSV measurement to visualize (default: {DEFAULT_DATA.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--bodies",
        type=Path,
        default=DEFAULT_BODIES,
        help=f"TSV file with body definitions (default: {DEFAULT_BODIES.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--dist-threshold",
        type=float,
        default=None,
        help="Override distance threshold (mm) used to detect the collision window.",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
        help="Max number of samples between segments under the collision threshold to merge.",
    )
    return parser.parse_args()


def compute_velocity_fits(
    time: np.ndarray,
    pos: dict[str, np.ndarray],
    bodies_order: tuple[str, str],
    start_idx: int,
    end_idx: int,
    t_minus: float,
    t_plus: float,
    t_p_minus: float,
    t_p_plus: float,
) -> tuple[dict[str, Dict[int, dict]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build polynomial fits for each component and evaluate v_pre/v_post."""
    velocity_fits: dict[str, Dict[int, dict]] = {body: {} for body in bodies_order}
    v_pre: dict[str, np.ndarray] = {}
    v_post: dict[str, np.ndarray] = {}

    for body in bodies_order:
        r = pos[body]
        v_pre_body = np.zeros(3, dtype=float)
        v_post_body = np.zeros(3, dtype=float)
        for comp in range(3):
            res_fit = polyfit_pre_post(
                time,
                r[:, comp],
                start_idx=start_idx,
                end_idx=end_idx,
                t_p_minus=t_minus,
                t_p_plus=t_plus,
                window=20,
                deg=2,
            )
            velocity_fits[body][comp] = res_fit
            dp_pre = res_fit["dp_pre"]
            dp_post = res_fit["dp_post"]
            v_pre_body[comp] = float(np.polyval(dp_pre, t_p_minus))
            v_post_body[comp] = float(np.polyval(dp_post, t_p_plus))

        v_pre[body] = v_pre_body
        v_post[body] = v_post_body

    return velocity_fits, v_pre, v_post


def determine_impact_times(
    time: np.ndarray,
    accels: dict[str, np.ndarray],
    bodies_order: tuple[str, str],
    start_idx: int,
    end_idx: int,
    t_minus: float,
    t_plus: float,
) -> tuple[float, float]:
    """Replicate the acceleration-based impact detection used in the analysis script."""
    a_mag_all_list = []
    a_mag_per_body: dict[str, np.ndarray] = {}
    for body in bodies_order:
        a = accels[body]
        a_mag = np.linalg.norm(a, axis=1)
        a_mag_per_body[body] = a_mag
        a_mag_all_list.append(a_mag)

    a_mag_all = np.concatenate(a_mag_all_list)
    base = float(np.median(a_mag_all))
    std = float(np.std(a_mag_all))
    if std > 0.0:
        threshold = base + std
    else:
        threshold = float(np.percentile(a_mag_all, 95.0))

    N = len(time)
    region_start = max(0, start_idx - 5)
    region_end = min(N - 1, end_idx + 5)
    mask_region = np.zeros(N, dtype=bool)
    mask_region[region_start : region_end + 1] = True

    mask_impact = np.zeros(N, dtype=bool)
    for body in bodies_order:
        mask_impact |= a_mag_per_body[body] >= threshold

    mask = mask_impact & mask_region
    idx_imp = np.where(mask)[0]
    if idx_imp.size == 0:
        return t_minus, t_plus
    return float(time[idx_imp[0]]), float(time[idx_imp[-1]])


def main() -> None:
    args = parse_args()
    file_path = args.file if args.file.is_absolute() else (PROJECT_ROOT / args.file)
    bodies_path = args.bodies if args.bodies.is_absolute() else (PROJECT_ROOT / args.bodies)

    time, marker_pos_all, _ = load_qtm_6d_file_version(file_path)
    bodies = load_bodies(bodies_path)

    for required_body in ("Puck1", "Puck2"):
        if required_body not in bodies:
            raise ValueError(f"{required_body} saknas i {bodies_path}")

    for marker in ("cm1", "cm2"):
        if marker not in marker_pos_all:
            raise ValueError(f"Markören {marker} saknas i {file_path}")

    bodies_order = ("Puck1", "Puck2")
    pos = {
        "Puck1": marker_pos_all["cm1"],
        "Puck2": marker_pos_all["cm2"],
    }

    velocities = compute_velocity_series(time, pos)
    accels = compute_acceleration_series(time, velocities)

    dist_threshold = args.dist_threshold
    if dist_threshold is None:
        dist_threshold = infer_dist_threshold_from_bodies(bodies_path)
    if dist_threshold is None:
        raise ValueError("Ingen distans-tröskel hittades – ange --dist-threshold manuellt.")

    collision = find_collision_interval(
        time,
        pos,
        dist_threshold=dist_threshold,
        max_gap_frames=args.max_gap_frames,
    )
    if not collision["threshold_hit"]:
        raise RuntimeError("Kunde inte hitta kollisionsfönster med angiven tröskel.")

    t_minus = collision["t_p_minus"]
    t_plus = collision["t_p_plus"]
    start_idx = collision["start_idx"]
    end_idx = collision["end_idx"]

    t_p_minus, t_p_plus = determine_impact_times(
        time,
        accels,
        bodies_order=bodies_order,
        start_idx=start_idx,
        end_idx=end_idx,
        t_minus=t_minus,
        t_plus=t_plus,
    )

    velocity_fits, v_pre, v_post = compute_velocity_fits(
        time,
        pos,
        bodies_order,
        start_idx,
        end_idx,
        t_minus,
        t_plus,
        t_p_minus,
        t_p_plus,
    )

    all_points = np.concatenate(list(pos.values()), axis=0)
    origin = all_points.mean(axis=0)

    plot_u_and_vu(
        time,
        pos,
        velocities,
        origin,
        bodies_order=bodies_order,
        pos_fits=velocity_fits,
        v_pre=v_pre,
        v_post=v_post,
        t_p_minus=t_p_minus,
        t_p_plus=t_p_plus,
        t_p_star=collision["t_min"],
        title=rf"{file_path.name} – Rörelse i $\hat{{u}}$-riktning",
        t_window_minus=t_minus,
        t_window_plus=t_plus,
    )
    plt.show()


if __name__ == "__main__":
    main()
