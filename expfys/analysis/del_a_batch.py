"""Batch helper that reproduces the old Del A conservation summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import del_a as del_a_mod

CAMERA_POS_ERROR_M = 0.02e-3  # 0.02 mm
FIT_WINDOW_DT = 0.5  # [s]


def compute_relative_delta(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0
    return (after - before) / before


def run_batch(
    base_dir: Path,
    prefix: str,
    start_idx: int = 1,
    end_idx: int = 10,
    ext: str = ".tsv",
    marker1: Optional[str] = None,
    marker2: Optional[str] = None,
    m1: float = 0.2,
    m2: float = 0.2,
    dist_threshold: float = 1.0,
) -> pd.DataFrame:
    rows = []
    cfg = del_a_mod.Config(
        m1=m1,
        m2=m2,
        dist_threshold=dist_threshold,
        save_vel=False,
        make_plots=False,
        marker1=marker1,
        marker2=marker2,
    )
    m_tot = m1 + m2

    for idx in range(start_idx, end_idx + 1):
        fname = f"{prefix}({idx}){ext}"
        path = base_dir / fname
        print(f"\n=== Analysis of file: {path.name} ===")
        if not path.exists():
            print("  -> Missing file, skipping.")
            continue

        try:
            result = del_a_mod.analyze_file(path, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  -> Error analysing {path.name}: {exc}")
            continue

        me = result.get("momentum_energy")
        collision = result.get("collision")
        if me is None or collision is None or not collision.get("threshold_hit", False):
            print("  -> No collision detected or missing data, skipping.")
            continue

        pre = me["pre"]
        post = me["post"]
        p_sys_pre = pre["p_sys"]
        p_sys_post = post["p_sys"]
        KE_sys_pre = pre["KE_sys"]
        KE_sys_post = post["KE_sys"]
        p_sys_pre_e = float(pre.get("p_sys_e", 0.0))
        p_sys_post_e = float(post.get("p_sys_e", 0.0))

        dp_x = p_sys_post[0] - p_sys_pre[0]
        dp_y = p_sys_post[1] - p_sys_pre[1]
        dp_z = p_sys_post[2] - p_sys_pre[2]
        dp_e = p_sys_post_e - p_sys_pre_e
        dKE = KE_sys_post - KE_sys_pre

        rel_dp_x = compute_relative_delta(p_sys_pre[0], p_sys_post[0])
        rel_dp_y = compute_relative_delta(p_sys_pre[1], p_sys_post[1])
        rel_dp_z = compute_relative_delta(p_sys_pre[2], p_sys_post[2])
        rel_dp_e = compute_relative_delta(p_sys_pre_e, p_sys_post_e)
        rel_dKE = compute_relative_delta(KE_sys_pre, KE_sys_post)

        eps_p_pre = 0.0
        eps_p_post = 0.0
        if p_sys_pre_e != 0.0:
            eps_p_pre = np.sqrt(2.0) * CAMERA_POS_ERROR_M * m_tot / (
                FIT_WINDOW_DT * abs(p_sys_pre_e)
            )
        if p_sys_post_e != 0.0:
            eps_p_post = np.sqrt(2.0) * CAMERA_POS_ERROR_M * m_tot / (
                FIT_WINDOW_DT * abs(p_sys_post_e)
            )

        sigma_p_pre_e = abs(p_sys_pre_e) * eps_p_pre
        sigma_p_post_e = abs(p_sys_post_e) * eps_p_post

        sigma_rel_dp_e = 0.0
        if p_sys_pre_e != 0.0:
            df_dp_pre = -p_sys_post_e / (p_sys_pre_e**2)
            df_dp_post = 1.0 / p_sys_pre_e
            sigma_rel_dp_e = np.sqrt(
                (df_dp_pre**2) * (sigma_p_pre_e**2)
                + (df_dp_post**2) * (sigma_p_post_e**2)
            )
        rel_dp_e_lo = rel_dp_e - sigma_rel_dp_e
        rel_dp_e_hi = rel_dp_e + sigma_rel_dp_e

        vel = result.get("vel_prepost") or {}
        row = {
            "file": path.name,
            "v1_pre": vel.get("v1_pre"),
            "v2_pre": vel.get("v2_pre"),
            "v1_post": vel.get("v1_post"),
            "v2_post": vel.get("v2_post"),
            "p_sys_pre_x": p_sys_pre[0],
            "p_sys_pre_y": p_sys_pre[1],
            "p_sys_pre_z": p_sys_pre[2],
            "p_sys_post_x": p_sys_post[0],
            "p_sys_post_y": p_sys_post[1],
            "p_sys_post_z": p_sys_post[2],
            "p_sys_pre_e": p_sys_pre_e,
            "p_sys_post_e": p_sys_post_e,
            "KE_sys_pre": KE_sys_pre,
            "KE_sys_post": KE_sys_post,
            "dp_x": dp_x,
            "dp_y": dp_y,
            "dp_z": dp_z,
            "dp_e": dp_e,
            "dKE": dKE,
            "rel_dp_x": rel_dp_x,
            "rel_dp_y": rel_dp_y,
            "rel_dp_z": rel_dp_z,
            "rel_dp_e": rel_dp_e,
            "rel_dp_e_sigma": sigma_rel_dp_e,
            "rel_dp_e_lo": rel_dp_e_lo,
            "rel_dp_e_hi": rel_dp_e_hi,
            "rel_dKE": rel_dKE,
            "t_collision": result.get("t_collision"),
        }
        rows.append(row)

    return pd.DataFrame(rows)
