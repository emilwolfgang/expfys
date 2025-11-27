#!/usr/bin/env python3
"""
Air table QTM analyzer for 3D markers (2D motion with angular momentum)

Usage (exempel):
  python main.py B1(1).tsv --bodies bodies.tsv --dist-threshold 63.5

Antaganden:
- QTM-export är FILE_VERSION-text med DATA_INCLUDED = 3D.
- MARKER_NAMES innehåller fyra markörer: cm1, ed1, cm2, ed2.
- cm1 och cm2 är markörer nära masscentrum för Puck1 respektive Puck2.
- ed1 och ed2 är "kantmarkörer" på respektive puck som används för orientering.
- bodies.tsv innehåller Puck1 och Puck2 med mass_kg och Izz_kgm2.

Vad skriptet gör (ny logik):
- Läser in 3D marker-data från QTM.
- Använder cm1/cm2 som masscentrum-markörer (COM) för Puck1/Puck2.
- Använder ed1/ed2 som "kantmarkörer" för orientering (vektor från COM till ed?-markören).
- Antar att rörelsen sker i x–y-planet och att z-axeln är normal (axeln för L).
- Detekterar ett kollisionsfönster där COM-distansen < dist_threshold.
  * Kanterna av detta fönster kallas t_- och t_+ och används för att definiera
    vilka data som används i polynomanpassning före/efter.
- Beräknar accelerationer |a(t)| från COM-data. Definierar t_P- och t_P+ som
  de tider inom ett område runt kollisionen där |a| ligger över en data-driven
  tröskel (impact-fönster, själva stöten).
- Fittar polynom före (pre) och efter (post) kollisionsfönstret baserat på t_- och t_+.
  * Pre-fit görs på data före t_-.
  * Post-fit görs på data efter t_+.
- Bestämmer pre-hastigheter genom att evaluera derivatan av pre-fit vid t_P-.
- Bestämmer post-hastigheter genom att evaluera derivatan av post-fit vid t_P+.
- Gör samma sak för rotation (θ(t) i planet -> ω(t)).
- Beräknar p, KE och L (orbitalt + spinn) före/efter kollisionen vid tiderna t_P-, t_P+.
- Ritar:
  * u, v, e (≈ x, y, z) vs tid,
  * v_x, v_y, v_z vs tid med fits,
  * banor i x–y-planet med orienteringslinjer,
  * θ(t) med fits (i grader).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expfys.dataio import load_qtm_6d_file_version


# =============================================================================
# QTM loading is provided by expfys.dataio.qtm
# =============================================================================


# =============================================================================
# Kollisiondetektion på COM (används för att bestämma t_- och t_+)
# =============================================================================


def find_collision_interval(
    time: np.ndarray,
    pos: Dict[str, np.ndarray],
    dist_threshold: float,
    max_gap_frames: int = 3,
    bodies_pair: Tuple[str, str] | None = None,
):
    """Detektera kollisionsfönster baserat på COM-avstånd.

    Distans-tröskeln används enbart för att definiera ett tidsintervall där
    vi senare gör polynomanpassningar (fönsterkanterna kallas t_- respektive t_+).

    Returnerar ett dict med:
      - threshold_hit (bool)
      - t_min, d_min, idx_min
      - t_p_minus, t_p_plus (vi tolkar dem som t_- och t_+)
      - start_idx, end_idx (index för fönstret)
    """
    if bodies_pair is not None:
        b1, b2 = bodies_pair
    else:
        bodies = list(pos.keys())
        if len(bodies) < 2:
            raise ValueError("find_collision_interval: förväntar minst två kroppar.")
        b1, b2 = bodies[0], bodies[1]
    r1 = pos[b1]
    r2 = pos[b2]
    diff = r1 - r2
    d = np.linalg.norm(diff, axis=1)  # (N,)

    idx_min = int(np.argmin(d))
    d_min = float(d[idx_min])
    t_min = float(time[idx_min])

    below = d < dist_threshold
    n = len(below)

    # Fyll små hål
    below_filled = below.copy()
    i = 0
    while i < n:
        if not below_filled[i]:
            start = i
            while i < n and not below_filled[i]:
                i += 1
            end = i  # [start, end) är ett "hål"
            gap_len = end - start
            left_true = start > 0 and below_filled[start - 1]
            right_true = end < n and below_filled[end] if end < n else False
            if left_true and right_true and gap_len <= max_gap_frames:
                below_filled[start:end] = True
        else:
            i += 1

    idx = np.where(below_filled)[0]
    if idx.size == 0:
        return {
            "threshold_hit": False,
            "t_min": t_min,
            "d_min": d_min,
            "idx_min": idx_min,
            "t_p_minus": None,
            "t_p_plus": None,
            "bodies": (b1, b2),
        }

    start_idx = idx[0]
    end_idx = idx[-1]
    t_p_minus = float(time[start_idx])  # vi tolkar detta som t_-
    t_p_plus = float(time[end_idx])     # vi tolkar detta som t_+

    return {
        "threshold_hit": True,
        "t_min": t_min,
        "d_min": d_min,
        "idx_min": idx_min,
        "t_p_minus": t_p_minus,
        "t_p_plus": t_p_plus,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "bodies": (b1, b2),
    }


# =============================================================================
# Orientering & vinkelhastighet från markörer (COM + kant)
# =============================================================================


def compute_orientation_from_markers(
    time: np.ndarray,
    cm_pos: Dict[str, np.ndarray],
    edge_pos: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """Beräkna orienteringsvinkel θ(t) och vinkelhastighet ω(t) från två markörer direkt i x–y-planet."""
    orient: Dict[str, dict] = {}
    t = time

    for body, r_cm in cm_pos.items():
        if body not in edge_pos:
            raise ValueError(f"MISSING edge marker for body {body!r}.")
        r_edge = edge_pos[body]
        if r_edge.shape != r_cm.shape:
            raise ValueError(
                f"cm_pos och edge_pos har olika form för {body!r}: "
                f"{r_cm.shape} vs {r_edge.shape}"
            )

        # Vektor från COM till kantmarkör
        a = r_edge - r_cm  # (N,3)

        # Använd direkt x–y-komponenterna i stället för projektion på ett PCA-plan
        a_xy = a[:, :2]  # (N,2) -> (ax, ay)

        theta = np.arctan2(a_xy[:, 1], a_xy[:, 0])
        theta_unwrap = np.unwrap(theta)
        omega = np.gradient(theta_unwrap, t)

        orient[body] = {
            "theta": theta_unwrap,
            "omega": omega,
            "rel_vec_xy": a_xy,
        }

    return orient


# =============================================================================
# Impuls, energi och rörelsemängdsmoment
# =============================================================================


def sample_at_times(time: np.ndarray, arr: np.ndarray, t_target: float) -> np.ndarray:
    """Ta värdet i arr (N,...) vid närmsta tidpunkt till t_target."""
    idx = int(np.argmin(np.abs(time - t_target)))
    return arr[idx]


def polyfit_pre_post(
    time: np.ndarray,
    y: np.ndarray,
    start_idx: int,
    end_idx: int,
    t_p_minus: float,
    t_p_plus: float,
    window: int = 20,
    deg: int = 2,
):
    """Gör en polynomanpassning (före/efter) kring kollisionsfönstret.

    t_p_minus och t_p_plus här ska tolkas som kanterna på fönstret (t_- och t_+),
    dvs de används för att beräkna dy/dt just vid fönsterkanterna. I den nya
    logiken används i stället dp_pre/dp_post vid t_P- och t_P+ för faktiska v_pre/v_post.

    Returnerar:
      {
        "pre_t": t_pre,
        "pre_fit": y_pre_fit,
        "post_t": t_post,
        "post_fit": y_post_fit,
        "dy_pre": dy_dt_pre vid t_p_minus,
        "dy_post": dy_dt_post vid t_p_plus,
        "p_pre": p_pre,
        "p_post": p_post,
        "dp_pre": dp_pre,
        "dp_post": dp_post,
      }
    """
    N = len(time)
    # pre-fönster: några punkter före kollisionsstart
    pre_start = max(0, start_idx - window)
    pre_end = start_idx + 1
    # post-fönster: från kollisionsslut och några punkter efter
    post_start = end_idx
    post_end = min(N, end_idx + 1 + window)

    t_pre = time[pre_start:pre_end]
    y_pre = y[pre_start:pre_end]
    t_post = time[post_start:post_end]
    y_post = y[post_start:post_end]

    # Fallback om det är för få punkter: använd vad som finns
    if t_pre.size < deg + 1:
        t_pre = time[: start_idx + 1]
        y_pre = y[: start_idx + 1]
    if t_post.size < deg + 1:
        t_post = time[end_idx:]
        y_post = y[end_idx:]

    # Anpassa polynom
    p_pre = np.polyfit(t_pre, y_pre, deg)
    p_post = np.polyfit(t_post, y_post, deg)

    # Derivera polynomen
    dp_pre = np.polyder(p_pre)
    dp_post = np.polyder(p_post)

    dy_pre = float(np.polyval(dp_pre, t_p_minus))
    dy_post = float(np.polyval(dp_post, t_p_plus))

    y_pre_fit = np.polyval(p_pre, t_pre)
    y_post_fit = np.polyval(p_post, t_post)

    return {
        "pre_t": t_pre,
        "pre_fit": y_pre_fit,
        "post_t": t_post,
        "post_fit": y_post_fit,
        "dy_pre": dy_pre,
        "dy_post": dy_post,
        "p_pre": p_pre,
        "p_post": p_post,
        "dp_pre": dp_pre,
        "dp_post": dp_post,
    }


def compute_momentum_energy_L(
    time: np.ndarray,
    pos: Dict[str, np.ndarray],
    bodies_phys: Dict[str, dict],
    origin: np.ndarray,
    n_hat: np.ndarray,
    t_pre: float,
    t_post: float,
    bodies_order: Tuple[str, str],
    v_pre: Dict[str, np.ndarray],
    v_post: Dict[str, np.ndarray],
    omega_pre: Dict[str, float],
    omega_post: Dict[str, float],
):
    """Beräkna p, KE och L (orbital + spinn) före/efter kollision.

    Före kollision: systemet behandlas som två separata kroppar (Puck1, Puck2)
    och p_sys, KE_sys och L_sys beräknas som summa av deras bidrag.

    Efter kollision: Puck1 och Puck2 antas ha fastnat i varandra och behandlas
    som en sammanslagen kropp ("Puck3"). Rörelsemängden efter definieras som
    (m1 + m2) * v_3(t_P+), och L efter definieras som r_3 × p_sys_post + I3 * omega_3,
    där I3 beräknas med parallellaxelsatsen runt Puck3:s masscentrum.

    v_pre/v_post och omega_pre/omega_post antas komma från polynomanpassning
    före/efter kollisionsfönstret. Vi evaluerar systemets tillstånd vid t_pre = t_P-
    och t_post = t_P+.
    """
    b1, b2 = bodies_order

    def stats_at_time(phase_time: float, v_dict: Dict[str, np.ndarray], omega_dict: Dict[str, float]):
        res: Dict[str, dict] = {}
        for body in (b1, b2):
            # r är i mm (direkt från Qualisys)
            r = sample_at_times(time, pos[body], phase_time)  # (3,) i mm
            # v är numerisk derivata av r, alltså i mm/s
            v_mm = v_dict[body]  # (3,) i mm/s
            omega = float(omega_dict[body])
            m = bodies_phys[body]["mass"]
            Izz_m2 = bodies_phys[body]["Izz"]       # kg·m² (från bodies.tsv)
            Izz_mm2 = Izz_m2 * 1e6                  # kg·mm², för att matcha r (mm) och p (kg·mm/s)

            # Linjär rörelsemängd i kg·mm/s
            p_vec = m * v_mm

            # För kinetisk energi vill vi ha m/s -> konvertera v från mm/s till m/s
            v_m = v_mm / 1000.0
            v2_m = float(np.dot(v_m, v_m))
            KE_trans = 0.5 * m * v2_m
            KE_rot = 0.5 * Izz_m2 * omega * omega
            KE_tot = KE_trans + KE_rot

            # Orbitalt L: r (mm) × p (kg·mm/s) -> kg·mm²/s
            r_rel = r - origin
            L_orb_vec = np.cross(r_rel, p_vec)
            L_orb_z = float(np.dot(L_orb_vec, n_hat))

            # Spinn-del i samma kg·mm²/s-enheter
            L_spin_z = Izz_mm2 * omega
            L_tot_z = L_orb_z + L_spin_z

            res[body] = {
                "r": r,
                "v": v_mm,
                "p": p_vec,
                "KE_trans": KE_trans,
                "KE_rot": KE_rot,
                "KE_tot": KE_tot,
                "L_orb_z": L_orb_z,
                "L_spin_z": L_spin_z,
                "L_tot_z": L_tot_z,
            }

        p_sys = res[b1]["p"] + res[b2]["p"]
        KE_trans_sys = res[b1]["KE_trans"] + res[b2]["KE_trans"]
        KE_rot_sys = res[b1]["KE_rot"] + res[b2]["KE_rot"]
        KE_tot_sys = res[b1]["KE_tot"] + res[b2]["KE_tot"]
        L_orb_sys = res[b1]["L_orb_z"] + res[b2]["L_orb_z"]
        L_spin_sys = res[b1]["L_spin_z"] + res[b2]["L_spin_z"]
        L_tot_sys = res[b1]["L_tot_z"] + res[b2]["L_tot_z"]

        return {
            "bodies": res,
            "p_sys": p_sys,
            "KE_trans_sys": KE_trans_sys,
            "KE_rot_sys": KE_rot_sys,
            "KE_tot_sys": KE_tot_sys,
            "L_orb_sys": L_orb_sys,
            "L_spin_sys": L_spin_sys,
            "L_tot_sys": L_tot_sys,
        }

    pre = stats_at_time(t_pre, v_pre, omega_pre)

    # Post: behandla Puck1 + Puck2 som en sammanslagen kropp "Puck3"
    b1, b2 = bodies_order
    m1 = bodies_phys[b1]["mass"]
    m2 = bodies_phys[b2]["mass"]
    m3 = m1 + m2

    # Positioner vid t_post (mm)
    r3 = sample_at_times(time, pos["Puck3"], t_post)

    # Linjär hastighet för Puck3 (mm/s) från polynomanpassningen
    v3_mm = v_post["Puck3"]
    omega3 = float(omega_post["Puck3"])

    # Rörelsemängd efter: (m1 + m2) * v3
    p_sys_post = m3 * v3_mm  # kg·mm/s

    # För kinetisk energi: konvertera till m/s
    v3_m = v3_mm / 1000.0
    v3_sq = float(np.dot(v3_m, v3_m))
    KE_trans_post = 0.5 * m3 * v3_sq

    # Tröghetsmoment för sammanslagen kropp runt dess COM (Puck3)
    # Beräknas med parallellaxelsatsen utifrån Puck1 och Puck2.
    I1 = bodies_phys[b1]["Izz"]  # kg·m²
    I2 = bodies_phys[b2]["Izz"]  # kg·m²

    r1_mm = sample_at_times(time, pos[b1], t_post)
    r2_mm = sample_at_times(time, pos[b2], t_post)
    r3_mm = r3

    r1_m = r1_mm / 1000.0
    r2_m = r2_mm / 1000.0
    r3_m = r3_mm / 1000.0

    d1_vec = r1_m - r3_m
    d2_vec = r2_m - r3_m
    d1_sq = float(np.dot(d1_vec, d1_vec))
    d2_sq = float(np.dot(d2_vec, d2_vec))

    # I3 runt COM3
    I3_m2 = I1 + m1 * d1_sq + I2 + m2 * d2_sq
    I3_mm2 = I3_m2 * 1e6  # kg·mm²

    KE_rot_post = 0.5 * I3_m2 * omega3 * omega3
    KE_tot_post = KE_trans_post + KE_rot_post

    # Rörelsemängdsmoment efter: r3 × p_sys_post + I3 * omega3 (runt origin och n_hat)
    r_rel3 = r3 - origin  # mm
    L_orb_vec_post = np.cross(r_rel3, p_sys_post)  # kg·mm²/s
    L_orb_sys_post = float(np.dot(L_orb_vec_post, n_hat))
    L_spin_sys_post = I3_mm2 * omega3
    L_tot_sys_post = L_orb_sys_post + L_spin_sys_post

    post = {
        "bodies": {
            "Puck3": {
                "r": r3,
                "v": v3_mm,
                "p": p_sys_post,
                "KE_trans": KE_trans_post,
                "KE_rot": KE_rot_post,
                "KE_tot": KE_tot_post,
                "L_orb_z": L_orb_sys_post,
                "L_spin_z": L_spin_sys_post,
                "L_tot_z": L_tot_sys_post,
            }
        },
        "p_sys": p_sys_post,
        "KE_trans_sys": KE_trans_post,
        "KE_rot_sys": KE_rot_post,
        "KE_tot_sys": KE_tot_post,
        "L_orb_sys": L_orb_sys_post,
        "L_spin_sys": L_spin_sys_post,
        "L_tot_sys": L_tot_sys_post,
    }

    return {"pre": pre, "post": post}


# =============================================================================
# Plot: scatter i planet med orienteringslinjer
# =============================================================================


def plot_plane_scatter_with_orientation(
    pos: Dict[str, np.ndarray],
    orient: Dict[str, dict],
    bodies_order: Tuple[str, str],
    title: str = "",
):
    """Rita 2D-bana (x, y) för varje puck + linjer som visar orienteringen i x–y-planet."""
    b1, b2 = bodies_order
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {b1: "C0", b2: "C1"}

    for body in (b1, b2):
        com = pos[body]        # (N,3)
        com2d = com[:, :2]     # (N,2) -> (x, y)
        rel2d = orient[body]["rel_vec_xy"]  # (N,2)
        c = colors[body]

        # Banan (linje + punkter)
        ax.plot(com2d[:, 0], com2d[:, 1], "-", label=f"{body} bana", color=c, alpha=0.8)
        ax.scatter(com2d[:, 0], com2d[:, 1], s=5, color=c, alpha=0.5)

        # Orienteringslinjer (var n:te frame för läsbarhet)
        N = com2d.shape[0]
        step = max(N // 80, 1)  # ungefär upp till ~80 linjer
        idxs = np.arange(0, N, step)
        for idx in idxs:
            x0, y0 = com2d[idx]
            dx, dy = rel2d[idx]
            scale = 0.05  # skala ned riktningsvektorn för tydlighet
            ax.plot(
                [x0, x0 + scale * dx],
                [y0, y0 + scale * dy],
                color=c,
                alpha=0.3,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()


# =============================================================================
# Plotta θ(t) med fits och markerade kollisionstider
# =============================================================================


def plot_theta_with_fits(
    time: np.ndarray,
    orient: Dict[str, dict],
    theta_fits: Dict[str, dict],
    bodies_order: Tuple[str, str],
    t_p_minus: float | None = None,
    t_p_plus: float | None = None,
    t_p_star: float | None = None,
    title: str = "",
    t_window_minus: float | None = None,
    t_window_plus: float | None = None,
):
    """Plotta rotationsvinkeln θ(t) (i grader) för båda puckarna med anpassade kurvor.

    Här används t_p_minus/t_p_plus som de viktigaste markeringarna (nu tolkar vi
    dem som t_P- och t_P+ i plottarna).
    """
    b1, b2 = bodies_order
    base_bodies = [b1, b2]
    extra_bodies = [b for b in orient.keys() if b not in base_bodies]
    bodies = base_bodies + extra_bodies

    color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    colors = {body: color_cycle[i % len(color_cycle)] for i, body in enumerate(bodies)}

    fig, ax = plt.subplots(figsize=(10, 5))

    for body in bodies:
        theta_rad = orient[body]["theta"]
        theta_deg = np.degrees(theta_rad)
        fit = theta_fits.get(body)
        c = colors[body]

        # Rådata i grader
        ax.plot(time, theta_deg, label=f"{body} θ(t)", color=c, alpha=0.7)

        # Anpassade kurvor före/efter (konvertera fit från rad -> deg)
        if fit is not None:
            pre_t = fit["pre_t"]
            post_t = fit["post_t"]
            pre_fit_deg = np.degrees(fit["pre_fit"])
            post_fit_deg = np.degrees(fit["post_fit"])

            ax.plot(
                pre_t,
                pre_fit_deg,
                linestyle=":",
                color=c,
                alpha=0.9,
                label=f"{body} fit pre",
            )
            ax.plot(
                post_t,
                post_fit_deg,
                linestyle="--",
                color=c,
                alpha=0.9,
                label=f"{body} fit post",
            )

    # Vertikala linjer
    if t_p_minus is not None:
        ax.axvline(t_p_minus, linestyle="--", color="k", alpha=0.7, label="t_P-")
    if t_p_plus is not None:
        ax.axvline(t_p_plus, linestyle="--", color="grey", alpha=0.7, label="t_P+")
    if t_p_star is not None:
        ax.axvline(t_p_star, linestyle="-.", color="red", alpha=0.8, label="t_min (dist)")
    if t_window_minus is not None:
        ax.axvline(
            t_window_minus,
            linestyle="-",
            color="orange",
            alpha=0.5,
            label="t_- (fönsterkant)",
        )
    if t_window_plus is not None:
        ax.axvline(
            t_window_plus,
            linestyle="-",
            color="orange",
            alpha=0.5,
            label="t_+ (fönsterkant)",
        )

    ax.set_xlabel("Tid (s)")
    ax.set_ylabel("θ (°)")
    ax.grid(True)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    plt.tight_layout()


# =============================================================================
# Plot: u, v, e (≈ x, y, z) för båda puckarna
# =============================================================================


def compute_velocity_series(
    time: np.ndarray, pos: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Approximerar dr/dt för varje kropp med numerisk derivata."""
    velocities: Dict[str, np.ndarray] = {}
    for body, r in pos.items():
        velocities[body] = np.gradient(r, time, axis=0)
    return velocities


def compute_acceleration_series(
    time: np.ndarray, velocities: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Approximerar dv/dt (acceleration) för varje kropp."""
    accels: Dict[str, np.ndarray] = {}
    for body, v in velocities.items():
        accels[body] = np.gradient(v, time, axis=0)
    return accels


# =============================================================================
# Plot: positionskomponenter (x, y, z) + fits
# =============================================================================


def plot_position_components(
    time: np.ndarray,
    pos: Dict[str, np.ndarray],
    bodies_order: Tuple[str, str],
    pos_fits: Dict[str, Dict[int, dict]] | None = None,
    t_p_minus: float | None = None,
    t_p_plus: float | None = None,
    t_p_star: float | None = None,
    title: str = "",
    t_window_minus: float | None = None,
    t_window_plus: float | None = None,
):
    """Plotta x/y/z-positioner och eventuella polynomanpassningar.

    Vertikala linjer vid t_p_minus/t_p_plus tolkas här som t_P-/t_P+.
    """
    b1, b2 = bodies_order
    base_bodies = [b1, b2]
    extra_bodies = [b for b in pos.keys() if b not in base_bodies]
    bodies = base_bodies + extra_bodies

    color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    colors = {body: color_cycle[i % len(color_cycle)] for i, body in enumerate(bodies)}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    comp_labels = ["x", "y", "z"]
    window_labels_done = False
    for comp_idx, (ax, label) in enumerate(zip(axes, comp_labels)):
        for body in bodies:
            c = colors[body]
            data = pos[body][:, comp_idx]
            line_label = body if comp_idx == 0 else None
            ax.plot(time, data, label=line_label, color=c, alpha=0.9)

            if pos_fits is not None:
                body_fit = pos_fits.get(body, {})
                fit = body_fit.get(comp_idx)
                if fit is not None:
                    pre_t = fit["pre_t"]
                    post_t = fit["post_t"]
                    pre_pos = fit["pre_fit"]
                    post_pos = fit["post_fit"]
                    ax.plot(pre_t, pre_pos, linestyle=":", color=c, alpha=0.9)
                    ax.plot(post_t, post_pos, linestyle="--", color=c, alpha=0.9)

        if t_p_minus is not None:
            ax.axvline(t_p_minus, linestyle="--", color="k", alpha=0.7, label="t_P-")
        if t_p_plus is not None:
            ax.axvline(t_p_plus, linestyle="--", color="grey", alpha=0.7, label="t_P+")
        if t_p_star is not None:
            ax.axvline(t_p_star, linestyle="-.", color="red", alpha=0.8, label="t_min (dist)")
        if t_window_minus is not None:
            ax.axvline(
                t_window_minus,
                linestyle="-",
                color="orange",
                alpha=0.5,
                label="t_- (fönsterkant)" if not window_labels_done else None,
            )
        if t_window_plus is not None:
            ax.axvline(
                t_window_plus,
                linestyle="-",
                color="orange",
                alpha=0.5,
                label="t_+ (fönsterkant)" if not window_labels_done else None,
            )
        window_labels_done = True

        ax.set_ylabel(f"{label} (mm)")
        ax.grid(True)

    axes[-1].set_xlabel("Tid (s)")
    if title:
        fig.suptitle(title)
    axes[0].legend(loc="best")
    plt.tight_layout()



# =============================================================================
# Plot: hastighetskomponenter (v_x, v_y, v_z) + fits
# =============================================================================


def plot_velocity_components(
    time: np.ndarray,
    velocities: Dict[str, np.ndarray],
    bodies_order: Tuple[str, str],
    vel_fits: Dict[str, Dict[int, dict]] | None = None,
    vel_star_points: Dict[str, Dict[str, np.ndarray]] | None = None,
    v_pre: Dict[str, np.ndarray] | None = None,
    v_post: Dict[str, np.ndarray] | None = None,
    t_p_minus: float | None = None,
    t_p_plus: float | None = None,
    t_p_star: float | None = None,
    title: str = "",
    t_window_minus: float | None = None,
    t_window_plus: float | None = None,
):
    """Plotta v_x/v_y/v_z och markera polynomanpassningar.

    vertikala linjer vid t_p_minus/t_p_plus tolkas här som t_P-/t_P+.
    """
    b1, b2 = bodies_order
    base_bodies = [b1, b2]
    extra_bodies = [b for b in velocities.keys() if b not in base_bodies]
    bodies = base_bodies + extra_bodies

    color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    colors = {body: color_cycle[i % len(color_cycle)] for i, body in enumerate(bodies)}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    comp_labels = ["v_x", "v_y", "v_z"]
    # Används för att bara sätta legend-label för t_- / t_+ en gång
    window_labels_done = False
    for comp_idx, (ax, label) in enumerate(zip(axes, comp_labels)):
        for body in bodies:
            c = colors[body]
            data = velocities[body][:, comp_idx]
            line_label = body if comp_idx == 0 else None
            ax.plot(time, data, label=line_label, color=c, alpha=0.9)

            if vel_fits is not None:
                body_fit = vel_fits.get(body, {})
                fit = body_fit.get(comp_idx)
                if fit is not None:
                    pre_t = fit["pre_t"]
                    post_t = fit["post_t"]
                    dp_pre = fit["dp_pre"]
                    dp_post = fit["dp_post"]
                    pre_vel = np.polyval(dp_pre, pre_t)
                    post_vel = np.polyval(dp_post, post_t)
                    ax.plot(pre_t, pre_vel, linestyle=":", color=c, alpha=0.9)
                    ax.plot(post_t, post_vel, linestyle="--", color=c, alpha=0.9)

            # Markera de hastigheter som bestämts av polynomanpassningen vid t_P- och t_P+
            # (dvs v_pre och v_post från analysen) med prickar.
            if v_pre is not None and v_post is not None:
                if t_p_minus is not None:
                    v_pre_body = v_pre.get(body)
                    if v_pre_body is not None:
                        y_pre_point = v_pre_body[comp_idx]
                        point_label_pre = "v_pre (t_P-)" if (body == bodies[0] and comp_idx == 0) else None
                        ax.scatter(
                            t_p_minus,
                            y_pre_point,
                            color=c,
                            edgecolors="k",
                            zorder=5,
                            label=point_label_pre,
                        )
                if t_p_plus is not None:
                    v_post_body = v_post.get(body)
                    if v_post_body is not None:
                        y_post_point = v_post_body[comp_idx]
                        point_label_post = "v_post (t_P+)" if (body == bodies[0] and comp_idx == 0) else None
                        ax.scatter(
                            t_p_plus,
                            y_post_point,
                            color=c,
                            edgecolors="k",
                            marker="s",
                            zorder=5,
                            label=point_label_post,
                        )

        if t_p_minus is not None:
            ax.axvline(t_p_minus, linestyle="--", color="k", alpha=0.7, label="t_P-")
        if t_p_plus is not None:
            ax.axvline(t_p_plus, linestyle="--", color="grey", alpha=0.7, label="t_P+")
        if t_p_star is not None:
            ax.axvline(t_p_star, linestyle="-.", color="red", alpha=0.8, label="t_min (dist)")
        if t_window_minus is not None:
            ax.axvline(
                t_window_minus,
                linestyle="-",
                color="orange",
                alpha=0.5,
                label="t_- (fönsterkant)" if not window_labels_done else None,
            )
        if t_window_plus is not None:
            ax.axvline(
                t_window_plus,
                linestyle="-",
                color="orange",
                alpha=0.5,
                label="t_+ (fönsterkant)" if not window_labels_done else None,
            )
        window_labels_done = True

        ax.set_ylabel(f"{label} (mm/s)")
        ax.grid(True)

    axes[-1].set_xlabel("Tid (s)")
    if title:
        fig.suptitle(title)
    axes[0].legend(loc="best")
    plt.tight_layout()

# =============================================================================
# Laddning av bodies.tsv (massor & Izz)
# =============================================================================


def load_bodies(path: Path) -> Dict[str, dict]:
    """Läs in TSV med kolumner: name, mass_kg, Izz_kgm2."""
    df = pd.read_csv(path, sep="\t")
    required = {"name", "mass_kg", "Izz_kgm2"}
    if not required <= set(df.columns):
        raise ValueError(f"bodies-filen måste ha kolumnerna: {required}")

    bodies: Dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        m = float(row["mass_kg"])
        Izz = float(row["Izz_kgm2"])
        bodies[name] = {"mass": m, "Izz": Izz}
    return bodies


def infer_dist_threshold_from_bodies(path: Path) -> float | None:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return None
    if "dist_threshold" not in df.columns:
        return None
    vals = df["dist_threshold"].dropna()
    if vals.empty:
        return None
    try:
        return float(vals.iloc[0])
    except Exception:
        return None


# =============================================================================
# CLI / hög-nivåanalys
# =============================================================================


@dataclass
class Config:
    bodies_path: Path
    dist_threshold: float
    max_gap_frames: int = 3
    make_plots: bool = True
    emit_summary: bool = True


def analyze_air_table(file_path: Path, cfg: Config) -> int:
    # 1) Läs 3D-markerdata (t.ex. cm1, ed1, cm2, ed2)
    time, marker_pos_all, header = load_qtm_6d_file_version(file_path)

    # 2) Läs massor och Izz
    bodies_phys_all = load_bodies(cfg.bodies_path)

    # 3) Puckar och markörer
    for required_body in ("Puck1", "Puck2"):
        if required_body not in bodies_phys_all:
            raise ValueError(
                f"Bodies-filen saknar {required_body!r}. Hittade: {list(bodies_phys_all.keys())}"
            )

    required_markers = ["cm1", "ed1", "cm2", "ed2"]
    for m_name in required_markers:
        if m_name not in marker_pos_all:
            raise ValueError(
                f"Markören {m_name!r} saknas i QTM-filen. Hittade: {list(marker_pos_all.keys())}"
            )

    bodies_order: Tuple[str, str] = ("Puck1", "Puck2")
    print(
        f"Kroppar (puckar): {bodies_order[0]} (cm1/ed1) och "
        f"{bodies_order[1]} (cm2/ed2)"
    )

    # COM-positioner: använd cm1/cm2 som (approximerade) masscentrum-markörer
    pos: Dict[str, np.ndarray] = {
        "Puck1": marker_pos_all["cm1"],
        "Puck2": marker_pos_all["cm2"],
    }

    # "Kantmarkörer" (för orientering / rotation) – ed1/ed2
    edge_pos: Dict[str, np.ndarray] = {
        "Puck1": marker_pos_all["ed1"],
        "Puck2": marker_pos_all["ed2"],
    }

    # Imaginär sammanslagen puck: Puck3 (medel av Puck1 och Puck2)
    pos["Puck3"] = 0.5 * (pos["Puck1"] + pos["Puck2"])

    velocities = compute_velocity_series(time, pos)
    accels = compute_acceleration_series(time, velocities)

    # Massor och tröghetsmoment
    bodies_phys = {b: bodies_phys_all[b] for b in bodies_order}

    # 4) Välj referensorigo och axel för L i globala koordinater
    all_points = np.concatenate(list(pos.values()), axis=0)  # (N_tot, 3)
    origin = all_points.mean(axis=0)
    n_hat = np.array([0.0, 0.0, 1.0])  # global z-axel

    print("\nReferens för L-beräkningar:")
    print(f"  origin  = {origin}")
    print(f"  n_hat   = {n_hat}  (axel för L)")

    # 5) Kollisiondetektion (fönster t_- och t_+)
    collision = find_collision_interval(
        time,
        pos,
        dist_threshold=cfg.dist_threshold,
        max_gap_frames=cfg.max_gap_frames,
        bodies_pair=bodies_order,
    )

    velocity_fits: Dict[str, Dict[int, dict]] | None = None
    summary_row: dict | None = None

    # För impact-tider (t_P-, t_P+)
    t_P_minus: float | None = None
    t_P_plus: float | None = None

    # Initialisera fönsterkanter t_- och t_+ till None (om ingen kollision hittas)
    t_minus: float | None = None
    t_plus: float | None = None

    if collision["threshold_hit"]:
        t_minus = collision["t_p_minus"]   # fönsterkant t_-
        t_plus = t_minus + 0.05            # fönsterkant t_+
        # Ursprungliga kollisionsindex (används för accelerations-regionen)
        coll_start_idx = collision["start_idx"]
        coll_end_idx = collision["end_idx"]
        t_min_dist = collision["t_min"]

        print(
            f"\nKollisionsfönster hittat när dist < {cfg.dist_threshold:.4f}:"
            f"\n  t_-  = {t_minus:.6f} s (första under tröskel)"
            f"\n  t_+  = {t_plus:.6f} s (t_- + 0.05 s)"
            f"\n  min dist = {collision['d_min']:.6f} vid t = {t_min_dist:.6f} s "
            f"(index {collision['idx_min']})"
        )

        # Index för anpassningsfönster (pre: före t_-, post: efter t_+)
        fit_start_idx = int(np.argmin(np.abs(time - t_minus)))
        fit_end_idx = int(np.argmin(np.abs(time - t_plus)))

        # 6) Bestäm t_P- och t_P+ från accelerationsspik
        # Beräkna |a| för båda kropparna
        a_mag_all_list = []
        a_mag_per_body: Dict[str, np.ndarray] = {}
        for body in bodies_order:
            a = accels[body]
            a_mag = np.linalg.norm(a, axis=1)
            a_mag_per_body[body] = a_mag
            a_mag_all_list.append(a_mag)

        a_mag_all = np.concatenate(a_mag_all_list)
        # Data-driven tröskel: median + 4*std, fallback till 95-percentil
        base = float(np.median(a_mag_all))
        std = float(np.std(a_mag_all))
        if std > 0:
            a_threshold = base + 1 * std
        else:
            a_threshold = float(np.percentile(a_mag_all, 95.0))

        print(
            f"\nAccelerationströskel för stöt: |a| > {a_threshold:.3e} mm/s² "
            f"(median={base:.3e}, std={std:.3e})"
        )

        N = len(time)
        # Begränsa sökning av impact till ett område runt kollisionen
        region_start = max(0, coll_start_idx - 5)
        region_end = min(N - 1, coll_end_idx + 5)
        mask_region = np.zeros(N, dtype=bool)
        mask_region[region_start : region_end + 1] = True

        mask_impact = np.zeros(N, dtype=bool)
        for body in bodies_order:
            mask_impact |= a_mag_per_body[body] >= a_threshold

        mask = mask_impact & mask_region
        idx_imp = np.where(mask)[0]

        if idx_imp.size == 0:
            print(
                "Varning: Hittade inga accelerationer över tröskeln nära kollisionen, "
                "använder fönsterkanten t_- som t_P- och t_+ som t_P+."
            )
            t_P_minus = t_minus
        else:
            t_P_minus = float(time[idx_imp[0]])

        # Sätt alltid t_P+ = t_+ (fönsterkanten)
        t_P_plus = t_plus

        print(
            f"\nImpacttider (t_P- baserat på acceleration, t_P+ = t_+):"
            f"\n  t_P- = {t_P_minus:.6f} s"
            f"\n  t_P+ = {t_P_plus:.6f} s"
        )

        # Bygg kantmarkör för Puck3 och frys valet efter kollisionstid t_P+
        cm1 = pos["Puck1"]
        cm2 = pos["Puck2"]
        cm3 = pos["Puck3"]
        ed1 = edge_pos["Puck1"]
        ed2 = edge_pos["Puck2"]

        # Välj, för varje tidssteg, den kantmarkör som ligger längst från cm3
        d1 = np.linalg.norm(ed1 - cm3, axis=1)
        d2 = np.linalg.norm(ed2 - cm3, axis=1)
        use_ed1 = d1 >= d2  # True där ed1 är längst från cm3, annars ed2

        # Frys valet vid och efter t_P+ så att vi inte hoppar mellan ed1/ed2 i sticky-fasen
        idx_lock = int(np.argmin(np.abs(time - t_P_plus)))
        use_ed1[idx_lock:] = use_ed1[idx_lock]

        edge3 = np.empty_like(cm3)
        edge3[use_ed1] = ed1[use_ed1]
        edge3[~use_ed1] = ed2[~use_ed1]
        edge_pos["Puck3"] = edge3

        # 7) Orienteringsvinklar och ω från markörer (COM + kant) i x–y-planet
        orient = compute_orientation_from_markers(
            time,
            pos,
            edge_pos,
        )

        # 8) Linjära hastigheter: polynomanpassning i global X, Y, Z (alla kroppar i pos)
        fit_bodies = list(pos.keys())
        v_pre: Dict[str, np.ndarray] = {}
        v_post: Dict[str, np.ndarray] = {}
        velocity_fit_segments: Dict[str, Dict[int, dict]] = {body: {} for body in fit_bodies}

        for body in fit_bodies:
            r = pos[body]  # (N,3)
            v_pre_body = np.zeros(3, dtype=float)
            v_post_body = np.zeros(3, dtype=float)
            for comp in range(3):
                res_fit = polyfit_pre_post(
                    time,
                    r[:, comp],
                    start_idx=fit_start_idx,
                    end_idx=fit_end_idx,
                    t_p_minus=t_minus,
                    t_p_plus=t_plus,
                    window=20,
                    deg=2,
                )
                # Nya definitionen: evaluera derivatan vid t_P- resp t_P+
                dp_pre = res_fit["dp_pre"]
                dp_post = res_fit["dp_post"]
                v_pre_body[comp] = float(np.polyval(dp_pre, t_P_minus))
                v_post_body[comp] = float(np.polyval(dp_post, t_P_plus))

                velocity_fit_segments[body][comp] = res_fit

            v_pre[body] = v_pre_body
            v_post[body] = v_post_body

        velocity_fits = velocity_fit_segments

        # 9) Vinkelhastigheter: polynomanpassning på θ(t) i planet (alla kroppar i orient)
        fit_bodies_theta = list(orient.keys())
        omega_pre: Dict[str, float] = {}
        omega_post: Dict[str, float] = {}
        theta_fits: Dict[str, dict] = {}
        for body in fit_bodies_theta:
            theta = orient[body]["theta"]
            res_fit_theta = polyfit_pre_post(
                time,
                theta,
                start_idx=fit_start_idx,
                end_idx=fit_end_idx,
                t_p_minus=t_minus,
                t_p_plus=t_plus,
                window=20,
                deg=2,
            )
            dp_pre_theta = res_fit_theta["dp_pre"]
            dp_post_theta = res_fit_theta["dp_post"]
            omega_pre[body] = float(np.polyval(dp_pre_theta, t_P_minus))
            omega_post[body] = float(np.polyval(dp_post_theta, t_P_plus))

            theta_fits[body] = res_fit_theta

        # 10) Rörelsemängd & L vid t_P- och t_P+
        stats = compute_momentum_energy_L(
            time,
            pos,
            bodies_phys,
            origin,
            n_hat,
            t_pre=t_P_minus,
            t_post=t_P_plus,
            bodies_order=bodies_order,
            v_pre=v_pre,
            v_post=v_post,
            omega_pre=omega_pre,
            omega_post=omega_post,
        )

        def fmt_vec(v: np.ndarray) -> str:
            return f"({v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f})"

        # Systemets rörelsemängd i globala (x, y, z)-komponenter
        p_sys_pre = stats["pre"]["p_sys"]
        p_sys_post = stats["post"]["p_sys"]

        # Norm av systemets rörelsemängd före kollision (används för relativ Δp)
        p_pre_mag = float(np.linalg.norm(p_sys_pre))

        print("\nLinjär rörelsemängd i (x, y, z) och kinetisk energi (evaluerat vid t_P- / t_P+):")
        print(
            f"  pre  (t_P-) : p_sys = {fmt_vec(p_sys_pre)} kg·mm/s, "
            f"|p_sys| = {p_pre_mag:.6e} kg·mm/s, "
            f"KE_trans = {stats['pre']['KE_trans_sys']:.6e} J, "
            f"KE_rot = {stats['pre']['KE_rot_sys']:.6e} J, "
            f"KE_tot = {stats['pre']['KE_tot_sys']:.6e} J"
        )
        print(
            f"  post (t_P+): p_sys = {fmt_vec(p_sys_post)} kg·mm/s, "
            f"KE_trans = {stats['post']['KE_trans_sys']:.6e} J, "
            f"KE_rot = {stats['post']['KE_rot_sys']:.6e} J, "
            f"KE_tot = {stats['post']['KE_tot_sys']:.6e} J"
        )

        # Skriv ut de bestämda linjära hastigheterna i (x, y, z) för varje puck
        print("\nBestämda linjära hastigheter i (x, y, z) vid t_P- och t_P+:")
        for body in bodies_order:
            v_pre_body = v_pre[body]
            v_post_body = v_post[body]

            print(
                f"  {body}:"
                f"\n    v(t_P-) = {fmt_vec(v_pre_body)} m/s"
                f"\n    v(t_P+) = {fmt_vec(v_post_body)} m/s"
            )

        # Skriv ut de bestämda vinkelhastigheterna (från θ)
        print("\nBestämda vinkelhastigheter (från θ i planet, rad/s) vid t_P- och t_P+:")
        for body in bodies_order:
            print(
                f"  {body}: ω(t_P-) = {omega_pre[body]:.6f}, "
                f"ω(t_P+) = {omega_post[body]:.6f}"
            )

        print("\nRörelsemängdsmoment runt n_hat (planet normal):")
        print(
            f"  pre  (t_P-): L_orb_sys = {stats['pre']['L_orb_sys']:.6e} kg·mm²/s,  "
            f"L_spin_sys = {stats['pre']['L_spin_sys']:.6e} kg·mm²/s,  "
            f"L_tot_sys = {stats['pre']['L_tot_sys']:.6e} kg·mm²/s"
        )
        print(
            f"  post (t_P+): L_orb_sys = {stats['post']['L_orb_sys']:.6e} kg·mm²/s,  "
            f"L_spin_sys = {stats['post']['L_spin_sys']:.6e} kg·mm²/s,  "
            f"L_tot_sys = {stats['post']['L_tot_sys']:.6e} kg·mm²/s"
        )

        # Skillnader i systemstorheter mellan t_P- och t_P+
        #dp_vec = p_sys_post - p_sys_pre
        #dp_u = float(dp_vec[0])   # komponent längs x (u)
        #dp_v = float(dp_vec[1])   # komponent längs y (v)
        #dp_e = float(dp_vec[2])   # komponent längs z (e ≈ n_hat)

        dp_u = np.abs(p_sys_post[0]) - np.abs(p_sys_pre[0])
        dp_v = np.abs(p_sys_post[1]) - np.abs(p_sys_pre[1])
        dp_e = np.abs(p_sys_post[2]) - np.abs(p_sys_pre[2])

        dp_E = float(stats["post"]["KE_tot_sys"] - stats["pre"]["KE_tot_sys"])
        dp_L_n = float(stats["post"]["L_tot_sys"] - stats["pre"]["L_tot_sys"])

        print(
            "\nSkillnader mellan pre (t_P-) och post (t_P+):"
            f"\n  dp_u = {dp_u:.6e} kg·mm/s"
            f"\n  dp_v = {dp_v:.6e} kg·mm/s"
            f"\n  dp_e = {dp_e:.6e} kg·mm/s"
            f"\n  dp_E = {dp_E:.6e} J"
            f"\n  dp_L_n = {dp_L_n:.6e} kg·mm²/s"
        )

        # En enkel maskinläsbar summary-rad (TAB-separerad) för scripts/run_batch.py
        # Kolumner: filnamn, dp_u, dp_v, dp_e, dp_E, dp_L_n, |p_pre|, KE_pre_tot, KE_post_tot,
        #           L_pre_n, L_post_n,
        #           omega_pre_mean, omega_post_mean,
        #           omega1_pre, omega2_pre, omega1_post, omega2_post,
        #           v1_pre_u, v1_pre_v, v2_pre_u, v2_pre_v, v1_post_u, v1_post_v, v2_post_u, v2_post_v
        KE_pre_tot = float(stats["pre"]["KE_tot_sys"])
        KE_post_tot = float(stats["post"]["KE_tot_sys"])
        L_pre_n = float(stats["pre"]["L_tot_sys"])
        L_post_n = float(stats["post"]["L_tot_sys"])

        # Rörelsemängd före/efter i komponenter
        p_pre_x, p_pre_y, p_pre_z = p_sys_pre
        p_post_x, p_post_y, p_post_z = p_sys_post

        # Skillnader i belopp av rörelsemängdskompogör nenter
        dp_u = abs(p_post_x) - abs(p_pre_x)
        dp_v = abs(p_post_y) - abs(p_pre_y)
        dp_e = abs(p_post_z) - abs(p_pre_z)

        # Skillnader i total energi och L längs n_hat
        dp_E = KE_post_tot - KE_pre_tot
        dp_L_n = L_post_n - L_pre_n

        # Storheter för SUMMARY-rad
        # Beloppet av rörelsemängden före
        p_pre_mag = float(np.linalg.norm(p_sys_pre))

        # Vinkelhastigheter för Puck1 och Puck2 före/efter
        omega1_pre = float(omega_pre.get("Puck1", np.nan))
        omega2_pre = float(omega_pre.get("Puck2", np.nan))
        omega1_post = float(omega_post.get("Puck1", np.nan))
        omega2_post = float(omega_post.get("Puck2", np.nan))

        # Medelvärden av vinkelhastighet (Puck1, Puck2)
        omega_pre_mean = 0.5 * (omega1_pre + omega2_pre)
        omega_post_mean = 0.5 * (omega1_post + omega2_post)

        # Projektera linjära hastigheter på u (x) och v (y)
        v1_pre_u = v_pre[bodies_order[0]][0]
        v1_pre_v = v_pre[bodies_order[0]][1]
        v2_pre_u = v_pre[bodies_order[1]][0]
        v2_pre_v = v_pre[bodies_order[1]][1]

        v1_post_u = v_post[bodies_order[0]][0]
        v1_post_v = v_post[bodies_order[0]][1]
        v2_post_u = v_post[bodies_order[1]][0]
        v2_post_v = v_post[bodies_order[1]][1]

        summary_row = {
            "file": file_path.name,
            "dp_u_kgm_s": float(dp_u),
            "dp_v_kgm_s": float(dp_v),
            "dp_e_kgm_s": float(dp_e),
            "dp_E_J": float(dp_E),
            "dp_L_n_kgm2_s": float(dp_L_n),
            "p_pre_mag": float(p_pre_mag),
            "KE_pre_tot_J": float(KE_pre_tot),
            "KE_post_tot_J": float(KE_post_tot),
            "L_pre_n_kgm2_s": float(L_pre_n),
            "L_post_n_kgm2_s": float(L_post_n),
            "omega_pre_mean_rad_s": float(omega_pre_mean),
            "omega_post_mean_rad_s": float(omega_post_mean),
            "omega1_pre_rad_s": float(omega1_pre),
            "omega2_pre_rad_s": float(omega2_pre),
            "omega1_post_rad_s": float(omega1_post),
            "omega2_post_rad_s": float(omega2_post),
            "v1_pre_u": float(v1_pre_u),
            "v1_pre_v": float(v1_pre_v),
            "v2_pre_u": float(v2_pre_u),
            "v2_pre_v": float(v2_pre_v),
            "v1_post_u": float(v1_post_u),
            "v1_post_v": float(v1_post_v),
            "v2_post_u": float(v2_post_u),
            "v2_post_v": float(v2_post_v),
        }

        if cfg.emit_summary:
            print(
                f"SUMMARY\t{summary_row['file']}\t"
                f"{summary_row['dp_u_kgm_s']:.6e}\t{summary_row['dp_v_kgm_s']:.6e}\t"
                f"{summary_row['dp_e_kgm_s']:.6e}\t{summary_row['dp_E_J']:.6e}\t"
                f"{summary_row['dp_L_n_kgm2_s']:.6e}\t"
                f"{summary_row['p_pre_mag']:.6e}\t{summary_row['KE_pre_tot_J']:.6e}\t"
                f"{summary_row['KE_post_tot_J']:.6e}\t"
                f"{summary_row['L_pre_n_kgm2_s']:.6e}\t{summary_row['L_post_n_kgm2_s']:.6e}\t"
                f"{summary_row['omega_pre_mean_rad_s']:.6e}\t{summary_row['omega_post_mean_rad_s']:.6e}\t"
                f"{summary_row['omega1_pre_rad_s']:.6e}\t{summary_row['omega2_pre_rad_s']:.6e}\t"
                f"{summary_row['omega1_post_rad_s']:.6e}\t{summary_row['omega2_post_rad_s']:.6e}\t"
                f"{summary_row['v1_pre_u']:.6e}\t{summary_row['v1_pre_v']:.6e}\t"
                f"{summary_row['v2_pre_u']:.6e}\t{summary_row['v2_pre_v']:.6e}\t"
                f"{summary_row['v1_post_u']:.6e}\t{summary_row['v1_post_v']:.6e}\t"
                f"{summary_row['v2_post_u']:.6e}\t{summary_row['v2_post_v']:.6e}"
            )

        # Spara rotationsfits för plottning
        collision["theta_fits"] = theta_fits
    else:
        print(
            f"\nIngen kollision (dist aldrig under tröskel {cfg.dist_threshold:.4f})."
            f"\n  min dist = {collision['d_min']:.6f} vid t = {collision['t_min']:.6f} s "
            f"(index {collision['idx_min']})"
        )
        # Om ingen kollision: bygg enkel kantmarkör för Puck3 (utan frysning) för plottar
        cm1 = pos["Puck1"]
        cm2 = pos["Puck2"]
        cm3 = pos["Puck3"]
        ed1 = edge_pos["Puck1"]
        ed2 = edge_pos["Puck2"]

        d1 = np.linalg.norm(ed1 - cm3, axis=1)
        d2 = np.linalg.norm(ed2 - cm3, axis=1)
        use_ed1 = d1 >= d2

        edge3 = np.empty_like(cm3)
        edge3[use_ed1] = ed1[use_ed1]
        edge3[~use_ed1] = ed2[~use_ed1]
        edge_pos["Puck3"] = edge3

        # Beräkna orient för plottar
        orient = compute_orientation_from_markers(
            time,
            pos,
            edge_pos,
        )
    # 11) Plot
    if cfg.make_plots:
        # t_min (dist-minimum) används ev. som extra markör
        t_min = collision["t_min"] if collision["threshold_hit"] else None

        # För plottar: om vi har t_P-/t_P+, använd dem, annars None
        t_plot_minus = t_P_minus if t_P_minus is not None else None
        t_plot_plus = t_P_plus if t_P_plus is not None else None

        # Plotta x, y, z med anpassade funktioner om vi har fits
        plot_position_components(
            time,
            pos,
            bodies_order=bodies_order,
            pos_fits=velocity_fits,
            t_p_minus=t_plot_minus,
            t_p_plus=t_plot_plus,
            t_p_star=t_min,
            title=f"{file_path.name} – positioner",
            t_window_minus=t_minus,
            t_window_plus=t_plus,
        )

        # Plotta vx, vy, vz med anpassade funktioner om vi har fits
        plot_velocity_components(
            time,
            velocities,
            bodies_order=bodies_order,
            vel_fits=velocity_fits,
            vel_star_points=None,
            v_pre=v_pre,
            v_post=v_post,
            t_p_minus=t_plot_minus,
            t_p_plus=t_plot_plus,
            t_p_star=t_min,
            title=f"{file_path.name} – hastigheter",
            t_window_minus=t_minus,
            t_window_plus=t_plus,
        )

        # Plotta banor i x–y-planet med orienteringslinjer
        title = f"{file_path.name} – banor i x–y-planet"
        plot_plane_scatter_with_orientation(
            pos,
            orient,
            bodies_order=bodies_order,
            title=title,
        )

        # Plotta rotation θ(t) med anpassade kurvor om vi har kollision och fits
        if collision.get("threshold_hit") and "theta_fits" in collision:
            plot_theta_with_fits(
                time,
                orient,
                collision["theta_fits"],
                bodies_order=bodies_order,
                t_p_minus=t_plot_minus,
                t_p_plus=t_plot_plus,
                t_p_star=t_min,
                title=f"{file_path.name} – rotation (θ)",
                t_window_minus=t_minus,
                t_window_plus=t_plus,
            )

        plt.show()

    return {
        "summary": summary_row,
        "collision": collision,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analysera QTM-3D-TSV för luftbord: 2D-rörelse, kollision, "
            "rörelsemängd, kinetisk energi och rörelsemängdsmoment runt planet."
        )
    )
    parser.add_argument("file", help="QTM FILE_VERSION TSV med 3D marker-data (cm1, ed1, cm2, ed2).")
    parser.add_argument(
        "--bodies",
        required=True,
        type=str,
        help="TSV med kolumner: name, mass_kg, Izz_kgm2 (t.ex. bodies.tsv).",
    )
    parser.add_argument(
        "--dist-threshold",
        type=float,
        help="Avståndströskel (valfritt, hämtas annars från bodies-filen).",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
        help="Max antal frames mellan dist<threshold-segment som ändå ska slås ihop (default: 3).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skapa inte plottar.",
    )

    args = parser.parse_args(argv)

    dist_threshold = args.dist_threshold
    if dist_threshold is None:
        dist_threshold = infer_dist_threshold_from_bodies(Path(args.bodies))
    if dist_threshold is None:
        print(
            "Fel: ingen distans-tröskel angiven och ingen kolumn 'dist_threshold' hittades i bodies-filen."
        )
        return 1

    cfg = Config(
        bodies_path=Path(args.bodies),
        dist_threshold=dist_threshold,
        max_gap_frames=args.max_gap_frames,
        make_plots=not args.no_plots,
    )

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Fel: filen {file_path} finns inte.")
        return 1

    try:
        analyze_air_table(file_path, cfg)
    except Exception as e:
        print(f"Fel under analys: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
