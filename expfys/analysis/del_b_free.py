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
    if len(pos) != 2:
        raise ValueError("find_collision_interval: förväntar exakt två kroppar.")

    bodies = list(pos.keys())
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
    post = stats_at_time(t_post, v_post, omega_post)
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
    bodies = [b1, b2]
    colors = {b1: "C0", b2: "C1"}

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


def plot_u_v_e(
    time: np.ndarray,
    pos: Dict[str, np.ndarray],
    origin: np.ndarray,
    n_hat: np.ndarray,
    bodies_order: Tuple[str, str],
    t_p_minus: float | None = None,
    t_p_plus: float | None = None,
    t_p_star: float | None = None,
    title: str = "",
    t_window_minus: float | None = None,
    t_window_plus: float | None = None,
):
    """Plotta "u, v, e" som tidsserier för båda puckarna.

    Här tolkas:
    - u ≈ x-komponenten relativt origin,
    - v ≈ y-komponenten relativt origin,
    - e = projektionen på n_hat (med n_hat = (0,0,1) i vårt fall).
    """
    b1, b2 = bodies_order
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    bodies = [b1, b2]
    colors = {b1: "C0", b2: "C1"}

    # För varje kropp, beräkna komponenter relativt origin
    rel_coords: Dict[str, Dict[str, np.ndarray]] = {}
    for body in bodies:
        r = pos[body] - origin  # (N,3)
        u = r[:, 0]
        v = r[:, 1]
        e = r @ n_hat  # för n_hat = (0,0,1) blir detta r[:,2]
        rel_coords[body] = {"u": u, "v": v, "e": e}

    comp_keys = ["u", "v", "e"]
    comp_labels = ["u (≈ x)", "v (≈ y)", "e (≈ z)"]
    # Används för att bara sätta legend-label för t_- / t_+ en gång
    window_labels_done = False

    for ax, comp, label in zip(axes, comp_keys, comp_labels):
        for body in bodies:
            c = colors[body]
            y = rel_coords[body][comp]
            # Sätt label bara i första subploten för snygg legend
            line_label = body if comp == "u" else None
            ax.plot(time, y, label=line_label, color=c, alpha=0.9)

        # Vertikala linjer
        if t_p_minus is not None:
            ax.axvline(t_p_minus, linestyle="--", color="k", alpha=0.7)
        if t_p_plus is not None:
            ax.axvline(t_p_plus, linestyle="--", color="grey", alpha=0.7)
        if t_p_star is not None:
            ax.axvline(t_p_star, linestyle="-.", color="red", alpha=0.8)
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

        ax.set_ylabel(label)
        ax.grid(True)

    axes[-1].set_xlabel("Tid (s)")
    if title:
        fig.suptitle(title)
    # Lägg till legend bara i första subploten om den finns
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
    bodies = [b1, b2]
    colors = {b1: "C0", b2: "C1"}
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
# Plot: u-position och v_u (v_x) i två subplots för båda puckarna
# =============================================================================

def plot_u_and_vu(
    time: np.ndarray,
    pos: Dict[str, np.ndarray],
    velocities: Dict[str, np.ndarray],
    origin: np.ndarray,
    bodies_order: Tuple[str, str],
    pos_fits: Dict[str, Dict[int, dict]] | None = None,
    v_pre: Dict[str, np.ndarray] | None = None,
    v_post: Dict[str, np.ndarray] | None = None,
    t_p_minus: float | None = None,
    t_p_plus: float | None = None,
    t_p_star: float | None = None,
    title: str = "",
    t_window_minus: float | None = None,
    t_window_plus: float | None = None,
):
    """Plotta u-position (≈ x) och v_u (v_x) i två subplots för båda puckarna.

    Vertikala linjer markerar t_P-/t_P+, t_min (dist) samt fönsterkanter t_- och t_+.
    """
    # Enlarge text sizes for this specific plot
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })
    b1, b2 = bodies_order
    bodies = [b1, b2]
    colors = {b1: "C0", b2: "C1"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Beräkna u-komponenten relativt origin och v_u = v_x
    u_coords: Dict[str, np.ndarray] = {}
    v_u_coords: Dict[str, np.ndarray] = {}
    for body in bodies:
        r = pos[body] - origin  # (N,3)
        u_coords[body] = r[:, 0]
        v_u_coords[body] = velocities[body][:, 0]  # v_x-komponenten

    # Subplot 0: u(t)
    ax_u = axes[0]
    for body in bodies:
        c = colors[body]
        label = r"Puck 1" if body == bodies[0] else r"Puck 2"
        ax_u.plot(time, u_coords[body], label=label, color=c, alpha=0.9)

        # Rita fittade polynom för u-komponenten (x) om vi har pos_fits
        if pos_fits is not None:
            body_fit = pos_fits.get(body, {})
            fit_u = body_fit.get(0)  # komponent 0 = x/u
            if fit_u is not None:
                pre_t = fit_u["pre_t"]
                post_t = fit_u["post_t"]
                pre_fit = fit_u["pre_fit"]
                post_fit = fit_u["post_fit"]

                # Justera till samma referens som u_coords (relativt origin)
                pre_fit_rel = pre_fit - origin[0]
                post_fit_rel = post_fit - origin[0]

                ax_u.plot(
                    pre_t,
                    pre_fit_rel,
                    linestyle=":",
                    color=c,
                    alpha=0.9,
                    label=r"$q_{1 innan}$" if body == bodies[0] else r"$q_{2 innan}$",
                    linewidth=2.5,
                )
                ax_u.plot(
                    post_t,
                    post_fit_rel,
                    linestyle="--",
                    color=c,
                    alpha=0.9,
                    label=r"$q_{1 efter}$" if body == bodies[0] else r"$q_{2 efter}$",
                    linewidth=2.5,
                )
    # Vertikala linjer (endast label i första subploten)
    if t_p_minus is not None:
        ax_u.axvline(
            t_p_minus,
            linestyle="--",
            color="black",
            alpha=0.7,
            label=r"$t_{P-}$",
        )
    if t_p_plus is not None:
        ax_u.axvline(
            t_p_plus,
            linestyle="--",
            color="black",
            alpha=0.7,
            label=r"$t_{P+}$",
        )
    if t_window_minus is not None:
        ax_u.axvline(
            t_window_minus,
            linestyle="-",
            color="orange",
            alpha=0.5,
            label=r"$t_{-}$",
        )
    if t_window_plus is not None:
        ax_u.axvline(
            t_window_plus,
            linestyle="-",
            color="orange",
            alpha=0.5,
            label=r"$t_{+}$",
        )
    ax_u.set_ylabel(r"Position $ \hat{u}\;[\mathrm{mm}]$")
    ax_u.grid(True)
    ax_u.legend(loc="best")

    # Subplot 1: v_u(t) = v_x(t)
    ax_v = axes[1]
    for body in bodies:
        c = colors[body]
        ax_v.plot(
            time,
            v_u_coords[body],
            label=r"$v_{1}$" if body == bodies[0] else r"$v_{2}$",
            color=c,
            alpha=0.9,
        )

        # Markera v_pre och v_post för x-komponenten om de finns
        if v_pre is not None and v_post is not None:
            if t_p_minus is not None:
                v_pre_body = v_pre.get(body)
                if v_pre_body is not None:
                    y_pre_point = v_pre_body[0]  # x-komponenten
                    label_pre = (
                        r"$v_{1}$"
                        if body == bodies[0]
                        else r"$v_{2}$"
                    )
                    ax_v.scatter(
                        t_p_minus,
                        y_pre_point,
                        color=c,
                        edgecolors="k",
                        zorder=5,
                        label=label_pre,
                    )
            if t_p_plus is not None:
                v_post_body = v_post.get(body)
                if v_post_body is not None:
                    y_post_point = v_post_body[0]
                    label_post = (
                        r"$v_{1}'$"
                        if body == bodies[0]
                        else r"$v_{2}'$"
                    )
                    ax_v.scatter(
                        t_p_plus,
                        y_post_point,
                        color=c,
                        edgecolors="k",
                        marker="s",
                        zorder=5,
                        label=label_post,
                    )
    # Vertikala linjer utan extra labels (dessa finns redan i första subploten)
    if t_p_minus is not None:
        ax_v.axvline(t_p_minus, linestyle="--", color="k", alpha=0.7)
    if t_p_plus is not None:
        ax_v.axvline(t_p_plus, linestyle="--", color="grey", alpha=0.7)
    if t_window_minus is not None:
        ax_v.axvline(
            t_window_minus,
            linestyle="-",
            color="orange",
            alpha=0.5,
        )
    if t_window_plus is not None:
        ax_v.axvline(
            t_window_plus,
            linestyle="-",
            color="orange",
            alpha=0.5,
        )
    ax_v.set_ylabel(r"$v_{\hat{u}}\;[\mathrm{mm/s}]$")
    ax_v.set_xlabel("Tid (s)")
    ax_v.grid(True)
    ax_v.legend(loc="best")

    if title:
        fig.suptitle(title)
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
        t_plus = collision["t_p_plus"]     # fönsterkant t_+
        start_idx = collision["start_idx"]
        end_idx = collision["end_idx"]
        t_min_dist = collision["t_min"]

        print(
            f"\nKollisionsfönster hittat när dist < {cfg.dist_threshold:.4f}:"
            f"\n  t_-  = {t_minus:.6f} s (första under tröskel)"
            f"\n  t_+  = {t_plus:.6f} s (sista under tröskel)"
            f"\n  min dist = {collision['d_min']:.6f} vid t = {t_min_dist:.6f} s "
            f"(index {collision['idx_min']})"
        )

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
        region_start = max(0, start_idx - 5)
        region_end = min(N - 1, end_idx + 5)
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
                "använder fönsterkanterna som t_P- och t_P+."
            )
            t_P_minus = t_minus
            t_P_plus = t_plus
        else:
            t_P_minus = float(time[idx_imp[0]])
            t_P_plus = float(time[idx_imp[-1]])

        print(
            f"\nImpacttider (baserat på acceleration):"
            f"\n  t_P- = {t_P_minus:.6f} s"
            f"\n  t_P+ = {t_P_plus:.6f} s"
        )

        # 7) Orienteringsvinklar och ω från markörer (COM + kant) i x–y-planet
        orient = compute_orientation_from_markers(
            time,
            pos,
            edge_pos,
        )

        # 8) Linjära hastigheter: polynomanpassning i global X, Y, Z
        v_pre: Dict[str, np.ndarray] = {}
        v_post: Dict[str, np.ndarray] = {}
        velocity_fit_segments: Dict[str, Dict[int, dict]] = {body: {} for body in bodies_order}

        for body in bodies_order:
            r = pos[body]  # (N,3)
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
                # Nya definitionen: evaluera derivatan vid t_P- resp t_P+
                dp_pre = res_fit["dp_pre"]
                dp_post = res_fit["dp_post"]
                v_pre_body[comp] = float(np.polyval(dp_pre, t_P_minus))
                v_post_body[comp] = float(np.polyval(dp_post, t_P_plus))

                velocity_fit_segments[body][comp] = res_fit

            v_pre[body] = v_pre_body
            v_post[body] = v_post_body

        velocity_fits = velocity_fit_segments

        # 9) Vinkelhastigheter: polynomanpassning på θ(t) i planet
        omega_pre: Dict[str, float] = {}
        omega_post: Dict[str, float] = {}
        theta_fits: Dict[str, dict] = {}
        for body in bodies_order:
            theta = orient[body]["theta"]
            res_fit_theta = polyfit_pre_post(
                time,
                theta,
                start_idx=start_idx,
                end_idx=end_idx,
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

        # Medel vinkelhastighet (signed mean) för de två puckarna vid t_P- och t_P+
        omega_pre_mean = 0.5 * (omega_pre[bodies_order[0]] + omega_pre[bodies_order[1]])
        omega_post_mean = 0.5 * (omega_post[bodies_order[0]] + omega_post[bodies_order[1]])

        # Individuella vinkelhastigheter (förtydligade variabelnamn)
        omega1_pre = omega_pre[bodies_order[0]]
        omega2_pre = omega_pre[bodies_order[1]]
        omega1_post = omega_post[bodies_order[0]]
        omega2_post = omega_post[bodies_order[1]]

        # Project velocities onto u (x) and v (y) directions
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
        # Om ingen kollision: beräkna orient för plottar ändå
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

        # Plotta "u, v, e" (här u ≈ x, v ≈ y, e ≈ z relativt origin) som tidsserier
        plot_u_v_e(
            time,
            pos,
            origin,
            n_hat,
            bodies_order=bodies_order,
            t_p_minus=t_plot_minus,
            t_p_plus=t_plot_plus,
            t_p_star=t_min,
            title=f"{file_path.name} – u/v/e (x, y, z) vs tid",
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

        # Extra figur: u-position och v_u (v_x) i två subplots
        plot_u_and_vu(
            time,
            pos,
            velocities,
            origin,
            bodies_order=bodies_order,
            pos_fits=velocity_fits,
            v_pre=v_pre,
            v_post=v_post,
            t_p_minus=t_plot_minus,
            t_p_plus=t_plot_plus,
            t_p_star=t_min,
            title=r"Rörelse i $\hat{u}$-riktning, position och hastighet, för mätning 1 i del B kontaktyta: plast",
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
