#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

# Extra datafiler och summary-filer som kan läggas till manuellt vid behov
# Fyll i dessa listor om du vill lägga till en annan mätserie utan att ändra körkommandot.
EXTRA_FILES: list[str] = [
    # exempel: "data/del_b/raw/B1(1).tsv",
    "data/del_a/raw/A1(1).tsv",
    "data/del_a/raw/A1(2).tsv",
    "data/del_a/raw/A1(3).tsv",
    "data/del_a/raw/A1(4).tsv",
    "data/del_a/raw/A1(5).tsv",
    "data/del_a/raw/A1(6).tsv",
    "data/del_a/raw/A1(7).tsv",
    "data/del_a/raw/A1(8).tsv",
    "data/del_a/raw/A1(9).tsv",
    "data/del_a/raw/A1(10).tsv",
]

EXTRA_SUMMARIES: list[str] = [
    # exempel: "data/del_b/summaries/summary_B.tsv",
    "data/del_a/summaries/A1_None",
    "data/del_a/summaries/A2_None",
    "data/del_a/summaries/A3_None",
]

# Ensure the repo root is importable for the analysis package
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

from expfys.analysis.del_a import load_qmt_any, find_markers_xyz_flexible
from expfys.plotting.style import FIG_WIDTH_IN, set_fixed_figwidth


def detect_time_series(df: pd.DataFrame) -> pd.Series:
    """Hittar tidskolumnen, annars index som fallback."""
    for col in df.columns:
        if str(col).lower() == "time":
            return df[col]
        if str(col).lower().startswith("time"):
            return df[col]
    return pd.Series(range(len(df)), index=df.index, name="index_time")


def find_collision_time(df: pd.DataFrame) -> float:
    """
    Exempel: definiera kollisionstid t_P som när |dp| är max,
    eller byt ut mot den metod DU använder.
    """
    if "dp_x" in df.columns and "dp_y" in df.columns and "dp_z" in df.columns:
        mag = np.sqrt(df["dp_x"]**2 + df["dp_y"]**2 + df["dp_z"]**2)
        idx = mag.idxmax()
        t = detect_time_series(df)
        return float(t.loc[idx])

    # Fallback – ingen kollisionstid hittad
    return 0.0


def plot_z_centered(files: list[str], marker="New 0000", energy_map: dict[str, float] | None = None):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3))
    set_fixed_figwidth(fig)
    any_plotted = False

    cmap = plt.get_cmap("RdYlGn")  # green (high) to red (low)
    color_clip_low = 0.4
    color_green_start = 0.5
    vmin = vmax = None
    if energy_map:
        vals = list(energy_map.values())
        vmin = min(vals)
        vmax = max(vals)
        if vmax == vmin:
            # avoid divide-by-zero; spread artificially
            vmin = vmax - 1.0

    for f in files:
        source = f
        df = load_qmt_any(f)
        df.attrs["source"] = source
        markers = find_markers_xyz_flexible(df)

        if marker not in markers:
            print(f"Marker {marker} saknas i {f}")
            continue

        # tid och z-komponent
        t = detect_time_series(df).astype(float)
        comps = markers[marker]
        z = df[comps["Z"]].astype(float).interpolate()

        # hitta t_P (kollision)
        tP = find_collision_time(df)

        # bestäm nollnivå som medelvärde av upp till 100 punkter före kollisionen
        # använd t < tP som "före"; om färre än 100 punkter finns, använd alla
        before_mask = t < tP
        z_before = z[before_mask]
        if len(z_before) >= 100:
            z0 = z_before.tail(100).mean()
        elif len(z_before) > 0:
            z0 = z_before.mean()
        else:
            # om ingen punkt ligger före tP (t ex om tP ligger vid första samplet) – använd första 100 totalt
            z0 = z.iloc[:100].mean()

        z = z - z0


        # centrera tiden
        t_shift = t - tP

        # välj färg baserat på andel bevarad energi (om finns)
        color = None
        stem = Path(f).stem
        if energy_map is not None and stem in energy_map and vmin is not None and vmax is not None:
            val = energy_map[stem]

            # --- Justerad färgskala ---
            # Klipp alla värden under 0.4 → samma mörkröda nyans
            val_adj = max(val, color_clip_low)

            # Övre gräns – tryck ihop toppområdet så att grönt börjar runt 0.5
            val_adj = min(val_adj, color_green_start)

            # Normalisera inom det effektiva intervallet
            norm = (val_adj - color_clip_low) / (color_green_start - color_clip_low)
            norm = max(0.0, min(1.0, norm))

            color = cmap(norm)

        # plot
        if color is not None:
            ax.plot(t_shift, z, label=stem, alpha=0.8, color=color)
        else:
            ax.plot(t_shift, z, label=stem, alpha=0.8)
        any_plotted = True

    # om inget plottades alls, avbryt innan vi försöker skapa colorbar/axlar
    if not any_plotted:
        print("Inget kunde plottas (marker saknas i alla filer eller tom filista).")
        return

    # färgskalans färgbar om vi har energidata
    if energy_map and vmin is not None and vmax is not None:
        sm = cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=color_clip_low, vmax=color_green_start),
            cmap=cmap,
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$Q_E$", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Tid relativt kollision [s]", fontsize=14)
    ax.set_ylabel("Vertikal position\nmarkör 1 [mm]", fontsize=14, labelpad=20)
    ax.set_title("Vertikal position kring kollisionstidpunkten för kontaktytan aluminium, del A", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Kör utan argument – använd endast hårdkodade filer
    summary_map: dict[str, float] = {}
    energy_map: dict[str, float] = {}

    def add_summary(path_str: str) -> None:
        path = Path(path_str)
        if not path.is_file():
            print(f"Varning: summary-fil hittades inte: {path}")
            return
        df_sum = pd.read_csv(path, sep="\t")
        for _, row in df_sum.iterrows():
            stem = str(Path(row["file"]).stem)
            summary_map[stem] = row.get("t_collision", 0.0)
            # rel_dKE antas vara (E_efter - E_före) / E_före
            if "rel_dKE" in row and not pd.isna(row["rel_dKE"]):
                rel_dKE = float(row["rel_dKE"])
                preserved = 1.0 + rel_dKE  # E_efter / E_före
                energy_map[stem] = preserved

    # läs alla hårdkodade summary-filer
    for s in EXTRA_SUMMARIES:
        add_summary(s)

    # definiera find_collision_time baserat på summary_map

    def find_collision_time(df):
        stem = Path(df.attrs.get("source", "")).stem
        return summary_map.get(stem, 0.0)

    # bygg fil-lista enbart från EXTRA_FILES
    all_files = EXTRA_FILES

    plot_z_centered(all_files, energy_map=energy_map)
