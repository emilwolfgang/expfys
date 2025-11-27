"""QTM loader utilities shared by Del A and Del B analyses."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_qmt_any(path: str) -> pd.DataFrame:
    """Load a QTM TSV file, supporting both FILE_VERSION exports and plain TSV."""
    with open(path, "r", errors="ignore") as f:
        first_line = f.readline()
        if first_line.startswith("FILE_VERSION"):
            f.seek(0)
            lines = f.readlines()
        else:
            lines = None

    # FILE_VERSION style export
    if first_line.startswith("FILE_VERSION"):
        data_start = None
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            try:
                [float(p) for p in parts]
            except ValueError:
                continue
            data_start = i
            break
        if data_start is None:
            raise ValueError("Could not locate numeric data section in QTM file.")

        df_num = pd.read_csv(path, sep="\t", header=None, skiprows=data_start)
        n_cols = df_num.shape[1]
        if n_cols % 3 != 0:
            return df_num

        n_markers = n_cols // 3

        marker_names: list[str] | None = None
        for line in lines[:data_start]:
            if line.startswith("MARKER_NAMES"):
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    marker_names = [p.strip() for p in parts[1:] if p.strip()]
                break

        col_names: list[str] = []
        if marker_names is not None and len(marker_names) == n_markers:
            for name in marker_names:
                col_names.extend([f"{name} : X", f"{name} : Y", f"{name} : Z"])
        else:
            for mid in range(1, n_markers + 1):
                col_names.extend([f"x{mid}", f"y{mid}", f"z{mid}"])

        df_num.columns = col_names

        freq = None
        for line in lines[:data_start]:
            if line.startswith("FREQUENCY"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        freq = float(parts[1])
                    except ValueError:
                        freq = None
                break

        if freq is not None and freq > 0:
            n = len(df_num)
            time = np.arange(n, dtype=np.float64) / freq
            df_num.insert(0, "time", time)

        return df_num

    df_tab = pd.read_csv(path, sep="\t", engine="python")
    if len(df_tab.columns) == 1:
        df_ws = pd.read_csv(path, delim_whitespace=True, engine="python")
        return df_ws
    return df_tab


def find_markers_xyz_flexible(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Return mapping {base: {X,Y,Z}} for all markers with complete coordinates."""
    markers: Dict[str, Dict[str, str]] = {}

    rx_named = re.compile(
        r"^(?P<base>.+?)[\s:_\-.]+(?P<comp>[XYZ])(?:\b|[\s\[\(].*)?$",
        re.IGNORECASE,
    )
    rx_indexed = re.compile(r"^(?P<comp>[XYZxyz])(?P<id>\d+)$")

    for col in df.columns:
        col_str = str(col).strip()

        m = rx_named.match(col_str)
        if m:
            base = m.group("base").strip()
            comp = m.group("comp").upper()
            if base.lower() in {"time", "frame"}:
                continue
            markers.setdefault(base, {})[comp] = col
            continue

        m2 = rx_indexed.match(col_str)
        if m2:
            comp = m2.group("comp").upper()
            idx = m2.group("id")
            base = f"marker_{idx}"
            markers.setdefault(base, {})[comp] = col
            continue

    return {
        b: comps
        for b, comps in markers.items()
        if {"X", "Y", "Z"} <= set(comps.keys())
    }


def detect_time_series(df: pd.DataFrame) -> pd.Series:
    """Return a time-like series if present, else the row index."""
    for col in df.columns:
        if str(col).strip().lower() == "time":
            return df[col]
    for col in df.columns:
        if str(col).strip().lower().startswith("time"):
            return df[col]
    return pd.Series(range(len(df)), index=df.index, name="index_time")


def _read_file_lines(path: Path) -> List[str]:
    with open(path, "r", errors="ignore") as f:
        return f.read().splitlines()


def load_qtm_6d_file_version(path: Path):
    """Load a FILE_VERSION export with DATA_INCLUDED=3D (Del B use-case)."""
    lines = _read_file_lines(path)
    if not lines or not lines[0].startswith("FILE_VERSION"):
        raise ValueError("Filen ser inte ut som en QTM FILE_VERSION-export.")

    freq: float | None = None
    no_of_markers: int | None = None
    marker_names: List[str] = []

    for line in lines:
        if line.startswith("FREQUENCY"):
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    freq = float(parts[1])
                except ValueError:
                    freq = None
        elif line.startswith("NO_OF_MARKERS"):
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    no_of_markers = int(parts[1])
                except ValueError:
                    no_of_markers = None
        elif line.startswith("MARKER_NAMES"):
            parts = line.split("\t")
            marker_names = [p.strip() for p in parts[1:] if p.strip()]

    if not freq or freq <= 0:
        raise ValueError("FREQUENCY saknas eller är ogiltig i headern.")
    if no_of_markers is None:
        raise ValueError("NO_OF_MARKERS saknas eller är ogiltig i headern.")
    if not marker_names:
        raise ValueError("MARKER_NAMES saknas i headern.")
    if len(marker_names) != no_of_markers:
        raise ValueError(
            f"NO_OF_MARKERS={no_of_markers} men MARKER_NAMES har {len(marker_names)} namn."
        )

    data_start: int | None = None
    for i, line in enumerate(lines):
        parts = line.split("\t")
        if not parts:
            continue
        try:
            float(parts[0])
        except ValueError:
            continue
        else:
            data_start = i
            break
    if data_start is None:
        raise ValueError("Kunde inte hitta numerisk datasektion i QTM-filen.")

    df_num = pd.read_csv(path, sep="\t", header=None, skiprows=data_start)
    n_frames, n_cols = df_num.shape
    expected_cols = 3 * no_of_markers
    if n_cols != expected_cols:
        raise ValueError(
            f"Förväntade {expected_cols} kolumner ({no_of_markers} markörer * 3), "
            f"men fick {n_cols}."
        )

    time = np.arange(n_frames, dtype=np.float64) / freq

    marker_pos: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(marker_names):
        c0 = 3 * idx
        c1 = c0 + 1
        c2 = c0 + 2
        pos = df_num.iloc[:, [c0, c1, c2]].to_numpy(dtype=float)
        marker_pos[name] = pos

    header_info = {
        "FREQUENCY": freq,
        "NO_OF_FRAMES": n_frames,
        "NO_OF_MARKERS": no_of_markers,
        "MARKER_NAMES": marker_names,
    }
    return time, marker_pos, header_info
