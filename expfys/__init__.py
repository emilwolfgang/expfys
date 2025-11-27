"""Shared utilities for the ExpFys analysis scripts.

This package centralises all reusable helpers so the plotting, batch and report
scripts can rely on the same behaviour. Comments stay in English while any user
facing text (prints, plot labels, legends) remains in Swedish to respect the
lab report.
"""

from __future__ import annotations

from pathlib import Path

# Repository layout ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAPPORT_DIR = REPO_ROOT / "rapport"
DEL_A_DATA_DIR = DATA_DIR / "del_a"
DEL_B_DATA_DIR = DATA_DIR / "del_b"

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "RAPPORT_DIR",
    "DEL_A_DATA_DIR",
    "DEL_B_DATA_DIR",
]
