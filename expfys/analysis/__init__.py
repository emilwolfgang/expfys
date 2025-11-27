"""High-level analysis entry points for Del A and Del B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import del_a, del_b_free, del_b_sticky

__all__ = [
    "del_a",
    "del_b_free",
    "del_b_sticky",
    "analyze_del_a_file",
    "analyze_del_b_free_file",
    "analyze_del_b_sticky_file",
]


def analyze_del_a_file(path: str | Path, cfg: del_a.Config | None = None) -> dict[str, Any]:
    """Convenience wrapper that runs the Del A pipeline on ``path``."""
    if cfg is None:
        cfg = del_a.Config()
    return del_a.analyze_file(Path(path), cfg)


def analyze_del_b_free_file(path: str | Path, cfg: del_b_free.Config | None = None) -> dict[str, Any]:
    """Run the non-sticking Del B analyzer."""
    if cfg is None:
        raise ValueError(
            "Del B free analyzer requires an explicit Config with bodies/dist threshold."
        )
    return del_b_free.analyze_air_table(Path(path), cfg)


def analyze_del_b_sticky_file(path: str | Path, cfg: del_b_sticky.Config | None = None) -> dict[str, Any]:
    """Run the sticking Del B analyzer."""
    if cfg is None:
        raise ValueError(
            "Del B sticky analyzer requires an explicit Config with bodies/dist threshold."
        )
    return del_b_sticky.analyze_air_table(Path(path), cfg)
