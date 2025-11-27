"""Material metadata shared between part A and part B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from . import REPO_ROOT

MATERIAL_COLOR_MAP: Dict[str, str] = {
    "Aluminium": "tab:blue",
    "Plast": "tab:red",
    "Kardborreband": "tab:orange",
    "Skumgummi": "tab:green",
}


@dataclass(frozen=True)
class MaterialSeries:
    """Immutable metadata bundle for one summary series."""

    path: Path | str
    label: str
    color_name: str | None = None
    key: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))

    @property
    def color(self) -> str | None:
        """Return the matplotlib color to use for the series."""
        if self.color_name:
            return self.color_name
        return MATERIAL_COLOR_MAP.get(self.label)

    @property
    def series_key(self) -> str:
        """Return a stable identifier for grouping."""
        if self.key:
            return self.key
        return self.path.stem

    @property
    def absolute_path(self) -> Path:
        """Absolute path to the summary file."""
        return (REPO_ROOT / self.path).resolve()


# Default summary locations -------------------------------------------------

SUMMARY_REGISTRY: Dict[str, List[MaterialSeries]] = {
    "A": [
        MaterialSeries("data/del_a/summaries/summary_A1.tsv", "Aluminium", key="summary_A1"),
        MaterialSeries("data/del_a/summaries/summary_A2.tsv", "Kardborreband", key="summary_A2"),
        MaterialSeries("data/del_a/summaries/summary_A3.tsv", "Skumgummi", key="summary_A3"),
    ],
    "B": [
        MaterialSeries("data/del_b/summaries/summary_B1.tsv", "Plast", key="summary_B1"),
        MaterialSeries("data/del_b/summaries/summary_B2.tsv", "Kardborreband", key="summary_B2"),
        MaterialSeries("data/del_b/summaries/summary_B3.tsv", "Skumgummi", key="summary_B3"),
    ],
}

PART_A_SERIES = SUMMARY_REGISTRY["A"]
PART_B_SERIES = SUMMARY_REGISTRY["B"]


def get_series(part: str) -> List[MaterialSeries]:
    """Return configured series for the requested part ('A' or 'B')."""
    key = part.upper()
    if key not in SUMMARY_REGISTRY:
        raise KeyError(f"Okänt del-ID: {part!r}")
    return SUMMARY_REGISTRY[key]


def iter_all_series() -> Iterable[MaterialSeries]:
    """Yield every configured series."""
    for series_list in SUMMARY_REGISTRY.values():
        yield from series_list

__all__ = [
    "MATERIAL_COLOR_MAP",
    "MaterialSeries",
    "PART_A_SERIES",
    "PART_B_SERIES",
    "SUMMARY_REGISTRY",
    "get_series",
    "iter_all_series",
]
