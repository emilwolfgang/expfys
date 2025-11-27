"""Helpers for loading TSV summary files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import re

import pandas as pd

from .materials import MaterialSeries


@dataclass(frozen=True)
class SummaryTable:
    """Container that bundles a loaded pandas DataFrame with metadata."""

    series: MaterialSeries
    frame: pd.DataFrame

    @property
    def label(self) -> str:
        return self.series.label

    @property
    def color(self) -> str | None:
        return self.series.color

    @property
    def path(self) -> Path:
        return self.series.absolute_path


def load_table(series: MaterialSeries) -> SummaryTable:
    """Read the TSV file described by ``series``."""
    frame = pd.read_csv(series.absolute_path, sep="\t")
    return SummaryTable(series=series, frame=frame)


def load_tables(series_list: Sequence[MaterialSeries]) -> List[SummaryTable]:
    """Load multiple series into memory."""
    return [load_table(series) for series in series_list]


def infer_attempt_numbers(frame: pd.DataFrame) -> pd.Series:
    """Return attempt indices for scatter plots.

    - Prefer ``file`` column and extract the integer inside parentheses.
    - Fall back to explicit ``attempt`` column if present.
    - If nothing is available use 1..N.
    """
    if "file" in frame.columns:
        rx = re.compile(r"\((\d+)\)")
        attempts = []
        for value in frame["file"].fillna(""):
            match = rx.search(str(value))
            attempts.append(int(match.group(1)) if match else None)

        fallback = 1
        for idx, val in enumerate(attempts):
            if val is None:
                attempts[idx] = fallback
            fallback += 1
        return pd.Series(attempts, name="attempt")

    if "attempt" in frame.columns:
        return frame["attempt"]

    return pd.Series(range(1, len(frame) + 1), name="attempt")


__all__ = [
    "SummaryTable",
    "load_table",
    "load_tables",
    "infer_attempt_numbers",
]
