"""Common helpers for reading and writing experiment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def write_summary(rows: Sequence[dict] | pd.DataFrame, path: str | Path) -> None:
    """Persist a list of dictionaries or DataFrame as a TSV summary."""
    if isinstance(rows, pd.DataFrame):
        frame = rows
    else:
        frame = pd.DataFrame(list(rows))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def read_summary(path: str | Path) -> pd.DataFrame:
    """Load a TSV summary into a pandas DataFrame."""
    return pd.read_csv(path, sep="\t")


__all__ = ["write_summary", "read_summary"]
