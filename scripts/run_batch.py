#!/usr/bin/env python3
"""Batch runner for both Del A and Del B prefixes.

Examples
--------
Process A1(1..10):
    python scripts/run_batch.py --prefix A1 --start 1 --end 10 --dist-threshold 30

Process sticky B2 series:
    python scripts/run_batch.py --prefix B2 --sticky --dist-threshold 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DEL_A_RAW = DATA_DIR / "del_a" / "raw"
DEL_A_SUMMARIES = DATA_DIR / "del_a" / "summaries"
DEL_B_RAW = DATA_DIR / "del_b" / "raw"
DEL_B_SUMMARIES = DATA_DIR / "del_b" / "summaries"
DEL_B_BODIES = DATA_DIR / "del_b" / "bodies"

from expfys.analysis import del_b_free, del_b_sticky
from expfys.analysis import del_a_batch
from expfys.analysis.summary import write_summary

DEFAULT_BODIES = {
    "B1": "bodiesB1.tsv",
    "B2": "bodiesB2.tsv",
    "B3": "bodiesB3.tsv",
}

DEFAULT_DIST_THRESHOLDS = {
    "A": 30.0,
    "B1": 63.5,
    "B2": 60.0,
    "B3": 62.0,
}

try:
    DEL_A_WEIGHTS = pd.read_csv(DATA_DIR / "del_a" / "aux" / "weights.tsv", sep="\t")
except Exception:
    DEL_A_WEIGHTS = None


def get_del_a_threshold(prefix: str) -> float | None:
    if DEL_A_WEIGHTS is None or "dist_threshold" not in DEL_A_WEIGHTS.columns:
        return None
    subset = DEL_A_WEIGHTS[DEL_A_WEIGHTS["prefix"].str.upper() == prefix.upper()]
    if subset.empty:
        return None
    vals = subset["dist_threshold"].dropna()
    if vals.empty:
        return None
    try:
        return float(vals.iloc[0])
    except Exception:
        return None


def get_dist_threshold_from_bodies(path: Path) -> float | None:
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


def run_del_a(prefix: str, args: argparse.Namespace) -> None:
    dist_threshold = args.dist_threshold
    if dist_threshold is None:
        dist_threshold = get_del_a_threshold(prefix)
    if dist_threshold is None:
        dist_threshold = DEFAULT_DIST_THRESHOLDS["A"]
    df = del_a_batch.run_batch(
        base_dir=DEL_A_RAW,
        prefix=prefix,
        start_idx=args.start,
        end_idx=args.end,
        ext=".tsv",
        marker1=args.marker1,
        marker2=args.marker2,
        m1=args.m1,
        m2=args.m2,
        dist_threshold=dist_threshold,
    )
    if df.empty:
        raise SystemExit("Inga giltiga försök analyserades för Del A.")

    summary_path = resolve_summary_path(
        args.output, DEL_A_SUMMARIES / f"summary_{prefix}.tsv"
    )
    write_summary(df, summary_path)
    print(f"Summary sparad till: {summary_path}")


def run_del_b(prefix: str, args: argparse.Namespace) -> None:
    prefix_key = prefix[:2].upper()
    bodies_name = args.bodies or DEFAULT_BODIES.get(prefix_key)
    if bodies_name is None:
        raise SystemExit(f"Saknar standardkroppar för prefix {prefix_key}; ange --bodies.")

    bodies_path = Path(bodies_name)
    if not bodies_path.is_absolute():
        bodies_path = DEL_B_BODIES / bodies_path

    if not bodies_path.exists():
        raise SystemExit(f"Bodies-fil saknas: {bodies_path}")

    dist_threshold = args.dist_threshold
    if dist_threshold is None:
        dist_threshold = get_dist_threshold_from_bodies(bodies_path)
    if dist_threshold is None:
        dist_threshold = DEFAULT_DIST_THRESHOLDS.get(prefix_key, 60.0)
    config_cls = del_b_sticky.Config if args.sticky else del_b_free.Config
    analyzer = del_b_sticky if args.sticky else del_b_free

    cfg = config_cls(
        bodies_path=bodies_path,
        dist_threshold=dist_threshold,
        max_gap_frames=args.max_gap_frames,
        make_plots=False,
        emit_summary=False,
    )

    rows: List[dict] = []
    for idx in range(args.start, args.end + 1):
        file_path = DEL_B_RAW / f"{prefix}({idx}).tsv"
        if not file_path.exists():
            print(f"Hoppar över {file_path.name} – filen finns inte.")
            continue

        print(f"=== Kör {file_path.name} ===")
        try:
            result = analyzer.analyze_air_table(file_path, cfg)
        except Exception as exc:
            print(f"Fel under analys av {file_path.name}: {exc}")
            continue

        summary = result.get("summary") if isinstance(result, dict) else None
        if summary:
            rows.append(summary)
        else:
            print(f"  -> Ingen summary genererades för {file_path.name}")

    if not rows:
        raise SystemExit("Inga sammanfattningar skapades – kontrollera prefix och intervall.")

    summary_path = resolve_summary_path(
        args.output, DEL_B_SUMMARIES / f"summary_{prefix}.tsv"
    )
    write_summary(rows, summary_path)
    print(f"Summary sparad till: {summary_path}")


def resolve_summary_path(output_arg: str | None, default_path: Path) -> Path:
    if output_arg:
        out_path = Path(output_arg)
        if not out_path.is_absolute():
            out_path = default_path.parent / out_path
    else:
        out_path = default_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Kör batchanalys för ett prefix (A* eller B*)."
    )
    parser.add_argument("--prefix", required=True, help="Prefix, t.ex. A1 eller B2.")
    parser.add_argument("--start", type=int, default=1, help="Första index (default 1).")
    parser.add_argument("--end", type=int, default=10, help="Sista index (default 10).")
    parser.add_argument("--dist-threshold", type=float, help="Överskriv standarddistans.")
    parser.add_argument("--output", help="Valfritt filnamn för summary (skrivs i summary-mappen).")

    # Del A specifika argument
    parser.add_argument("--marker1", help="Del A: namn på första markören.")
    parser.add_argument("--marker2", help="Del A: namn på andra markören.")
    parser.add_argument("--m1", type=float, default=0.2, help="Del A: massa för kropp 1 [kg].")
    parser.add_argument("--m2", type=float, default=0.2, help="Del A: massa för kropp 2 [kg].")

    # Del B specifika argument
    parser.add_argument("--sticky", action="store_true", help="Använd sticky-analys för Del B.")
    parser.add_argument(
        "--bodies",
        help="Del B: stig till bodies.tsv (default väljs efter prefix om ej angiven).",
    )
    parser.add_argument(
        "--max-gap-frames", type=int, default=3, help="Del B: max gap mellan dist<d segment."
    )

    args = parser.parse_args(argv)
    prefix = args.prefix.strip()
    if not prefix:
        raise SystemExit("Prefix måste anges.")

    prefix_upper = prefix.upper()
    if prefix_upper.startswith("A"):
        run_del_a(prefix_upper, args)
    elif prefix_upper.startswith("B"):
        run_del_b(prefix_upper, args)
    else:
        raise SystemExit("Prefix måste börja med 'A' eller 'B'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
