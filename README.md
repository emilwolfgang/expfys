# ExpFys Collision Analysis

This repository contains all code, data, and plots for the ExpFys collision experiments (Del A: 1D air-track riders, Del B: 2D air-table pucks).

## Layout

- `data/` – canonical storage for measurements and results:
  - `del_a/raw/` raw TSV exports from Qualisys (A1–A3).
  - `del_a/derived/` velocity derivatives (auto-generated when saving).
  - `del_a/summaries/` analysis outputs (`summary_A*.tsv`, `A*_None` tables).
  - `del_a/aux/weights.tsv` measured rider masses and collision thresholds.
  - `del_b/raw/` raw TSVs for B-series (including test datasets).
  - `del_b/summaries/` Del B summaries (`summary_B*.tsv` etc.).
  - `del_b/bodies/` puck mass/inertia files with thresholds.
- `expfys/` – reusable package:
  - `analysis/` per-part analyzers (`del_a.py`, `del_b_free.py`, `del_b_sticky.py`, `del_a_batch.py`).
  - `dataio/` shared QTM loaders.
  - `plotting/` publication-ready figure helpers.
  - `materials.py`, `summary_io.py`, `__init__.py` provide metadata and path utilities.
- `rapport/` – scripts that render the report figures (`plots.py`, auxiliary plot scripts).
- `scripts/run_batch.py` – unified CLI for batch processing any prefix.
- `tests/` – regression smoke tests.

## Usage

1. Install dependencies (NumPy, pandas, matplotlib, pytest).
2. Run batch analyses:
   - Del A: `python scripts/run_batch.py --prefix A1`
   - Del B free: `python scripts/run_batch.py --prefix B1`
   - Del B sticky: `python scripts/run_batch.py --prefix B2 --sticky`
   Results are written to `data/del_*/summaries/summary_<prefix>.tsv`.
3. Plot/report:
   - `python rapport/plots.py` regenerates the momentum/energy figures.
   - Individual Del B plots (`expfys_delB/plot_d*.py`) accept `--summary data/del_b/summaries/summary_B*.tsv`.

All code assumes the repository root is the working directory so relative paths resolve correctly.
