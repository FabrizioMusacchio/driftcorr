"""Input/output utilities for IMARIS-like tracking tables.

This module standardizes reading and light validation for track tables with
columns such as `Position X/Y/Z`, `Time`, and `TrackID`.

Supported file formats:
- `.xls`, `.xlsx`: expected to contain sheet `Position` (skip first row by default).
- `.csv`: expected to contain the same tracking columns directly.

Mathematical role:
- None. This module performs schema handling and numeric coercion only.

Controlling parameters:
- `sheet_name`, `skiprows` in `read_imaris_position_xls` control Excel parsing.
- `ensure_numeric_tracks` chooses which position/time/id columns are coerced.

Effects:
- Stricter required-column checks fail early on malformed inputs.
- Numeric coercion with `errors='coerce'` followed by `dropna` removes invalid rows.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

REQUIRED_TRACK_COLUMNS = ["Position X", "Position Y", "Position Z", "Time", "TrackID"]


def read_imaris_position_xls(
    path: str | Path,
    sheet_name: str = "Position",
    skiprows: int = 1,
) -> pd.DataFrame:
    """Read IMARIS tracking export from the Position sheet."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p, sheet_name=sheet_name, skiprows=skiprows)
    missing = [c for c in REQUIRED_TRACK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")
    return df


def list_imaris_xls_files(imaris_dir: str | Path) -> list[Path]:
    """List IMARIS-like tracking files (.xls/.xlsx/.csv) in deterministic order."""
    d = Path(imaris_dir)
    files = sorted([*d.glob("*.xls"), *d.glob("*.xlsx"), *d.glob("*.csv")])
    return files


def load_imaris_directory(imaris_dir: str | Path) -> Mapping[str, pd.DataFrame]:
    """Load all IMARIS exports in a directory, keyed by filename."""
    out: dict[str, pd.DataFrame] = {}
    for path in list_imaris_xls_files(imaris_dir):
        out[path.name] = read_imaris_position_xls(path)
    return out


def ensure_numeric_tracks(
    df: pd.DataFrame,
    pos_cols: tuple[str, str, str] = ("Position X", "Position Y", "Position Z"),
    time_col: str = "Time",
    tid_col: str = "TrackID",
) -> pd.DataFrame:
    """Coerce track columns to numeric and drop incomplete rows."""
    out = df.copy()
    for col in [*pos_cols, time_col, tid_col]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[*pos_cols, time_col, tid_col]).copy()
    return out
