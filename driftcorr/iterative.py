"""Generic iterative drift-correction runner.

This module provides a small wrapper for applying any drift-correction method
repeatedly to residual coordinates. Each iteration fits the supplied correction
function on the current coordinates, then uses the corrected output as the next
iteration's input.

Author
------
author: Fabrizio Musacchio
date: Jul 2026
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks

POS_COLS = ("Position X", "Position Y", "Position Z")
CORR_COLS = ("Position X_corr", "Position Y_corr", "Position Z_corr")
ROW_ID_COL = "_driftcorr_iter_row_id"


CorrectionFn = Callable[[pd.DataFrame, int, Path | None], pd.DataFrame]


def _require_corrected_columns(df_corr: pd.DataFrame, iteration: int) -> None:
    missing = [col for col in CORR_COLS if col not in df_corr.columns]
    if missing:
        raise ValueError(f"Iteration {iteration} correction output missing columns: {missing}")


def _as_cumulative_output(original_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    out = original_df.drop(columns=[ROW_ID_COL], errors="ignore").copy()
    current_xyz = current_df.loc[:, POS_COLS].to_numpy(float)
    out.loc[:, CORR_COLS] = current_xyz
    return out


def _coerce_corrected_table(corrected_df: pd.DataFrame) -> pd.DataFrame:
    out = corrected_df.copy()
    for col in [*CORR_COLS, "TrackID", "Time"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _align_corrected_xyz(
    current_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Align corrected coordinates to the current row order.

    Some correction methods leave rows uncorrected as NaN when no reliable model
    can be fitted for a timepoint. Missing corrected rows are treated as zero
    additional residual correction for that iteration.
    """
    corrected = _coerce_corrected_table(corrected_df)
    _require_corrected_columns(corrected, iteration=-1)

    current_xyz = current_df.loc[:, POS_COLS].to_numpy(float)
    corrected_xyz = current_xyz.copy()

    if ROW_ID_COL in current_df.columns and ROW_ID_COL in corrected.columns:
        key_cols = [ROW_ID_COL]
    else:
        key_cols = ["TrackID", "Time"]

    corr_keyed = corrected.loc[:, [*key_cols, *CORR_COLS]].copy()
    corr_keyed = corr_keyed.dropna(subset=[*key_cols, *CORR_COLS])
    corr_keyed = corr_keyed.drop_duplicates(subset=key_cols, keep="first")

    left = current_df.loc[:, key_cols].copy()
    left["_current_row_index"] = np.arange(current_df.shape[0], dtype=int)
    aligned = left.merge(corr_keyed, on=key_cols, how="left", sort=False)

    corr_values = aligned.loc[:, CORR_COLS].to_numpy(float)
    valid = np.all(np.isfinite(corr_values), axis=1)
    corrected_xyz[valid] = corr_values[valid]
    return corrected_xyz, valid


def _next_iteration_input(current_df: pd.DataFrame, corrected_xyz: np.ndarray) -> pd.DataFrame:
    nxt = current_df.copy()
    nxt.loc[:, POS_COLS] = corrected_xyz
    for col in CORR_COLS:
        if col in nxt.columns:
            nxt = nxt.drop(columns=[col])
    return ensure_numeric_tracks(nxt)


def run_iterative_drift_correction(
    df: pd.DataFrame,
    correct_once: CorrectionFn,
    n_iters: int = 3,
    convergence_tol: float | None = None,
    min_iters: int = 1,
    save_intermediate: bool = False,
    out_prefix: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a drift-correction function repeatedly to residual coordinates.

    Parameters
    ----------
    df:
        Original track table with ``Position X/Y/Z`` columns.
    correct_once:
        Callable receiving ``(current_df, iteration, iteration_out_prefix)`` and
        returning a dataframe with ``Position X_corr/Y_corr/Z_corr`` columns.
    n_iters:
        Maximum number of correction iterations.
    convergence_tol:
        Optional stopping threshold on the median per-row correction magnitude
        applied in one iteration. If ``None`` or ``<= 0``, no early stopping is
        used.
    min_iters:
        Minimum number of iterations before early stopping may trigger.
    save_intermediate:
        Whether cumulative corrected coordinates after each iteration should be
        written to ``<out_prefix>_iterXX.csv``.
    out_prefix:
        Optional path prefix for intermediate outputs and the iteration summary.

    Returns
    -------
    final_df, summary_df:
        ``final_df`` keeps the original coordinates and adds the final
        cumulative corrected coordinates. ``summary_df`` contains one row per
        completed iteration.
    """
    original = ensure_numeric_tracks(df).copy().reset_index(drop=True)
    if ROW_ID_COL in original.columns:
        original = original.drop(columns=[ROW_ID_COL])
    original[ROW_ID_COL] = np.arange(original.shape[0], dtype=int)
    current = original.copy()
    for col in CORR_COLS:
        if col in current.columns:
            current = current.drop(columns=[col])

    n_max = max(1, int(n_iters))
    n_min = max(1, int(min_iters))
    tol = None if convergence_tol is None else float(convergence_tol)
    prefix = Path(out_prefix) if out_prefix is not None else None
    if prefix is not None:
        prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | bool]] = []
    stopped = False

    original_xyz = original.loc[:, POS_COLS].to_numpy(float)
    for iteration in range(1, n_max + 1):
        iter_prefix = None
        if save_intermediate and prefix is not None:
            iter_prefix = prefix.with_name(f"{prefix.name}_iter{iteration:02d}")

        corrected = correct_once(current, iteration, iter_prefix)
        corrected = _coerce_corrected_table(corrected)
        _require_corrected_columns(corrected, iteration)

        current_xyz = current.loc[:, POS_COLS].to_numpy(float)
        corrected_xyz, valid_correction = _align_corrected_xyz(current, corrected)

        step_delta = current_xyz - corrected_xyz
        step_norm = np.linalg.norm(step_delta, axis=1)
        cumulative_norm = np.linalg.norm(original_xyz - corrected_xyz, axis=1)
        finite_step = step_norm[np.isfinite(step_norm) & valid_correction]
        finite_cum = cumulative_norm[np.isfinite(cumulative_norm)]

        step_median = float(np.median(finite_step)) if finite_step.size else float("nan")
        row = {
            "iteration": int(iteration),
            "n_rows": int(current.shape[0]),
            "n_rows_correction_output": int(corrected.shape[0]),
            "n_rows_with_valid_correction": int(np.sum(valid_correction)),
            "n_rows_without_valid_correction": int(current.shape[0] - np.sum(valid_correction)),
            "step_correction_median": step_median,
            "step_correction_mean": float(np.mean(finite_step)) if finite_step.size else float("nan"),
            "step_correction_p95": float(np.percentile(finite_step, 95)) if finite_step.size else float("nan"),
            "step_correction_max": float(np.max(finite_step)) if finite_step.size else float("nan"),
            "cumulative_correction_median": float(np.median(finite_cum)) if finite_cum.size else float("nan"),
            "cumulative_correction_mean": float(np.mean(finite_cum)) if finite_cum.size else float("nan"),
            "stopped_by_tol": False,
        }

        current = _next_iteration_input(current, corrected_xyz)
        cumulative = _as_cumulative_output(original, current)
        if save_intermediate and prefix is not None:
            cumulative.to_csv(prefix.with_name(f"{prefix.name}_iter{iteration:02d}.csv"), index=False)

        if tol is not None and tol > 0 and iteration >= n_min and np.isfinite(step_median) and step_median <= tol:
            row["stopped_by_tol"] = True
            stopped = True
            rows.append(row)
            break
        rows.append(row)

    final = _as_cumulative_output(original, current)
    summary = pd.DataFrame(rows)
    if prefix is not None:
        summary.to_csv(prefix.with_name(f"{prefix.name}_iteration_summary.csv"), index=False)
    if stopped:
        final.attrs["iterative_stopped_by_tol"] = True
    return final, summary
