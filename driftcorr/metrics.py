"""Low-level metric primitives for drift correction validation.

This module provides tensorization and metric functions used by evaluation
pipelines and downstream analyses.

Implemented metric families:
- Track tensor pivoting (`N x T x 3`).
- Displacements relative to anchor:
  `d_i(t) = x_i(t) - x_i(t_anchor)`.
- Pairwise trace correlation on displacement norms.
- Neighbor-distance stability over time.
- MSD per track and summary statistics.
- RMS displacement per track and near-stationary fraction.

Key controlling parameters:
- `min_overlap` in pairwise correlations.
- `k_neighbors` and `min_pairs` in neighbor-distance statistics.
- `threshold_um` in near-stationary fraction.

Effects:
- Stricter overlaps and pair-count constraints trade robustness for sample size.
- Neighbor count changes spatial locality sensitivity.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks
# %% FUNCTIONS
def pivot_tracks(
    df: pd.DataFrame,
    xcol: str = "Position X",
    ycol: str = "Position Y",
    zcol: str = "Position Z",
    time_col: str = "Time",
    tid_col: str = "TrackID",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot row-wise tracks into a dense tensor (N tracks, T times, 3 axes)."""
    d = df[[tid_col, time_col, xcol, ycol, zcol]].copy()
    d[tid_col] = pd.to_numeric(d[tid_col], errors="coerce")
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d[zcol] = pd.to_numeric(d[zcol], errors="coerce")
    d = d.dropna()

    times = np.array(sorted(d[time_col].unique()), dtype=float)
    track_ids = np.array(sorted(d[tid_col].unique()), dtype=float)

    tid_to_idx = {float(tid): i for i, tid in enumerate(track_ids)}
    t_to_idx = {float(t): j for j, t in enumerate(times)}

    x = np.full((len(track_ids), len(times), 3), np.nan, dtype=float)

    dfr = d.rename(
        columns={
            tid_col: "TrackID_",
            time_col: "Time_",
            xcol: "PX_",
            ycol: "PY_",
            zcol: "PZ_",
        }
    )

    for row in dfr.itertuples(index=False):
        i = tid_to_idx[float(row.TrackID_)]
        j = t_to_idx[float(row.Time_)]
        x[i, j, 0] = float(row.PX_)
        x[i, j, 1] = float(row.PY_)
        x[i, j, 2] = float(row.PZ_)

    return times, track_ids, x

def displacements_relative_to_anchor(
    x: np.ndarray,
    anchor_index: int | None = None,
) -> np.ndarray:
    """Compute displacement traces relative to per-track first valid point or shared anchor index."""
    n, t, _ = x.shape
    d = np.full_like(x, np.nan, dtype=float)
    for i in range(n):
        xi = x[i]
        valid = np.all(np.isfinite(xi), axis=1)
        if not np.any(valid):
            continue
        if anchor_index is None:
            t0 = int(np.where(valid)[0][0])
        else:
            t0 = int(anchor_index)
            if t0 < 0 or t0 >= t or not np.all(np.isfinite(xi[t0])):
                continue
        x0 = xi[t0]
        for j in range(t):
            if np.all(np.isfinite(xi[j])):
                d[i, j] = xi[j] - x0
    return d

def pairwise_trace_correlation(d: np.ndarray, min_overlap: int = 4) -> np.ndarray:
    """Pairwise correlation of displacement norms across tracks."""
    n, _, _ = d.shape
    s = np.linalg.norm(d, axis=2)
    corrs: list[float] = []
    for i in range(n):
        si = s[i]
        vi = np.isfinite(si)
        for j in range(i + 1, n):
            sj = s[j]
            vj = np.isfinite(sj)
            vv = vi & vj
            if int(np.sum(vv)) < int(min_overlap):
                continue
            a = si[vv]
            b = sj[vv]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(c):
                corrs.append(c)
    return np.array(corrs, dtype=float)

def _mean_position_per_track(x: np.ndarray) -> np.ndarray:
    return np.nanmean(x, axis=1)

def _knn_from_points(p: np.ndarray, k: int = 8) -> np.ndarray:
    n = p.shape[0]
    nbrs = np.full((n, k), -1, dtype=int)

    finite = np.all(np.isfinite(p), axis=1)
    idx = np.where(finite)[0]
    if idx.size < 2:
        return nbrs

    pf = p[idx]
    diff = pf[:, None, :] - pf[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)

    for a, ii in enumerate(idx):
        order = np.argsort(d2[a])
        take = order[:k]
        nbrs[ii, : len(take)] = idx[take]
    return nbrs

def neighbor_distance_stats(
    x: np.ndarray,
    k_neighbors: int = 8,
    min_pairs: int = 20,
) -> dict[str, float | bool]:
    """Summarize local-neighbor distance stability over time."""
    p = _mean_position_per_track(x)
    nbrs = _knn_from_points(p, k=k_neighbors)

    n, _, _ = x.shape
    rel_changes: list[float] = []

    for i in range(n):
        for nn in nbrs[i]:
            if nn < 0:
                continue
            xi = x[i]
            xj = x[nn]
            valid = np.all(np.isfinite(xi), axis=1) & np.all(np.isfinite(xj), axis=1)
            if np.sum(valid) < 4:
                continue
            dist = np.linalg.norm(xi[valid] - xj[valid], axis=1)
            med = np.nanmedian(dist)
            if med < 1e-9:
                continue
            rel = (dist - med) / med
            rel_changes.append(float(np.nanstd(rel)))

    arr = np.array(rel_changes, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_pairs:
        return {
            "ok": False,
            "n_pairs": int(arr.size),
            "rel_std_median": float("nan"),
            "rel_std_p90": float("nan"),
        }

    return {
        "ok": True,
        "n_pairs": int(arr.size),
        "rel_std_median": float(np.nanmedian(arr)),
        "rel_std_p90": float(np.nanpercentile(arr, 90)),
    }

def msd_per_track(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-track MSD curves for all valid lags."""
    n, t, _ = x.shape
    tau = np.arange(1, t, dtype=int)
    msd = np.full((n, t - 1), np.nan, dtype=float)

    for i in range(n):
        xi = x[i]
        valid = np.all(np.isfinite(xi), axis=1)
        if np.sum(valid) < 3:
            continue
        for dt in tau:
            v = valid[:-dt] & valid[dt:]
            if np.sum(v) < 2:
                continue
            dx = xi[dt:][v] - xi[:-dt][v]
            msd[i, dt - 1] = float(np.mean(np.sum(dx * dx, axis=1)))
    return tau.astype(float), msd

def summarize_msd(msd: np.ndarray) -> dict[str, float | bool]:
    """Summarize MSD via medians across tracks."""
    if msd.size == 0:
        return {"ok": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(msd, axis=0)
    return {
        "ok": True,
        "msd_tau1_median": float(med[0]) if med.size >= 1 else float("nan"),
        "msd_tau2_median": float(med[1]) if med.size >= 2 else float("nan"),
        "msd_tau3_median": float(med[2]) if med.size >= 3 else float("nan"),
        "msd_last_median": float(med[-1]) if med.size >= 1 else float("nan"),
    }

def msd_summary(msd: np.ndarray) -> dict[str, float | bool]:
    """Alias for summarize_msd for API clarity."""
    return summarize_msd(msd)

def rms_displacement_per_track(
    x: np.ndarray,
    anchor_index: int | None = None,
) -> np.ndarray:
    """Per-track RMS displacement magnitude after anchoring."""
    d = displacements_relative_to_anchor(x, anchor_index=anchor_index)
    mag = np.linalg.norm(d, axis=2)
    rms = np.sqrt(np.nanmean(mag * mag, axis=1))
    return rms

def fraction_near_stationary(rms_per_track: np.ndarray, threshold_um: float = 2.0) -> float:
    """Fraction of tracks with RMS displacement below threshold."""
    a = np.asarray(rms_per_track, dtype=float)
    m = np.isfinite(a)
    if not np.any(m):
        return float("nan")
    return float(np.mean(a[m] < float(threshold_um)))

def align_orig_and_corr(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    orig_pos_cols: tuple[str, str, str] = ("Position X", "Position Y", "Position Z"),
    corr_pos_cols: tuple[str, str, str] = ("Position X_corr", "Position Y_corr", "Position Z_corr"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned tensors for original and corrected tracks over common IDs and times."""
    do = ensure_numeric_tracks(df_orig, pos_cols=orig_pos_cols)
    dc = ensure_numeric_tracks(df_corr, pos_cols=corr_pos_cols)

    times_o, tids_o, xo = pivot_tracks(do, *orig_pos_cols)
    times_c, tids_c, xc = pivot_tracks(dc, *corr_pos_cols)

    common_t = np.intersect1d(times_o, times_c)
    common_id = np.intersect1d(tids_o, tids_c)

    tid_to_i_o = {float(t): i for i, t in enumerate(tids_o)}
    tid_to_i_c = {float(t): i for i, t in enumerate(tids_c)}
    t_to_j_o = {float(t): j for j, t in enumerate(times_o)}
    t_to_j_c = {float(t): j for j, t in enumerate(times_c)}

    i_o = np.array([tid_to_i_o[float(tid)] for tid in common_id], dtype=int)
    i_c = np.array([tid_to_i_c[float(tid)] for tid in common_id], dtype=int)
    j_o = np.array([t_to_j_o[float(t)] for t in common_t], dtype=int)
    j_c = np.array([t_to_j_c[float(t)] for t in common_t], dtype=int)

    xo2 = xo[i_o][:, j_o]
    xc2 = xc[i_c][:, j_c]
    return common_t, common_id, xo2, xc2

def scalarize_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str | bool]:
    """Return only scalar metric entries for tabular summaries."""
    out: dict[str, float | int | str | bool] = {}
    for k, v in metrics.items():
        if np.isscalar(v) or isinstance(v, (str, bool)):
            out[k] = v
    return out

def _rms_ci_from_squared_errors(e2: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    """Approximate RMS confidence interval from per-sample squared errors."""
    v = np.asarray(e2, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(v))
    if v.size == 1:
        rms = float(np.sqrt(max(mu, 0.0)))
        return rms, rms, rms
    se = float(np.std(v, ddof=1) / np.sqrt(v.size))
    lo = max(mu - z * se, 0.0)
    hi = max(mu + z * se, 0.0)
    return float(np.sqrt(mu)), float(np.sqrt(lo)), float(np.sqrt(hi))

def per_time_groundtruth_metrics(
    df_corr: pd.DataFrame,
    df_groundtruth: pd.DataFrame,
    drift_field_groundtruth: pd.DataFrame | None = None,
    near_stationary_threshold_um: float = 2.0,
    ci_z: float = 1.96,
) -> pd.DataFrame:
    """Compute per-time synthetic metrics with simple confidence intervals across tracks.

    Metrics per time and group (`all`, `stationary`, `migrating`):
    - RMS corrected-vs-groundtruth position error with CI.
    - Drift bias magnitude: ||median(pred_drift)-median(true_drift)|| (if drift GT provided).
    - Drift variance ratio: var(pred_drift)/var(true_drift) (if drift GT provided).
    - Fraction near-stationary after correction using corrected displacement to per-track anchor.
    """
    dc = ensure_numeric_tracks(df_corr, pos_cols=("Position X_corr", "Position Y_corr", "Position Z_corr"))
    dg = ensure_numeric_tracks(df_groundtruth, pos_cols=("Position X", "Position Y", "Position Z"))

    mc = dc[["TrackID", "Time", "Position X", "Position Y", "Position Z", "Position X_corr", "Position Y_corr", "Position Z_corr"]].copy()
    mg = dg[["TrackID", "Time", "Position X", "Position Y", "Position Z"]].copy()
    mg = mg.rename(columns={"Position X": "X_gt", "Position Y": "Y_gt", "Position Z": "Z_gt"})
    mm = mc.merge(mg, on=["TrackID", "Time"], how="inner")
    if "CellType" in dg.columns:
        mm = mm.merge(dg[["TrackID", "CellType"]].drop_duplicates("TrackID"), on="TrackID", how="left")

    mm["err2"] = (
        (mm["Position X_corr"].to_numpy(float) - mm["X_gt"].to_numpy(float)) ** 2
        + (mm["Position Y_corr"].to_numpy(float) - mm["Y_gt"].to_numpy(float)) ** 2
        + (mm["Position Z_corr"].to_numpy(float) - mm["Z_gt"].to_numpy(float)) ** 2
    )
    mm["pred_dx"] = mm["Position X"].to_numpy(float) - mm["Position X_corr"].to_numpy(float)
    mm["pred_dy"] = mm["Position Y"].to_numpy(float) - mm["Position Y_corr"].to_numpy(float)
    mm["pred_dz"] = mm["Position Z"].to_numpy(float) - mm["Position Z_corr"].to_numpy(float)

    # Corrected displacement-to-anchor magnitude per row.
    anc = (
        dc.sort_values(["TrackID", "Time"])
        .groupby("TrackID", as_index=False)[["Position X_corr", "Position Y_corr", "Position Z_corr"]]
        .first()
        .rename(columns={"Position X_corr": "X0", "Position Y_corr": "Y0", "Position Z_corr": "Z0"})
    )
    mm = mm.merge(anc, on="TrackID", how="left")
    mm["corr_disp_mag"] = np.sqrt(
        (mm["Position X_corr"].to_numpy(float) - mm["X0"].to_numpy(float)) ** 2
        + (mm["Position Y_corr"].to_numpy(float) - mm["Y0"].to_numpy(float)) ** 2
        + (mm["Position Z_corr"].to_numpy(float) - mm["Z0"].to_numpy(float)) ** 2
    )

    ddr: pd.DataFrame | None = None
    if drift_field_groundtruth is not None and {"drift_x_total", "drift_y_total", "drift_z_total"}.issubset(drift_field_groundtruth.columns):
        ddr = drift_field_groundtruth[["TrackID", "Time", "drift_x_total", "drift_y_total", "drift_z_total"]].copy()
        mm = mm.merge(ddr, on=["TrackID", "Time"], how="left")

    groups: list[tuple[str, pd.DataFrame]] = [("all", mm)]
    if "CellType" in mm.columns:
        groups.append(("stationary", mm[mm["CellType"] == "stationary"].copy()))
        groups.append(("migrating", mm[mm["CellType"] == "migrating"].copy()))

    rows: list[dict[str, float | str | int]] = []
    for gname, gg in groups:
        if gg.shape[0] == 0:
            continue
        for tt, gt in gg.groupby("Time", as_index=False):
            e2 = gt["err2"].to_numpy(float)
            rms, rms_lo, rms_hi = _rms_ci_from_squared_errors(e2, z=float(ci_z))
            row: dict[str, float | str | int] = {
                "Time": float(tt),
                "group": str(gname),
                "n_tracks": int(gt["TrackID"].nunique()),
                "rms_error": float(rms),
                "rms_error_ci_low": float(rms_lo),
                "rms_error_ci_high": float(rms_hi),
                "near_stationary_fraction": float(
                    np.mean(gt["corr_disp_mag"].to_numpy(float) <= float(near_stationary_threshold_um))
                )
                if gt.shape[0]
                else float("nan"),
            }
            if ddr is not None and {"drift_x_total", "drift_y_total", "drift_z_total"}.issubset(gt.columns):
                pred_med = np.array(
                    [
                        np.nanmedian(gt["pred_dx"].to_numpy(float)),
                        np.nanmedian(gt["pred_dy"].to_numpy(float)),
                        np.nanmedian(gt["pred_dz"].to_numpy(float)),
                    ],
                    dtype=float,
                )
                true_med = np.array(
                    [
                        np.nanmedian(gt["drift_x_total"].to_numpy(float)),
                        np.nanmedian(gt["drift_y_total"].to_numpy(float)),
                        np.nanmedian(gt["drift_z_total"].to_numpy(float)),
                    ],
                    dtype=float,
                )
                row["drift_bias_norm"] = float(np.linalg.norm(pred_med - true_med))
                pred_var = np.nanvar(gt[["pred_dx", "pred_dy", "pred_dz"]].to_numpy(float), axis=0)
                true_var = np.nanvar(gt[["drift_x_total", "drift_y_total", "drift_z_total"]].to_numpy(float), axis=0)
                num = float(np.nansum(pred_var))
                den = float(np.nansum(true_var))
                row["drift_var_ratio"] = float(num / den) if den > 1e-12 else float("nan")
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.shape[0]:
        out = out.sort_values(["group", "Time"]).reset_index(drop=True)
    return out
# %% END