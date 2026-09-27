"""Diagnostics for association ambiguity in dx-sweep benchmarks.

This module provides per-time diagnostics independent of assignment quality:
- nearest-neighbor spacing in each frame, ``d_NN(t)``,
- true displacement magnitude between consecutive frames, ``Δx(t)``,
- ambiguity ratio ``ρ(t) = Δx(t) / d_NN(t)``.

These curves explain when association becomes difficult as displacement
approaches or exceeds inter-cell spacing.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import PointCloudFrame

# %% FUNCTIONS
def compute_nn_distance(frames: list[PointCloudFrame]) -> pd.DataFrame:
    """Compute per-frame nearest-neighbor distance statistics.

    Parameters
    ----------
    frames:
        Sequence of point-cloud frames.

    Returns
    -------
    pd.DataFrame
        Columns: ``Time, nn_median, nn_mean, nn_p90, n_points``.
    """
    rows: list[dict[str, float]] = []
    for f in sorted(frames, key=lambda v: float(v.time)):
        x = np.asarray(f.xyz, dtype=float)
        n = int(x.shape[0])
        if n < 2:
            rows.append(
                {
                    "Time": float(f.time),
                    "nn_median": np.nan,
                    "nn_mean": np.nan,
                    "nn_p90": np.nan,
                    "n_points": n,
                }
            )
            continue
        # Pairwise squared distances.
        diff = x[:, None, :] - x[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        rows.append(
            {
                "Time": float(f.time),
                "nn_median": float(np.nanmedian(nn)) if nn.size else np.nan,
                "nn_mean": float(np.nanmean(nn)) if nn.size else np.nan,
                "nn_p90": float(np.nanpercentile(nn, 90.0)) if nn.size else np.nan,
                "n_points": n,
            }
        )
    return pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)


def compute_displacement_from_gt(gt_tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-time GT displacement statistics between consecutive frames.

    Parameters
    ----------
    gt_tracks_df:
        Ground-truth tracks with columns ``TrackID, Time`` and xyz coordinates
        in either ``x/y/z`` or ``Position X/Y/Z``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Time, disp_median, disp_mean, disp_p90, n_pairs``.
        ``Time`` corresponds to the source frame of each transition ``t->t+1``.
    """
    if gt_tracks_df.empty:
        return pd.DataFrame(columns=["Time", "disp_median", "disp_mean", "disp_p90", "n_pairs"])

    d = gt_tracks_df.copy()
    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    d["TrackID"] = pd.to_numeric(d["TrackID"], errors="coerce")
    if {"x", "y", "z"}.issubset(d.columns):
        cols = ["x", "y", "z"]
    elif {"Position X", "Position Y", "Position Z"}.issubset(d.columns):
        cols = ["Position X", "Position Y", "Position Z"]
    else:
        raise ValueError("gt_tracks_df missing xyz columns")
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Time", "TrackID", *cols]).copy()

    times = np.sort(d["Time"].unique())
    if times.size < 2:
        return pd.DataFrame(columns=["Time", "disp_median", "disp_mean", "disp_p90", "n_pairs"])
    nxt = {float(times[i]): float(times[i + 1]) for i in range(times.size - 1)}

    rows: list[dict[str, float]] = []
    for t, t1 in nxt.items():
        a = d[d["Time"] == float(t)][["TrackID", *cols]].rename(columns={cols[0]: "x0", cols[1]: "y0", cols[2]: "z0"})
        b = d[d["Time"] == float(t1)][["TrackID", *cols]].rename(columns={cols[0]: "x1", cols[1]: "y1", cols[2]: "z1"})
        m = a.merge(b, on="TrackID", how="inner")
        if m.empty:
            rows.append({"Time": float(t), "disp_median": np.nan, "disp_mean": np.nan, "disp_p90": np.nan, "n_pairs": 0})
            continue
        dv = np.sqrt(
            (m["x1"].to_numpy(float) - m["x0"].to_numpy(float)) ** 2
            + (m["y1"].to_numpy(float) - m["y0"].to_numpy(float)) ** 2
            + (m["z1"].to_numpy(float) - m["z0"].to_numpy(float)) ** 2
        )
        rows.append(
            {
                "Time": float(t),
                "disp_median": float(np.nanmedian(dv)) if dv.size else np.nan,
                "disp_mean": float(np.nanmean(dv)) if dv.size else np.nan,
                "disp_p90": float(np.nanpercentile(dv, 90.0)) if dv.size else np.nan,
                "n_pairs": int(dv.size),
            }
        )
    return pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)


def compute_rho(nn_df: pd.DataFrame, disp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ambiguity ratio ``ρ = Δx / d_NN`` per time."""
    if nn_df.empty or disp_df.empty:
        return pd.DataFrame(columns=["Time", "rho_median", "rho_mean", "nn_median", "disp_median"])
    n = nn_df.copy()
    d = disp_df.copy()
    for c in ["Time", "nn_median", "nn_mean"]:
        if c in n.columns:
            n[c] = pd.to_numeric(n[c], errors="coerce")
    for c in ["Time", "disp_median", "disp_mean"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    m = n.merge(d, on="Time", how="inner")
    if m.empty:
        return pd.DataFrame(columns=["Time", "rho_median", "rho_mean", "nn_median", "disp_median"])
    m["rho_median"] = m["disp_median"] / np.maximum(m["nn_median"], 1e-12)
    m["rho_mean"] = m["disp_mean"] / np.maximum(m["nn_mean"], 1e-12)
    cols = [
        "Time",
        "rho_median",
        "rho_mean",
        "nn_median",
        "nn_mean",
        "disp_median",
        "disp_mean",
    ]
    return m[cols].sort_values("Time").reset_index(drop=True)
# %% END