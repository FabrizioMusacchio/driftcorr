"""Stationary-track discovery for robust drift model fitting.

This module identifies tracks likely dominated by imaging drift rather than
biological migration. The selected subset can be used to fit drift models more
robustly in mixed populations.

Scoring criteria:
- RMS displacement magnitude per track.
- Per-track neighborhood-distance variability.

Selection logic:
- Keep tracks below configurable RMS and neighbor-variability thresholds.
- If too few are selected, enforce a minimum fraction by taking lowest-RMS tracks.

Mathematical proxies:
- `RMS_i = sqrt(E_t ||d_i(t)||^2)`.
- Neighbor relative stability from std of:
  `(dist_ij(t) - median_t dist_ij) / median_t dist_ij`.

Controlling parameters:
- `rms_threshold_um` / `rms_quantile`.
- `neighbor_relstd_threshold` / `neighbor_quantile`.
- `k_neighbors`, `min_fraction`.

Effects:
- Lower thresholds yield stricter subsets.
- Higher `min_fraction` forces broader coverage when strict criteria fail.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks
from .metrics import displacements_relative_to_anchor, pivot_tracks, rms_displacement_per_track


@dataclass
class StationarySelection:
    track_ids: set[float]
    table: pd.DataFrame
    diagnostics: dict[str, float | int]


def _knn_from_points(p: np.ndarray, k: int = 6) -> np.ndarray:
    n = p.shape[0]
    nbrs = np.full((n, k), -1, dtype=int)
    if n < 2:
        return nbrs
    diff = p[:, None, :] - p[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    for i in range(n):
        order = np.argsort(d2[i])
        take = order[:k]
        nbrs[i, : len(take)] = take
    return nbrs


def _per_track_neighbor_relstd(x: np.ndarray, k_neighbors: int = 6) -> np.ndarray:
    p = np.nanmean(x, axis=1)
    nbrs = _knn_from_points(p, k=k_neighbors)

    n = x.shape[0]
    out = np.full((n,), np.nan, dtype=float)
    for i in range(n):
        vals = []
        for j in nbrs[i]:
            if j < 0:
                continue
            xi = x[i]
            xj = x[j]
            valid = np.all(np.isfinite(xi), axis=1) & np.all(np.isfinite(xj), axis=1)
            if np.sum(valid) < 4:
                continue
            d = np.linalg.norm(xi[valid] - xj[valid], axis=1)
            med = np.nanmedian(d)
            if med <= 1e-9:
                continue
            rel = (d - med) / med
            vals.append(float(np.nanstd(rel)))
        if vals:
            out[i] = float(np.nanmedian(vals))
    return out


def find_stationary_tracks(
    df: pd.DataFrame,
    times: np.ndarray | None = None,
    rms_threshold_um: float | None = None,
    neighbor_relstd_threshold: float | None = None,
    rms_quantile: float = 0.4,
    neighbor_quantile: float = 0.5,
    k_neighbors: int = 6,
    min_fraction: float = 0.1,
) -> StationarySelection:
    """Discover likely stationary tracks using displacement and neighborhood stability."""
    d = ensure_numeric_tracks(df)
    _, track_ids, x = pivot_tracks(d)

    if times is not None:
        t_all = np.array(sorted(d["Time"].unique()), dtype=float)
        t_to_idx = {float(t): i for i, t in enumerate(t_all)}
        valid_t = [t_to_idx[float(t)] for t in times if float(t) in t_to_idx]
        if valid_t:
            x = x[:, np.array(valid_t, dtype=int), :]

    rms = rms_displacement_per_track(x)
    nbr_relstd = _per_track_neighbor_relstd(x, k_neighbors=k_neighbors)

    finite_rms = np.isfinite(rms)
    finite_nbr = np.isfinite(nbr_relstd)

    if rms_threshold_um is None:
        rms_threshold_um = float(np.nanquantile(rms[finite_rms], np.clip(rms_quantile, 0.05, 0.95))) if np.any(finite_rms) else np.nan
    if neighbor_relstd_threshold is None:
        neighbor_relstd_threshold = (
            float(np.nanquantile(nbr_relstd[finite_nbr], np.clip(neighbor_quantile, 0.05, 0.95))) if np.any(finite_nbr) else np.nan
        )

    mask = finite_rms & finite_nbr & (rms <= float(rms_threshold_um)) & (nbr_relstd <= float(neighbor_relstd_threshold))

    min_count = max(1, int(np.ceil(float(min_fraction) * track_ids.size)))
    if np.sum(mask) < min_count:
        order = np.argsort(np.where(np.isfinite(rms), rms, np.inf))
        top = order[:min_count]
        mask = np.zeros_like(mask)
        mask[top] = True

    selected_ids = set(track_ids[mask].astype(float).tolist())

    means = np.nanmean(x, axis=1)
    table = pd.DataFrame(
        {
            "TrackID": track_ids.astype(float),
            "selected_stationary": mask.astype(bool),
            "rms_displacement": rms,
            "neighbor_relstd": nbr_relstd,
            "mean_x": means[:, 0],
            "mean_y": means[:, 1],
            "mean_z": means[:, 2],
        }
    )

    sel_means = means[mask]
    diagnostics: dict[str, float | int] = {
        "n_tracks_total": int(track_ids.size),
        "n_tracks_selected": int(np.sum(mask)),
        "fraction_selected": float(np.mean(mask)) if track_ids.size else 0.0,
        "rms_threshold_um": float(rms_threshold_um),
        "neighbor_relstd_threshold": float(neighbor_relstd_threshold),
    }
    if sel_means.size:
        diagnostics.update(
            {
                "selected_mean_x": float(np.nanmean(sel_means[:, 0])),
                "selected_mean_y": float(np.nanmean(sel_means[:, 1])),
                "selected_mean_z": float(np.nanmean(sel_means[:, 2])),
                "selected_spread_x": float(np.nanstd(sel_means[:, 0])),
                "selected_spread_y": float(np.nanstd(sel_means[:, 1])),
                "selected_spread_z": float(np.nanstd(sel_means[:, 2])),
            }
        )

    return StationarySelection(track_ids=selected_ids, table=table, diagnostics=diagnostics)


def evaluate_stationary_selection_impact(
    metrics_without_subset: dict[str, float | int | bool],
    metrics_with_subset: dict[str, float | int | bool],
) -> dict[str, float]:
    """Compute metric deltas for diagnostics."""
    keys = [
        "corr_mean_corr",
        "neighbor_relstd_median_corr",
        "rms_disp_median_corr",
        "near_stationary_fraction_corr",
    ]
    out: dict[str, float] = {}
    for k in keys:
        a = metrics_without_subset.get(k, np.nan)
        b = metrics_with_subset.get(k, np.nan)
        try:
            out[f"delta_{k}"] = float(b) - float(a)
        except Exception:
            out[f"delta_{k}"] = float("nan")
    return out
