"""Stationary-track discovery for robust drift model fitting.

This module identifies tracks likely dominated by imaging drift rather than
biological migration.

Selection criteria combine:
- low short-lag MSD (tau=1 and tau=2),
- stable neighbor distances over time,
- optional displacement-correlation outlier score,
- optional RMS displacement threshold.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
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


def _msd_tau_per_track(x: np.ndarray, tau: int) -> np.ndarray:
    n, t, _ = x.shape
    out = np.full((n,), np.nan, dtype=float)
    if tau <= 0 or tau >= t:
        return out
    for i in range(n):
        xi = x[i]
        valid = np.all(np.isfinite(xi), axis=1)
        v = valid[:-tau] & valid[tau:]
        if np.sum(v) < 2:
            continue
        dx = xi[tau:][v] - xi[:-tau][v]
        out[i] = float(np.mean(np.sum(dx * dx, axis=1)))
    return out


def _corr_outlier_score(d: np.ndarray, min_overlap: int = 4) -> np.ndarray:
    """Per-track outlier score based on displacement-norm correlations.

    score_i = 1 - median_j corr(||d_i||, ||d_j||), lower is more globally coherent.
    """
    n = d.shape[0]
    s = np.linalg.norm(d, axis=2)
    out = np.full((n,), np.nan, dtype=float)

    for i in range(n):
        vals = []
        si = s[i]
        vi = np.isfinite(si)
        for j in range(n):
            if i == j:
                continue
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
                vals.append(c)
        if vals:
            out[i] = 1.0 - float(np.median(vals))
    return out


def find_stationary_tracks(
    df: pd.DataFrame,
    reference_time: float | None = None,
    times: np.ndarray | None = None,
    rms_threshold_um: float | None = None,
    neighbor_relstd_threshold: float | None = None,
    msd_tau1_threshold_um2: float | None = None,
    msd_tau2_threshold_um2: float | None = None,
    corr_outlier_threshold: float | None = None,
    rms_quantile: float = 0.4,
    neighbor_quantile: float = 0.5,
    msd_tau1_quantile: float = 0.45,
    msd_tau2_quantile: float = 0.45,
    corr_outlier_quantile: float = 0.6,
    k_neighbors: int = 6,
    min_fraction: float = 0.1,
    use_corr_outlier: bool = True,
) -> StationarySelection:
    """Discover likely stationary tracks from short-lag motion and local geometry stability."""
    d = ensure_numeric_tracks(df)
    t_all, track_ids, x = pivot_tracks(d)

    if times is not None:
        t_to_idx = {float(t): i for i, t in enumerate(t_all)}
        valid_t = [t_to_idx[float(t)] for t in times if float(t) in t_to_idx]
        if valid_t:
            x = x[:, np.array(valid_t, dtype=int), :]
            t_all = np.array([t_all[i] for i in valid_t], dtype=float)

    anchor_idx = None
    if reference_time is not None:
        idx = np.where(np.isclose(t_all, float(reference_time)))[0]
        if idx.size:
            anchor_idx = int(idx[0])

    drel = displacements_relative_to_anchor(x, anchor_index=anchor_idx)

    rms = rms_displacement_per_track(x, anchor_index=anchor_idx)
    nbr_relstd = _per_track_neighbor_relstd(x, k_neighbors=k_neighbors)
    msd_tau1 = _msd_tau_per_track(x, tau=1)
    msd_tau2 = _msd_tau_per_track(x, tau=2)
    corr_out = _corr_outlier_score(drel) if use_corr_outlier else np.full((track_ids.size,), np.nan, dtype=float)

    finite_rms = np.isfinite(rms)
    finite_nbr = np.isfinite(nbr_relstd)
    finite_m1 = np.isfinite(msd_tau1)
    finite_m2 = np.isfinite(msd_tau2)
    finite_co = np.isfinite(corr_out)

    if rms_threshold_um is None:
        rms_threshold_um = float(np.nanquantile(rms[finite_rms], np.clip(rms_quantile, 0.05, 0.95))) if np.any(finite_rms) else np.nan
    if neighbor_relstd_threshold is None:
        neighbor_relstd_threshold = float(np.nanquantile(nbr_relstd[finite_nbr], np.clip(neighbor_quantile, 0.05, 0.95))) if np.any(finite_nbr) else np.nan
    if msd_tau1_threshold_um2 is None:
        msd_tau1_threshold_um2 = float(np.nanquantile(msd_tau1[finite_m1], np.clip(msd_tau1_quantile, 0.05, 0.95))) if np.any(finite_m1) else np.nan
    if msd_tau2_threshold_um2 is None:
        msd_tau2_threshold_um2 = float(np.nanquantile(msd_tau2[finite_m2], np.clip(msd_tau2_quantile, 0.05, 0.95))) if np.any(finite_m2) else np.nan
    if corr_outlier_threshold is None and use_corr_outlier:
        corr_outlier_threshold = float(np.nanquantile(corr_out[finite_co], np.clip(corr_outlier_quantile, 0.05, 0.95))) if np.any(finite_co) else np.nan

    mask = finite_nbr & finite_m1 & finite_m2
    mask = mask & (nbr_relstd <= float(neighbor_relstd_threshold))
    mask = mask & (msd_tau1 <= float(msd_tau1_threshold_um2))
    mask = mask & (msd_tau2 <= float(msd_tau2_threshold_um2))

    if np.isfinite(rms_threshold_um):
        mask = mask & finite_rms & (rms <= float(rms_threshold_um))

    if use_corr_outlier and np.isfinite(corr_outlier_threshold):
        mask = mask & finite_co & (corr_out <= float(corr_outlier_threshold))

    min_count = max(1, int(np.ceil(float(min_fraction) * track_ids.size)))
    if np.sum(mask) < min_count:
        # Fallback: prefer low MSD tau1, then low neighbor variability.
        rank_m1 = np.argsort(np.where(np.isfinite(msd_tau1), msd_tau1, np.inf))
        rank_n = np.argsort(np.where(np.isfinite(nbr_relstd), nbr_relstd, np.inf))
        score = np.full((track_ids.size,), np.inf, dtype=float)
        inv_rank_m1 = np.empty_like(rank_m1)
        inv_rank_m1[rank_m1] = np.arange(track_ids.size)
        inv_rank_n = np.empty_like(rank_n)
        inv_rank_n[rank_n] = np.arange(track_ids.size)
        score = inv_rank_m1.astype(float) + inv_rank_n.astype(float)

        take = np.argsort(score)[:min_count]
        mask = np.zeros_like(mask)
        mask[take] = True

    selected_ids = set(track_ids[mask].astype(float).tolist())

    means = np.nanmean(x, axis=1)
    table = pd.DataFrame(
        {
            "TrackID": track_ids.astype(float),
            "selected_stationary": mask.astype(bool),
            "rms_displacement": rms,
            "neighbor_relstd": nbr_relstd,
            "msd_tau1": msd_tau1,
            "msd_tau2": msd_tau2,
            "corr_outlier": corr_out,
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
        "msd_tau1_threshold_um2": float(msd_tau1_threshold_um2),
        "msd_tau2_threshold_um2": float(msd_tau2_threshold_um2),
        "corr_outlier_threshold": float(corr_outlier_threshold) if corr_outlier_threshold is not None else float("nan"),
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


def plot_stationary_spatial_distribution(selection_table: pd.DataFrame, out_path: str) -> None:
    """Plot selected vs non-selected tracks in mean XY and XZ space."""
    if selection_table is None or selection_table.shape[0] == 0:
        return
    if not {"mean_x", "mean_y", "mean_z", "selected_stationary"}.issubset(selection_table.columns):
        return

    tab = selection_table.copy()
    sel = tab["selected_stationary"].astype(bool).to_numpy()

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax1.scatter(tab.loc[~sel, "mean_x"], tab.loc[~sel, "mean_y"], s=18, alpha=0.4, label="not selected")
    ax1.scatter(tab.loc[sel, "mean_x"], tab.loc[sel, "mean_y"], s=20, alpha=0.8, label="selected")
    ax1.set_xlabel("mean X")
    ax1.set_ylabel("mean Y")
    ax1.set_title("Stationary subset spatial distribution (XY)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(tab.loc[~sel, "mean_x"], tab.loc[~sel, "mean_z"], s=18, alpha=0.4, label="not selected")
    ax2.scatter(tab.loc[sel, "mean_x"], tab.loc[sel, "mean_z"], s=20, alpha=0.8, label="selected")
    ax2.set_xlabel("mean X")
    ax2.set_ylabel("mean Z")
    ax2.set_title("Stationary subset spatial distribution (XZ)")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
