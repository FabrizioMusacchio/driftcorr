"""Controlled synthetic drift experiments with explicit ground truth.

This module generates benchmark datasets where true trajectories and injected
drift components are known. It supports reproducible method evaluation beyond
proxy metrics by directly comparing corrected outputs to ground truth.

Generated components:
- Ground-truth cell motion (stationary + migrating subsets).
- Global translation drift over time.
- Spatially varying smooth local drift (low-frequency RBF field).
- Optional temporal smoothness in drift dynamics.
- Measurement noise.

Observation model:
- `x_obs(track,t) = x_true(track,t) + d_global(t) + d_local(x_true,t) + noise`.

Outputs:
- `df_orig`: observed drift-contaminated tracks.
- `df_groundtruth`: drift-free tracks.
- `drift_field_groundtruth`: per-row global/local/total drift vectors.

Ground-truth evaluation:
- RMS error per track vs ground truth.
- RMS error over time.
- Summary tables and standard plots for corrected-vs-groundtruth comparisons.

Controlling parameters:
- dataset size (`n_tracks`, `n_times`, `n_files`),
- motion composition (`stationary_fraction`, `migrating_fraction`),
- drift structure (`global_drift_scale_xyz`, `local_lengthscale_um`,
  `local_drift_scale_um`, `n_rbf_centers`),
- smoothness/noise (`temporal_smoothness`, `measurement_noise_um`).

Effects:
- Larger local drift scale/shorter lengthscale make correction harder.
- Higher temporal smoothness yields more slowly varying drift.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks


def _smooth_random_series(
    rng: np.random.Generator,
    t: int,
    dim: int,
    scale: float,
    temporal_smoothness: float,
) -> np.ndarray:
    temporal_smoothness = float(np.clip(temporal_smoothness, 0.0, 0.999))
    out = np.zeros((t, dim), dtype=float)
    out[0] = rng.normal(scale=scale, size=dim)
    for i in range(1, t):
        noise = rng.normal(scale=scale, size=dim)
        out[i] = temporal_smoothness * out[i - 1] + (1.0 - temporal_smoothness) * noise
    return out


def _rbf_kernel(points: np.ndarray, centers: np.ndarray, lengthscale: float) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    k = np.exp(-d2 / (2.0 * lengthscale * lengthscale))
    ksum = np.sum(k, axis=1, keepdims=True)
    ksum[ksum <= 1e-12] = 1.0
    return k / ksum


def generate_controlled_drift_dataset(
    n_tracks: int = 80,
    n_times: int = 20,
    stationary_fraction: float = 0.45,
    migrating_fraction: float = 0.35,
    seed: int = 0,
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((80.0, 620.0), (80.0, 620.0), (20.0, 320.0)),
    global_drift_scale_xyz: tuple[float, float, float] = (2.8, 2.2, 1.1),
    local_lengthscale_um: float = 220.0,
    local_drift_scale_um: float = 2.4,
    n_rbf_centers: int = 10,
    temporal_smoothness: float = 0.9,
    measurement_noise_um: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate 3D tracks with known drift and known no-drift groundtruth.

    Returns:
      df_orig: observed positions with drift applied
      df_groundtruth: true positions without imaging drift
      drift_field_groundtruth: per-row drift vectors and components
    """
    rng = np.random.default_rng(seed)

    n = int(n_tracks)
    t_count = int(n_times)
    times = np.arange(1, t_count + 1, dtype=float)
    track_ids = np.arange(100000, 100000 + n, dtype=float)

    x_rng, y_rng, z_rng = bounds_xyz
    p0 = np.column_stack(
        [
            rng.uniform(x_rng[0], x_rng[1], size=n),
            rng.uniform(y_rng[0], y_rng[1], size=n),
            rng.uniform(z_rng[0], z_rng[1], size=n),
        ]
    )

    n_stationary = max(1, int(np.floor(float(stationary_fraction) * n)))
    n_migrating = max(1, int(np.floor(float(migrating_fraction) * n)))
    n_migrating = min(n - n_stationary, n_migrating)

    perm = rng.permutation(n)
    idx_stationary = perm[:n_stationary]
    idx_migrating = perm[n_stationary : n_stationary + n_migrating]
    idx_slow = perm[n_stationary + n_migrating :]
    mask_stationary = np.zeros((n,), dtype=bool)
    mask_migrating = np.zeros((n,), dtype=bool)
    mask_slow = np.zeros((n,), dtype=bool)
    mask_stationary[idx_stationary] = True
    mask_migrating[idx_migrating] = True
    mask_slow[idx_slow] = True

    cell_type = np.array(["slow"] * n, dtype=object)
    cell_type[idx_stationary] = "stationary"
    cell_type[idx_migrating] = "migrating"

    x_true = np.zeros((n, t_count, 3), dtype=float)
    x_true[:, 0, :] = p0

    dir_mask = rng.random(n) < 0.5
    vel = rng.normal(scale=np.array([0.8, 0.8, 0.35]), size=(n, 3))
    vel[mask_stationary] = 0.0
    vel[mask_slow] *= 0.2
    vel[mask_migrating & (~dir_mask)] *= 0.0

    rw_step = rng.normal(scale=np.array([0.6, 0.6, 0.25]), size=(n, t_count, 3))
    rw_step[mask_stationary] *= 0.05
    rw_step[mask_slow] *= 0.25
    rw_step[mask_migrating & dir_mask] *= 0.2

    for ti in range(1, t_count):
        x_true[:, ti, :] = x_true[:, ti - 1, :] + vel + rw_step[:, ti, :]

    x_true[:, :, 0] = np.clip(x_true[:, :, 0], x_rng[0], x_rng[1])
    x_true[:, :, 1] = np.clip(x_true[:, :, 1], y_rng[0], y_rng[1])
    x_true[:, :, 2] = np.clip(x_true[:, :, 2], z_rng[0], z_rng[1])

    global_drift = _smooth_random_series(
        rng,
        t=t_count,
        dim=3,
        scale=1.0,
        temporal_smoothness=temporal_smoothness,
    )
    global_drift = np.cumsum(global_drift, axis=0)
    global_drift *= np.array(global_drift_scale_xyz, dtype=float)[None, :]

    centers = np.column_stack(
        [
            rng.uniform(x_rng[0], x_rng[1], size=n_rbf_centers),
            rng.uniform(y_rng[0], y_rng[1], size=n_rbf_centers),
            rng.uniform(z_rng[0], z_rng[1], size=n_rbf_centers),
        ]
    )
    wt = _smooth_random_series(
        rng,
        t=t_count,
        dim=3 * n_rbf_centers,
        scale=1.0,
        temporal_smoothness=temporal_smoothness,
    ).reshape(t_count, n_rbf_centers, 3)
    wt *= float(local_drift_scale_um)

    z_trend = _smooth_random_series(rng, t=t_count, dim=1, scale=1.0, temporal_smoothness=temporal_smoothness)[:, 0]
    z_trend *= 1.2

    drift_global_all = np.zeros_like(x_true)
    drift_local_all = np.zeros_like(x_true)

    z_mid = 0.5 * (z_rng[0] + z_rng[1])
    z_half = max(1e-6, 0.5 * (z_rng[1] - z_rng[0]))

    for ti in range(t_count):
        p_t = x_true[:, ti, :]
        k = _rbf_kernel(p_t, centers=centers, lengthscale=float(local_lengthscale_um))
        local = k @ wt[ti]

        # Smooth vertical trend: top cells drift more than middle/bottom, without hard clusters.
        z_norm = ((p_t[:, 2] - z_mid) / z_half)[:, None]
        z_component = z_norm * np.array([z_trend[ti], -0.35 * z_trend[ti], 0.15 * z_trend[ti]])[None, :]
        local = local + z_component

        drift_global_all[:, ti, :] = global_drift[ti][None, :]
        drift_local_all[:, ti, :] = local

    drift_total = drift_global_all + drift_local_all
    x_obs = x_true + drift_total + rng.normal(scale=float(measurement_noise_um), size=x_true.shape)

    rows_obs: list[dict[str, Any]] = []
    rows_gt: list[dict[str, Any]] = []
    rows_drift: list[dict[str, Any]] = []

    row_id = 1
    for i, tid in enumerate(track_ids):
        for ti, t in enumerate(times):
            xo = x_obs[i, ti]
            xg = x_true[i, ti]
            dg = drift_global_all[i, ti]
            dl = drift_local_all[i, ti]
            dtot = drift_total[i, ti]

            rows_obs.append(
                {
                    "Position X": float(xo[0]),
                    "Position Y": float(xo[1]),
                    "Position Z": float(xo[2]),
                    "Time": float(t),
                    "TrackID": float(tid),
                    "ID": int(row_id),
                    "CellType": str(cell_type[i]),
                }
            )
            rows_gt.append(
                {
                    "Position X": float(xg[0]),
                    "Position Y": float(xg[1]),
                    "Position Z": float(xg[2]),
                    "Time": float(t),
                    "TrackID": float(tid),
                    "ID": int(row_id),
                    "CellType": str(cell_type[i]),
                }
            )
            rows_drift.append(
                {
                    "Time": float(t),
                    "TrackID": float(tid),
                    "drift_x_global": float(dg[0]),
                    "drift_y_global": float(dg[1]),
                    "drift_z_global": float(dg[2]),
                    "drift_x_local": float(dl[0]),
                    "drift_y_local": float(dl[1]),
                    "drift_z_local": float(dl[2]),
                    "drift_x_total": float(dtot[0]),
                    "drift_y_total": float(dtot[1]),
                    "drift_z_total": float(dtot[2]),
                    "CellType": str(cell_type[i]),
                }
            )
            row_id += 1

    df_orig = pd.DataFrame(rows_obs)
    df_groundtruth = pd.DataFrame(rows_gt)
    drift_field_groundtruth = pd.DataFrame(rows_drift)
    return df_orig, df_groundtruth, drift_field_groundtruth


def _to_imaris_position_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Unit"] = "um"
    out["Category"] = "Spot"
    out["Collection"] = "Position"
    cols = ["Position X", "Position Y", "Position Z", "Unit", "Category", "Collection", "Time", "TrackID", "ID"]
    return out[cols]


def write_synthetic_imaris_table(df_orig: pd.DataFrame, out_path: str | Path) -> Path:
    """Write synthetic observed tracks as IMARIS-like Position table (.csv)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = _to_imaris_position_table(df_orig)
    table.to_csv(out_path, index=False)
    return out_path


def generate_synthetic_imaris_folder(
    out_dir: str | Path,
    n_files: int = 5,
    n_tracks: int = 80,
    n_times: int = 20,
    stationary_fraction: float = 0.45,
    migrating_fraction: float = 0.35,
    seed: int = 0,
    temporal_smoothness: float = 0.9,
) -> dict[str, Any]:
    """Create a synthetic IMARIS folder plus ground-truth sidecar files."""
    out_dir = Path(out_dir)
    gt_dir = out_dir / "_groundtruth"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for i in range(int(n_files)):
        file_seed = int(seed + i * 37)
        df_orig, df_gt, df_drift = generate_controlled_drift_dataset(
            n_tracks=n_tracks,
            n_times=n_times,
            stationary_fraction=stationary_fraction,
            migrating_fraction=migrating_fraction,
            seed=file_seed,
            temporal_smoothness=temporal_smoothness,
        )

        base = f"synthetic_drift_{i + 1:03d}"
        xlsx_path = out_dir / f"{base}.csv"
        write_synthetic_imaris_table(df_orig, xlsx_path)

        df_gt.to_csv(gt_dir / f"{base}_groundtruth.csv", index=False)
        df_drift.to_csv(gt_dir / f"{base}_drift_field_groundtruth.csv", index=False)

        meta = {
            "seed": file_seed,
            "n_tracks": int(n_tracks),
            "n_times": int(n_times),
            "stationary_fraction": float(stationary_fraction),
            "migrating_fraction": float(migrating_fraction),
            "temporal_smoothness": float(temporal_smoothness),
        }
        pd.DataFrame([meta]).to_csv(gt_dir / f"{base}_meta.csv", index=False)
        written_files.append(xlsx_path)

    return {"imaris_dir": out_dir, "groundtruth_dir": gt_dir, "files": written_files}


def load_groundtruth_for_imaris_file(imaris_file: str | Path) -> pd.DataFrame | None:
    """Load matching groundtruth CSV for one IMARIS file if available."""
    p = Path(imaris_file)
    gt_path = p.parent / "_groundtruth" / f"{p.stem}_groundtruth.csv"
    if not gt_path.exists():
        return None
    return pd.read_csv(gt_path)


def evaluate_corrected_vs_groundtruth(
    df_corr: pd.DataFrame,
    df_groundtruth: pd.DataFrame,
    corrected_cols: tuple[str, str, str] = ("Position X_corr", "Position Y_corr", "Position Z_corr"),
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare corrected positions against no-drift groundtruth.

    Returns:
      summary metrics
      merged table with per-row errors
      per-track RMS table
      per-time RMS table
    """
    dc = df_corr.copy()
    dgt = df_groundtruth.copy()

    if corrected_cols[0] not in dc.columns:
        raise ValueError(f"Corrected dataframe missing columns {corrected_cols}")

    dc = ensure_numeric_tracks(dc, pos_cols=corrected_cols)
    dgt = ensure_numeric_tracks(dgt, pos_cols=("Position X", "Position Y", "Position Z"))

    mc = dc[["TrackID", "Time", *corrected_cols]].copy()
    mc = mc.rename(columns={corrected_cols[0]: "Xc", corrected_cols[1]: "Yc", corrected_cols[2]: "Zc"})
    mg = dgt[["TrackID", "Time", "Position X", "Position Y", "Position Z"]].copy()
    mg = mg.rename(columns={"Position X": "Xg", "Position Y": "Yg", "Position Z": "Zg"})

    merged = mc.merge(mg, on=["TrackID", "Time"], how="inner")
    merged["err_x"] = merged["Xc"] - merged["Xg"]
    merged["err_y"] = merged["Yc"] - merged["Yg"]
    merged["err_z"] = merged["Zc"] - merged["Zg"]
    merged["err_norm"] = np.sqrt(merged["err_x"] ** 2 + merged["err_y"] ** 2 + merged["err_z"] ** 2)

    if "CellType" in dgt.columns:
        ttype = dgt[["TrackID", "CellType"]].drop_duplicates("TrackID")
        merged = merged.merge(ttype, on="TrackID", how="left")

    per_track = (
        merged.groupby("TrackID", as_index=False)
        .agg(rms_error=("err_norm", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))))
        .sort_values("TrackID")
        .reset_index(drop=True)
    )
    if "CellType" in merged.columns:
        per_track = per_track.merge(merged[["TrackID", "CellType"]].drop_duplicates("TrackID"), on="TrackID", how="left")

    per_time = (
        merged.groupby("Time", as_index=False)
        .agg(rms_error=("err_norm", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))))
        .sort_values("Time")
        .reset_index(drop=True)
    )

    summary = {
        "n_points": float(merged.shape[0]),
        "n_tracks": float(per_track.shape[0]),
        "overall_rms_error": float(np.sqrt(np.mean(merged["err_norm"].to_numpy(float) ** 2))) if merged.shape[0] else float("nan"),
        "median_track_rms_error": float(per_track["rms_error"].median()) if per_track.shape[0] else float("nan"),
        "p90_track_rms_error": float(per_track["rms_error"].quantile(0.90)) if per_track.shape[0] else float("nan"),
        "mean_time_rms_error": float(per_time["rms_error"].mean()) if per_time.shape[0] else float("nan"),
    }
    return summary, merged, per_track, per_time


def save_groundtruth_evaluation_outputs(
    base_out_prefix: str | Path,
    groundtruth_df: pd.DataFrame,
    corrected_by_method: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """Save groundtruth-based method comparison metrics and plots."""
    base_out_prefix = Path(base_out_prefix)
    base_out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_summary: list[dict[str, Any]] = []
    per_method_track: dict[str, pd.DataFrame] = {}
    per_method_time: dict[str, pd.DataFrame] = {}
    per_method_time_stationary: dict[str, pd.DataFrame] = {}
    per_method_time_migrating: dict[str, pd.DataFrame] = {}

    for method, df_corr in corrected_by_method.items():
        summary, merged, per_track, per_time = evaluate_corrected_vs_groundtruth(df_corr, groundtruth_df)
        summary["method"] = method
        rows_summary.append(summary)

        per_method_track[method] = per_track
        per_method_time[method] = per_time

        merged.to_csv(base_out_prefix.with_name(base_out_prefix.name + f"_gt_errors_{method}.csv"), index=False)
        per_track.to_csv(base_out_prefix.with_name(base_out_prefix.name + f"_gt_rms_per_track_{method}.csv"), index=False)
        per_time.to_csv(base_out_prefix.with_name(base_out_prefix.name + f"_gt_rms_over_time_{method}.csv"), index=False)

        if "CellType" in merged.columns:
            for ctype, store, suffix in [
                ("stationary", per_method_time_stationary, "stationary"),
                ("migrating", per_method_time_migrating, "migrating"),
            ]:
                tab = (
                    merged[merged["CellType"] == ctype]
                    .groupby("Time", as_index=False)
                    .agg(rms_error=("err_norm", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))))
                    .sort_values("Time")
                    .reset_index(drop=True)
                )
                store[method] = tab
                tab.to_csv(base_out_prefix.with_name(base_out_prefix.name + f"_gt_rms_{suffix}_over_time_{method}.csv"), index=False)

    summary_df = pd.DataFrame(rows_summary).sort_values("method").reset_index(drop=True)
    out_summary = base_out_prefix.with_name(base_out_prefix.name + "_gt_metrics_summary.csv")
    summary_df.to_csv(out_summary, index=False)

    out_hist = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_per_track_hist.png")
    out_time = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_over_time.png")
    out_time_stationary = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_stationary_over_time.png")
    out_time_migrating = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_migrating_over_time.png")

    fig = plt.figure(figsize=(9, 5))
    for method, tab in per_method_track.items():
        arr = tab["rms_error"].to_numpy(float)
        if arr.size:
            plt.hist(arr, bins=40, alpha=0.4, label=method)
    plt.xlabel("RMS error per track (um)")
    plt.ylabel("count")
    plt.title("Corrected vs Groundtruth: per-track RMS error")
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist, dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    for method, tab in per_method_time.items():
        if tab.shape[0]:
            plt.plot(tab["Time"].to_numpy(float), tab["rms_error"].to_numpy(float), marker="o", label=method)
    plt.xlabel("Time")
    plt.ylabel("RMS error (um)")
    plt.title("Corrected vs Groundtruth: RMS error over time")
    plt.ylim(bottom=0.0)
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_time, dpi=300)
    plt.close(fig)

    if per_method_time_stationary:
        fig = plt.figure(figsize=(9, 5))
        for method, tab in per_method_time_stationary.items():
            if tab.shape[0]:
                plt.plot(tab["Time"].to_numpy(float), tab["rms_error"].to_numpy(float), marker="o", label=method)
        plt.xlabel("Time")
        plt.ylabel("RMS error (um)")
        plt.title("Corrected vs Groundtruth: RMS_stationary_over_time")
        plt.ylim(bottom=0.0)
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()
        plt.savefig(out_time_stationary, dpi=300)
        plt.close(fig)

    if per_method_time_migrating:
        fig = plt.figure(figsize=(9, 5))
        for method, tab in per_method_time_migrating.items():
            if tab.shape[0]:
                plt.plot(tab["Time"].to_numpy(float), tab["rms_error"].to_numpy(float), marker="o", label=method)
        plt.xlabel("Time")
        plt.ylabel("RMS error (um)")
        plt.title("Corrected vs Groundtruth: RMS_migrating_over_time")
        plt.ylim(bottom=0.0)
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()
        plt.savefig(out_time_migrating, dpi=300)
        plt.close(fig)

    out: dict[str, Path] = {
        "gt_metrics_summary": out_summary,
        "gt_rms_per_track_hist": out_hist,
        "gt_rms_over_time": out_time,
    }
    if per_method_time_stationary:
        out["gt_rms_stationary_over_time"] = out_time_stationary
    if per_method_time_migrating:
        out["gt_rms_migrating_over_time"] = out_time_migrating
    return out
