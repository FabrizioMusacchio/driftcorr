"""Controlled synthetic drift experiments with explicit ground truth.

This module generates benchmark datasets where true trajectories and injected
drift components are known. It supports reproducible method evaluation beyond
proxy metrics by directly comparing corrected outputs to ground truth.

Generated components:
- Ground-truth cell motion (stationary + migrating subsets).
- Global translation drift over time.
- Spatially varying smooth local drift (low-frequency RBF field).
- Optional non-rigid regimes: global rotation drift, piecewise spatial drift,
  and abrupt jump-event drift.
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
from .metrics import per_time_groundtruth_metrics


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


def _rotate_about_z(points: np.ndarray, center_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate 3D points around z axis with fixed center."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    x = points[:, 0] - float(center_xyz[0])
    y = points[:, 1] - float(center_xyz[1])
    z = points[:, 2]
    xr = c * x - s * y + float(center_xyz[0])
    yr = s * x + c * y + float(center_xyz[1])
    return np.column_stack([xr, yr, z])


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
    local_mode: str = "smooth",
    piecewise_regions: int = 3,
    piecewise_transition_um: float = 90.0,
    dx_scale_um: float = 0.8,
    dx_distribution: str = "gaussian",
    dx_track_smoothness: float = 0.7,
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
    mode = str(local_mode)

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

    if mode == "nodrift_dx_sweep":
        # Association-only regime: no imaging drift, controlled per-frame motion magnitude.
        rng_dx = np.random.default_rng(int(seed) + 19013)
        x_true = np.zeros((n, t_count, 3), dtype=float)
        x_true[:, 0, :] = p0
        a = float(np.clip(dx_track_smoothness, 0.0, 0.98))
        sxy = float(max(dx_scale_um, 1e-6))
        sz = 0.6 * sxy

        # Persistent per-track drift-free motion direction and speed.
        base_dir = rng_dx.normal(size=(n, 3))
        base_norm = np.linalg.norm(base_dir, axis=1, keepdims=True)
        base_norm[base_norm <= 1e-9] = 1.0
        base_dir = base_dir / base_norm
        speed = rng_dx.uniform(0.8, 1.8, size=(n, 1)) * (2.8 * sxy)
        base_vel = base_dir * speed

        # Smooth oscillatory component for curvature / non-linear trajectories.
        phase_xy = rng_dx.uniform(0.0, 2.0 * np.pi, size=(n, 1))
        phase_z = rng_dx.uniform(0.0, 2.0 * np.pi, size=(n, 1))
        freq_xy = rng_dx.uniform(0.15, 0.65, size=(n, 1))
        freq_z = rng_dx.uniform(0.10, 0.45, size=(n, 1))
        osc_amp_xy = rng_dx.uniform(0.25, 0.55, size=(n, 1)) * (2.2 * sxy)
        osc_amp_z = rng_dx.uniform(0.15, 0.40, size=(n, 1)) * (1.2 * sxy)

        v_prev = base_vel.copy()
        for ti in range(1, t_count):
            if str(dx_distribution).lower() == "bounded_uniform":
                noise = rng_dx.uniform(low=[-sxy, -sxy, -sz], high=[sxy, sxy, sz], size=(n, 3))
            else:
                noise = rng_dx.normal(scale=np.array([sxy, sxy, sz]), size=(n, 3))

            # AR(1) velocity with stationary std approx equal to noise std.
            innovation = np.sqrt(max(1e-9, 1.0 - a * a)) * noise
            v_prev = a * v_prev + innovation + (1.0 - a) * 0.35 * base_vel

            tt = float(ti)
            osc = np.column_stack(
                [
                    (osc_amp_xy[:, 0] * np.sin(freq_xy[:, 0] * tt + phase_xy[:, 0])),
                    (osc_amp_xy[:, 0] * np.cos(freq_xy[:, 0] * tt + phase_xy[:, 0])),
                    (osc_amp_z[:, 0] * np.sin(freq_z[:, 0] * tt + phase_z[:, 0])),
                ]
            )
            # Small frame-to-frame zigzag component.
            zig_sign = -1.0 if (ti % 2 == 0) else 1.0
            zig = zig_sign * 0.22 * noise

            step = v_prev + osc + zig
            step[mask_stationary] *= 0.05
            step[mask_slow] *= 0.55
            x_true[:, ti, :] = x_true[:, ti - 1, :] + step

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

    piecewise_regions = max(1, int(piecewise_regions))
    region_centers = np.column_stack(
        [
            rng.uniform(x_rng[0], x_rng[1], size=piecewise_regions),
            rng.uniform(y_rng[0], y_rng[1], size=piecewise_regions),
            rng.uniform(z_rng[0], z_rng[1], size=piecewise_regions),
        ]
    )
    region_w = _smooth_random_series(
        rng,
        t=t_count,
        dim=3 * piecewise_regions,
        scale=1.0,
        temporal_smoothness=temporal_smoothness,
    ).reshape(t_count, piecewise_regions, 3)
    region_w *= float(local_drift_scale_um) * 1.2

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

        if str(local_mode) in {"piecewise_smooth", "piecewise_abrupt"}:
            d2r = np.sum((p_t[:, None, :] - region_centers[None, :, :]) ** 2, axis=2)
            if str(local_mode) == "piecewise_abrupt":
                idx = np.argmin(d2r, axis=1)
                wr = np.zeros_like(d2r)
                wr[np.arange(d2r.shape[0]), idx] = 1.0
            else:
                sig = max(float(piecewise_transition_um), 1e-6)
                wr = np.exp(-d2r / (2.0 * sig * sig))
                wsum = np.sum(wr, axis=1, keepdims=True)
                wsum[wsum <= 1e-12] = 1.0
                wr = wr / wsum
            local = local + wr @ region_w[ti]

        drift_global_all[:, ti, :] = global_drift[ti][None, :]
        drift_local_all[:, ti, :] = local

    if mode in {"rotation_drift", "piecewise_spatial_drift", "jump_events"}:
        # Use a dedicated RNG stream for new non-rigid regimes so existing modes
        # (`smooth`, `piecewise_smooth`, `piecewise_abrupt`) remain bitwise stable.
        rng_non_rigid = np.random.default_rng(int(seed) + 9173)
        ref_center = np.mean(x_true[:, 0, :], axis=0)
        base_global = drift_global_all.copy()
        base_local = drift_local_all.copy()

        if mode == "rotation_drift":
            total_deg = float(rng_non_rigid.uniform(1.0, 10.0))
            total_rad = np.deg2rad(total_deg) * float(rng_non_rigid.choice([-1.0, 1.0]))
            angle_base = np.linspace(0.0, total_rad, t_count, dtype=float)
            angle_pert = _smooth_random_series(
                rng_non_rigid,
                t=t_count,
                dim=1,
                scale=1.0,
                temporal_smoothness=temporal_smoothness,
            )[:, 0]
            angle_pert = 0.12 * total_rad * angle_pert
            angles = angle_base + angle_pert - float(angle_pert[0])
            trans = _smooth_random_series(
                rng_non_rigid,
                t=t_count,
                dim=3,
                scale=0.35,
                temporal_smoothness=temporal_smoothness,
            )
            trans = np.cumsum(trans, axis=0)
            trans *= np.array(global_drift_scale_xyz, dtype=float)[None, :] * 0.25
            for ti in range(t_count):
                p_t = x_true[:, ti, :]
                p_rot = _rotate_about_z(p_t, ref_center, float(angles[ti]))
                drift_local_all[:, ti, :] = p_rot - p_t
                drift_global_all[:, ti, :] = trans[ti][None, :]

        elif mode == "piecewise_spatial_drift":
            n_regions = int(np.clip(piecewise_regions, 2, 4))
            x_ref = x_true[:, 0, 0]
            q = np.linspace(0.0, 1.0, n_regions + 1, dtype=float)
            bounds = np.quantile(x_ref, q)
            x_centers = 0.5 * (bounds[:-1] + bounds[1:])
            region_traj = _smooth_random_series(
                rng_non_rigid,
                t=t_count,
                dim=3 * n_regions,
                scale=1.0,
                temporal_smoothness=temporal_smoothness,
            ).reshape(t_count, n_regions, 3)
            region_traj *= float(local_drift_scale_um) * 1.9

            # Keep a smaller global component while emphasizing region-wise divergence.
            drift_global_all = base_global * 0.45
            sig = max(float(piecewise_transition_um), 1e-6)
            for ti in range(t_count):
                p_t = x_true[:, ti, :]
                dx = p_t[:, 0:1] - x_centers[None, :]
                if sig <= 5.0:
                    idx = np.argmin(np.abs(dx), axis=1)
                    wr = np.zeros((p_t.shape[0], n_regions), dtype=float)
                    wr[np.arange(p_t.shape[0]), idx] = 1.0
                else:
                    wr = np.exp(-(dx * dx) / (2.0 * sig * sig))
                    wsum = np.sum(wr, axis=1, keepdims=True)
                    wsum[wsum <= 1e-12] = 1.0
                    wr = wr / wsum
                reg_local = wr @ region_traj[ti]
                z_norm = ((p_t[:, 2] - z_mid) / z_half)[:, None]
                z_extra = z_norm * np.array([0.45, -0.15, 0.25])[None, :] * float(local_drift_scale_um)
                drift_local_all[:, ti, :] = reg_local + 0.3 * base_local[:, ti, :] + z_extra

        elif mode == "jump_events":
            drift_local_all = 0.45 * base_local
            jump_count = int(min(3, max(2, t_count // 7)))
            jump_candidates = np.arange(2, max(3, t_count - 1), dtype=int)
            if jump_candidates.size > 0:
                jt = sorted(rng_non_rigid.choice(jump_candidates, size=min(jump_count, jump_candidates.size), replace=False).tolist())
            else:
                jt = []
            jump_vec = rng_non_rigid.normal(
                scale=np.array(global_drift_scale_xyz, dtype=float)[None, :] * 1.15,
                size=(len(jt), 3),
            )
            jump_offset = np.zeros((t_count, 3), dtype=float)
            acc = np.zeros((3,), dtype=float)
            j = 0
            for ti in range(t_count):
                while j < len(jt) and ti >= int(jt[j]):
                    acc = acc + jump_vec[j]
                    j += 1
                jump_offset[ti] = acc
            drift_global_all = base_global + jump_offset[None, :, :]

            # Optional small rotational jumps to create discontinuities beyond translation.
            rot_steps = rng_non_rigid.normal(scale=np.deg2rad(0.25), size=t_count)
            for t_jump in jt:
                rot_steps[int(t_jump)] += rng_non_rigid.uniform(np.deg2rad(-1.8), np.deg2rad(1.8))
            ang = np.cumsum(rot_steps)
            for ti in range(t_count):
                p_t = x_true[:, ti, :]
                p_rot = _rotate_about_z(p_t, ref_center, float(ang[ti]))
                drift_local_all[:, ti, :] += 0.6 * (p_rot - p_t)

    if mode == "nodrift_dx_sweep":
        drift_global_all = np.zeros_like(x_true)
        drift_local_all = np.zeros_like(x_true)

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
    local_mode: str = "smooth",
    piecewise_regions: int = 3,
    piecewise_transition_um: float = 90.0,
    global_drift_scale_xyz: tuple[float, float, float] = (2.8, 2.2, 1.1),
    local_lengthscale_um: float = 220.0,
    local_drift_scale_um: float = 2.4,
    n_rbf_centers: int = 10,
    measurement_noise_um: float = 0.2,
    dx_scale_um: float = 0.8,
    dx_distribution: str = "gaussian",
    dx_track_smoothness: float = 0.7,
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
            local_mode=local_mode,
            piecewise_regions=piecewise_regions,
            piecewise_transition_um=piecewise_transition_um,
            global_drift_scale_xyz=global_drift_scale_xyz,
            local_lengthscale_um=local_lengthscale_um,
            local_drift_scale_um=local_drift_scale_um,
            n_rbf_centers=n_rbf_centers,
            measurement_noise_um=measurement_noise_um,
            dx_scale_um=dx_scale_um,
            dx_distribution=dx_distribution,
            dx_track_smoothness=dx_track_smoothness,
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
            "local_mode": str(local_mode),
            "piecewise_regions": int(piecewise_regions),
            "piecewise_transition_um": float(piecewise_transition_um),
            "global_drift_scale_x": float(global_drift_scale_xyz[0]),
            "global_drift_scale_y": float(global_drift_scale_xyz[1]),
            "global_drift_scale_z": float(global_drift_scale_xyz[2]),
            "local_lengthscale_um": float(local_lengthscale_um),
            "local_drift_scale_um": float(local_drift_scale_um),
            "n_rbf_centers": int(n_rbf_centers),
            "measurement_noise_um": float(measurement_noise_um),
            "dx_scale_um": float(dx_scale_um),
            "dx_distribution": str(dx_distribution),
            "dx_track_smoothness": float(dx_track_smoothness),
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


def load_drift_field_for_imaris_file(imaris_file: str | Path) -> pd.DataFrame | None:
    """Load matching drift-field groundtruth CSV for one IMARIS file if available."""
    p = Path(imaris_file)
    drift_path = p.parent / "_groundtruth" / f"{p.stem}_drift_field_groundtruth.csv"
    if not drift_path.exists():
        return None
    return pd.read_csv(drift_path)


def load_meta_for_imaris_file(imaris_file: str | Path) -> dict[str, Any] | None:
    """Load matching synthetic meta CSV as dict for one IMARIS file if available."""
    p = Path(imaris_file)
    meta_path = p.parent / "_groundtruth" / f"{p.stem}_meta.csv"
    if not meta_path.exists():
        return None
    tab = pd.read_csv(meta_path)
    if tab.shape[0] == 0:
        return None
    row = tab.iloc[0].to_dict()
    return {str(k): v for k, v in row.items()}


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
    if "CellType" in merged.columns:
        for ct in ("stationary", "migrating"):
            mm = merged[merged["CellType"] == ct]
            if mm.shape[0] > 0:
                summary[f"overall_rms_error_{ct}"] = float(np.sqrt(np.mean(mm["err_norm"].to_numpy(float) ** 2)))
            else:
                summary[f"overall_rms_error_{ct}"] = float("nan")
    return summary, merged, per_track, per_time


def save_groundtruth_evaluation_outputs(
    base_out_prefix: str | Path,
    groundtruth_df: pd.DataFrame,
    corrected_by_method: dict[str, pd.DataFrame],
    drift_field_groundtruth: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Save groundtruth-based method comparison metrics and plots."""
    base_out_prefix = Path(base_out_prefix)
    base_out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_summary: list[dict[str, Any]] = []
    per_method_track: dict[str, pd.DataFrame] = {}
    per_method_time: dict[str, pd.DataFrame] = {}
    per_method_time_stationary: dict[str, pd.DataFrame] = {}
    per_method_time_migrating: dict[str, pd.DataFrame] = {}
    per_method_drift_bias: dict[str, pd.DataFrame] = {}
    per_method_time_ci: dict[str, pd.DataFrame] = {}

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

        if drift_field_groundtruth is not None and {"drift_x_total", "drift_y_total", "drift_z_total"}.issubset(drift_field_groundtruth.columns):
            pred = df_corr[["TrackID", "Time", "Position X", "Position Y", "Position Z", "Position X_corr", "Position Y_corr", "Position Z_corr"]].copy()
            pred["pred_dx"] = pred["Position X"].to_numpy(float) - pred["Position X_corr"].to_numpy(float)
            pred["pred_dy"] = pred["Position Y"].to_numpy(float) - pred["Position Y_corr"].to_numpy(float)
            pred["pred_dz"] = pred["Position Z"].to_numpy(float) - pred["Position Z_corr"].to_numpy(float)
            tru = drift_field_groundtruth[["TrackID", "Time", "drift_x_total", "drift_y_total", "drift_z_total"]].copy()
            mm = pred.merge(tru, on=["TrackID", "Time"], how="inner")
            if mm.shape[0] > 0:
                bt = (
                    mm.groupby("Time", as_index=False)
                    .agg(
                        pred_dx_median=("pred_dx", "median"),
                        pred_dy_median=("pred_dy", "median"),
                        pred_dz_median=("pred_dz", "median"),
                        true_dx_median=("drift_x_total", "median"),
                        true_dy_median=("drift_y_total", "median"),
                        true_dz_median=("drift_z_total", "median"),
                    )
                    .sort_values("Time")
                    .reset_index(drop=True)
                )
                bias = np.sqrt(
                    (bt["pred_dx_median"].to_numpy(float) - bt["true_dx_median"].to_numpy(float)) ** 2
                    + (bt["pred_dy_median"].to_numpy(float) - bt["true_dy_median"].to_numpy(float)) ** 2
                    + (bt["pred_dz_median"].to_numpy(float) - bt["true_dz_median"].to_numpy(float)) ** 2
                )
                bt["drift_bias_norm"] = bias
                bt.to_csv(base_out_prefix.with_name(base_out_prefix.name + f"_gt_drift_bias_over_time_{method}.csv"), index=False)
                per_method_drift_bias[method] = bt

        # Publication-quality per-time metrics with CI across tracks.
        time_ci = per_time_groundtruth_metrics(
            df_corr=df_corr,
            df_groundtruth=groundtruth_df,
            drift_field_groundtruth=drift_field_groundtruth,
            near_stationary_threshold_um=2.0,
            ci_z=1.96,
        )
        if time_ci.shape[0] > 0:
            per_method_time_ci[method] = time_ci
            time_ci.to_csv(base_out_prefix.with_name(base_out_prefix.name + f"_gt_per_time_metrics_ci_{method}.csv"), index=False)

    summary_df = pd.DataFrame(rows_summary).sort_values("method").reset_index(drop=True)
    out_summary = base_out_prefix.with_name(base_out_prefix.name + "_gt_metrics_summary.csv")
    summary_df.to_csv(out_summary, index=False)

    out_hist = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_per_track_hist.png")
    out_time = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_over_time.png")
    out_time_stationary = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_stationary_over_time.png")
    out_time_migrating = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_migrating_over_time.png")
    out_drift_bias = base_out_prefix.with_name(base_out_prefix.name + "_gt_drift_bias_over_time.png")

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

    if per_method_drift_bias:
        fig = plt.figure(figsize=(9, 5))
        for method, tab in per_method_drift_bias.items():
            if tab.shape[0]:
                plt.plot(tab["Time"].to_numpy(float), tab["drift_bias_norm"].to_numpy(float), marker="o", label=method)
        plt.xlabel("Time")
        plt.ylabel("||median(pred_drift)-median(true_drift)|| (um)")
        plt.title("Groundtruth drift bias over time")
        plt.ylim(bottom=0.0)
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()
        plt.savefig(out_drift_bias, dpi=300)
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
    if per_method_drift_bias:
        out["gt_drift_bias_over_time"] = out_drift_bias
    return out
