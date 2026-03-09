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
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks
from .metrics import per_time_groundtruth_metrics
# %% FUNCTIONS

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
    dx_mode: str = "fixed",
    dx_min_um: float = 0.2,
    dx_max_um: float = 4.0,
    dx_varying_distribution: str = "uniform",
    birthdeath_enable: bool = False,
    p_birth_late: float = 0.35,
    p_death_early: float = 0.35,
    min_life_frames: int = 5,
    p_birth_random: float = 0.5,
    birth_near_dist_min_um: float = 0.2,
    birth_near_dist_max_um: float = 4.0,
    birth_near_pick: str = "uniform_over_active_tracks",
    stationary_mode: str = "hard_zero",
    stationary_apply_to: str = "all_tracks",
    drift_rotation_enable: bool = True,
    drift_rotation_deg_scale: float = 2.0,
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
    dx_schedule = np.full((n, t_count), float(dx_scale_um), dtype=float)
    dx_source_all = np.full((n, t_count), "fixed", dtype=object)

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

    if mode in {
        "nodrift_dx_sweep",
        "nodrift_varying_dx",
        "nodrift_dx_varying_birthdeath",
        "varying_dx_birthdeath_stationary",
        "varying_dx_birthdeath_stationary_density",
        "varying_dx_birthdeath_with_drift",
    }:
        # Association-only regimes: no imaging drift with controlled motion
        # amplitudes. "nodrift_varying_dx" uses heteroscedastic dx schedules.
        if mode == "nodrift_dx_sweep":
            rng_offset = 19013
        elif mode == "nodrift_varying_dx":
            rng_offset = 29013
        elif mode == "nodrift_dx_varying_birthdeath":
            rng_offset = 39013
        elif mode in {"varying_dx_birthdeath_stationary", "varying_dx_birthdeath_stationary_density"}:
            rng_offset = 49013
        else:
            rng_offset = 59013
        rng_dx = np.random.default_rng(int(seed) + int(rng_offset))
        x_true = np.zeros((n, t_count, 3), dtype=float)
        x_true[:, 0, :] = p0
        a = float(np.clip(dx_track_smoothness, 0.0, 0.98))

        if mode in {
            "nodrift_varying_dx",
            "nodrift_dx_varying_birthdeath",
            "varying_dx_birthdeath_stationary",
            "varying_dx_birthdeath_stationary_density",
            "varying_dx_birthdeath_with_drift",
        }:
            dx_mode_l = str(dx_mode).lower()
            dx_lo = float(min(dx_min_um, dx_max_um))
            dx_hi = float(max(dx_min_um, dx_max_um))
            if dx_mode_l == "varying_per_track":
                dx_track = rng_dx.uniform(dx_lo, dx_hi, size=(n, 1))
                dx_schedule = np.repeat(dx_track, t_count, axis=1)
                dx_source_all[:, :] = "varying_per_track"
            elif dx_mode_l == "fixed":
                dx_schedule = np.full((n, t_count), float(max(dx_scale_um, 1e-6)), dtype=float)
                dx_source_all[:, :] = "fixed"
            else:
                if str(dx_varying_distribution).lower() == "uniform":
                    dx_schedule = rng_dx.uniform(dx_lo, dx_hi, size=(n, t_count))
                else:
                    # Fallback to uniform until more varying distributions are added.
                    dx_schedule = rng_dx.uniform(dx_lo, dx_hi, size=(n, t_count))
                dx_source_all[:, :] = "varying_per_track_time"
            dx_schedule = np.clip(dx_schedule, dx_lo, dx_hi)
        else:
            dx_schedule = np.full((n, t_count), float(max(dx_scale_um, 1e-6)), dtype=float)
            dx_source_all[:, :] = "fixed"

        # Persistent per-track drift-free motion direction and speed.
        base_dir = rng_dx.normal(size=(n, 3))
        base_norm = np.linalg.norm(base_dir, axis=1, keepdims=True)
        base_norm[base_norm <= 1e-9] = 1.0
        base_dir = base_dir / base_norm
        speed = rng_dx.uniform(0.8, 1.8, size=(n, 1))
        base_vel = base_dir * speed

        # Smooth oscillatory component for curvature / non-linear trajectories.
        phase_xy = rng_dx.uniform(0.0, 2.0 * np.pi, size=(n, 1))
        phase_z = rng_dx.uniform(0.0, 2.0 * np.pi, size=(n, 1))
        freq_xy = rng_dx.uniform(0.15, 0.65, size=(n, 1))
        freq_z = rng_dx.uniform(0.10, 0.45, size=(n, 1))
        osc_amp_xy = rng_dx.uniform(0.25, 0.55, size=(n, 1)) * 2.2
        osc_amp_z = rng_dx.uniform(0.15, 0.40, size=(n, 1)) * 1.2

        coh_centers = None
        coh_flow = None
        if mode == "varying_dx_birthdeath_with_drift":
            n_coh = 4
            coh_centers = np.column_stack(
                [
                    rng_dx.uniform(x_rng[0], x_rng[1], size=n_coh),
                    rng_dx.uniform(y_rng[0], y_rng[1], size=n_coh),
                    rng_dx.uniform(z_rng[0], z_rng[1], size=n_coh),
                ]
            )
            coh_flow = _smooth_random_series(
                rng_dx,
                t=t_count,
                dim=3 * n_coh,
                scale=1.0,
                temporal_smoothness=temporal_smoothness,
            ).reshape(t_count, n_coh, 3)
            coh_flow *= 0.65

        v_prev = base_vel * (2.8 * dx_schedule[:, [0]])
        for ti in range(1, t_count):
            step_scale = dx_schedule[:, [ti]]
            sxy = step_scale
            sz = 0.6 * step_scale
            if str(dx_distribution).lower() == "bounded_uniform":
                if mode == "nodrift_dx_sweep":
                    # Preserve legacy nodrift_dx_sweep behavior for reproducibility.
                    sxy0 = float(step_scale[0, 0])
                    sz0 = float(0.6 * sxy0)
                    noise = rng_dx.uniform(low=[-sxy0, -sxy0, -sz0], high=[sxy0, sxy0, sz0], size=(n, 3))
                else:
                    noise = np.column_stack(
                        [
                            rng_dx.uniform(low=-sxy[:, 0], high=sxy[:, 0], size=n),
                            rng_dx.uniform(low=-sxy[:, 0], high=sxy[:, 0], size=n),
                            rng_dx.uniform(low=-sz[:, 0], high=sz[:, 0], size=n),
                        ]
                    )
            else:
                noise = rng_dx.normal(size=(n, 3)) * np.column_stack([sxy[:, 0], sxy[:, 0], sz[:, 0]])

            # AR(1) velocity with stationary std approx equal to noise std.
            innovation = np.sqrt(max(1e-9, 1.0 - a * a)) * noise
            v_prev = a * v_prev + innovation + (1.0 - a) * 0.35 * (2.8 * step_scale * base_vel)

            tt = float(ti)
            osc = np.column_stack(
                [
                    (osc_amp_xy[:, 0] * step_scale[:, 0] * np.sin(freq_xy[:, 0] * tt + phase_xy[:, 0])),
                    (osc_amp_xy[:, 0] * step_scale[:, 0] * np.cos(freq_xy[:, 0] * tt + phase_xy[:, 0])),
                    (osc_amp_z[:, 0] * step_scale[:, 0] * np.sin(freq_z[:, 0] * tt + phase_z[:, 0])),
                ]
            )
            # Small frame-to-frame zigzag component.
            zig_sign = -1.0 if (ti % 2 == 0) else 1.0
            zig = zig_sign * 0.22 * noise

            step = v_prev + osc + zig
            if coh_centers is not None and coh_flow is not None:
                kc = _rbf_kernel(x_true[:, ti - 1, :], centers=coh_centers, lengthscale=180.0)
                cstep = kc @ coh_flow[ti]
                step[mask_slow] += (step_scale[mask_slow] * cstep[mask_slow])
            step[mask_stationary] *= 0.05
            step[mask_slow] *= 0.55
            x_true[:, ti, :] = x_true[:, ti - 1, :] + step

    # Optional birth/death process for association-focused heteroscedastic regime.
    t_birth_idx = np.zeros((n,), dtype=int)
    t_death_idx = np.full((n,), t_count - 1, dtype=int)
    birth_mode = np.full((n,), "initial", dtype=object)
    parent_track_id = np.full((n,), np.nan, dtype=float)
    init_dist_to_parent = np.full((n,), np.nan, dtype=float)

    birthdeath_active = bool(birthdeath_enable) or (
        mode in {
            "nodrift_dx_varying_birthdeath",
            "varying_dx_birthdeath_stationary",
            "varying_dx_birthdeath_stationary_density",
            "varying_dx_birthdeath_with_drift",
        }
    )
    if birthdeath_active:
        rng_bd = np.random.default_rng(int(seed) + 49013)
        min_life = int(max(2, min(int(min_life_frames), t_count)))
        birth_prob = float(np.clip(p_birth_late, 0.0, 1.0))
        death_prob = float(np.clip(p_death_early, 0.0, 1.0))
        p_random = float(np.clip(p_birth_random, 0.0, 1.0))
        near_d_lo = float(min(birth_near_dist_min_um, birth_near_dist_max_um))
        near_d_hi = float(max(birth_near_dist_min_um, birth_near_dist_max_um))

        # 1) Sample lifetimes.
        for i in range(n):
            if t_count - min_life >= 1 and rng_bd.random() < birth_prob:
                t_birth_idx[i] = int(rng_bd.integers(1, t_count - min_life + 1))
            else:
                t_birth_idx[i] = 0

            td_min = int(t_birth_idx[i] + min_life - 1)
            td_min = int(min(max(td_min, t_birth_idx[i]), t_count - 1))
            if td_min < (t_count - 1) and rng_bd.random() < death_prob:
                t_death_idx[i] = int(rng_bd.integers(td_min, t_count))
            else:
                t_death_idx[i] = t_count - 1

        # 2) Re-anchor tracks that start later than first frame.
        step_delta = x_true[:, 1:, :] - x_true[:, :-1, :]
        late_idx = np.flatnonzero(t_birth_idx > 0)
        late_idx = late_idx[np.argsort(t_birth_idx[late_idx])]

        for i in late_idx:
            tb = int(t_birth_idx[i])
            candidates = np.flatnonzero((t_birth_idx < tb) & (t_death_idx >= tb) & (np.arange(n) != i))
            pick_near = (rng_bd.random() > p_random) and (candidates.size > 0)

            if pick_near:
                if str(birth_near_pick).lower() == "uniform_over_active_tracks":
                    parent = int(rng_bd.choice(candidates))
                else:
                    parent = int(rng_bd.choice(candidates))
                parent_tid = float(track_ids[parent])
                parent_pos = x_true[parent, tb, :].copy()
                dir_vec = rng_bd.normal(size=3)
                dn = float(np.linalg.norm(dir_vec))
                if dn <= 1e-12:
                    dir_vec = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    dir_vec = dir_vec / dn
                dist = float(rng_bd.uniform(near_d_lo, near_d_hi))
                init_pos = parent_pos + dist * dir_vec
                birth_mode[i] = "near"
                parent_track_id[i] = parent_tid
                init_dist_to_parent[i] = dist
            else:
                init_pos = np.array(
                    [
                        rng_bd.uniform(x_rng[0], x_rng[1]),
                        rng_bd.uniform(y_rng[0], y_rng[1]),
                        rng_bd.uniform(z_rng[0], z_rng[1]),
                    ],
                    dtype=float,
                )
                birth_mode[i] = "random"

            x_true[i, tb, :] = init_pos
            for ti in range(tb + 1, t_count):
                x_true[i, ti, :] = x_true[i, ti - 1, :] + step_delta[i, ti - 1, :]

            # Keep pre-birth positions out-of-range-safe; these rows will be dropped.
            for ti in range(0, tb):
                x_true[i, ti, :] = init_pos

    # Optional hard-stationary subset for varying-dx birthdeath regime.
    is_stationary_track = np.zeros((n,), dtype=bool)
    if mode in {"varying_dx_birthdeath_stationary", "varying_dx_birthdeath_stationary_density"}:
        rng_sta = np.random.default_rng(int(seed) + 79013)
        frac = float(np.clip(stationary_fraction, 0.0, 1.0))
        apply_to = str(stationary_apply_to).lower()
        if apply_to == "initial_tracks_only":
            eligible = np.flatnonzero(t_birth_idx == 0)
        else:
            eligible = np.arange(n, dtype=int)
        if eligible.size > 0 and frac > 0.0:
            is_stationary_track[eligible] = rng_sta.random(eligible.size) < frac
        if str(stationary_mode).lower() == "hard_zero":
            for i in np.flatnonzero(is_stationary_track):
                tb = int(t_birth_idx[i])
                td = int(t_death_idx[i])
                anchor = x_true[i, tb, :].copy()
                x_true[i, tb : td + 1, :] = anchor[None, :]
                dx_schedule[i, tb : td + 1] = 0.0
                dx_source_all[i, tb : td + 1] = "stationary_hard_zero"
            cell_type[is_stationary_track] = "stationary"

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
    drift_tx = np.zeros((t_count,), dtype=float)
    drift_ty = np.zeros((t_count,), dtype=float)
    drift_tz = np.zeros((t_count,), dtype=float)
    drift_rot_deg = np.zeros((t_count,), dtype=float)

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
        drift_tx[ti] = float(global_drift[ti, 0])
        drift_ty[ti] = float(global_drift[ti, 1])
        drift_tz[ti] = float(global_drift[ti, 2])

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
                drift_tx[ti] = float(trans[ti, 0])
                drift_ty[ti] = float(trans[ti, 1])
                drift_tz[ti] = float(trans[ti, 2])
                drift_rot_deg[ti] = float(np.rad2deg(angles[ti]))

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
                drift_tx[ti] = float(drift_global_all[0, ti, 0])
                drift_ty[ti] = float(drift_global_all[0, ti, 1])
                drift_tz[ti] = float(drift_global_all[0, ti, 2])
                drift_rot_deg[ti] = 0.0

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
                drift_tx[ti] = float(drift_global_all[0, ti, 0])
                drift_ty[ti] = float(drift_global_all[0, ti, 1])
                drift_tz[ti] = float(drift_global_all[0, ti, 2])
                drift_rot_deg[ti] = float(np.rad2deg(ang[ti]))

    if mode == "varying_dx_birthdeath_with_drift":
        # Apply global rigid-like drift after intrinsic motion generation.
        rng_drift = np.random.default_rng(int(seed) + 69013)
        trans_step = _smooth_random_series(
            rng_drift,
            t=t_count,
            dim=3,
            scale=1.0,
            temporal_smoothness=temporal_smoothness,
        )
        trans = np.cumsum(trans_step, axis=0)
        trans *= np.array(global_drift_scale_xyz, dtype=float)[None, :] * 0.85

        if bool(drift_rotation_enable):
            rot_steps = _smooth_random_series(
                rng_drift,
                t=t_count,
                dim=1,
                scale=1.0,
                temporal_smoothness=temporal_smoothness,
            )[:, 0]
            rot_steps = np.cumsum(rot_steps)
            max_abs = float(np.max(np.abs(rot_steps))) if rot_steps.size else 1.0
            if max_abs <= 1e-9:
                max_abs = 1.0
            rot_deg = (rot_steps / max_abs) * float(max(0.0, drift_rotation_deg_scale))
            rot_rad = np.deg2rad(rot_deg)
        else:
            rot_deg = np.zeros((t_count,), dtype=float)
            rot_rad = np.zeros((t_count,), dtype=float)

        ref_center = np.mean(x_true[:, 0, :], axis=0)
        for ti in range(t_count):
            p_t = x_true[:, ti, :]
            p_rot = _rotate_about_z(p_t, ref_center, float(rot_rad[ti]))
            drift_local_all[:, ti, :] = p_rot - p_t
            drift_global_all[:, ti, :] = trans[ti][None, :]
            drift_tx[ti] = float(trans[ti, 0])
            drift_ty[ti] = float(trans[ti, 1])
            drift_tz[ti] = float(trans[ti, 2])
            drift_rot_deg[ti] = float(rot_deg[ti])

    if mode in {
        "nodrift_dx_sweep",
        "nodrift_varying_dx",
        "nodrift_dx_varying_birthdeath",
        "varying_dx_birthdeath_stationary",
        "varying_dx_birthdeath_stationary_density",
    }:
        drift_global_all = np.zeros_like(x_true)
        drift_local_all = np.zeros_like(x_true)
        drift_tx[:] = 0.0
        drift_ty[:] = 0.0
        drift_tz[:] = 0.0
        drift_rot_deg[:] = 0.0

    drift_total = drift_global_all + drift_local_all
    x_obs = x_true + drift_total + rng.normal(scale=float(measurement_noise_um), size=x_true.shape)

    rows_obs: list[dict[str, Any]] = []
    rows_gt: list[dict[str, Any]] = []
    rows_drift: list[dict[str, Any]] = []

    row_id = 1
    for i, tid in enumerate(track_ids):
        tb_i = int(t_birth_idx[i])
        td_i = int(t_death_idx[i])
        for ti, t in enumerate(times):
            if ti < tb_i or ti > td_i:
                continue
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
                    "t_birth_time": float(times[tb_i]),
                    "t_death_time": float(times[td_i]),
                    "birth_mode": str(birth_mode[i]),
                    "parent_track_id": float(parent_track_id[i]) if np.isfinite(parent_track_id[i]) else np.nan,
                    "init_dist_to_parent": float(init_dist_to_parent[i]) if np.isfinite(init_dist_to_parent[i]) else np.nan,
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
                    "t_birth_time": float(times[tb_i]),
                    "t_death_time": float(times[td_i]),
                    "birth_mode": str(birth_mode[i]),
                    "parent_track_id": float(parent_track_id[i]) if np.isfinite(parent_track_id[i]) else np.nan,
                    "init_dist_to_parent": float(init_dist_to_parent[i]) if np.isfinite(init_dist_to_parent[i]) else np.nan,
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
                    "drift_tx": float(drift_tx[ti]),
                    "drift_ty": float(drift_ty[ti]),
                    "drift_tz": float(drift_tz[ti]),
                    "drift_trans_norm": float(np.sqrt(drift_tx[ti] ** 2 + drift_ty[ti] ** 2 + drift_tz[ti] ** 2)),
                    "drift_rot_deg": float(drift_rot_deg[ti]),
                    "dx_scale_um": float(dx_schedule[i, ti]),
                    "dx_scale_source": str(dx_source_all[i, ti]),
                    "CellType": str(cell_type[i]),
                    "t_birth_time": float(times[tb_i]),
                    "t_death_time": float(times[td_i]),
                    "birth_mode": str(birth_mode[i]),
                    "parent_track_id": float(parent_track_id[i]) if np.isfinite(parent_track_id[i]) else np.nan,
                    "init_dist_to_parent": float(init_dist_to_parent[i]) if np.isfinite(init_dist_to_parent[i]) else np.nan,
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
    dx_mode: str = "fixed",
    dx_min_um: float = 0.2,
    dx_max_um: float = 4.0,
    dx_varying_distribution: str = "uniform",
    birthdeath_enable: bool = False,
    p_birth_late: float = 0.35,
    p_death_early: float = 0.35,
    min_life_frames: int = 5,
    p_birth_random: float = 0.5,
    birth_near_dist_min_um: float = 0.2,
    birth_near_dist_max_um: float = 4.0,
    birth_near_pick: str = "uniform_over_active_tracks",
    stationary_mode: str = "hard_zero",
    stationary_apply_to: str = "all_tracks",
    drift_rotation_enable: bool = True,
    drift_rotation_deg_scale: float = 2.0,
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((80.0, 620.0), (80.0, 620.0), (20.0, 320.0)),
) -> dict[str, Any]:
    """Create a synthetic IMARIS folder plus ground-truth sidecar files."""
    out_dir = Path(out_dir)
    gt_dir = out_dir / "_groundtruth"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    birthdeath_track_rows: list[pd.DataFrame] = []
    active_per_time_rows: list[pd.DataFrame] = []
    drift_schedule_rows: list[pd.DataFrame] = []
    for i in range(int(n_files)):
        file_seed = int(seed + i * 37)
        df_orig, df_gt, df_drift = generate_controlled_drift_dataset(
            n_tracks=n_tracks,
            n_times=n_times,
            stationary_fraction=stationary_fraction,
            migrating_fraction=migrating_fraction,
            seed=file_seed,
            bounds_xyz=bounds_xyz,
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
            dx_mode=dx_mode,
            dx_min_um=dx_min_um,
            dx_max_um=dx_max_um,
            dx_varying_distribution=dx_varying_distribution,
            birthdeath_enable=birthdeath_enable,
            p_birth_late=p_birth_late,
            p_death_early=p_death_early,
            min_life_frames=min_life_frames,
            p_birth_random=p_birth_random,
            birth_near_dist_min_um=birth_near_dist_min_um,
            birth_near_dist_max_um=birth_near_dist_max_um,
            birth_near_pick=birth_near_pick,
            stationary_mode=stationary_mode,
            stationary_apply_to=stationary_apply_to,
            drift_rotation_enable=drift_rotation_enable,
            drift_rotation_deg_scale=drift_rotation_deg_scale,
        )

        base = f"synthetic_drift_{i + 1:03d}"
        xlsx_path = out_dir / f"{base}.csv"
        write_synthetic_imaris_table(df_orig, xlsx_path)

        df_gt.to_csv(gt_dir / f"{base}_groundtruth.csv", index=False)
        df_drift.to_csv(gt_dir / f"{base}_drift_field_groundtruth.csv", index=False)
        if "dx_scale_um" in df_drift.columns:
            cols = ["Time", "TrackID", "dx_scale_um"]
            if "dx_scale_source" in df_drift.columns:
                cols.append("dx_scale_source")
            dx_schedule = df_drift[cols].copy()
            dx_schedule.to_csv(gt_dir / f"{base}_dx_schedule.csv", index=False)

            dx_time = (
                dx_schedule.groupby("Time", as_index=False)["dx_scale_um"]
                .agg(
                    dx_q10=lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.10)),
                    dx_q50=lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.50)),
                    dx_q90=lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.90)),
                    dx_mean=lambda x: float(np.mean(np.asarray(x, dtype=float))),
                )
                .sort_values("Time")
                .reset_index(drop=True)
            )
            dx_time.to_csv(gt_dir / f"{base}_dx_schedule_summary_time.csv", index=False)

            dx_track = (
                dx_schedule.groupby("TrackID", as_index=False)["dx_scale_um"]
                .agg(
                    dx_median=lambda x: float(np.median(np.asarray(x, dtype=float))),
                    dx_std=lambda x: float(np.std(np.asarray(x, dtype=float))),
                )
                .sort_values("TrackID")
                .reset_index(drop=True)
            )
            dx_track.to_csv(gt_dir / f"{base}_dx_schedule_summary_track.csv", index=False)

        if {"drift_tx", "drift_ty", "drift_tz", "drift_trans_norm", "drift_rot_deg"}.issubset(df_drift.columns):
            drift_tab = (
                df_drift.groupby("Time", as_index=False)
                .agg(
                    drift_tx=("drift_tx", "median"),
                    drift_ty=("drift_ty", "median"),
                    drift_tz=("drift_tz", "median"),
                    drift_trans_norm=("drift_trans_norm", "median"),
                    drift_rot_deg=("drift_rot_deg", "median"),
                )
                .sort_values("Time")
                .reset_index(drop=True)
            )
            drift_tab.to_csv(gt_dir / f"{base}_drift_schedule.csv", index=False)
            drift_tab["File"] = str(base)
            drift_schedule_rows.append(
                drift_tab[
                    [
                        "File",
                        "Time",
                        "drift_tx",
                        "drift_ty",
                        "drift_tz",
                        "drift_trans_norm",
                        "drift_rot_deg",
                    ]
                ]
            )

        if {"t_birth_time", "t_death_time", "birth_mode"}.issubset(df_gt.columns):
            track_life = (
                df_gt[
                    [
                        "TrackID",
                        "t_birth_time",
                        "t_death_time",
                        "birth_mode",
                        "parent_track_id",
                        "init_dist_to_parent",
                    ]
                ]
                .drop_duplicates("TrackID")
                .copy()
            )
            track_life["File"] = str(base)
            track_life = track_life[
                [
                    "File",
                    "TrackID",
                    "t_birth_time",
                    "t_death_time",
                    "birth_mode",
                    "parent_track_id",
                    "init_dist_to_parent",
                ]
            ]
            birthdeath_track_rows.append(track_life)
            track_life.to_csv(gt_dir / f"{base}_track_birthdeath_summary.csv", index=False)

            active_tab = (
                df_gt.groupby("Time", as_index=False)["TrackID"]
                .nunique()
                .rename(columns={"TrackID": "active_tracks"})
                .sort_values("Time")
                .reset_index(drop=True)
            )
            active_tab["File"] = str(base)
            active_per_time_rows.append(active_tab[["File", "Time", "active_tracks"]])
            active_tab.to_csv(gt_dir / f"{base}_active_tracks_per_time.csv", index=False)

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
            "dx_mode": str(dx_mode),
            "dx_min_um": float(dx_min_um),
            "dx_max_um": float(dx_max_um),
            "dx_varying_distribution": str(dx_varying_distribution),
            "birthdeath_enable": bool(birthdeath_enable),
            "p_birth_late": float(p_birth_late),
            "p_death_early": float(p_death_early),
            "min_life_frames": int(min_life_frames),
            "p_birth_random": float(p_birth_random),
            "birth_near_dist_min_um": float(birth_near_dist_min_um),
            "birth_near_dist_max_um": float(birth_near_dist_max_um),
            "birth_near_pick": str(birth_near_pick),
            "stationary_mode": str(stationary_mode),
            "stationary_apply_to": str(stationary_apply_to),
            "drift_rotation_enable": bool(drift_rotation_enable),
            "drift_rotation_deg_scale": float(drift_rotation_deg_scale),
        }
        pd.DataFrame([meta]).to_csv(gt_dir / f"{base}_meta.csv", index=False)
        written_files.append(xlsx_path)

    if birthdeath_track_rows:
        track_summary = pd.concat(birthdeath_track_rows, ignore_index=True)
        track_summary.to_csv(gt_dir / "track_birthdeath_summary.csv", index=False)
    if active_per_time_rows:
        active_summary = pd.concat(active_per_time_rows, ignore_index=True)
        active_summary.to_csv(gt_dir / "active_tracks_per_time.csv", index=False)
    if drift_schedule_rows:
        drift_summary = pd.concat(drift_schedule_rows, ignore_index=True)
        drift_summary.to_csv(gt_dir / "drift_schedule.csv", index=False)

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
    """
    Save groundtruth-based method comparison metrics and plots.
    
    DEBUG: 
    
    base_out_prefix = out_prefix
    groundtruth_df = gt
    corrected_by_method = corr_by_method
    drift_field_groundtruth=drift_gt
    """
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

    out_hist = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_per_track_hist.pdf")
    out_time = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_over_time.pdf")
    out_time_stationary = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_stationary_over_time.pdf")
    out_time_migrating = base_out_prefix.with_name(base_out_prefix.name + "_gt_rms_migrating_over_time.pdf")
    out_drift_bias = base_out_prefix.with_name(base_out_prefix.name + "_gt_drift_bias_over_time.pdf")

    #fig = plt.figure(figsize=(9, 5))
    cm2inch = 1/2.54
    fig = plt.figure(figsize=(6.0*cm2inch, 4*cm2inch))
    for method, tab in per_method_track.items():
        arr = tab["rms_error"].to_numpy(float)
        if arr.size:
            plt.hist(arr, bins=40, alpha=0.4, label=method)
    plt.xlabel("RMS error per track (um)")
    plt.ylabel("count")
    plt.title("Corrected vs Groundtruth: per-track RMS error")
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=300, transparent=True)
    plt.close(fig)

    #fig = plt.figure(figsize=(9, 5))
    cm2inch = 1/2.54
    fig = plt.figure(figsize=(6.0*cm2inch, 4*cm2inch))
    for method, tab in per_method_time.items():
        if tab.shape[0]:
            plt.plot(tab["Time"].to_numpy(float), tab["rms_error"].to_numpy(float), 
                     marker="o", label=method, markersize=2, lw=0.7)
    plt.xlabel("Time")
    plt.ylabel("RMS error (um)")
    plt.title("Corrected vs Groundtruth: RMS error over time")
    plt.ylim(bottom=0.0, top=11.0)
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(ncol=2, loc="upper left", fontsize=4)
    if tab["Time"].max() <= 10:
        plt.xticks(np.arange(0, tab["Time"].max() + 1, step=2))
    elif tab["Time"].max() <= 20:
        plt.xticks(np.arange(0, tab["Time"].max() + 1, step=5))
    plt.tight_layout()
    plt.savefig(out_time, dpi=300, transparent=True)
    plt.close(fig)

    if per_method_time_stationary:
        #fig = plt.figure(figsize=(9, 5))
        cm2inch = 1/2.54
        fig = plt.figure(figsize=(6.0*cm2inch, 4*cm2inch))
        for method, tab in per_method_time_stationary.items():
            if tab.shape[0]:
                plt.plot(tab["Time"].to_numpy(float), tab["rms_error"].to_numpy(float), marker="o", label=method)
        plt.xlabel("Time")
        plt.ylabel("RMS error (um)")
        plt.title("Corrected vs Groundtruth: RMS_stationary_over_time")
        plt.ylim(bottom=0.0, top=11.0)
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(ncol=2, loc="upper left")
        if tab["Time"].max() <= 10:
            plt.xticks(np.arange(0, tab["Time"].max() + 1, step=2))
        elif tab["Time"].max() <= 20:
            plt.xticks(np.arange(0, tab["Time"].max() + 1, step=5))
        plt.tight_layout()
        plt.savefig(out_time_stationary, dpi=300, transparent=True)
        plt.close(fig)

    if per_method_time_migrating:
        #fig = plt.figure(figsize=(9, 5))
        cm2inch = 1/2.54
        fig = plt.figure(figsize=(6.0*cm2inch, 4*cm2inch))
        for method, tab in per_method_time_migrating.items():
            if tab.shape[0]:
                plt.plot(tab["Time"].to_numpy(float), tab["rms_error"].to_numpy(float), marker="o", label=method)
        plt.xlabel("Time")
        plt.ylabel("RMS error (um)")
        plt.title("Corrected vs Groundtruth: RMS_migrating_over_time")
        plt.ylim(bottom=0.0, top=11.0)
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(ncol=2, loc="upper left")
        if tab["Time"].max() <= 10:
            plt.xticks(np.arange(0, tab["Time"].max() + 1, step=2))
        elif tab["Time"].max() <= 20:
            plt.xticks(np.arange(0, tab["Time"].max() + 1, step=5))
        plt.tight_layout()
        plt.savefig(out_time_migrating, dpi=300, transparent=True)
        plt.close(fig)

    if per_method_drift_bias:
        #fig = plt.figure(figsize=(9, 5))
        cm2inch = 1/2.54
        fig = plt.figure(figsize=(6.0*cm2inch, 4*cm2inch))
        for method, tab in per_method_drift_bias.items():
            if tab.shape[0]:
                plt.plot(tab["Time"].to_numpy(float), tab["drift_bias_norm"].to_numpy(float), marker="o", label=method)
        plt.xlabel("Time")
        plt.ylabel("||median(pred_drift)-median(true_drift)|| (um)")
        plt.title("Groundtruth drift bias over time")
        plt.ylim(bottom=0.0, top=11.0)
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(ncol=2, loc="upper left")
        if groundtruth_df["Time"].max() <= 10:
            plt.xticks(np.arange(0, groundtruth_df["Time"].max() + 1, step=2))
        elif groundtruth_df["Time"].max() <= 20:
            plt.xticks(np.arange(0, groundtruth_df["Time"].max() + 1, step=5))
        plt.tight_layout()
        plt.savefig(out_drift_bias, dpi=300, transparent=True)
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
# %% END