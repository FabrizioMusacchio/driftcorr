"""Joint space-time RBF drift model in 4D input space.

This method fits a smooth drift function over both spatial coordinates and time.
The implementation includes:
- configurable spatial input mode (`row_position`, `track_anchor`, `ref_position`),
- normalized inputs (space robustly scaled, time in [0,1]),
- kernel bias augmentation (`K_aug = [K, 1]`),
- optional two-stage decomposition (global translation + residual RBF),
- iterative refinement with sequential correction stages.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks


@dataclass
class SpaceTimeStageModel:
    """One refinement stage of the space-time RBF model."""

    centers_u: np.ndarray
    weights_aug: np.ndarray
    ell_space: float
    ell_time: float
    lam: float
    reference_time: float
    robust_keep_fraction: float
    robust_mode: str
    space_input_mode: str
    two_stage: bool
    rbf_linear_terms: bool
    spatial_center: np.ndarray
    spatial_scale: float
    time_min: float
    time_max: float
    global_time_values: np.ndarray
    global_drift_values: np.ndarray
    cond_proxy: float
    per_time_table: pd.DataFrame


@dataclass
class SpaceTimeRBFModel:
    """Full model with one or multiple refinement stages."""

    stages: list[SpaceTimeStageModel]
    reference_time: float
    space_input_mode: str
    n_refine: int


def _kernel_spacetime(
    u: np.ndarray,
    c: np.ndarray,
    ell_space: float,
    ell_time: float,
) -> np.ndarray:
    diff_space = u[:, None, :3] - c[None, :, :3]
    d2_space = np.sum(diff_space * diff_space, axis=2)
    dt2 = (u[:, None, 3] - c[None, :, 3]) ** 2
    es2 = max(float(ell_space) ** 2, 1e-12)
    et2 = max(float(ell_time) ** 2, 1e-12)
    return np.exp(-(d2_space / (2.0 * es2) + dt2 / (2.0 * et2)))


def _augment_kernel(k: np.ndarray) -> np.ndarray:
    return np.hstack([k, np.ones((k.shape[0], 1), dtype=float)])


def _augment_kernel_with_linear(k: np.ndarray, u: np.ndarray, rbf_linear_terms: bool = False) -> np.ndarray:
    n = int(k.shape[0])
    cols: list[np.ndarray] = [k]
    if bool(rbf_linear_terms):
        cols.extend([u[:, [0]], u[:, [1]], u[:, [2]], u[:, [3]]])
    cols.append(np.ones((n, 1), dtype=float))
    return np.hstack(cols)


def _choose_centers(u: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if u.shape[0] <= m:
        return u.copy()
    idx = rng.choice(u.shape[0], size=m, replace=False)
    return u[idx].copy()


def _weighted_ridge_aug(k_aug: np.ndarray, d: np.ndarray, w: np.ndarray, lam: float, lam_bias: float = 0.0) -> np.ndarray:
    sw = np.sqrt(np.clip(w, 0.0, np.inf))[:, None]
    kw = k_aug * sw
    dw = d * sw
    m_aug = k_aug.shape[1]
    a = kw.T @ kw
    reg = np.full((m_aug,), float(lam), dtype=float)
    reg[-1] = float(lam_bias)
    a.flat[:: m_aug + 1] += reg
    b = kw.T @ dw
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        a2 = a.copy()
        a2.flat[:: a2.shape[0] + 1] += 1e-8
        return np.linalg.solve(a2, b)


def _condition_proxy(k_aug: np.ndarray, lam: float, lam_bias: float = 0.0) -> float:
    m_aug = int(k_aug.shape[1])
    a = k_aug.T @ k_aug
    reg = np.full((m_aug,), float(lam), dtype=float)
    reg[-1] = float(lam_bias)
    a.flat[:: m_aug + 1] += reg
    try:
        return float(np.linalg.cond(a))
    except np.linalg.LinAlgError:
        return float("inf")


def _robust_spatial_scale(x: np.ndarray) -> float:
    c = np.nanmedian(x, axis=0)
    r = np.linalg.norm(x - c[None, :], axis=1)
    s = float(np.nanmedian(r))
    if not np.isfinite(s) or s <= 1e-9:
        mins = np.nanmin(x, axis=0)
        maxs = np.nanmax(x, axis=0)
        s = float(np.linalg.norm(maxs - mins) / 3.0)
    if not np.isfinite(s) or s <= 1e-9:
        s = 1.0
    return s


def _normalize_inputs(
    spatial: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    c = np.nanmedian(spatial, axis=0)
    s = _robust_spatial_scale(spatial)

    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    tr = max(tmax - tmin, 1e-9)

    xn = (spatial - c[None, :]) / s
    tn = (t - tmin) / tr
    return xn, tn, c, float(s), tmin, tmax


def _spatial_inputs_for_rows(df: pd.DataFrame, reference_time: float, mode: str) -> np.ndarray:
    d = df.copy()
    mode = str(mode)

    if mode == "row_position":
        return d[["Position X", "Position Y", "Position Z"]].to_numpy(float)

    means = d.groupby("TrackID", as_index=False)[["Position X", "Position Y", "Position Z"]].mean()
    means = means.rename(columns={"Position X": "X_mean", "Position Y": "Y_mean", "Position Z": "Z_mean"})

    if mode == "track_anchor":
        dm = d[["TrackID"]].merge(means, on="TrackID", how="left")
        return dm[["X_mean", "Y_mean", "Z_mean"]].to_numpy(float)

    if mode == "ref_position":
        ref = d[d["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
        ref = ref.rename(columns={"Position X": "X_ref", "Position Y": "Y_ref", "Position Z": "Z_ref"})
        dm = d[["TrackID"]].merge(ref, on="TrackID", how="left")
        miss = ~np.all(np.isfinite(dm[["X_ref", "Y_ref", "Z_ref"]].to_numpy(float)), axis=1)
        if np.any(miss):
            mm = d[["TrackID"]].merge(means, on="TrackID", how="left")
            arr = dm[["X_ref", "Y_ref", "Z_ref"]].to_numpy(float)
            arr[miss] = mm[["X_mean", "Y_mean", "Z_mean"]].to_numpy(float)[miss]
            return arr
        return dm[["X_ref", "Y_ref", "Z_ref"]].to_numpy(float)

    raise ValueError(f"Unknown space_input_mode: {mode}")


def _build_training_samples(
    df: pd.DataFrame,
    reference_time: float,
    fit_track_ids: np.ndarray | None,
    space_input_mode: str,
    two_stage: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized u, drift target, raw_times, raw_space, raw_total_drift."""
    d = ensure_numeric_tracks(df)
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        d = d[d["TrackID"].astype(float).isin(ids)].copy()

    ref = d[d["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    ref = ref.rename(columns={"Position X": "X_ref", "Position Y": "Y_ref", "Position Z": "Z_ref"})
    mdf = d.merge(ref, on="TrackID", how="inner")

    if mdf.shape[0] == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    spatial_all = _spatial_inputs_for_rows(d, reference_time=reference_time, mode=space_input_mode)
    d_with_spatial = d.copy()
    d_with_spatial["_sx"] = spatial_all[:, 0]
    d_with_spatial["_sy"] = spatial_all[:, 1]
    d_with_spatial["_sz"] = spatial_all[:, 2]

    mdf = mdf[["TrackID", "Time", "Position X", "Position Y", "Position Z", "X_ref", "Y_ref", "Z_ref"]].copy()
    sp = d_with_spatial[["TrackID", "Time", "_sx", "_sy", "_sz"]].copy()
    mdf = mdf.merge(sp, on=["TrackID", "Time"], how="left")

    t = mdf["Time"].to_numpy(float)
    spatial = mdf[["_sx", "_sy", "_sz"]].to_numpy(float)

    d_total = np.column_stack(
        [
            mdf["Position X"].to_numpy(float) - mdf["X_ref"].to_numpy(float),
            mdf["Position Y"].to_numpy(float) - mdf["Y_ref"].to_numpy(float),
            mdf["Position Z"].to_numpy(float) - mdf["Z_ref"].to_numpy(float),
        ]
    )

    if two_stage:
        gtab = pd.DataFrame({"Time": t, "dx": d_total[:, 0], "dy": d_total[:, 1], "dz": d_total[:, 2]})
        gtab = gtab.groupby("Time", as_index=False).median().sort_values("Time")
        gmap = {float(row.Time): np.array([row.dx, row.dy, row.dz], dtype=float) for row in gtab.itertuples(index=False)}
        g = np.vstack([gmap[float(tt)] for tt in t])
        d_target = d_total - g
    else:
        d_target = d_total

    xn, tn, _, _, _, _ = _normalize_inputs(spatial, t)
    u = np.column_stack([xn, tn])
    return u, d_target, t, spatial, d_total


def _build_fit_table(
    df: pd.DataFrame,
    reference_time: float,
    fit_track_ids: np.ndarray | None,
    space_input_mode: str,
) -> pd.DataFrame:
    d = ensure_numeric_tracks(df)
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        d = d[d["TrackID"].astype(float).isin(ids)].copy()

    ref = d[d["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    ref = ref.rename(columns={"Position X": "X_ref", "Position Y": "Y_ref", "Position Z": "Z_ref"})
    mdf = d.merge(ref, on="TrackID", how="inner")
    if mdf.shape[0] == 0:
        return mdf

    sp_all = _spatial_inputs_for_rows(d, reference_time=reference_time, mode=space_input_mode)
    d_sp = d.copy()
    d_sp["_sx"] = sp_all[:, 0]
    d_sp["_sy"] = sp_all[:, 1]
    d_sp["_sz"] = sp_all[:, 2]
    sp = d_sp[["TrackID", "Time", "_sx", "_sy", "_sz"]].copy()

    mdf = mdf[["TrackID", "Time", "Position X", "Position Y", "Position Z", "X_ref", "Y_ref", "Z_ref"]].copy()
    mdf = mdf.merge(sp, on=["TrackID", "Time"], how="left")
    return mdf


def _fit_stage_weights(
    k_aug: np.ndarray,
    d_target: np.ndarray,
    track_ids: np.ndarray,
    lam: float,
    robust_keep_fraction: float,
    robust_iters: int,
    robust_mode: str,
    track_trim_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(k_aug.shape[0])
    keep_fraction = float(np.clip(robust_keep_fraction, 0.1, 0.99))
    mode = str(robust_mode)
    if mode not in {"none", "sample_trim", "track_trim"}:
        raise ValueError(f"Unknown robust_mode: {mode}")

    if mode == "none":
        mask = np.ones((n,), dtype=bool)
        w_aug = _weighted_ridge_aug(k_aug, d_target, mask.astype(float), lam=lam, lam_bias=0.0)
        return w_aug, mask

    if mode == "sample_trim":
        k_keep = max(20, int(np.floor(keep_fraction * n)))
        wv = np.ones((n,), dtype=float)
        w_aug = _weighted_ridge_aug(k_aug, d_target, wv, lam=lam, lam_bias=0.0)
        for _ in range(max(1, int(robust_iters))):
            pred = k_aug @ w_aug
            res = np.linalg.norm(pred - d_target, axis=1)
            idx = np.argsort(res)[:k_keep]
            wv[:] = 0.0
            wv[idx] = 1.0
            w_aug = _weighted_ridge_aug(k_aug, d_target, wv, lam=lam, lam_bias=0.0)
        return w_aug, wv > 0

    # track_trim
    tids = np.asarray(track_ids, dtype=float)
    uniq = np.array(sorted(np.unique(tids)), dtype=float)
    if uniq.size == 0:
        mask = np.ones((n,), dtype=bool)
        w_aug = _weighted_ridge_aug(k_aug, d_target, mask.astype(float), lam=lam, lam_bias=0.0)
        return w_aug, mask

    keep_tracks = max(2, int(np.floor(keep_fraction * uniq.size)))
    selected = uniq.copy()
    prev: set[float] | None = None
    n_iters = max(1, int(track_trim_iters))

    for _ in range(n_iters):
        mask_sel = np.isin(tids, selected)
        if int(np.sum(mask_sel)) < 10:
            break
        w_aug = _weighted_ridge_aug(k_aug[mask_sel], d_target[mask_sel], np.ones((int(np.sum(mask_sel)),), dtype=float), lam=lam, lam_bias=0.0)
        pred_all = k_aug @ w_aug
        res_all = np.linalg.norm(pred_all - d_target, axis=1)

        scores: list[tuple[float, float]] = []
        for tid in uniq:
            mt = tids == float(tid)
            if not np.any(mt):
                continue
            scores.append((float(tid), float(np.median(res_all[mt]))))
        scores.sort(key=lambda x: x[1])
        selected = np.array([tid for tid, _ in scores[:keep_tracks]], dtype=float)
        cur = set(selected.tolist())
        if prev is not None and cur == prev:
            break
        prev = cur

    mask_final = np.isin(tids, selected)
    if int(np.sum(mask_final)) < 10:
        mask_final = np.ones((n,), dtype=bool)
    w_aug = _weighted_ridge_aug(k_aug[mask_final], d_target[mask_final], np.ones((int(np.sum(mask_final)),), dtype=float), lam=lam, lam_bias=0.0)
    return w_aug, mask_final


def _fit_stage(
    df: pd.DataFrame,
    reference_time: float,
    m_centers: int,
    ell_space: float,
    ell_time: float,
    lam: float,
    robust_keep_fraction: float,
    robust_iters: int,
    seed: int,
    fit_track_ids: np.ndarray | None,
    space_input_mode: str,
    two_stage: bool,
    robust_mode: str = "sample_trim",
    track_trim_iters: int = 4,
    global_time_values: np.ndarray | None = None,
    global_drift_values: np.ndarray | None = None,
    rbf_linear_terms: bool = False,
) -> tuple[SpaceTimeStageModel, dict[str, float | int | str | bool]]:
    mdf = _build_fit_table(df, reference_time=reference_time, fit_track_ids=fit_track_ids, space_input_mode=space_input_mode)
    if mdf.shape[0] < max(20, int(m_centers)):
        raise ValueError("Too few training samples for space-time RBF fit.")

    t_raw = mdf["Time"].to_numpy(float)
    spatial_raw = mdf[["_sx", "_sy", "_sz"]].to_numpy(float)
    track_ids = mdf["TrackID"].to_numpy(float)

    d_total = np.column_stack(
        [
            mdf["Position X"].to_numpy(float) - mdf["X_ref"].to_numpy(float),
            mdf["Position Y"].to_numpy(float) - mdf["Y_ref"].to_numpy(float),
            mdf["Position Z"].to_numpy(float) - mdf["Z_ref"].to_numpy(float),
        ]
    )

    if bool(two_stage) and global_time_values is not None and global_drift_values is not None and global_time_values.size:
        g_time = np.asarray(global_time_values, dtype=float)
        g_drift = np.asarray(global_drift_values, dtype=float)
        g_per_sample = _interp_global_drift(t_raw, g_time, g_drift)
        d_target = d_total - g_per_sample
    else:
        g_time = np.array(sorted(np.unique(t_raw)), dtype=float)
        g_drift = np.zeros((g_time.size, 3), dtype=float)
        d_target = d_total

    xn, tn, center_spatial, scale_spatial, tmin, tmax = _normalize_inputs(spatial_raw, t_raw)
    u = np.column_stack([xn, tn])

    centers = _choose_centers(u, m=int(m_centers), seed=seed)
    k = _kernel_spacetime(u, centers, ell_space=float(ell_space), ell_time=float(ell_time))
    k_aug = _augment_kernel_with_linear(k, u=u, rbf_linear_terms=rbf_linear_terms)

    keep_fraction = float(np.clip(robust_keep_fraction, 0.1, 0.99))
    weights_aug, inlier_mask = _fit_stage_weights(
        k_aug=k_aug,
        d_target=d_target,
        track_ids=track_ids,
        lam=lam,
        robust_keep_fraction=keep_fraction,
        robust_iters=robust_iters,
        robust_mode=robust_mode,
        track_trim_iters=track_trim_iters,
    )

    pred_res = k_aug @ weights_aug
    pred_total = pred_res + _interp_global_drift(t_raw, g_time, g_drift)
    res_total = np.linalg.norm(pred_total - d_total, axis=1)

    cond_proxy = _condition_proxy(k_aug, lam=lam, lam_bias=0.0)
    ill_conditioned = bool(np.isfinite(cond_proxy) and cond_proxy > 1e12)
    weight_max_abs = float(np.nanmax(np.abs(weights_aug))) if weights_aug.size else float("nan")
    weights_all_finite = bool(np.all(np.isfinite(weights_aug)))

    # Per-time diagnostics (coverage + predicted drift magnitude).
    rows_t: list[dict[str, float]] = []
    for tt in np.array(sorted(np.unique(t_raw)), dtype=float):
        mt = t_raw == float(tt)
        if not np.any(mt):
            continue
        ut = u[mt]
        st = spatial_raw[mt]

        d2 = np.sum((ut[:, None, :3] - centers[None, :, :3]) ** 2, axis=2)
        nn = np.sqrt(np.min(d2, axis=1)) * float(scale_spatial)

        mag = np.linalg.norm(pred_total[mt], axis=1)
        rows_t.append(
            {
                "Time": float(tt),
                "center_nn_median": float(np.median(nn)) if nn.size else float("nan"),
                "pred_drift_median": float(np.median(mag)) if mag.size else float("nan"),
                "res_median": float(np.median(res_total[mt])) if np.any(mt) else float("nan"),
            }
        )

    per_time_table = pd.DataFrame(rows_t).sort_values("Time").reset_index(drop=True)

    stage = SpaceTimeStageModel(
        centers_u=centers,
        weights_aug=weights_aug,
        ell_space=float(ell_space),
        ell_time=float(ell_time),
        lam=float(lam),
        reference_time=float(reference_time),
        robust_keep_fraction=float(keep_fraction),
        robust_mode=str(robust_mode),
        space_input_mode=str(space_input_mode),
        two_stage=bool(two_stage and global_time_values is not None and global_drift_values is not None and np.asarray(global_time_values).size > 0),
        rbf_linear_terms=bool(rbf_linear_terms),
        spatial_center=center_spatial,
        spatial_scale=float(scale_spatial),
        time_min=float(tmin),
        time_max=float(tmax),
        global_time_values=g_time,
        global_drift_values=g_drift,
        cond_proxy=cond_proxy,
        per_time_table=per_time_table,
    )

    info: dict[str, float | int | str | bool] = {
        "n_samples": int(u.shape[0]),
        "n_centers": int(centers.shape[0]),
        "res_median": float(np.median(res_total)),
        "res_p90": float(np.percentile(res_total, 90)),
        "ell_space": float(ell_space),
        "ell_time": float(ell_time),
        "lam": float(lam),
        "robust_keep_fraction": float(keep_fraction),
        "robust_iters": int(robust_iters),
        "robust_mode": str(robust_mode),
        "track_trim_iters": int(track_trim_iters),
        "space_input_mode": str(space_input_mode),
        "two_stage": bool(stage.two_stage),
        "rbf_linear_terms": bool(rbf_linear_terms),
        "cond_proxy": float(cond_proxy),
        "ill_conditioned": bool(ill_conditioned),
        "weight_max_abs": float(weight_max_abs),
        "weights_all_finite": bool(weights_all_finite),
        "inlier_fraction": float(np.mean(inlier_mask)) if inlier_mask.size else float("nan"),
    }
    return stage, info


def _interp_global_drift(times: np.ndarray, g_time: np.ndarray, g_drift: np.ndarray) -> np.ndarray:
    if g_time.size == 0:
        return np.zeros((times.size, 3), dtype=float)
    out = np.zeros((times.size, 3), dtype=float)
    for ax in range(3):
        out[:, ax] = np.interp(times, g_time, g_drift[:, ax])
    return out


def _predict_stage_drift(stage: SpaceTimeStageModel, df_current: pd.DataFrame) -> np.ndarray:
    d = ensure_numeric_tracks(df_current)

    spatial = _spatial_inputs_for_rows(d, reference_time=stage.reference_time, mode=stage.space_input_mode)
    t_raw = d["Time"].to_numpy(float)

    tr = max(stage.time_max - stage.time_min, 1e-9)
    xn = (spatial - stage.spatial_center[None, :]) / max(stage.spatial_scale, 1e-9)
    tn = (t_raw - stage.time_min) / tr
    u = np.column_stack([xn, tn])

    k = _kernel_spacetime(u, stage.centers_u, stage.ell_space, stage.ell_time)
    pred_res = _augment_kernel_with_linear(k, u=u, rbf_linear_terms=stage.rbf_linear_terms) @ stage.weights_aug

    if stage.two_stage:
        g = _interp_global_drift(t_raw, stage.global_time_values, stage.global_drift_values)
    else:
        g = np.zeros_like(pred_res)

    return pred_res + g


def fit_spacetime_rbf_model(
    df: pd.DataFrame,
    reference_time: float | None = None,
    m_centers: int = 70,
    ell_space: float = 1.2,
    ell_time: float = 0.3,
    lam: float = 3e-4,
    robust_keep_fraction: float = 0.8,
    robust_iters: int = 4,
    seed: int = 0,
    fit_track_ids: np.ndarray | None = None,
    space_input_mode: str = "track_anchor",
    n_refine: int = 2,
    two_stage: bool = False,
    robust_mode: str = "sample_trim",
    track_trim_iters: int = 4,
    rbf_linear_terms: bool = False,
    weight_abort_threshold: float = 1e4,
) -> tuple[SpaceTimeRBFModel, dict[str, float | int | str | bool]]:
    """Fit space-time RBF drift model with optional refinement stages."""
    d0 = ensure_numeric_tracks(df)
    times = np.array(sorted(d0["Time"].unique()), dtype=float)
    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    stages: list[SpaceTimeStageModel] = []
    stage_infos: list[dict[str, float | int | str | bool]] = []

    current = d0.copy()
    n_ref = max(1, int(n_refine))
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        dfit = d0[d0["TrackID"].astype(float).isin(ids)].copy()
    else:
        dfit = d0.copy()

    if bool(two_stage):
        fit_tab0 = _build_fit_table(dfit, reference_time=reference_time, fit_track_ids=None, space_input_mode="row_position")
        if fit_tab0.shape[0] > 0:
            tt0 = fit_tab0["Time"].to_numpy(float)
            d0_total = np.column_stack(
                [
                    fit_tab0["Position X"].to_numpy(float) - fit_tab0["X_ref"].to_numpy(float),
                    fit_tab0["Position Y"].to_numpy(float) - fit_tab0["Y_ref"].to_numpy(float),
                    fit_tab0["Position Z"].to_numpy(float) - fit_tab0["Z_ref"].to_numpy(float),
                ]
            )
            gtab = (
                pd.DataFrame({"Time": tt0, "dx": d0_total[:, 0], "dy": d0_total[:, 1], "dz": d0_total[:, 2]})
                .groupby("Time", as_index=False)
                .median()
                .sort_values("Time")
                .reset_index(drop=True)
            )
            global_time_values = gtab["Time"].to_numpy(float)
            global_drift_values = gtab[["dx", "dy", "dz"]].to_numpy(float)
        else:
            global_time_values = np.array([], dtype=float)
            global_drift_values = np.zeros((0, 3), dtype=float)
    else:
        global_time_values = np.array([], dtype=float)
        global_drift_values = np.zeros((0, 3), dtype=float)

    for i in range(n_ref):
        stage_two_stage = bool(two_stage and i == 0 and global_time_values.size > 0)
        stage, info = _fit_stage(
            df=current,
            reference_time=reference_time,
            m_centers=m_centers,
            ell_space=ell_space,
            ell_time=ell_time,
            lam=lam,
            robust_keep_fraction=robust_keep_fraction,
            robust_iters=robust_iters,
            seed=int(seed + i),
            fit_track_ids=fit_track_ids,
            space_input_mode=space_input_mode,
            two_stage=stage_two_stage,
            robust_mode=robust_mode,
            track_trim_iters=track_trim_iters,
            global_time_values=global_time_values if stage_two_stage else None,
            global_drift_values=global_drift_values if stage_two_stage else None,
            rbf_linear_terms=rbf_linear_terms,
        )
        info = dict(info)
        info["refine_iter"] = int(i + 1)

        wmax = float(info.get("weight_max_abs", np.nan))
        wf = bool(info.get("weights_all_finite", True))
        if (not wf) or (np.isfinite(wmax) and wmax > float(weight_abort_threshold)):
            print(
                f"[WARN] ST-RBF refinement aborted at stage {i+1}: "
                f"unstable weights (finite={wf}, max_abs={wmax:.3g})"
            )
            break

        stages.append(stage)
        stage_infos.append(info)

        if i < n_ref - 1:
            drift_pred = _predict_stage_drift(stage, current)
            xyz = current[["Position X", "Position Y", "Position Z"]].to_numpy(float)
            xyz_corr = xyz - drift_pred
            current.loc[:, "Position X"] = xyz_corr[:, 0]
            current.loc[:, "Position Y"] = xyz_corr[:, 1]
            current.loc[:, "Position Z"] = xyz_corr[:, 2]

    model = SpaceTimeRBFModel(
        stages=stages,
        reference_time=float(reference_time),
        space_input_mode=str(space_input_mode),
        n_refine=int(n_ref),
    )

    if stage_infos:
        last = stage_infos[-1]
    else:
        last = {
            "n_samples": 0,
            "n_centers": 0,
            "res_median": float("nan"),
            "res_p90": float("nan"),
            "ell_space": float(ell_space),
            "ell_time": float(ell_time),
            "lam": float(lam),
            "robust_keep_fraction": float(robust_keep_fraction),
            "robust_iters": int(robust_iters),
            "cond_proxy": float("nan"),
            "ill_conditioned": False,
            "weight_max_abs": float("nan"),
            "weights_all_finite": True,
            "inlier_fraction": float("nan"),
        }
    info_out: dict[str, float | int | str | bool] = {
        "n_samples": int(last["n_samples"]),
        "n_centers": int(last["n_centers"]),
        "res_median": float(last["res_median"]),
        "res_p90": float(last["res_p90"]),
        "ell_space": float(last["ell_space"]),
        "ell_time": float(last["ell_time"]),
        "lam": float(last["lam"]),
        "robust_keep_fraction": float(last["robust_keep_fraction"]),
        "robust_iters": int(last["robust_iters"]),
        "robust_mode": str(robust_mode),
        "track_trim_iters": int(track_trim_iters),
        "space_input_mode": str(space_input_mode),
        "two_stage": bool(two_stage),
        "n_refine": int(n_ref),
        "rbf_linear_terms": bool(rbf_linear_terms),
        "cond_proxy_max": float(np.max([float(ii["cond_proxy"]) for ii in stage_infos])) if stage_infos else float("nan"),
        "ill_conditioned_any": bool(np.any([bool(ii["ill_conditioned"]) for ii in stage_infos])) if stage_infos else False,
        "weight_max_abs_max": float(np.max([float(ii["weight_max_abs"]) for ii in stage_infos])) if stage_infos else float("nan"),
        "weights_all_finite_all": bool(np.all([bool(ii["weights_all_finite"]) for ii in stage_infos])) if stage_infos else True,
        "n_stages_fitted": int(len(stage_infos)),
    }
    return model, info_out


def predict_spacetime_drift(model: SpaceTimeRBFModel, u: np.ndarray) -> np.ndarray:
    """Predict drift for raw u=(x,y,z,t) using the first stage as approximation.

    This function is kept for compatibility. For full sequential prediction over
    all refinement stages, use `apply_correction`.
    """
    if len(model.stages) == 0:
        return np.zeros((u.shape[0], 3), dtype=float)

    stage = model.stages[0]
    spatial = u[:, :3]
    t_raw = u[:, 3]

    tr = max(stage.time_max - stage.time_min, 1e-9)
    xn = (spatial - stage.spatial_center[None, :]) / max(stage.spatial_scale, 1e-9)
    tn = (t_raw - stage.time_min) / tr
    uu = np.column_stack([xn, tn])

    k = _kernel_spacetime(uu, stage.centers_u, stage.ell_space, stage.ell_time)
    pred_res = _augment_kernel_with_linear(k, u=uu, rbf_linear_terms=stage.rbf_linear_terms) @ stage.weights_aug
    if stage.two_stage:
        g = _interp_global_drift(t_raw, stage.global_time_values, stage.global_drift_values)
    else:
        g = np.zeros_like(pred_res)
    return pred_res + g


def apply_correction(df: pd.DataFrame, model: SpaceTimeRBFModel) -> pd.DataFrame:
    """Apply space-time RBF correction (all refinement stages sequentially)."""
    d = ensure_numeric_tracks(df)

    current = d.copy()
    xyz_current = current[["Position X", "Position Y", "Position Z"]].to_numpy(float)

    for stage in model.stages:
        drift_pred = _predict_stage_drift(stage, current)
        xyz_current = xyz_current - drift_pred
        current.loc[:, "Position X"] = xyz_current[:, 0]
        current.loc[:, "Position Y"] = xyz_current[:, 1]
        current.loc[:, "Position Z"] = xyz_current[:, 2]

    out = d.copy()
    out["Position X_corr"] = xyz_current[:, 0]
    out["Position Y_corr"] = xyz_current[:, 1]
    out["Position Z_corr"] = xyz_current[:, 2]
    return out


def make_spacetime_diagnostic_plots(model: SpaceTimeRBFModel, out_prefix: str | Path) -> None:
    """Save ST-RBF diagnostics: center coverage and predicted drift magnitude over time."""
    if not model.stages:
        return
    out_prefix = str(out_prefix)

    for i, stage in enumerate(model.stages, start=1):
        tab = stage.per_time_table
        if tab is None or tab.shape[0] == 0:
            continue

        suf = "" if len(model.stages) == 1 else f"_stage{i}"

        if "center_nn_median" in tab.columns:
            fig = plt.figure()
            plt.plot(tab["Time"].to_numpy(float), tab["center_nn_median"].to_numpy(float), marker="o")
            plt.xlabel("Time")
            plt.ylabel("Median nearest-center distance (um)")
            plt.title("ST-RBF center coverage over time")
            plt.tight_layout()
            plt.savefig(out_prefix + f"_st_rbf_center_coverage{suf}.png", dpi=300)
            plt.close(fig)

        if "pred_drift_median" in tab.columns:
            fig = plt.figure()
            plt.plot(tab["Time"].to_numpy(float), tab["pred_drift_median"].to_numpy(float), marker="o")
            plt.xlabel("Time")
            plt.ylabel("Median predicted drift magnitude (um)")
            plt.title("ST-RBF predicted drift magnitude over time")
            plt.tight_layout()
            plt.savefig(out_prefix + f"_st_rbf_pred_drift_median{suf}.png", dpi=300)
            plt.close(fig)
