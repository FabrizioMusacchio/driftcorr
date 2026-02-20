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
    space_input_mode: str
    two_stage: bool
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
    return np.linalg.solve(a, b)


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
) -> tuple[SpaceTimeStageModel, dict[str, float | int | str | bool]]:
    d = ensure_numeric_tracks(df)

    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        dfit = d[d["TrackID"].astype(float).isin(ids)].copy()
    else:
        dfit = d.copy()

    ref = dfit[dfit["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    ref = ref.rename(columns={"Position X": "X_ref", "Position Y": "Y_ref", "Position Z": "Z_ref"})
    mdf = dfit.merge(ref, on="TrackID", how="inner")
    if mdf.shape[0] < max(20, int(m_centers)):
        raise ValueError("Too few training samples for space-time RBF fit.")

    # Spatial inputs over full fitting dataframe, then align to merged rows.
    sp_all = _spatial_inputs_for_rows(dfit, reference_time=reference_time, mode=space_input_mode)
    dfit_sp = dfit.copy()
    dfit_sp["_sx"] = sp_all[:, 0]
    dfit_sp["_sy"] = sp_all[:, 1]
    dfit_sp["_sz"] = sp_all[:, 2]
    sp = dfit_sp[["TrackID", "Time", "_sx", "_sy", "_sz"]].copy()

    mdf = mdf[["TrackID", "Time", "Position X", "Position Y", "Position Z", "X_ref", "Y_ref", "Z_ref"]].copy()
    mdf = mdf.merge(sp, on=["TrackID", "Time"], how="left")

    t_raw = mdf["Time"].to_numpy(float)
    spatial_raw = mdf[["_sx", "_sy", "_sz"]].to_numpy(float)

    d_total = np.column_stack(
        [
            mdf["Position X"].to_numpy(float) - mdf["X_ref"].to_numpy(float),
            mdf["Position Y"].to_numpy(float) - mdf["Y_ref"].to_numpy(float),
            mdf["Position Z"].to_numpy(float) - mdf["Z_ref"].to_numpy(float),
        ]
    )

    if two_stage:
        gtab = (
            pd.DataFrame({"Time": t_raw, "dx": d_total[:, 0], "dy": d_total[:, 1], "dz": d_total[:, 2]})
            .groupby("Time", as_index=False)
            .median()
            .sort_values("Time")
            .reset_index(drop=True)
        )
        g_time = gtab["Time"].to_numpy(float)
        g_drift = gtab[["dx", "dy", "dz"]].to_numpy(float)
        g_map = {float(tt): g_drift[i] for i, tt in enumerate(g_time)}
        g_per_sample = np.vstack([g_map[float(tt)] for tt in t_raw])
        d_target = d_total - g_per_sample
    else:
        g_time = np.array(sorted(np.unique(t_raw)), dtype=float)
        g_drift = np.zeros((g_time.size, 3), dtype=float)
        d_target = d_total

    xn, tn, center_spatial, scale_spatial, tmin, tmax = _normalize_inputs(spatial_raw, t_raw)
    u = np.column_stack([xn, tn])

    centers = _choose_centers(u, m=int(m_centers), seed=seed)
    k = _kernel_spacetime(u, centers, ell_space=float(ell_space), ell_time=float(ell_time))
    k_aug = _augment_kernel(k)

    wv = np.ones((u.shape[0],), dtype=float)
    keep_fraction = float(np.clip(robust_keep_fraction, 0.1, 0.99))
    k_keep = max(20, int(np.floor(keep_fraction * u.shape[0])))

    weights_aug = _weighted_ridge_aug(k_aug, d_target, wv, lam=lam, lam_bias=0.0)
    for _ in range(max(1, int(robust_iters))):
        pred_res = k_aug @ weights_aug
        res = np.linalg.norm(pred_res - d_target, axis=1)
        idx = np.argsort(res)[:k_keep]
        wv[:] = 0.0
        wv[idx] = 1.0
        weights_aug = _weighted_ridge_aug(k_aug, d_target, wv, lam=lam, lam_bias=0.0)

    pred_res = k_aug @ weights_aug

    # Add stage-1 global term back for diagnostics in total-drift space.
    if two_stage:
        g_map = {float(tt): g_drift[i] for i, tt in enumerate(g_time)}
        g_per_sample = np.vstack([g_map[float(tt)] for tt in t_raw])
        pred_total = pred_res + g_per_sample
    else:
        pred_total = pred_res

    res_total = np.linalg.norm(pred_total - d_total, axis=1)

    # Condition proxy for numerical stability.
    a = k_aug.T @ k_aug
    a.flat[:: a.shape[0] + 1] += np.r_[np.full((a.shape[0] - 1,), float(lam)), 0.0]
    cond_proxy = float(np.linalg.cond(a))
    ill_conditioned = bool(np.isfinite(cond_proxy) and cond_proxy > 1e12)

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
        space_input_mode=str(space_input_mode),
        two_stage=bool(two_stage),
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
        "space_input_mode": str(space_input_mode),
        "two_stage": bool(two_stage),
        "cond_proxy": float(cond_proxy),
        "ill_conditioned": bool(ill_conditioned),
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
    pred_res = _augment_kernel(k) @ stage.weights_aug

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
    space_input_mode: str = "row_position",
    n_refine: int = 2,
    two_stage: bool = False,
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

    for i in range(n_ref):
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
            two_stage=two_stage,
        )
        stages.append(stage)
        info = dict(info)
        info["refine_iter"] = int(i + 1)
        stage_infos.append(info)

        if i < n_ref - 1 and str(space_input_mode) == "row_position":
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

    last = stage_infos[-1]
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
        "space_input_mode": str(space_input_mode),
        "two_stage": bool(two_stage),
        "n_refine": int(n_ref),
        "cond_proxy_max": float(np.max([float(ii["cond_proxy"]) for ii in stage_infos])) if stage_infos else float("nan"),
        "ill_conditioned_any": bool(np.any([bool(ii["ill_conditioned"]) for ii in stage_infos])) if stage_infos else False,
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
    pred_res = _augment_kernel(k) @ stage.weights_aug
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
