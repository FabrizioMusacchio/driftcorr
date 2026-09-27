"""Mixture-of-experts extension for the space-time RBF drift model.

This module clusters tracks by anchor position and fits one ST-RBF expert per
cluster. During inference, expert drift predictions are blended with smooth
Gaussian weights in space.

Mathematically:
- Cluster anchors `p_i` into experts with centers `c_k`.
- Fit expert models `f_k(p,t)` using only tracks in cluster `k`.
- Blend predictions:
  `d_hat(p,t) = sum_k w_k(p) f_k(p,t)`,
  with `w_k(p) ∝ exp(-||p-c_k||^2 / (2*sigma_blend^2))`.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks
from .method_spacetime_rbf import SpaceTimeRBFModel, apply_correction, fit_spacetime_rbf_model

# %% DATA CLASSES
@dataclass
class SpaceTimeRBFExpert:
    center_xyz: np.ndarray
    track_ids: np.ndarray
    model: SpaceTimeRBFModel
    fit_info: dict[str, float | int | str | bool]

@dataclass
class SpaceTimeRBFMoEModel:
    experts: list[SpaceTimeRBFExpert]
    blend_sigma_um: float
    cluster_mode: str
    reference_time: float
    shared_global_time_values: np.ndarray
    shared_global_drift_values: np.ndarray
    drift_cap_um: float
    group_coupling_alpha: float
# %% FUNCTIONS
def _fit_table_total_drift(df: pd.DataFrame, reference_time: float, fit_track_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Build table with total drift targets D(t)=X(t)-X(ref) for residual diagnostics."""
    d = ensure_numeric_tracks(df)
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        d = d[d["TrackID"].astype(float).isin(ids)].copy()
    ref = d[d["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    ref = ref.rename(columns={"Position X": "X_ref", "Position Y": "Y_ref", "Position Z": "Z_ref"})
    tab = d.merge(ref, on="TrackID", how="inner")
    if tab.shape[0] == 0:
        return tab
    tab["dx_true"] = tab["Position X"].to_numpy(float) - tab["X_ref"].to_numpy(float)
    tab["dy_true"] = tab["Position Y"].to_numpy(float) - tab["Y_ref"].to_numpy(float)
    tab["dz_true"] = tab["Position Z"].to_numpy(float) - tab["Z_ref"].to_numpy(float)
    return tab

def _estimate_global_translation(
    df: pd.DataFrame,
    reference_time: float,
    fit_track_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate shared global translation g(t) from robust per-time medians."""
    tab = _fit_table_total_drift(df, reference_time=reference_time, fit_track_ids=fit_track_ids)
    if tab.shape[0] == 0:
        return np.array([], dtype=float), np.zeros((0, 3), dtype=float)
    gtab = (
        tab.groupby("Time", as_index=False)
        .agg(dx=("dx_true", "median"), dy=("dy_true", "median"), dz=("dz_true", "median"))
        .sort_values("Time")
        .reset_index(drop=True)
    )
    return gtab["Time"].to_numpy(float), gtab[["dx", "dy", "dz"]].to_numpy(float)

def _interp_global_drift(times: np.ndarray, g_time: np.ndarray, g_drift: np.ndarray) -> np.ndarray:
    if g_time.size == 0:
        return np.zeros((times.size, 3), dtype=float)
    out = np.zeros((times.size, 3), dtype=float)
    for ax in range(3):
        out[:, ax] = np.interp(times, g_time, g_drift[:, ax])
    return out

def _subtract_global_from_positions(df: pd.DataFrame, g_time: np.ndarray, g_drift: np.ndarray) -> pd.DataFrame:
    d = ensure_numeric_tracks(df).copy()
    if g_time.size == 0:
        return d
    t = d["Time"].to_numpy(float)
    g = _interp_global_drift(t, g_time, g_drift)
    xyz = d[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    xyz_res = xyz - g
    d.loc[:, "Position X"] = xyz_res[:, 0]
    d.loc[:, "Position Y"] = xyz_res[:, 1]
    d.loc[:, "Position Z"] = xyz_res[:, 2]
    return d

def _kmeans_simple(points: np.ndarray, k: int, seed: int = 0, n_iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)
    if pts.shape[0] <= k:
        return pts.copy(), np.arange(pts.shape[0], dtype=int)

    rng = np.random.default_rng(seed)
    idx0 = rng.choice(pts.shape[0], size=k, replace=False)
    centers = pts[idx0].copy()
    labels = np.zeros((pts.shape[0],), dtype=int)

    for _ in range(max(1, int(n_iters))):
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mi = labels == i
            if np.any(mi):
                new_centers[i] = np.mean(pts[mi], axis=0)
            else:
                new_centers[i] = pts[rng.integers(0, pts.shape[0])]
        if np.linalg.norm(new_centers - centers) < 1e-8:
            centers = new_centers
            break
        centers = new_centers
    return centers, labels

def _track_anchor_table(df: pd.DataFrame, reference_time: float, mode: str) -> pd.DataFrame:
    d = ensure_numeric_tracks(df)
    mode = str(mode)
    means = d.groupby("TrackID", as_index=False)[["Position X", "Position Y", "Position Z"]].mean()
    means = means.rename(columns={"Position X": "ax", "Position Y": "ay", "Position Z": "az"})
    if mode == "track_anchor":
        return means
    if mode == "ref_position":
        ref = d[d["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
        ref = ref.rename(columns={"Position X": "ax", "Position Y": "ay", "Position Z": "az"})
        out = means.merge(ref, on="TrackID", how="left", suffixes=("_mean", ""))
        miss = ~np.all(np.isfinite(out[["ax", "ay", "az"]].to_numpy(float)), axis=1)
        if np.any(miss):
            out.loc[miss, "ax"] = out.loc[miss, "ax_mean"]
            out.loc[miss, "ay"] = out.loc[miss, "ay_mean"]
            out.loc[miss, "az"] = out.loc[miss, "az_mean"]
        return out[["TrackID", "ax", "ay", "az"]]
    raise ValueError(f"Unknown cluster_mode: {mode}")

def _anchors_for_rows(df: pd.DataFrame, anchor_table: pd.DataFrame) -> np.ndarray:
    d = ensure_numeric_tracks(df)
    m = d[["TrackID"]].merge(anchor_table, on="TrackID", how="left")
    a = m[["ax", "ay", "az"]].to_numpy(float)
    miss = ~np.all(np.isfinite(a), axis=1)
    if np.any(miss):
        a[miss] = d.loc[miss, ["Position X", "Position Y", "Position Z"]].to_numpy(float)
    return a

def fit_spacetime_rbf_moe_model(
    df: pd.DataFrame,
    n_experts: int = 3,
    blend_sigma_um: float = 120.0,
    cluster_mode: str = "track_anchor",
    seed: int = 0,
    reference_time: float | None = None,
    fit_track_ids: np.ndarray | None = None,
    drift_cap_factor: float = 3.0,
    cond_abort_threshold: float = 1e12,
    group_coupling_alpha: float = 0.0,
    **st_kwargs: object,
) -> tuple[SpaceTimeRBFMoEModel, pd.DataFrame]:
    """Fit a spatial mixture of ST-RBF experts."""
    d = ensure_numeric_tracks(df)
    times = np.array(sorted(d["Time"].unique()), dtype=float)
    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)
    g_time, g_drift = _estimate_global_translation(d, reference_time=reference_time, fit_track_ids=fit_track_ids)
    d_res = _subtract_global_from_positions(d, g_time=g_time, g_drift=g_drift)

    anchors = _track_anchor_table(d, reference_time=reference_time, mode=cluster_mode)
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        anchors = anchors[anchors["TrackID"].astype(float).isin(ids)].copy()
    pts = anchors[["ax", "ay", "az"]].to_numpy(float)
    tids = anchors["TrackID"].to_numpy(float)
    if pts.shape[0] == 0:
        raise ValueError("No tracks available for ST-RBF-MoE fit.")

    k = max(1, min(int(n_experts), int(pts.shape[0])))
    centers, labels = _kmeans_simple(pts, k=k, seed=seed)

    experts: list[SpaceTimeRBFExpert] = []
    expert_ids: list[int] = []
    rows: list[dict[str, float | int | str | bool]] = []
    expert_raw_corrs: list[pd.DataFrame] = []
    for i in range(k):
        mi = labels == i
        ids_i = tids[mi]
        if ids_i.size == 0:
            continue
        expert_kwargs = dict(st_kwargs)
        n_rows_i = int(d_res[d_res["TrackID"].astype(float).isin(set(ids_i.tolist()))].shape[0])
        max_centers_i = max(8, n_rows_i // 2)
        m_req = int(expert_kwargs.get("m_centers", 70))
        expert_kwargs["m_centers"] = int(min(m_req, max_centers_i))
        expert_kwargs["two_stage"] = False
        model_i, info_i = fit_spacetime_rbf_model(
            d_res,
            reference_time=reference_time,
            fit_track_ids=ids_i,
            seed=int(seed + i),
            **expert_kwargs,
        )
        cond_max = float(info_i.get("cond_proxy_max", np.nan))
        if np.isfinite(cond_max) and cond_max > float(cond_abort_threshold):
            raise ValueError(f"ST-RBF-MoE expert {i} ill-conditioned (cond={cond_max:.3g}).")
        experts.append(
            SpaceTimeRBFExpert(
                center_xyz=centers[i].copy(),
                track_ids=np.asarray(ids_i, dtype=float),
                model=model_i,
                fit_info=dict(info_i),
            )
        )
        expert_ids.append(int(i))
        rows.append(
            {
                "expert_id": int(i),
                "center_x": float(centers[i, 0]),
                "center_y": float(centers[i, 1]),
                "center_z": float(centers[i, 2]),
                "n_tracks": int(ids_i.size),
                "res_median": float(info_i.get("res_median", np.nan)),
                "res_p90": float(info_i.get("res_p90", np.nan)),
                "n_stages_fitted": int(info_i.get("n_stages_fitted", 0)),
            }
        )
        corr_i = apply_correction(d_res, model_i)
        corr_i = corr_i[["TrackID", "Time", "Position X_corr", "Position Y_corr", "Position Z_corr"]].copy()
        corr_i = corr_i.rename(
            columns={
                "Position X_corr": f"Xc_{i}",
                "Position Y_corr": f"Yc_{i}",
                "Position Z_corr": f"Zc_{i}",
            }
        )
        expert_raw_corrs.append(corr_i)

    if not experts:
        raise ValueError("Failed to fit ST-RBF-MoE experts.")

    # Optional soft coupling between neighboring experts to reduce overfitting.
    alpha = float(np.clip(group_coupling_alpha, 0.0, 1.0))
    if alpha > 0.0 and len(experts) > 1:
        cexp = np.vstack([ex.center_xyz for ex in experts])
        d2 = np.sum((cexp[:, None, :] - cexp[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        w_nb = np.exp(-d2 / (2.0 * float(max(blend_sigma_um, 1e-6)) ** 2))
        for i, ex in enumerate(experts):
            wi = w_nb[i]
            s = float(np.sum(wi))
            if s <= 1e-12:
                continue
            wi = wi / s
            for si in range(min([len(ex.model.stages) for ex in experts])):
                ref = np.zeros_like(ex.model.stages[si].weights_aug)
                for j, exj in enumerate(experts):
                    if j == i:
                        continue
                    ref += wi[j] * exj.model.stages[si].weights_aug
                ex.model.stages[si].weights_aug = (1.0 - alpha) * ex.model.stages[si].weights_aug + alpha * ref

    tab_tot = _fit_table_total_drift(d, reference_time=reference_time, fit_track_ids=fit_track_ids)
    if tab_tot.shape[0] > 0:
        dmag = np.linalg.norm(tab_tot[["dx_true", "dy_true", "dz_true"]].to_numpy(float), axis=1)
        drift_cap_um = float(np.percentile(dmag, 99)) * float(max(drift_cap_factor, 1.0))
    else:
        drift_cap_um = float("inf")

    model = SpaceTimeRBFMoEModel(
        experts=experts,
        blend_sigma_um=float(max(blend_sigma_um, 1e-6)),
        cluster_mode=str(cluster_mode),
        reference_time=float(reference_time),
        shared_global_time_values=np.asarray(g_time, dtype=float),
        shared_global_drift_values=np.asarray(g_drift, dtype=float),
        drift_cap_um=float(drift_cap_um),
        group_coupling_alpha=float(alpha),
    )
    et = pd.DataFrame(rows).sort_values("expert_id").reset_index(drop=True)
    et["shared_global_translation"] = True
    et["drift_cap_um"] = float(drift_cap_um)
    et["group_coupling_alpha"] = float(alpha)

    # Blended residual diagnostics on fit samples: compare predicted blended drift vs D(t)=X(t)-X(ref).
    fit_tab = _fit_table_total_drift(d, reference_time=reference_time, fit_track_ids=fit_track_ids)
    if fit_tab.shape[0] > 0 and len(expert_raw_corrs) == len(experts):
        anchor_table = _track_anchor_table(d, reference_time=reference_time, mode=cluster_mode)
        a = _anchors_for_rows(fit_tab, anchor_table)
        cexp = np.vstack([ex.center_xyz for ex in experts])
        d2 = np.sum((a[:, None, :] - cexp[None, :, :]) ** 2, axis=2)
        w = np.exp(-d2 / (2.0 * float(max(blend_sigma_um, 1e-6)) ** 2))
        wsum = np.sum(w, axis=1, keepdims=True)
        wsum[wsum <= 1e-12] = 1.0
        w = w / wsum

        pred_stack = []
        fit_obs_res = d_res[["TrackID", "Time", "Position X", "Position Y", "Position Z"]].copy()
        fit_obs_res = fit_tab[["TrackID", "Time"]].merge(fit_obs_res, on=["TrackID", "Time"], how="left")
        for i, corr_i in enumerate(expert_raw_corrs):
            mm = fit_obs_res.merge(corr_i, on=["TrackID", "Time"], how="left")
            pred_i = np.column_stack(
                [
                    mm["Position X"].to_numpy(float) - mm[f"Xc_{i}"].to_numpy(float),
                    mm["Position Y"].to_numpy(float) - mm[f"Yc_{i}"].to_numpy(float),
                    mm["Position Z"].to_numpy(float) - mm[f"Zc_{i}"].to_numpy(float),
                ]
            )
            pred_i = np.nan_to_num(pred_i, nan=0.0, posinf=0.0, neginf=0.0)
            pred_stack.append(pred_i)
        pred_stack_arr = np.stack(pred_stack, axis=2)  # (N,3,K)
        pred_blend = np.zeros((fit_tab.shape[0], 3), dtype=float)
        for i in range(pred_stack_arr.shape[2]):
            pred_blend += pred_stack_arr[:, :, i] * w[:, [i]]

        dtrue = fit_tab[["dx_true", "dy_true", "dz_true"]].to_numpy(float)
        tt = fit_tab["Time"].to_numpy(float)
        gfit = _interp_global_drift(tt, g_time, g_drift)
        pred_total_blend = pred_blend + gfit
        res_blend = np.linalg.norm(pred_total_blend - dtrue, axis=1)
        et["blend_res_median"] = float(np.median(res_blend)) if res_blend.size else float("nan")
        et["blend_res_p90"] = float(np.percentile(res_blend, 90)) if res_blend.size else float("nan")

        assigned_map = {float(tid): int(lbl) for tid, lbl in zip(tids.tolist(), labels.tolist())}
        assigned = np.array([assigned_map.get(float(tid), -1) for tid in fit_tab["TrackID"].to_numpy(float)], dtype=int)
        for i_local, i_global in enumerate(expert_ids):
            resid_i = np.linalg.norm(pred_stack_arr[:, :, i_local] + gfit - dtrue, axis=1)
            ma = assigned == int(i_global)
            if np.any(ma):
                et.loc[et["expert_id"] == int(i_global), "assigned_res_median"] = float(np.median(resid_i[ma]))
                et.loc[et["expert_id"] == int(i_global), "assigned_res_p90"] = float(np.percentile(resid_i[ma], 90))
            else:
                et.loc[et["expert_id"] == int(i_global), "assigned_res_median"] = float("nan")
                et.loc[et["expert_id"] == int(i_global), "assigned_res_p90"] = float("nan")

    return model, et

def apply_spacetime_rbf_moe_correction(df: pd.DataFrame, model: SpaceTimeRBFMoEModel) -> pd.DataFrame:
    """Apply blended ST-RBF expert correction to all rows."""
    d = ensure_numeric_tracks(df)
    t = d["Time"].to_numpy(float)
    g = _interp_global_drift(t, model.shared_global_time_values, model.shared_global_drift_values)
    d_res = d.copy()
    xyz_obs = d[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    xyz_res = xyz_obs - g
    d_res.loc[:, "Position X"] = xyz_res[:, 0]
    d_res.loc[:, "Position Y"] = xyz_res[:, 1]
    d_res.loc[:, "Position Z"] = xyz_res[:, 2]

    anchor_table = _track_anchor_table(d, reference_time=model.reference_time, mode=model.cluster_mode)
    a = _anchors_for_rows(d, anchor_table)

    centers = np.vstack([ex.center_xyz for ex in model.experts])
    d2 = np.sum((a[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    w = np.exp(-d2 / (2.0 * float(model.blend_sigma_um) ** 2))
    w = np.clip(w, 1e-8, 1.0)
    wsum = np.sum(w, axis=1, keepdims=True)
    wsum[wsum <= 1e-12] = 1.0
    w = w / wsum

    drifts = []
    for ex in model.experts:
        corr = apply_correction(d_res, ex.model)
        xyz_corr = corr[["Position X_corr", "Position Y_corr", "Position Z_corr"]].to_numpy(float)
        drifts.append(xyz_res - xyz_corr)
    drift_stack = np.stack(drifts, axis=2)  # (N,3,K)

    drift_blend = np.zeros((d.shape[0], 3), dtype=float)
    for k in range(drift_stack.shape[2]):
        drift_blend += drift_stack[:, :, k] * w[:, [k]]

    drift_total = drift_blend + g
    cap = float(model.drift_cap_um)
    if np.isfinite(cap) and cap > 0:
        mag = np.linalg.norm(drift_total, axis=1)
        over = mag > cap
        if np.any(over):
            scale = (cap / np.maximum(mag[over], 1e-12))[:, None]
            drift_total[over] = drift_total[over] * scale

    out = d.copy()
    xyz_corr = xyz_obs - drift_total
    out["Position X_corr"] = xyz_corr[:, 0]
    out["Position Y_corr"] = xyz_corr[:, 1]
    out["Position Z_corr"] = xyz_corr[:, 2]
    return out

def make_st_moe_diagnostic_plots(expert_table: pd.DataFrame, out_prefix: str | Path) -> None:
    """Save ST-RBF-MoE expert-size and per-expert residual plots."""
    if expert_table is None or expert_table.shape[0] == 0:
        return
    out_prefix = str(out_prefix)
    tab = expert_table.sort_values("expert_id")

    fig = plt.figure(figsize=(7, 4))
    plt.bar(tab["expert_id"].to_numpy(int), tab["n_tracks"].to_numpy(float))
    plt.xlabel("Expert")
    plt.ylabel("Assigned tracks")
    plt.title("ST-RBF-MoE expert sizes")
    plt.tight_layout()
    plt.savefig(out_prefix + "_st_rbf_moe_expert_sizes.png", dpi=300)
    plt.close(fig)

    if "res_median" in tab.columns:
        fig = plt.figure(figsize=(7, 4))
        plt.bar(tab["expert_id"].to_numpy(int), tab["res_median"].to_numpy(float))
        plt.xlabel("Expert")
        plt.ylabel("Residual median (um)")
        plt.title("ST-RBF-MoE per-expert residuals")
        plt.tight_layout()
        plt.savefig(out_prefix + "_st_rbf_moe_expert_residuals.png", dpi=300)
        plt.close(fig)

    if {"assigned_res_median", "blend_res_median"}.issubset(tab.columns):
        fig = plt.figure(figsize=(8, 4))
        plt.bar(tab["expert_id"].to_numpy(int), tab["assigned_res_median"].to_numpy(float), label="assigned expert")
        blend = tab["blend_res_median"].to_numpy(float)
        if blend.size:
            plt.axhline(float(np.nanmedian(blend)), color="k", linestyle="--", label="blended median")
        plt.xlabel("Expert")
        plt.ylabel("Residual median (um)")
        plt.title("ST-RBF-MoE assigned vs blended residuals")
        handles, _ = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_st_rbf_moe_blended_residuals.png", dpi=300)
        plt.close(fig)
# %% END