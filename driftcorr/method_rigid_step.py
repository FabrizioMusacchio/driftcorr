"""Rigid and piecewise-rigid drift correction per time step.

This module estimates inter-frame transforms between consecutive timepoints and
maps all observations into a reference frame.

Global rigid model:
- Fit `Y ≈ R X + b` between matched tracks at `t0 -> t1`.
- `R` is estimated by Kabsch SVD (or optional z-only rotation).
- Robust trimmed re-estimation keeps the lowest-residual fraction repeatedly.
- Step transforms are composed and inverted to correct each row.

Piecewise rigid model:
- Tracks are clustered by mean position into `K` spatial groups.
- One rigid step model per group is estimated.
- Corrections are blended in space via Gaussian weights over group centers.

Core formulas:
- Composition: `(R2,b2) o (R1,b1) = (R2 R1, R2 b1 + b2)`.
- Inverse: `(R,b)^-1 = (R^T, -R^T b)`.
- Blending: `x_corr = Σ_k α_k(p) (R_k x + b_k)`, `α_k` Gaussian in space.

Controlling parameters:
- `keep_fraction`: robust inlier fraction.
- `inlier_reestimate_iters`: number of trimmed refit iterations.
- `mode`: `rigid` or `rotz`.
- `n_groups` / `blend_sigma_um`: piecewise rigid complexity and smoothness.

Effects:
- Lower `keep_fraction` is more robust but may discard valid motion.
- Higher `n_groups` captures spatially varying drift but risks overfitting.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .drift_smoothing import detect_stage_jumps, plot_drift_timeseries, smooth_drift_timeseries
from .io_imaris import ensure_numeric_tracks

# %% PIECEWISE RIGID MODEL DATA CLASS
@dataclass
class PiecewiseRigidModel:
    """Spatially partitioned rigid model with Gaussian blending."""

    centers: np.ndarray
    track_to_group: dict[float, int]
    group_steps: dict[int, dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]]
    global_step: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]
    times: np.ndarray
    ref_time: float
    blend_sigma_um: float

# %% RIGID FITTING FUNCTIONS
def kabsch_rigid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rigid transform y ~= R x + b via Kabsch."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    h = xc.T @ yc
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    b = y.mean(axis=0) - r @ x.mean(axis=0)
    return r, b

def rotz_only_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid fit constrained to z-axis rotation plus translation."""
    x2 = x[:, :2]
    y2 = y[:, :2]
    xc = x2 - x2.mean(axis=0)
    yc = y2 - y2.mean(axis=0)
    h = xc.T @ yc
    u, _, vt = np.linalg.svd(h)
    r2 = vt.T @ u.T
    if np.linalg.det(r2) < 0:
        vt[-1, :] *= -1
        r2 = vt.T @ u.T
    theta = np.arctan2(r2[1, 0], r2[0, 0])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    b = y.mean(axis=0) - r @ x.mean(axis=0)
    return r, b, float(theta)

def robust_rigid_trimmed(
    x: np.ndarray,
    y: np.ndarray,
    keep_fraction: float = 0.7,
    inlier_reestimate_iters: int = 30,
    mode: str = "rigid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Trimmed iterative rigid fit with explicit inlier re-estimation iterations."""
    n = int(x.shape[0])
    k = max(6, int(np.floor(float(keep_fraction) * n)))
    idx = np.arange(n)

    theta: float | None = None
    for _ in range(max(1, int(inlier_reestimate_iters))):
        if mode == "rotz":
            r, b, theta = rotz_only_fit(x[idx], y[idx])
        else:
            r, b = kabsch_rigid(x[idx], y[idx])

        yp = (x @ r.T) + b
        res = np.linalg.norm(yp - y, axis=1)
        new_idx = np.argsort(res)[:k]
        if np.array_equal(np.sort(new_idx), np.sort(idx)):
            break
        idx = new_idx

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[idx] = True
    yp = (x @ r.T) + b
    res = np.linalg.norm(yp - y, axis=1)
    return r, b, inlier_mask, res, theta

def estimate_step_rigid_from_tracks(
    df: pd.DataFrame,
    t0: float,
    t1: float,
    keep_fraction: float = 0.7,
    min_pairs: int = 15,
    mode: str = "rigid",
    inlier_reestimate_iters: int = 30,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float | int | bool]]:
    """Estimate one step transform t0 -> t1 based on shared tracks."""
    df0 = df[df["Time"] == t0][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    df1 = df[df["Time"] == t1][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    m = df0.merge(df1, on="TrackID", suffixes=("_t0", "_t1"))
    n = int(m.shape[0])
    if n < min_pairs:
        return None, None, {"ok": False, "t0": float(t0), "t1": float(t1), "n_pairs": n}

    x = m[["Position X_t0", "Position Y_t0", "Position Z_t0"]].to_numpy(float)
    y = m[["Position X_t1", "Position Y_t1", "Position Z_t1"]].to_numpy(float)

    r, b, inlier_mask, res, theta = robust_rigid_trimmed(
        x,
        y,
        keep_fraction=keep_fraction,
        inlier_reestimate_iters=inlier_reestimate_iters,
        mode=mode,
    )

    info: dict[str, float | int | bool] = {
        "ok": True,
        "t0": float(t0),
        "t1": float(t1),
        "n_pairs": n,
        "keep_fraction": float(keep_fraction),
        "inlier_reestimate_iters": int(inlier_reestimate_iters),
        "res_median": float(np.median(res)),
        "res_p90": float(np.percentile(res, 90)),
        "rot_deg": float(np.degrees(theta)) if theta is not None else np.nan,
        "b_norm": float(np.linalg.norm(b)),
        "inlier_fraction": float(np.mean(inlier_mask)),
    }
    return r, b, info

def compose_rb(r2: np.ndarray, b2: np.ndarray, r1: np.ndarray, b1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compose two rigid transforms (r1,b1) then (r2,b2)."""
    r = r2 @ r1
    b = r2 @ b1 + b2
    return r, b

def invert_rb(r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert rigid transform y = r x + b."""
    ri = r.T
    bi = -ri @ b
    return ri, bi

def build_step_transforms(
    df: pd.DataFrame,
    keep_fraction: float = 0.7,
    min_pairs: int = 15,
    mode: str = "rigid",
    inlier_reestimate_iters: int = 30,
) -> tuple[dict[tuple[float, float], tuple[np.ndarray, np.ndarray]], pd.DataFrame, np.ndarray]:
    """Build per-step transforms for all consecutive timepoints."""
    df = ensure_numeric_tracks(df)
    times = np.array(sorted(df["Time"].unique()), dtype=float)
    step: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, float | int | bool]] = []

    for i in range(len(times) - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        r, b, info = estimate_step_rigid_from_tracks(
            df,
            t0,
            t1,
            keep_fraction,
            min_pairs,
            mode,
            inlier_reestimate_iters=inlier_reestimate_iters,
        )
        rows.append(info)
        if info["ok"]:
            assert r is not None and b is not None
            step[(t0, t1)] = (r, b)
    return step, pd.DataFrame(rows), times

def cumulative_transform_to_ref(
    step: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    times: np.ndarray,
    ref_time: float,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return inverse mapping from each t to reference frame."""
    times = np.array(sorted(times), dtype=float)
    ref_time = float(ref_time)
    ref_idx = int(np.where(times == ref_time)[0][0])

    fwd: dict[float, tuple[np.ndarray, np.ndarray]] = {ref_time: (np.eye(3), np.zeros(3))}
    rcum = np.eye(3)
    bcum = np.zeros(3)

    for i in range(ref_idx, len(times) - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        if (t0, t1) in step:
            rstep, bstep = step[(t0, t1)]
            rcum, bcum = compose_rb(rstep, bstep, rcum, bcum)
        fwd[t1] = (rcum.copy(), bcum.copy())

    inv: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for t, (rft, bft) in fwd.items():
        inv[t] = invert_rb(rft, bft)
    return inv

def apply_rigid_correction_to_tracks(
    df: pd.DataFrame,
    step: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    times: np.ndarray,
    ref_time: float,
) -> pd.DataFrame:
    """Apply global rigid correction to all rows in a track dataframe."""
    df_corr = ensure_numeric_tracks(df)
    inv = cumulative_transform_to_ref(step, times, ref_time)

    x = df_corr[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    t = df_corr["Time"].to_numpy(float)

    xcorr = np.empty_like(x)
    for i in range(x.shape[0]):
        r, b = inv.get(float(t[i]), (np.eye(3), np.zeros(3)))
        xcorr[i] = r @ x[i] + b

    df_corr["Position X_corr"] = xcorr[:, 0]
    df_corr["Position Y_corr"] = xcorr[:, 1]
    df_corr["Position Z_corr"] = xcorr[:, 2]
    return df_corr

def _estimate_time_drift_from_corrected(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-time drift vectors as robust median(original - corrected)."""
    d0 = ensure_numeric_tracks(df_orig)
    dc = ensure_numeric_tracks(df_corr)
    if d0.shape[0] == dc.shape[0]:
        # Fast path: same row count and preserved row order.
        m = pd.DataFrame(
            {
                "Time": d0["Time"].to_numpy(float),
                "dx": d0["Position X"].to_numpy(float) - dc["Position X_corr"].to_numpy(float),
                "dy": d0["Position Y"].to_numpy(float) - dc["Position Y_corr"].to_numpy(float),
                "dz": d0["Position Z"].to_numpy(float) - dc["Position Z_corr"].to_numpy(float),
            }
        )
    else:
        m = d0.merge(dc, on=["TrackID", "Time"], how="inner", suffixes=("_orig", "_corr"))
        m["dx"] = m["Position X_orig"].to_numpy(float) - m["Position X_corr"].to_numpy(float)
        m["dy"] = m["Position Y_orig"].to_numpy(float) - m["Position Y_corr"].to_numpy(float)
        m["dz"] = m["Position Z_orig"].to_numpy(float) - m["Position Z_corr"].to_numpy(float)
        m = m[["Time", "dx", "dy", "dz"]]

    tab = (
        m.groupby("Time", as_index=False)
        .agg(
            dx=("dx", "median"),
            dy=("dy", "median"),
            dz=("dz", "median"),
        )
        .sort_values("Time")
        .reset_index(drop=True)
    )
    times = tab["Time"].to_numpy(float)
    drift = tab[["dx", "dy", "dz"]].to_numpy(float)
    return times, drift

def _smooth_drift_piecewise(
    times: np.ndarray,
    drift_raw: np.ndarray,
    method: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth drift trace, preserving optional jump discontinuities via segmentation."""
    detect_jumps = bool(params.get("detect_stage_jumps", True))
    if not detect_jumps or drift_raw.shape[0] <= 2:
        return smooth_drift_timeseries(times, drift_raw, method=method, params=params), np.zeros((drift_raw.shape[0],), dtype=bool)

    jump_mask = detect_stage_jumps(times, drift_raw, method="mad_threshold", params=params)
    if not np.any(jump_mask):
        return smooth_drift_timeseries(times, drift_raw, method=method, params=params), jump_mask

    out = np.array(drift_raw, dtype=float, copy=True)
    split_idx = [0] + [int(i) for i in np.where(jump_mask)[0].tolist()] + [int(drift_raw.shape[0])]
    for a, b in zip(split_idx[:-1], split_idx[1:]):
        seg_len = int(b - a)
        if seg_len <= 2:
            continue
        out[a:b] = smooth_drift_timeseries(times[a:b], drift_raw[a:b], method=method, params=params)
    return out, jump_mask

def apply_rigid_correction_with_optional_smoothing(
    df: pd.DataFrame,
    step: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    times: np.ndarray,
    ref_time: float,
    smoothing_enabled: bool = False,
    smoothing_method: str = "savgol",
    smoothing_params: dict[str, Any] | None = None,
    out_prefix: str | Path | None = None,
    tag: str = "rigid",
) -> pd.DataFrame:
    """Apply rigid correction with optional temporal smoothing of global drift.

    Default behavior is unchanged: when ``smoothing_enabled=False`` this calls
    ``apply_rigid_correction_to_tracks`` directly.
    """

    df_raw = apply_rigid_correction_to_tracks(df, step, times, ref_time)
    if not bool(smoothing_enabled):
        return df_raw

    params = {} if smoothing_params is None else dict(smoothing_params)
    tvals, drift_raw = _estimate_time_drift_from_corrected(df, df_raw)
    drift_smooth, jump_mask = _smooth_drift_piecewise(
        tvals,
        drift_raw,
        method=smoothing_method,
        params=params,
    )

    # Posthoc drift-only smoothing:
    # x_corr_s = x_corr_raw + (d_raw(t) - d_smooth(t))
    df_sm = df_raw.copy()
    tt = df_sm["Time"].to_numpy(float)
    map_raw = {float(t): drift_raw[i] for i, t in enumerate(tvals)}
    map_sm = {float(t): drift_smooth[i] for i, t in enumerate(tvals)}
    delta = np.vstack([map_raw.get(float(t), np.zeros(3)) - map_sm.get(float(t), np.zeros(3)) for t in tt])
    df_sm["Position X_corr"] = df_sm["Position X_corr"].to_numpy(float) + delta[:, 0]
    df_sm["Position Y_corr"] = df_sm["Position Y_corr"].to_numpy(float) + delta[:, 1]
    df_sm["Position Z_corr"] = df_sm["Position Z_corr"].to_numpy(float) + delta[:, 2]

    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        tab = pd.DataFrame(
            {
                "time": tvals,
                "dx_raw": drift_raw[:, 0],
                "dy_raw": drift_raw[:, 1],
                "dz_raw": drift_raw[:, 2],
                "dx_smooth": drift_smooth[:, 0],
                "dy_smooth": drift_smooth[:, 1],
                "dz_smooth": drift_smooth[:, 2],
                "jump_detected": jump_mask.astype(int),
            }
        )
        tab.to_csv(Path(str(out_prefix) + f"_{tag}_drift_timeseries.csv"), index=False)
        plot_drift_timeseries(tvals, drift_raw, drift_smooth, out_prefix=out_prefix, tag=tag)

    return df_sm

def _cluster_tracks_by_mean_position(
    df: pd.DataFrame,
    n_groups: int,
    seed: int = 0,
) -> tuple[dict[float, int], np.ndarray]:
    d = ensure_numeric_tracks(df)
    grp = d.groupby("TrackID")[["Position X", "Position Y", "Position Z"]].mean().reset_index()
    points = grp[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    tids = grp["TrackID"].to_numpy(float)

    n_groups = int(max(1, min(int(n_groups), points.shape[0])))
    if n_groups == 1:
        labels = np.zeros(points.shape[0], dtype=int)
        centers = np.nanmean(points, axis=0, keepdims=True)
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
        labels = km.fit_predict(points)
        centers = km.cluster_centers_

    track_to_group = {float(tid): int(labels[i]) for i, tid in enumerate(tids)}
    return track_to_group, centers

def build_piecewise_rigid_model(
    df: pd.DataFrame,
    n_groups: int = 3,
    keep_fraction: float = 0.7,
    min_pairs: int = 15,
    mode: str = "rigid",
    inlier_reestimate_iters: int = 30,
    blend_sigma_um: float = 120.0,
    seed: int = 0,
) -> tuple[PiecewiseRigidModel, pd.DataFrame]:
    """Build piecewise rigid transforms and blending model."""
    d = ensure_numeric_tracks(df)
    global_step, global_table, times = build_step_transforms(
        d,
        keep_fraction=keep_fraction,
        min_pairs=min_pairs,
        mode=mode,
        inlier_reestimate_iters=inlier_reestimate_iters,
    )

    track_to_group, centers = _cluster_tracks_by_mean_position(d, n_groups=n_groups, seed=seed)

    group_steps: dict[int, dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]] = {}
    rows: list[dict[str, float | int | bool]] = []

    for g in range(centers.shape[0]):
        tids = [tid for tid, gid in track_to_group.items() if gid == g]
        dfg = d[d["TrackID"].astype(float).isin(tids)].copy()
        step_g, table_g, _ = build_step_transforms(
            dfg,
            keep_fraction=keep_fraction,
            min_pairs=min_pairs,
            mode=mode,
            inlier_reestimate_iters=inlier_reestimate_iters,
        )
        group_steps[g] = step_g
        if not table_g.empty:
            tg = table_g.copy()
            tg["group"] = int(g)
            tg["center_x"] = float(centers[g, 0])
            tg["center_y"] = float(centers[g, 1])
            tg["center_z"] = float(centers[g, 2])
            rows.extend(tg.to_dict(orient="records"))

    ref_time = float(times.min())
    model = PiecewiseRigidModel(
        centers=centers,
        track_to_group=track_to_group,
        group_steps=group_steps,
        global_step=global_step,
        times=times,
        ref_time=ref_time,
        blend_sigma_um=float(blend_sigma_um),
    )

    table_piecewise = pd.DataFrame(rows)
    if table_piecewise.empty:
        table_piecewise = global_table.copy()
        table_piecewise["group"] = 0
        table_piecewise["center_x"] = float(centers[0, 0]) if centers.size else 0.0
        table_piecewise["center_y"] = float(centers[0, 1]) if centers.size else 0.0
        table_piecewise["center_z"] = float(centers[0, 2]) if centers.size else 0.0
    return model, table_piecewise

def apply_piecewise_rigid_correction_to_tracks(
    df: pd.DataFrame,
    model: PiecewiseRigidModel,
    use_track_mean_position: bool = True,
) -> pd.DataFrame:
    """Apply piecewise rigid correction using soft blending across spatial groups."""
    d = ensure_numeric_tracks(df)
    x = d[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    t = d["Time"].to_numpy(float)

    if use_track_mean_position:
        ptrack = d.groupby("TrackID")[["Position X", "Position Y", "Position Z"]].mean().reset_index()
        track_to_p = {
            float(r.TrackID): np.array([r["Position X"], r["Position Y"], r["Position Z"]], dtype=float)
            for _, r in ptrack.iterrows()
        }
        p = np.vstack([track_to_p.get(float(tid), np.array([np.nan, np.nan, np.nan])) for tid in d["TrackID"].to_numpy(float)])
    else:
        p = x.copy()

    inv_by_group: dict[int, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for g, step in model.group_steps.items():
        inv_by_group[g] = cumulative_transform_to_ref(step, model.times, model.ref_time)

    global_inv = cumulative_transform_to_ref(model.global_step, model.times, model.ref_time)

    centers = model.centers
    sigma = float(max(1e-9, model.blend_sigma_um))

    xcorr = np.empty_like(x)
    for i in range(x.shape[0]):
        dif = centers - p[i][None, :]
        d2 = np.sum(dif * dif, axis=1)
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 1e-12:
            w = np.ones((centers.shape[0],), dtype=float)
        w = w / np.sum(w)

        x_i = np.zeros(3, dtype=float)
        for g in range(centers.shape[0]):
            inv = inv_by_group.get(g, {})
            r, b = inv.get(float(t[i]), global_inv.get(float(t[i]), (np.eye(3), np.zeros(3))))
            x_i += w[g] * (r @ x[i] + b)
        xcorr[i] = x_i

    d["Position X_corr"] = xcorr[:, 0]
    d["Position Y_corr"] = xcorr[:, 1]
    d["Position Z_corr"] = xcorr[:, 2]
    return d

def run_rigid_global_vs_piecewise(
    df: pd.DataFrame,
    n_groups: int = 3,
    keep_fraction: float = 0.7,
    min_pairs: int = 15,
    mode: str = "rigid",
    inlier_reestimate_iters: int = 30,
    blend_sigma_um: float = 120.0,
    seed: int = 0,
) -> dict[str, object]:
    """Evaluation hook for comparing global rigid vs piecewise rigid on one dataset."""
    d = ensure_numeric_tracks(df)

    step, table_global, times = build_step_transforms(
        d,
        keep_fraction=keep_fraction,
        min_pairs=min_pairs,
        mode=mode,
        inlier_reestimate_iters=inlier_reestimate_iters,
    )
    ref_time = float(times.min())
    df_global = apply_rigid_correction_to_tracks(d, step, times, ref_time)

    piecewise_model, table_piecewise = build_piecewise_rigid_model(
        d,
        n_groups=n_groups,
        keep_fraction=keep_fraction,
        min_pairs=min_pairs,
        mode=mode,
        inlier_reestimate_iters=inlier_reestimate_iters,
        blend_sigma_um=blend_sigma_um,
        seed=seed,
    )
    df_piecewise = apply_piecewise_rigid_correction_to_tracks(d, piecewise_model)

    return {
        "df_global": df_global,
        "df_piecewise": df_piecewise,
        "table_global": table_global,
        "table_piecewise": table_piecewise,
        "piecewise_model": piecewise_model,
    }
# %% END