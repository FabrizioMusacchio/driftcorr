"""RBF-based drift correction to a fixed reference frame.

This module models imaging drift as a smooth vector field per timepoint (or
jointly across time with temporal regularization). For matched tracks between
time `t` and reference `t_ref`, target drift vectors are:
`D_t = X_t - X_ref`.

Per-timepoint model:
- `D_t ≈ K(X_t, C) W_t`, where `K` is a Gaussian RBF kernel.
- Ridge regularization:
  `min_W ||K W - D||^2 + lambda ||W||^2`.
- Optional robust trimming via iterative inlier selection.

Temporal-consistent model:
- Shared centers `C`, time-indexed weights `W(t)`.
- Objective:
  `Σ_t ||K_t W(t) - D_t||^2 + lambda ||W(t)||^2 + gamma ||W(t)-W(t-1)||^2`.
- Solved as a block-tridiagonal linear system.

Kernel augmentation:
- A constant bias column is appended to the kernel:
  `K_aug = [K, 1]`.
- This lets the model represent global translation even with sparse center
  coverage.

Optional two-stage model:
- Stage 1: per-timepoint global translation from median drift over tracks.
- Stage 2: RBF residual field fit after removing Stage 1.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks


@dataclass
class RBFField:
    """RBF drift field parameters with augmented bias and optional Stage-1 shift."""

    centers: np.ndarray
    sigma: float
    W_aug: np.ndarray
    lam: float
    global_stage1: np.ndarray
    center_mode: str
    robust_mode: str


def _rbf_kernel(x: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    diff = x[:, None, :] - c[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    return np.exp(-r2 / (2.0 * sigma * sigma))


def _augment_kernel(k: np.ndarray) -> np.ndarray:
    return np.hstack([k, np.ones((k.shape[0], 1), dtype=float)])


def _choose_centers_random(x: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if x.shape[0] <= m:
        return x.copy()
    idx = rng.choice(x.shape[0], size=m, replace=False)
    return x[idx].copy()


def _kmeans_simple(points: np.ndarray, k: int, seed: int = 0, n_iters: int = 30) -> np.ndarray:
    """Small NumPy-only k-means used for kmeans_pooled center mode."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= k:
        return pts.copy()

    rng = np.random.default_rng(seed)
    centers = _choose_centers_random(pts, m=k, seed=seed)

    for _ in range(max(1, int(n_iters))):
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        lab = np.argmin(d2, axis=1)

        new_centers = np.empty_like(centers)
        for i in range(k):
            mask = lab == i
            if np.any(mask):
                new_centers[i] = np.mean(pts[mask], axis=0)
            else:
                new_centers[i] = pts[rng.integers(0, pts.shape[0])]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < 1e-9:
            break

    return centers


def _default_sigma(c: np.ndarray) -> float:
    m = c.shape[0]
    if m < 2:
        return 1.0
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    s = float(np.median(nn))
    if not np.isfinite(s) or s <= 0:
        finite = d2[np.isfinite(d2)]
        s = float(np.sqrt(np.median(finite))) if finite.size else 1.0
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return s


def _ridge_solve_aug(k_aug: np.ndarray, d: np.ndarray, lam: float, lam_bias: float = 0.0) -> np.ndarray:
    m_aug = k_aug.shape[1]
    a = k_aug.T @ k_aug
    reg = np.full((m_aug,), float(lam), dtype=float)
    reg[-1] = float(lam_bias)
    a.flat[:: m_aug + 1] += reg
    b = k_aug.T @ d
    return np.linalg.solve(a, b)


def _stable_solve(a: np.ndarray, b: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        aa = a.copy()
        aa.flat[:: aa.shape[0] + 1] += jitter
        return np.linalg.solve(aa, b)


def _merge_with_reference(
    df: pd.DataFrame,
    t: float,
    tref: float,
    fit_track_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    dft = df[df["Time"] == t][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    dfr = df[df["Time"] == tref][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    mdf = dft.merge(dfr, on="TrackID", suffixes=("_t", "_ref"))
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        mdf = mdf[mdf["TrackID"].astype(float).isin(ids)]
    return mdf


def _reference_track_ids(df: pd.DataFrame, reference_time: float, fit_track_ids: np.ndarray | None = None) -> np.ndarray:
    dfr = df[df["Time"] == float(reference_time)][["TrackID"]].copy()
    ids = np.array(sorted(dfr["TrackID"].astype(float).unique()), dtype=float)
    if fit_track_ids is not None:
        wanted = set(np.asarray(fit_track_ids, dtype=float).tolist())
        ids = np.array([tid for tid in ids if float(tid) in wanted], dtype=float)
    return ids


def _choose_centers(
    df: pd.DataFrame,
    reference_time: float,
    m: int,
    center_mode: str,
    seed: int = 0,
    fit_track_ids: np.ndarray | None = None,
    pooled_time_stride: int = 1,
) -> np.ndarray:
    """Choose centers by mode from reference or pooled positions."""
    d = ensure_numeric_tracks(df)
    center_mode = str(center_mode)

    ref_ids = _reference_track_ids(d, reference_time=reference_time, fit_track_ids=fit_track_ids)
    if ref_ids.size == 0:
        return np.zeros((0, 3), dtype=float)

    ids_set = set(ref_ids.tolist())
    dsel = d[d["TrackID"].astype(float).isin(ids_set)].copy()

    if center_mode == "ref":
        pref = dsel[dsel["Time"] == float(reference_time)][["Position X", "Position Y", "Position Z"]].to_numpy(float)
        return _choose_centers_random(pref, m=int(m), seed=seed)

    times = np.array(sorted(dsel["Time"].unique()), dtype=float)
    stride = max(1, int(pooled_time_stride))
    if stride > 1:
        times = times[::stride]

    pooled_rows: list[np.ndarray] = []
    for t in times:
        arr = dsel[dsel["Time"] == float(t)][["Position X", "Position Y", "Position Z"]].to_numpy(float)
        if arr.size:
            pooled_rows.append(arr)
    pooled = np.vstack(pooled_rows) if pooled_rows else np.zeros((0, 3), dtype=float)

    if pooled.shape[0] == 0:
        pref = dsel[dsel["Time"] == float(reference_time)][["Position X", "Position Y", "Position Z"]].to_numpy(float)
        return _choose_centers_random(pref, m=int(m), seed=seed)

    if center_mode == "pooled":
        return _choose_centers_random(pooled, m=int(m), seed=seed)
    if center_mode == "kmeans_pooled":
        return _kmeans_simple(pooled, k=min(int(m), pooled.shape[0]), seed=seed)

    raise ValueError(f"Unknown center_mode: {center_mode}")


def _nearest_center_median_distance(x: np.ndarray, centers: np.ndarray) -> float:
    if x.shape[0] == 0 or centers.shape[0] == 0:
        return float("nan")
    d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    dmin = np.sqrt(np.min(d2, axis=1))
    return float(np.median(dmin))


def robust_fit_rbf_field(
    x: np.ndarray,
    d: np.ndarray,
    centers: np.ndarray,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    max_iters: int = 25,
    robust_mode: str = "sample_trim",
    lam_bias: float = 0.0,
    center_mode: str = "ref",
    global_stage1: np.ndarray | None = None,
) -> tuple[RBFField, np.ndarray, np.ndarray]:
    """Fit D ~= [K(X,C),1]W with sample trimming or plain ridge."""
    n = int(x.shape[0])
    if n < 8:
        raise ValueError("Too few correspondences for RBF field fit.")

    if sigma is None:
        sigma = _default_sigma(centers)
    sigma = float(sigma)

    k = _rbf_kernel(x, centers, sigma)
    k_aug = _augment_kernel(k)

    if robust_mode not in {"sample_trim", "none"}:
        raise ValueError(f"Unsupported robust_mode in robust_fit_rbf_field: {robust_mode}")

    if robust_mode == "none":
        w_aug = _ridge_solve_aug(k_aug, d, lam=lam, lam_bias=lam_bias)
        pred = k_aug @ w_aug
        res = np.linalg.norm(pred - d, axis=1)
        inlier_mask = np.ones((n,), dtype=bool)
    else:
        keep_fraction = float(np.clip(keep_fraction, 0.1, 0.99))
        k_keep = max(8, int(np.floor(keep_fraction * n)))

        w_aug = _ridge_solve_aug(k_aug, d, lam=lam, lam_bias=lam_bias)
        prev = np.inf
        inlier_idx = np.arange(n)

        for _ in range(int(max_iters)):
            pred = k_aug @ w_aug
            res = np.linalg.norm(pred - d, axis=1)
            order = np.argsort(res)
            inlier_idx = order[:k_keep]

            ki = k_aug[inlier_idx]
            di = d[inlier_idx]
            wi = _ridge_solve_aug(ki, di, lam=lam, lam_bias=lam_bias)

            pred_i = ki @ wi
            cost = float(np.mean(np.sum((pred_i - di) ** 2, axis=1)))
            w_aug = wi
            if abs(prev - cost) < 1e-12:
                break
            prev = cost

        pred = k_aug @ w_aug
        res = np.linalg.norm(pred - d, axis=1)
        inlier_mask = np.zeros((n,), dtype=bool)
        inlier_mask[inlier_idx] = True

    field = RBFField(
        centers=centers.copy(),
        sigma=float(sigma),
        W_aug=w_aug,
        lam=float(lam),
        global_stage1=np.zeros((3,), dtype=float) if global_stage1 is None else np.asarray(global_stage1, dtype=float),
        center_mode=str(center_mode),
        robust_mode=str(robust_mode),
    )
    return field, inlier_mask, res


def predict_rbf_field(field: RBFField, x: np.ndarray) -> np.ndarray:
    """Predict drift vectors at positions x."""
    k = _rbf_kernel(x, field.centers, field.sigma)
    k_aug = _augment_kernel(k)
    return k_aug @ field.W_aug + field.global_stage1[None, :]


def _solve_temporal_block_tridiagonal(
    diag_blocks: list[np.ndarray],
    off_blocks: list[np.ndarray],
    rhs_blocks: list[np.ndarray],
) -> list[np.ndarray]:
    """Solve block-tridiagonal system for W(t) with RHS shape (M_aug,3) per time."""
    t_count = len(diag_blocks)
    if t_count == 1:
        return [_stable_solve(diag_blocks[0], rhs_blocks[0])]

    dmod = [b.copy() for b in diag_blocks]
    g: list[np.ndarray] = []
    y: list[np.ndarray] = [_stable_solve(dmod[0], rhs_blocks[0])]

    for i in range(t_count - 1):
        gi = _stable_solve(dmod[i], off_blocks[i])
        g.append(gi)
        dmod[i + 1] = dmod[i + 1] - off_blocks[i].T @ gi
        yi = _stable_solve(dmod[i + 1], rhs_blocks[i + 1] - off_blocks[i].T @ y[i])
        y.append(yi)

    w: list[np.ndarray] = [np.zeros_like(rhs_blocks[0]) for _ in range(t_count)]
    w[-1] = y[-1]
    for i in range(t_count - 2, -1, -1):
        w[i] = y[i] - g[i] @ w[i + 1]
    return w


def _collect_pair_blocks(
    df: pd.DataFrame,
    times: np.ndarray,
    reference_time: float,
    min_pairs: int,
    fit_track_ids: np.ndarray | None,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    valid_times: list[float] = []
    x_blocks: list[np.ndarray] = []
    d_blocks: list[np.ndarray] = []
    tid_blocks: list[np.ndarray] = []

    for t in times:
        t = float(t)
        if t == float(reference_time):
            continue
        mdf = _merge_with_reference(df, t=t, tref=reference_time, fit_track_ids=fit_track_ids)
        if int(mdf.shape[0]) < int(min_pairs):
            continue
        xt = mdf[["Position X_t", "Position Y_t", "Position Z_t"]].to_numpy(float)
        xr = mdf[["Position X_ref", "Position Y_ref", "Position Z_ref"]].to_numpy(float)
        d = xt - xr
        tids = mdf["TrackID"].to_numpy(float)

        valid_times.append(t)
        x_blocks.append(xt)
        d_blocks.append(d)
        tid_blocks.append(tids)

    return valid_times, x_blocks, d_blocks, tid_blocks


def fit_temporal_rbf_fields_to_reference(
    df: pd.DataFrame,
    times: np.ndarray,
    reference_time: float,
    centers: np.ndarray,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    temporal_gamma: float = 0.0,
    fit_track_ids: np.ndarray | None = None,
    center_mode: str = "ref",
    two_stage: bool = False,
    lam_bias: float = 0.0,
) -> tuple[dict[float, RBFField], dict[float, dict[str, float | int | bool | str]]]:
    """Fit all non-reference timepoint RBF fields jointly with temporal smoothness."""
    rows_by_time: dict[float, dict[str, float | int | bool | str]] = {}
    fields_by_time: dict[float, RBFField] = {}

    if centers.shape[0] == 0:
        return fields_by_time, rows_by_time
    if sigma is None:
        sigma = _default_sigma(centers)
    sigma = float(sigma)

    valid_times, x_blocks, d_blocks_raw, _ = _collect_pair_blocks(
        df=df,
        times=times,
        reference_time=reference_time,
        min_pairs=min_pairs,
        fit_track_ids=fit_track_ids,
    )
    t_count = len(valid_times)
    if t_count == 0:
        return fields_by_time, rows_by_time

    m_count = centers.shape[0]
    m_aug = m_count + 1

    d_blocks: list[np.ndarray] = []
    stage1: list[np.ndarray] = []
    k_aug_blocks: list[np.ndarray] = []
    for i in range(t_count):
        d_raw = d_blocks_raw[i]
        g = np.median(d_raw, axis=0) if two_stage else np.zeros((3,), dtype=float)
        d_res = d_raw - g[None, :]
        k_aug = _augment_kernel(_rbf_kernel(x_blocks[i], centers, sigma))

        stage1.append(np.asarray(g, dtype=float))
        d_blocks.append(d_res)
        k_aug_blocks.append(k_aug)

    reg = np.full((m_aug,), float(lam), dtype=float)
    reg[-1] = float(lam_bias)

    ident = np.eye(m_aug, dtype=float)
    reg_mat = np.diag(reg)

    diag_blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    for i in range(t_count):
        k_aug = k_aug_blocks[i]
        a = (k_aug.T @ k_aug) + reg_mat
        if temporal_gamma > 0:
            if i == 0 or i == t_count - 1:
                a = a + float(temporal_gamma) * ident
            else:
                a = a + 2.0 * float(temporal_gamma) * ident
        diag_blocks.append(a)
        rhs_blocks.append(k_aug.T @ d_blocks[i])

    off_blocks: list[np.ndarray] = []
    if temporal_gamma > 0:
        off = -float(temporal_gamma) * ident
        for _ in range(t_count - 1):
            off_blocks.append(off.copy())
    else:
        for _ in range(t_count - 1):
            off_blocks.append(np.zeros_like(ident))

    w_blocks = _solve_temporal_block_tridiagonal(diag_blocks, off_blocks, rhs_blocks)

    smoothness_prev: dict[int, float] = {}
    if t_count > 1:
        for i in range(1, t_count):
            pred_prev = k_aug_blocks[i - 1] @ w_blocks[i - 1] + stage1[i - 1][None, :]
            pred_cur = k_aug_blocks[i] @ w_blocks[i] + stage1[i][None, :]
            mmin = min(pred_prev.shape[0], pred_cur.shape[0])
            step = np.linalg.norm(pred_cur[:mmin] - pred_prev[:mmin], axis=1)
            smoothness_prev[i] = float(np.median(step)) if step.size else float("nan")

    for i, t in enumerate(valid_times):
        field = RBFField(
            centers=centers.copy(),
            sigma=sigma,
            W_aug=w_blocks[i],
            lam=float(lam),
            global_stage1=stage1[i],
            center_mode=str(center_mode),
            robust_mode="sample_trim",
        )
        fields_by_time[t] = field

        d_pred_total = predict_rbf_field(field, x_blocks[i])
        d_target_total = d_blocks_raw[i]
        res = np.linalg.norm(d_pred_total - d_target_total, axis=1)

        rows_by_time[t] = {
            "diag_type": "timepoint",
            "ok": True,
            "t": float(t),
            "tref": float(reference_time),
            "n_pairs": int(d_blocks_raw[i].shape[0]),
            "inlier_fraction": 1.0,
            "res_median": float(np.median(res)),
            "res_p90": float(np.percentile(res, 90)),
            "sigma": float(sigma),
            "lam": float(lam),
            "M": int(m_count),
            "center_mode": str(center_mode),
            "robust_mode": "sample_trim",
            "temporal_gamma": float(temporal_gamma),
            "temporal_res_median": float(np.median(res)),
            "temporal_smoothness_prev": float(smoothness_prev.get(i, np.nan)),
            "center_nn_median": _nearest_center_median_distance(x_blocks[i], centers),
            "pred_drift_median": float(np.median(np.linalg.norm(d_pred_total, axis=1))),
            "global_stage1_mag": float(np.linalg.norm(stage1[i])),
            "two_stage": bool(two_stage),
        }

    return fields_by_time, rows_by_time


def estimate_rbf_field_to_reference(
    df: pd.DataFrame,
    t: float,
    tref: float,
    centers: np.ndarray,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    fit_track_ids: np.ndarray | None = None,
    robust_mode: str = "sample_trim",
    center_mode: str = "ref",
    two_stage: bool = False,
    lam_bias: float = 0.0,
) -> tuple[RBFField | None, dict[str, float | int | bool | str]]:
    """Fit a drift field for one timepoint against the reference time."""
    mdf = _merge_with_reference(df, t=t, tref=tref, fit_track_ids=fit_track_ids)

    n = int(mdf.shape[0])
    if n < int(min_pairs):
        info: dict[str, float | int | bool | str] = {
            "diag_type": "timepoint",
            "ok": False,
            "t": float(t),
            "tref": float(tref),
            "n_pairs": n,
            "reason": "Too few pairs",
            "center_mode": str(center_mode),
            "robust_mode": str(robust_mode),
            "two_stage": bool(two_stage),
        }
        return None, info

    xt = mdf[["Position X_t", "Position Y_t", "Position Z_t"]].to_numpy(float)
    xr = mdf[["Position X_ref", "Position Y_ref", "Position Z_ref"]].to_numpy(float)
    d_total = xt - xr

    g = np.median(d_total, axis=0) if two_stage else np.zeros((3,), dtype=float)
    d_res = d_total - g[None, :]

    field, inlier_mask, _ = robust_fit_rbf_field(
        x=xt,
        d=d_res,
        centers=centers,
        keep_fraction=keep_fraction,
        lam=lam,
        sigma=sigma,
        robust_mode="none" if robust_mode == "track_trim" else robust_mode,
        lam_bias=lam_bias,
        center_mode=center_mode,
        global_stage1=g,
    )

    d_pred = predict_rbf_field(field, xt)
    res = np.linalg.norm(d_pred - d_total, axis=1)

    info = {
        "diag_type": "timepoint",
        "ok": True,
        "t": float(t),
        "tref": float(tref),
        "n_pairs": n,
        "inlier_fraction": float(np.mean(inlier_mask)),
        "res_median": float(np.median(res)),
        "res_p90": float(np.percentile(res, 90)),
        "sigma": float(field.sigma),
        "lam": float(field.lam),
        "M": int(field.centers.shape[0]),
        "center_mode": str(center_mode),
        "robust_mode": str(robust_mode),
        "center_nn_median": _nearest_center_median_distance(xt, field.centers),
        "pred_drift_median": float(np.median(np.linalg.norm(d_pred, axis=1))),
        "global_stage1_mag": float(np.linalg.norm(g)),
        "two_stage": bool(two_stage),
    }
    return field, info


def _compute_track_trim_residuals(
    df: pd.DataFrame,
    reference_time: float,
    fields_by_time: dict[float, Optional[RBFField]],
    candidate_track_ids: np.ndarray,
) -> pd.DataFrame:
    """Compute per-track residual medians across all available timepoints."""
    residuals: dict[float, list[float]] = {float(tid): [] for tid in candidate_track_ids.tolist()}

    for t, field in fields_by_time.items():
        if field is None:
            continue
        mdf = _merge_with_reference(df, t=float(t), tref=float(reference_time), fit_track_ids=candidate_track_ids)
        if mdf.shape[0] == 0:
            continue
        xt = mdf[["Position X_t", "Position Y_t", "Position Z_t"]].to_numpy(float)
        xr = mdf[["Position X_ref", "Position Y_ref", "Position Z_ref"]].to_numpy(float)
        d = xt - xr
        pred = predict_rbf_field(field, xt)
        res = np.linalg.norm(pred - d, axis=1)
        tids = mdf["TrackID"].to_numpy(float)
        for tid, rr in zip(tids, res):
            residuals[float(tid)].append(float(rr))

    rows: list[dict[str, float | bool]] = []
    for tid in candidate_track_ids.tolist():
        vals = residuals.get(float(tid), [])
        med = float(np.median(vals)) if vals else float("inf")
        rows.append({"TrackID": float(tid), "track_residual_median": med})
    return pd.DataFrame(rows)


def _fit_fields_once(
    df: pd.DataFrame,
    times: np.ndarray,
    reference_time: float,
    centers: np.ndarray,
    keep_fraction: float,
    lam: float,
    sigma: float,
    min_pairs: int,
    temporal_gamma: float,
    fit_track_ids: np.ndarray | None,
    robust_mode: str,
    center_mode: str,
    two_stage: bool,
    lam_bias: float,
) -> tuple[dict[float, Optional[RBFField]], list[dict[str, float | int | bool | str]]]:
    fields_by_time: dict[float, Optional[RBFField]] = {float(reference_time): None}
    tf_rows: list[dict[str, float | int | bool | str]] = []

    tf_rows.append(
        {
            "diag_type": "timepoint",
            "ok": True,
            "t": float(reference_time),
            "tref": float(reference_time),
            "n_pairs": np.nan,
            "inlier_fraction": np.nan,
            "res_median": 0.0,
            "res_p90": 0.0,
            "sigma": float(sigma),
            "lam": float(lam),
            "M": int(centers.shape[0]),
            "temporal_gamma": float(temporal_gamma),
            "temporal_res_median": 0.0,
            "temporal_smoothness_prev": np.nan,
            "center_mode": str(center_mode),
            "robust_mode": str(robust_mode),
            "center_nn_median": _nearest_center_median_distance(
                df[df["Time"] == float(reference_time)][["Position X", "Position Y", "Position Z"]].to_numpy(float),
                centers,
            ),
            "pred_drift_median": 0.0,
            "global_stage1_mag": 0.0,
            "two_stage": bool(two_stage),
        }
    )

    if float(temporal_gamma) > 0.0:
        temporal_fields, temporal_rows = fit_temporal_rbf_fields_to_reference(
            df=df,
            times=times,
            reference_time=reference_time,
            centers=centers,
            lam=lam,
            sigma=sigma,
            min_pairs=min_pairs,
            temporal_gamma=float(temporal_gamma),
            fit_track_ids=fit_track_ids,
            center_mode=center_mode,
            two_stage=two_stage,
            lam_bias=lam_bias,
        )
        for t in times:
            t = float(t)
            if t == float(reference_time):
                continue
            row = temporal_rows.get(
                t,
                {
                    "diag_type": "timepoint",
                    "ok": False,
                    "t": t,
                    "tref": float(reference_time),
                    "n_pairs": 0,
                    "reason": "No temporal fit row",
                    "center_mode": str(center_mode),
                    "robust_mode": str(robust_mode),
                    "temporal_gamma": float(temporal_gamma),
                    "two_stage": bool(two_stage),
                },
            )
            tf_rows.append(row)
            field = temporal_fields.get(t)
            if field is not None:
                fields_by_time[t] = field
    else:
        for t in times:
            t = float(t)
            if t == float(reference_time):
                continue
            field, info = estimate_rbf_field_to_reference(
                df=df,
                t=t,
                tref=reference_time,
                centers=centers,
                keep_fraction=keep_fraction,
                lam=lam,
                sigma=sigma,
                min_pairs=min_pairs,
                fit_track_ids=fit_track_ids,
                robust_mode=robust_mode,
                center_mode=center_mode,
                two_stage=two_stage,
                lam_bias=lam_bias,
            )
            info["temporal_gamma"] = float(temporal_gamma)
            info.setdefault("temporal_res_median", info.get("res_median", np.nan))
            info.setdefault("temporal_smoothness_prev", np.nan)
            tf_rows.append(info)
            if bool(info.get("ok", False)) and field is not None:
                fields_by_time[t] = field

    return fields_by_time, tf_rows


def correct_positions_rbf_to_reference(
    df: pd.DataFrame,
    reference_time: float | None = None,
    m: int = 25,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    seed: int = 0,
    fit_track_ids: np.ndarray | None = None,
    temporal_gamma: float = 0.0,
    center_mode: str = "ref",
    pooled_time_stride: int = 1,
    robust_mode: str = "sample_trim",
    track_trim_iters: int = 4,
    two_stage: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, Optional[RBFField]], np.ndarray, float]:
    """Correct all timepoints with robust RBF fits to a fixed reference.

    New options:
    - center_mode: {'ref','pooled','kmeans_pooled'}
    - robust_mode: {'sample_trim','track_trim'}
    - two_stage: stage-1 global translation + residual RBF
    """
    d = ensure_numeric_tracks(df)
    times = np.array(sorted(d["Time"].unique()), dtype=float)
    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    centers = _choose_centers(
        d,
        reference_time=reference_time,
        m=int(m),
        center_mode=center_mode,
        seed=seed,
        fit_track_ids=fit_track_ids,
        pooled_time_stride=pooled_time_stride,
    )
    if centers.shape[0] == 0:
        # Last fallback: sample from all rows.
        all_pos = d[["Position X", "Position Y", "Position Z"]].to_numpy(float)
        centers = _choose_centers_random(all_pos, m=max(1, int(m)), seed=seed)

    if sigma is None:
        sigma = _default_sigma(centers)
    sigma = float(sigma)

    used_track_ids = fit_track_ids
    track_diag_rows: list[dict[str, float | int | bool | str]] = []

    if robust_mode == "track_trim":
        base_ids = _reference_track_ids(d, reference_time=reference_time, fit_track_ids=fit_track_ids)
        if base_ids.size:
            selected = np.array(base_ids, dtype=float)
            prev_set: set[float] | None = None
            n_iters = max(1, int(track_trim_iters))

            for it in range(n_iters):
                fields_tmp, _ = _fit_fields_once(
                    df=d,
                    times=times,
                    reference_time=reference_time,
                    centers=centers,
                    keep_fraction=keep_fraction,
                    lam=lam,
                    sigma=sigma,
                    min_pairs=min_pairs,
                    temporal_gamma=temporal_gamma,
                    fit_track_ids=selected,
                    robust_mode="track_trim",
                    center_mode=center_mode,
                    two_stage=two_stage,
                    lam_bias=0.0,
                )
                tab = _compute_track_trim_residuals(d, reference_time, fields_tmp, candidate_track_ids=base_ids)
                if tab.shape[0] == 0:
                    break

                k_keep = max(int(min_pairs), int(np.floor(float(np.clip(keep_fraction, 0.1, 0.99)) * tab.shape[0])))
                order = np.argsort(tab["track_residual_median"].to_numpy(float))
                keep_idx = order[:k_keep]

                keep_mask = np.zeros((tab.shape[0],), dtype=bool)
                keep_mask[keep_idx] = True
                tab = tab.copy()
                tab["selected_track"] = keep_mask
                tab["diag_type"] = "track_trim_track"
                tab["track_trim_iter"] = int(it + 1)
                for row in tab.to_dict(orient="records"):
                    track_diag_rows.append(row)

                selected_new = tab.loc[tab["selected_track"] == True, "TrackID"].to_numpy(float)
                cur_set = set(selected_new.tolist())
                if prev_set is not None and cur_set == prev_set:
                    selected = selected_new
                    break
                prev_set = cur_set
                selected = selected_new

            used_track_ids = np.array(sorted(prev_set if prev_set is not None else set(base_ids.tolist())), dtype=float)

    fields_by_time, tf_rows = _fit_fields_once(
        df=d,
        times=times,
        reference_time=reference_time,
        centers=centers,
        keep_fraction=keep_fraction,
        lam=lam,
        sigma=sigma,
        min_pairs=min_pairs,
        temporal_gamma=temporal_gamma,
        fit_track_ids=used_track_ids,
        robust_mode=robust_mode,
        center_mode=center_mode,
        two_stage=two_stage,
        lam_bias=0.0,
    )

    if robust_mode == "track_trim" and track_diag_rows:
        for row in track_diag_rows:
            tf_rows.append(row)

    df_corr = d.copy()
    df_corr["Position X_corr"] = np.nan
    df_corr["Position Y_corr"] = np.nan
    df_corr["Position Z_corr"] = np.nan

    mref = df_corr["Time"] == float(reference_time)
    df_corr.loc[mref, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = df_corr.loc[
        mref, ["Position X", "Position Y", "Position Z"]
    ].to_numpy(float)

    for t in times:
        t = float(t)
        if t == float(reference_time):
            continue
        field = fields_by_time.get(t)
        if field is None:
            continue
        mask_t = df_corr["Time"] == t
        x = df_corr.loc[mask_t, ["Position X", "Position Y", "Position Z"]].to_numpy(float)
        dhat = predict_rbf_field(field, x)
        xc = x - dhat
        df_corr.loc[mask_t, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = xc

    tf_table = pd.DataFrame(tf_rows)
    if "diag_type" not in tf_table.columns:
        tf_table["diag_type"] = "timepoint"
    if "t" in tf_table.columns:
        tf_table = tf_table.sort_values(["diag_type", "t"], na_position="last").reset_index(drop=True)

    return df_corr, tf_table, fields_by_time, times, reference_time


def make_rbf_diagnostic_plots(tf_table: pd.DataFrame, out_prefix: str | Path) -> None:
    """Save per-timepoint RBF diagnostics, including center coverage and track-trim stats."""
    out_prefix = str(out_prefix)
    if tf_table is None or tf_table.shape[0] == 0:
        return

    ttab = tf_table.copy()
    if "diag_type" in ttab.columns:
        ttab = ttab[ttab["diag_type"] == "timepoint"].copy()
    if ttab.shape[0] == 0:
        return

    ok = ttab["ok"] == True if "ok" in ttab.columns else np.ones(len(ttab), dtype=bool)
    if not np.any(ok):
        return

    if "t1" not in ttab.columns and "t" in ttab.columns:
        ttab["t1"] = ttab["t"]

    if "inlier_fraction" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "inlier_fraction"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Inlier fraction")
        plt.title("RBF drift fit: inlier fraction per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_inlier_fraction.png", dpi=300)
        plt.close(fig)

    fig = plt.figure()
    if "res_median" in ttab.columns:
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "res_median"], marker="o", label="median")
    if "temporal_res_median" in ttab.columns:
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "temporal_res_median"], marker="o", label="temporal median")
    if "res_p90" in ttab.columns:
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "res_p90"], marker="o", label="p90")
    plt.xlabel("Time (t)")
    plt.ylabel("Residual norm (um)")
    plt.title("RBF drift fit: residual statistics per timepoint")
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_rbf_residual_stats.png", dpi=300)
    plt.close(fig)

    if "sigma" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "sigma"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("sigma (um)")
        plt.title("RBF drift fit: sigma per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_sigma.png", dpi=300)
        plt.close(fig)

    if "lam" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "lam"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("lambda (ridge)")
        plt.title("RBF drift fit: ridge lambda per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_lambda.png", dpi=300)
        plt.close(fig)

    if "M" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "M"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of centers (M)")
        plt.title("RBF drift fit: number of centers per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_M.png", dpi=300)
        plt.close(fig)

    if "temporal_smoothness_prev" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "temporal_smoothness_prev"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Median step change in predicted drift (um)")
        plt.title("RBF temporal smoothness diagnostic")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_temporal_smoothness.png", dpi=300)
        plt.close(fig)

    if "center_nn_median" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "center_nn_median"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Median nearest-center distance (um)")
        plt.title("RBF center coverage over time")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_center_coverage.png", dpi=300)
        plt.close(fig)

    if "pred_drift_median" in ttab.columns:
        fig = plt.figure()
        plt.plot(ttab.loc[ok, "t1"], ttab.loc[ok, "pred_drift_median"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Median predicted drift magnitude (um)")
        plt.title("RBF predicted drift magnitude over time")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_pred_drift_median.png", dpi=300)
        plt.close(fig)

    if "diag_type" in tf_table.columns:
        tracks = tf_table[tf_table["diag_type"] == "track_trim_track"].copy()
        if tracks.shape[0] > 0 and "track_residual_median" in tracks.columns:
            if "selected_track" in tracks.columns:
                sel = tracks[tracks["selected_track"] == True]
                rem = tracks[tracks["selected_track"] != True]

                fig = plt.figure()
                if rem.shape[0] > 0:
                    plt.hist(rem["track_residual_median"].to_numpy(float), bins=40, alpha=0.4, label="removed")
                if sel.shape[0] > 0:
                    plt.hist(sel["track_residual_median"].to_numpy(float), bins=40, alpha=0.4, label="kept")
                plt.xlabel("Track residual median (um)")
                plt.ylabel("count")
                plt.title("RBF track-trim residuals")
                handles, _ = plt.gca().get_legend_handles_labels()
                if handles:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(out_prefix + "_rbf_track_trim_residual_hist.png", dpi=300)
                plt.close(fig)

                sel_ids = sel[["TrackID"]].drop_duplicates().sort_values("TrackID")
                sel_ids.to_csv(out_prefix + "_rbf_track_trim_selected_trackids.csv", index=False)
