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

Correction:
- `X_corr(t) = X(t) - d_hat_t(X(t))`.

Controlling parameters:
- `m`: number of RBF centers (capacity).
- `sigma`: kernel width (spatial smoothness scale).
- `lam`: ridge strength (smoother/smaller weights when larger).
- `keep_fraction`: robust trimming fraction.
- `temporal_gamma`: inter-time smoothness coupling.

Effects:
- Larger `sigma` and `lam` increase smoothness and reduce overfitting.
- Larger `temporal_gamma` enforces slower temporal changes in drift fields.

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
    """RBF drift field parameters."""

    centers: np.ndarray
    sigma: float
    W: np.ndarray
    lam: float


def _rbf_kernel(x: np.ndarray, c: np.ndarray, sigma: float) -> np.ndarray:
    diff = x[:, None, :] - c[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    return np.exp(-r2 / (2.0 * sigma * sigma))


def _choose_centers(x: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if x.shape[0] <= m:
        return x.copy()
    idx = rng.choice(x.shape[0], size=m, replace=False)
    return x[idx].copy()


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


def _ridge_solve(k: np.ndarray, d: np.ndarray, lam: float) -> np.ndarray:
    m = k.shape[1]
    a = k.T @ k
    a.flat[:: m + 1] += lam
    b = k.T @ d
    return np.linalg.solve(a, b)


def _stable_solve(a: np.ndarray, b: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    """Solve linear system with tiny diagonal fallback for numerical stability."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        aa = a.copy()
        aa.flat[:: aa.shape[0] + 1] += jitter
        return np.linalg.solve(aa, b)


def robust_fit_rbf_field(
    x: np.ndarray,
    d: np.ndarray,
    m: int = 25,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    max_iters: int = 25,
    seed: int = 0,
) -> tuple[RBFField, np.ndarray, np.ndarray]:
    """Fit D ~= K(X,C)W with trimmed least squares."""
    n = int(x.shape[0])
    if n < 8:
        raise ValueError("Too few correspondences for RBF field fit.")

    keep_fraction = float(np.clip(keep_fraction, 0.1, 0.99))
    k_keep = max(8, int(np.floor(keep_fraction * n)))

    c = _choose_centers(x, m=m, seed=seed)
    if sigma is None:
        sigma = _default_sigma(c)
    sigma = float(sigma)

    k = _rbf_kernel(x, c, sigma)
    w = _ridge_solve(k, d, lam=lam)

    prev = np.inf
    inlier_idx = np.arange(n)

    for _ in range(int(max_iters)):
        dp = k @ w
        res = np.linalg.norm(dp - d, axis=1)
        order = np.argsort(res)
        inlier_idx = order[:k_keep]

        ki = k[inlier_idx]
        di = d[inlier_idx]
        wi = _ridge_solve(ki, di, lam=lam)

        dp_i = ki @ wi
        cost = float(np.mean(np.sum((dp_i - di) ** 2, axis=1)))
        if abs(prev - cost) < 1e-12:
            w = wi
            break
        w = wi
        prev = cost

    dp = k @ w
    res = np.linalg.norm(dp - d, axis=1)

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idx] = True

    field = RBFField(centers=c, sigma=float(sigma), W=w, lam=float(lam))
    return field, inlier_mask, res


def predict_rbf_field(field: RBFField, x: np.ndarray) -> np.ndarray:
    """Predict drift vectors at positions x."""
    k = _rbf_kernel(x, field.centers, field.sigma)
    return k @ field.W


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


def _solve_temporal_block_tridiagonal(
    diag_blocks: list[np.ndarray],
    off_blocks: list[np.ndarray],
    rhs_blocks: list[np.ndarray],
) -> list[np.ndarray]:
    """Solve block-tridiagonal system for W(t) with RHS shape (M,3) per time."""
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


def fit_temporal_rbf_fields_to_reference(
    df: pd.DataFrame,
    times: np.ndarray,
    reference_time: float,
    m: int = 25,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    seed: int = 0,
    temporal_gamma: float = 0.0,
    fit_track_ids: np.ndarray | None = None,
) -> tuple[dict[float, RBFField], dict[float, dict[str, float | int | bool | str]]]:
    """Fit all non-reference timepoint RBF fields jointly with temporal smoothness."""
    rows_by_time: dict[float, dict[str, float | int | bool | str]] = {}
    fields_by_time: dict[float, RBFField] = {}

    ref_df = df[df["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        ref_df = ref_df[ref_df["TrackID"].astype(float).isin(ids)]
    x_ref = ref_df[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    if x_ref.shape[0] == 0:
        return fields_by_time, rows_by_time

    centers = _choose_centers(x_ref, m=m, seed=seed)
    if sigma is None:
        sigma = _default_sigma(centers)
    sigma = float(sigma)

    valid_times: list[float] = []
    k_blocks: list[np.ndarray] = []
    x_blocks: list[np.ndarray] = []
    d_blocks: list[np.ndarray] = []

    for t in times:
        t = float(t)
        if t == float(reference_time):
            continue
        mdf = _merge_with_reference(df, t=t, tref=reference_time, fit_track_ids=fit_track_ids)
        n = int(mdf.shape[0])
        if n < int(min_pairs):
            rows_by_time[t] = {
                "ok": False,
                "t": t,
                "tref": float(reference_time),
                "n_pairs": n,
                "reason": "Too few pairs",
                "temporal_gamma": float(temporal_gamma),
            }
            continue

        xt = mdf[["Position X_t", "Position Y_t", "Position Z_t"]].to_numpy(float)
        xr = mdf[["Position X_ref", "Position Y_ref", "Position Z_ref"]].to_numpy(float)
        d = xt - xr
        k = _rbf_kernel(xt, centers, sigma)

        valid_times.append(t)
        x_blocks.append(xt)
        d_blocks.append(d)
        k_blocks.append(k)

    t_count = len(valid_times)
    if t_count == 0:
        return fields_by_time, rows_by_time

    m_count = centers.shape[0]
    ident = np.eye(m_count, dtype=float)

    diag_blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    for i in range(t_count):
        k = k_blocks[i]
        a = (k.T @ k) + (float(lam) * ident)
        if temporal_gamma > 0:
            if i == 0 or i == t_count - 1:
                a = a + float(temporal_gamma) * ident
            else:
                a = a + 2.0 * float(temporal_gamma) * ident
        diag_blocks.append(a)
        rhs_blocks.append(k.T @ d_blocks[i])

    off_blocks: list[np.ndarray] = []
    if temporal_gamma > 0:
        for _ in range(t_count - 1):
            off_blocks.append(-float(temporal_gamma) * ident)
    else:
        for _ in range(t_count - 1):
            off_blocks.append(np.zeros_like(ident))

    w_blocks = _solve_temporal_block_tridiagonal(diag_blocks, off_blocks, rhs_blocks)

    k_cc = _rbf_kernel(centers, centers, sigma)
    drift_centers = [k_cc @ w for w in w_blocks]

    smoothness_prev: dict[int, float] = {}
    if len(drift_centers) > 1:
        for i in range(1, len(drift_centers)):
            step_norm = np.linalg.norm(drift_centers[i] - drift_centers[i - 1], axis=1)
            smoothness_prev[i] = float(np.median(step_norm))

    for i, t in enumerate(valid_times):
        w = w_blocks[i]
        field = RBFField(centers=centers.copy(), sigma=sigma, W=w, lam=float(lam))
        fields_by_time[t] = field

        d_pred = k_blocks[i] @ w
        res = np.linalg.norm(d_pred - d_blocks[i], axis=1)
        rows_by_time[t] = {
            "ok": True,
            "t": float(t),
            "tref": float(reference_time),
            "n_pairs": int(d_blocks[i].shape[0]),
            "inlier_fraction": 1.0,
            "res_median": float(np.median(res)),
            "res_p90": float(np.percentile(res, 90)),
            "sigma": float(sigma),
            "lam": float(lam),
            "M": int(m_count),
            "temporal_gamma": float(temporal_gamma),
            "temporal_res_median": float(np.median(res)),
            "temporal_smoothness_prev": float(smoothness_prev.get(i, np.nan)),
        }

    return fields_by_time, rows_by_time


def estimate_rbf_field_to_reference(
    df: pd.DataFrame,
    t: float,
    tref: float,
    m: int = 25,
    keep_fraction: float = 0.7,
    lam: float = 1e-2,
    sigma: float | None = None,
    min_pairs: int = 15,
    seed: int = 0,
    fit_track_ids: np.ndarray | None = None,
) -> tuple[RBFField | None, dict[str, float | int | bool | str]]:
    """Fit a drift field for one timepoint against the reference time."""
    mdf = _merge_with_reference(df, t=t, tref=tref, fit_track_ids=fit_track_ids)

    n = int(mdf.shape[0])
    if n < min_pairs:
        info: dict[str, float | int | bool | str] = {
            "ok": False,
            "t": float(t),
            "tref": float(tref),
            "n_pairs": n,
            "reason": "Too few pairs",
        }
        return None, info

    xt = mdf[["Position X_t", "Position Y_t", "Position Z_t"]].to_numpy(float)
    xr = mdf[["Position X_ref", "Position Y_ref", "Position Z_ref"]].to_numpy(float)
    d = xt - xr

    field, inlier_mask, res = robust_fit_rbf_field(
        xt,
        d,
        m=m,
        keep_fraction=keep_fraction,
        lam=lam,
        sigma=sigma,
        seed=seed,
    )

    info = {
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
    }
    return field, info


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, Optional[RBFField]], np.ndarray, float]:
    """Correct all timepoints with per-timepoint or temporally-smoothed RBF fits."""
    df = ensure_numeric_tracks(df)
    times = np.array(sorted(df["Time"].unique()), dtype=float)
    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    df_corr = df.copy()
    df_corr["Position X_corr"] = np.nan
    df_corr["Position Y_corr"] = np.nan
    df_corr["Position Z_corr"] = np.nan

    mref = df_corr["Time"] == reference_time
    df_corr.loc[mref, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = df_corr.loc[
        mref, ["Position X", "Position Y", "Position Z"]
    ].to_numpy(float)

    tf_rows: list[dict[str, float | int | bool | str]] = []
    fields_by_time: dict[float, Optional[RBFField]] = {reference_time: None}

    tf_rows.append(
        {
            "ok": True,
            "t": float(reference_time),
            "tref": float(reference_time),
            "n_pairs": np.nan,
            "inlier_fraction": np.nan,
            "res_median": 0.0,
            "res_p90": 0.0,
            "sigma": np.nan,
            "lam": float(lam),
            "M": int(m),
            "temporal_gamma": float(temporal_gamma),
            "temporal_res_median": 0.0,
            "temporal_smoothness_prev": np.nan,
        }
    )

    if float(temporal_gamma) > 0.0:
        temporal_fields, temporal_rows = fit_temporal_rbf_fields_to_reference(
            df=df_corr,
            times=times,
            reference_time=reference_time,
            m=m,
            lam=lam,
            sigma=sigma,
            min_pairs=min_pairs,
            seed=seed,
            temporal_gamma=float(temporal_gamma),
            fit_track_ids=fit_track_ids,
        )
        for t in times:
            t = float(t)
            if t == reference_time:
                continue
            row = temporal_rows.get(
                t,
                {
                    "ok": False,
                    "t": t,
                    "tref": float(reference_time),
                    "n_pairs": 0,
                    "reason": "No temporal fit row",
                    "temporal_gamma": float(temporal_gamma),
                },
            )
            tf_rows.append(row)
            field = temporal_fields.get(t)
            if field is None:
                continue
            fields_by_time[t] = field
            mask_t = df_corr["Time"] == t
            x = df_corr.loc[mask_t, ["Position X", "Position Y", "Position Z"]].to_numpy(float)
            dhat = predict_rbf_field(field, x)
            xc = x - dhat
            df_corr.loc[mask_t, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = xc
    else:
        for i, t in enumerate(times):
            t = float(t)
            if t == reference_time:
                continue

            field, info = estimate_rbf_field_to_reference(
                df_corr,
                t=t,
                tref=reference_time,
                m=m,
                keep_fraction=keep_fraction,
                lam=lam,
                sigma=sigma,
                min_pairs=min_pairs,
                seed=seed + i,
                fit_track_ids=fit_track_ids,
            )
            info["temporal_gamma"] = float(temporal_gamma)
            info.setdefault("temporal_res_median", info.get("res_median", np.nan))
            info.setdefault("temporal_smoothness_prev", np.nan)
            tf_rows.append(info)
            if not info["ok"]:
                continue

            assert field is not None
            fields_by_time[t] = field

            mask_t = df_corr["Time"] == t
            x = df_corr.loc[mask_t, ["Position X", "Position Y", "Position Z"]].to_numpy(float)
            dhat = predict_rbf_field(field, x)
            xc = x - dhat
            df_corr.loc[mask_t, ["Position X_corr", "Position Y_corr", "Position Z_corr"]] = xc

    tf_table = pd.DataFrame(tf_rows)
    if "t" in tf_table.columns:
        tf_table = tf_table.sort_values("t").reset_index(drop=True)
    return df_corr, tf_table, fields_by_time, times, reference_time


def make_rbf_diagnostic_plots(tf_table: pd.DataFrame, out_prefix: str | Path) -> None:
    """Save per-timepoint RBF fit diagnostics."""
    out_prefix = str(out_prefix)
    if tf_table is None or tf_table.shape[0] == 0:
        return

    ok = tf_table["ok"] == True if "ok" in tf_table.columns else np.ones(len(tf_table), dtype=bool)
    if not np.any(ok):
        return

    if "t1" not in tf_table.columns and "t" in tf_table.columns:
        tf_table = tf_table.copy()
        tf_table["t1"] = tf_table["t"]

    if "inlier_fraction" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "inlier_fraction"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Inlier fraction")
        plt.title("RBF drift fit: inlier fraction per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_inlier_fraction.png", dpi=300)
        plt.close(fig)

    fig = plt.figure()
    if "res_median" in tf_table.columns:
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "res_median"], marker="o", label="median")
    if "temporal_res_median" in tf_table.columns:
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "temporal_res_median"], marker="o", label="temporal median")
    if "res_p90" in tf_table.columns:
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "res_p90"], marker="o", label="p90")
    plt.xlabel("Time (t)")
    plt.ylabel("Residual norm (um)")
    plt.title("RBF drift fit: residual statistics per timepoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_rbf_residual_stats.png", dpi=300)
    plt.close(fig)

    if "sigma" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "sigma"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("sigma (um)")
        plt.title("RBF drift fit: sigma per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_sigma.png", dpi=300)
        plt.close(fig)

    if "lam" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "lam"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("lambda (ridge)")
        plt.title("RBF drift fit: ridge lambda per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_lambda.png", dpi=300)
        plt.close(fig)

    if "M" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "M"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Number of centers (M)")
        plt.title("RBF drift fit: number of centers per timepoint")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_M.png", dpi=300)
        plt.close(fig)

    if "temporal_smoothness_prev" in tf_table.columns:
        fig = plt.figure()
        plt.plot(tf_table.loc[ok, "t1"], tf_table.loc[ok, "temporal_smoothness_prev"], marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("Median step change in predicted drift (um)")
        plt.title("RBF temporal smoothness diagnostic")
        plt.tight_layout()
        plt.savefig(out_prefix + "_rbf_temporal_smoothness.png", dpi=300)
        plt.close(fig)
