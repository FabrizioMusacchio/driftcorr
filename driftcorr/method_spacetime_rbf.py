"""Joint space-time RBF drift model in 4D input space.

This method fits a single smooth drift function over both spatial coordinates
and time:
- input `u = (x, y, z, t)`,
- output `d(u)` in 3D.

Training targets are drift vectors relative to a reference state:
`d_target = x_obs - x_ref(track)`.

Model:
- `d(u) ≈ K((x,y,z,t), C) W` with Gaussian kernel
  `K = exp(-||Δx||^2/(2*ell_space^2) - Δt^2/(2*ell_time^2))`.
- Ridge regularization:
  `min_W ||K W - D||^2 + lambda ||W||^2`.
- Robustness by iterative trimming/reweighting of high-residual samples.

Correction:
- `x_corr = x_obs - d_hat(x_obs, t)`.

Controlling parameters:
- `m_centers`: model capacity.
- `ell_space`, `ell_time`: smoothness scales in space and time.
- `lam`: regularization strength.
- `robust_keep_fraction`, `robust_iters`: outlier resistance.

Effects:
- Larger `ell_space/ell_time` impose smoother fields.
- Smaller `robust_keep_fraction` ignores more high-motion samples.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_imaris import ensure_numeric_tracks


@dataclass
class SpaceTimeRBFModel:
    """Space-time RBF drift model."""

    centers: np.ndarray
    weights: np.ndarray
    ell_space: float
    ell_time: float
    lam: float
    reference_time: float
    robust_keep_fraction: float


def _kernel_spacetime(
    u: np.ndarray,
    c: np.ndarray,
    ell_space: float,
    ell_time: float,
) -> np.ndarray:
    diff_space = u[:, None, :3] - c[None, :, :3]
    d2_space = np.sum(diff_space * diff_space, axis=2)
    dt2 = (u[:, None, 3] - c[None, :, 3]) ** 2
    return np.exp(-(d2_space / (2.0 * ell_space * ell_space) + dt2 / (2.0 * ell_time * ell_time)))


def _choose_centers(u: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if u.shape[0] <= m:
        return u.copy()
    idx = rng.choice(u.shape[0], size=m, replace=False)
    return u[idx].copy()


def _weighted_ridge(k: np.ndarray, d: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    sw = np.sqrt(np.clip(w, 0.0, np.inf))[:, None]
    kw = k * sw
    dw = d * sw
    m = k.shape[1]
    a = kw.T @ kw
    a.flat[:: m + 1] += lam
    b = kw.T @ dw
    return np.linalg.solve(a, b)


def _build_training_samples(
    df: pd.DataFrame,
    reference_time: float,
    fit_track_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    d = ensure_numeric_tracks(df)
    if fit_track_ids is not None:
        ids = set(np.asarray(fit_track_ids, dtype=float).tolist())
        d = d[d["TrackID"].astype(float).isin(ids)].copy()

    ref = d[d["Time"] == float(reference_time)][["TrackID", "Position X", "Position Y", "Position Z"]].copy()
    ref = ref.rename(columns={"Position X": "X_ref", "Position Y": "Y_ref", "Position Z": "Z_ref"})

    mdf = d.merge(ref, on="TrackID", how="inner")

    u = mdf[["Position X", "Position Y", "Position Z", "Time"]].to_numpy(float)
    drift = np.column_stack(
        [
            mdf["Position X"].to_numpy(float) - mdf["X_ref"].to_numpy(float),
            mdf["Position Y"].to_numpy(float) - mdf["Y_ref"].to_numpy(float),
            mdf["Position Z"].to_numpy(float) - mdf["Z_ref"].to_numpy(float),
        ]
    )
    return u, drift


def fit_spacetime_rbf_model(
    df: pd.DataFrame,
    reference_time: float | None = None,
    m_centers: int = 40,
    ell_space: float = 120.0,
    ell_time: float = 5.0,
    lam: float = 1e-2,
    robust_keep_fraction: float = 0.7,
    robust_iters: int = 4,
    seed: int = 0,
    fit_track_ids: np.ndarray | None = None,
) -> tuple[SpaceTimeRBFModel, dict[str, float | int]]:
    """Fit smooth space-time RBF drift model with robust trimming."""
    d = ensure_numeric_tracks(df)
    times = np.array(sorted(d["Time"].unique()), dtype=float)
    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    u, drift = _build_training_samples(d, reference_time=reference_time, fit_track_ids=fit_track_ids)
    if u.shape[0] < max(20, m_centers):
        raise ValueError("Too few training samples for space-time RBF fit.")

    centers = _choose_centers(u, m=m_centers, seed=seed)
    k = _kernel_spacetime(u, centers, ell_space=ell_space, ell_time=ell_time)

    w = np.ones((u.shape[0],), dtype=float)
    keep_fraction = float(np.clip(robust_keep_fraction, 0.1, 0.99))
    k_keep = max(20, int(np.floor(keep_fraction * u.shape[0])))

    weights = _weighted_ridge(k, drift, w, lam=lam)
    for _ in range(max(1, int(robust_iters))):
        pred = k @ weights
        res = np.linalg.norm(pred - drift, axis=1)
        idx = np.argsort(res)[:k_keep]
        w = np.zeros_like(w)
        w[idx] = 1.0
        weights = _weighted_ridge(k, drift, w, lam=lam)

    pred = k @ weights
    res = np.linalg.norm(pred - drift, axis=1)

    model = SpaceTimeRBFModel(
        centers=centers,
        weights=weights,
        ell_space=float(ell_space),
        ell_time=float(ell_time),
        lam=float(lam),
        reference_time=float(reference_time),
        robust_keep_fraction=float(keep_fraction),
    )

    info: dict[str, float | int] = {
        "n_samples": int(u.shape[0]),
        "n_centers": int(centers.shape[0]),
        "res_median": float(np.median(res)),
        "res_p90": float(np.percentile(res, 90)),
        "ell_space": float(ell_space),
        "ell_time": float(ell_time),
        "lam": float(lam),
        "robust_keep_fraction": float(keep_fraction),
        "robust_iters": int(robust_iters),
    }
    return model, info


def predict_spacetime_drift(model: SpaceTimeRBFModel, u: np.ndarray) -> np.ndarray:
    """Predict drift vectors at input space-time coordinates u=(x,y,z,t)."""
    k = _kernel_spacetime(u, model.centers, model.ell_space, model.ell_time)
    return k @ model.weights


def apply_correction(df: pd.DataFrame, model: SpaceTimeRBFModel) -> pd.DataFrame:
    """Apply space-time RBF correction to track dataframe."""
    d = ensure_numeric_tracks(df)
    u = d[["Position X", "Position Y", "Position Z", "Time"]].to_numpy(float)
    drift_pred = predict_spacetime_drift(model, u)

    xyz = d[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    corr = xyz - drift_pred

    d["Position X_corr"] = corr[:, 0]
    d["Position Y_corr"] = corr[:, 1]
    d["Position Z_corr"] = corr[:, 2]
    return d
