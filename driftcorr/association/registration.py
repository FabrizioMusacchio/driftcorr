"""Registration helpers for drift-aware point-cloud association.

Strategy 1 uses pre-registration to reduce apparent drift before assignment:
- rigid Procrustes fit for known correspondences,
- ICP rigid fit with nearest-neighbor correspondences and trimming,
- pluggable non-rigid registrar interface (optional integrations).

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def procrustes_rigid(
    x: np.ndarray,
    y: np.ndarray,
    correspondences: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate rigid transform x -> y by SVD Procrustes.

    Parameters
    ----------
    x, y:
        Source/target point arrays (N,3).
    correspondences:
        Optional integer array shape (K,2), each row `(i,j)` linking x[i] to y[j].
        If None, x and y are matched by row index with K=min(Nx,Ny).

    Returns
    -------
    R, b, residuals:
        Rotation matrix, translation vector, per-pair Euclidean residuals.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if correspondences is None:
        k = min(xs.shape[0], ys.shape[0])
        idx_x = np.arange(k, dtype=int)
        idx_y = np.arange(k, dtype=int)
    else:
        cc = np.asarray(correspondences, dtype=int)
        if cc.ndim != 2 or cc.shape[1] != 2:
            raise ValueError("correspondences must have shape (K,2)")
        idx_x = cc[:, 0]
        idx_y = cc[:, 1]
    if idx_x.size < 3:
        return np.eye(3, dtype=float), np.zeros((3,), dtype=float), np.array([], dtype=float)

    xx = xs[idx_x]
    yy = ys[idx_y]
    mx = np.mean(xx, axis=0)
    my = np.mean(yy, axis=0)
    x0 = xx - mx
    y0 = yy - my

    h = x0.T @ y0
    u, _, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    b = my - (r @ mx)
    pred = (r @ xx.T).T + b[None, :]
    residuals = np.linalg.norm(pred - yy, axis=1)
    return r, b, residuals


def icp_rigid(
    x: np.ndarray,
    y: np.ndarray,
    n_iters: int = 20,
    max_corr_dist: float = np.inf,
    keep_fraction: float = 0.8,
    init: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trimmed ICP rigid registration x -> y.

    Returns
    -------
    R, b, correspondences, residuals
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return np.eye(3, dtype=float), np.zeros((3,), dtype=float), np.empty((0, 2), dtype=int), np.array([], dtype=float)

    if init is None:
        r_tot = np.eye(3, dtype=float)
        b_tot = np.zeros((3,), dtype=float)
    else:
        r_tot = np.asarray(init[0], dtype=float)
        b_tot = np.asarray(init[1], dtype=float)

    tree = cKDTree(ys)
    corr = np.empty((0, 2), dtype=int)
    residuals = np.array([], dtype=float)
    max_corr = float(max_corr_dist)
    keep = float(np.clip(keep_fraction, 1e-3, 1.0))

    for _ in range(max(1, int(n_iters))):
        x_cur = (r_tot @ xs.T).T + b_tot[None, :]
        d, j = tree.query(x_cur, k=1)
        d = np.asarray(d, dtype=float)
        j = np.asarray(j, dtype=int)

        valid = np.isfinite(d) & (d <= max_corr)
        if not np.any(valid):
            break
        idx = np.flatnonzero(valid)
        dist = d[idx]
        order = np.argsort(dist)
        k = max(3, int(np.floor(keep * float(order.size))))
        sel = idx[order[:k]]
        corr = np.column_stack([sel, j[sel]]).astype(int)

        r_inc, b_inc, residuals = procrustes_rigid(x_cur, ys, correspondences=corr)
        r_tot = r_inc @ r_tot
        b_tot = (r_inc @ b_tot) + b_inc

    return r_tot, b_tot, corr, residuals


@dataclass(slots=True)
class NonRigidRegistrar:
    """Pluggable interface for non-rigid registration backends."""

    name: str = "nonrigid_unavailable"

    def fit_transform(self, *_: Any, **__: Any) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError(
            "No non-rigid registrar is configured. Install and integrate an external backend (e.g. CPD)."
        )


def optional_cpd_registrar() -> NonRigidRegistrar:
    """Return optional CPD registrar if available, else raise clear error."""
    try:
        # Optional dependency example; not required by driftcorr defaults.
        from pycpd import DeformableRegistration  # type: ignore
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError("CPD backend not installed. Install `pycpd` to enable non-rigid registration.") from exc

    class _CPDRegistrar(NonRigidRegistrar):
        name = "cpd"

        def fit_transform(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            reg = DeformableRegistration(X=np.asarray(y, dtype=float), Y=np.asarray(x, dtype=float), **kwargs)
            ty, _ = reg.register()
            return np.asarray(ty, dtype=float), {"backend": "pycpd", "kwargs": dict(kwargs)}

    return _CPDRegistrar()
