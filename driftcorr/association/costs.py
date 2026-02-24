"""Cost functions for frame-to-frame point association.

The assignment objective uses a pairwise cost matrix C where C[i, j] measures
how likely point i at time t corresponds to point j at time t+1.

Implemented costs:
- squared Euclidean distance,
- Huberized robust distance,
- Mahalanobis distance using an inverse covariance metric.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

import numpy as np


def sqeuclidean_cost(x_t: np.ndarray, x_tp1: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean pairwise costs.

    C[i, j] = ||x_t[i] - x_tp1[j]||_2^2
    """
    a = np.asarray(x_t, dtype=float)
    b = np.asarray(x_tp1, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("x_t and x_tp1 must have shape (N,3)/(M,3)")
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def huberized_cost(x_t: np.ndarray, x_tp1: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Compute robust Huberized distance cost.

    Let d = ||x_i - y_j||_2 and \delta > 0.
    rho(d) = 0.5 d^2                  if d <= delta
             = delta*(d - 0.5*delta)  otherwise
    """
    delta = max(float(delta), 1e-9)
    sq = sqeuclidean_cost(x_t, x_tp1)
    d = np.sqrt(np.maximum(sq, 0.0))
    quad = 0.5 * d * d
    lin = delta * (d - 0.5 * delta)
    return np.where(d <= delta, quad, lin)


def mahalanobis_cost(x_t: np.ndarray, x_tp1: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
    """Compute Mahalanobis pairwise costs with covariance matrix.

    C[i, j] = (x_i - y_j)^T Sigma^{-1} (x_i - y_j)

    Parameters
    ----------
    cov:
        3x3 covariance. If None, identity is used.
    """
    a = np.asarray(x_t, dtype=float)
    b = np.asarray(x_tp1, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("x_t and x_tp1 must have shape (N,3)/(M,3)")
    if cov is None:
        cov = np.eye(3, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError(f"cov must have shape (3,3), got {cov.shape}")
    cov_inv = np.linalg.pinv(cov)
    diff = a[:, None, :] - b[None, :, :]
    return np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
