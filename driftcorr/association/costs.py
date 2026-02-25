"""Cost functions for frame-to-frame point association.

The assignment objective uses a pairwise cost matrix ``C`` where ``C[i, j]``
measures how likely point ``i`` at time ``t`` corresponds to point ``j`` at
time ``t+1``.

Implemented families:
- squared Euclidean and Euclidean distance,
- Huber robust cost,
- Mahalanobis distance (full covariance or diagonal scaling).

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

import numpy as np


def _validate_xyz(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape[1] != 3 or bb.shape[1] != 3:
        raise ValueError("x_t and x_tp1 must have shape (N,3)/(M,3)")
    return aa, bb


def cost_sqeuclidean(x_t: np.ndarray, x_tp1: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean pairwise costs.

    C[i, j] = ||x_t[i] - x_tp1[j]||_2^2
    """
    a, b = _validate_xyz(x_t, x_tp1)
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def cost_euclidean(x_t: np.ndarray, x_tp1: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances."""
    return np.sqrt(np.maximum(cost_sqeuclidean(x_t, x_tp1), 0.0))


def cost_huber(x_t: np.ndarray, x_tp1: np.ndarray, delta: float = 1.0) -> np.ndarray:
    r"""Compute robust Huberized distance cost.

    Let ``d = ||x_i - y_j||_2`` and ``\delta > 0``.
    ``rho(d) = 0.5 d^2`` if ``d <= delta``, else
    ``rho(d) = delta * (d - 0.5 * delta)``.
    """
    delta = max(float(delta), 1e-9)
    sq = cost_sqeuclidean(x_t, x_tp1)
    d = np.sqrt(np.maximum(sq, 0.0))
    quad = 0.5 * d * d
    lin = delta * (d - 0.5 * delta)
    return np.where(d <= delta, quad, lin)


def cost_mahalanobis(
    x_t: np.ndarray,
    x_tp1: np.ndarray,
    cov: np.ndarray | None = None,
    scales: tuple[float, float, float] | list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute Mahalanobis pairwise costs with covariance matrix.

    C[i, j] = (x_i - y_j)^T Sigma^{-1} (x_i - y_j)

    Parameters
    ----------
    cov:
        3x3 covariance. If None, identity is used.
    """
    a, b = _validate_xyz(x_t, x_tp1)
    if scales is not None:
        s = np.asarray(scales, dtype=float).reshape(-1)
        if s.size != 3:
            raise ValueError("scales must contain 3 positive values for x/y/z")
        s = np.maximum(np.abs(s), 1e-9)
        cov = np.diag(s * s)
    if cov is None:
        cov = np.eye(3, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError(f"cov must have shape (3,3), got {cov.shape}")
    cov_inv = np.linalg.pinv(cov)
    diff = a[:, None, :] - b[None, :, :]
    return np.einsum("...i,ij,...j->...", diff, cov_inv, diff)


# Backward-compatible wrappers used by older call sites.
def sqeuclidean_cost(x_t: np.ndarray, x_tp1: np.ndarray) -> np.ndarray:
    return cost_sqeuclidean(x_t, x_tp1)


def huberized_cost(x_t: np.ndarray, x_tp1: np.ndarray, delta: float = 1.0) -> np.ndarray:
    return cost_huber(x_t, x_tp1, delta=delta)


def mahalanobis_cost(x_t: np.ndarray, x_tp1: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
    return cost_mahalanobis(x_t, x_tp1, cov=cov)
