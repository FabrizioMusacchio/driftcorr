"""Frame-to-frame assignment routines.

This module provides cost construction, gating, and Hungarian assignment for
matching points across consecutive frames.

Mathematical objective:
- Given cost matrix C in R^{N x M} and allowed mask G,
  solve min_{pi} sum_i C[i, pi(i)] over one-to-one assignments with forbidden
  edges excluded by G.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .costs import huberized_cost, mahalanobis_cost, sqeuclidean_cost
from .gating import apply_gate_to_cost, combine_gates, gate_by_max_cost, gate_by_max_distance, gate_by_percentile


def build_cost_matrix(
    x_t: np.ndarray,
    x_tp1: np.ndarray,
    metric: str = "sqeuclidean",
    cov: np.ndarray | None = None,
    huber_delta: float = 1.0,
) -> np.ndarray:
    """Construct pairwise cost matrix for association.

    Parameters
    ----------
    metric:
        One of `sqeuclidean`, `huber`, `mahalanobis`.
    """
    m = str(metric).lower()
    if m == "sqeuclidean":
        return sqeuclidean_cost(x_t, x_tp1)
    if m == "huber":
        return huberized_cost(x_t, x_tp1, delta=float(huber_delta))
    if m == "mahalanobis":
        return mahalanobis_cost(x_t, x_tp1, cov=cov)
    raise ValueError(f"Unknown metric '{metric}'")


def apply_gating(
    cost_matrix: np.ndarray,
    max_dist: float | None = None,
    max_cost: float | None = None,
    keep_percentile: float | None = None,
    assume_squared_cost: bool = True,
) -> np.ndarray:
    """Create allowed-edge mask by combining selected gates."""
    c = np.asarray(cost_matrix, dtype=float)
    gates: list[np.ndarray | None] = [np.isfinite(c)]
    if max_dist is not None:
        gates.append(gate_by_max_distance(c, max_dist=float(max_dist), assume_squared_cost=assume_squared_cost))
    if max_cost is not None:
        gates.append(gate_by_max_cost(c, max_cost=float(max_cost)))
    if keep_percentile is not None:
        gates.append(gate_by_percentile(c, keep_percentile=float(keep_percentile)))
    g = combine_gates(*gates)
    return np.asarray(g if g is not None else np.ones_like(c, dtype=bool), dtype=bool)


def hungarian_assignment(
    cost_matrix: np.ndarray,
    gating_mask: np.ndarray | None = None,
    forbidden_cost: float | None = None,
) -> dict[str, Any]:
    """Solve gated one-to-one assignment via Hungarian algorithm.

    Returns
    -------
    dict with keys:
    - row_ind, col_ind: matched indices after gating
    - costs: matched costs
    - unmatched_t, unmatched_tp1: unmatched source/destination indices
    - matched_mask: boolean assignment mask over all candidate row/col outputs
    - total_cost: sum of accepted match costs
    """
    c = np.asarray(cost_matrix, dtype=float)
    n, m = c.shape if c.ndim == 2 else (0, 0)
    if n == 0 or m == 0:
        return {
            "row_ind": np.array([], dtype=int),
            "col_ind": np.array([], dtype=int),
            "costs": np.array([], dtype=float),
            "matched_mask": np.array([], dtype=bool),
            "unmatched_t": np.arange(n, dtype=int),
            "unmatched_tp1": np.arange(m, dtype=int),
            "total_cost": 0.0,
        }

    g = np.asarray(gating_mask, dtype=bool) if gating_mask is not None else np.ones_like(c, dtype=bool)
    gated_cost = apply_gate_to_cost(c, g, forbidden_cost=forbidden_cost)

    rr, cc = linear_sum_assignment(gated_cost)
    valid = g[rr, cc] & np.isfinite(c[rr, cc])
    rr_ok = rr[valid]
    cc_ok = cc[valid]
    costs_ok = c[rr_ok, cc_ok]

    used_r = np.zeros((n,), dtype=bool)
    used_c = np.zeros((m,), dtype=bool)
    used_r[rr_ok] = True
    used_c[cc_ok] = True

    return {
        "row_ind": rr_ok.astype(int),
        "col_ind": cc_ok.astype(int),
        "costs": costs_ok.astype(float),
        "matched_mask": valid.astype(bool),
        "unmatched_t": np.flatnonzero(~used_r).astype(int),
        "unmatched_tp1": np.flatnonzero(~used_c).astype(int),
        "total_cost": float(np.sum(costs_ok)) if costs_ok.size else 0.0,
    }


def nearest_neighbor_assignment(
    cost_matrix: np.ndarray,
    gating_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Greedy nearest-neighbor one-to-one assignment baseline.

    This baseline is deterministic and useful for comparison against Hungarian.
    """
    c = np.asarray(cost_matrix, dtype=float)
    n, m = c.shape if c.ndim == 2 else (0, 0)
    if n == 0 or m == 0:
        return {
            "row_ind": np.array([], dtype=int),
            "col_ind": np.array([], dtype=int),
            "costs": np.array([], dtype=float),
            "matched_mask": np.array([], dtype=bool),
            "unmatched_t": np.arange(n, dtype=int),
            "unmatched_tp1": np.arange(m, dtype=int),
            "total_cost": 0.0,
        }
    g = np.asarray(gating_mask, dtype=bool) if gating_mask is not None else np.ones_like(c, dtype=bool)
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(m):
            if g[i, j] and np.isfinite(c[i, j]):
                edges.append((float(c[i, j]), int(i), int(j)))
    edges.sort(key=lambda x: (x[0], x[1], x[2]))

    used_i = np.zeros((n,), dtype=bool)
    used_j = np.zeros((m,), dtype=bool)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for cost, i, j in edges:
        if used_i[i] or used_j[j]:
            continue
        used_i[i] = True
        used_j[j] = True
        rows.append(i)
        cols.append(j)
        vals.append(cost)

    rr = np.asarray(rows, dtype=int)
    cc = np.asarray(cols, dtype=int)
    vv = np.asarray(vals, dtype=float)
    return {
        "row_ind": rr,
        "col_ind": cc,
        "costs": vv,
        "matched_mask": np.ones((rr.size,), dtype=bool),
        "unmatched_t": np.flatnonzero(~used_i).astype(int),
        "unmatched_tp1": np.flatnonzero(~used_j).astype(int),
        "total_cost": float(np.sum(vv)) if vv.size else 0.0,
    }
