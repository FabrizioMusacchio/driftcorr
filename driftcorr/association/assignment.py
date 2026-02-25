"""Frame-to-frame assignment routines.

This module provides cost construction, gating, Hungarian assignment, and
nearest-neighbor baselines for matching points across consecutive frames.

Mathematical objective:
- Given cost matrix C in R^{N x M} and allowed mask G,
  solve min_{pi} sum_i C[i, pi(i)] over one-to-one assignments with forbidden
  edges excluded by G.

The module also supports explicit unmatched handling via dummy nodes, which
prevents forced links in ambiguous settings.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .costs import cost_euclidean, cost_huber, cost_mahalanobis, cost_sqeuclidean
from .gating import apply_gate_to_cost, combine_gates, gate_by_max_cost, gate_by_max_distance, gate_by_percentile


def build_cost_matrix(
    x_t: np.ndarray,
    x_tp1: np.ndarray,
    metric: str = "sqeuclidean",
    cov: np.ndarray | None = None,
    huber_delta: float = 1.0,
    mahalanobis_scales: tuple[float, float, float] | list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Construct pairwise cost matrix for association.

    Parameters
    ----------
    metric:
        One of ``sqeuclidean``, ``euclidean``, ``huber``, ``mahalanobis``,
        ``mahalanobis_diag``.
    """
    m = str(metric).lower()
    if m == "sqeuclidean":
        return cost_sqeuclidean(x_t, x_tp1)
    if m == "euclidean":
        return cost_euclidean(x_t, x_tp1)
    if m == "huber":
        return cost_huber(x_t, x_tp1, delta=float(huber_delta))
    if m == "mahalanobis":
        return cost_mahalanobis(x_t, x_tp1, cov=cov)
    if m == "mahalanobis_diag":
        return cost_mahalanobis(x_t, x_tp1, cov=None, scales=mahalanobis_scales)
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


def _default_forbidden_cost(cost_matrix: np.ndarray) -> float:
    c = np.asarray(cost_matrix, dtype=float)
    finite = c[np.isfinite(c)]
    base = float(np.nanmax(finite)) if finite.size else 1.0
    return max(1e9, base * 1e6 + 1.0)


def _default_miss_cost(cost_matrix: np.ndarray) -> float:
    c = np.asarray(cost_matrix, dtype=float)
    finite = c[np.isfinite(c)]
    if finite.size == 0:
        return 1.0
    q = float(np.nanpercentile(finite, 95.0))
    med = float(np.nanmedian(finite))
    return max(q, med, 1e-6)


def hungarian_assignment(
    cost_matrix: np.ndarray,
    gating_mask: np.ndarray | None = None,
    forbidden_cost: float | None = None,
    use_dummy: bool = False,
    miss_cost: float | None = None,
    birth_cost: float | None = None,
    death_cost: float | None = None,
) -> dict[str, Any]:
    """Solve gated one-to-one assignment via Hungarian algorithm.

    Parameters
    ----------
    use_dummy:
        If True, augment the assignment with dummy nodes so that points can be
        left unmatched with configurable penalties (`birth_cost`/`death_cost`).
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
    if g.shape != c.shape:
        raise ValueError(f"gating_mask shape {g.shape} must match cost_matrix shape {c.shape}")

    if not bool(use_dummy):
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

    inf_cost = float(forbidden_cost) if forbidden_cost is not None else _default_forbidden_cost(c)
    miss = float(_default_miss_cost(c) if miss_cost is None else miss_cost)
    bcost = float(miss if birth_cost is None else birth_cost)
    dcost = float(miss if death_cost is None else death_cost)

    big = np.full((n + m, n + m), inf_cost, dtype=float)
    # Top-left: real edges.
    real = c.copy()
    real[~g] = inf_cost
    real[~np.isfinite(real)] = inf_cost
    big[:n, :m] = real
    # Top-right: source unmatched (death).
    for i in range(n):
        big[i, m + i] = dcost
    # Bottom-left: destination unmatched (birth).
    for j in range(m):
        big[n + j, j] = bcost
    # Bottom-right: dummy<->dummy zero-fill.
    big[n:, m:] = 0.0

    rr, cc = linear_sum_assignment(big)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    unmatched_t: list[int] = []
    unmatched_tp1: list[int] = []
    total_cost = 0.0

    for r, col in zip(rr.tolist(), cc.tolist()):
        if r < n and col < m:
            if g[r, col] and np.isfinite(c[r, col]) and big[r, col] < inf_cost:
                rows.append(int(r))
                cols.append(int(col))
                cv = float(c[r, col])
                vals.append(cv)
                total_cost += cv
            else:
                unmatched_t.append(int(r))
                unmatched_tp1.append(int(col))
        elif r < n and col >= m:
            unmatched_t.append(int(r))
            total_cost += dcost
        elif r >= n and col < m:
            unmatched_tp1.append(int(col))
            total_cost += bcost

    # Ensure no duplicates and sorted output for deterministic behavior.
    unmatched_t_arr = np.unique(np.asarray(unmatched_t, dtype=int))
    unmatched_tp1_arr = np.unique(np.asarray(unmatched_tp1, dtype=int))

    # Remove potentially matched indices from unmatched arrays.
    if rows:
        matched_r = np.unique(np.asarray(rows, dtype=int))
        unmatched_t_arr = np.setdiff1d(unmatched_t_arr, matched_r, assume_unique=False)
    if cols:
        matched_c = np.unique(np.asarray(cols, dtype=int))
        unmatched_tp1_arr = np.setdiff1d(unmatched_tp1_arr, matched_c, assume_unique=False)

    return {
        "row_ind": np.asarray(rows, dtype=int),
        "col_ind": np.asarray(cols, dtype=int),
        "costs": np.asarray(vals, dtype=float),
        "matched_mask": np.ones((len(rows),), dtype=bool),
        "unmatched_t": unmatched_t_arr.astype(int),
        "unmatched_tp1": unmatched_tp1_arr.astype(int),
        "total_cost": float(total_cost),
    }


def nearest_neighbor_assignment(
    cost_matrix: np.ndarray,
    gating_mask: np.ndarray | None = None,
    use_dummy: bool = False,
    miss_cost: float | None = None,
) -> dict[str, Any]:
    """Greedy nearest-neighbor one-to-one assignment baseline.

    This baseline is deterministic and useful for comparison against Hungarian.
    With `use_dummy=True`, rows may remain unmatched if best edge cost exceeds
    `miss_cost`.
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

    miss = float(_default_miss_cost(c) if miss_cost is None else miss_cost)
    used_i = np.zeros((n,), dtype=bool)
    used_j = np.zeros((m,), dtype=bool)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Deterministic row-wise nearest neighbor assignment with one-to-one usage.
    for i in range(n):
        cand_j = np.flatnonzero(g[i] & np.isfinite(c[i]) & (~used_j))
        if cand_j.size == 0:
            continue
        j_loc = cand_j[np.argmin(c[i, cand_j])]
        best = float(c[i, j_loc])
        if bool(use_dummy) and best > miss:
            continue
        used_i[i] = True
        used_j[j_loc] = True
        rows.append(int(i))
        cols.append(int(j_loc))
        vals.append(best)

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
