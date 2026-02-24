"""Gating utilities for association constraints.

Gating marks implausible correspondences as forbidden before assignment.
Given a cost matrix C in R^{N x M}, we construct a boolean mask G where
G[i, j] = True means edge (i,j) is allowed.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

import numpy as np


def gate_by_max_cost(cost_matrix: np.ndarray, max_cost: float) -> np.ndarray:
    """Allow only pairs with cost <= max_cost."""
    c = np.asarray(cost_matrix, dtype=float)
    thr = float(max_cost)
    return np.isfinite(c) & (c <= thr)


def gate_by_max_distance(cost_matrix: np.ndarray, max_dist: float, assume_squared_cost: bool = True) -> np.ndarray:
    """Allow only pairs within a displacement threshold.

    Parameters
    ----------
    assume_squared_cost:
        If True, interprets matrix as squared distances and uses sqrt before
        comparing to `max_dist`.
    """
    c = np.asarray(cost_matrix, dtype=float)
    d = np.sqrt(np.maximum(c, 0.0)) if bool(assume_squared_cost) else c
    return np.isfinite(d) & (d <= float(max_dist))


def gate_by_percentile(cost_matrix: np.ndarray, keep_percentile: float = 95.0) -> np.ndarray:
    """Keep edges up to a global cost percentile.

    This is useful when absolute distance scales vary between experiments.
    """
    c = np.asarray(cost_matrix, dtype=float)
    finite = np.isfinite(c)
    if not np.any(finite):
        return np.zeros_like(c, dtype=bool)
    pct = float(np.clip(keep_percentile, 0.0, 100.0))
    thr = float(np.percentile(c[finite], pct))
    return finite & (c <= thr)


def combine_gates(*masks: np.ndarray | None) -> np.ndarray | None:
    """Combine multiple boolean gate masks via logical AND."""
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    out = np.asarray(valid[0], dtype=bool).copy()
    for m in valid[1:]:
        out &= np.asarray(m, dtype=bool)
    return out


def apply_gate_to_cost(cost_matrix: np.ndarray, gating_mask: np.ndarray | None, forbidden_cost: float | None = None) -> np.ndarray:
    """Return gated cost matrix with forbidden entries set to large cost.

    This matrix can be passed directly to Hungarian assignment.
    """
    c = np.asarray(cost_matrix, dtype=float).copy()
    if gating_mask is None:
        return c
    g = np.asarray(gating_mask, dtype=bool)
    if g.shape != c.shape:
        raise ValueError(f"gating_mask shape {g.shape} must match cost_matrix shape {c.shape}")
    if forbidden_cost is None:
        finite = c[np.isfinite(c)]
        base = float(np.nanmax(finite)) if finite.size else 1.0
        forbidden_cost = max(1e9, base * 1e6 + 1.0)
    c[~g] = float(forbidden_cost)
    return c
