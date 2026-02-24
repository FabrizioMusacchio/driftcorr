"""Typed containers for point-cloud association without persistent input IDs.

This module formalizes the E-step data structures for association from raw
point clouds. The mathematical setting is:

- Input at each timepoint t: X_t in R^{N_t x 3} with xyz coordinates only.
- No stable track IDs are provided.
- Goal: infer pseudo track IDs by linking points across consecutive frames.

Outputs represent inferred trajectories and diagnostics:
- pseudo tracks with inferred `TrackID`,
- per-link assignment details,
- per-frame statistics,
- aggregate quality metrics.

Core assumptions:
- per-frame displacement is moderate relative to local spacing,
- N_t is approximately stable but births/deaths are allowed,
- unmatched points are allowed via gating and birth/death logic.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PointCloudFrame:
    """Single timepoint point cloud without identity labels.

    Parameters
    ----------
    time:
        Scalar frame time value.
    xyz:
        Point coordinates with shape (N, 3).
    meta:
        Optional metadata dictionary (source file, regime, frame index, etc.).

    Notes
    -----
    The association module intentionally treats `xyz` as unlabeled observations.
    Any hidden GT identities used for evaluation must be kept outside this
    container, e.g. in `GroundTruthMapping`.
    """

    time: float
    xyz: np.ndarray
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.xyz, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"PointCloudFrame.xyz must have shape (N,3), got {arr.shape}")
        self.xyz = arr
        self.time = float(self.time)
        if self.meta is None:
            self.meta = {}


@dataclass(slots=True)
class AssociationResult:
    """Result bundle of point-cloud association.

    Attributes
    ----------
    tracks_df:
        Inferred pseudo tracks with at least:
        `TrackID`, `Time`, `x`, `y`, `z`, `PointIndex`.
        Convenience columns `Position X/Y/Z` are typically included for
        compatibility with driftcorr fitters.
    links_df:
        Per-link assignment table with columns such as:
        `t`, `src_index`, `dst_index`, `cost`, `gated`, `matched`,
        optionally confidence and motion residual fields.
    frame_stats_df:
        Per-frame and per-transition diagnostics:
        counts of points, matches, unmatched sources/destinations, etc.
    quality:
        Global summary metrics (mean cost, unmatched rate, fragmentation,
        switch-like proxy, confidence summary).
    params:
        Algorithm parameter dictionary used to produce this result.
    """

    tracks_df: pd.DataFrame
    links_df: pd.DataFrame
    frame_stats_df: pd.DataFrame
    quality: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroundTruthMapping:
    """Ground-truth mapping utilities for blinded synthetic datasets.

    Attributes
    ----------
    point_to_gt_df:
        Mapping table with at least columns:
        `File`, `Time`, `PointIndex`, `GT_TrackID`.
        This table unblinds synthetic point-cloud rows to GT track IDs for
        evaluation.
    gt_tracks_df:
        Optional full ground-truth track table.
    meta:
        Metadata such as regime name, seed, generation parameters.

    Notes
    -----
    This container is optional at runtime. Association itself does not require
    GT and should remain usable without this structure.
    """

    point_to_gt_df: pd.DataFrame
    gt_tracks_df: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)
