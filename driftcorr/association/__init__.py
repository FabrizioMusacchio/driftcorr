"""Optional point-cloud association module for driftcorr.

Public API
----------
- `associate_nodrift`: direct frame-to-frame association without drift model.
- `associate_with_registration`: registration-assisted association.
- `em_associate_and_driftfit`: alternating association/drift fit loop.
- `load_blind_frames`: load blinded point clouds from CSV.
- `synthetic_generate_blind_pair`: build paired GT/blind synthetic datasets.

This package is intentionally optional and independent from the core drift
correction CLI workflows.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from .runner import (
    DriftModelAdapterRbfToReference,
    DriftModelAdapterRigidStep,
    DriftModelAdapterSpacetimeRbf,
    associate_nodrift,
    associate_with_registration,
    em_associate_and_driftfit,
    frames_to_points_df,
    frames_to_tracks_df,
    tracks_df_to_frames,
)
from .synthetic import load_blind_frames, load_groundtruth_mapping, synthetic_generate_blind_pair
from .types import AssociationResult, GroundTruthMapping, PointCloudFrame

__all__ = [
    "PointCloudFrame",
    "AssociationResult",
    "GroundTruthMapping",
    "associate_nodrift",
    "associate_with_registration",
    "em_associate_and_driftfit",
    "DriftModelAdapterRigidStep",
    "DriftModelAdapterRbfToReference",
    "DriftModelAdapterSpacetimeRbf",
    "frames_to_points_df",
    "frames_to_tracks_df",
    "tracks_df_to_frames",
    "load_blind_frames",
    "load_groundtruth_mapping",
    "synthetic_generate_blind_pair",
]
