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
from .kalman import associate_kalman_cv
from .min_cost_flow import associate_min_cost_flow
from .tracklets import associate_tracklet_stitching
from .diagnostics import compute_displacement_from_gt, compute_nn_distance, compute_rho
from .hungarian_cv_variants import (
    associate_hungarian_adaptive_maxdist_cv,
    associate_hungarian_innovation_percentile_cv,
    associate_hungarian_nis_gating_cv,
    associate_hungarian_switch_penalty_nis_cv,
    associate_hungarian_switch_penalty_cv,
    associate_hungarian_two_stage_nis_cv,
    associate_hungarian_two_stage_cv,
)
from .synthetic import (
    load_blind_frames,
    load_groundtruth_mapping,
    synthetic_generate_blind_pair,
    synthetic_generate_blind_pair_random_pointcloud,
    synthetic_generate_blind_pair_varying_dx_birthdeath_drift,
)
from .types import AssociationResult, GroundTruthMapping, PointCloudFrame

__all__ = [
    "PointCloudFrame",
    "AssociationResult",
    "GroundTruthMapping",
    "associate_nodrift",
    "associate_with_registration",
    "em_associate_and_driftfit",
    "associate_min_cost_flow",
    "associate_tracklet_stitching",
    "associate_kalman_cv",
    "associate_hungarian_adaptive_maxdist_cv",
    "associate_hungarian_innovation_percentile_cv",
    "associate_hungarian_nis_gating_cv",
    "associate_hungarian_switch_penalty_cv",
    "associate_hungarian_switch_penalty_nis_cv",
    "associate_hungarian_two_stage_cv",
    "associate_hungarian_two_stage_nis_cv",
    "DriftModelAdapterRigidStep",
    "DriftModelAdapterRbfToReference",
    "DriftModelAdapterSpacetimeRbf",
    "frames_to_points_df",
    "frames_to_tracks_df",
    "tracks_df_to_frames",
    "load_blind_frames",
    "load_groundtruth_mapping",
    "synthetic_generate_blind_pair",
    "synthetic_generate_blind_pair_random_pointcloud",
    "synthetic_generate_blind_pair_varying_dx_birthdeath_drift",
    "compute_nn_distance",
    "compute_displacement_from_gt",
    "compute_rho",
]
