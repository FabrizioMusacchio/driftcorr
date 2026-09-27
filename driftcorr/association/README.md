# driftcorr/association

## Purpose
This submodule solves point-cloud association for 3D+t data when stable IDs are missing.
It is intentionally usable without the rest of the drift correction CLI and can also be
coupled to drift fitting (EM loop).

Core problem:
- Input per timepoint: unlabeled 3D point cloud.
- Output: inferred pseudo tracks, link table, frame statistics, quality metrics.

## Main API Entry Points
Public imports are defined in `__init__.py`.

Primary methods:
- `associate_nodrift(...)`
- `associate_with_registration(...)`
- `em_associate_and_driftfit(...)`
- `associate_kalman_cv(...)`
- `associate_min_cost_flow(...)`
- `associate_tracklet_stitching(...)`
- Hungarian CV variant family in `hungarian_cv_variants.py`

Shared output type:
- `AssociationResult` (tracks, links, frame_stats, quality, params)

## Typical Data Flow
1. Build / load frame list (`PointCloudFrame`).
2. Compute pairwise costs and gates.
3. Solve assignment (Hungarian / nearest / specialized method).
4. Link assignments across time into pseudo tracks.
5. Score with no-GT metrics or GT-aware metrics.
6. Optionally visualize or run diagnostics.

## File-by-File Structure

### Public entry + data contracts

#### `__init__.py`
Good for:
- Finding what is considered the public API.

Contains:
- Re-exports of runners, specialized methods, synthetic helpers, diagnostics, and types.

#### `types.py`
Good for:
- Understanding data contracts across all methods.

Contains:
- `PointCloudFrame`: one frame (`time`, `xyz`, `meta`).
- `AssociationResult`: standard method output container.
- `GroundTruthMapping`: GT helper container for benchmark evaluation.

---

### Core assignment building blocks

#### `costs.py`
Good for:
- Defining or extending pairwise cost functions.

Contains:
- `cost_sqeuclidean`
- `cost_euclidean`
- `cost_huber`
- `cost_mahalanobis` (full covariance or diagonal scales)

#### `gating.py`
Good for:
- Restricting implausible candidate links before assignment.

Contains:
- `gate_by_max_cost`
- `gate_by_max_distance`
- `gate_by_percentile`
- `combine_gates`
- `apply_gate_to_cost` (mask forbidden edges)

#### `assignment.py`
Good for:
- Low-level assignment mechanics independent of high-level runner logic.

Contains:
- `build_cost_matrix(...)` with metric switch.
- `apply_gating(...)` helper wrapper around `gating.py`.
- `compute_nis_cost_matrix(...)` and `gate_by_nis(...)`.
- `hungarian_assignment(...)` with optional dummy unmatched handling.
- `nearest_neighbor_assignment(...)`.

Notes:
- Dummy mode supports `miss_cost`, `birth_cost`, `death_cost`.
- This is the core place for one-frame-step matching behavior.

#### `linking.py`
Good for:
- Turning frame-wise assignments into full pseudo tracks.

Contains:
- `link_assignments_to_tracks(...)`

Behavior:
- Unmatched destinations create new tracks (birth).
- Unmatched sources terminate tracks (death).
- Builds `tracks_df`, `links_df`, `frame_stats_df`.

---

### Motion and registration helpers

#### `motion_models.py`
Good for:
- Stateful prediction models for frame-to-frame association.

Contains:
- `ConstantPositionModel`
- `ConstantVelocityModel`
- `SimpleKalmanFilter3D` (lightweight helper)

#### `registration.py`
Good for:
- Pre-alignment before association in drifted scenarios.

Contains:
- `procrustes_rigid(...)`
- `icp_rigid(...)`
- Optional non-rigid registrar protocol hooks.

Notes:
- Registration-assisted runner currently uses rigid ICP path.

---

### High-level orchestration

#### `runner.py`
Good for:
- Main "glue" logic and default execution path.

Contains:
- Format converters:
  - `frames_to_points_df`
  - `frames_to_tracks_df`
  - `tracks_df_to_frames`
- Internal helpers:
  - `_run_pairwise_assignments`
  - `_build_result`
  - velocity estimation from inferred tracks
- Public runners:
  - `associate_nodrift`
  - `associate_with_registration`
  - `em_associate_and_driftfit`

Also contains:
- Drift adapters for EM:
  - `DriftModelAdapterRigidStep`
  - `DriftModelAdapterRbfToReference`
  - `DriftModelAdapterSpacetimeRbf`

---

### Specialized association strategies

#### `hungarian_cv_variants.py`
Good for:
- Experimental Hungarian + constant-velocity variants and gating studies.

Contains public methods:
- `associate_hungarian_adaptive_maxdist_cv`
- `associate_hungarian_innovation_percentile_cv`
- `associate_hungarian_switch_penalty_cv`
- `associate_hungarian_two_stage_cv`
- `associate_hungarian_nis_gating_cv`
- `associate_hungarian_switch_penalty_nis_cv`
- `associate_hungarian_two_stage_nis_cv`

Notes:
- Supports Euclidean and NIS gating modes.
- Includes sigma adaptation and variant-specific heuristics.

#### `kalman.py`
Good for:
- Full CV state-space association with innovation gating.

Contains:
- `associate_kalman_cv(...)`
- Internal track state, transition/process/measurement model helpers.

#### `min_cost_flow.py`
Good for:
- Global multi-frame linking instead of strictly local pairwise linking.

Contains:
- `associate_min_cost_flow(...)`
- Candidate edge generation and global optimization path cover logic.

Notes:
- Uses optional `networkx` if available; has fallback behavior.

#### `tracklets.py`
Good for:
- Two-stage strategy: conservative local linking then stitch tracklets.

Contains:
- `associate_tracklet_stitching(...)`
- Tracklet build/summarize/stitch internals.

---

### Evaluation, diagnostics, synthetic support, visualization

#### `scoring.py`
Good for:
- Quantitative evaluation and quality reporting.

Contains:
- No-GT metrics:
  - `link_confidence`
  - `score_links_df`
  - `score_tracks_df`
  - `global_association_metrics`
- GT-aware metrics:
  - `evaluate_against_groundtruth(...)`

GT-aware outputs include:
- coverage
- track purity
- fragmentation
- ID switches
- per-time link precision/recall/F1
- ID-consistent per-time RMS and aggregates

#### `diagnostics.py`
Good for:
- Understanding association difficulty regimes independent of method quality.

Contains:
- `compute_nn_distance(...)` -> spacing curve `d_NN(t)`
- `compute_displacement_from_gt(...)` -> GT displacement `Delta x(t)`
- `compute_rho(...)` -> ambiguity ratio `rho(t)=Delta x/d_NN`

#### `synthetic.py`
Good for:
- Building reproducible blind-pair datasets for benchmarks.

Contains:
- `synthetic_generate_blind_pair(...)`
- `synthetic_generate_blind_pair_varying_dx_birthdeath_drift(...)`
- `load_blind_frames(...)`
- `load_groundtruth_mapping(...)`

#### `visualization.py`
Good for:
- Plot-level sanity checks and benchmark reporting figures.

Contains:
- `plot_3d_frames_comparison(...)`
- `plot_rms_over_time(...)`
- `plot_accuracy_over_time(...)`
- Internal helpers for regime/file extraction and optional drift schedule overlays.

---

## Which File To Edit For Which Change

- New cost function: `costs.py` + metric switch in `assignment.py`.
- New gate rule: `gating.py` + integrate in `assignment.py` or variant module.
- New base association runner behavior: `runner.py`.
- New Hungarian CV experimental variant: `hungarian_cv_variants.py`.
- Kalman model tuning: `kalman.py`.
- Global association strategy changes: `min_cost_flow.py`.
- Tracklet strategy changes: `tracklets.py`.
- Birth/death linking policy changes: `linking.py`.
- New benchmark metric: `scoring.py` (and maybe `diagnostics.py`).
- New synthetic benchmark generator format: `synthetic.py`.
- New plots for reports: `visualization.py`.

## Minimal Usage Example

```python
from driftcorr.association import (
    load_blind_frames,
    associate_nodrift,
    evaluate_against_groundtruth,  # from scoring.py if imported directly
)

frames = load_blind_frames("path/to/file_blind.csv")
result = associate_nodrift(
    frames,
    method="hungarian",
    motion_model="constant_velocity",
    gating={"max_dist": 4.0, "keep_percentile": 99.0},
    params={"metric": "sqeuclidean", "motion_refine_iters": 2},
)

tracks = result.tracks_df
links = result.links_df
quality = result.quality
```

## Important Conventions

- Coordinates are expected as xyz (or `Position X/Y/Z` in helper conversions).
- Time is numeric and sorted for most routines.
- Frame-to-frame assignment is the core primitive; multi-frame behavior is produced by linking or global methods.
- Most benchmark scripts assume reproducibility via explicit seeds and deterministic folder naming.
