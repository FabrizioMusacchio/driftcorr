"""Segment carcinoma 3D+T TIFF data and export GT tables compatible with driftcorr.

This script:
1. reads the `carcinoma_xyzt.tif` stack with `tifffile`,
2. segments cells in each 3D timepoint,
3. links labels over time via voxel overlap with centroid-distance fallback,
4. writes a tracked 3D+T label TIFF for napari,
5. exports IMARIS-like GT CSV tables,
6. simulates a mild rotational drift on centroids,
7. exports a blind point-cloud CSV plus unblinding table,
8. saves 3D plots for quick inspection.

Default input:
`example_image_data/Test datasets for segmentation/test_datasets_segmentation/Carcinoa_3DT/carcinoma_xyzt.tif`

Run example:
`conda run -n manual_registration env PYTHONPATH="$PWD" python user_scripts/segment_carcinoma_3dt_and_build_gt.py`
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

MPL_DIR = Path.cwd() / ".mplconfig_user_scripts"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tifffile
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires `tifffile`. Install it in your environment, then rerun."
    ) from exc

try:
    from scipy import ndimage as ndi
    from scipy import stats as sstats
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires `scipy`. Install it in your environment, then rerun."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftcorr.association import (
    associate_hungarian_nis_gating_cv,
    associate_hungarian_switch_penalty_cv,
    associate_hungarian_switch_penalty_nis_cv,
    associate_hungarian_two_stage_cv,
    associate_hungarian_two_stage_nis_cv,
    associate_nodrift,
)
from driftcorr.association.runner import tracks_df_to_frames
from driftcorr.association.scoring import evaluate_against_groundtruth
from driftcorr.association.synthetic import load_blind_frames
from driftcorr.association.visualization import (
    plot_3d_frames_comparison,
    plot_accuracy_over_time,
    plot_rms_over_time,
)
from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
from driftcorr.io_imaris import ensure_numeric_tracks
from driftcorr.method_local_communities import run_local_drift_correction_for_file
from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks, build_step_transforms
from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
from driftcorr.synthetic_experiments import save_groundtruth_evaluation_outputs
from driftcorr.visualization import save_visual_inspection_plots
# %% CONSTANTS AND PARAMETERS
INPUT_TIF = (
    PROJECT_ROOT
    / "example_image_data"
    / "Test datasets for segmentation"
    / "test_datasets_segmentation"
    / "Carcinoa_3DT"
    / "carcinoma_xyzt.tif"
)
OUTPUT_DIR = INPUT_TIF.parent / "carcinoma_xyzt_tracking_exports"
HCV_BENCHMARK_STATS_PATH = (
    PROJECT_ROOT
    / "Synthetic files track inference parameter sweep"
    / "hungarian_cv_variants_sweep_nis_dx_blind_varying_dx"
    / "aggregated_hcv_nis_dx_blind_varying_dx_stats.csv"
)
PCA_MAX_FRAME_GAP = 2
RIGID_FAIRNESS_STATS_PATH = PROJECT_ROOT / "Synthetic files parameter sweep" / "rigid_fairness_sweep" / "rigid_fairness_aggregated.csv"
LOCAL_FAIRNESS_STATS_PATH = PROJECT_ROOT / "Synthetic files parameter sweep" / "local_fairness_sweep" / "local_fairness_aggregated.csv"
ST_RBF_GRID_STATS_PATH = PROJECT_ROOT / "Synthetic files parameter sweep" / "st_rbf_parameter_selection" / "st_rbf_grid_aggregated.csv"
ST_RBF_AB_STATS_PATH = PROJECT_ROOT / "Synthetic files parameter sweep" / "st_rbf_parameter_selection" / "st_rbf_stationary_ab_summary.csv"

EXPECTED_SHAPE = (12, 38, 512, 512)
EXPECTED_AXES = "TZYX"

GAUSSIAN_SIGMA_ZYX = (0.7, 1.0, 1.0)
BACKGROUND_SIGMA_ZYX = (2.0, 8.0, 8.0)
PRIMARY_MIN_COMPONENT_SIZE = 150
RESCUE_MIN_COMPONENT_SIZE = 120
FINAL_MIN_COMPONENT_SIZE = 120
LOW_THRESHOLD_FRACTION = 0.35
LOW_THRESHOLD_FLOOR = 0.12
RESCUE_LOW_THRESHOLD_FRACTION = 0.60
RESCUE_LOW_THRESHOLD_FLOOR = 0.008
RESCUE_PEAK_MIN = 0.24
MAX_TRACK_LINK_DISTANCE = 18.0
REMOVE_SINGLE_FRAME_TRACKS = True
MOTION_CLASSIFICATION_MODE = "median_gmm"  # options: "mean_otsu_log", "median_gmm", "median_manual"
MANUAL_STATIONARY_THRESHOLD = 6.0  # px/frame; used only when MOTION_CLASSIFICATION_MODE == "median_manual"
ROTATION_TOTAL_DEG_XYZ = (3.0, -2.5, 2.0)
RIGID_DRIFT_PER_FRAME_XYZ = (2.5, -1.5, 0.8)
IMARIS_UNIT = "px"
VOXEL_SIZE_XYZ = (1.0, 1.0, 1.0)
HCV_DEFAULT_KEEP_PERCENTILE = 99.0
HCV_DEFAULT_MAX_DIST_FACTOR = 6.0
BEST_ASSOCIATION_METHOD = "hungarian_two_stage_nis_cv"
CM = 1.0 / 2.54
BASELINE_METHODS = ["rigid", "local", "rbf", "st_rbf"]
BASELINE_DEFAULT_OPTIONS: dict[str, Any] = {
    "reference_time": None,              # all methods: reference frame
    "seed": 0,                           # rbf, st_rbf: random seed
    "lam": 3e-4,                         # rbf: ridge regularization
    "keep_fraction": 0.7,                # rigid, rbf: inlier/trim fraction
    "min_pairs": 15,                     # rigid, rbf: minimum matched track pairs
    "m_centers": 45,                     # rbf: number of RBF centers
    "sigma": 180.0,                      # rbf: spatial kernel width
    "temporal_gamma": 1.0,               # rbf: temporal smoothing strength
    "center_mode": "ref",                # rbf: center selection mode
    "center_pooled_time_stride": 1,      # rbf: time stride for pooled centers
    "robust_mode": "sample_trim",        # rbf: robust fitting mode
    "track_trim_iters": 4,               # rbf: track trimming iterations
    "two_stage": 0,                      # rbf, st_rbf: two-stage fit on/off
    "rbf_linear_terms": 0,               # rbf, st_rbf: add linear terms
    "rbf_space_input_mode": "track_anchor",  # rbf: spatial input mode
    "row_position_iters": 3,             # rbf: row-position fit iterations
    "row_position_tol": 1e-3,            # rbf: row-position tolerance
    "weight_abort_threshold": 1e4,       # rbf, st_rbf: stability cutoff
    "rigid_mode": "rigid",               # rigid: transform model type
    "inlier_reestimate_iters": 30,       # rigid: inlier re-estimation iterations
    "k_neighbors": 8,                   # local: number of neighbors
    "radius_um": None,                   # local: optional radius instead of kNN
    "ell_um_graph": 80.0,                # local: graph weighting length scale
    "ell_um_field": 240.0,               # local: field smoothing length scale
    "min_edge_weight": 0.25,             # local: minimum graph edge weight
    "max_pairs_per_node": 50,            # local: max pairs per node
    "st_m_centers": 70,                  # st_rbf: number of space-time centers
    "st_lam": 8.5e-4,                    # st_rbf: ridge regularization
    "st_ell_space": 4.0,                 # st_rbf: spatial kernel length scale
    "st_ell_time": 0.4,                  # st_rbf: temporal kernel length scale
    "st_ell_x": None,                    # st_rbf: optional anisotropic x scale
    "st_ell_y": None,                    # st_rbf: optional anisotropic y scale
    "st_ell_z": None,                    # st_rbf: optional anisotropic z scale
    "st_robust_keep_fraction": 0.83,     # st_rbf: inlier fraction
    "st_robust_iters": 2,                # st_rbf: robust fitting iterations
    "space_input_mode": "ref_position",  # st_rbf: spatial input mode
    "st_refine_iters": 3,                # st_rbf: refinement stages
    "st_refine_tol": 0.0,                # st_rbf: refinement tolerance
    "st_robust_mode": "sample_trim",     # st_rbf: robust fitting mode
    "st_track_trim_iters": 4,            # st_rbf: track trimming iterations
    "st_per_axis_scaling": 1,            # st_rbf: per-axis scaling on/off
    "st_linear_interactions": 0,         # st_rbf: space-time interaction terms
}


HCV_METHOD_DEFS: list[dict[str, Any]] = [
    {
        "name": "hungarian_gating_cv",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_velocity",
        "params": {"metric": "sqeuclidean"},
    },
    {
        "name": "hungarian_two_stage_cv",
        "kind": "hungarian_cv_variant",
        "variant": "two_stage",
        "motion_model": "constant_velocity",
        "params": {
            "stage1_factor": 0.5,
            "stage1_keep_percentile": 90.0,
            "stage2_keep_percentile": 99.0,
            "stage2_allow_unmatched": True,
            "metric": "sqeuclidean",
            "use_dummy": True,
        },
    },
    {
        "name": "hungarian_switch_penalty_cv",
        "kind": "hungarian_cv_variant",
        "variant": "switch_penalty",
        "motion_model": "constant_velocity",
        "params": {"lambda_switch": 0.2, "use_normalized": True, "metric": "sqeuclidean", "use_dummy": True},
    },
    {
        "name": "hungarian_nis_gating_cv",
        "kind": "hungarian_cv_variant",
        "variant": "nis_gating",
        "motion_model": "constant_velocity",
        "params": {"gating_mode": "nis", "nis_alpha": 0.997, "use_dummy": True},
    },
    {
        "name": "hungarian_two_stage_nis_cv",
        "kind": "hungarian_cv_variant",
        "variant": "two_stage_nis",
        "motion_model": "constant_velocity",
        "params": {
            "gating_mode": "nis",
            "nis_alpha": 0.997,
            "stage1_factor": 0.5,
            "stage1_keep_percentile": 90.0,
            "stage2_keep_percentile": 99.0,
            "stage2_allow_unmatched": True,
            "use_dummy": True,
        },
    },
    {
        "name": "hungarian_switch_penalty_nis_cv",
        "kind": "hungarian_cv_variant",
        "variant": "switch_penalty_nis",
        "motion_model": "constant_velocity",
        "params": {
            "gating_mode": "nis",
            "nis_alpha": 0.997,
            "lambda_switch": 0.2,
            "use_normalized": True,
            "use_dummy": True,
        },
    },
]

HCV_METHOD_ORDER = [str(cfg["name"]) for cfg in HCV_METHOD_DEFS]
DRIFT_REGIMES: list[dict[str, Any]] = [
    {
        "name": "no_drift",
        "dataset_label": "carcinoma_3dt_no_drift",
        "plot_title": "Ground Truth tracks",
    },
    {
        "name": "rigid_drift",
        "dataset_label": "carcinoma_3dt_rigid_drift",
        "plot_title": "Rigidly drifted tracks",
    },
    {
        "name": "rotation_drift",
        "dataset_label": "carcinoma_3dt_rotation_drift",
        "plot_title": "Rotated tracks",
    },
]

# %% FUNCTIONS
def otsu_threshold(values: np.ndarray, bins: int = 256, value_range: tuple[float, float] | None = None) -> float:
    """Compute an Otsu threshold from a 1D numeric array."""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    if float(np.max(data)) <= float(np.min(data)):
        return float(np.min(data))

    hist, edges = np.histogram(data, bins=int(bins), range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = hist.sum() - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    between = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[1:]) ** 2
    if between.size == 0:
        return float(np.median(data))
    return float(centers[:-1][int(np.argmax(between))])

def load_stack_tzyx(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the TIFF stack and normalize it to TZYX."""
    with tifffile.TiffFile(path) as tif:
        array = tif.asarray()
        axes = getattr(tif.series[0], "axes", "")

    arr = np.asarray(array)
    info: dict[str, Any] = {
        "raw_shape": tuple(int(v) for v in arr.shape),
        "raw_axes": str(axes),
        "dtype": str(arr.dtype),
    }

    if arr.ndim == 4:
        if axes:
            axis_to_index = {ax: i for i, ax in enumerate(axes)}
            required = ["T", "Z", "Y", "X"]
            if all(ax in axis_to_index for ax in required):
                arr = np.transpose(arr, [axis_to_index[ax] for ax in required])
            else:
                raise ValueError(f"Unexpected TIFF axes {axes!r}; expected to include {EXPECTED_AXES}.")
        elif tuple(int(v) for v in arr.shape) != EXPECTED_SHAPE:
            raise ValueError(f"4D TIFF has shape {arr.shape}, but expected {EXPECTED_SHAPE}.")
    elif arr.ndim == 3:
        if tuple(int(v) for v in arr.shape) == (EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1], EXPECTED_SHAPE[2], EXPECTED_SHAPE[3]):
            arr = arr.reshape(EXPECTED_SHAPE)
        else:
            raise ValueError(f"3D TIFF has shape {arr.shape}; cannot infer T/Z reliably.")
    else:
        raise ValueError(f"Unsupported TIFF dimensionality: {arr.ndim}.")

    if tuple(int(v) for v in arr.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Normalized TIFF shape is {arr.shape}; expected exactly {EXPECTED_SHAPE}.")

    info["normalized_shape"] = tuple(int(v) for v in arr.shape)
    info["normalized_axes"] = EXPECTED_AXES
    return arr, info

def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than `min_size` voxels."""
    labels, _ = ndi.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= int(min_size))
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep)

def relabel_consecutive(mask: np.ndarray) -> np.ndarray:
    """Label connected components as consecutive positive integers."""
    labels, _ = ndi.label(mask)
    if labels.max() == 0:
        return labels.astype(np.uint32)
    return labels.astype(np.uint32)


def filter_components_by_peak(mask: np.ndarray, reference: np.ndarray, min_size: int, min_peak: float) -> np.ndarray:
    """Keep only components with sufficient size and peak response."""
    labels, _ = ndi.label(mask)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    out = np.zeros_like(mask, dtype=bool)
    for comp_id in range(1, int(labels.max()) + 1):
        if int(sizes[comp_id]) < int(min_size):
            continue
        comp_mask = labels == comp_id
        peak_value = float(np.max(reference[comp_mask]))
        if peak_value < float(min_peak):
            continue
        out[comp_mask] = True
    return out


def segment_volume(volume_zyx: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Segment one 3D frame using primary + rescue 3D segmentation branches."""
    vol = np.asarray(volume_zyx, dtype=np.float32)
    p_low, p_high = np.percentile(vol, [5.0, 99.8])
    denom = max(float(p_high - p_low), 1e-6)
    normalized = np.clip((vol - float(p_low)) / denom, 0.0, 1.0)
    smoothed = ndi.gaussian_filter(normalized, sigma=GAUSSIAN_SIGMA_ZYX)

    high_thr = otsu_threshold(smoothed[smoothed > 0.0], bins=256, value_range=(0.0, 1.0))
    low_thr = max(float(LOW_THRESHOLD_FLOOR), float(high_thr) * float(LOW_THRESHOLD_FRACTION))

    seeds = smoothed > high_thr
    support = smoothed > low_thr
    primary_mask = ndi.binary_propagation(seeds, mask=support)
    primary_mask = ndi.binary_closing(primary_mask, structure=ndi.generate_binary_structure(3, 1), iterations=1)
    primary_mask = ndi.binary_fill_holes(primary_mask)
    primary_mask = remove_small_components(primary_mask, min_size=PRIMARY_MIN_COMPONENT_SIZE)

    background = ndi.gaussian_filter(normalized, sigma=BACKGROUND_SIGMA_ZYX)
    enhanced = np.clip(smoothed - background, 0.0, None)
    rescue_thr = otsu_threshold(
        enhanced[enhanced > 0.0],
        bins=256,
        value_range=(0.0, max(float(enhanced.max()), 1e-6)),
    )
    rescue_low_thr = max(float(RESCUE_LOW_THRESHOLD_FLOOR), float(RESCUE_LOW_THRESHOLD_FRACTION) * float(rescue_thr))
    rescue_mask = ndi.binary_propagation(enhanced > rescue_thr, mask=enhanced > rescue_low_thr)
    rescue_mask = ndi.binary_closing(rescue_mask, structure=ndi.generate_binary_structure(3, 1), iterations=1)
    rescue_mask = ndi.binary_fill_holes(rescue_mask)
    rescue_mask = filter_components_by_peak(
        rescue_mask,
        reference=smoothed,
        min_size=RESCUE_MIN_COMPONENT_SIZE,
        min_peak=RESCUE_PEAK_MIN,
    )

    mask = primary_mask | rescue_mask
    mask = ndi.binary_closing(mask, structure=ndi.generate_binary_structure(3, 1), iterations=1)
    mask = ndi.binary_fill_holes(mask)
    mask = remove_small_components(mask, min_size=FINAL_MIN_COMPONENT_SIZE)
    labels = relabel_consecutive(mask)

    stats = {
        "p_low": float(p_low),
        "p_high": float(p_high),
        "high_threshold": float(high_thr),
        "low_threshold": float(low_thr),
        "rescue_threshold": float(rescue_thr),
        "rescue_low_threshold": float(rescue_low_thr),
        "seed_voxels": int(seeds.sum()),
        "primary_mask_voxels": int(primary_mask.sum()),
        "rescue_mask_voxels": int(rescue_mask.sum()),
        "mask_voxels": int(mask.sum()),
        "n_components": int(labels.max()),
    }
    return labels, stats


def centroid_xyz_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return centroid as x/y/z using voxel coordinates."""
    zyx = np.asarray(ndi.center_of_mass(mask), dtype=np.float64)
    return np.array(
        [
            float(zyx[2]) * float(VOXEL_SIZE_XYZ[0]),
            float(zyx[1]) * float(VOXEL_SIZE_XYZ[1]),
            float(zyx[0]) * float(VOXEL_SIZE_XYZ[2]),
        ],
        dtype=np.float64,
    )

def build_frame_components(local_labels: np.ndarray) -> dict[int, dict[str, Any]]:
    """Collect masks and centroids for all components in one frame."""
    ids = np.unique(local_labels)
    ids = ids[ids > 0]
    components: dict[int, dict[str, Any]] = {}
    for local_id in ids.tolist():
        mask = local_labels == int(local_id)
        components[int(local_id)] = {
            "mask": mask,
            "centroid_xyz": centroid_xyz_from_mask(mask),
            "size": int(mask.sum()),
        }
    return components


def assign_tracks(
    local_stack_tzyx: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Assign stable TrackIDs over time using overlap first, then centroid distance."""
    n_t = int(local_stack_tzyx.shape[0])
    global_stack = np.zeros_like(local_stack_tzyx, dtype=np.uint32)

    next_track_id = 1
    row_id = 1
    prev_frame = np.zeros(local_stack_tzyx.shape[1:], dtype=np.uint32)
    prev_centroids: dict[int, np.ndarray] = {}

    track_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for t_index in range(n_t):
        time_value = float(t_index + 1)
        local_labels = local_stack_tzyx[t_index]
        components = build_frame_components(local_labels)

        assignments: dict[int, tuple[int, str, float]] = {}
        used_track_ids: set[int] = set()

        overlap_candidates: list[tuple[int, int, int]] = []
        for local_id, info in components.items():
            overlapping = prev_frame[info["mask"]]
            overlapping = overlapping[overlapping > 0]
            if overlapping.size == 0:
                continue
            track_ids, counts = np.unique(overlapping, return_counts=True)
            best_idx = int(np.argmax(counts))
            overlap_candidates.append((int(counts[best_idx]), int(local_id), int(track_ids[best_idx])))

        for overlap_voxels, local_id, track_id in sorted(overlap_candidates, reverse=True):
            if local_id in assignments or track_id in used_track_ids:
                continue
            assignments[local_id] = (track_id, "overlap", float(overlap_voxels))
            used_track_ids.add(track_id)

        distance_candidates: list[tuple[float, int, int]] = []
        unmatched_local_ids = [lid for lid in components if lid not in assignments]
        for local_id in unmatched_local_ids:
            curr_centroid = components[local_id]["centroid_xyz"]
            for track_id, prev_centroid in prev_centroids.items():
                if track_id in used_track_ids:
                    continue
                dist = float(np.linalg.norm(curr_centroid - prev_centroid))
                if dist <= float(MAX_TRACK_LINK_DISTANCE):
                    distance_candidates.append((dist, int(local_id), int(track_id)))

        for dist, local_id, track_id in sorted(distance_candidates):
            if local_id in assignments or track_id in used_track_ids:
                continue
            assignments[local_id] = (track_id, "distance", dist)
            used_track_ids.add(track_id)

        for local_id in sorted(components):
            if local_id not in assignments:
                assignments[local_id] = (next_track_id, "new", np.nan)
                next_track_id += 1

        frame_global = np.zeros_like(local_labels, dtype=np.uint32)
        curr_centroids: dict[int, np.ndarray] = {}
        link_summary = {"overlap": 0, "distance": 0, "new": 0}

        for local_id, (track_id, link_mode, link_metric) in sorted(assignments.items()):
            mask = components[local_id]["mask"]
            frame_global[mask] = int(track_id)
            centroid_xyz = components[local_id]["centroid_xyz"]
            curr_centroids[int(track_id)] = centroid_xyz
            link_summary[link_mode] += 1

            track_rows.append(
                {
                    "Position X": float(centroid_xyz[0]),
                    "Position Y": float(centroid_xyz[1]),
                    "Position Z": float(centroid_xyz[2]),
                    "Unit": IMARIS_UNIT,
                    "Category": "Spot",
                    "Collection": "Position",
                    "Time": float(time_value),
                    "TrackID": float(track_id),
                    "ID": int(row_id),
                    "Source Local Label": int(local_id),
                    "Voxel Count": int(components[local_id]["size"]),
                    "Link Mode": str(link_mode),
                    "Link Metric": float(link_metric) if np.isfinite(link_metric) else np.nan,
                }
            )
            row_id += 1

        global_stack[t_index] = frame_global
        prev_frame = frame_global
        prev_centroids = curr_centroids

        stats_rows.append(
            {
                "Time": float(time_value),
                "tracked_components": int(len(assignments)),
                "linked_by_overlap": int(link_summary["overlap"]),
                "linked_by_distance": int(link_summary["distance"]),
                "new_tracks": int(link_summary["new"]),
                "active_track_ids": int(len(curr_centroids)),
            }
        )

    tracks_df = pd.DataFrame(track_rows).sort_values(["Time", "TrackID"]).reset_index(drop=True)
    frame_stats_df = pd.DataFrame(stats_rows).sort_values("Time").reset_index(drop=True)
    return global_stack, tracks_df, frame_stats_df


def remove_single_frame_tracks(
    labels_tzyx: np.ndarray,
    tracks_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Remove tracks that appear in only one frame from labels and centroid table."""
    tracks = tracks_df.copy()
    tracks["TrackID"] = pd.to_numeric(tracks["TrackID"], errors="coerce")
    tracks["Time"] = pd.to_numeric(tracks["Time"], errors="coerce")
    tracks = tracks.dropna(subset=["TrackID", "Time"]).copy()
    if tracks.empty:
        cleanup_df = pd.DataFrame(
            [{"removed_single_frame_tracks": 0, "remaining_tracks": 0, "removed_rows": 0}]
        )
        return labels_tzyx.copy(), tracks_df.copy(), cleanup_df

    frames_per_track = (
        tracks.groupby("TrackID")["Time"]
        .nunique()
        .rename("n_frames")
        .reset_index()
    )
    remove_ids = frames_per_track.loc[frames_per_track["n_frames"] <= 1, "TrackID"].to_numpy(np.float64)
    if remove_ids.size == 0:
        cleanup_df = pd.DataFrame(
            [{
                "removed_single_frame_tracks": 0,
                "remaining_tracks": int(frames_per_track.shape[0]),
                "removed_rows": 0,
            }]
        )
        return labels_tzyx.copy(), tracks_df.copy(), cleanup_df

    remove_id_set = set(remove_ids.tolist())
    filtered_tracks_df = tracks_df.loc[~pd.to_numeric(tracks_df["TrackID"], errors="coerce").isin(remove_id_set)].copy()
    filtered_tracks_df = filtered_tracks_df.sort_values(["Time", "TrackID"]).reset_index(drop=True)

    filtered_labels = labels_tzyx.copy()
    remove_ids_u32 = np.asarray(remove_ids, dtype=np.uint32)
    filtered_labels[np.isin(filtered_labels, remove_ids_u32)] = 0

    cleanup_df = pd.DataFrame(
        [{
            "removed_single_frame_tracks": int(remove_ids.size),
            "remaining_tracks": int(pd.to_numeric(filtered_tracks_df["TrackID"], errors="coerce").dropna().nunique()),
            "removed_rows": int(len(tracks_df) - len(filtered_tracks_df)),
        }]
    )
    return filtered_labels, filtered_tracks_df, cleanup_df


def rotation_matrix_xyz(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Build a 3D rotation matrix for rotations around X, Y, then Z."""
    cx, sx = float(np.cos(angle_x)), float(np.sin(angle_x))
    cy, sy = float(np.cos(angle_y)), float(np.sin(angle_y))
    cz, sz = float(np.cos(angle_z)), float(np.sin(angle_z))

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def apply_rotational_drift(df: pd.DataFrame, max_deg_xyz: tuple[float, float, float]) -> pd.DataFrame:
    """Apply a mild cumulative frame-to-frame rotational drift around the global centroid center."""
    out = df.copy()
    times = np.array(sorted(out["Time"].astype(float).unique()), dtype=np.float64)
    center = out[["Position X", "Position Y", "Position Z"]].to_numpy(np.float64).mean(axis=0)
    total_deg_xyz = np.array(max_deg_xyz, dtype=np.float64)

    n_steps = max(len(times) - 1, 1)
    step_idx = np.arange(1, n_steps + 1, dtype=np.float64)
    weight_x = 1.0 + 0.20 * np.sin(np.pi * step_idx / max(n_steps, 1))
    weight_y = 1.0 + 0.15 * np.cos(np.pi * step_idx / max(n_steps, 1))
    weight_z = 1.0 + 0.10 * np.sin(2.0 * np.pi * step_idx / max(n_steps, 1))
    weights = np.column_stack([weight_x, weight_y, weight_z])

    step_deg = np.zeros((len(times), 3), dtype=np.float64)
    if len(times) > 1:
        step_deg[1:, :] = total_deg_xyz[None, :] * (weights / np.sum(weights, axis=0, keepdims=True))
    cum_deg = np.cumsum(step_deg, axis=0)
    cum_rad = np.deg2rad(cum_deg)

    rotated_rows: list[pd.DataFrame] = []
    for time_idx, time_value in enumerate(times.tolist()):
        frame = out[out["Time"] == float(time_value)].copy()
        rot = rotation_matrix_xyz(*cum_rad[time_idx])
        points = frame[["Position X", "Position Y", "Position Z"]].to_numpy(np.float64)
        rotated = ((points - center[None, :]) @ rot.T) + center[None, :]
        frame["Position X"] = rotated[:, 0]
        frame["Position Y"] = rotated[:, 1]
        frame["Position Z"] = rotated[:, 2]
        frame["Rotation Step X deg"] = float(step_deg[time_idx, 0])
        frame["Rotation Step Y deg"] = float(step_deg[time_idx, 1])
        frame["Rotation Step Z deg"] = float(step_deg[time_idx, 2])
        frame["Rotation Cum X deg"] = float(cum_deg[time_idx, 0])
        frame["Rotation Cum Y deg"] = float(cum_deg[time_idx, 1])
        frame["Rotation Cum Z deg"] = float(cum_deg[time_idx, 2])
        rotated_rows.append(frame)

    out_rot = pd.concat(rotated_rows, ignore_index=True)
    out_rot = out_rot.sort_values(["Time", "TrackID"]).reset_index(drop=True)
    out_rot["ID"] = np.arange(1, len(out_rot) + 1, dtype=int)
    return out_rot


def apply_rigid_translation_drift(df: pd.DataFrame, step_xyz: tuple[float, float, float]) -> pd.DataFrame:
    """Apply a cumulative rigid translation drift with constant per-frame step in x/y/z."""
    out = df.copy()
    times = np.array(sorted(out["Time"].astype(float).unique()), dtype=np.float64)
    step = np.asarray(step_xyz, dtype=np.float64).reshape(3)

    drifted_frames: list[pd.DataFrame] = []
    for time_idx, time_value in enumerate(times.tolist()):
        frame = out[out["Time"] == float(time_value)].copy()
        shift = float(time_idx) * step
        frame["Position X"] = frame["Position X"].to_numpy(float) + float(shift[0])
        frame["Position Y"] = frame["Position Y"].to_numpy(float) + float(shift[1])
        frame["Position Z"] = frame["Position Z"].to_numpy(float) + float(shift[2])
        frame["Rigid Drift Step X px"] = float(step[0])
        frame["Rigid Drift Step Y px"] = float(step[1])
        frame["Rigid Drift Step Z px"] = float(step[2])
        frame["Rigid Drift Cum X px"] = float(shift[0])
        frame["Rigid Drift Cum Y px"] = float(shift[1])
        frame["Rigid Drift Cum Z px"] = float(shift[2])
        drifted_frames.append(frame)

    out_drift = pd.concat(drifted_frames, ignore_index=True)
    out_drift = out_drift.sort_values(["Time", "TrackID"]).reset_index(drop=True)
    out_drift["ID"] = np.arange(1, len(out_drift) + 1, dtype=int)
    return out_drift


def make_blind_pointcloud_table(
    df_with_ids: pd.DataFrame,
    regime_name: str,
    file_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a blind point-cloud CSV and a synthetic-style unblinding table."""
    blind_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []

    for time_value, frame in df_with_ids.groupby("Time", sort=True):
        frame_sorted = frame.sort_values("TrackID").reset_index(drop=True)
        points = frame_sorted[["Position X", "Position Y", "Position Z", "TrackID"]].to_numpy(np.float64)
        for point_index in range(points.shape[0]):
            blind_rows.append(
                {
                    "Time": float(time_value),
                    "PointIndex": int(point_index),
                    "x": float(points[point_index, 0]),
                    "y": float(points[point_index, 1]),
                    "z": float(points[point_index, 2]),
                }
            )
            map_rows.append(
                {
                    "Regime": str(regime_name),
                    "File": str(file_name),
                    "Time": float(time_value),
                    "PointIndex": int(point_index),
                    "GT_TrackID": float(points[point_index, 3]),
                }
            )

    return pd.DataFrame(blind_rows), pd.DataFrame(map_rows)


def set_equal_box_aspect(ax: Any, xyz: np.ndarray) -> None:
    """Make 3D axes use roughly equal scaling."""
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    try:
        ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))
    except Exception:
        pass


def compute_xyz_limits(xyz: np.ndarray, pad_fraction: float = 0.06) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return padded XYZ limits with a cubic bounding box."""
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.size == 0:
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-6)
    half = 0.5 * float(np.max(span)) * (1.0 + float(pad_fraction))
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
        (float(center[2] - half), float(center[2] + half)),
    )


def style_benchmark_3d_axis(
    ax: Any,
    xyz: np.ndarray,
    title: str,
    add_equal_box: bool = True,
    set_xylim: bool = True,
) -> None:
    """Style a single 3D axis to match the compact benchmark panel appearance."""
    (xlim, ylim, zlim) = compute_xyz_limits(xyz)
    ax.set_title(title, pad=-55, fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if set_xylim:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
    ax.view_init(elev=25, azim=-60)
    ax.set_position([0.1, 0.1, 0.7, 0.8])
    ax.tick_params(axis="both", which="major", pad=-3, labelsize=6)
    ax.xaxis.labelpad = -7
    ax.yaxis.labelpad = -7
    ax.zaxis.labelpad = -7
    if add_equal_box:
        set_equal_box_aspect(ax, xyz)


def plot_tracks_3d(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Plot one 3D trajectory line per TrackID using benchmark-style single panels.
    
    DEBUG:
    df = regime_tracks_df
    out_path = regime_tracks_plot 
    title=str(regime_cfg["plot_title"])
    """
    fig = plt.figure(figsize=(6.0 * CM, 6.0 * CM), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    track_ids = np.array(sorted(df["TrackID"].astype(float).unique()), dtype=np.float64)
    cmap = plt.get_cmap("turbo")
    denom = max(len(track_ids) - 1, 1)
    xyz_all = df[["Position X", "Position Y", "Position Z"]].to_numpy(np.float64)

    for idx, track_id in enumerate(track_ids.tolist()):
        track = df[df["TrackID"] == float(track_id)].sort_values("Time")
        color = cmap(idx / denom)
        ax.plot(
            track["Position X"].to_numpy(float),
            track["Position Y"].to_numpy(float),
            track["Position Z"].to_numpy(float),
            color=color,
            linewidth=1.0,
            alpha=0.8)

    style_benchmark_3d_axis(ax, xyz_all, title=title, set_xylim=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)


def plot_blind_pointcloud_3d(df_blind: pd.DataFrame, out_path: Path, title: str) -> None:
    """Plot blind point clouds, color-coded by frame, using benchmark-style layout."""
    fig = plt.figure(figsize=(6.0 * CM, 6.0 * CM), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df_blind["x"].to_numpy(float),
        df_blind["y"].to_numpy(float),
        df_blind["z"].to_numpy(float),
        c=df_blind["Time"].to_numpy(float),
        cmap="viridis",
        s=6,
        alpha=0.55,
        linewidths=0.0,
    )
    xyz = df_blind[["x", "y", "z"]].to_numpy(np.float64)
    style_benchmark_3d_axis(ax, xyz, title=title)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.tick_params(labelsize=6, pad=1)
    cbar.outline.set_visible(False)
    cbar.set_label("Frame / Time")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)


def save_tracked_label_tiff(labels_tzyx: np.ndarray, out_path: Path) -> Path:
    """Write tracked labels as a TZYX TIFF for napari inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        out_path,
        labels_tzyx.astype(np.uint32, copy=False),
        metadata={"axes": EXPECTED_AXES},
        imagej=False,
    )
    return out_path


def save_qc_overlay_frames(raw_stack_tzyx: np.ndarray, labels_tzyx: np.ndarray, out_dir: Path) -> list[Path]:
    """Save max-projection QC overlays for a few representative frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = sorted({0, raw_stack_tzyx.shape[0] // 2, raw_stack_tzyx.shape[0] - 1})
    saved: list[Path] = []

    for frame_idx in frame_indices:
        raw_proj = np.max(raw_stack_tzyx[frame_idx], axis=0)
        label_proj = np.max(labels_tzyx[frame_idx], axis=0)
        out_path = out_dir / f"qc_overlay_frame_{frame_idx + 1:02d}.png"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(raw_proj, cmap="gray")
        axes[0].set_title(f"Raw Max Projection t={frame_idx + 1}")

        axes[1].imshow(raw_proj, cmap="gray")
        axes[1].imshow(np.ma.masked_where(label_proj == 0, label_proj), cmap="nipy_spectral", alpha=0.45)
        axes[1].set_title(f"Overlay t={frame_idx + 1}")

        axes[2].imshow(label_proj > 0, cmap="gray")
        axes[2].set_title(f"Binary Label Projection t={frame_idx + 1}")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved.append(out_path)

    return saved


def estimate_dx_scale_from_blind_frames(blind_frames: list[Any]) -> float:
    """Estimate a characteristic frame-to-frame displacement scale from blind frames."""
    if len(blind_frames) < 2:
        return 1.0

    medians: list[float] = []
    for idx in range(len(blind_frames) - 1):
        xyz0 = np.asarray(blind_frames[idx].xyz, dtype=np.float64)
        xyz1 = np.asarray(blind_frames[idx + 1].xyz, dtype=np.float64)
        if xyz0.size == 0 or xyz1.size == 0:
            continue
        diff = xyz0[:, None, :] - xyz1[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        nn = np.min(dist, axis=1)
        if nn.size > 0:
            medians.append(float(np.median(nn)))
    if not medians:
        return 1.0
    return max(float(np.median(np.asarray(medians, dtype=np.float64))), 1e-3)


def load_best_hcv_configs(stats_path: Path) -> dict[str, dict[str, Any]]:
    """Load best HCV configs from benchmark aggregation, fallback to defaults if unavailable."""
    fallback = {
        "hungarian_gating_cv": {
            "max_dist_factor": 6.0,
            "keep_percentile": 99.0,
            "motion_refine_iters": 5,
            "nn_scale_factor": 0.3,
            "sigma_init_mode": "dx",
        },
        "hungarian_two_stage_cv": {
            "max_dist_factor": 6.0,
            "keep_percentile": 99.0,
            "motion_refine_iters": 5,
            "nn_scale_factor": 0.3,
            "sigma_init_mode": "dx",
        },
        "hungarian_switch_penalty_cv": {
            "max_dist_factor": 6.0,
            "keep_percentile": 99.0,
            "motion_refine_iters": 5,
            "nn_scale_factor": 0.3,
            "sigma_init_mode": "dx",
        },
        "hungarian_nis_gating_cv": {
            "max_dist_factor": 6.0,
            "keep_percentile": 99.0,
            "motion_refine_iters": 0,
            "nn_scale_factor": 0.3,
            "sigma_init_mode": "nn_based",
        },
        "hungarian_two_stage_nis_cv": {
            "max_dist_factor": 6.0,
            "keep_percentile": 99.0,
            "motion_refine_iters": 0,
            "nn_scale_factor": 0.4,
            "sigma_init_mode": "nn_based",
        },
        "hungarian_switch_penalty_nis_cv": {
            "max_dist_factor": 6.0,
            "keep_percentile": 99.0,
            "motion_refine_iters": 0,
            "nn_scale_factor": 0.3,
            "sigma_init_mode": "nn_based",
        },
    }
    if not stats_path.exists():
        return fallback

    try:
        stats = pd.read_csv(stats_path)
    except Exception:
        return fallback

    req = {"method", "metric", "mean", "max_dist_factor", "keep_percentile", "motion_refine_iters", "nn_scale_factor", "sigma_init_mode"}
    if not req.issubset(stats.columns):
        return fallback

    link = stats[stats["metric"].astype(str) == "link_f1_mean"].copy()
    if link.empty:
        return fallback

    for col in ["mean", "max_dist_factor", "keep_percentile", "motion_refine_iters", "nn_scale_factor"]:
        link[col] = pd.to_numeric(link[col], errors="coerce")
    if "std" in link.columns:
        link["std"] = pd.to_numeric(link["std"], errors="coerce")
    else:
        link["std"] = np.nan

    link = link.sort_values(
        ["method", "mean", "std", "motion_refine_iters", "nn_scale_factor"],
        ascending=[True, False, True, True, True],
    )
    best = link.groupby("method", as_index=False).first()
    out = dict(fallback)
    for row in best.itertuples(index=False):
        method_name = str(getattr(row, "method"))
        out[method_name] = {
            "max_dist_factor": float(getattr(row, "max_dist_factor")),
            "keep_percentile": float(getattr(row, "keep_percentile")),
            "motion_refine_iters": int(round(float(getattr(row, "motion_refine_iters")))),
            "nn_scale_factor": float(getattr(row, "nn_scale_factor")),
            "sigma_init_mode": str(getattr(row, "sigma_init_mode")),
            "benchmark_link_f1_mean": float(getattr(row, "mean")),
            "benchmark_link_f1_std": float(getattr(row, "std")),
        }
    return out


def attach_pointindex_to_gt_tracks(gt_df: pd.DataFrame, unblinding_df: pd.DataFrame) -> pd.DataFrame:
    """Attach PointIndex to GT track rows using the unblinding table."""
    map_df = unblinding_df[["Time", "PointIndex", "GT_TrackID"]].copy()
    map_df["Time"] = pd.to_numeric(map_df["Time"], errors="coerce")
    map_df["GT_TrackID"] = pd.to_numeric(map_df["GT_TrackID"], errors="coerce")
    out = gt_df.copy()
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce")
    out["TrackID"] = pd.to_numeric(out["TrackID"], errors="coerce")
    out = out.merge(
        map_df,
        left_on=["Time", "TrackID"],
        right_on=["Time", "GT_TrackID"],
        how="left",
    )
    out = out.drop(columns=["GT_TrackID"])
    out["PointIndex"] = pd.to_numeric(out["PointIndex"], errors="coerce")
    return out.sort_values(["Time", "PointIndex", "TrackID"]).reset_index(drop=True)


def compute_accuracy_over_time_table(gt_tracks_df: pd.DataFrame, inferred_tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Compute GT-ID-consistent association accuracy over time."""
    gt = gt_tracks_df[["Time", "PointIndex", "TrackID"]].rename(columns={"TrackID": "GT_TrackID"}).copy()
    inf = inferred_tracks_df[["Time", "PointIndex", "TrackID"]].rename(columns={"TrackID": "INF_TrackID"}).copy()
    gt["Time"] = pd.to_numeric(gt["Time"], errors="coerce")
    gt["PointIndex"] = pd.to_numeric(gt["PointIndex"], errors="coerce")
    gt["GT_TrackID"] = pd.to_numeric(gt["GT_TrackID"], errors="coerce")
    inf["Time"] = pd.to_numeric(inf["Time"], errors="coerce")
    inf["PointIndex"] = pd.to_numeric(inf["PointIndex"], errors="coerce")
    inf["INF_TrackID"] = pd.to_numeric(inf["INF_TrackID"], errors="coerce")
    merged = inf.merge(gt, on=["Time", "PointIndex"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["Time", "accuracy"])

    dominant = (
        merged.groupby(["GT_TrackID", "INF_TrackID"], as_index=False)
        .size()
        .sort_values(["GT_TrackID", "size", "INF_TrackID"], ascending=[True, False, True])
        .drop_duplicates(["GT_TrackID"], keep="first")
        .rename(columns={"INF_TrackID": "INF_dom"})
    )
    dom_map = {float(r["GT_TrackID"]): float(r["INF_dom"]) for _, r in dominant.iterrows()}

    rows: list[dict[str, float]] = []
    for time_value, grp in merged.groupby("Time"):
        gt_ids = sorted(set(pd.to_numeric(grp["GT_TrackID"], errors="coerce").dropna().to_numpy(float).tolist()))
        if not gt_ids:
            continue
        correct = 0
        inf_arr = pd.to_numeric(grp["INF_TrackID"], errors="coerce").to_numpy(float)
        gt_arr = pd.to_numeric(grp["GT_TrackID"], errors="coerce").to_numpy(float)
        for gtid in gt_ids:
            inf_dom = dom_map.get(float(gtid))
            if inf_dom is None:
                continue
            ok = np.any((gt_arr == float(gtid)) & (inf_arr == float(inf_dom)))
            if bool(ok):
                correct += 1
        rows.append({"Time": float(time_value), "accuracy": float(correct / max(len(gt_ids), 1))})
    return pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)


def make_hcv_gating_and_params(
    method_cfg: dict[str, Any],
    best_cfg: dict[str, Any],
    dx_scale_est: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct benchmark-like gating and params for one HCV method."""
    params = dict(method_cfg.get("params", {}))
    params.update(
        {
            "motion_refine_iters": int(best_cfg.get("motion_refine_iters", 0)),
            "max_dist_factor": float(best_cfg.get("max_dist_factor", HCV_DEFAULT_MAX_DIST_FACTOR)),
            "keep_percentile": float(best_cfg.get("keep_percentile", HCV_DEFAULT_KEEP_PERCENTILE)),
            "dx_scale": float(dx_scale_est),
            "sigma_init_mode": str(best_cfg.get("sigma_init_mode", "dx")),
            "nn_scale_factor": float(best_cfg.get("nn_scale_factor", 0.3)),
            "max_gap": int(PCA_MAX_FRAME_GAP),
        }
    )
    gating = {
        "max_dist": float(max(0.5, params["max_dist_factor"] * max(float(dx_scale_est), 1e-6))),
        "keep_percentile": float(params["keep_percentile"]),
        "huber_delta": float(params.get("huber_delta", 1.0)),
    }
    return gating, params


def run_hcv_method(
    blind_frames: list[Any],
    method_cfg: dict[str, Any],
    gating: dict[str, Any],
    params: dict[str, Any],
):
    """Dispatch one HCV method using the same family as the benchmark."""
    kind = str(method_cfg.get("kind", "nodrift"))
    variant = str(method_cfg.get("variant", ""))
    if kind == "nodrift":
        return associate_nodrift(
            blind_frames,
            method=str(method_cfg.get("method", "hungarian")),
            gating=gating,
            motion_model=str(method_cfg.get("motion_model", "constant_velocity")),
            params=params,
        )
    if kind == "hungarian_cv_variant":
        if variant == "two_stage":
            return associate_hungarian_two_stage_cv(blind_frames, gating=gating, params=params)
        if variant == "switch_penalty":
            return associate_hungarian_switch_penalty_cv(blind_frames, gating=gating, params=params)
        if variant == "nis_gating":
            return associate_hungarian_nis_gating_cv(blind_frames, gating=gating, params=params)
        if variant == "two_stage_nis":
            return associate_hungarian_two_stage_nis_cv(blind_frames, gating=gating, params=params)
        if variant == "switch_penalty_nis":
            return associate_hungarian_switch_penalty_nis_cv(blind_frames, gating=gating, params=params)
    raise ValueError(f"Unsupported HCV method config: {method_cfg}")


def plot_heatmap_from_pivot(
    pivot: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap_name: str = "viridis",
    clim: tuple[float, float] | None = None,
    figsize: tuple = (7.2, 4.8),
    show_title: bool =True,
    rotate_xlabels: bool = False,
    show_xlabels: bool = True) -> Path:
    """Plot a heatmap from a prepared pivot table, analogous to benchmark plots."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f0f0f0")
    #fig_h = max(4.2, 0.48 * max(3, pivot.shape[0]))
    #fig_w = max(4.6, 1.25 * max(1, pivot.shape[1]))
    #fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig, ax = plt.subplots(figsize=figsize)
    mat = pivot.to_numpy(float)
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    if clim is not None:
        im.set_clim(*clim)
    #cbar = fig.colorbar(im, ax=ax, fraction=0.18, pad=0.03)
    cbar = fig.colorbar(im, ax=ax, fraction=0.5, pad=0.02)
    cbar.outline.set_visible(False)
    cbar_label_use = cbar_label.replace("_", " ")
    # scan cbar_label_use: every 16 characters, insert a newline if there is a space to break on:
    if len(cbar_label_use) > 20:
        parts = []
        start = 0
        while start < len(cbar_label_use):
            end = min(start + 16, len(cbar_label_use))
            if end < len(cbar_label_use):
                # try to break on last space:
                space_pos = cbar_label_use.rfind(" ", start, end)
                if space_pos != -1 and space_pos > start:
                    end = space_pos
            parts.append(cbar_label_use[start:end].strip())
            start = end
        cbar_label_use = "\n".join(parts)
    cbar.set_label(cbar_label_use, labelpad=1)
    ax.set_xticks(np.arange(pivot.shape[1]))
    if rotate_xlabels:
        rot_angle=20
    else:
        rot_angle=0
    if show_xlabels:
        ax.set_xticklabels([str(v) for v in pivot.columns.tolist()], rotation=rot_angle)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(v) for v in pivot.index.tolist()])
    if show_title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def order_hcv_methods_df(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    """Return a dataframe ordered by the canonical HCV method definition sequence."""
    if method_col not in df.columns:
        return df.copy()
    out = df.copy()
    order_map = {name: idx for idx, name in enumerate(HCV_METHOD_ORDER)}
    out["_hcv_order"] = out[method_col].astype(str).map(order_map).fillna(len(HCV_METHOD_ORDER)).astype(int)
    out = out.sort_values(["_hcv_order", method_col]).drop(columns=["_hcv_order"]).reset_index(drop=True)
    return out


def plot_metric_heatmap(summary_df: pd.DataFrame, 
                        metric: str, 
                        out_path: Path, 
                        title: str, 
                        dataset_label: str,
                        show_title: bool = True,
                        figure_size: tuple[float, float] = (7.2, 4.8),
                        clim: tuple[float, float] | None = None) -> Path:
    """Create a single-metric heatmap across methods for this dataset."""
    tab = summary_df[["method", metric]].copy()
    tab[metric] = pd.to_numeric(tab[metric], errors="coerce")
    tab = tab.dropna(subset=[metric]).reset_index(drop=True)
    tab = order_hcv_methods_df(tab, method_col="method")
    pivot = tab.set_index("method")[[metric]]
    pivot.columns = [str(dataset_label)]
    return plot_heatmap_from_pivot(pivot, title=title, cbar_label=metric, out_path=out_path, figsize=figure_size, clim=clim, show_title=show_title)


def plot_precision_recall_accuracy(summary_df: pd.DataFrame, 
                                   out_path: Path,
                                   dataset_label: str,
                                   figure_size: tuple[float, float] = (7.2, 4.8),
                                   show_title=True) -> Path:
    """Plot a heatmap for precision, recall and accuracy across methods."""
    metrics = ["link_precision_mean", "link_recall_mean", "accuracy_mean"]
    cols = ["method", *metrics]
    if "link_f1_mean" in summary_df.columns:
        cols.append("link_f1_mean")
    tab = summary_df[cols].copy()
    for metric in metrics:
        tab[metric] = pd.to_numeric(tab[metric], errors="coerce")
    tab = order_hcv_methods_df(tab, method_col="method")
    pivot = tab.set_index("method")[metrics].T
    pivot.columns = tab["method"].astype(str).tolist()
    pivot.index = [f"{dataset_label}: precision", f"{dataset_label}: recall", f"{dataset_label}: accuracy"]
    return plot_heatmap_from_pivot(
        pivot,
        title=f"Precision / Recall / Accuracy across HCV methods ({dataset_label})",
        cbar_label="score",
        out_path=out_path,
        clim=(0.0, 1.0),
        figsize=figure_size,
        show_title=show_title,
        rotate_xlabels=False)


def load_best_rigid_params(stats_path: Path) -> dict[str, Any]:
    """Choose rigid parameters from fairness sweep by across-regime RMS then drift bias."""
    fallback = {"keep_fraction": 0.85, "inlier_reestimate_iters": 5}
    if not stats_path.exists():
        return fallback
    try:
        df = pd.read_csv(stats_path)
    except Exception:
        return fallback
    req = {"keep_fraction", "inlier_reestimate_iters", "mean_rms_all", "mean_drift_bias_abs"}
    if not req.issubset(df.columns):
        return fallback
    cur = df[list(req)].copy()
    for col in req:
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
    cur = cur.dropna()
    if cur.empty:
        return fallback
    grouped = (
        cur.groupby(["keep_fraction", "inlier_reestimate_iters"], as_index=False)[["mean_rms_all", "mean_drift_bias_abs"]]
        .mean()
        .sort_values(["mean_rms_all", "mean_drift_bias_abs", "inlier_reestimate_iters"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    best = grouped.iloc[0]
    return {
        "keep_fraction": float(best["keep_fraction"]),
        "inlier_reestimate_iters": int(round(float(best["inlier_reestimate_iters"]))),
    }


def load_best_local_params(stats_path: Path) -> dict[str, Any]:
    """Choose local parameters from fairness sweep by across-regime RMS then drift bias."""
    fallback = {"k_neighbors": 30, "ell_um_field": 240.0}
    if not stats_path.exists():
        return fallback
    try:
        df = pd.read_csv(stats_path)
    except Exception:
        return fallback
    req = {"k_neighbors", "ell_um_field", "mean_rms_all", "mean_drift_bias_abs"}
    if not req.issubset(df.columns):
        return fallback
    cur = df[list(req)].copy()
    for col in req:
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
    cur = cur.dropna()
    if cur.empty:
        return fallback
    grouped = (
        cur.groupby(["k_neighbors", "ell_um_field"], as_index=False)[["mean_rms_all", "mean_drift_bias_abs"]]
        .mean()
        .sort_values(["mean_rms_all", "mean_drift_bias_abs", "k_neighbors"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    best = grouped.iloc[0]
    return {
        "k_neighbors": int(round(float(best["k_neighbors"]))),
        "ell_um_field": float(best["ell_um_field"]),
    }


def load_best_st_rbf_params(grid_stats_path: Path, ab_stats_path: Path) -> dict[str, Any]:
    """Choose ST-RBF parameters from sweep using combined RMS x drift-bias heuristic."""
    fallback = {"st_ell_space": 10.0, "st_ell_time": 0.2, "use_stationary_subset": False}
    if not grid_stats_path.exists():
        return fallback
    try:
        grid = pd.read_csv(grid_stats_path)
    except Exception:
        return fallback
    req = {"ell_space", "ell_time", "mean_rms_all", "mean_drift_bias_abs"}
    if not req.issubset(grid.columns):
        return fallback
    cur = grid[list(req)].copy()
    for col in req:
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
    cur = cur.dropna()
    if cur.empty:
        return fallback
    grouped = cur.groupby(["ell_space", "ell_time"], as_index=False)[["mean_rms_all", "mean_drift_bias_abs"]].mean()
    grouped["combined_score"] = grouped["mean_rms_all"] * grouped["mean_drift_bias_abs"]
    grouped = grouped.sort_values(["combined_score", "mean_rms_all", "mean_drift_bias_abs"], ascending=[True, True, True]).reset_index(drop=True)
    best = grouped.iloc[0]
    out = {
        "st_ell_space": float(best["ell_space"]),
        "st_ell_time": float(best["ell_time"]),
        "use_stationary_subset": False,
    }

    if ab_stats_path.exists():
        try:
            ab = pd.read_csv(ab_stats_path)
        except Exception:
            ab = pd.DataFrame()
        req_ab = {"ell_space", "ell_time", "use_stationary_subset", "mean_rms_all", "mean_drift_bias_abs"}
        if req_ab.issubset(ab.columns):
            cur_ab = ab[list(req_ab)].copy()
            for col in req_ab:
                cur_ab[col] = pd.to_numeric(cur_ab[col], errors="coerce")
            cur_ab = cur_ab.dropna()
            if not cur_ab.empty:
                mask = np.isclose(cur_ab["ell_space"].to_numpy(float), out["st_ell_space"]) & np.isclose(cur_ab["ell_time"].to_numpy(float), out["st_ell_time"])
                pick = cur_ab.loc[mask].copy()
                if not pick.empty:
                    pick["combined_score"] = pick["mean_rms_all"] * pick["mean_drift_bias_abs"]
                    pick = pick.sort_values(["combined_score", "mean_rms_all", "mean_drift_bias_abs", "use_stationary_subset"], ascending=[True, True, True, True]).reset_index(drop=True)
                    out["use_stationary_subset"] = bool(int(round(float(pick.iloc[0]["use_stationary_subset"]))))
    return out


def make_baseline_options_from_sweeps() -> dict[str, Any]:
    """Merge baseline defaults with sweep-derived best parameters."""
    opts = dict(BASELINE_DEFAULT_OPTIONS)
    opts.update(load_best_rigid_params(RIGID_FAIRNESS_STATS_PATH))
    opts.update(load_best_local_params(LOCAL_FAIRNESS_STATS_PATH))
    opts.update(load_best_st_rbf_params(ST_RBF_GRID_STATS_PATH, ST_RBF_AB_STATS_PATH))
    return opts


def run_baseline_methods_for_dataframe(
    df: pd.DataFrame,
    out_prefix: Path,
    methods: list[str],
    opts: dict[str, Any],
    use_stationary_subset: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run rigid/local/rbf/st_rbf correction methods on one dataframe."""
    df = ensure_numeric_tracks(df)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    corr_by_method: dict[str, pd.DataFrame] = {}

    if "rigid" in methods:
        step, tf_rigid, times = build_step_transforms(
            df,
            keep_fraction=opts["keep_fraction"],
            min_pairs=opts["min_pairs"],
            mode=opts["rigid_mode"],
            inlier_reestimate_iters=opts["inlier_reestimate_iters"],
        )
        ref_time = float(times.min()) if opts["reference_time"] is None else float(opts["reference_time"])
        df_rigid = apply_rigid_correction_to_tracks(df, step, times, ref_time)
        tf_rigid.to_csv(str(out_prefix) + "_rigid_transforms.csv", index=False)
        df_rigid.to_csv(str(out_prefix) + "_positions_corrected_rigid.csv", index=False)
        corr_by_method["rigid"] = df_rigid

    if "local" in methods:
        df_local, *_ = run_local_drift_correction_for_file(
            df,
            out_prefix=out_prefix,
            reference_time=opts["reference_time"],
            k_neighbors=opts["k_neighbors"],
            radius_um=opts["radius_um"],
            ell_um_graph=opts["ell_um_graph"],
            ell_um_field=opts["ell_um_field"],
            min_edge_weight=opts["min_edge_weight"],
            max_pairs_per_node=opts["max_pairs_per_node"],
            use_track_mean_position=True,
        )
        corr_by_method["local"] = df_local

    if "rbf" in methods:
        df_rbf, tf_table, _, _, _ = correct_positions_rbf_to_reference(
            df,
            reference_time=opts["reference_time"],
            m=opts["m_centers"],
            keep_fraction=opts["keep_fraction"],
            lam=opts["lam"],
            sigma=opts["sigma"],
            min_pairs=opts["min_pairs"],
            seed=opts["seed"],
            fit_track_ids=None,
            temporal_gamma=opts["temporal_gamma"],
            center_mode=opts["center_mode"],
            pooled_time_stride=opts["center_pooled_time_stride"],
            robust_mode=opts["robust_mode"],
            track_trim_iters=opts["track_trim_iters"],
            two_stage=bool(opts["two_stage"]),
            space_input_mode=opts.get("rbf_space_input_mode", opts["space_input_mode"]),
            rbf_linear_terms=bool(opts["rbf_linear_terms"]),
            row_position_iters=opts["row_position_iters"],
            row_position_tol=opts["row_position_tol"],
            weight_abort_threshold=opts["weight_abort_threshold"],
        )
        df_rbf.to_csv(str(out_prefix) + "_positions_corrected_rbf.csv", index=False)
        tf_table.to_csv(str(out_prefix) + "_rbf_fit_table.csv", index=False)
        make_rbf_diagnostic_plots(tf_table, out_prefix)
        corr_by_method["rbf"] = df_rbf

    if "st_rbf" in methods:
        st_model, st_info = fit_spacetime_rbf_model(
            df,
            reference_time=opts["reference_time"],
            m_centers=opts["st_m_centers"],
            ell_space=opts["st_ell_space"],
            ell_time=opts["st_ell_time"],
            ell_x=opts["st_ell_x"],
            ell_y=opts["st_ell_y"],
            ell_z=opts["st_ell_z"],
            lam=opts.get("st_lam", opts["lam"]),
            robust_keep_fraction=opts["st_robust_keep_fraction"],
            robust_iters=opts["st_robust_iters"],
            seed=opts["seed"],
            fit_track_ids=None,
            space_input_mode=opts["space_input_mode"],
            n_refine=opts["st_refine_iters"],
            two_stage=bool(opts["two_stage"]),
            robust_mode=opts["st_robust_mode"],
            track_trim_iters=opts["st_track_trim_iters"],
            rbf_linear_terms=bool(opts["rbf_linear_terms"]),
            st_linear_interactions=bool(opts["st_linear_interactions"]),
            weight_abort_threshold=opts["weight_abort_threshold"],
            per_axis_scaling=bool(opts["st_per_axis_scaling"]),
            refine_tol=opts["st_refine_tol"],
        )
        df_st = apply_spacetime_correction(df, st_model)
        pd.DataFrame([st_info]).to_csv(str(out_prefix) + "_st_rbf_fit_info.csv", index=False)
        df_st.to_csv(str(out_prefix) + "_positions_corrected_st_rbf.csv", index=False)
        corr_by_method["st_rbf"] = df_st

    return corr_by_method


def build_rotation_drift_groundtruth(df_groundtruth: pd.DataFrame, df_rotated: pd.DataFrame) -> pd.DataFrame:
    """Build a drift-groundtruth table analogous to synthetic benchmark sidecars."""
    gt = ensure_numeric_tracks(df_groundtruth)
    rot = ensure_numeric_tracks(df_rotated)
    merged = rot.merge(
        gt[["TrackID", "Time", "Position X", "Position Y", "Position Z"]],
        on=["TrackID", "Time"],
        how="inner",
        suffixes=("_obs", "_gt"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["TrackID", "Time", "drift_x_total", "drift_y_total", "drift_z_total"])
    merged["drift_x_total"] = merged["Position X_obs"].to_numpy(float) - merged["Position X_gt"].to_numpy(float)
    merged["drift_y_total"] = merged["Position Y_obs"].to_numpy(float) - merged["Position Y_gt"].to_numpy(float)
    merged["drift_z_total"] = merged["Position Z_obs"].to_numpy(float) - merged["Position Z_gt"].to_numpy(float)
    return merged[["TrackID", "Time", "drift_x_total", "drift_y_total", "drift_z_total"]].copy()


def choose_one_to_one_track_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Select a deterministic one-to-one GT/inferred mapping from association counts."""
    if mapping_df.empty:
        return pd.DataFrame(columns=["GT_TrackID", "TrackID", "count"])
    cur = mapping_df[["GT_TrackID", "TrackID", "count"]].copy()
    for col in ["GT_TrackID", "TrackID", "count"]:
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
    cur = cur.dropna().sort_values(["count", "GT_TrackID", "TrackID"], ascending=[False, True, True]).reset_index(drop=True)
    chosen_rows: list[dict[str, float]] = []
    used_gt: set[float] = set()
    used_inf: set[float] = set()
    for row in cur.itertuples(index=False):
        gt_id = float(row.GT_TrackID)
        inf_id = float(row.TrackID)
        if gt_id in used_gt or inf_id in used_inf:
            continue
        used_gt.add(gt_id)
        used_inf.add(inf_id)
        chosen_rows.append({"GT_TrackID": gt_id, "TrackID": inf_id, "count": float(row.count)})
    return pd.DataFrame(chosen_rows).sort_values(["GT_TrackID", "TrackID"]).reset_index(drop=True)


def remap_tracks_for_gt_evaluation(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Map inferred TrackIDs to unique GT TrackIDs for GT-based evaluation."""
    if df.empty or mapping_df.empty:
        out = df.copy()
        out["TrackID"] = pd.to_numeric(out.get("TrackID", np.nan), errors="coerce")
        return out.dropna(subset=["TrackID"]).copy()
    map_df = mapping_df[["GT_TrackID", "TrackID"]].copy()
    map_df["GT_TrackID"] = pd.to_numeric(map_df["GT_TrackID"], errors="coerce")
    map_df["TrackID"] = pd.to_numeric(map_df["TrackID"], errors="coerce")
    out = df.copy()
    out["TrackID"] = pd.to_numeric(out["TrackID"], errors="coerce")
    out = out.merge(map_df, on="TrackID", how="inner")
    out["TrackID"] = pd.to_numeric(out["GT_TrackID"], errors="coerce")
    out = out.drop(columns=["GT_TrackID"]).dropna(subset=["TrackID"]).copy()
    if "ID" in out.columns:
        out["ID"] = np.arange(1, len(out) + 1, dtype=int)
    return out.sort_values(["TrackID", "Time"]).reset_index(drop=True)


def build_regime_tracks(gt_tracks_df: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    """Create one centroid table for a requested drift regime."""
    name = str(regime_name)
    if name == "no_drift":
        out = gt_tracks_df.copy()
        out["ID"] = np.arange(1, len(out) + 1, dtype=int)
        return out
    if name == "rigid_drift":
        return apply_rigid_translation_drift(gt_tracks_df, step_xyz=RIGID_DRIFT_PER_FRAME_XYZ)
    if name == "rotation_drift":
        return apply_rotational_drift(gt_tracks_df, max_deg_xyz=ROTATION_TOTAL_DEG_XYZ)
    raise ValueError(f"Unknown drift regime: {regime_name}")


def compute_track_average_displacements(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track step-displacement summaries used for motion classification."""
    rows: list[dict[str, float]] = []
    for track_id, grp in df.groupby("TrackID", sort=True):
        track = grp.sort_values("Time")
        pts = track[["Position X", "Position Y", "Position Z"]].to_numpy(np.float64)
        if pts.shape[0] < 2:
            mean_disp = 0.0
            median_disp = 0.0
        else:
            diffs = np.diff(pts, axis=0)
            step_disp = np.linalg.norm(diffs, axis=1)
            mean_disp = float(np.mean(step_disp)) if step_disp.size else 0.0
            median_disp = float(np.median(step_disp)) if step_disp.size else 0.0
        rows.append(
            {
                "TrackID": float(track_id),
                "mean_step_displacement": float(mean_disp),
                "median_step_displacement": float(median_disp),
            }
        )
    return pd.DataFrame(rows).sort_values("TrackID").reset_index(drop=True)


def corrected_tracks_csv_path(drift_root: Path, regime_name: str, method_name: str) -> Path:
    """Return the corrected-track CSV path for one drift-correction method."""
    prefix = drift_root / f"{INPUT_TIF.stem}_{regime_name}_inferred"
    suffix_by_method = {
        "rigid": "_positions_corrected_rigid.csv",
        "local": "_positions_corrected_localdrift.csv",
        "rbf": "_positions_corrected_rbf.csv",
        "st_rbf": "_positions_corrected_st_rbf.csv",
    }
    if method_name not in suffix_by_method:
        raise ValueError(f"Unsupported drift-correction method: {method_name}")
    return prefix.with_name(prefix.name + suffix_by_method[method_name])


def get_motion_metric_config() -> tuple[str, str]:
    """Return the active per-track displacement column and display label."""
    mode = str(MOTION_CLASSIFICATION_MODE).strip().lower()
    if mode == "mean_otsu_log":
        return "mean_step_displacement", "average step displacement"
    if mode in {"median_gmm", "median_manual"}:
        return "median_step_displacement", "median step displacement"
    raise ValueError(f"Unsupported MOTION_CLASSIFICATION_MODE: {MOTION_CLASSIFICATION_MODE}")


def _fit_two_component_gmm_threshold(values: np.ndarray) -> float:
    """Fit a simple 1D two-component Gaussian mixture and return the density-intersection threshold."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5 or np.allclose(vals, vals[0]):
        return float(np.median(vals)) if vals.size else 0.0

    x = np.log1p(vals)
    q25, q75 = np.percentile(x, [25.0, 75.0])
    means = np.array([q25, q75], dtype=np.float64)
    vars_ = np.array([max(np.var(x), 1e-3), max(np.var(x), 1e-3)], dtype=np.float64)
    weights = np.array([0.5, 0.5], dtype=np.float64)

    for _ in range(100):
        stds = np.sqrt(np.maximum(vars_, 1e-6))
        pdfs = np.column_stack(
            [
                weights[k] * np.exp(-0.5 * ((x - means[k]) / stds[k]) ** 2) / max(stds[k], 1e-6)
                for k in range(2)
            ]
        )
        denom = np.maximum(np.sum(pdfs, axis=1, keepdims=True), 1e-12)
        resp = pdfs / denom
        nk = np.maximum(resp.sum(axis=0), 1e-6)
        new_weights = nk / float(x.size)
        new_means = (resp * x[:, None]).sum(axis=0) / nk
        new_vars = (resp * (x[:, None] - new_means[None, :]) ** 2).sum(axis=0) / nk
        if np.allclose(means, new_means, atol=1e-5) and np.allclose(vars_, new_vars, atol=1e-5):
            weights, means, vars_ = new_weights, new_means, np.maximum(new_vars, 1e-6)
            break
        weights, means, vars_ = new_weights, new_means, np.maximum(new_vars, 1e-6)

    order = np.argsort(means)
    weights = weights[order]
    means = means[order]
    vars_ = vars_[order]
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 2048)
    stds = np.sqrt(np.maximum(vars_, 1e-6))
    pdf0 = weights[0] * np.exp(-0.5 * ((grid - means[0]) / stds[0]) ** 2) / max(stds[0], 1e-6)
    pdf1 = weights[1] * np.exp(-0.5 * ((grid - means[1]) / stds[1]) ** 2) / max(stds[1], 1e-6)
    between = (grid >= means[0]) & (grid <= means[1])
    if np.any(between):
        idx_rel = int(np.argmin(np.abs(pdf0[between] - pdf1[between])))
        thr_log = float(grid[between][idx_rel])
    else:
        thr_log = float(0.5 * (means[0] + means[1]))
    thr = float(np.expm1(thr_log))
    return float(np.clip(thr, np.min(vals), np.max(vals)))


def determine_stationary_threshold(avg_disp_df: pd.DataFrame) -> float:
    """Estimate a threshold separating semi-stationary and migrating cells."""
    metric_col, _ = get_motion_metric_config()
    vals = pd.to_numeric(avg_disp_df[metric_col], errors="coerce").dropna().to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    mode = str(MOTION_CLASSIFICATION_MODE).strip().lower()
    if mode == "median_manual":
        return float(MANUAL_STATIONARY_THRESHOLD)
    if vals.size < 5 or np.allclose(vals, vals[0]):
        return float(np.median(vals))
    if mode == "median_gmm":
        return _fit_two_component_gmm_threshold(vals)
    thr_log = otsu_threshold(np.log1p(vals), bins=128, value_range=(float(np.min(np.log1p(vals))), float(np.max(np.log1p(vals)))))
    thr = float(np.expm1(thr_log))
    return float(np.clip(thr, np.min(vals), np.max(vals)))


def classify_tracks_by_displacement(avg_disp_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Attach stationary/migrating class labels using a shared threshold."""
    out = avg_disp_df.copy()
    metric_col, _ = get_motion_metric_config()
    out["motion_class"] = np.where(
        pd.to_numeric(out[metric_col], errors="coerce").to_numpy(np.float64) <= float(threshold),
        "stationary",
        "migrating",
    )
    return out


def compute_birth_death_counts_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Count births and deaths between consecutive frames, excluding the first frame."""
    times = np.array(sorted(pd.to_numeric(df["Time"], errors="coerce").dropna().unique()), dtype=np.float64)
    rows: list[dict[str, float]] = []
    if times.size < 2:
        return pd.DataFrame(columns=["Time", "births", "deaths"])
    for idx in range(1, len(times)):
        t_prev = float(times[idx - 1])
        t_cur = float(times[idx])
        prev_ids = set(pd.to_numeric(df.loc[df["Time"] == t_prev, "TrackID"], errors="coerce").dropna().to_numpy(np.float64).tolist())
        cur_ids = set(pd.to_numeric(df.loc[df["Time"] == t_cur, "TrackID"], errors="coerce").dropna().to_numpy(np.float64).tolist())
        rows.append(
            {
                "Time": t_cur,
                "births": float(len(cur_ids - prev_ids)),
                "deaths": float(len(prev_ids - cur_ids)),
            }
        )
    return pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)


def sem_or_zero(values: np.ndarray) -> float:
    """Return SEM or 0 if insufficient samples."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def run_statistical_test(values_gt: np.ndarray, values_pca: np.ndarray, paired: bool) -> dict[str, Any]:
    """Run a simple normality-aware statistical comparison."""
    gt = np.asarray(values_gt, dtype=np.float64)
    pca = np.asarray(values_pca, dtype=np.float64)
    gt = gt[np.isfinite(gt)]
    pca = pca[np.isfinite(pca)]
    result: dict[str, Any] = {
        "paired": bool(paired),
        "n_gt": int(gt.size),
        "n_pca": int(pca.size),
        "normality_test": "shapiro",
        "gt_normality_p": np.nan,
        "pca_normality_p": np.nan,
        "difference_normality_p": np.nan,
        "all_normal": False,
        "test_used": "not_applicable",
        "p_value": np.nan,
    }
    if gt.size == 0 or pca.size == 0:
        return result

    def _shapiro_safe(arr: np.ndarray) -> float:
        if arr.size < 3 or arr.size > 5000 or np.allclose(arr, arr[0]):
            return np.nan
        try:
            return float(sstats.shapiro(arr).pvalue)
        except Exception:
            return np.nan

    if paired:
        n = min(gt.size, pca.size)
        gt = gt[:n]
        pca = pca[:n]
        diff = gt - pca
        result["n_gt"] = int(n)
        result["n_pca"] = int(n)
        result["difference_normality_p"] = _shapiro_safe(diff)
        all_normal = bool(np.isfinite(result["difference_normality_p"]) and result["difference_normality_p"] > 0.05)
        result["all_normal"] = all_normal
        if n >= 2:
            if all_normal:
                test = sstats.ttest_rel(gt, pca, nan_policy="omit")
                result["test_used"] = "paired_t_test"
                result["p_value"] = float(test.pvalue)
            else:
                if np.any(np.abs(diff) > 1e-12):
                    test = sstats.wilcoxon(gt, pca, zero_method="wilcox")
                    result["test_used"] = "wilcoxon_signed_rank"
                    result["p_value"] = float(test.pvalue)
        return result

    result["gt_normality_p"] = _shapiro_safe(gt)
    result["pca_normality_p"] = _shapiro_safe(pca)
    all_normal = bool(
        np.isfinite(result["gt_normality_p"])
        and np.isfinite(result["pca_normality_p"])
        and result["gt_normality_p"] > 0.05
        and result["pca_normality_p"] > 0.05
    )
    result["all_normal"] = all_normal
    if gt.size >= 2 and pca.size >= 2:
        if all_normal:
            test = sstats.ttest_ind(gt, pca, equal_var=False, nan_policy="omit")
            result["test_used"] = "welch_t_test"
            result["p_value"] = float(test.pvalue)
        else:
            test = sstats.mannwhitneyu(gt, pca, alternative="two-sided")
            result["test_used"] = "mann_whitney_u"
            result["p_value"] = float(test.pvalue)
    return result


def annotate_pvalue(ax: Any, x0: float, x1: float, y: float, p_value: float) -> None:
    """Draw a simple significance bracket with a p-value label."""
    if not np.isfinite(p_value):
        return
    h = max(0.02 * max(y, 1.0), 0.1)
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text(0.5 * (x0 + x1), y + h, f"p={p_value:.3f}", ha="center", va="bottom", fontsize=8)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    if not p_values:
        return []
    arr = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(arr)
    adjusted = np.empty_like(arr)
    m = len(arr)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = float(arr[idx]) * float(m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def run_three_group_statistical_analysis(
    groups: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run normality-aware omnibus and post-hoc statistics for three groups."""
    clean_groups: dict[str, np.ndarray] = {}
    normality_rows: list[dict[str, Any]] = []

    def _shapiro_safe(arr: np.ndarray) -> float:
        if arr.size < 3 or arr.size > 5000 or np.allclose(arr, arr[0]):
            return np.nan
        try:
            return float(sstats.shapiro(arr).pvalue)
        except Exception:
            return np.nan

    for name, values in groups.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        clean_groups[name] = arr
        normality_rows.append(
            {
                "row_type": "normality",
                "group": str(name),
                "n": int(arr.size),
                "normality_test": "shapiro",
                "normality_p_value": _shapiro_safe(arr),
            }
        )

    analyzable = {name: arr for name, arr in clean_groups.items() if arr.size > 0}
    omnibus_row: dict[str, Any] = {
        "row_type": "omnibus",
        "groups": ",".join(analyzable.keys()),
        "global_test": "not_applicable",
        "global_p_value": np.nan,
        "all_groups_normal": False,
    }
    pairwise_rows: list[dict[str, Any]] = []
    if len(analyzable) < 2:
        return omnibus_row, [*normality_rows]

    all_normal = all(
        np.isfinite(row["normality_p_value"]) and float(row["normality_p_value"]) > 0.05
        for row in normality_rows
        if row["group"] in analyzable
    )
    omnibus_row["all_groups_normal"] = bool(all_normal)
    arrays = [arr for arr in analyzable.values()]
    if all_normal and all(arr.size >= 2 for arr in arrays):
        try:
            omnibus = sstats.f_oneway(*arrays)
            omnibus_row["global_test"] = "one_way_anova"
            omnibus_row["global_p_value"] = float(omnibus.pvalue)
        except Exception:
            pass
    elif all(arr.size >= 2 for arr in arrays):
        try:
            omnibus = sstats.kruskal(*arrays)
            omnibus_row["global_test"] = "kruskal_wallis"
            omnibus_row["global_p_value"] = float(omnibus.pvalue)
        except Exception:
            pass

    raw_pairwise_p: list[float] = []
    pair_indices: list[tuple[str, str, str]] = []
    names = list(analyzable.keys())
    for idx_a in range(len(names)):
        for idx_b in range(idx_a + 1, len(names)):
            name_a = names[idx_a]
            name_b = names[idx_b]
            arr_a = analyzable[name_a]
            arr_b = analyzable[name_b]
            if arr_a.size < 2 or arr_b.size < 2:
                p_val = np.nan
                test_used = "not_applicable"
            elif all_normal:
                test = sstats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
                p_val = float(test.pvalue)
                test_used = "welch_t_test"
            else:
                test = sstats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
                p_val = float(test.pvalue)
                test_used = "mann_whitney_u"
            raw_pairwise_p.append(p_val)
            pair_indices.append((name_a, name_b, test_used))

    finite_ps = [float(p) for p in raw_pairwise_p if np.isfinite(p)]
    adjusted_finite = holm_bonferroni(finite_ps)
    finite_iter = iter(adjusted_finite)
    for (name_a, name_b, test_used), raw_p in zip(pair_indices, raw_pairwise_p):
        adj_p = float(next(finite_iter)) if np.isfinite(raw_p) else np.nan
        pairwise_rows.append(
            {
                "row_type": "pairwise",
                "group_a": str(name_a),
                "group_b": str(name_b),
                "pairwise_test": str(test_used),
                "raw_p_value": float(raw_p) if np.isfinite(raw_p) else np.nan,
                "adjustment": "holm_bonferroni",
                "adjusted_p_value": adj_p,
            }
        )

    return omnibus_row, [*normality_rows, *pairwise_rows]


def plot_two_group_bar(
    values_gt: np.ndarray,
    values_other: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    stats_csv_path: Path | None = None,
    paired: bool = False,
    group_labels: tuple[str, str] = ("GT", "PCA"),
    show_title: bool = True,
    figsize: tuple[float, float] = (5.0 * CM, 4.2 * CM)) -> None:
    """Plot GT vs PCA mean bars with SEM, points, and optional test annotation."""
    gt = np.asarray(values_gt, dtype=np.float64)
    other = np.asarray(values_other, dtype=np.float64)
    gt = gt[np.isfinite(gt)]
    other = other[np.isfinite(other)]
    means = [float(np.mean(gt)) if gt.size else 0.0, float(np.mean(other)) if other.size else 0.0]
    sems = [sem_or_zero(gt), sem_or_zero(other)]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.array([0.0, 1.0], dtype=np.float64)
    ax.bar(x, means, yerr=sems, capsize=4, color=["#4C72B0", "#DD8452"], alpha=0.85, width=0.65)

    rng = np.random.default_rng(0)
    for xi, arr, color in zip(x.tolist(), [gt, other], ["#1f3552", "#7a3d14"]):
        if arr.size:
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            ax.scatter(np.full(arr.size, xi) + jitter, arr, s=16, color=color, alpha=0.8, zorder=3, lw=0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(group_labels[0]), str(group_labels[1])])
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title)
    

    stats_row = run_statistical_test(gt, other, paired=paired)
    ymax = max([m + s for m, s in zip(means, sems)] + ([float(np.max(gt))] if gt.size else [0.0]) + ([float(np.max(other))] if other.size else [0.0]))
    annotate_pvalue(ax, x0=0.0, x1=1.0, y=max(ymax * 1.03, 0.5), p_value=float(stats_row["p_value"]) if np.isfinite(stats_row["p_value"]) else np.nan)

    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)

    if stats_csv_path is not None:
        pd.DataFrame([stats_row]).to_csv(stats_csv_path, index=False)


def plot_total_count_bars(
    count_gt: int,
    count_other: int,
    title: str,
    ylabel: str,
    out_path: Path,
    group_labels: tuple[str, str] = ("CCA", "PCA"),
    show_title: bool = True,
    figsize: tuple[float, float] = (5.0 * CM, 4.2 * CM)) -> None:
    """Plot simple GT vs PCA total counts."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.array([0.0, 1.0], dtype=np.float64)
    heights = [float(count_gt), float(count_other)]
    bars = ax.bar(x, heights, color=["#4C72B0", "#DD8452"], alpha=0.85, width=0.65)
    for bar, val in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{int(round(val))}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(group_labels[0]), str(group_labels[1])])
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(title, fontsize=8)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)


def plot_birth_death_bars(
    gt_counts: pd.DataFrame,
    other_counts: pd.DataFrame,
    out_path: Path,
    stats_csv_path: Path,
    group_labels: tuple[str, str] = ("CCA", "PCA"),
    show_title: bool = True,
    figsize: tuple[float, float] = (10.0 * CM, 4.2 * CM)) -> None:
    """Plot births/deaths GT vs PCA as mean-per-frame bars with SEM and paired tests."""
    merged = gt_counts.merge(other_counts, on="Time", suffixes=("_gt", "_other"), how="inner")
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    for ax, metric, title in zip(axes, ["births", "deaths"], ["Births after t0", "Deaths after t0"]):
        gt = pd.to_numeric(merged[f"{metric}_gt"], errors="coerce").dropna().to_numpy(np.float64)
        other = pd.to_numeric(merged[f"{metric}_other"], errors="coerce").dropna().to_numpy(np.float64)
        means = [float(np.mean(gt)) if gt.size else 0.0, float(np.mean(other)) if other.size else 0.0]
        sems = [sem_or_zero(gt), sem_or_zero(other)]
        x = np.array([0.0, 1.0], dtype=np.float64)
        ax.bar(x, means, yerr=sems, capsize=4, color=["#4C72B0", "#DD8452"], alpha=0.85, width=0.65)
        rng = np.random.default_rng(0)
        for xi, arr, color in zip(x.tolist(), [gt, other], ["#1f3552", "#7a3d14"]):
            if arr.size:
                jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                ax.scatter(np.full(arr.size, xi) + jitter, arr, s=14, color=color, alpha=0.8, zorder=3, lw=0)
        stats_row = run_statistical_test(gt, other, paired=True)
        stats_row["comparison"] = metric
        rows.append(stats_row)
        ymax = max([m + s for m, s in zip(means, sems)] + ([float(np.max(gt))] if gt.size else [0.0]) + ([float(np.max(other))] if other.size else [0.0]))
        annotate_pvalue(ax, 0.0, 1.0, max(ymax * 1.03, 0.5), float(stats_row["p_value"]) if np.isfinite(stats_row["p_value"]) else np.nan)
        ax.set_xticks(x)
        ax.set_xticklabels([str(group_labels[0]), str(group_labels[1])])
        if show_title:
            ax.set_title(title)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(f"{metric}/frame")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(stats_csv_path, index=False)


def plot_stationary_migrating_count_bars(
    gt_class_df: pd.DataFrame,
    other_class_df: pd.DataFrame,
    out_path: Path,
    group_labels: tuple[str, str] = ("CCA", "PCA"),
    show_title: bool = True,
    figsize: tuple[float, float] = (6.0 * CM, 4.5 * CM)) -> None:
    """Plot total stationary vs migrating track counts for GT and PCA."""
    categories = ["stationary", "migrating"]
    gt_counts = gt_class_df["motion_class"].value_counts().to_dict()
    other_counts = other_class_df["motion_class"].value_counts().to_dict()
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    vals_gt = [float(gt_counts.get(cat, 0)) for cat in categories]
    vals_other = [float(other_counts.get(cat, 0)) for cat in categories]
    ax.bar(x - width / 2.0, vals_gt, width=width, color="#4C72B0", alpha=0.85, label=str(group_labels[0]))
    ax.bar(x + width / 2.0, vals_other, width=width, color="#DD8452", alpha=0.85, label=str(group_labels[1]))
    ax.set_xticks(x)
    ax.set_xticklabels(["stationary", "migrating"])
    ax.set_ylabel("unique tracks")
    if show_title:
        ax.set_title("Stationary vs migrating cells")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)


def plot_three_group_count_bars(
    gt_unique: int,
    pca_unique: int,
    driftcorr_unique: int,
    gt_stationary: int,
    pca_stationary: int,
    driftcorr_stationary: int,
    gt_migrating: int,
    pca_migrating: int,
    driftcorr_migrating: int,
    out_path: Path,
    show_title: bool = True,
) -> Path:
    """Plot grouped bars for unique, stationary and migrating track counts."""
    categories = ["unique tracks", "stationary cells", "migrating cells"]
    gt_vals = [float(gt_unique), float(gt_stationary), float(gt_migrating)]
    pca_vals = [float(pca_unique), float(pca_stationary), float(pca_migrating)]
    drift_vals = [float(driftcorr_unique), float(driftcorr_stationary), float(driftcorr_migrating)]
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.5 * CM, 3.0 * CM))
    ax.bar(x - width, gt_vals, width=width, color="#4C72B0", alpha=0.85, label="CCA")
    ax.bar(x, pca_vals, width=width, color="#DD8452", alpha=0.85, label="PCA")
    ax.bar(x + width, drift_vals, width=width, color="#55A868", alpha=0.85, label="DriftCorr")
    ax.set_xticks(x)
    ax.set_xticklabels(["unique", "stationary", "migrating"])
    ax.set_ylabel("tracks")
    ax.set_ylim(bottom=0.0)
    if show_title:
        ax.set_title("Track-count summary", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    return out_path


def plot_average_displacement_histogram(
    gt_avg_disp: pd.DataFrame,
    other_avg_disp: pd.DataFrame,
    threshold: float,
    out_path: Path,
    group_labels: tuple[str, str] = ("GT", "PCA"),
    show_title: bool = True,
    figsize: tuple[float, float] = (6.2 * CM, 4.5 * CM)) -> None:
    """Plot GT/PCA average-displacement histograms with threshold marker."""
    gt = pd.to_numeric(gt_avg_disp["mean_step_displacement"], errors="coerce").dropna().to_numpy(np.float64)
    other = pd.to_numeric(other_avg_disp["mean_step_displacement"], errors="coerce").dropna().to_numpy(np.float64)
    max_val = max(float(np.max(gt)) if gt.size else 0.0, float(np.max(other)) if other.size else 0.0, float(threshold))
    bins = np.linspace(0.0, max(max_val * 1.05, 1.0), 20)
    fig, ax = plt.subplots(figsize=figsize)
    if gt.size:
        ax.hist(gt, bins=bins, alpha=0.45, color="#4C72B0", label=str(group_labels[0]))
    if other.size:
        ax.hist(other, bins=bins, alpha=0.45, color="#DD8452", label=str(group_labels[1]))
    ax.axvline(float(threshold), color="black", linestyle="--", linewidth=1.2, label=f"migration\nthreshold={threshold:.3f}")
    ax.set_xlabel("average step displacement")
    ax.set_ylabel("tracks")
    if show_title:
        ax.set_title("Average displacement histogram")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)


def plot_average_displacement_histogram_threeway(
    gt_avg_disp: pd.DataFrame,
    pca_avg_disp: pd.DataFrame,
    driftcorr_avg_disp: pd.DataFrame,
    threshold: float,
    out_path: Path,
    show_title: bool = True,
) -> Path:
    """Overlay histograms of the active step-displacement metric for CCA, PCA and drift-corrected tracks."""
    metric_col, axis_label = get_motion_metric_config()
    arrays = [
        pd.to_numeric(gt_avg_disp[metric_col], errors="coerce").dropna().to_numpy(np.float64),
        pd.to_numeric(pca_avg_disp[metric_col], errors="coerce").dropna().to_numpy(np.float64),
        pd.to_numeric(driftcorr_avg_disp[metric_col], errors="coerce").dropna().to_numpy(np.float64),
    ]
    max_val = max([float(np.max(arr)) if arr.size else 0.0 for arr in arrays] + [float(threshold), 1.0])
    bins = np.linspace(0.0, max_val * 1.05, 24)
    fig, ax = plt.subplots(figsize=(6.2 * CM, 3.5 * CM))
    labels_colors = [("CCA", "#4C72B0"), ("PCA", "#DD8452"), ("DriftCorr", "#55A868")]
    """ for arr, (label, color) in zip(arrays, labels_colors):
        if arr.size:
            ax.hist(arr, bins=bins, alpha=0.35, color=color, label=label) """
    for arr, (label, color) in zip(arrays, labels_colors):
        if not arr.size:
            continue
        if label == "DriftCorr":
            hatch_color = (0.184, 0.365, 0.227, 0.45) # RGBA version of "#2f5d3a" with alpha for visibility
            with matplotlib.rc_context({"hatch.linewidth": 0.8}):
                _, _, patches = ax.hist(arr, bins=bins, label=label, facecolor="none",
                    edgecolor=hatch_color, linewidth=0.0, hatch="///", zorder=3)
            for patch in patches:
                patch.set_facecolor("none")
                patch.set_edgecolor(hatch_color)
                patch.set_linewidth(0.0)
                patch.set_hatch("///")
        else:
            ax.hist(arr, bins=bins, alpha=0.35, color=color, label=label, linewidth=0.0, zorder=2)
    ax.axvline(float(threshold), color="black", linestyle="--", linewidth=1.2, label=f"migration\nthreshold={threshold:.3f}")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("tracks")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    if show_title:
        ax.set_title("Average displacement histogram", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    return out_path


def plot_motion_class_average_displacement_threeway(
    gt_class_df: pd.DataFrame,
    pca_class_df: pd.DataFrame,
    driftcorr_class_df: pd.DataFrame,
    out_path: Path,
    stats_csv_path: Path | None = None,
    group_labels: tuple[str, str, str] = ("CCA", "PCA", "DriftCorr"),
) -> Path:
    """Plot the active displacement metric for stationary vs migrating tracks across CCA/PCA/driftcorr."""
    stats_rows: list[dict[str, Any]] = []
    metric_col, axis_label = get_motion_metric_config()
    fig, axes = plt.subplots(1, 2, figsize=(9.0 * CM, 3.6 * CM), sharey=False)
    for ax, motion_class, title in zip(axes, ["stationary", "migrating"], ["Stationary", "Migrating"]):
        groups = [
            pd.to_numeric(gt_class_df.loc[gt_class_df["motion_class"] == motion_class, metric_col], errors="coerce").dropna().to_numpy(np.float64),
            pd.to_numeric(pca_class_df.loc[pca_class_df["motion_class"] == motion_class, metric_col], errors="coerce").dropna().to_numpy(np.float64),
            pd.to_numeric(driftcorr_class_df.loc[driftcorr_class_df["motion_class"] == motion_class, metric_col], errors="coerce").dropna().to_numpy(np.float64),
        ]
        means = [float(np.mean(arr)) if arr.size else 0.0 for arr in groups]
        medians = [float(np.median(arr)) if arr.size else 0.0 for arr in groups]
        sems = [sem_or_zero(arr) for arr in groups]
        x = np.arange(3, dtype=np.float64)
        group_map = {
            str(group_labels[0]): groups[0],
            str(group_labels[1]): groups[1],
            str(group_labels[2]): groups[2],
        }
        omnibus_row, test_rows = run_three_group_statistical_analysis(group_map)
        stats_rows.append(
            {
                "motion_class": str(motion_class),
                "row_type": "summary",
                "groups": ",".join(group_map.keys()),
                "global_test": omnibus_row.get("global_test", "not_applicable"),
                "global_p_value": omnibus_row.get("global_p_value", np.nan),
                "all_groups_normal": omnibus_row.get("all_groups_normal", False),
            }
        )
        for label, arr, mean_val, median_val, sem_val in zip(group_labels, groups, means, medians, sems):
            stats_rows.append(
                {
                    "motion_class": str(motion_class),
                    "row_type": "group_summary",
                    "group": str(label),
                    "n": int(arr.size),
                    "mean": float(mean_val),
                    "median": float(median_val),
                    "sem": float(sem_val),
                }
            )
        stats_rows.extend([{**row, "motion_class": str(motion_class)} for row in [omnibus_row, *test_rows]])

        # ax.bar(
        #     x,
        #     means,
        #     yerr=sems,
        #     capsize=4,
        #     color=["#4C72B0", "#DD8452", "#55A868"],
        #     alpha=0.85,
        #     width=0.65,
        #     zorder=2,
        # )
        # rng = np.random.default_rng(0)
        # for idx, arr, color in zip(x.tolist(), groups, ["#1f3552", "#7a3d14", "#1d5f34"]):
        #     if arr.size:
        #         jitter = rng.uniform(-0.08, 0.08, size=arr.size)
        #         ax.scatter(np.full(arr.size, idx) + jitter, arr, s=10, color=color, alpha=0.7, zorder=3, lw=0)
        # for xi, mean_val, sem_val, color in zip(x, means, sems, ["#0b1f33", "#4a2208", "#10351d"]):
        #     ax.errorbar(xi, mean_val, yerr=sem_val, fmt="none", ecolor=color, elinewidth=1.25, capsize=4, capthick=1.25, zorder=5)
        #     ax.scatter(xi, mean_val, s=14, color=color, zorder=6)

        vp = ax.violinplot(
            groups,
            positions=x,
            widths=0.65,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        violin_colors = ["#4C72B0", "#DD8452", "#55A868"]
        for body, color in zip(vp["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_linewidth(0)
            body.set_alpha(0.6)
            body.set_zorder(1)
        for xi, median_val, color in zip(x, medians, ["#0b1f33", "#4a2208", "#10351d"]):
            ax.scatter(xi, median_val, s=14, color=color, zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(list(group_labels))
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(axis_label.replace(" ", "\n", 1))

        data_max = max([float(np.max(arr)) if arr.size else 0.0 for arr in groups] + [0.0])
        data_span = data_max if data_max > 0.0 else 1.0
        pairwise_rows = [row for row in test_rows if row.get("row_type") == "pairwise"]
        pair_lookup = {
            (str(row.get("group_a")), str(row.get("group_b"))): float(row.get("adjusted_p_value", np.nan))
            for row in pairwise_rows
        }
        annotation_specs = [
            (0.0, 1.0, pair_lookup.get((str(group_labels[0]), str(group_labels[1])), np.nan)),
            (1.0, 2.0, pair_lookup.get((str(group_labels[1]), str(group_labels[2])), np.nan)),
            (0.0, 2.0, pair_lookup.get((str(group_labels[0]), str(group_labels[2])), np.nan)),
        ]
        base_y = data_max + 0.08 * data_span
        step_y = 0.26 * data_span
        for idx, (x0, x1, p_value) in enumerate(annotation_specs):
            if np.isfinite(p_value):
                annotate_pvalue(ax, x0, x1, base_y + idx * step_y, float(p_value))
        top_margin = 0.08 if motion_class == "migrating" else 0.18
        ax.set_ylim(bottom=0, top=base_y + max(len(annotation_specs) - 1, 0) * step_y + top_margin * data_span)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    if stats_csv_path is not None:
        stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stats_rows).to_csv(stats_csv_path, index=False)
    return out_path


def run_regime_summary_analysis(
    gt_tracks_df: pd.DataFrame,
    association_root: Path,
    drift_root: Path,
    drift_summary_df: pd.DataFrame,
    regime_name: str,
    out_dir: Path) -> dict[str, Path]:
    """Create regime summary plots: counts/birth-death vs PCA, motion vs drift-corrected result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    method_dir = association_root / BEST_ASSOCIATION_METHOD
    inferred_tracks_path = method_dir / f"{BEST_ASSOCIATION_METHOD}_inferred_tracks.csv"
    if not inferred_tracks_path.exists():
        raise FileNotFoundError(f"Best PCA inferred tracks not found: {inferred_tracks_path}")
    pca_tracks_df = ensure_numeric_tracks(pd.read_csv(inferred_tracks_path))
    gt_tracks_df = ensure_numeric_tracks(gt_tracks_df)
    if drift_summary_df.empty:
        raise ValueError("Drift-correction summary is empty; cannot build motion comparison plots.")
    best_drift_method = str(drift_summary_df.sort_values(["overall_rms_error", "drift_bias_abs"]).iloc[0]["method"])
    corrected_tracks_path = corrected_tracks_csv_path(drift_root=drift_root, regime_name=regime_name, method_name=best_drift_method)
    if not corrected_tracks_path.exists():
        raise FileNotFoundError(f"Best drift-corrected tracks not found: {corrected_tracks_path}")
    driftcorr_tracks_df = ensure_numeric_tracks(pd.read_csv(corrected_tracks_path))

    gt_unique = int(pd.to_numeric(gt_tracks_df["TrackID"], errors="coerce").dropna().nunique())
    pca_unique = int(pd.to_numeric(pca_tracks_df["TrackID"], errors="coerce").dropna().nunique())
    driftcorr_unique = int(pd.to_numeric(driftcorr_tracks_df["TrackID"], errors="coerce").dropna().nunique())

    gt_birthdeath = compute_birth_death_counts_by_time(gt_tracks_df)
    pca_birthdeath = compute_birth_death_counts_by_time(pca_tracks_df)
    birthdeath_plot = out_dir / "gt_vs_pca_births_deaths_barplot.pdf"
    birthdeath_stats = out_dir / "gt_vs_pca_births_deaths_stats.csv"
    plot_birth_death_bars(gt_birthdeath, pca_birthdeath, out_path=birthdeath_plot, stats_csv_path=birthdeath_stats,
                          show_title=False, figsize=(5.15 * CM, 3.0 * CM))

    gt_avg_disp = compute_track_average_displacements(gt_tracks_df)
    pca_avg_disp = compute_track_average_displacements(pca_tracks_df)
    driftcorr_avg_disp = compute_track_average_displacements(driftcorr_tracks_df)
    threshold = determine_stationary_threshold(gt_avg_disp)
    gt_class = classify_tracks_by_displacement(gt_avg_disp, threshold)
    driftcorr_class = classify_tracks_by_displacement(driftcorr_avg_disp, threshold)
    pca_class = classify_tracks_by_displacement(pca_avg_disp, threshold)

    three_bar_plot = out_dir / "gt_pca_driftcorr_track_count_summary.pdf"
    plot_three_group_count_bars(
        gt_unique=gt_unique,
        pca_unique=pca_unique,
        driftcorr_unique=driftcorr_unique,
        gt_stationary=int((gt_class["motion_class"] == "stationary").sum()),
        pca_stationary=int((pca_class["motion_class"] == "stationary").sum()),
        driftcorr_stationary=int((driftcorr_class["motion_class"] == "stationary").sum()),
        gt_migrating=int((gt_class["motion_class"] == "migrating").sum()),
        pca_migrating=int((pca_class["motion_class"] == "migrating").sum()),
        driftcorr_migrating=int((driftcorr_class["motion_class"] == "migrating").sum()),
        out_path=three_bar_plot,
        show_title=False,
    )

    hist_plot = out_dir / "gt_pca_driftcorr_average_displacement_histogram.pdf"
    plot_average_displacement_histogram_threeway(
        gt_avg_disp=gt_avg_disp,
        pca_avg_disp=pca_avg_disp,
        driftcorr_avg_disp=driftcorr_avg_disp,
        threshold=threshold,
        out_path=hist_plot,
        show_title=False,
    )

    motion_bar_plot = out_dir / "gt_pca_driftcorr_average_displacement_stationary_vs_migrating.pdf"
    motion_bar_stats = out_dir / "gt_pca_driftcorr_average_displacement_stationary_vs_migrating_stats.csv"
    plot_motion_class_average_displacement_threeway(
        gt_class_df=gt_class,
        pca_class_df=pca_class,
        driftcorr_class_df=driftcorr_class,
        out_path=motion_bar_plot,
        stats_csv_path=motion_bar_stats,
    )

    summary_csv = out_dir / "gt_vs_pca_regime_summary.csv"
    metric_col, _ = get_motion_metric_config()
    pd.DataFrame(
        [
            {
                "gt_unique_tracks": gt_unique,
                "pca_unique_tracks": pca_unique,
                "driftcorr_unique_tracks": driftcorr_unique,
                "motion_classification_mode": str(MOTION_CLASSIFICATION_MODE),
                "motion_metric_column": str(metric_col),
                "stationary_threshold": float(threshold),
                "best_driftcorr_method_from_pca_tracks": str(best_drift_method),
                "gt_stationary_tracks": int((gt_class["motion_class"] == "stationary").sum()),
                "pca_stationary_tracks": int((pca_class["motion_class"] == "stationary").sum()),
                "gt_migrating_tracks": int((gt_class["motion_class"] == "migrating").sum()),
                "pca_migrating_tracks": int((pca_class["motion_class"] == "migrating").sum()),
                "driftcorr_stationary_tracks": int((driftcorr_class["motion_class"] == "stationary").sum()),
                "driftcorr_migrating_tracks": int((driftcorr_class["motion_class"] == "migrating").sum()),
            }
        ]
    ).to_csv(summary_csv, index=False)

    return {
        "birthdeath_plot": birthdeath_plot,
        "birthdeath_stats": birthdeath_stats,
        "three_bar_plot": three_bar_plot,
        "histogram_plot": hist_plot,
        "motion_bar_plot": motion_bar_plot,
        "motion_bar_stats": motion_bar_stats,
        "summary_csv": summary_csv,
    }


def build_drift_summary_extended(summary_df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """Convert GT summary metrics into a benchmark-style long summary table."""
    metric_cols = ["overall_rms_error", "median_error_norm", "p95_error_norm", "drift_bias_abs"]
    rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        method_name = str(getattr(row, "method"))
        for metric_name in metric_cols:
            if not hasattr(row, metric_name):
                continue
            metric_value = float(getattr(row, metric_name))
            rows.append(
                {
                    "mode": str(dataset_label),
                    "config": "baseline_suite",
                    "method": method_name,
                    "metric_name": metric_name,
                    "mean": metric_value,
                    "std": np.nan,
                    "n": 1,
                }
            )
    return pd.DataFrame(rows)


def plot_drift_summary_heatmap(summary_ext: pd.DataFrame, 
                               metric_name: str, 
                               out_path: Path,
                               title: str,
                               figure_size: tuple[float, float] = (7.2, 4.8)) -> Path | None:
    """Plot one dataset x methods heatmap for a drift-correction metric."""
    cur = summary_ext[summary_ext["metric_name"].astype(str) == str(metric_name)].copy()
    if cur.empty:
        return None
    pivot = (
        cur.pivot_table(index="method", columns="mode", values="mean", aggfunc="mean", observed=False)
        .sort_index(axis=0)
        .sort_index(axis=1))
    return plot_heatmap_from_pivot(pivot, title=title, cbar_label=metric_name, out_path=out_path, figsize=figure_size, show_title=False, show_xlabels=False)


def plot_drift_rms_vs_p95_scatter(summary_ext: pd.DataFrame, out_path: Path) -> Path | None:
    """Plot a compact RMS-vs-P95 scatter analogous to the benchmark summary."""
    rms_tab = summary_ext[summary_ext["metric_name"].astype(str) == "overall_rms_error"][["mode", "method", "mean"]].rename(columns={"mean": "rms_mean"})
    p95_tab = summary_ext[summary_ext["metric_name"].astype(str) == "p95_error_norm"][["mode", "method", "mean"]].rename(columns={"mean": "p95_mean"})
    scatter_df = rms_tab.merge(p95_tab, on=["mode", "method"], how="inner")
    if scatter_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.0 * CM, 6.0 * CM))
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(sorted(scatter_df["method"].astype(str).unique().tolist()))}
    for row in scatter_df.itertuples(index=False):
        ax.scatter(
            float(row.rms_mean),
            float(row.p95_mean),
            s=30,
            color=colors[str(row.method)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.0,
        )
        ax.text(float(row.rms_mean), float(row.p95_mean), str(row.method), fontsize=5, ha="left", va="bottom")
    ax.set_xlabel("RMS (mean)")
    ax.set_ylabel("P95 error norm (mean)")
    ax.set_title("Baseline suite: RMS vs P95")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    return out_path


def run_drift_correction_suite(
    gt_tracks_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    association_root: Path,
    out_root: Path,
    dataset_label: str,
    regime_name: str,
) -> tuple[pd.DataFrame, list[Path]]:
    """Run benchmark-style drift correction on best inferred association tracks."""
    out_root.mkdir(parents=True, exist_ok=True)
    method_dir = association_root / BEST_ASSOCIATION_METHOD
    inferred_tracks_path = method_dir / f"{BEST_ASSOCIATION_METHOD}_inferred_tracks.csv"
    mapping_path = method_dir / f"{BEST_ASSOCIATION_METHOD}_mapping_table.csv"
    if not inferred_tracks_path.exists():
        raise FileNotFoundError(f"Best association tracks not found: {inferred_tracks_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Best association mapping not found: {mapping_path}")

    df_inferred = ensure_numeric_tracks(pd.read_csv(inferred_tracks_path))
    mapping_df = pd.read_csv(mapping_path)
    chosen_mapping_df = choose_one_to_one_track_mapping(mapping_df)
    chosen_mapping_path = out_root / f"{BEST_ASSOCIATION_METHOD}_eval_track_mapping.csv"
    chosen_mapping_df.to_csv(chosen_mapping_path, index=False)

    opts = make_baseline_options_from_sweeps()
    out_prefix = out_root / f"{INPUT_TIF.stem}_{regime_name}_inferred"
    corr_by_method = run_baseline_methods_for_dataframe(
        df=df_inferred,
        out_prefix=out_prefix,
        methods=list(BASELINE_METHODS),
        opts=opts,
        use_stationary_subset=bool(opts.get("use_stationary_subset", False)))

    eval_summary_df, eval_details = run_all_methods_and_compare(df_inferred, corr_by_method)
    eval_artifacts = save_evaluation_outputs(out_prefix, eval_summary_df, eval_details)

    gt_eval_df = ensure_numeric_tracks(gt_tracks_df)
    drift_gt_df = build_rotation_drift_groundtruth(df_groundtruth=gt_tracks_df, df_rotated=observed_df)
    df_inferred_eval = remap_tracks_for_gt_evaluation(df_inferred, chosen_mapping_df)
    corr_eval_by_method = {name: remap_tracks_for_gt_evaluation(df_corr, chosen_mapping_df) for name, df_corr in corr_by_method.items()}

    visual_paths: list[Path] = []
    for method_name, df_corr_eval in corr_eval_by_method.items():
        visual_outputs = save_visual_inspection_plots(
            df_orig=df_inferred_eval,
            df_corr=df_corr_eval,
            out_prefix=out_prefix,
            method_name=method_name,
            df_groundtruth=gt_eval_df,
            use_data_box_aspect_tracks_over_time=True,
        )
        visual_paths.extend(visual_outputs.values())

    gt_outputs = save_groundtruth_evaluation_outputs(
        base_out_prefix=out_prefix,
        groundtruth_df=gt_eval_df,
        corrected_by_method=corr_eval_by_method,
        drift_field_groundtruth=drift_gt_df)
    gt_summary_path = gt_outputs["gt_metrics_summary"]
    gt_summary_df = pd.read_csv(gt_summary_path) if gt_summary_path.exists() else pd.DataFrame()

    summary_ext = build_drift_summary_extended(gt_summary_df, dataset_label=dataset_label)

    # sort summary_ext according the values in the "method" column in such a way, 
    # that the order is "rigid", "local", "rbf" and "st_rbf":
    method_order = ["rigid", "local", "rbf", "st_rbf"]
    summary_ext["method"] = pd.Categorical(summary_ext["method"], categories=method_order, ordered=True)
    summary_ext = summary_ext.sort_values("method").reset_index(drop=True)
    summary_ext_path = out_root / "baseline_suite_summary_extended.csv"
    summary_ext.to_csv(summary_ext_path, index=False)

    saved_paths: list[Path] = [chosen_mapping_path, summary_ext_path, *list(eval_artifacts.values()), *visual_paths, *list(gt_outputs.values())]
    metric_specs = [
        ("overall_rms_error", "Overall RMS"),
        ("median_error_norm", "Median error"),
        ("p95_error_norm", "P95 error"),
        ("drift_bias_abs", "Drift bias")]
    for metric_name, title in metric_specs:
        out_path = out_root / f"baseline_suite_{metric_name}_heatmap_plot.pdf"
        plotted = plot_drift_summary_heatmap(summary_ext, metric_name=metric_name, out_path=out_path, title=title,
                                             figure_size=(2.45 * CM, 2.95 * CM))
        if plotted is not None:
            saved_paths.append(plotted)
    scatter_path = plot_drift_rms_vs_p95_scatter(summary_ext, out_root / "baseline_suite_rms_vs_p95_scatter_plot.pdf")
    if scatter_path is not None:
        saved_paths.append(scatter_path)

    sweep_params_df = pd.DataFrame(
        [
            {
                "best_association_method": BEST_ASSOCIATION_METHOD,
                "rigid_keep_fraction": float(opts["keep_fraction"]),
                "rigid_inlier_reestimate_iters": int(opts["inlier_reestimate_iters"]),
                "local_k_neighbors": int(opts["k_neighbors"]),
                "local_ell_um_field": float(opts["ell_um_field"]),
                "rbf_sigma": float(opts["sigma"]),
                "rbf_m_centers": int(opts["m_centers"]),
                "st_rbf_ell_space": float(opts["st_ell_space"]),
                "st_rbf_ell_time": float(opts["st_ell_time"]),
                "st_rbf_use_stationary_subset": bool(opts.get("use_stationary_subset", False)),
            }
        ]
    )
    sweep_params_path = out_root / "selected_baseline_parameters.csv"
    sweep_params_df.to_csv(sweep_params_path, index=False)
    saved_paths.append(sweep_params_path)

    if not gt_summary_df.empty:
        gt_summary_sorted = gt_summary_df.sort_values("overall_rms_error").reset_index(drop=True)
    else:
        gt_summary_sorted = gt_summary_df
    return gt_summary_sorted, saved_paths


def run_drift_correction_suite_gt_input(
    gt_tracks_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    out_root: Path,
    dataset_label: str,
    regime_name: str,
) -> tuple[pd.DataFrame, list[Path]]:
    """Run benchmark-style drift correction directly on observed GT-ID tracks."""
    out_root.mkdir(parents=True, exist_ok=True)
    df_input = ensure_numeric_tracks(observed_df)
    gt_eval_df = ensure_numeric_tracks(gt_tracks_df)
    drift_gt_df = build_rotation_drift_groundtruth(df_groundtruth=gt_tracks_df, df_rotated=observed_df)

    opts = make_baseline_options_from_sweeps()
    out_prefix = out_root / f"{INPUT_TIF.stem}_{regime_name}_gttrackinput"
    corr_by_method = run_baseline_methods_for_dataframe(
        df=df_input,
        out_prefix=out_prefix,
        methods=list(BASELINE_METHODS),
        opts=opts,
        use_stationary_subset=bool(opts.get("use_stationary_subset", False)),
    )

    eval_summary_df, eval_details = run_all_methods_and_compare(df_input, corr_by_method)
    eval_artifacts = save_evaluation_outputs(out_prefix, eval_summary_df, eval_details)

    visual_paths: list[Path] = []
    for method_name, df_corr in corr_by_method.items():
        visual_outputs = save_visual_inspection_plots(
            df_orig=df_input,
            df_corr=df_corr,
            out_prefix=out_prefix,
            method_name=method_name,
            df_groundtruth=gt_eval_df,
            use_data_box_aspect_tracks_over_time=True,
        )
        visual_paths.extend(visual_outputs.values())

    gt_outputs = save_groundtruth_evaluation_outputs(
        base_out_prefix=out_prefix,
        groundtruth_df=gt_eval_df,
        corrected_by_method=corr_by_method,
        drift_field_groundtruth=drift_gt_df,
    )
    gt_summary_path = gt_outputs["gt_metrics_summary"]
    gt_summary_df = pd.read_csv(gt_summary_path) if gt_summary_path.exists() else pd.DataFrame()

    summary_ext = build_drift_summary_extended(gt_summary_df, dataset_label=f"{dataset_label}_gttrackinput")
    method_order = ["rigid", "local", "rbf", "st_rbf"]
    summary_ext["method"] = pd.Categorical(summary_ext["method"], categories=method_order, ordered=True)
    summary_ext = summary_ext.sort_values("method").reset_index(drop=True)
    summary_ext_path = out_root / "baseline_suite_summary_extended.csv"
    summary_ext.to_csv(summary_ext_path, index=False)

    saved_paths: list[Path] = [summary_ext_path, *list(eval_artifacts.values()), *visual_paths, *list(gt_outputs.values())]
    metric_specs = [
        ("overall_rms_error", "Overall RMS"),
        ("median_error_norm", "Median error"),
        ("p95_error_norm", "P95 error"),
        ("drift_bias_abs", "Drift bias"),
    ]
    for metric_name, title in metric_specs:
        out_path = out_root / f"baseline_suite_{metric_name}_heatmap_plot.pdf"
        plotted = plot_drift_summary_heatmap(
            summary_ext,
            metric_name=metric_name,
            out_path=out_path,
            title=title,
            figure_size=(2.65 * CM, 3.15 * CM),
        )
        if plotted is not None:
            saved_paths.append(plotted)
    scatter_path = plot_drift_rms_vs_p95_scatter(summary_ext, out_root / "baseline_suite_rms_vs_p95_scatter_plot.pdf")
    if scatter_path is not None:
        saved_paths.append(scatter_path)

    sweep_params_df = pd.DataFrame(
        [
            {
                "input_source": "groundtruth_trackids",
                "rigid_keep_fraction": float(opts["keep_fraction"]),
                "rigid_inlier_reestimate_iters": int(opts["inlier_reestimate_iters"]),
                "local_k_neighbors": int(opts["k_neighbors"]),
                "local_ell_um_field": float(opts["ell_um_field"]),
                "rbf_sigma": float(opts["sigma"]),
                "rbf_m_centers": int(opts["m_centers"]),
                "st_rbf_ell_space": float(opts["st_ell_space"]),
                "st_rbf_ell_time": float(opts["st_ell_time"]),
                "st_rbf_use_stationary_subset": bool(opts.get("use_stationary_subset", False)),
            }
        ]
    )
    sweep_params_path = out_root / "selected_baseline_parameters.csv"
    sweep_params_df.to_csv(sweep_params_path, index=False)
    saved_paths.append(sweep_params_path)

    if not gt_summary_df.empty:
        gt_summary_sorted = gt_summary_df.sort_values("overall_rms_error").reset_index(drop=True)
    else:
        gt_summary_sorted = gt_summary_df
    return gt_summary_sorted, saved_paths


def run_blind_point_cloud_association_suite(
    blind_csv_path: Path,
    gt_tracks_df: pd.DataFrame,
    unblinding_df: pd.DataFrame,
    out_root: Path,
    dataset_label: str,
    plot_reference_tracks_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Run the 6 HCV methods on the blind CSV and save benchmark-like outputs.
    
    DEBUG:
    blind_csv_path=blind_path
    gt_tracks_df=regime_tracks_df
    unblinding_df=unblinding_df
    out_root=association_root
    dataset_label=dataset_label
    plot_reference_tracks_df=regime_tracks_df
    
    """
    out_root.mkdir(parents=True, exist_ok=True)
    blind_frames = load_blind_frames(blind_csv_path)
    gt_tracks_with_pointindex = attach_pointindex_to_gt_tracks(gt_tracks_df, unblinding_df)
    plot_reference_df = gt_tracks_df if plot_reference_tracks_df is None else plot_reference_tracks_df
    plot_reference_with_pointindex = attach_pointindex_to_gt_tracks(plot_reference_df, unblinding_df)
    gt_frames = tracks_df_to_frames(gt_tracks_with_pointindex)
    plot_reference_frames = tracks_df_to_frames(plot_reference_with_pointindex)
    dx_scale_est = estimate_dx_scale_from_blind_frames(blind_frames)
    best_configs = load_best_hcv_configs(HCV_BENCHMARK_STATS_PATH)

    summary_rows: list[dict[str, Any]] = []
    saved_paths: list[Path] = []

    for method_cfg in HCV_METHOD_DEFS:
        # method_cfg = HCV_METHOD_DEFS[0]
        method_name = str(method_cfg["name"])
        method_dir = out_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        gating, params = make_hcv_gating_and_params(
            method_cfg=method_cfg,
            best_cfg=best_configs.get(method_name, {}),
            dx_scale_est=dx_scale_est,
        )
        assoc = run_hcv_method(blind_frames=blind_frames, method_cfg=method_cfg, gating=gating, params=params)

        tracks_path = method_dir / f"{method_name}_inferred_tracks.csv"
        links_path = method_dir / f"{method_name}_links.csv"
        frame_stats_path = method_dir / f"{method_name}_frame_stats.csv"
        assoc.tracks_df.to_csv(tracks_path, index=False)
        assoc.links_df.to_csv(links_path, index=False)
        assoc.frame_stats_df.to_csv(frame_stats_path, index=False)
        saved_paths.extend([tracks_path, links_path, frame_stats_path])

        summary_gt, per_time_gt, mapping_df = evaluate_against_groundtruth(
            inferred_tracks_df=assoc.tracks_df,
            point_to_gt_df=unblinding_df,
            links_df=assoc.links_df,
            gt_tracks_df=gt_tracks_with_pointindex)
        accuracy_df = compute_accuracy_over_time_table(gt_tracks_with_pointindex, assoc.tracks_df)
        accuracy_path = method_dir / f"{method_name}_accuracy_over_time.pdf"
        rms_path = method_dir / f"{method_name}_rms_over_time.pdf"
        panel_path = method_dir / f"{method_name}_panel_3d.pdf"

        if not per_time_gt.empty:
            per_time_gt = per_time_gt.merge(accuracy_df, on="Time", how="left")
        else:
            per_time_gt = accuracy_df.copy()
        per_time_path = method_dir / f"{method_name}_per_time_metrics.csv"
        mapping_path = method_dir / f"{method_name}_mapping_table.csv"
        per_time_gt.to_csv(per_time_path, index=False)
        mapping_df.to_csv(mapping_path, index=False)
        saved_paths.extend([per_time_path, mapping_path])

        plot_3d_frames_comparison(
            frames_gt=plot_reference_frames,
            frames_blind=blind_frames,
            frames_inferred=tracks_df_to_frames(assoc.tracks_df),
            out_path=panel_path,
            gt_tracks_df=plot_reference_with_pointindex,
            inferred_tracks_df=assoc.tracks_df,
            title=f"{dataset_label} | {method_name}",
            use_data_box_aspect=True,
            view_elev=25.0,
            view_azim=-60.0,
        )
        plot_rms_over_time(gt_tracks_with_pointindex, assoc.tracks_df, rms_path)
        plot_accuracy_over_time(gt_tracks_with_pointindex, assoc.tracks_df, accuracy_path)
        saved_paths.extend([panel_path, rms_path, accuracy_path])

        summary_row = {
            "method": method_name,
            "dx_scale_est": float(dx_scale_est),
            "gate_max_dist": float(gating["max_dist"]),
            "gate_keep_percentile": float(gating["keep_percentile"]),
            "max_gap": int(params.get("max_gap", 0)),
            **{f"assoc_{k}": v for k, v in assoc.quality.items()},
            **summary_gt,
            "accuracy_mean": float(pd.to_numeric(accuracy_df.get("accuracy", np.nan), errors="coerce").mean()) if not accuracy_df.empty else float("nan"),
        }
        for key, value in params.items():
            if key in {"metric", "motion_refine_iters", "max_dist_factor", "keep_percentile", "dx_scale", "sigma_init_mode", "nn_scale_factor", "lambda_switch", "stage1_factor", "stage1_keep_percentile", "stage2_keep_percentile"}:
                summary_row[f"param_{key}"] = value
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("link_f1_mean", ascending=False).reset_index(drop=True)
    summary_csv = out_root / "association_methods_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    saved_paths.append(summary_csv)

    aggregate_plots = [
        plot_metric_heatmap(summary_df, "link_f1_mean", dataset_label=dataset_label, clim=None, figure_size=(6.0 * CM, 3.9 * CM),
                            out_path=out_root / "best_value__link_f1_mean.pdf", 
                            title = "HCV methods: link_f1_mean", show_title=False),
        plot_metric_heatmap(summary_df, "id_switches_mean_per_track", dataset_label=dataset_label,
                            figure_size=(6.0 * CM, 3.9 * CM),
                            out_path=out_root / "best_value__id_switches_mean_per_track.pdf",
                            title="HCV methods: id_switches_mean_per_track", show_title=False),
        plot_metric_heatmap(summary_df, "fragmentation_rate", dataset_label=dataset_label,
                            figure_size=(6.0 * CM, 3.9 * CM),
                            out_path=out_root / "best_value__fragmentation_rate.pdf",
                            title="HCV methods: fragmentation_rate", show_title=False),
        plot_metric_heatmap(summary_df, "link_precision_mean", dataset_label=dataset_label, figure_size=(6.0 * CM, 3.9 * CM),
                            out_path=out_root / "best_value__link_precision_mean.pdf", 
                            title="HCV methods: link_precision_mean", show_title=False),
        plot_metric_heatmap(summary_df, "link_recall_mean", dataset_label=dataset_label, figure_size=(6.0 * CM, 3.9 * CM),
                            out_path=out_root / "best_value__link_recall_mean.pdf", 
                            title="HCV methods: link_recall_mean", show_title=False),
        plot_metric_heatmap(summary_df, "accuracy_mean", dataset_label=dataset_label, figure_size=(6.0 * CM, 3.9 * CM),
                            out_path=out_root / "best_value__accuracy_mean.pdf", 
                            title="HCV methods: accuracy_mean", show_title=False),
        plot_precision_recall_accuracy(summary_df, out_root / "best_value__precision_recall_accuracy.pdf", dataset_label=dataset_label,
                                       figure_size=(9.25 * CM, 4.35 * CM), show_title=False),
    ]
    saved_paths.extend(aggregate_plots)
    return summary_df, saved_paths
# %% MAIN FUNCTION
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading TIFF: {INPUT_TIF}")
    stack_tzyx, stack_info = load_stack_tzyx(INPUT_TIF)
    print(f"Loaded stack with raw shape {stack_info['raw_shape']} and axes {stack_info['raw_axes']!r}")
    print(f"Normalized stack shape: {stack_info['normalized_shape']} ({stack_info['normalized_axes']})")

    local_labels: list[np.ndarray] = []
    segmentation_rows: list[dict[str, Any]] = []
    for t_index in range(stack_tzyx.shape[0]):
        labels_t, seg_stats = segment_volume(stack_tzyx[t_index])
        local_labels.append(labels_t)
        segmentation_rows.append({"Time": float(t_index + 1), **seg_stats})
        print(
            f"Frame {t_index + 1:02d}: "
            f"{int(seg_stats['n_components'])} components, "
            f"high_thr={seg_stats['high_threshold']:.3f}, "
            f"low_thr={seg_stats['low_threshold']:.3f}")

    local_stack_tzyx = np.stack(local_labels, axis=0).astype(np.uint32, copy=False)
    tracked_labels_tzyx, gt_tracks_df, tracking_stats_df = assign_tracks(local_stack_tzyx)
    cleanup_summary_df = pd.DataFrame()
    if bool(REMOVE_SINGLE_FRAME_TRACKS):
        tracked_labels_tzyx, gt_tracks_df, cleanup_summary_df = remove_single_frame_tracks(
            tracked_labels_tzyx,
            gt_tracks_df,
        )
    segmentation_df = pd.DataFrame(segmentation_rows)
    summary_df = segmentation_df.merge(tracking_stats_df, on="Time", how="left")

    label_tif_path = save_tracked_label_tiff(tracked_labels_tzyx, OUTPUT_DIR / f"{INPUT_TIF.stem}_tracked_labels.tif")
    gt_imaris_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_groundtruth_imaris.csv"
    summary_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_segmentation_tracking_summary.csv"
    cleanup_summary_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_single_frame_track_cleanup_summary.csv"

    gt_tracks_df.to_csv(gt_imaris_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    if not cleanup_summary_df.empty:
        cleanup_summary_df.to_csv(cleanup_summary_path, index=False)
    qc_paths = save_qc_overlay_frames(
        raw_stack_tzyx=stack_tzyx,
        labels_tzyx=tracked_labels_tzyx,
        out_dir=OUTPUT_DIR / "qc_overlays")

    print()
    print("Common outputs:")
    print(f"  Labels TIFF: {label_tif_path}")
    print(f"  GT IMARIS CSV: {gt_imaris_path}")
    print(f"  Summary CSV: {summary_path}")
    if not cleanup_summary_df.empty:
        removed_tracks = int(cleanup_summary_df.iloc[0]["removed_single_frame_tracks"])
        remaining_tracks = int(cleanup_summary_df.iloc[0]["remaining_tracks"])
        print(f"  Single-frame cleanup CSV: {cleanup_summary_path}")
        print(f"  Single-frame cleanup: removed {removed_tracks} tracks, kept {remaining_tracks}")
    for qc_path in qc_paths:
        print(f"  QC overlay: {qc_path}")

    for regime_cfg in DRIFT_REGIMES:
        # regime_cfg = DRIFT_REGIMES[1]
        regime_name = str(regime_cfg["name"])
        dataset_label = str(regime_cfg["dataset_label"])
        regime_dir = OUTPUT_DIR / regime_name
        association_root = regime_dir / "blind_association_hcv"
        drift_root = regime_dir / "drift_correction_baselines"
        drift_gt_input_root = regime_dir / "drift_correction_baselines_gt_input"
        regime_dir.mkdir(parents=True, exist_ok=True)

        regime_tracks_df = build_regime_tracks(gt_tracks_df, regime_name=regime_name)
        regime_stem = f"{INPUT_TIF.stem}_{regime_name}"
        regime_imaris_path = regime_dir / f"{regime_stem}_imaris.csv"
        regime_no_trackid_path = regime_dir / f"{regime_stem}_imaris_no_trackid.csv"
        regime_tracks_plot = regime_dir / f"{regime_stem}_tracks_3d.pdf"
        blind_path = regime_dir / f"{regime_stem}_blind.csv"
        unblinding_path = regime_dir / f"{regime_stem}_unblinding.csv"
        blind_plot_path = regime_dir / f"{regime_stem}_blind_pointcloud_3d.pdf"

        regime_tracks_df.to_csv(regime_imaris_path, index=False)
        regime_tracks_df.drop(columns=["TrackID"]).to_csv(regime_no_trackid_path, index=False)
        plot_tracks_3d(regime_tracks_df, regime_tracks_plot, title=str(regime_cfg["plot_title"]))

        blind_df, unblinding_df = make_blind_pointcloud_table(
            regime_tracks_df,
            regime_name=dataset_label,
            file_name=regime_stem)
        blind_df.to_csv(blind_path, index=False)
        unblinding_df.to_csv(unblinding_path, index=False)
        plot_blind_pointcloud_3d(blind_df, blind_plot_path, title="Blind point clouds")

        association_summary_df, association_paths = run_blind_point_cloud_association_suite(
            blind_csv_path=blind_path,
            gt_tracks_df=regime_tracks_df,
            unblinding_df=unblinding_df,
            out_root=association_root,
            dataset_label=dataset_label,
            plot_reference_tracks_df=regime_tracks_df)
        
        drift_summary_df, drift_paths = run_drift_correction_suite(
            gt_tracks_df=gt_tracks_df,
            observed_df=regime_tracks_df,
            association_root=association_root,
            out_root=drift_root,
            dataset_label=dataset_label,
            regime_name=regime_name)
        
        drift_gt_input_summary_df, drift_gt_input_paths = run_drift_correction_suite_gt_input(
            gt_tracks_df=gt_tracks_df,
            observed_df=regime_tracks_df,
            out_root=drift_gt_input_root,
            dataset_label=dataset_label,
            regime_name=regime_name)
        
        regime_analysis_dir = regime_dir / "gt_vs_pca_analysis"
        regime_analysis_paths = run_regime_summary_analysis(
            gt_tracks_df=gt_tracks_df,
            association_root=association_root,
            drift_root=drift_root,
            drift_summary_df=drift_summary_df,
            regime_name=regime_name,
            out_dir=regime_analysis_dir)

        print()
        print(f"Regime: {regime_name}")
        print(f"  Output root: {regime_dir}")
        print(f"  IMARIS CSV: {regime_imaris_path}")
        print(f"  IMARIS without TrackID: {regime_no_trackid_path}")
        print(f"  Track plot: {regime_tracks_plot}")
        print(f"  Blind CSV: {blind_path}")
        print(f"  Unblinding CSV: {unblinding_path}")
        print(f"  Blind point-cloud plot: {blind_plot_path}")
        print(f"  Association output root: {association_root}")
        print(f"  GT-vs-PCA analysis root: {regime_analysis_dir}")
        print(f"  Drift-correction output root: {drift_root}")
        print(f"  Drift-correction GT-input root: {drift_gt_input_root}")
        if not association_summary_df.empty:
            print("  HCV association ranking by link_f1_mean:")
            for row in association_summary_df[["method", "link_f1_mean", "accuracy_mean"]].itertuples(index=False):
                print(
                    f"    {str(row.method)} | "
                    f"link_f1_mean={float(row.link_f1_mean):.4f} | "
                    f"accuracy_mean={float(row.accuracy_mean):.4f}"
                )
        for assoc_path in association_paths[:6]:
            print(f"  Assoc artifact: {assoc_path}")
        for key in ["summary_csv", "three_bar_plot", "birthdeath_plot", "histogram_plot"]:
            print(f"  Regime analysis artifact: {regime_analysis_paths[key]}")
        if not drift_summary_df.empty:
            print("  Drift-correction ranking by overall_rms_error:")
            for row in drift_summary_df[["method", "overall_rms_error", "drift_bias_abs"]].itertuples(index=False):
                print(
                    f"    {str(row.method)} | "
                    f"overall_rms_error={float(row.overall_rms_error):.4f} | "
                    f"drift_bias_abs={float(row.drift_bias_abs):.4f}"
                )
        for drift_path in drift_paths[:6]:
            print(f"  Drift artifact: {drift_path}")
        if not drift_gt_input_summary_df.empty:
            print("  Drift-correction GT-input ranking by overall_rms_error:")
            for row in drift_gt_input_summary_df[["method", "overall_rms_error", "drift_bias_abs"]].itertuples(index=False):
                print(
                    f"    {str(row.method)} | "
                    f"overall_rms_error={float(row.overall_rms_error):.4f} | "
                    f"drift_bias_abs={float(row.drift_bias_abs):.4f}"
                )
        for drift_gt_input_path in drift_gt_input_paths[:6]:
            print(f"  Drift GT-input artifact: {drift_gt_input_path}")
# %% MAIN
if __name__ == "__main__":
    main()
# %% END
