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
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires `scipy`. Install it in your environment, then rerun."
    ) from exc
# %% CONSTANTS AND PARAMETERS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_TIF = (
    PROJECT_ROOT
    / "example_image_data"
    / "Test datasets for segmentation"
    / "test_datasets_segmentation"
    / "Carcinoa_3DT"
    / "carcinoma_xyzt.tif"
)
OUTPUT_DIR = INPUT_TIF.parent / "carcinoma_xyzt_tracking_exports"

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
ROTATION_TOTAL_DEG_XYZ = (3.0, -2.5, 2.0)
IMARIS_UNIT = "px"
VOXEL_SIZE_XYZ = (1.0, 1.0, 1.0)

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


def plot_tracks_3d(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Plot one 3D trajectory line per TrackID."""
    fig = plt.figure(figsize=(10, 8))
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
            linewidth=1.5,
            alpha=0.9,
        )
        first = track.iloc[0]
        ax.scatter(
            [float(first["Position X"])],
            [float(first["Position Y"])],
            [float(first["Position Z"])],
            color=[color],
            s=20,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_box_aspect(ax, xyz_all)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_blind_pointcloud_3d(df_blind: pd.DataFrame, out_path: Path, title: str) -> None:
    """Plot blind point clouds, color-coded by frame."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df_blind["x"].to_numpy(float),
        df_blind["y"].to_numpy(float),
        df_blind["z"].to_numpy(float),
        c=df_blind["Time"].to_numpy(float),
        cmap="viridis",
        s=20,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_box_aspect(ax, df_blind[["x", "y", "z"]].to_numpy(np.float64))
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08)
    cbar.set_label("Frame / Time")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
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
            f"low_thr={seg_stats['low_threshold']:.3f}"
        )

    local_stack_tzyx = np.stack(local_labels, axis=0).astype(np.uint32, copy=False)
    tracked_labels_tzyx, gt_tracks_df, tracking_stats_df = assign_tracks(local_stack_tzyx)
    segmentation_df = pd.DataFrame(segmentation_rows)
    summary_df = segmentation_df.merge(tracking_stats_df, on="Time", how="left")

    label_tif_path = save_tracked_label_tiff(
        tracked_labels_tzyx,
        OUTPUT_DIR / f"{INPUT_TIF.stem}_tracked_labels.tif",
    )
    gt_imaris_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_groundtruth_imaris.csv"
    summary_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_segmentation_tracking_summary.csv"
    gt_tracks_plot = OUTPUT_DIR / f"{INPUT_TIF.stem}_groundtruth_tracks_3d.png"

    gt_tracks_df.to_csv(gt_imaris_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_tracks_3d(gt_tracks_df, gt_tracks_plot, title="Ground-truth centroids from tracked 3D segmentation")
    qc_paths = save_qc_overlay_frames(
        raw_stack_tzyx=stack_tzyx,
        labels_tzyx=tracked_labels_tzyx,
        out_dir=OUTPUT_DIR / "qc_overlays",
    )

    rotated_df = apply_rotational_drift(gt_tracks_df, max_deg_xyz=ROTATION_TOTAL_DEG_XYZ)
    rotated_imaris_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_rotated_imaris.csv"
    rotated_plot_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_rotated_tracks_3d.png"
    rotated_df.to_csv(rotated_imaris_path, index=False)
    plot_tracks_3d(rotated_df, rotated_plot_path, title="Centroid tracks after simulated rotational drift")

    file_stem = f"{INPUT_TIF.stem}_rotated"
    regime_name = "carcinoma_3dt_rotated"
    rotated_no_trackid_path = OUTPUT_DIR / f"{INPUT_TIF.stem}_rotated_imaris_no_trackid.csv"
    rotated_df.drop(columns=["TrackID"]).to_csv(rotated_no_trackid_path, index=False)
    blind_df, unblinding_df = make_blind_pointcloud_table(rotated_df, regime_name=regime_name, file_name=file_stem)
    blind_path = OUTPUT_DIR / f"{file_stem}_blind.csv"
    unblinding_path = OUTPUT_DIR / f"{file_stem}_unblinding.csv"
    blind_plot_path = OUTPUT_DIR / f"{file_stem}_blind_pointcloud_3d.png"
    blind_df.to_csv(blind_path, index=False)
    unblinding_df.to_csv(unblinding_path, index=False)
    plot_blind_pointcloud_3d(blind_df, blind_plot_path, title="Rotated blind point clouds colored by frame")

    print()
    print("Written outputs:")
    print(f"  Labels TIFF: {label_tif_path}")
    print(f"  GT IMARIS CSV: {gt_imaris_path}")
    print(f"  Rotated IMARIS CSV: {rotated_imaris_path}")
    print(f"  Rotated IMARIS without TrackID: {rotated_no_trackid_path}")
    print(f"  Blind CSV: {blind_path}")
    print(f"  Unblinding CSV: {unblinding_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  GT plot: {gt_tracks_plot}")
    print(f"  Rotated plot: {rotated_plot_path}")
    print(f"  Blind point-cloud plot: {blind_plot_path}")
    for qc_path in qc_paths:
        print(f"  QC overlay: {qc_path}")
# %% MAIN
if __name__ == "__main__":
    main()
# %% END
