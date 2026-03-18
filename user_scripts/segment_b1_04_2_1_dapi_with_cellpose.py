"""Segment the B1_04_2_1 DAPI 2D+T stack with Cellpose and link cells over time.

This script:
1. reads `example_image_data/B1_04_2_1_DAPI/B1_04_2_1_DAPI.ome.tif`,
2. normalizes the image stack to `TYX`,
3. optionally histogram-matches every frame to frame 0 to reduce bleaching,
4. segments each frame with Cellpose (`model_type="nuclei"`),
5. links detections with two post-hoc methods:
   - a Hungarian linker with centroid distance, overlap, area consistency and gap closing,
   - a connected-component baseline on the frame-to-frame overlap graph,
6. writes the raw Cellpose masks and both linked label stacks as TIFF,
7. exports track tables with `Position Z = 0` for downstream tools.

Example:
`conda run -n cellpose env PYTHONPATH="$PWD" python user_scripts/segment_b1_04_2_1_dapi_with_cellpose.py`

Installation:
conda create -n cellpose python=3.10 mamba
conda activate cellpose
pip install --upgrade pip
pip install cellpose tifffile numpy pandas scipy matplotlib
mamba install -y ipykernel
"""
# %% IMPORTS
from __future__ import annotations

import math
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

MPL_DIR = Path.cwd() / ".mplconfig_user_scripts"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR / ".cache"))

try:
    from cellpose import models
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires `cellpose`. Create a dedicated conda env and install "
        "`cellpose`, `tifffile`, `numpy`, `pandas`, and `scipy`, then rerun."
    ) from exc

# %% PARAMETERS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_TIF = (
    PROJECT_ROOT
    / "example_image_data"
    / "B1_04_2_1_DAPI"
    / "B1_04_2_1_DAPI.ome.tif")
OUTPUT_DIR = INPUT_TIF.parent / "B1_04_2_1_DAPI_cellpose_exports"
EXPECTED_AXES = "TYX"

CELLPOSE_MODEL_TYPE = "nuclei"
CELLPOSE_PRETRAINED_MODEL = "cpsam"
CELLPOSE_DIAMETER = 14.0
USE_GPU = False
MATCH_HISTOGRAMS = True
CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
MIN_OBJECT_AREA = 40
MAX_OBJECT_AREA = 4000

HUNGARIAN_MAX_DISTANCE = 32.0
HUNGARIAN_MAX_FRAME_GAP = 1
HUNGARIAN_COST_THRESHOLD = 0.98
DISTANCE_WEIGHT = 0.55
IOU_WEIGHT = 0.30
AREA_WEIGHT = 0.15
# %% FUNCTIONS
@dataclass
class Object2D:
    time: int
    local_label: int
    centroid_y: float
    centroid_x: float
    area: int
    bbox: tuple[int, int, int, int]
    mask_crop: np.ndarray


def load_stack_tyx(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the TIFF stack and normalize it to TYX."""
    with tifffile.TiffFile(path) as tif:
        array = tif.asarray()
        axes = getattr(tif.series[0], "axes", "")
        ome_xml = tif.ome_metadata or ""

    arr = np.asarray(array)
    info: dict[str, Any] = {
        "raw_shape": tuple(int(v) for v in arr.shape),
        "raw_axes": str(axes),
        "dtype": str(arr.dtype),
        "ome_present": bool(ome_xml),
    }

    if arr.ndim == 2:
        arr = arr[None, ...]
        info["axis_inference"] = "single 2D plane expanded to one timepoint"
    elif arr.ndim == 3:
        if axes in {"TYX", "QYX", "IYX"}:
            pass
        elif axes == "ZYX" or not axes:
            info["axis_inference"] = "interpreted leading Z axis as T"
        else:
            axis_to_index = {ax: i for i, ax in enumerate(axes)}
            if all(ax in axis_to_index for ax in ["T", "Y", "X"]):
                arr = np.transpose(arr, [axis_to_index[ax] for ax in ["T", "Y", "X"]])
            elif all(ax in axis_to_index for ax in ["Z", "Y", "X"]):
                arr = np.transpose(arr, [axis_to_index[ax] for ax in ["Z", "Y", "X"]])
                info["axis_inference"] = "interpreted leading Z axis as T"
            else:
                raise ValueError(f"Cannot infer TYX from shape {arr.shape} and axes {axes!r}.")
    elif arr.ndim == 4:
        axis_to_index = {ax: i for i, ax in enumerate(axes)}
        if all(ax in axis_to_index for ax in ["T", "Y", "X"]):
            arr = np.transpose(arr, [axis_to_index[ax] for ax in ["T", "Y", "X"]])
        elif all(ax in axis_to_index for ax in ["Z", "Y", "X"]):
            arr = np.transpose(arr, [axis_to_index[ax] for ax in ["Z", "Y", "X"]])
            info["axis_inference"] = "interpreted Z as T"
        else:
            raise ValueError(f"Cannot infer TYX from shape {arr.shape} and axes {axes!r}.")
    else:
        raise ValueError(f"Unsupported TIFF dimensionality: {arr.ndim}")

    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Normalized TIFF shape is {arr.shape}; expected TYX.")

    info["normalized_shape"] = tuple(int(v) for v in arr.shape)
    info["normalized_axes"] = EXPECTED_AXES
    return arr, info


def match_histograms_to_reference(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match source intensities to a reference frame using a binned CDF mapping."""
    src = np.asarray(source, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if src.size == 0 or ref.size == 0:
        return src.copy()
    lo = float(min(np.min(src), np.min(ref)))
    hi = float(max(np.max(src), np.max(ref)))
    if hi <= lo:
        return src.copy()
    bin_edges = np.linspace(lo, hi, 2049, dtype=np.float32)
    src_hist, _ = np.histogram(src.ravel(), bins=bin_edges)
    ref_hist, _ = np.histogram(ref.ravel(), bins=bin_edges)
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= max(src_cdf[-1], 1.0)
    ref_cdf /= max(ref_cdf[-1], 1.0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    matched_centers = np.interp(src_cdf, ref_cdf, bin_centers)
    src_bins = np.clip(np.digitize(src.ravel(), bin_edges) - 1, 0, len(matched_centers) - 1)
    matched = matched_centers[src_bins].reshape(src.shape)
    return matched.astype(np.float32, copy=False)


def preprocess_stack_for_cellpose(stack_tyx: np.ndarray, match_histograms: bool = True) -> np.ndarray:
    """Apply mild per-frame preprocessing for more stable Cellpose segmentation."""
    stack = np.asarray(stack_tyx, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError("Expected TYX stack.")
    reference = stack[0]
    processed = np.empty_like(stack, dtype=np.float32)
    for t in range(stack.shape[0]):
        frame = stack[t]
        if match_histograms:
            frame = match_histograms_to_reference(frame, reference)
        processed[t] = frame
    return processed


def keep_labels_in_size_range(label_frame: np.ndarray, min_area: int, max_area: int | None = None) -> np.ndarray:
    """Filter instance labels by area and relabel consecutively."""
    labels = np.asarray(label_frame, dtype=np.int32)
    if labels.max() <= 0:
        return np.zeros_like(labels, dtype=np.uint32)
    sizes = np.bincount(labels.ravel())
    out = np.zeros_like(labels, dtype=np.uint32)
    next_label = 1
    for local_label in range(1, int(labels.max()) + 1):
        area = int(sizes[local_label]) if local_label < sizes.size else 0
        if area < int(min_area):
            continue
        if max_area is not None and area > int(max_area):
            continue
        out[labels == local_label] = np.uint32(next_label)
        next_label += 1
    return out


def run_cellpose_segmentation(
    stack_tyx: np.ndarray,
    diameter: float,
    use_gpu: bool,
    model_type: str,
    flow_threshold: float,
    cellprob_threshold: float,
    min_area: int,
    max_area: int | None,
) -> np.ndarray:
    """Segment each frame with Cellpose and return a TYX instance-label stack."""
    model_cls = getattr(models, "CellposeModel", None)
    if model_cls is not None:
        # Cellpose v4 uses `CellposeModel` and ignores `model_type`.
        model = model_cls(gpu=bool(use_gpu), pretrained_model=str(CELLPOSE_PRETRAINED_MODEL))
    else:
        legacy_cls = getattr(models, "Cellpose", None)
        if legacy_cls is None:
            raise AttributeError(
                "Neither `cellpose.models.CellposeModel` nor `cellpose.models.Cellpose` "
                "is available in the installed Cellpose package."
            )
        model = legacy_cls(gpu=bool(use_gpu), model_type=str(model_type))
    label_stack = np.zeros_like(stack_tyx, dtype=np.uint32)
    for t in range(stack_tyx.shape[0]):
        frame = np.asarray(stack_tyx[t], dtype=np.float32)
        masks, _, _, _ = model.eval(
            frame,
            channels=[0, 0],
            diameter=float(diameter),
            do_3D=False,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
            min_size=int(min_area),
        )
        label_stack[t] = keep_labels_in_size_range(masks, min_area=min_area, max_area=max_area)
        print(
            f"[Cellpose] frame {t + 1:03d}/{stack_tyx.shape[0]:03d}: "
            f"{int(label_stack[t].max())} kept objects",
            flush=True,
        )
    return label_stack


def extract_objects_from_frame(label_frame: np.ndarray, time_index: int) -> list[Object2D]:
    """Extract object properties from one 2D label frame."""
    labels = np.asarray(label_frame, dtype=np.int32)
    if labels.max() <= 0:
        return []
    objects: list[Object2D] = []
    slices = ndi.find_objects(labels)
    for local_label, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        y_slice, x_slice = slc
        crop = labels[y_slice, x_slice] == local_label
        area = int(crop.sum())
        if area <= 0:
            continue
        centroid = ndi.center_of_mass(crop.astype(np.uint8))
        centroid_y = float(y_slice.start) + float(centroid[0])
        centroid_x = float(x_slice.start) + float(centroid[1])
        bbox = (
            int(y_slice.start),
            int(y_slice.stop),
            int(x_slice.start),
            int(x_slice.stop),
        )
        objects.append(
            Object2D(
                time=int(time_index),
                local_label=int(local_label),
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                area=area,
                bbox=bbox,
                mask_crop=np.asarray(crop, dtype=bool),
            )
        )
    return objects


def compute_iou(obj_a: Object2D, obj_b: Object2D) -> float:
    """Compute IoU between two 2D objects from possibly different frames."""
    ay0, ay1, ax0, ax1 = obj_a.bbox
    by0, by1, bx0, bx1 = obj_b.bbox
    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    a_crop = obj_a.mask_crop[(iy0 - ay0):(iy1 - ay0), (ix0 - ax0):(ix1 - ax0)]
    b_crop = obj_b.mask_crop[(iy0 - by0):(iy1 - by0), (ix0 - bx0):(ix1 - bx0)]
    intersection = int(np.logical_and(a_crop, b_crop).sum())
    if intersection == 0:
        return 0.0
    union = int(obj_a.area + obj_b.area - intersection)
    return float(intersection / max(union, 1))


def build_object_lists(label_stack_tyx: np.ndarray) -> list[list[Object2D]]:
    """Extract object metadata for all frames."""
    return [extract_objects_from_frame(label_stack_tyx[t], time_index=t) for t in range(label_stack_tyx.shape[0])]


def link_with_hungarian(
    label_stack_tyx: np.ndarray,
    max_distance: float,
    max_frame_gap: int,
    cost_threshold: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Link objects over time using Hungarian assignment with simple motion cues."""
    objects_by_time = build_object_lists(label_stack_tyx)
    linked = np.zeros_like(label_stack_tyx, dtype=np.uint32)
    track_records: list[dict[str, Any]] = []
    next_track_id = 1
    active_tracks: dict[int, dict[str, Any]] = {}

    for t, objects in enumerate(objects_by_time):
        if not active_tracks:
            assignments: dict[int, int] = {}
        else:
            active_items = [
                (track_id, state)
                for track_id, state in active_tracks.items()
                if int(t - state["last_time"]) <= int(max_frame_gap) + 1
            ]
            assignments = {}
            if active_items and objects:
                n_tracks = len(active_items)
                n_objects = len(objects)
                cost = np.full((n_tracks, n_objects), fill_value=np.inf, dtype=np.float64)
                for i, (_, state) in enumerate(active_items):
                    prev_y = float(state["last_y"])
                    prev_x = float(state["last_x"])
                    vy = float(state["velocity_y"])
                    vx = float(state["velocity_x"])
                    time_gap = max(int(t - state["last_time"]), 1)
                    pred_y = prev_y + vy * float(time_gap)
                    pred_x = prev_x + vx * float(time_gap)
                    prev_area = max(float(state["last_area"]), 1.0)
                    prev_obj = state["last_obj"]
                    for j, obj in enumerate(objects):
                        dy = float(obj.centroid_y - pred_y)
                        dx = float(obj.centroid_x - pred_x)
                        dist = math.hypot(dy, dx)
                        if dist > float(max_distance) * float(time_gap):
                            continue
                        iou = compute_iou(prev_obj, obj) if time_gap == 1 else 0.0
                        area_rel = abs(float(obj.area) - prev_area) / prev_area
                        cost[i, j] = (
                            DISTANCE_WEIGHT * min(dist / (float(max_distance) * float(time_gap)), 1.0)
                            + IOU_WEIGHT * (1.0 - iou)
                            + AREA_WEIGHT * min(area_rel, 1.0)
                        )
                row_ind, col_ind = linear_sum_assignment(cost)
                for row, col in zip(row_ind.tolist(), col_ind.tolist()):
                    if not np.isfinite(cost[row, col]):
                        continue
                    if float(cost[row, col]) > float(cost_threshold):
                        continue
                    track_id = int(active_items[row][0])
                    assignments[col] = track_id

        used_track_ids = set(assignments.values())
        for obj_index, obj in enumerate(objects):
            track_id = assignments.get(obj_index)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
            track_records.append(
                {
                    "TrackID": int(track_id),
                    "Time": int(t),
                    "Position X": float(obj.centroid_x),
                    "Position Y": float(obj.centroid_y),
                    "Position Z": 0.0,
                    "Area": int(obj.area),
                    "Source Label": int(obj.local_label),
                    "Linker": "hungarian",
                }
            )
            linked[t][label_stack_tyx[t] == obj.local_label] = np.uint32(track_id)
            prev_state = active_tracks.get(track_id)
            if prev_state is None:
                velocity_y = 0.0
                velocity_x = 0.0
            else:
                dt = max(int(t - prev_state["last_time"]), 1)
                velocity_y = (float(obj.centroid_y) - float(prev_state["last_y"])) / float(dt)
                velocity_x = (float(obj.centroid_x) - float(prev_state["last_x"])) / float(dt)
            active_tracks[track_id] = {
                "last_time": int(t),
                "last_y": float(obj.centroid_y),
                "last_x": float(obj.centroid_x),
                "last_area": int(obj.area),
                "velocity_y": float(velocity_y),
                "velocity_x": float(velocity_x),
                "last_obj": obj,
            }
            used_track_ids.add(track_id)

        stale_track_ids = [
            track_id
            for track_id, state in active_tracks.items()
            if int(t - state["last_time"]) > int(max_frame_gap) + 1
        ]
        for track_id in stale_track_ids:
            active_tracks.pop(track_id, None)

    tracks_df = pd.DataFrame(track_records).sort_values(["TrackID", "Time"]).reset_index(drop=True)
    return linked, tracks_df


class UnionFind:
    """Minimal disjoint-set data structure."""

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def add(self, item: tuple[int, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        root = self.parent.setdefault(item, item)
        if root != item:
            root = self.find(root)
            self.parent[item] = root
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def link_with_overlap_connected_components(label_stack_tyx: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Link objects by connected components on the frame-to-frame overlap graph."""
    objects_by_time = build_object_lists(label_stack_tyx)
    uf = UnionFind()
    for t, objects in enumerate(objects_by_time):
        for obj in objects:
            uf.add((int(t), int(obj.local_label)))
    for t in range(len(objects_by_time) - 1):
        current = objects_by_time[t]
        nxt = objects_by_time[t + 1]
        if not current or not nxt:
            continue
        for obj_a in current:
            for obj_b in nxt:
                if compute_iou(obj_a, obj_b) > 0.0:
                    uf.union((obj_a.time, obj_a.local_label), (obj_b.time, obj_b.local_label))

    root_to_track_id: dict[tuple[int, int], int] = {}
    next_track_id = 1
    linked = np.zeros_like(label_stack_tyx, dtype=np.uint32)
    records: list[dict[str, Any]] = []
    for t, objects in enumerate(objects_by_time):
        for obj in objects:
            node = (int(t), int(obj.local_label))
            root = uf.find(node)
            if root not in root_to_track_id:
                root_to_track_id[root] = next_track_id
                next_track_id += 1
            track_id = int(root_to_track_id[root])
            linked[t][label_stack_tyx[t] == obj.local_label] = np.uint32(track_id)
            records.append(
                {
                    "TrackID": int(track_id),
                    "Time": int(t),
                    "Position X": float(obj.centroid_x),
                    "Position Y": float(obj.centroid_y),
                    "Position Z": 0.0,
                    "Area": int(obj.area),
                    "Source Label": int(obj.local_label),
                    "Linker": "cc_overlap",
                }
            )
    tracks_df = pd.DataFrame(records).sort_values(["TrackID", "Time"]).reset_index(drop=True)
    return linked, tracks_df


def save_label_tiff(path: Path, label_stack_tyx: np.ndarray) -> Path:
    """Write a TYX label stack as TIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.asarray(label_stack_tyx, dtype=np.uint32),
        metadata={"axes": EXPECTED_AXES},
        imagej=False,
    )
    return path


def save_track_csv(path: Path, tracks_df: pd.DataFrame) -> Path:
    """Write track table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks_df.to_csv(path, index=False)
    return path
# %% MAIN FUNCTION
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stack_tyx, info = load_stack_tyx(INPUT_TIF)
    print(f"[I/O] loaded {INPUT_TIF.name}: {info}")

    processed_tyx = preprocess_stack_for_cellpose(stack_tyx, match_histograms=MATCH_HISTOGRAMS)
    masks_tyx = run_cellpose_segmentation(
        stack_tyx=processed_tyx,
        diameter=CELLPOSE_DIAMETER,
        use_gpu=USE_GPU,
        model_type=CELLPOSE_MODEL_TYPE,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_area=MIN_OBJECT_AREA,
        max_area=MAX_OBJECT_AREA)

    raw_masks_path = OUTPUT_DIR / "B1_04_2_1_DAPI_cellpose_masks.tif"
    save_label_tiff(raw_masks_path, masks_tyx)

    hungarian_labels, hungarian_tracks = link_with_hungarian(
        masks_tyx,
        max_distance=HUNGARIAN_MAX_DISTANCE,
        max_frame_gap=HUNGARIAN_MAX_FRAME_GAP,
        cost_threshold=HUNGARIAN_COST_THRESHOLD,
    )
    hungarian_tif = OUTPUT_DIR / "B1_04_2_1_DAPI_cellpose_linked_hungarian.tif"
    hungarian_csv = OUTPUT_DIR / "B1_04_2_1_DAPI_cellpose_linked_hungarian_tracks.csv"
    save_label_tiff(hungarian_tif, hungarian_labels)
    save_track_csv(hungarian_csv, hungarian_tracks)

    cc_labels, cc_tracks = link_with_overlap_connected_components(masks_tyx)
    cc_tif = OUTPUT_DIR / "B1_04_2_1_DAPI_cellpose_linked_cc_overlap.tif"
    cc_csv = OUTPUT_DIR / "B1_04_2_1_DAPI_cellpose_linked_cc_overlap_tracks.csv"
    save_label_tiff(cc_tif, cc_labels)
    save_track_csv(cc_csv, cc_tracks)

    summary = pd.DataFrame(
        [
            {
                "stack_shape_tyx": tuple(int(v) for v in stack_tyx.shape),
                "cellpose_model": CELLPOSE_MODEL_TYPE,
                "cellpose_diameter": float(CELLPOSE_DIAMETER),
                "histogram_matching": bool(MATCH_HISTOGRAMS),
                "hungarian_unique_tracks": int(hungarian_tracks["TrackID"].nunique()),
                "cc_overlap_unique_tracks": int(cc_tracks["TrackID"].nunique()),
            }
        ]
    )
    summary_path = OUTPUT_DIR / "B1_04_2_1_DAPI_cellpose_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Done] wrote outputs to {OUTPUT_DIR}")
# %% MAIN
if __name__ == "__main__":
    main()
# %% END
