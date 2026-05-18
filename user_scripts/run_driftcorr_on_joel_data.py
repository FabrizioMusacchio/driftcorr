"""Run all drift-correction candidates on IMARIS spot tables in Joel_data.

This script mirrors the standard Python workflow used elsewhere in the project,
but targets the real IMARIS-exported 3D spot tables placed in
`example_image_data/Joel_data/`.

Current goal:
- apply all drift-correction candidate methods,
- save corrected position tables per method,
- write standard method-comparison metrics and qualitative plots.

Later, additional morphology/movement summaries can be added on top of these
outputs without touching the core correction runs.

Author
------
author: Fabrizio Musacchio
date: May 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import proj3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
    from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
    from driftcorr.method_local_communities import (
        run_local_drift_correction_for_file,
        run_local_drift_correction_for_file_with_optional_smoothing,
    )
    from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
    from driftcorr.method_rigid_step import (
        apply_rigid_correction_to_tracks,
        apply_rigid_correction_with_optional_smoothing,
        build_step_transforms,
    )
    from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
    from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
    from driftcorr.stationary import find_stationary_tracks
    from driftcorr.visualization import save_visual_inspection_plots
except ModuleNotFoundError as exc:
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
        from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
        from driftcorr.method_local_communities import (
            run_local_drift_correction_for_file,
            run_local_drift_correction_for_file_with_optional_smoothing,
        )
        from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
        from driftcorr.method_rigid_step import (
            apply_rigid_correction_to_tracks,
            apply_rigid_correction_with_optional_smoothing,
            build_step_transforms,
        )
        from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
        from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
        from driftcorr.stationary import find_stationary_tracks
        from driftcorr.visualization import save_visual_inspection_plots
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import 'driftcorr'. Install the package or run this script from inside the cloned project."
        ) from exc

# %% GLOBAL SETTINGS
JOEL_DIR = PROJECT_ROOT / "example_image_data" / "Joel_data"
RESULTS_ROOT = JOEL_DIR / "driftcorr_results"

DEFAULT_METHODS = ["rigid", "local", "rbf", "st_rbf"]
WRITE_INPUT_SUMMARY = False
SAVE_ONLY_MINIMAL_OUTPUTS = True
ANNOTATE_TRACK_IDS_IN_COMPARISON_PLOTS = True
TRACK_ID_ANNOTATION_FONTSIZE = 4
COLOR_TRACK_ID_TEXT_BY_TRACK = True

METHOD_COLORS = {
    "original": "#7A7A7A",
    "rigid": "#4C78A8",
    "local": "#59A14F",
    "rbf": "#F28E2B",
    "st_rbf": "#B07AA1",
    "rigid_smooth": "#76B7B2",
    "local_smooth": "#E15759",
}

OPTIONS = {
    "reference_time": None,
    "seed": 0,
    "lam": 3e-4,
    "keep_fraction": 0.7,
    "min_pairs": 15,
    "m_centers": 45,
    "sigma": 180.0,
    "temporal_gamma": 1.0,
    "center_mode": "ref",
    "center_pooled_time_stride": 1,
    "robust_mode": "sample_trim",
    "track_trim_iters": 4,
    "two_stage": 0,
    "rbf_linear_terms": 0,
    "rbf_space_input_mode": "track_anchor",
    "row_position_iters": 3,
    "row_position_tol": 1e-3,
    "weight_abort_threshold": 1e4,
    "rigid_mode": "rigid",
    "inlier_reestimate_iters": 30,
    "k_neighbors": 12,
    "radius_um": None,
    "ell_um_graph": 80.0,
    "ell_um_field": 120.0,
    "min_edge_weight": 0.25,
    "max_pairs_per_node": 50,
    "st_m_centers": 70,
    "st_lam": 8.5e-4,
    "st_ell_space": 4.0,
    "st_ell_time": 0.4,
    "st_ell_x": None,
    "st_ell_y": None,
    "st_ell_z": None,
    "st_robust_keep_fraction": 0.83,
    "st_robust_iters": 2,
    "space_input_mode": "ref_position",
    "st_refine_iters": 3,
    "st_refine_tol": 0.0,
    "st_robust_mode": "hybrid_track_irls",
    "st_track_trim_iters": 4,
    "st_per_axis_scaling": 1,
    "st_linear_interactions": 0,
    "rigid_smooth_enable": 0,
    "rigid_smooth_method": "savgol",
    "rigid_smooth_window": 9,
    "rigid_smooth_polyorder": 2,
    "local_smooth_enable": 0,
    "local_smooth_method": "savgol",
    "local_smooth_window": 9,
    "local_smooth_polyorder": 2,
    "smooth_detect_stage_jumps": 1,
    "smooth_jump_thresh_mad": 6.0,
}
# %% HELPERS
def _read_joel_imaris_file(path: Path) -> pd.DataFrame:
    """Read one Joel IMARIS export with a helpful error if xlrd is missing."""
    try:
        return ensure_numeric_tracks(read_imaris_position_xls(path))
    except ImportError as exc:
        raise ImportError(
            f"Reading legacy .xls files requires 'xlrd'. Install it, then re-run this script.\n"
            f"Failed while loading: {path}"
        ) from exc


def _write_input_summary(df: pd.DataFrame, out_prefix: Path) -> None:
    """Persist a compact summary of the original track table."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    times = sorted(pd.unique(df["Time"]))
    track_ids = sorted(pd.unique(df["TrackID"]))
    summary = pd.DataFrame(
        [
            {
                "n_rows": int(df.shape[0]),
                "n_tracks": int(len(track_ids)),
                "n_timepoints": int(len(times)),
                "time_min": float(min(times)) if times else None,
                "time_max": float(max(times)) if times else None,
                "x_min": float(df["Position X"].min()),
                "x_max": float(df["Position X"].max()),
                "y_min": float(df["Position Y"].min()),
                "y_max": float(df["Position Y"].max()),
                "z_min": float(df["Position Z"].min()),
                "z_max": float(df["Position Z"].max()),
            }
        ]
    )
    summary.to_csv(str(out_prefix) + "_input_summary.csv", index=False)
    df.to_csv(str(out_prefix) + "_input_numeric.csv", index=False)


def _compute_track_limits(df_orig: pd.DataFrame, df_corr: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
    """Compute shared plot limits across original and corrected coordinates."""
    x_min = min(float(df_orig["Position X"].min()), float(df_corr["Position X_corr"].min()))
    x_max = max(float(df_orig["Position X"].max()), float(df_corr["Position X_corr"].max()))
    y_min = min(float(df_orig["Position Y"].min()), float(df_corr["Position Y_corr"].min()))
    y_max = max(float(df_orig["Position Y"].max()), float(df_corr["Position Y_corr"].max()))
    z_min = min(float(df_orig["Position Z"].min()), float(df_corr["Position Z_corr"].min()))
    z_max = max(float(df_orig["Position Z"].max()), float(df_corr["Position Z_corr"].max()))
    return x_min, x_max, y_min, y_max, z_min, z_max


def _estimate_label_bbox(display_xy: np.ndarray, label: str, fig_dpi: float, font_size: float) -> tuple[float, float, float, float]:
    """Approximate a text bounding box in display coordinates."""
    char_w = 0.58 * float(font_size) * float(fig_dpi) / 72.0
    text_w = max(10.0, char_w * max(len(str(label)), 1))
    text_h = max(8.0, 1.25 * float(font_size) * float(fig_dpi) / 72.0)
    x0 = float(display_xy[0])
    y0 = float(display_xy[1] - 0.5 * text_h)
    return (x0, y0, x0 + text_w, y0 + text_h)


def _bbox_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float = 2.0) -> bool:
    """Return whether two display-space bounding boxes overlap."""
    return not (
        a[2] + pad < b[0]
        or a[0] - pad > b[2]
        or a[3] + pad < b[1]
        or a[1] - pad > b[3]
    )


def _bbox_inside_axis(bbox: tuple[float, float, float, float], axis_bbox) -> bool:
    """Return whether a label bbox stays inside an axis bbox."""
    return (
        bbox[0] >= axis_bbox.x0
        and bbox[2] <= axis_bbox.x1
        and bbox[1] >= axis_bbox.y0
        and bbox[3] <= axis_bbox.y1
    )


def _choose_label_position_2d(
    ax,
    renderer,
    x: float,
    y: float,
    label: str,
    dx: float,
    dy: float,
    used_bboxes: list[tuple[float, float, float, float]],
    font_size: float,
) -> tuple[float, float]:
    """Choose a non-overlapping 2D label position from a set of candidate offsets."""
    candidates = [
        (x + dx, y + dy),
        (x + dx, y - dy),
        (x - dx, y + dy),
        (x - dx, y - dy),
        (x + 1.8 * dx, y),
        (x - 1.8 * dx, y),
        (x, y + 1.8 * dy),
        (x, y - 1.8 * dy),
        (x + 1.8 * dx, y + 1.8 * dy),
        (x - 1.8 * dx, y + 1.8 * dy),
        (x + 1.8 * dx, y - 1.8 * dy),
        (x - 1.8 * dx, y - 1.8 * dy),
    ]
    axis_bbox = ax.get_window_extent(renderer=renderer)
    dpi = float(ax.figure.dpi)
    fallback = candidates[0]
    for cx, cy in candidates:
        disp = ax.transData.transform((cx, cy))
        bbox = _estimate_label_bbox(disp, label, dpi, font_size)
        if not _bbox_inside_axis(bbox, axis_bbox):
            continue
        if any(_bbox_overlaps(bbox, ub) for ub in used_bboxes):
            continue
        used_bboxes.append(bbox)
        return float(cx), float(cy)
    disp = ax.transData.transform(fallback)
    used_bboxes.append(_estimate_label_bbox(disp, label, dpi, font_size))
    return float(fallback[0]), float(fallback[1])


def _choose_label_position_3d(
    ax,
    renderer,
    x: float,
    y: float,
    z: float,
    label: str,
    dx: float,
    dy: float,
    dz: float,
    used_bboxes: list[tuple[float, float, float, float]],
    font_size: float,
) -> tuple[float, float, float]:
    """Choose a non-overlapping 3D label position based on projected display-space overlap."""
    candidates = [
        (x + dx, y + dy, z + dz),
        (x + dx, y - dy, z + dz),
        (x - dx, y + dy, z + dz),
        (x - dx, y - dy, z + dz),
        (x + 1.8 * dx, y, z + 0.6 * dz),
        (x - 1.8 * dx, y, z + 0.6 * dz),
        (x, y + 1.8 * dy, z + 0.6 * dz),
        (x, y - 1.8 * dy, z + 0.6 * dz),
        (x + 1.8 * dx, y + 1.8 * dy, z),
        (x - 1.8 * dx, y + 1.8 * dy, z),
        (x + 1.8 * dx, y - 1.8 * dy, z),
        (x - 1.8 * dx, y - 1.8 * dy, z),
    ]
    axis_bbox = ax.get_window_extent(renderer=renderer)
    dpi = float(ax.figure.dpi)
    fallback = candidates[0]
    for cx, cy, cz in candidates:
        xp, yp, _ = proj3d.proj_transform(cx, cy, cz, ax.get_proj())
        disp = ax.transData.transform((xp, yp))
        bbox = _estimate_label_bbox(disp, label, dpi, font_size)
        if not _bbox_inside_axis(bbox, axis_bbox):
            continue
        if any(_bbox_overlaps(bbox, ub) for ub in used_bboxes):
            continue
        used_bboxes.append(bbox)
        return float(cx), float(cy), float(cz)
    xp, yp, _ = proj3d.proj_transform(*fallback, ax.get_proj())
    disp = ax.transData.transform((xp, yp))
    used_bboxes.append(_estimate_label_bbox(disp, label, dpi, font_size))
    return float(fallback[0]), float(fallback[1]), float(fallback[2])


def _annotate_track_end_2d(
    ax,
    x: float,
    y: float,
    x2: float,
    y2: float,
    label: str,
    font_size: float,
    text_color: str = "black",
) -> None:
    """Annotate one track end in a 2D axis with a thin connector line."""
    ha = "left" if x2 >= x else "right"
    ax.annotate(
        str(label),
        xy=(x, y),
        xytext=(x2, y2),
        textcoords="data",
        fontsize=float(font_size),
        color=text_color,
        ha=ha,
        va="center",
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.5, "shrinkA": 0, "shrinkB": 0},
        clip_on=True,
    )


def _annotate_track_end_3d(
    ax,
    x: float,
    y: float,
    z: float,
    x2: float,
    y2: float,
    z2: float,
    label: str,
    font_size: float,
    text_color: str = "black",
) -> None:
    """Annotate one track end in a 3D axis with a thin connector line."""
    ax.plot([x, x2], [y, y2], [z, z2], color="black", linewidth=0.5, alpha=0.9)
    ax.text(x2, y2, z2, str(label), fontsize=float(font_size), color=text_color)


def _add_track_id_annotations(
    ax3d,
    ax_xy,
    ax_xz,
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    font_size: float,
    track_colors: dict[float, str] | None = None,
) -> None:
    """Add track-end ID labels to 3D, XY, and XZ comparison panels."""
    if df_plot.empty or "TrackID" not in df_plot.columns:
        return
    fig = ax_xy.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_vals = pd.to_numeric(df_plot[x_col], errors="coerce").to_numpy(np.float64)
    y_vals = pd.to_numeric(df_plot[y_col], errors="coerce").to_numpy(np.float64)
    z_vals = pd.to_numeric(df_plot[z_col], errors="coerce").to_numpy(np.float64)
    span_x = max(float(np.nanmax(x_vals) - np.nanmin(x_vals)), 1e-6)
    span_y = max(float(np.nanmax(y_vals) - np.nanmin(y_vals)), 1e-6)
    span_z = max(float(np.nanmax(z_vals) - np.nanmin(z_vals)), 1e-6)
    dx = 0.018 * span_x
    dy = 0.018 * span_y
    dz = 0.018 * span_z
    used_3d: list[tuple[float, float, float, float]] = []
    used_xy: list[tuple[float, float, float, float]] = []
    used_xz: list[tuple[float, float, float, float]] = []

    for track_id, grp in df_plot.groupby("TrackID", sort=True):
        g = grp.sort_values("Time")
        if g.empty:
            continue
        x = float(g.iloc[-1][x_col])
        y = float(g.iloc[-1][y_col])
        z = float(g.iloc[-1][z_col])
        label = int(track_id) if float(track_id).is_integer() else f"{float(track_id):g}"
        text_color = "black"
        if COLOR_TRACK_ID_TEXT_BY_TRACK and track_colors is not None:
            text_color = str(track_colors.get(float(track_id), "black"))
        x3, y3, z3 = _choose_label_position_3d(ax3d, renderer, x, y, z, label, dx, dy, dz, used_3d, font_size)
        xxy, yxy = _choose_label_position_2d(ax_xy, renderer, x, y, label, dx, dy, used_xy, font_size)
        xxz, yxz = _choose_label_position_2d(ax_xz, renderer, x, z, label, dx, dz, used_xz, font_size)
        _annotate_track_end_3d(ax3d, x, y, z, x3, y3, z3, label, font_size, text_color=text_color)
        _annotate_track_end_2d(ax_xy, x, y, xxy, yxy, label, font_size, text_color=text_color)
        _annotate_track_end_2d(ax_xz, x, z, xxz, yxz, label, font_size, text_color=text_color)


def _plot_tracks_comparison_for_joel(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    out_path: Path,
    title: str = "",
    annotate_track_ids: bool = False,
) -> Path:
    """Create Joel-specific comparison plot with optional TrackID annotations."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 18))
    x_min, x_max, y_min, y_max, z_min, z_max = _compute_track_limits(df_orig, df_corr)
    orig_track_colors: dict[float, str] = {}
    corr_track_colors: dict[float, str] = {}

    ax1 = fig.add_subplot(321, projection="3d")
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid].sort_values("Time")
        line = ax1.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.9)[0]
        orig_track_colors[float(tid)] = str(line.get_color())
    ax1.set_title("Original (3D)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(322, projection="3d")
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid].sort_values("Time")
        line = ax2.plot(d["Position X_corr"], d["Position Y_corr"], d["Position Z_corr"], alpha=0.9)[0]
        corr_track_colors[float(tid)] = str(line.get_color())
    ax2.set_title("Corrected (3D)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    ax3 = fig.add_subplot(323)
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid].sort_values("Time")
        ax3.plot(d["Position X"], d["Position Y"], alpha=0.7, color=orig_track_colors.get(float(tid)))
    ax3.set_title("Original XY")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    ax4 = fig.add_subplot(324)
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid].sort_values("Time")
        ax4.plot(d["Position X_corr"], d["Position Y_corr"], alpha=0.7, color=corr_track_colors.get(float(tid)))
    ax4.set_title("Corrected XY")
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    ax5 = fig.add_subplot(325)
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid].sort_values("Time")
        ax5.plot(d["Position X"], d["Position Z"], alpha=0.7, color=orig_track_colors.get(float(tid)))
    ax5.set_title("Original XZ")
    ax5.set_aspect("equal", adjustable="box")
    ax5.set_xlim(x_min, x_max)
    ax5.set_ylim(z_min, z_max)
    ax5.set_xlabel("X")
    ax5.set_ylabel("Z")

    ax6 = fig.add_subplot(326)
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid].sort_values("Time")
        ax6.plot(d["Position X_corr"], d["Position Z_corr"], alpha=0.7, color=corr_track_colors.get(float(tid)))
    ax6.set_title("Corrected XZ")
    ax6.set_aspect("equal", adjustable="box")
    ax6.set_xlim(x_min, x_max)
    ax6.set_ylim(z_min, z_max)
    ax6.set_xlabel("X")
    ax6.set_ylabel("Z")

    if annotate_track_ids:
        _add_track_id_annotations(
            ax1,
            ax3,
            ax5,
            df_orig,
            "Position X",
            "Position Y",
            "Position Z",
            font_size=TRACK_ID_ANNOTATION_FONTSIZE,
            track_colors=orig_track_colors,
        )
        _add_track_id_annotations(
            ax2,
            ax4,
            ax6,
            df_corr,
            "Position X_corr",
            "Position Y_corr",
            "Position Z_corr",
            font_size=TRACK_ID_ANNOTATION_FONTSIZE,
            track_colors=corr_track_colors,
        )

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def _cleanup_minimal_outputs(out_prefix: Path) -> None:
    """Keep only corrected-coordinate CSVs, comparison plots, and timeseries directories."""
    base_dir = out_prefix.parent
    stem = out_prefix.name
    if not base_dir.exists():
        return

    keep_files = {
        f"{stem}_input_duplicate_track_time_rows.csv",
        f"{stem}_input_deduplication_summary.csv",
        f"{stem}_failure_diagnostics_summary.csv",
        f"{stem}_failure_duplicate_track_time_rows.csv",
        f"{stem}_failure_reference_time_duplicate_track_ids.csv",
        f"{stem}_failure_traceback.txt",
    }
    keep_prefixes = [
        f"{stem}_positions_corrected_",
        f"{stem}_tracks_comparison_",
    ]
    keep_dir_prefix = f"{stem}_timeseries_"

    for path in base_dir.iterdir():
        if path.is_dir():
            if path.name.startswith(keep_dir_prefix):
                continue
            continue
        if path.name in keep_files:
            continue
        if any(path.name.startswith(prefix) for prefix in keep_prefixes):
            continue
        path.unlink(missing_ok=True)


def _collect_track_table_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return compact diagnostics plus duplicate-row tables for one track dataframe."""
    d = ensure_numeric_tracks(df)
    if d.empty:
        summary = pd.DataFrame([{"n_rows": 0, "n_tracks": 0, "n_timepoints": 0}])
        return summary, d.copy(), d.copy()

    ref_time = float(pd.to_numeric(d["Time"], errors="coerce").min())
    pair_counts = (
        d.groupby(["TrackID", "Time"], as_index=False)
        .size()
        .rename(columns={"size": "rows_per_track_time"})
    )
    dup_pairs = pair_counts[pair_counts["rows_per_track_time"] > 1].copy()

    dup_rows = d.merge(
        dup_pairs[["TrackID", "Time"]],
        on=["TrackID", "Time"],
        how="inner",
    ).sort_values(["TrackID", "Time"]).reset_index(drop=True)

    ref_rows = d[d["Time"] == ref_time].copy()
    ref_track_counts = (
        ref_rows.groupby("TrackID", as_index=False)
        .size()
        .rename(columns={"size": "rows_at_reference_time"})
    )
    ref_dup_tracks = ref_track_counts[ref_track_counts["rows_at_reference_time"] > 1].copy()

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(d.shape[0]),
                "n_tracks": int(pd.to_numeric(d["TrackID"], errors="coerce").dropna().nunique()),
                "n_timepoints": int(pd.to_numeric(d["Time"], errors="coerce").dropna().nunique()),
                "reference_time": float(ref_time),
                "duplicate_track_time_pairs": int(dup_pairs.shape[0]),
                "duplicate_track_time_rows": int(dup_rows.shape[0]),
                "max_rows_per_track_time": int(pair_counts["rows_per_track_time"].max()),
                "reference_time_duplicate_track_ids": int(ref_dup_tracks.shape[0]),
                "reference_time_max_rows_per_track": int(ref_track_counts["rows_at_reference_time"].max()),
            }
        ]
    )
    return summary, dup_rows, ref_dup_tracks


def _first_nonnull_value(series: pd.Series):
    """Return the first non-null value from a series, or the first entry if all are null."""
    nonnull = series.dropna()
    if not nonnull.empty:
        return nonnull.iloc[0]
    return series.iloc[0] if len(series) else None


def _deduplicate_track_time_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse duplicate (TrackID, Time) rows deterministically and preserve schema where possible."""
    d = ensure_numeric_tracks(df)
    pair_counts = (
        d.groupby(["TrackID", "Time"], as_index=False)
        .size()
        .rename(columns={"size": "rows_per_track_time"})
    )
    dup_pairs = pair_counts[pair_counts["rows_per_track_time"] > 1].copy()
    if dup_pairs.empty:
        summary = pd.DataFrame(
            [
                {
                    "n_rows_before": int(d.shape[0]),
                    "n_rows_after": int(d.shape[0]),
                    "n_duplicate_pairs": 0,
                    "n_rows_removed": 0,
                }
            ]
        )
        return d.copy(), pd.DataFrame(columns=d.columns), summary

    dup_rows = d.merge(dup_pairs[["TrackID", "Time"]], on=["TrackID", "Time"], how="inner")
    agg_map: dict[str, str | callable] = {}
    for col in d.columns:
        if col in {"TrackID", "Time"}:
            continue
        if pd.api.types.is_numeric_dtype(d[col]):
            agg_map[col] = "median"
        else:
            agg_map[col] = _first_nonnull_value

    dedup = d.groupby(["TrackID", "Time"], as_index=False, sort=False).agg(agg_map)
    ordered_cols = [col for col in d.columns if col in dedup.columns]
    dedup = dedup[ordered_cols].copy()
    dedup = ensure_numeric_tracks(dedup)

    summary = pd.DataFrame(
        [
            {
                "n_rows_before": int(d.shape[0]),
                "n_rows_after": int(dedup.shape[0]),
                "n_duplicate_pairs": int(dup_pairs.shape[0]),
                "n_rows_removed": int(d.shape[0] - dedup.shape[0]),
            }
        ]
    )
    return dedup, dup_rows.sort_values(["TrackID", "Time"]).reset_index(drop=True), summary


def _write_failure_diagnostics(
    file_path: Path,
    df: pd.DataFrame,
    out_prefix: Path,
    exc: Exception,
) -> None:
    """Persist diagnostics for one failed batch item."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    diag_summary, dup_rows, ref_dup_tracks = _collect_track_table_diagnostics(df)
    diag_summary.insert(0, "source_file", str(file_path))
    diag_summary.to_csv(str(out_prefix) + "_failure_diagnostics_summary.csv", index=False)
    dup_rows.to_csv(str(out_prefix) + "_failure_duplicate_track_time_rows.csv", index=False)
    ref_dup_tracks.to_csv(str(out_prefix) + "_failure_reference_time_duplicate_track_ids.csv", index=False)
    with open(str(out_prefix) + "_failure_traceback.txt", "w", encoding="utf-8") as fh:
        fh.write(f"source_file: {file_path}\n")
        fh.write(f"exception_type: {type(exc).__name__}\n")
        fh.write(f"exception_message: {exc}\n\n")
        fh.write(traceback.format_exc())


def _compute_step_displacement_table(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    method_name: str,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Compute consecutive-step planar and axial displacement metrics per track."""
    d = ensure_numeric_tracks(df)
    rows: list[dict[str, float | int | str]] = []
    for track_id, grp in d.groupby("TrackID", sort=True):
        g = grp.sort_values("Time").copy()
        coords = g[[x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        times = pd.to_numeric(g["Time"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times) & np.all(np.isfinite(coords), axis=1)
        if int(np.sum(valid)) < 2:
            continue
        coords = coords[valid]
        times = times[valid]
        delta = coords[1:] - coords[:-1]
        delta_t = times[1:] - times[:-1]
        keep = np.isfinite(delta_t) & (delta_t > 0)
        if int(np.sum(keep)) == 0:
            continue
        delta = delta[keep]
        delta_t = delta_t[keep]
        t0 = times[:-1][keep]
        t1 = times[1:][keep]
        d_xy = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        d_z = np.abs(delta[:, 2])
        ratio = d_z / np.maximum(d_xy, eps)
        for idx in range(delta.shape[0]):
            rows.append(
                {
                    "method": method_name,
                    "TrackID": float(track_id),
                    "time_start": float(t0[idx]),
                    "time_end": float(t1[idx]),
                    "delta_t": float(delta_t[idx]),
                    "dx": float(delta[idx, 0]),
                    "dy": float(delta[idx, 1]),
                    "dz_signed": float(delta[idx, 2]),
                    "d_xy": float(d_xy[idx]),
                    "d_z": float(d_z[idx]),
                    "d_z_over_d_xy": float(ratio[idx]),
                }
            )
    return pd.DataFrame(rows)


def _summarize_track_displacement_metrics(step_df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Aggregate stepwise displacement metrics into one robust row per track."""
    if step_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "TrackID",
                "n_steps",
                "mean_d_xy",
                "median_d_xy",
                "std_d_xy",
                "mean_d_z",
                "median_d_z",
                "std_d_z",
                "mean_d_z_over_d_xy_stepwise",
                "median_d_z_over_d_xy_stepwise",
                "ratio_mean_d_z_over_d_xy",
                "ratio_median_d_z_over_d_xy",
            ]
        )

    rows: list[dict[str, float | int | str]] = []
    for (method_name, track_id), grp in step_df.groupby(["method", "TrackID"], sort=True):
        dxy = pd.to_numeric(grp["d_xy"], errors="coerce").to_numpy(dtype=float)
        dz = pd.to_numeric(grp["d_z"], errors="coerce").to_numpy(dtype=float)
        ratio_step = pd.to_numeric(grp["d_z_over_d_xy"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "method": str(method_name),
                "TrackID": float(track_id),
                "n_steps": int(grp.shape[0]),
                "mean_d_xy": float(np.nanmean(dxy)),
                "median_d_xy": float(np.nanmedian(dxy)),
                "std_d_xy": float(np.nanstd(dxy)),
                "mean_d_z": float(np.nanmean(dz)),
                "median_d_z": float(np.nanmedian(dz)),
                "std_d_z": float(np.nanstd(dz)),
                "mean_d_z_over_d_xy_stepwise": float(np.nanmean(ratio_step)),
                "median_d_z_over_d_xy_stepwise": float(np.nanmedian(ratio_step)),
                "ratio_mean_d_z_over_d_xy": float(np.nanmean(dz) / max(np.nanmean(dxy), eps)),
                "ratio_median_d_z_over_d_xy": float(np.nanmedian(dz) / max(np.nanmedian(dxy), eps)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_metrics_across_tracks(track_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Summarize one or more track-level metrics across tracks for each method."""
    rows: list[dict[str, float | int | str]] = []
    for method_name, grp in track_df.groupby("method", sort=False):
        for metric in metric_cols:
            vals = pd.to_numeric(grp[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "method": str(method_name),
                    "metric": metric,
                    "n_tracks": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan"),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    return pd.DataFrame(rows)


def _get_method_color(method_name: str) -> str:
    """Return a stable, muted color for a method name."""
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    fallback = plt.get_cmap("tab10")
    return fallback(abs(hash(method_name)) % 10)


def _plot_violin_with_points(
    track_df: pd.DataFrame,
    methods: list[str],
    metrics: list[tuple[str, str]],
    out_path: Path,
) -> Path:
    """Plot per-track metric distributions as violin plots with jittered points."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 3.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    max_val = 0.0
    for metric_col, _ in metrics:
        vals = pd.to_numeric(track_df[metric_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            max_val = max(max_val, float(np.max(vals)))
    y_max = max(max_val, 1.0)

    for ax, (metric_col, title) in zip(axes, metrics):
        values_by_method: list[np.ndarray] = []
        for method_name in methods:
            vals = pd.to_numeric(
                track_df.loc[track_df["method"] == method_name, metric_col],
                errors="coerce",
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            values_by_method.append(vals)

        positions = np.arange(1, len(methods) + 1)
        non_empty = [(pos, vals) for pos, vals in zip(positions, values_by_method) if vals.size > 0]
        if non_empty:
            non_empty_pos = [pos for pos, _ in non_empty]
            non_empty_vals = [vals for _, vals in non_empty]
            parts = ax.violinplot(non_empty_vals, positions=non_empty_pos, widths=0.85, showmedians=True)
            for body, pos in zip(parts["bodies"], non_empty_pos):
                body.set_facecolor(_get_method_color(methods[int(pos) - 1]))
                body.set_edgecolor(_get_method_color(methods[int(pos) - 1]))
                body.set_alpha(0.35)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.9)

        for pos, vals, method_name in zip(positions, values_by_method, methods):
            if vals.size == 0:
                continue
            x_jitter = pos + rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(
                x_jitter,
                vals,
                s=14,
                alpha=0.7,
                color=_get_method_color(method_name),
                edgecolors="none",
            )

        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylim(0.0, y_max)
        if "ratio" in metric_col:
            ax.set_ylabel("value (unitless)")
        else:
            ax.set_ylabel("distance (IMARIS units)")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Per-track displacement summaries", fontsize=13)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_metric_histograms_by_method(
    track_df: pd.DataFrame,
    methods: list[str],
    metric_col: str,
    metric_title: str,
    out_path: Path,
) -> Path:
    """Plot one histogram panel per method for a selected track-level metric."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(methods), 1, figsize=(7.0, 2.2 * len(methods)), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    all_vals = pd.to_numeric(track_df[metric_col], errors="coerce").to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size > 0:
        x_max = float(np.max(all_vals))
        if not np.isfinite(x_max) or x_max <= 0:
            x_max = 1.0
    else:
        x_max = 1.0

    for ax, method_name in zip(axes, methods):
        vals = pd.to_numeric(
            track_df.loc[track_df["method"] == method_name, metric_col],
            errors="coerce",
        ).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            ax.hist(vals, bins=20, color=_get_method_color(method_name), alpha=0.8, edgecolor="white")
            ax.axvline(float(np.median(vals)), color="black", linestyle="--", linewidth=1.0, label="median")
        ax.set_xlim(left=0.0, right=x_max)
        ax.set_ylabel(method_name)
        ax.grid(axis="y", alpha=0.2)
        if vals.size > 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[0].set_title(metric_title)
    if "ratio" in metric_col:
        axes[-1].set_xlabel(f"{metric_col} (unitless)")
    else:
        axes[-1].set_xlabel(f"{metric_col} (IMARIS units)")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_method_summary_vs_original(
    summary_df: pd.DataFrame,
    methods: list[str],
    metrics: list[tuple[str, str]],
    out_path: Path,
) -> Path:
    """Plot method-level medians/IQR against the original-data baseline."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.3 * len(metrics), 4.7), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    positions = np.arange(1, len(methods) + 1)
    for ax, (metric_col, title) in zip(axes, metrics):
        s_method = summary_df[(summary_df["metric"] == metric_col) & (summary_df["method"].isin(methods))].copy()
        s_method = s_method.set_index("method").reindex(methods).reset_index()
        s_orig = summary_df[(summary_df["metric"] == metric_col) & (summary_df["method"] == "original")]
        if not s_orig.empty:
            orig_med = float(s_orig.iloc[0]["median"])
            ax.axhline(orig_med, color="#C44E52", linestyle="--", linewidth=1.4, label="original median")

        med = pd.to_numeric(s_method["median"], errors="coerce").to_numpy(dtype=float)
        q25 = pd.to_numeric(s_method["q25"], errors="coerce").to_numpy(dtype=float)
        q75 = pd.to_numeric(s_method["q75"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack([med - q25, q75 - med])
        ax.errorbar(positions, med, yerr=yerr, fmt="o", color="black", capsize=4, linewidth=1.1)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        if not s_orig.empty:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Method summary against original baseline", fontsize=13)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _save_displacement_metric_outputs(
    df_orig: pd.DataFrame,
    corr_by_method: dict[str, pd.DataFrame],
    out_prefix: Path,
) -> None:
    """Compute and save Joel-specific planar/axial displacement comparison outputs."""
    out_dir = out_prefix.parent / "displacement_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_methods = [name for name in DEFAULT_METHODS if name in corr_by_method]
    if not corr_methods:
        return
    display_methods = ["original"] + corr_methods

    step_tables = [
        _compute_step_displacement_table(
            df_orig,
            x_col="Position X",
            y_col="Position Y",
            z_col="Position Z",
            method_name="original",
        )
    ]
    for method_name in corr_methods:
        step_tables.append(
            _compute_step_displacement_table(
                corr_by_method[method_name],
                x_col="Position X_corr",
                y_col="Position Y_corr",
                z_col="Position Z_corr",
                method_name=method_name,
            )
        )

    step_df = pd.concat(step_tables, ignore_index=True) if step_tables else pd.DataFrame()
    track_df = _summarize_track_displacement_metrics(step_df)
    summary_df = _summarize_metrics_across_tracks(
        track_df,
        metric_cols=[
            "median_d_xy",
            "median_d_z",
            "ratio_median_d_z_over_d_xy",
            "mean_d_xy",
            "mean_d_z",
            "ratio_mean_d_z_over_d_xy",
        ],
    )

    step_df.to_csv(out_dir / f"{out_prefix.name}_displacement_metrics_per_step.csv", index=False)
    track_df.to_csv(out_dir / f"{out_prefix.name}_displacement_metrics_per_track.csv", index=False)
    summary_df.to_csv(out_dir / f"{out_prefix.name}_displacement_metrics_summary.csv", index=False)

    plot_metrics = [
        ("median_d_xy", r"median $d_{xy}$ per track"),
        ("median_d_z", r"median $d_z$ per track"),
        ("ratio_median_d_z_over_d_xy", r"median $d_z/d_{xy}$ per track"),
    ]

    _plot_violin_with_points(
        track_df=track_df[track_df["method"].isin(display_methods)].copy(),
        methods=display_methods,
        metrics=plot_metrics,
        out_path=out_dir / f"{out_prefix.name}_violin_displacement_metrics.png",
    )
    _plot_metric_histograms_by_method(
        track_df=track_df[track_df["method"].isin(display_methods)].copy(),
        methods=display_methods,
        metric_col="median_d_xy",
        metric_title=r"Distribution of median $d_{xy}$ per track",
        out_path=out_dir / f"{out_prefix.name}_histogram_d_xy.png",
    )
    _plot_metric_histograms_by_method(
        track_df=track_df[track_df["method"].isin(display_methods)].copy(),
        methods=display_methods,
        metric_col="median_d_z",
        metric_title=r"Distribution of median $d_z$ per track",
        out_path=out_dir / f"{out_prefix.name}_histogram_d_z.png",
    )
    _plot_metric_histograms_by_method(
        track_df=track_df[track_df["method"].isin(display_methods)].copy(),
        methods=display_methods,
        metric_col="ratio_median_d_z_over_d_xy",
        metric_title=r"Distribution of median $d_z/d_{xy}$ per track",
        out_path=out_dir / f"{out_prefix.name}_histogram_dz_over_dxy.png",
    )


def run_methods_for_dataframe(
    df: pd.DataFrame,
    out_prefix: Path,
    methods: list[str],
    opts: dict,
    use_stationary_subset: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run selected drift-correction methods and write method-specific artifacts."""
    df = ensure_numeric_tracks(df)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    stationary_ids = None
    if use_stationary_subset:
        sel = find_stationary_tracks(df)
        sel.table.to_csv(str(out_prefix) + "_stationary_subset.csv", index=False)
        if len(sel.track_ids) >= int(opts["min_pairs"]):
            stationary_ids = sorted(sel.track_ids)

    corr_by_method: dict[str, pd.DataFrame] = {}

    if "rbf" in methods:
        print(f"  [method] rbf | file={out_prefix.stem}")
        df_rbf, tf_table, _, _, _ = correct_positions_rbf_to_reference(
            df,
            reference_time=opts["reference_time"],
            m=opts["m_centers"],
            keep_fraction=opts["keep_fraction"],
            lam=opts["lam"],
            sigma=opts["sigma"],
            min_pairs=opts["min_pairs"],
            seed=opts["seed"],
            fit_track_ids=stationary_ids,
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
        corr_by_method["rbf"] = df_rbf

    if "rigid" in methods:
        print(f"  [method] rigid | file={out_prefix.stem}")
        step, tf_rigid, times = build_step_transforms(
            df,
            keep_fraction=opts["keep_fraction"],
            min_pairs=opts["min_pairs"],
            mode=opts["rigid_mode"],
            inlier_reestimate_iters=opts["inlier_reestimate_iters"],
        )
        ref_time = float(times.min()) if opts["reference_time"] is None else float(opts["reference_time"])
        df_rigid = apply_rigid_correction_to_tracks(df, step, times, ref_time)
        df_rigid.to_csv(str(out_prefix) + "_positions_corrected_rigid.csv", index=False)
        corr_by_method["rigid"] = df_rigid
        if bool(opts.get("rigid_smooth_enable", 0)):
            df_rigid_s = apply_rigid_correction_with_optional_smoothing(
                df=df,
                step=step,
                times=times,
                ref_time=ref_time,
                smoothing_enabled=True,
                smoothing_method=str(opts.get("rigid_smooth_method", "savgol")),
                smoothing_params={
                    "window_length": int(opts.get("rigid_smooth_window", 9)),
                    "polyorder": int(opts.get("rigid_smooth_polyorder", 2)),
                    "detect_stage_jumps": bool(opts.get("smooth_detect_stage_jumps", 1)),
                    "jump_thresh_mad": float(opts.get("smooth_jump_thresh_mad", 6.0)),
                },
                out_prefix=out_prefix,
                tag="rigid_smooth",
            )
            df_rigid_s.to_csv(str(out_prefix) + "_positions_corrected_rigid_smooth.csv", index=False)
            corr_by_method["rigid_smooth"] = df_rigid_s

    if "local" in methods:
        print(f"  [method] local | file={out_prefix.stem}")
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
        if bool(opts.get("local_smooth_enable", 0)):
            local_smooth_prefix = Path(str(out_prefix) + "_local_smooth")
            df_local_s, *_ = run_local_drift_correction_for_file_with_optional_smoothing(
                df=df,
                out_prefix=local_smooth_prefix,
                reference_time=opts["reference_time"],
                k_neighbors=opts["k_neighbors"],
                radius_um=opts["radius_um"],
                ell_um_graph=opts["ell_um_graph"],
                ell_um_field=opts["ell_um_field"],
                min_edge_weight=opts["min_edge_weight"],
                max_pairs_per_node=opts["max_pairs_per_node"],
                use_track_mean_position=True,
                smoothing_enabled=True,
                smoothing_method=str(opts.get("local_smooth_method", "savgol")),
                smoothing_params={
                    "window_length": int(opts.get("local_smooth_window", 9)),
                    "polyorder": int(opts.get("local_smooth_polyorder", 2)),
                    "detect_stage_jumps": bool(opts.get("smooth_detect_stage_jumps", 1)),
                    "jump_thresh_mad": float(opts.get("smooth_jump_thresh_mad", 6.0)),
                },
            )
            df_local_s.to_csv(str(out_prefix) + "_positions_corrected_local_smooth.csv", index=False)
            corr_by_method["local_smooth"] = df_local_s

    if "st_rbf" in methods:
        diag_summary, _, _ = _collect_track_table_diagnostics(df)
        diag_row = diag_summary.iloc[0].to_dict()
        print(
            "  [method] st_rbf | "
            f"file={out_prefix.stem} | "
            f"rows={int(diag_row['n_rows'])} | "
            f"tracks={int(diag_row['n_tracks'])} | "
            f"times={int(diag_row['n_timepoints'])} | "
            f"dup_track_time_pairs={int(diag_row['duplicate_track_time_pairs'])} | "
            f"ref_dup_track_ids={int(diag_row['reference_time_duplicate_track_ids'])}"
        )
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
            fit_track_ids=stationary_ids,
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
        df_st.to_csv(str(out_prefix) + "_positions_corrected_st_rbf.csv", index=False)
        corr_by_method["st_rbf"] = df_st

    return corr_by_method


def postprocess_outputs(
    df_orig: pd.DataFrame,
    out_prefix: Path,
    corr_by_method: dict[str, pd.DataFrame],
) -> None:
    """Create Joel-specific qualitative plots and optionally trim auxiliary outputs."""
    for method_name, df_corr in corr_by_method.items():
        vis_outputs = save_visual_inspection_plots(df_orig, df_corr, out_prefix, method_name)
        if ANNOTATE_TRACK_IDS_IN_COMPARISON_PLOTS:
            annotated_path = vis_outputs["tracks_comparison"].with_name(
                vis_outputs["tracks_comparison"].stem + "_with_ids" + vis_outputs["tracks_comparison"].suffix
            )
            _plot_tracks_comparison_for_joel(
                df_orig=df_orig,
                df_corr=df_corr,
                out_path=annotated_path,
                title=f"Track comparison: {method_name} (with IDs)",
                annotate_track_ids=True,
            )
    _save_displacement_metric_outputs(df_orig=df_orig, corr_by_method=corr_by_method, out_prefix=out_prefix)
    if SAVE_ONLY_MINIMAL_OUTPUTS:
        _cleanup_minimal_outputs(out_prefix)
# %% HOW TO RUN A SINGLE FILE
RUN_SINGLE_FILE = True
if RUN_SINGLE_FILE:
    SINGLE_FILE_NAME = "registered_stack_5D_251008_P2_485um_analysis.xls"
    METHODS_SINGLE = list(DEFAULT_METHODS)
    USE_STATIONARY_SUBSET_SINGLE = False

    file_path = JOEL_DIR / SINGLE_FILE_NAME
    df_single = _read_joel_imaris_file(file_path)

    out_prefix_single = RESULTS_ROOT / "single_file" / file_path.stem / file_path.stem
    if WRITE_INPUT_SUMMARY:
        _write_input_summary(df_single, out_prefix_single)

    corr_single = run_methods_for_dataframe(
        df=df_single,
        out_prefix=out_prefix_single,
        methods=METHODS_SINGLE,
        opts=OPTIONS,
        use_stationary_subset=USE_STATIONARY_SUBSET_SINGLE,
    )
    postprocess_outputs(df_single, out_prefix_single, corr_single)
print("Completed single-file run.")
# %% HOW TO BATCH RUN ALL JOEL FILES
RUN_BATCH = True
if RUN_BATCH:
    METHODS_BATCH = list(DEFAULT_METHODS)
    USE_STATIONARY_SUBSET_BATCH = False
    RUN_BATCH_SMOOTHED = False
    SELECTED_FILES: list[str] = []

    opts_batch = dict(OPTIONS)
    batch_root = RESULTS_ROOT / "batch_default"
    if RUN_BATCH_SMOOTHED:
        opts_batch["rigid_smooth_enable"] = 1
        opts_batch["local_smooth_enable"] = 1
        batch_root = RESULTS_ROOT / "batch_with_smoothing"

    all_files = list_imaris_xls_files(JOEL_DIR)
    if SELECTED_FILES:
        selected = {name for name in SELECTED_FILES}
        all_files = [p for p in all_files if p.name in selected]

    for idx, file_path in enumerate(all_files, start=1):
        df_batch = _read_joel_imaris_file(file_path)
        out_prefix_batch = batch_root / file_path.stem / file_path.stem
        diag_summary, _, _ = _collect_track_table_diagnostics(df_batch)
        diag_row = diag_summary.iloc[0].to_dict()

        print(
            f"[BATCH {idx}/{len(all_files)}] {file_path.name} | "
            f"rows={int(diag_row['n_rows'])} | "
            f"tracks={int(diag_row['n_tracks'])} | "
            f"times={int(diag_row['n_timepoints'])} | "
            f"dup_track_time_pairs={int(diag_row['duplicate_track_time_pairs'])} | "
            f"ref_dup_track_ids={int(diag_row['reference_time_duplicate_track_ids'])}"
        )

        if int(diag_row["duplicate_track_time_pairs"]) > 0:
            df_batch_dedup, dup_rows, dedup_summary = _deduplicate_track_time_rows(df_batch)
            out_prefix_batch.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"  [warn] deduplicating {int(dedup_summary.iloc[0]['n_duplicate_pairs'])} duplicate "
                f"(TrackID, Time) pairs in {file_path.name}; "
                f"rows {int(dedup_summary.iloc[0]['n_rows_before'])} -> {int(dedup_summary.iloc[0]['n_rows_after'])}"
            )
            dup_rows.to_csv(str(out_prefix_batch) + "_input_duplicate_track_time_rows.csv", index=False)
            dedup_summary.to_csv(str(out_prefix_batch) + "_input_deduplication_summary.csv", index=False)
            df_batch = df_batch_dedup

        if WRITE_INPUT_SUMMARY:
            _write_input_summary(df_batch, out_prefix_batch)
        try:
            corr_batch = run_methods_for_dataframe(
                df=df_batch,
                out_prefix=out_prefix_batch,
                methods=METHODS_BATCH,
                opts=opts_batch,
                use_stationary_subset=USE_STATIONARY_SUBSET_BATCH,
            )
            postprocess_outputs(df_batch, out_prefix_batch, corr_batch)
        except Exception as exc:
            print(f"[ERROR] Failed on {file_path.name}: {type(exc).__name__}: {exc}")
            _write_failure_diagnostics(file_path, df_batch, out_prefix_batch, exc)
            raise

# %% END
