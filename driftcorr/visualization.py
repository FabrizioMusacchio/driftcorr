"""Visualization utilities for qualitative drift-correction inspection.

This module produces method-comparison figures designed for manual review:
- trajectory-level comparison plots (original vs corrected),
- frame-by-frame 3D snapshots for scrolling inspection,
- optional ground-truth-aware panels on synthetic data:
  (GT | Original input | Corrected | GT-Corrected residuals).

These plots complement scalar metrics by revealing local geometric artifacts,
over-correction, or residual drift patterns that are difficult to diagnose from
single summary numbers.

Mathematical role:
- No optimization is performed here.
- The module visualizes outputs from models that typically apply:
  `x_corr = x_obs - d_hat(x,t)` or rigid equivalents.

Controlling parameters:
- figure sizing and axis limits are derived from data ranges,
- optional `df_groundtruth` enables a third GT panel in frame plots.

Effects:
- Consistent axis limits make before/after comparisons visually trustworthy.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %% FUNCTIONS
def _compute_limits(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    df_groundtruth: pd.DataFrame | None = None,
) -> tuple[float, float, float, float, float, float]:
    x_min = min(float(df_orig["Position X"].min()), float(df_corr["Position X_corr"].min()))
    x_max = max(float(df_orig["Position X"].max()), float(df_corr["Position X_corr"].max()))
    y_min = min(float(df_orig["Position Y"].min()), float(df_corr["Position Y_corr"].min()))
    y_max = max(float(df_orig["Position Y"].max()), float(df_corr["Position Y_corr"].max()))
    z_min = min(float(df_orig["Position Z"].min()), float(df_corr["Position Z_corr"].min()))
    z_max = max(float(df_orig["Position Z"].max()), float(df_corr["Position Z_corr"].max()))
    if df_groundtruth is not None and df_groundtruth.shape[0] > 0:
        x_min = min(x_min, float(df_groundtruth["Position X"].min()))
        x_max = max(x_max, float(df_groundtruth["Position X"].max()))
        y_min = min(y_min, float(df_groundtruth["Position Y"].min()))
        y_max = max(y_max, float(df_groundtruth["Position Y"].max()))
        z_min = min(z_min, float(df_groundtruth["Position Z"].min()))
        z_max = max(z_max, float(df_groundtruth["Position Z"].max()))
    return x_min, x_max, y_min, y_max, z_min, z_max

def _compute_box_aspect_from_limits(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> tuple[float, float, float]:
    """Return positive XYZ spans for use with set_box_aspect."""
    span_x = max(float(x_max) - float(x_min), 1e-6)
    span_y = max(float(y_max) - float(y_min), 1e-6)
    span_z = max(float(z_max) - float(z_min), 1e-6)
    return span_x, span_y, span_z

def _apply_box_aspect_from_limits(
    ax,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """Apply data-derived 3D box aspect if supported."""
    try:
        ax.set_box_aspect(_compute_box_aspect_from_limits(x_min, x_max, y_min, y_max, z_min, z_max))
    except Exception:
        pass

def plot_tracks_comparison(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    out_path: str | Path,
    title: str = "",
) -> None:
    """Plot original vs corrected tracks as full trajectories in 3D/2D views."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 18))

    x_min, x_max, y_min, y_max, z_min, z_max = _compute_limits(df_orig, df_corr)

    ax1 = fig.add_subplot(321, projection="3d")
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid]
        ax1.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.9)
    ax1.set_title("Original (3D)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(322, projection="3d")
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid]
        ax2.plot(d["Position X_corr"], d["Position Y_corr"], d["Position Z_corr"], alpha=0.9)
    ax2.set_title("Corrected (3D)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    ax3 = fig.add_subplot(323)
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid]
        ax3.plot(d["Position X"], d["Position Y"], alpha=0.7)
    ax3.set_title("Original XY")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    ax4 = fig.add_subplot(324)
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid]
        ax4.plot(d["Position X_corr"], d["Position Y_corr"], alpha=0.7)
    ax4.set_title("Corrected XY")
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    ax5 = fig.add_subplot(325)
    for tid in df_orig["TrackID"].unique():
        d = df_orig[df_orig["TrackID"] == tid]
        ax5.plot(d["Position X"], d["Position Z"], alpha=0.7)
    ax5.set_title("Original XZ")
    ax5.set_aspect("equal", adjustable="box")
    ax5.set_xlim(x_min, x_max)
    ax5.set_ylim(z_min, z_max)
    ax5.set_xlabel("X")
    ax5.set_ylabel("Z")

    ax6 = fig.add_subplot(326)
    for tid in df_corr["TrackID"].unique():
        d = df_corr[df_corr["TrackID"] == tid]
        ax6.plot(d["Position X_corr"], d["Position Z_corr"], alpha=0.7)
    ax6.set_title("Corrected XZ")
    ax6.set_aspect("equal", adjustable="box")
    ax6.set_xlim(x_min, x_max)
    ax6.set_ylim(z_min, z_max)
    ax6.set_xlabel("X")
    ax6.set_ylabel("Z")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def _format_time_for_filename(t: float) -> str:
    s = f"{float(t):.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")

def _save_frames_as_gif(frame_paths: list[Path], gif_path: Path, fps: float = 4.0) -> Path | None:
    """Create a GIF animation from frame images if Pillow is available."""
    if not frame_paths:
        return None
    try:
        from PIL import Image
    except Exception:
        return None

    duration_ms = int(round(1000.0 / max(float(fps), 0.1)))
    images: list[Image.Image] = []
    try:
        for p in frame_paths:
            with Image.open(p) as im:
                images.append(im.convert("P", palette=Image.ADAPTIVE))
        if not images:
            return None
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        first, rest = images[0], images[1:]
        first.save(
            gif_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return gif_path
    finally:
        for im in images:
            try:
                im.close()
            except Exception:
                pass

def plot_timepoint_side_by_side_3d(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    out_dir: str | Path,
    title_prefix: str = "",
    df_groundtruth: pd.DataFrame | None = None,
) -> list[Path]:
    """Save one 3D figure per timepoint: optionally GT, then original and corrected."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_min, x_max, y_min, y_max, z_min, z_max = _compute_limits(df_orig, df_corr, df_groundtruth=df_groundtruth)

    all_times = np.array(sorted(np.intersect1d(df_orig["Time"].unique(), df_corr["Time"].unique())), dtype=float)
    if df_groundtruth is not None:
        all_times = np.array(sorted(np.intersect1d(all_times, df_groundtruth["Time"].unique())), dtype=float)
    all_tids = np.array(sorted(np.intersect1d(df_orig["TrackID"].unique(), df_corr["TrackID"].unique())), dtype=float)
    if df_groundtruth is not None:
        all_tids = np.array(sorted(np.intersect1d(all_tids, df_groundtruth["TrackID"].unique())), dtype=float)
    if all_tids.size == 0:
        all_tids = np.array(sorted(np.union1d(df_orig["TrackID"].unique(), df_corr["TrackID"].unique())), dtype=float)
    if df_groundtruth is not None:
        all_tids = np.array(sorted(np.union1d(all_tids, df_groundtruth["TrackID"].unique())), dtype=float)

    vmin = float(np.min(all_tids)) if all_tids.size else 0.0
    vmax = float(np.max(all_tids)) if all_tids.size else 1.0

    saved: list[Path] = []
    for idx, t in enumerate(all_times):
        do = df_orig[df_orig["Time"] == float(t)]
        dc = df_corr[df_corr["Time"] == float(t)]
        dgt = df_groundtruth[df_groundtruth["Time"] == float(t)] if df_groundtruth is not None else None

        has_gt = dgt is not None
        if has_gt:
            fig = plt.figure(figsize=(20, 6), constrained_layout=True)
            ax_gt = fig.add_subplot(131, projection="3d")
            ax1 = fig.add_subplot(132, projection="3d")
            ax2 = fig.add_subplot(133, projection="3d")
            ax_gt.scatter(
                dgt["Position X"],
                dgt["Position Y"],
                dgt["Position Z"],
                c=dgt["TrackID"],
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
                s=18,
                alpha=0.9,
            )
            ax_gt.set_title(f"Ground Truth (no imaging drift) | Time={t}")
            ax_gt.set_xlim(x_min, x_max)
            ax_gt.set_ylim(y_min, y_max)
            ax_gt.set_zlim(z_min, z_max)
            ax_gt.set_xlabel("X")
            ax_gt.set_ylabel("Y")
            ax_gt.set_zlabel("Z")
        else:
            fig = plt.figure(figsize=(14, 6), constrained_layout=True)
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")

        ax1.scatter(
            do["Position X"],
            do["Position Y"],
            do["Position Z"],
            c=do["TrackID"],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            s=18,
            alpha=0.9,
        )
        ax1.set_title(f"Original | Time={t}")
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2.scatter(
            dc["Position X_corr"],
            dc["Position Y_corr"],
            dc["Position Z_corr"],
            c=dc["TrackID"],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            s=18,
            alpha=0.9,
        )
        ax2.set_title(f"Corrected | Time={t}")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_zlim(z_min, z_max)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        if has_gt:
            ax_gt.view_init(elev=25, azim=-60)
        ax1.view_init(elev=25, azim=-60)
        ax2.view_init(elev=25, azim=-60)

        fig.suptitle(f"{title_prefix} | Time={t}", fontsize=12)
        t_tag = _format_time_for_filename(float(t))
        out_path = out_dir / f"frame_{idx:04d}_t_{t_tag}.png"
        plt.savefig(out_path, dpi=220)
        plt.close(fig)
        saved.append(out_path)

    return saved

def plot_tracks_over_time_panels(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    out_path: str | Path,
    title_prefix: str = "",
    df_groundtruth: pd.DataFrame | None = None,
    use_data_box_aspect: bool = False,
) -> Path:
    """Plot full track trajectories over all timepoints.

    Layout:
    - with ground truth: 2x2 (GT, Original input, Corrected, Overlay GT+Corrected)
    - without ground truth: 1x2 (Original input, Corrected)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max, y_min, y_max, z_min, z_max = _compute_limits(df_orig, df_corr, df_groundtruth=df_groundtruth)

    has_gt = df_groundtruth is not None and df_groundtruth.shape[0] > 0
    if has_gt:
        fig = plt.figure(figsize=(16, 14), constrained_layout=True)
        ax_gt = fig.add_subplot(221, projection="3d")
        ax1 = fig.add_subplot(222, projection="3d")
        ax2 = fig.add_subplot(223, projection="3d")
        ax_overlay = fig.add_subplot(224, projection="3d")
    else:
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

    # Panel: GT
    if has_gt:
        assert df_groundtruth is not None
        for tid in df_groundtruth["TrackID"].dropna().unique():
            d = df_groundtruth[df_groundtruth["TrackID"] == tid].sort_values("Time")
            ax_gt.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.8, linewidth=1.0)
        ax_gt.set_title("Ground Truth tracks (all frames)")
        ax_gt.set_xlim(x_min, x_max)
        ax_gt.set_ylim(y_min, y_max)
        ax_gt.set_zlim(z_min, z_max)
        ax_gt.set_xlabel("X")
        ax_gt.set_ylabel("Y")
        ax_gt.set_zlabel("Z")
        ax_gt.view_init(elev=25, azim=-60)
        if use_data_box_aspect:
            _apply_box_aspect_from_limits(ax_gt, x_min, x_max, y_min, y_max, z_min, z_max)

    # Panel: original input
    for tid in df_orig["TrackID"].dropna().unique():
        d = df_orig[df_orig["TrackID"] == tid].sort_values("Time")
        ax1.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.8, linewidth=1.0)
    ax1.set_title("Original input tracks (all frames)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=25, azim=-60)
    if use_data_box_aspect:
        _apply_box_aspect_from_limits(ax1, x_min, x_max, y_min, y_max, z_min, z_max)

    # Panel: corrected
    for tid in df_corr["TrackID"].dropna().unique():
        d = df_corr[df_corr["TrackID"] == tid].sort_values("Time")
        ax2.plot(d["Position X_corr"], d["Position Y_corr"], d["Position Z_corr"], alpha=0.8, linewidth=1.0)
    ax2.set_title("Corrected tracks (all frames)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=25, azim=-60)
    if use_data_box_aspect:
        _apply_box_aspect_from_limits(ax2, x_min, x_max, y_min, y_max, z_min, z_max)

    # Panel: overlay GT + corrected (no residual computation, no merge)
    if has_gt:
        assert df_groundtruth is not None

        gt_ids = set(df_groundtruth["TrackID"].dropna().unique().tolist())
        corr_ids = set(df_corr["TrackID"].dropna().unique().tolist())
        common_track_ids = sorted(gt_ids.intersection(corr_ids))

        if common_track_ids:
            cmap = plt.get_cmap("turbo")
            n_tracks = max(len(common_track_ids), 1)
            denom = max(n_tracks - 1, 1)
            color_by_tid = {tid: cmap(i / denom) for i, tid in enumerate(common_track_ids)}

            for tid in common_track_ids:
                c = color_by_tid[tid]

                d_gt = df_groundtruth[df_groundtruth["TrackID"] == tid].sort_values("Time")
                d_corr = df_corr[df_corr["TrackID"] == tid].sort_values("Time")

                # GT:
                ax_overlay.plot(
                    d_gt["Position X"],
                    d_gt["Position Y"],
                    d_gt["Position Z"],
                    color=c,
                    linewidth=1.0,
                    alpha=1.0,
                    linestyle="-",
                )

                # Corrected:
                ax_overlay.plot(
                    d_corr["Position X_corr"],
                    d_corr["Position Y_corr"],
                    d_corr["Position Z_corr"],
                    color=c,
                    linewidth=1.0,
                    alpha=0.5,
                    linestyle="-",
                )

            ax_overlay.set_title("Overlay: GT and corrected tracks (all frames)")
            ax_overlay.set_xlim(x_min, x_max)
            ax_overlay.set_ylim(y_min, y_max)
            ax_overlay.set_zlim(z_min, z_max)
            ax_overlay.set_xlabel("X")
            ax_overlay.set_ylabel("Y")
            ax_overlay.set_zlabel("Z")
            ax_overlay.view_init(elev=25, azim=-60)
            if use_data_box_aspect:
                _apply_box_aspect_from_limits(ax_overlay, x_min, x_max, y_min, y_max, z_min, z_max)
            ax_overlay.text2D(
                0.02,
                0.98,
                "GT: semi-transparent tracks, corrected: solid tracks",
                transform=ax_overlay.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
        else:
            ax_overlay.set_title("Overlay: GT and corrected tracks (all frames)")
            ax_overlay.text2D(0.2, 0.5, "No common GT/corrected TrackIDs", transform=ax_overlay.transAxes)

    if title_prefix:
        fig.suptitle(f"{title_prefix} | Track trajectories over all frames", fontsize=12)

    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    
    # now, save each panel separately:
    def _safe_name(s: str) -> str:
        s = str(s).strip().lower()
        for old, new in [
            (" ", "_"),
            (":", ""),
            ("/", "_"),
            ("\\", "_"),
            ("(", ""),
            (")", ""),
            (",", ""),
            ("+", "plus"),]:
            s = s.replace(old, new)
        return s

    panel_specs: list[tuple[int, str, str]] = []
    if has_gt:
        panel_specs.append((1, "Ground Truth tracks", "ground_truth_tracks"))
    panel_specs.extend(
        [
            (2 if has_gt else 1, "Original input tracks", "original_input_tracks"),
            (3 if has_gt else 2, "Corrected tracks", "corrected_tracks"),
        ])
    if has_gt:
        panel_specs.append((4, "Overlay: GT and corrected tracks", "overlay_gt_and_corrected"))

    cm2inch = 1 / 2.54
    for panel_id, panel_title, name_stub in panel_specs:
        fig_i, ax_i = plt.subplots(
            figsize=(6.0 * cm2inch, 6.0 * cm2inch),
            subplot_kw={"projection": "3d"},
        )
        
        """ 
        # generate some dummy scatter points:
        np.random.seed(42)
        n_points = 100
        x = np.random.uniform(0, 300, n_points)
        y = np.random.uniform(0, 300, n_points)
        z = np.random.uniform(0, 300, n_points)

        cm2inch = 0.393701
        fig = plt.figure(figsize=(6.0 * cm2inch, 4.5 * cm2inch), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c="gray", s=10, alpha=0.5, label="Dummy points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(r"$x^{true}$", pad=10)
        # reduce gap between the title and plot:
        ax.view_init(elev=25, azim=-60)
        # move the plot a bit up:
        ax.set_position([0.1, 0.2, 0.8, 0.7])
        # reduce gap between the axis ticks and plot:
        ax.tick_params(axis="both", which="major", pad=-2)
        # reduce gap between the axis label and plot:
        ax.xaxis.labelpad = -2
        ax.yaxis.labelpad = -2
        ax.zaxis.labelpad = -2
        fig.savefig("/Users/husker/Science/Python/Projekte/Correct uniform movements/paper/figures/test_scatter.pdf", dpi=300, transparent=True)
        #plt.tight_layout()
        plt.close(fig)
        """

        if name_stub == "ground_truth_tracks":
            assert df_groundtruth is not None
            for tid in df_groundtruth["TrackID"].dropna().unique():
                d = df_groundtruth[df_groundtruth["TrackID"] == tid].sort_values("Time")
                ax_i.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.8, linewidth=1.0)

        elif name_stub == "original_input_tracks":
            for tid in df_orig["TrackID"].dropna().unique():
                d = df_orig[df_orig["TrackID"] == tid].sort_values("Time")
                ax_i.plot(d["Position X"], d["Position Y"], d["Position Z"], alpha=0.8, linewidth=1.0)

        elif name_stub == "corrected_tracks":
            for tid in df_corr["TrackID"].dropna().unique():
                d = df_corr[df_corr["TrackID"] == tid].sort_values("Time")
                ax_i.plot(d["Position X_corr"], d["Position Y_corr"], d["Position Z_corr"], alpha=0.8, linewidth=1.0)

        elif name_stub == "overlay_gt_and_corrected":
            assert df_groundtruth is not None

            gt_ids = set(df_groundtruth["TrackID"].dropna().unique().tolist())
            corr_ids = set(df_corr["TrackID"].dropna().unique().tolist())
            common_track_ids = sorted(gt_ids.intersection(corr_ids))

            if common_track_ids:
                cmap = plt.get_cmap("turbo")
                n_tracks = max(len(common_track_ids), 1)
                denom = max(n_tracks - 1, 1)
                color_by_tid = {tid: cmap(i / denom) for i, tid in enumerate(common_track_ids)}

                for tid in common_track_ids:
                    c = color_by_tid[tid]

                    d_gt = df_groundtruth[df_groundtruth["TrackID"] == tid].sort_values("Time")
                    d_corr = df_corr[df_corr["TrackID"] == tid].sort_values("Time")

                    ax_i.plot(
                        d_gt["Position X"],
                        d_gt["Position Y"],
                        d_gt["Position Z"],
                        color=c,
                        linewidth=1.0,
                        alpha=0.25,
                        linestyle="-",
                    )
                    ax_i.plot(
                        d_corr["Position X_corr"],
                        d_corr["Position Y_corr"],
                        d_corr["Position Z_corr"],
                        color=c,
                        linewidth=0.5,
                        alpha=1.0,
                        linestyle="-",
                    )

                """ ax_i.text2D(
                    0.02,
                    0.98,
                    "GT: semi-transparent tracks, corrected: solid tracks",
                    transform=ax_i.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
                ) """
            else:
                ax_i.text2D(0.2, 0.5, "No common GT/corrected TrackIDs", transform=ax_i.transAxes)

        ax_i.set_title(panel_title, pad=-55, fontsize=8)
        ax_i.set_xlim(x_min, x_max)
        ax_i.set_ylim(y_min, y_max)
        ax_i.set_zlim(z_min, z_max)
        ax_i.set_xlabel("X")
        ax_i.set_ylabel("Y")
        ax_i.set_zlabel("Z")
        ax_i.view_init(elev=25, azim=-60)
        if use_data_box_aspect:
            _apply_box_aspect_from_limits(ax_i, x_min, x_max, y_min, y_max, z_min, z_max)
        
        ax_i.set_position([0.1, 0.1, 0.7, 0.8])
        ax_i.tick_params(axis="both", which="major", pad=-3)
        ax_i.xaxis.labelpad = -7
        ax_i.yaxis.labelpad = -7
        ax_i.zaxis.labelpad = -7

        fig_i.tight_layout()
        out_i = out_path.parent / f"{out_path.stem}_panel_{panel_id}_{_safe_name(name_stub)}.pdf"
        fig_i.savefig(out_i, dpi=300, transparent=True)
        plt.close(fig_i)
    
    
    return out_path

def save_visual_inspection_plots(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    out_prefix: str | Path,
    method_name: str,
    df_groundtruth: pd.DataFrame | None = None,
    use_data_box_aspect_tracks_over_time: bool = False,
) -> dict[str, Path]:
    """Save both trajectory and per-timepoint visual inspection plots."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    tracks_path = out_prefix.with_name(out_prefix.name + f"_tracks_comparison_{method_name}.png")
    frames_dir = out_prefix.parent / f"{out_prefix.name}_timeseries_{method_name}"

    plot_tracks_comparison(
        df_orig=df_orig,
        df_corr=df_corr,
        out_path=tracks_path,
        title=f"Track comparison: {method_name}",
    )
    frame_paths = plot_timepoint_side_by_side_3d(
        df_orig=df_orig,
        df_corr=df_corr,
        out_dir=frames_dir,
        title_prefix=method_name,
        df_groundtruth=df_groundtruth,
    )
    tracks_over_time_path = plot_tracks_over_time_panels(
        df_orig=df_orig,
        df_corr=df_corr,
        out_path=frames_dir / "tracks_over_time_panels.png",
        title_prefix=method_name,
        df_groundtruth=df_groundtruth,
        use_data_box_aspect=use_data_box_aspect_tracks_over_time,
    )
    gif_path = _save_frames_as_gif(frame_paths, frames_dir / "timeseries.gif", fps=4.0)

    out: dict[str, Path] = {
        "tracks_comparison": tracks_path,
        "timeseries_dir": frames_dir,
        "tracks_over_time_panels": tracks_over_time_path,
    }
    if gif_path is not None:
        out["timeseries_gif"] = gif_path
    return out
# %% END
