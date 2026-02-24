"""Benchmark point-cloud association (Module 1) on synthetic blinded datasets.

This script is intentionally API-first and independent from CLI workflows.
It evaluates association quality when frame-wise point clouds are provided
without IDs.

Outputs are written under:
`<PROJECT_ROOT>/Synthetic files parameter sweep/driftcorr_results_assoc`

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

# Import association APIs, synthetic helpers, and plotting utilities.
# Why this section exists:
# - Keep this benchmark runnable from a cloned repo without pip install.
# - Ensure all imports resolve through either installed package or local path.

from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from driftcorr.association import (
        DriftModelAdapterRigidStep,
        associate_nodrift,
        associate_with_registration,
        em_associate_and_driftfit,
        load_blind_frames,
        synthetic_generate_blind_pair,
    )
    from driftcorr.association.scoring import evaluate_against_groundtruth
    from driftcorr.association.visualization import (
        plot_3d_frames_comparison,
        plot_accuracy_over_time,
        plot_rms_over_time,
    )
    from driftcorr.io_imaris import read_imaris_position_xls
    from driftcorr.association.types import PointCloudFrame
except ModuleNotFoundError as exc:
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from driftcorr.association import (
            DriftModelAdapterRigidStep,
            associate_nodrift,
            associate_with_registration,
            em_associate_and_driftfit,
            load_blind_frames,
            synthetic_generate_blind_pair,
        )
        from driftcorr.association.scoring import evaluate_against_groundtruth
        from driftcorr.association.visualization import (
            plot_3d_frames_comparison,
            plot_accuracy_over_time,
            plot_rms_over_time,
        )
        from driftcorr.io_imaris import read_imaris_position_xls
        from driftcorr.association.types import PointCloudFrame
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import driftcorr association module. Run from repository root or install driftcorr."
        ) from exc

# %% SETTINGS
"""Define centralized dataset and result roots and benchmark toggles.

What is tested:
- A) no-drift dx sweep (core module-1 validation)
- B) drift-present strategy comparison (optional scaffold)

Why it matters:
- The no-drift sweep isolates association behavior from drift-model confounds.
- Optional drift strategy runs test registration and EM integration paths.
"""

ASSOC_ROOT = PROJECT_ROOT / "Synthetic files track inference parameter sweep" 
DATA_ROOT = ASSOC_ROOT / "_blind_pairs"

RUN_GENERATE_DATASETS = True
RUN_NODRIFT_DX_SWEEP = True
RUN_DRIFT_STRATEGY_COMPARISON = False
OVERWRITE_DATASETS = True

N_FILES = 3
N_TRACKS = 80
N_TIMES = 20
BASE_SEED = 42

DX_SCALES = [0.2, 0.5, 1.0, 2.0, 4.0]
NODRIFT_METHODS = [
    {
        "name": "hungarian_gating_cp",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_position",
    },
    {
        "name": "hungarian_gating_cv",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_velocity",
    },
    {
        "name": "nearest_gating_cp",
        "kind": "nodrift",
        "method": "nearest",
        "motion_model": "constant_position",
    },
]

DRIFT_REGIMES = ["rotation_drift", "smooth"]

# %% FUNCTIONS
def _results_root(tag: str) -> Path:
    safe = str(tag).strip().replace(" ", "_").replace("-", "_")
    p = ASSOC_ROOT / safe
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fmt_token(x: float) -> str:
    return f"{float(x):.2f}".replace(".", "p")

def _plot_link_cost_over_time(links_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    if not links_df.empty and {"t", "cost"}.issubset(links_df.columns):
        tab = links_df.copy()
        tab["t"] = pd.to_numeric(tab["t"], errors="coerce")
        tab["cost"] = pd.to_numeric(tab["cost"], errors="coerce")
        tab = tab.dropna(subset=["t", "cost"])
        if not tab.empty:
            agg = tab.groupby("t", as_index=False)["cost"].median().rename(columns={"cost": "median_cost"})
            ax.plot(agg["t"], agg["median_cost"], marker="o")
    ax.set_title("Median link cost over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cost")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_metric_vs_dx(agg: pd.DataFrame, metric: str, out_path: Path, title: str) -> bool:
    if agg.empty or metric not in agg.columns or "dx_scale" not in agg.columns or "method" not in agg.columns:
        return False
    d = agg.copy()
    d["dx_scale"] = pd.to_numeric(d["dx_scale"], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["dx_scale", metric, "method"])
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method in sorted(d["method"].astype(str).unique().tolist()):
        g = d[d["method"].astype(str) == str(method)].copy()
        stat = g.groupby("dx_scale", as_index=False)[metric].agg(mean="mean", std="std").sort_values("dx_scale")
        x = stat["dx_scale"].to_numpy(float)
        y = stat["mean"].to_numpy(float)
        s = pd.to_numeric(stat["std"], errors="coerce").fillna(0.0).to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=method)
        ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("dx_scale (um)")
    ax.set_ylabel(metric)
    if metric in {"link_precision_mean", "link_recall_mean", "link_f1_mean", "track_purity", "coverage"}:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _plot_metric_heatmap_dx_method(agg: pd.DataFrame, metric: str, out_path: Path, title: str) -> bool:
    if agg.empty or metric not in agg.columns or "dx_scale" not in agg.columns or "method" not in agg.columns:
        return False
    d = agg.copy()
    d["dx_scale"] = pd.to_numeric(d["dx_scale"], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["dx_scale", metric, "method"])
    if d.empty:
        return False
    piv = d.pivot_table(index="method", columns="dx_scale", values=metric, aggfunc="mean")
    if piv.empty:
        return False
    x_vals = sorted([float(v) for v in piv.columns.tolist()])
    methods = sorted([str(v) for v in piv.index.tolist()])
    piv = piv.reindex(index=methods, columns=x_vals)
    mat = piv.to_numpy(float)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("dx_scale (um)")
    ax.set_ylabel("method")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _plot_metric_by_regime_strategy(agg: pd.DataFrame, metric: str, out_path: Path, title: str) -> bool:
    if agg.empty or metric not in agg.columns or "regime" not in agg.columns or "strategy" not in agg.columns:
        return False
    d = agg.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric, "regime", "strategy"])
    if d.empty:
        return False
    regimes = sorted(d["regime"].astype(str).unique().tolist())
    strategies = sorted(d["strategy"].astype(str).unique().tolist())
    piv_mean = d.pivot_table(index="regime", columns="strategy", values=metric, aggfunc="mean").reindex(index=regimes, columns=strategies)
    piv_std = d.pivot_table(index="regime", columns="strategy", values=metric, aggfunc="std").reindex(index=regimes, columns=strategies)
    x = np.arange(len(regimes), dtype=float)
    w = min(0.8 / max(len(strategies), 1), 0.28)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for i, s in enumerate(strategies):
        xi = x + (i - (len(strategies) - 1) / 2.0) * w
        yy = pd.to_numeric(piv_mean[s], errors="coerce").to_numpy(float)
        ee = pd.to_numeric(piv_std[s], errors="coerce").fillna(0.0).to_numpy(float)
        ax.bar(xi, yy, width=w, yerr=ee, capsize=3, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    if metric in {"link_precision_mean", "link_recall_mean", "link_f1_mean", "track_purity", "coverage"}:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _plot_metric_heatmap_regime_strategy(agg: pd.DataFrame, metric: str, out_path: Path, title: str) -> bool:
    if agg.empty or metric not in agg.columns or "regime" not in agg.columns or "strategy" not in agg.columns:
        return False
    d = agg.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric, "regime", "strategy"])
    if d.empty:
        return False
    piv = d.pivot_table(index="strategy", columns="regime", values=metric, aggfunc="mean")
    if piv.empty:
        return False
    strategies = sorted([str(v) for v in piv.index.tolist()])
    regimes = sorted([str(v) for v in piv.columns.tolist()])
    piv = piv.reindex(index=strategies, columns=regimes)
    mat = piv.to_numpy(float)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(regimes, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("regime")
    ax.set_ylabel("strategy")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _gt_tracks_with_pointindex(gt_root: Path, regime: str, file_stem: str, mapping_df: pd.DataFrame) -> pd.DataFrame:
    gt_path = gt_root / regime / "_groundtruth" / f"{file_stem}_groundtruth.csv"
    gt = pd.read_csv(gt_path)
    gt = gt[["Time", "TrackID", "Position X", "Position Y", "Position Z"]].copy()

    map_sub = mapping_df[(mapping_df["Regime"].astype(str) == str(regime)) & (mapping_df["File"].astype(str) == str(file_stem))].copy()
    map_sub = map_sub[["Time", "PointIndex", "GT_TrackID"]]
    gt = gt.merge(map_sub, left_on=["Time", "TrackID"], right_on=["Time", "GT_TrackID"], how="inner")
    gt = gt.drop(columns=["GT_TrackID"])
    return gt

def _frames_from_gt_tracks(gt_tracks: pd.DataFrame) -> list[PointCloudFrame]:
    frames: list[PointCloudFrame] = []
    for t, grp in gt_tracks.groupby("Time"):
        g = grp.sort_values("PointIndex") if "PointIndex" in grp.columns else grp
        xyz = g[["Position X", "Position Y", "Position Z"]].to_numpy(float)
        frames.append(PointCloudFrame(time=float(t), xyz=xyz, meta={"source": "gt"}))
    return sorted(frames, key=lambda f: float(f.time))

def _evaluate_and_save_run(
    out_dir: Path,
    assoc_result,
    gt_tracks_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    blind_frames: list[PointCloudFrame],
    regime: str,
    file_stem: str,
    method_name: str,
    extras: dict[str, float | str],
) -> dict[str, float | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_df = assoc_result.tracks_df.copy()
    links_df = assoc_result.links_df.copy()
    frame_stats_df = assoc_result.frame_stats_df.copy()

    summary_gt, per_time, mapping_table = evaluate_against_groundtruth(
        tracks_df,
        mapping_df,
        links_df=links_df,
        gt_tracks_df=gt_tracks_df,
    )

    tracks_df.to_csv(out_dir / f"{file_stem}_tracks_inferred_{method_name}.csv", index=False)
    links_df.to_csv(out_dir / f"{file_stem}_links_{method_name}.csv", index=False)
    frame_stats_df.to_csv(out_dir / f"{file_stem}_frame_stats_{method_name}.csv", index=False)
    per_time.to_csv(out_dir / f"{file_stem}_accuracy_over_time_{method_name}.csv", index=False)
    mapping_table.to_csv(out_dir / f"{file_stem}_gt_inferred_mapping_{method_name}.csv", index=False)

    gt_frames = _frames_from_gt_tracks(gt_tracks_df)
    inferred_frames = []
    if not tracks_df.empty:
        for t, grp in tracks_df.groupby("Time"):
            g = grp.sort_values("PointIndex") if "PointIndex" in grp.columns else grp
            cols = ["x", "y", "z"] if {"x", "y", "z"}.issubset(g.columns) else ["Position X", "Position Y", "Position Z"]
            inferred_frames.append(PointCloudFrame(time=float(t), xyz=g[cols].to_numpy(float), meta={"source": "inferred"}))

    plot_3d_frames_comparison(
        frames_gt=gt_frames,
        frames_blind=blind_frames,
        frames_inferred=inferred_frames,
        out_path=out_dir / f"{file_stem}_panel_3d_{method_name}.png",
        gt_tracks_df=gt_tracks_df,
        inferred_tracks_df=tracks_df,
        title=f"{regime} | {file_stem} | {method_name}",
    )
    plot_accuracy_over_time(gt_tracks_df, tracks_df, out_dir / f"{file_stem}_plot_accuracy_over_time_{method_name}.png")
    plot_rms_over_time(gt_tracks_df, tracks_df, out_dir / f"{file_stem}_plot_rms_over_time_{method_name}.png")
    _plot_link_cost_over_time(links_df, out_dir / f"{file_stem}_plot_link_cost_over_time_{method_name}.png")

    row: dict[str, float | str] = {
        "regime": str(regime),
        "file_stem": str(file_stem),
        "method": str(method_name),
        **{k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in extras.items()},
    }
    row.update({k: float(v) if np.isscalar(v) and isinstance(v, (int, float, np.floating)) else v for k, v in summary_gt.items()})
    for k, v in assoc_result.quality.items():
        if isinstance(v, (int, float, np.floating)):
            row[f"assoc_{k}"] = float(v)
    pd.DataFrame([row]).to_csv(out_dir / f"{file_stem}_metrics_summary_{method_name}.csv", index=False)
    return row

# %% DATASET GENERATION
"""Generate or reuse paired GT/blind datasets.

What is tested:
- Creation and reuse of blinded point-cloud datasets with unblinding maps.

Why it matters:
- Association benchmarking requires no-ID input plus GT mapping for evaluation.

Success criteria:
- For each requested regime, blind CSVs and unblinding CSV exist.
"""

if RUN_GENERATE_DATASETS:
    # Generate nodrift datasets over dx scales.
    for dx in DX_SCALES:
        tag = f"nodrift_dx_{_fmt_token(dx)}"
        root = DATA_ROOT / tag
        synthetic_generate_blind_pair(
            synthetic_dir=root,
            seed=BASE_SEED,
            n_files=N_FILES,
            n_tracks=N_TRACKS,
            n_times=N_TIMES,
            regimes=["nodrift_dx_sweep"],
            drift_params={},
            motion_params={
                "stationary_fraction": 0.10,
                "migrating_fraction": 0.85,
                "temporal_smoothness": 0.9,
                "dx_scale_um": float(dx),
                "dx_distribution": "gaussian",
                "dx_track_smoothness": 0.7,
            },
            out_blind_dir=root / "blind",
            out_gt_dir=root / "gt",
            unblinding_path=root / "unblinding.csv",
            overwrite=bool(OVERWRITE_DATASETS),
        )

    # Optional drift strategy dataset scaffold.
    drift_root = DATA_ROOT / "drift_strategies"
    synthetic_generate_blind_pair(
        synthetic_dir=drift_root,
        seed=BASE_SEED,
        n_files=N_FILES,
        n_tracks=N_TRACKS,
        n_times=N_TIMES,
        regimes=DRIFT_REGIMES,
        drift_params={
            "piecewise_regions": 3,
            "piecewise_transition_um": 80.0,
            "local_drift_scale_um": 2.4,
            "local_lengthscale_um": 200.0,
        },
        motion_params={
            "stationary_fraction": 0.45,
            "migrating_fraction": 0.35,
            "temporal_smoothness": 0.9,
        },
        out_blind_dir=drift_root / "blind",
        out_gt_dir=drift_root / "gt",
        unblinding_path=drift_root / "unblinding.csv",
        overwrite=bool(OVERWRITE_DATASETS),
    )

# %% NO-DRIFT DX SWEEP
"""Benchmark association-only behavior as displacement magnitude increases.

What is tested:
- Hungarian + gating, with/without constant-velocity refinement.
- Nearest-neighbor + gating baseline.

Why it matters:
- This isolates module-1 quality before introducing drift correction confounds.

Interpretation:
- Link precision/recall/F1 and purity should decline as dx increases.
- Constant-velocity should help at moderate dx if trajectories are smooth.
"""

if RUN_NODRIFT_DX_SWEEP:
    out_root = _results_root("nodrift_dx_sweep")
    rows: list[dict[str, float | str]] = []

    for dx in DX_SCALES:
        tag = f"nodrift_dx_{_fmt_token(dx)}"
        data_root = DATA_ROOT / tag
        blind_reg_dir = data_root / "blind" / "nodrift_dx_sweep"
        gt_reg_dir = data_root / "gt" / "nodrift_dx_sweep"
        map_path = data_root / "unblinding.csv"
        if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
            print(f"[WARN] Missing dataset for dx={dx}, skip: {data_root}")
            continue

        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == "nodrift_dx_sweep")
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", "nodrift_dx_sweep", file_stem, mapping_all)

            for m in NODRIFT_METHODS:
                gating = {"max_dist": float(max(0.5, 4.0 * float(dx))), "keep_percentile": 99.0}
                assoc = associate_nodrift(
                    blind_frames,
                    method=str(m["method"]),
                    gating=gating,
                    motion_model=str(m["motion_model"]),
                    params={"metric": "sqeuclidean", "motion_refine_iters": 2},
                )
                run_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / str(m["name"])
                row = _evaluate_and_save_run(
                    out_dir=run_dir,
                    assoc_result=assoc,
                    gt_tracks_df=gt_tracks,
                    mapping_df=map_sub,
                    blind_frames=blind_frames,
                    regime="nodrift_dx_sweep",
                    file_stem=file_stem,
                    method_name=str(m["name"]),
                    extras={"dx_scale": float(dx)},
                )
                rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_csv = out_root / "aggregated_summary.csv"
        agg.to_csv(agg_csv, index=False)

        # Compact summary tables.
        metric_cols = [
            c
            for c in [
                "link_precision_mean",
                "link_recall_mean",
                "link_f1_mean",
                "track_purity",
                "coverage",
                "fragmentation_rate",
                "mean_fragments_per_gt",
                "id_switches_total",
                "id_switches_mean_per_track",
                "rms_mean",
                "rms_std",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(id_vars=["dx_scale", "method", "regime"], value_vars=metric_cols, var_name="metric", value_name="value")
            summary = (
                long.groupby(["dx_scale", "method", "metric"], as_index=False)["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_scale"])
                .reset_index(drop=True)
            )
            summary.to_csv(out_root / "aggregated_summary_stats.csv", index=False)

        # Publication-style aggregate plots:
        generated: list[str] = []
        plot_metrics = [
            "link_precision_mean",
            "link_recall_mean",
            "link_f1_mean",
            "fragmentation_rate",
            "id_switches_mean_per_track",
            "rms_mean",
        ]
        for metric in plot_metrics:
            if metric not in agg.columns:
                continue
            p1 = out_root / f"plot_dx_sweep_{metric}_lines.png"
            if _plot_metric_vs_dx(agg, metric=metric, out_path=p1, title=f"No-drift dx sweep: {metric}"):
                generated.append(p1.name)
            p2 = out_root / f"plot_dx_sweep_{metric}_heatmap.png"
            if _plot_metric_heatmap_dx_method(agg, metric=metric, out_path=p2, title=f"No-drift dx sweep: {metric} (method x dx)"):
                generated.append(p2.name)

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dx_scales: {DX_SCALES}",
                    f"methods: {[m['name'] for m in NODRIFT_METHODS]}",
                    f"aggregated_csv: {agg_csv}",
                    "description: No-drift association benchmark with blinded point clouds.",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% DRIFT-PRESENT STRATEGY COMPARISON (OPTIONAL)
"""Compare strategy 1 vs strategy 2 under drift regimes.

What is tested:
- Strategy 1: registration then assignment (ICP rigid + Hungarian)
- Strategy 2: EM loop with rigid drift-fit adapter

Why it matters:
- Demonstrates integration hooks between module 1 association and module 2
  drift correction while keeping module-1 independently usable.

Interpretation:
- Better link precision/recall/F1 and purity under drift indicates better coupling strategy.
"""

if RUN_DRIFT_STRATEGY_COMPARISON:
    out_root = _results_root("drift_strategy_comparison")
    data_root = DATA_ROOT / "drift_strategies"
    map_path = data_root / "unblinding.csv"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing mapping CSV: {map_path}")
    mapping_all = pd.read_csv(map_path)
    rows: list[dict[str, float | str]] = []

    for regime in DRIFT_REGIMES:
        blind_reg_dir = data_root / "blind" / regime
        if not blind_reg_dir.exists():
            continue
        for blind_file in sorted(blind_reg_dir.glob("*_blind.csv")):
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == str(regime))
                & (mapping_all["File"].astype(str) == str(file_stem))
            ].copy()
            if map_sub.empty:
                continue

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", regime, file_stem, mapping_all)

            # Strategy 1: registration + assignment.
            assoc_reg = associate_with_registration(
                blind_frames,
                registration="icp_rigid",
                then_assignment="hungarian",
                params={
                    "metric": "sqeuclidean",
                    "gating": {"max_dist": 16.0, "keep_percentile": 99.0},
                    "icp_n_iters": 20,
                    "icp_keep_fraction": 0.8,
                    "icp_max_corr_dist": 25.0,
                },
            )
            row_reg = _evaluate_and_save_run(
                out_dir=out_root / regime / file_stem / "strategy_registration",
                assoc_result=assoc_reg,
                gt_tracks_df=gt_tracks,
                mapping_df=map_sub,
                blind_frames=blind_frames,
                regime=regime,
                file_stem=file_stem,
                method_name="strategy_registration",
                extras={"strategy": "registration_then_assignment"},
            )
            rows.append(row_reg)

            # Strategy 2: EM with rigid drift fitter.
            assoc_em, _ = em_associate_and_driftfit(
                blind_frames,
                drift_fit_adapter=DriftModelAdapterRigidStep(),
                n_iters=3,
                params={
                    "association_method": "hungarian",
                    "motion_model": "constant_position",
                    "metric": "sqeuclidean",
                    "gating": {"max_dist": 16.0, "keep_percentile": 99.0},
                    "assignment_change_tol": 1e-3,
                    "drift_options": {"keep_fraction": 0.7, "min_pairs": 8, "inlier_reestimate_iters": 10},
                },
            )
            row_em = _evaluate_and_save_run(
                out_dir=out_root / regime / file_stem / "strategy_em_rigid",
                assoc_result=assoc_em,
                gt_tracks_df=gt_tracks,
                mapping_df=map_sub,
                blind_frames=blind_frames,
                regime=regime,
                file_stem=file_stem,
                method_name="strategy_em_rigid",
                extras={"strategy": "em_with_rigid_adapter"},
            )
            rows.append(row_em)

    if rows:
        agg = pd.DataFrame(rows)
        agg.to_csv(out_root / "aggregated_summary.csv", index=False)
        generated: list[str] = []
        plot_metrics = [
            "link_precision_mean",
            "link_recall_mean",
            "link_f1_mean",
            "fragmentation_rate",
            "id_switches_mean_per_track",
            "rms_mean",
        ]
        for metric in plot_metrics:
            if metric not in agg.columns:
                continue
            p1 = out_root / f"plot_strategy_{metric}_bars.png"
            if _plot_metric_by_regime_strategy(agg, metric=metric, out_path=p1, title=f"Drift strategy comparison: {metric}"):
                generated.append(p1.name)
            p2 = out_root / f"plot_strategy_{metric}_heatmap.png"
            if _plot_metric_heatmap_regime_strategy(
                agg,
                metric=metric,
                out_path=p2,
                title=f"Drift strategy comparison: {metric} (strategy x regime)",
            ):
                generated.append(p2.name)

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"regimes: {DRIFT_REGIMES}",
                    "strategies: registration_then_assignment, em_with_rigid_adapter",
                    "description: Optional drift-present strategy comparison scaffold.",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )


# %% END
"""End of association benchmark script.

How to use:
- set `RUN_GENERATE_DATASETS=True` once,
- then run `RUN_NODRIFT_DX_SWEEP=True` for module-1 core results,
- optionally run drift strategy comparisons.
"""
