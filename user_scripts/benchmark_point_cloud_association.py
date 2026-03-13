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
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from driftcorr.association import (
        DriftModelAdapterRigidStep,
        associate_hungarian_adaptive_maxdist_cv,
        associate_hungarian_innovation_percentile_cv,
        associate_hungarian_nis_gating_cv,
        associate_hungarian_switch_penalty_nis_cv,
        associate_hungarian_switch_penalty_cv,
        associate_hungarian_two_stage_nis_cv,
        associate_hungarian_two_stage_cv,
        associate_kalman_cv,
        associate_min_cost_flow,
        associate_nodrift,
        associate_tracklet_stitching,
        associate_with_registration,
        compute_displacement_from_gt,
        compute_nn_distance,
        compute_rho,
        em_associate_and_driftfit,
        load_blind_frames,
        synthetic_generate_blind_pair,
        synthetic_generate_blind_pair_varying_dx_birthdeath_drift,
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
            associate_hungarian_adaptive_maxdist_cv,
            associate_hungarian_innovation_percentile_cv,
            associate_hungarian_nis_gating_cv,
            associate_hungarian_switch_penalty_nis_cv,
            associate_hungarian_switch_penalty_cv,
            associate_hungarian_two_stage_nis_cv,
            associate_hungarian_two_stage_cv,
            associate_kalman_cv,
            associate_min_cost_flow,
            associate_nodrift,
            associate_tracklet_stitching,
            associate_with_registration,
            compute_displacement_from_gt,
            compute_nn_distance,
            compute_rho,
            em_associate_and_driftfit,
            load_blind_frames,
            synthetic_generate_blind_pair,
            synthetic_generate_blind_pair_varying_dx_birthdeath_drift,
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

CM = 1/2.54  # centimeters in inches

ASSOC_ROOT = PROJECT_ROOT / "Synthetic files track inference parameter sweep" 
DATA_ROOT = ASSOC_ROOT / "_blind_pairs"
DATA_ROOT_VARYING_DX = ASSOC_ROOT / "_blind_pairs_varying_dx"
DATA_ROOT_VARYING_DX_BIRTHDEATH = ASSOC_ROOT / "_blind_pairs_varying_dx_birthdeath"
DATA_ROOT_VARYING_DX_BIRTHDEATH_DRIFT = ASSOC_ROOT / "_blind_pairs_varying_dx_birthdeath_drift"
DATA_ROOT_VARYING_DX_STATIONARY = ASSOC_ROOT / "_blind_pairs_varying_dx_stationary"
DATA_ROOT_VARYING_DX_DENSITY = ASSOC_ROOT / "_blind_pairs_varying_dx_density"
NODRIFT_VARYING_DX_REGIME = "nodrift_varying_dx"
NODRIFT_VARYING_DX_TAG = "varying_dx_0p20_4p00"
NODRIFT_VARYING_DX_NOMINAL_SCALE = 1.0
NODRIFT_VARYING_DX_BIRTHDEATH_REGIME = "nodrift_dx_varying_birthdeath"
NODRIFT_VARYING_DX_BIRTHDEATH_TAG = "varying_dx_birthdeath_0p20_4p00"
NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE = 1.0
NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME = "varying_dx_birthdeath_with_drift"
NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_TAG = "varying_dx_birthdeath_with_drift"
NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE = 1.0
NODRIFT_VARYING_DX_STATIONARY_REGIME = "varying_dx_birthdeath_stationary"
NODRIFT_VARYING_DX_STATIONARY_TAG = "varying_dx_birthdeath_stationary"
NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE = 1.0
NODRIFT_VARYING_DX_STATIONARY_STATIONARY_FRACTION = 0.25
NODRIFT_VARYING_DX_DENSITY_REGIME = "varying_dx_birthdeath_stationary_density"
NODRIFT_VARYING_DX_DENSITY_TAG = "varying_dx_birthdeath_stationary_density"
DENSITY_INITIAL_TRACKS = 80
DENSITY_SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0]
VOLUME_SIZE = (700.0, 700.0, 350.0)
DRIFT_PARAMS_BIRTHDEATH = {
    "global_drift_scale_xyz": (2.8, 2.2, 1.1),
    "local_drift_scale_um": 0.0,
    "local_lengthscale_um": 220.0,
    "piecewise_regions": 3,
    "piecewise_transition_um": 90.0,
    "drift_rotation_enable": True,
    "drift_rotation_deg_scale": 2.0,
}
MOTION_MIX_PARAMS = {
    "stationary_fraction": 0.30,
    "migrating_fraction": 0.50,
    "locally_coherent_fraction": 0.20,
}

RUN_GENERATE_DATASETS = True
RUN_NODRIFT_DX_SWEEP = True
RUN_NODRIFT_PARAM_SWEEP = True
RUN_HUNGARIAN_CV_VARIANTS_SWEEP = True
RUN_HCV_NIS_GATING_COMPARISON = True
RUN_HCV_NIS_DX_BLIND_SWEEP = True
RUN_HCV_NIS_DX_BLIND_SWEEP_VARYING_DX = True
RUN_HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH = True
RUN_HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH_DRIFT = True
RUN_HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH_STATIONARY = True
RUN_HCV_NIS_DX_BLIND_SWEEP_DENSITY = True
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
    {
        "name": "hungarian_dummy_unmatched_cp",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_position",
        "params": {"use_dummy": True},
    },
    {
        "name": "min_cost_flow_cp",
        "kind": "min_cost_flow",
        "motion_model": "constant_position",
    },
    {
        "name": "min_cost_flow_cv",
        "kind": "min_cost_flow",
        "motion_model": "constant_velocity",
        "params": {"allow_skip": True, "skip_cost_factor": 1.2},
    },
    {
        "name": "tracklet_stitching_cv",
        "kind": "tracklet_stitching",
        "motion_model": "constant_velocity",
    },
    {
        "name": "kalman_cv_innovation_gating",
        "kind": "kalman",
        "motion_model": "constant_velocity",
    },
    {
        "name": "hungarian_huber_dummy_cp",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_position",
        "params": {"metric": "huber", "use_dummy": True, "huber_delta": 2.0},
    },
    {
        "name": "hungarian_mahalanobis_diag_cp",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_position",
        "params": {"metric": "mahalanobis_diag", "use_dummy": True, "mahalanobis_scales": [2.0, 2.0, 2.0]},
    },
]

DRIFT_REGIMES = ["rotation_drift", "smooth"]
NODRIFT_PARAM_SWEEP_DX = [0.2, 0.5, 1.0, 2.0, 4.0]
NODRIFT_PARAM_SWEEP_MAX_FILES = 1
NODRIFT_PARAM_SWEEP_MAX_DIST_FACTORS = [2.0, 3.0, 4.0, 6.0]
NODRIFT_PARAM_SWEEP_KEEP_PERCENTILES = [90, 95, 99]
NODRIFT_PARAM_SWEEP_REFINE_ITERS = [0, 1, 2, 5]

HCV_KEEP_PERCENTILE = 99
HCV_MAX_DIST_FACTORS = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
HCV_REFINE_ITERS = [0, 1, 2, 5]
HCV_DX_SCALES = DX_SCALES
HCV_MAX_FILES = 1
HUNGARIAN_CV_VARIANTS = [
    {
        "name": "hungarian_gating_cv",
        "kind": "nodrift",
        "method": "hungarian",
        "motion_model": "constant_velocity",
        "params": {"metric": "sqeuclidean"},
    },
    {
        "name": "hungarian_adaptive_maxdist_cv",
        "kind": "hungarian_cv_variant",
        "variant": "adaptive_maxdist",
        "motion_model": "constant_velocity",
        "params": {"sigma_mad_clip": 3.0, "metric": "sqeuclidean", "use_dummy": True},
    },
    {
        "name": "hungarian_innovation_percentile_cv",
        "kind": "hungarian_cv_variant",
        "variant": "innovation_percentile",
        "motion_model": "constant_velocity",
        "params": {"sigma_mad_clip": 3.0, "use_dummy": True},
    },
    {
        "name": "hungarian_switch_penalty_cv",
        "kind": "hungarian_cv_variant",
        "variant": "switch_penalty",
        "motion_model": "constant_velocity",
        "params": {"lambda_switch": 0.2, "use_normalized": True, "metric": "sqeuclidean", "use_dummy": True},
    },
    {
        "name": "hungarian_two_stage_cv",
        "kind": "hungarian_cv_variant",
        "variant": "two_stage",
        "motion_model": "constant_velocity",
        "params": {"stage1_factor": 0.5, "stage1_keep_percentile": 90.0, "stage2_keep_percentile": 99.0, "stage2_allow_unmatched": True, "metric": "sqeuclidean", "use_dummy": True},
    },
]
HCV_NIS_KEEP_PERCENTILE = 99.0
HCV_NIS_FIXED_MAX_DIST_FACTOR = 6.0
HCV_NIS_DX_BLIND_NN_SCALE_FACTORS = [0.2, 0.3, 0.4]
HCV_MAX_FILES_VARYING_DX = 1 # defines how many files to generate and run per dx_scale in the varying dx sweep
HCV_MAX_FILES_BIRTHDEATH = 1
HCV_MAX_FILES_BIRTHDEATH_DRIFT = 1
HCV_MAX_FILES_BIRTHDEATH_STATIONARY = 1
HCV_MAX_FILES_DENSITY = 1
HCV_NIS_METHODS = [
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
        "params": {"stage1_factor": 0.5, "stage1_keep_percentile": 90.0, "stage2_keep_percentile": 99.0, "stage2_allow_unmatched": True, "metric": "sqeuclidean", "use_dummy": True},
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
        "params": {"gating_mode": "nis", "nis_alpha": 0.997, "stage1_factor": 0.5, "stage1_keep_percentile": 90.0, "stage2_keep_percentile": 99.0, "stage2_allow_unmatched": True, "use_dummy": True},
    },
    {
        "name": "hungarian_switch_penalty_nis_cv",
        "kind": "hungarian_cv_variant",
        "variant": "switch_penalty_nis",
        "motion_model": "constant_velocity",
        "params": {"gating_mode": "nis", "nis_alpha": 0.997, "lambda_switch": 0.2, "use_normalized": True, "use_dummy": True},
    },
]
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

def _plot_simple_series(tab: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str, out_path: Path) -> bool:
    if tab.empty or x_col not in tab.columns or y_col not in tab.columns:
        return False
    d = tab.copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(d[x_col].to_numpy(float), d[y_col].to_numpy(float), marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _compute_and_save_diagnostics(
    out_dir: Path,
    blind_frames: list[PointCloudFrame],
    gt_tracks_df: pd.DataFrame,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nn_df = compute_nn_distance(blind_frames)
    disp_df = compute_displacement_from_gt(gt_tracks_df)
    rho_df = compute_rho(nn_df, disp_df)
    nn_df.to_csv(out_dir / "diagnostics_nn_distance_over_time.csv", index=False)
    disp_df.to_csv(out_dir / "diagnostics_displacement_over_time.csv", index=False)
    rho_df.to_csv(out_dir / "diagnostics_rho_over_time.csv", index=False)

    _plot_simple_series(
        nn_df,
        x_col="Time",
        y_col="nn_median",
        title="Nearest-neighbor distance over time",
        y_label="median d_NN (um)",
        out_path=out_dir / "plot_nn_distance_over_time.png",
    )
    _plot_simple_series(
        disp_df,
        x_col="Time",
        y_col="disp_median",
        title="GT displacement over time",
        y_label="median Δx (um)",
        out_path=out_dir / "plot_displacement_over_time.png",
    )
    _plot_simple_series(
        rho_df,
        x_col="Time",
        y_col="rho_median",
        title="Ambiguity ratio over time",
        y_label="ρ = Δx / d_NN",
        out_path=out_dir / "plot_rho_over_time.png",
    )

    return {
        "diag_nn_median": float(np.nanmedian(pd.to_numeric(nn_df.get("nn_median", np.nan), errors="coerce"))),
        "diag_disp_median": float(np.nanmedian(pd.to_numeric(disp_df.get("disp_median", np.nan), errors="coerce"))),
        "diag_rho_median": float(np.nanmedian(pd.to_numeric(rho_df.get("rho_median", np.nan), errors="coerce"))),
    }

def _dispatch_association_method(
    blind_frames: list[PointCloudFrame],
    method_cfg: dict[str, object],
    dx_scale: float,
    gating_override: dict[str, object] | None = None,
    params_override: dict[str, object] | None = None,
):
    """Run one association method config through the correct API."""
    kind = str(method_cfg.get("kind", "nodrift"))
    base_gating = {
        "max_dist": float(max(0.5, 4.0 * float(dx_scale))),
        "keep_percentile": 99.0,
        "huber_delta": float(method_cfg.get("params", {}).get("huber_delta", 1.0)) if isinstance(method_cfg.get("params"), dict) else 1.0,
    }
    if gating_override:
        base_gating.update(dict(gating_override))
    p_base = {"metric": "sqeuclidean", "motion_refine_iters": 2}
    if isinstance(method_cfg.get("params"), dict):
        p_base.update(dict(method_cfg.get("params", {})))
    if params_override:
        p_base.update(dict(params_override))
    if "dx_scale" not in p_base:
        p_base["dx_scale"] = float(dx_scale)
    if "max_dist_factor" not in p_base:
        base_md = float(base_gating.get("max_dist", max(0.5, 4.0 * float(dx_scale))))
        p_base["max_dist_factor"] = float(base_md / max(float(dx_scale), 1e-9))

    if kind == "min_cost_flow":
        return associate_min_cost_flow(
            blind_frames,
            gating=base_gating,
            motion_model=str(method_cfg.get("motion_model", "constant_position")),
            params=p_base,
        )
    if kind == "tracklet_stitching":
        strict = {
            "max_dist": float(base_gating.get("max_dist", 4.0)),
            "keep_percentile": float(p_base.get("keep_percentile_strict", 90.0)),
            "huber_delta": float(p_base.get("huber_delta", 1.0)),
        }
        loose = {
            "max_dist": float(base_gating.get("max_dist", 4.0)),
            "keep_percentile": float(base_gating.get("keep_percentile", 99.0)),
        }
        return associate_tracklet_stitching(
            blind_frames,
            gating_strict=strict,
            gating_loose=loose,
            motion_model=str(method_cfg.get("motion_model", "constant_velocity")),
            params=p_base,
        )
    if kind == "kalman":
        return associate_kalman_cv(
            blind_frames,
            gating=base_gating,
            params=p_base,
        )
    if kind == "hungarian_cv_variant":
        variant_name = str(method_cfg.get("variant", ""))
        if variant_name == "adaptive_maxdist":
            return associate_hungarian_adaptive_maxdist_cv(blind_frames, gating=base_gating, params=p_base)
        if variant_name == "innovation_percentile":
            return associate_hungarian_innovation_percentile_cv(blind_frames, gating=base_gating, params=p_base)
        if variant_name == "nis_gating":
            return associate_hungarian_nis_gating_cv(blind_frames, gating=base_gating, params=p_base)
        if variant_name == "switch_penalty":
            return associate_hungarian_switch_penalty_cv(blind_frames, gating=base_gating, params=p_base)
        if variant_name == "switch_penalty_nis":
            return associate_hungarian_switch_penalty_nis_cv(blind_frames, gating=base_gating, params=p_base)
        if variant_name == "two_stage":
            return associate_hungarian_two_stage_cv(blind_frames, gating=base_gating, params=p_base)
        if variant_name == "two_stage_nis":
            return associate_hungarian_two_stage_nis_cv(blind_frames, gating=base_gating, params=p_base)
        raise ValueError(f"Unknown hungarian_cv_variant '{variant_name}'")
    # default: pairwise nodrift
    return associate_nodrift(
        blind_frames,
        method=str(method_cfg.get("method", "hungarian")),
        gating=base_gating,
        motion_model=str(method_cfg.get("motion_model", "constant_position")),
        params=p_base,
    )

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

def _plot_metric_heatmap_dx_method(agg: pd.DataFrame, 
                                   metric: str, 
                                   out_path: Path, 
                                   title: str,
                                   figsize: tuple = (7.2, 4.8),
                                   clims: tuple = None,
                                   title_on: bool = True) -> bool:
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
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    if clims is not None:
        im.set_clim(*clims)
    #cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label(metric)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.02)
    # remove frame around colorbar:
    cbar.outline.set_visible(False)
    # replace all "_" with " ":
    cbar.set_label(metric.replace("_", " "), labelpad=1)
    cbar.ax.tick_params(pad=1)
    
    
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("dx scale ($\mu$m)")
    #ax.set_ylabel("method")
    if title_on:
        ax.set_title(title.replace("_", " "), fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
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

def _plot_param_heatmap(
    agg: pd.DataFrame,
    metric: str,
    method_name: str,
    dx_scale: float,
    motion_refine_iters: int,
    out_path: Path,
) -> bool:
    req = {"method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", metric}
    if agg.empty or not req.issubset(set(agg.columns)):
        return False
    d = agg.copy()
    d = d[d["method"].astype(str) == str(method_name)]
    d["dx_scale"] = pd.to_numeric(d["dx_scale"], errors="coerce")
    d = d[np.isclose(d["dx_scale"].to_numpy(float), float(dx_scale))]
    d["motion_refine_iters"] = pd.to_numeric(d["motion_refine_iters"], errors="coerce")
    d = d[d["motion_refine_iters"] == float(motion_refine_iters)]
    if d.empty:
        return False
    d["max_dist_factor"] = pd.to_numeric(d["max_dist_factor"], errors="coerce")
    d["keep_percentile"] = pd.to_numeric(d["keep_percentile"], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["max_dist_factor", "keep_percentile", metric])
    if d.empty:
        return False

    x_vals = sorted(d["keep_percentile"].unique().tolist())
    y_vals = sorted(d["max_dist_factor"].unique().tolist())
    piv = d.pivot_table(index="max_dist_factor", columns="keep_percentile", values=metric, aggfunc="mean")
    piv = piv.reindex(index=y_vals, columns=x_vals)
    mat = piv.to_numpy(float)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([str(int(v)) for v in x_vals])
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{float(v):g}" for v in y_vals])
    ax.set_xlabel("keep_percentile")
    ax.set_ylabel("max_dist_factor")
    ax.set_title(f"{method_name} | dx={dx_scale:g} | iters={int(motion_refine_iters)} | {metric}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _metric_higher_is_better(metric: str) -> bool:
    return str(metric) in {"link_f1_mean"}

def _fmt_method_token(method: str) -> str:
    return str(method).strip().replace(" ", "_").replace("-", "_")

def _plot_heatmap_from_pivot(
    pivot: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap_name: str = "viridis",
) -> bool:
    if pivot.empty:
        return False
    mat = pivot.to_numpy(float)
    if mat.size == 0:
        return False
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f0f0f0")
    fig_w = max(7.0, 1.2 * max(1, pivot.shape[1]))
    fig_h = max(4.8, 0.45 * max(3, pivot.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(v) for v in pivot.columns.tolist()])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(v) for v in pivot.index.tolist()])
    ax.set_xlabel("dx_scale")
    ax.set_ylabel("method")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _prepare_param_sweep_config_table(agg: pd.DataFrame) -> pd.DataFrame:
    req = ["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters"]
    if agg.empty or not set(req).issubset(set(agg.columns)):
        return pd.DataFrame()
    d = agg.copy()
    for c in req:
        if c in {"method"}:
            d[c] = d[c].astype(str)
        else:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "config_id" not in d.columns:
        d["config_id"] = d.apply(
            lambda r: f"md{_fmt_token(float(r['max_dist_factor']))}_kp{int(r['keep_percentile'])}_it{int(r['motion_refine_iters'])}",
            axis=1,
        )
    metric_cols = [
        c
        for c in [
            "link_f1_mean",
            "rms_mean",
            "id_switches_mean_per_track",
            "fragmentation_rate",
            "unmatched_rate_mean",
            "link_precision_mean",
            "link_recall_mean",
        ]
        if c in d.columns
    ]
    for c in metric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    opt_param_cols = [c for c in ["nn_scale_factor", "sigma_init_mode"] if c in d.columns]
    if "nn_scale_factor" in opt_param_cols:
        d["nn_scale_factor"] = pd.to_numeric(d["nn_scale_factor"], errors="coerce")
    if "sigma_init_mode" in opt_param_cols:
        d["sigma_init_mode"] = d["sigma_init_mode"].astype(str)
    gcols = ["method", "dx_scale", "config_id", "max_dist_factor", "keep_percentile", "motion_refine_iters", *opt_param_cols]
    cfg = d.groupby(gcols, as_index=False)[metric_cols].mean(numeric_only=True)
    cfg = cfg.sort_values(["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", *opt_param_cols]).reset_index(drop=True)
    return cfg

def _best_config_by_metric(cfg: pd.DataFrame, metric: str) -> pd.DataFrame:
    if cfg.empty or metric not in cfg.columns:
        return pd.DataFrame()
    d = cfg.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric, "method", "dx_scale"])
    if d.empty:
        return pd.DataFrame()
    higher = _metric_higher_is_better(metric)
    d = d.sort_values(
        ["method", "dx_scale", metric, "max_dist_factor", "keep_percentile", "motion_refine_iters"],
        ascending=[True, True, not higher, True, True, True],
    )
    best = d.groupby(["method", "dx_scale"], as_index=False).first()
    return best

def _plot_param_sweep_overview(out_root: Path) -> list[str]:
    generated: list[str] = []
    agg_path = out_root / "aggregated_param_sweep.csv"
    stats_path = out_root / "aggregated_param_sweep_stats.csv"
    if not agg_path.exists():
        return generated

    agg = pd.read_csv(agg_path)
    _ = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    cfg = _prepare_param_sweep_config_table(agg)
    if cfg.empty:
        return generated

    core_metrics = [
        m
        for m in [
            "link_f1_mean",
            "rms_mean",
            "id_switches_mean_per_track",
            "fragmentation_rate",
            "unmatched_rate_mean",
        ]
        if m in cfg.columns
    ]

    # Plot 1 + Plot 2: best-by-metric value and selected params.
    for metric in core_metrics:
        best = _best_config_by_metric(cfg, metric)
        if best.empty:
            continue
        piv_val = best.pivot_table(index="method", columns="dx_scale", values=metric, aggfunc="mean")
        piv_val = piv_val.reindex(index=sorted(piv_val.index.astype(str).tolist()), columns=sorted([float(v) for v in piv_val.columns.tolist()]))
        p_val = out_root / f"best_value__{metric}.png"
        if _plot_heatmap_from_pivot(
            piv_val,
            title=f"Best value by method/dx: {metric}",
            cbar_label=metric,
            out_path=p_val,
        ):
            generated.append(p_val.name)

        for param_name in ["max_dist_factor", "keep_percentile", "motion_refine_iters"]:
            if param_name not in best.columns:
                continue
            piv_p = best.pivot_table(index="method", columns="dx_scale", values=param_name, aggfunc="mean")
            piv_p = piv_p.reindex(index=sorted(piv_p.index.astype(str).tolist()), columns=sorted([float(v) for v in piv_p.columns.tolist()]))
            p_param = out_root / f"best_param__{metric}__{param_name}.png"
            if _plot_heatmap_from_pivot(
                piv_p,
                title=f"Selected {param_name} for best {metric}",
                cbar_label=param_name,
                out_path=p_param,
            ):
                generated.append(p_param.name)

    # Plot 3: sensitivity curves per method/metric.
    for method in sorted(cfg["method"].astype(str).unique().tolist()):
        sub = cfg[cfg["method"].astype(str) == str(method)].copy()
        if sub.empty:
            continue
        for metric in core_metrics:
            if metric not in sub.columns:
                continue
            fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.6))
            panels = [
                ("max_dist_factor", "max_dist_factor"),
                ("keep_percentile", "keep_percentile"),
                ("motion_refine_iters", "motion_refine_iters"),
            ]
            for ax, (col, xlab) in zip(axs, panels):
                d = sub.copy()
                d[col] = pd.to_numeric(d[col], errors="coerce")
                d[metric] = pd.to_numeric(d[metric], errors="coerce")
                d = d.dropna(subset=[col, metric])
                if d.empty:
                    ax.set_title(xlab)
                    continue
                stat = d.groupby(col, as_index=False)[metric].mean().sort_values(col)
                ax.plot(stat[col].to_numpy(float), stat[metric].to_numpy(float), marker="o")
                ax.set_xlabel(xlab)
                ax.set_ylabel(metric)
                if metric in {"link_f1_mean"}:
                    ax.set_ylim(0.0, 1.0)
                else:
                    ax.set_ylim(bottom=0.0)
                ax.set_title(xlab)
            fig.suptitle(f"Sensitivity | {method} | {metric}")
            fig.tight_layout()
            p = out_root / f"sensitivity__{_fmt_method_token(method)}__{metric}.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            generated.append(p.name)

    # Plot 4: Pareto (best by F1 per method/dx).
    if "link_f1_mean" in cfg.columns and "rms_mean" in cfg.columns:
        for dx in sorted(pd.to_numeric(cfg["dx_scale"], errors="coerce").dropna().unique().tolist()):
            sub = cfg[pd.to_numeric(cfg["dx_scale"], errors="coerce") == float(dx)].copy()
            if sub.empty:
                continue
            sub["link_f1_mean"] = pd.to_numeric(sub["link_f1_mean"], errors="coerce")
            sub["rms_mean"] = pd.to_numeric(sub["rms_mean"], errors="coerce")
            sub = sub.dropna(subset=["method", "link_f1_mean", "rms_mean"])
            if sub.empty:
                continue
            sub = sub.sort_values(["method", "link_f1_mean", "rms_mean"], ascending=[True, False, True])
            best_f1 = sub.groupby("method", as_index=False).first()
            fig, ax = plt.subplots(figsize=(7.0, 5.2))
            ax.scatter(best_f1["rms_mean"].to_numpy(float), best_f1["link_f1_mean"].to_numpy(float), s=50.0)
            for _, r in best_f1.iterrows():
                ax.annotate(str(r["method"]), (float(r["rms_mean"]), float(r["link_f1_mean"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_xlabel("rms_mean")
            ax.set_ylabel("link_f1_mean")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Pareto (best by F1) | dx={float(dx):g}")
            fig.tight_layout()
            p = out_root / f"pareto_bestbyF1__dx{_fmt_token(float(dx))}.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            generated.append(p.name)

    # Plot 5: top-5 table by F1 for each dx.
    if "link_f1_mean" in cfg.columns:
        show_cols = [
            c
            for c in [
                "method",
                "max_dist_factor",
                "keep_percentile",
                "motion_refine_iters",
                "link_f1_mean",
                "rms_mean",
                "id_switches_mean_per_track",
                "fragmentation_rate",
            ]
            if c in cfg.columns
        ]
        for dx in sorted(pd.to_numeric(cfg["dx_scale"], errors="coerce").dropna().unique().tolist()):
            sub = cfg[pd.to_numeric(cfg["dx_scale"], errors="coerce") == float(dx)].copy()
            if sub.empty:
                continue
            sub["link_f1_mean"] = pd.to_numeric(sub["link_f1_mean"], errors="coerce")
            sub["rms_mean"] = pd.to_numeric(sub.get("rms_mean", np.nan), errors="coerce")
            sub = sub.dropna(subset=["link_f1_mean"])
            if sub.empty:
                continue
            top = sub.sort_values(["link_f1_mean", "rms_mean"], ascending=[False, True]).head(5).copy()
            top = top[show_cols]
            for c in top.columns:
                if c != "method":
                    top[c] = pd.to_numeric(top[c], errors="coerce")
            fig, ax = plt.subplots(figsize=(13.5, 2.8))
            ax.axis("off")
            cell_text = []
            for _, r in top.iterrows():
                row = []
                for c in show_cols:
                    if c == "method":
                        row.append(str(r[c]))
                    else:
                        v = r[c]
                        if np.isfinite(v):
                            row.append(f"{float(v):.4g}")
                        else:
                            row.append("nan")
                cell_text.append(row)
            table = ax.table(cellText=cell_text, colLabels=show_cols, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.2)
            ax.set_title(f"Top-5 configs by link_f1_mean | dx={float(dx):g}", fontsize=10)
            fig.tight_layout()
            p = out_root / f"top5_byF1__dx{_fmt_token(float(dx))}.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            generated.append(p.name)

    return generated


def _plot_hcv_variants_overview(out_root: Path, agg: pd.DataFrame) -> list[str]:
    """Create compact overview plots for Hungarian-CV variants sweep."""
    generated: list[str] = []
    if agg.empty:
        return generated
    cfg = _prepare_param_sweep_config_table(agg)
    if cfg.empty:
        return generated

    metrics = [
        m
        for m in [
            "link_f1_mean",
            "rms_mean",
            "id_switches_mean_per_track",
            "fragmentation_rate",
            "unmatched_rate_mean",
        ]
        if m in cfg.columns
    ]

    baseline_method = "hungarian_gating_cv"
    best_variant_method: str | None = None
    if "link_f1_mean" in cfg.columns:
        best_by_dx_all = _best_config_by_metric(cfg, "link_f1_mean")
        if not best_by_dx_all.empty:
            perf = best_by_dx_all.groupby("method", as_index=False)["link_f1_mean"].mean().sort_values("link_f1_mean", ascending=False)
            for m in perf["method"].astype(str).tolist():
                if m != baseline_method:
                    best_variant_method = m
                    break

    # Group 1: value heatmaps + selected parameters using configs chosen by best link_f1_mean.
    # This keeps parameter selection consistent across all reported metrics.
    best_by_f1 = _best_config_by_metric(cfg, "link_f1_mean") if "link_f1_mean" in cfg.columns else pd.DataFrame()
    for metric in metrics:
        best = best_by_f1.copy()
        if best.empty:
            # Fallback when F1 is unavailable: keep legacy per-metric behavior.
            best = _best_config_by_metric(cfg, metric)
        if best.empty or metric not in best.columns:
            continue
        piv_val = best.pivot_table(index="method", columns="dx_scale", values=metric, aggfunc="mean")
        piv_val = piv_val.reindex(index=sorted([str(v) for v in piv_val.index.tolist()]), columns=sorted([float(v) for v in piv_val.columns.tolist()]))
        p_val = out_root / f"best_value__{metric}.png"
        if _plot_heatmap_from_pivot(
            piv_val,
            title=f"HCV variants: {metric} at config selected by best link_f1_mean",
            cbar_label=metric,
            out_path=p_val,
        ):
            generated.append(p_val.name)

        for pname in ["max_dist_factor", "motion_refine_iters"]:
            if pname not in best.columns:
                continue
            piv_p = best.pivot_table(index="method", columns="dx_scale", values=pname, aggfunc="mean")
            piv_p = piv_p.reindex(index=sorted([str(v) for v in piv_p.index.tolist()]), columns=sorted([float(v) for v in piv_p.columns.tolist()]))
            p = out_root / f"best_param__{metric}__{pname}.png"
            if _plot_heatmap_from_pivot(
                piv_p,
                title=f"HCV variants: selected {pname} (config chosen by best link_f1_mean)",
                cbar_label=pname,
                out_path=p,
            ):
                generated.append(p.name)

    # Group 2: Pareto per dx using best-by-F1 configs.
    if "link_f1_mean" in cfg.columns and "rms_mean" in cfg.columns:
        for dx in sorted(pd.to_numeric(cfg["dx_scale"], errors="coerce").dropna().unique().tolist()):
            sub = cfg[pd.to_numeric(cfg["dx_scale"], errors="coerce") == float(dx)].copy()
            if sub.empty:
                continue
            sub["link_f1_mean"] = pd.to_numeric(sub["link_f1_mean"], errors="coerce")
            sub["rms_mean"] = pd.to_numeric(sub["rms_mean"], errors="coerce")
            sub = sub.dropna(subset=["method", "link_f1_mean", "rms_mean"])
            if sub.empty:
                continue
            best_f1 = sub.sort_values(["method", "link_f1_mean", "rms_mean"], ascending=[True, False, True]).groupby("method", as_index=False).first()
            fig, ax = plt.subplots(figsize=(7.0, 5.2))
            ax.scatter(best_f1["rms_mean"].to_numpy(float), best_f1["link_f1_mean"].to_numpy(float), s=55.0)
            for _, r in best_f1.iterrows():
                ax.annotate(str(r["method"]), (float(r["rms_mean"]), float(r["link_f1_mean"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_xlabel("rms_mean")
            ax.set_ylabel("link_f1_mean")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"HCV variants Pareto (best by F1) | dx={float(dx):g}")
            fig.tight_layout()
            p = out_root / f"pareto_hcv_bestbyF1__dx{_fmt_token(float(dx))}.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            generated.append(p.name)

    # Group 3: top-5 configs by F1 per dx.
    if "link_f1_mean" in cfg.columns:
        cols = [
            c
            for c in [
                "method",
                "max_dist_factor",
                "motion_refine_iters",
                "link_f1_mean",
                "rms_mean",
                "id_switches_mean_per_track",
                "fragmentation_rate",
                "unmatched_rate_mean",
            ]
            if c in cfg.columns
        ]
        for dx in sorted(pd.to_numeric(cfg["dx_scale"], errors="coerce").dropna().unique().tolist()):
            sub = cfg[pd.to_numeric(cfg["dx_scale"], errors="coerce") == float(dx)].copy()
            methods_subset = [baseline_method]
            if best_variant_method is not None:
                methods_subset.append(best_variant_method)
            sub = sub[sub["method"].astype(str).isin(methods_subset)].copy()
            if sub.empty:
                continue
            sub["link_f1_mean"] = pd.to_numeric(sub["link_f1_mean"], errors="coerce")
            sub["rms_mean"] = pd.to_numeric(sub.get("rms_mean", np.nan), errors="coerce")
            sub = sub.dropna(subset=["link_f1_mean"])
            if sub.empty:
                continue
            top_raw = sub.sort_values(["link_f1_mean", "rms_mean"], ascending=[False, True]).head(5).copy()
            top = top_raw[cols].copy()

            # Export the same top-5 table as CSV with an explicit parameter-set column.
            def _fmt_num(v: object) -> str:
                vn = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
                return f"{float(vn):g}" if np.isfinite(vn) else str(v)

            def _param_set_label(r: pd.Series) -> str:
                parts: list[str] = []
                for key in ["max_dist_factor", "keep_percentile", "motion_refine_iters", "nn_scale_factor", "sigma_init_mode"]:
                    if key in r.index:
                        vv = r.get(key)
                        if key == "sigma_init_mode":
                            if isinstance(vv, str) and vv.strip():
                                parts.append(f"{key}={vv}")
                        else:
                            vn = pd.to_numeric(pd.Series([vv]), errors="coerce").iloc[0]
                            if np.isfinite(vn):
                                parts.append(f"{key}={_fmt_num(vn)}")
                cfg = str(r.get("config_id", "")).strip()
                if cfg:
                    parts.append(f"config_id={cfg}")
                return "; ".join(parts)

            top_csv = top.copy()
            top_csv.insert(1, "parameter set", top_raw.apply(_param_set_label, axis=1).tolist())
            p_csv = out_root / f"top5_hcv_byF1__dx{_fmt_token(float(dx))}.csv"
            top_csv.to_csv(p_csv, index=False)

            fig, ax = plt.subplots(figsize=(12.6, 2.7))
            ax.axis("off")
            cell_text: list[list[str]] = []
            for _, r in top.iterrows():
                row: list[str] = []
                for c in cols:
                    if c == "method":
                        row.append(str(r[c]))
                    else:
                        v = pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0]
                        row.append(f"{float(v):.4g}" if np.isfinite(v) else "nan")
                cell_text.append(row)
            tbl = ax.table(cellText=cell_text, colLabels=cols, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.2)
            ax.set_title(f"HCV variants top-5 by F1 | dx={float(dx):g}", fontsize=10)
            fig.tight_layout()
            p = out_root / f"top5_hcv_byF1__dx{_fmt_token(float(dx))}.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            generated.append(p.name)

    # Group 4: sensitivity for all tested methods with one line per dx.
    dx_values = sorted(pd.to_numeric(cfg["dx_scale"], errors="coerce").dropna().unique().tolist())
    if dx_values:
        for method in sorted(cfg["method"].astype(str).unique().tolist()):
            sub_m = cfg[cfg["method"].astype(str) == str(method)].copy()
            if sub_m.empty:
                continue
            for metric in metrics:
                if metric not in sub_m.columns:
                    continue
                fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9))
                for ax, xcol in zip(axs, ["max_dist_factor", "motion_refine_iters"]):
                    has_line = False
                    for dx in dx_values:
                        d = sub_m[np.isclose(pd.to_numeric(sub_m["dx_scale"], errors="coerce").to_numpy(float), float(dx))].copy()
                        if d.empty:
                            continue
                        d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
                        d[metric] = pd.to_numeric(d[metric], errors="coerce")
                        d = d.dropna(subset=[xcol, metric])
                        if d.empty:
                            continue
                        st = d.groupby(xcol, as_index=False)[metric].mean().sort_values(xcol)
                        if st.empty:
                            continue
                        ax.plot(st[xcol].to_numpy(float), st[metric].to_numpy(float), marker="o", label=f"dx={float(dx):g}")
                        has_line = True
                    ax.set_xlabel(xcol)
                    ax.set_ylabel(metric)
                    if metric == "link_f1_mean":
                        ax.set_ylim(0.0, 1.0)
                    else:
                        ax.set_ylim(bottom=0.0)
                    ax.set_title(xcol)
                    if has_line:
                        ax.legend(fontsize=8)
                fig.suptitle(f"Sensitivity | {method} | all dx | {metric}")
                fig.tight_layout()
                p = out_root / f"sensitivity__{_fmt_method_token(method)}__all_dx__{metric}.png"
                fig.savefig(p, dpi=300)
                plt.close(fig)
                generated.append(p.name)

    return sorted(set(generated))


def _plot_nis_gate_diagnostics(out_root: Path, agg: pd.DataFrame) -> list[str]:
    """Plot NIS gate diagnostics from per-run diagnostic CSV files."""
    generated: list[str] = []
    if agg.empty:
        return generated
    d = agg.copy()
    d["dx_scale"] = pd.to_numeric(d.get("dx_scale", np.nan), errors="coerce")
    d = d[d["method"].astype(str).str.contains("_nis_", regex=False)].copy()
    if d.empty:
        return generated

    per_run_summary_rows: list[dict[str, float | str]] = []
    for dx in sorted(d["dx_scale"].dropna().unique().tolist()):
        sub_dx = d[np.isclose(d["dx_scale"].to_numpy(float), float(dx))].copy()
        if sub_dx.empty:
            continue
        # Representative file: first lexicographic.
        file_stem = sorted(sub_dx["file_stem"].astype(str).unique().tolist())[0]
        sub_file = sub_dx[sub_dx["file_stem"].astype(str) == str(file_stem)].copy()
        if sub_file.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        plotted = False
        for method in sorted(sub_file["method"].astype(str).unique().tolist()):
            sub_m = sub_file[sub_file["method"].astype(str) == str(method)].copy()
            if sub_m.empty:
                continue
            sub_m["link_f1_mean"] = pd.to_numeric(sub_m.get("link_f1_mean", np.nan), errors="coerce")
            sub_m["rms_mean"] = pd.to_numeric(sub_m.get("rms_mean", np.nan), errors="coerce")
            sub_m = sub_m.sort_values(["link_f1_mean", "rms_mean"], ascending=[False, True])
            best = sub_m.iloc[0]
            cfg_id = str(best.get("config_id", ""))
            base_candidates = [
                out_root / f"dx_{_fmt_token(float(dx))}",
                out_root / f"density_{_fmt_token(float(dx))}",
            ]
            csv_path = None
            for b in base_candidates:
                cp = b / str(file_stem) / str(method) / cfg_id / f"{file_stem}_{method}_nis_gate_diagnostics.csv"
                if cp.exists():
                    csv_path = cp
                    break
            if csv_path is None:
                continue
            tab = pd.read_csv(csv_path)
            if tab.empty or "Time" not in tab.columns or "r_eq_median" not in tab.columns:
                continue
            tab["Time"] = pd.to_numeric(tab["Time"], errors="coerce")
            tab["r_eq_median"] = pd.to_numeric(tab["r_eq_median"], errors="coerce")
            tab = tab.dropna(subset=["Time", "r_eq_median"]).sort_values("Time")
            if tab.empty:
                continue
            ax.plot(tab["Time"].to_numpy(float), tab["r_eq_median"].to_numpy(float), marker="o", label=method)
            plotted = True
            per_run_summary_rows.append(
                {
                    "dx_scale": float(dx),
                    "method": str(method),
                    "file_stem": str(file_stem),
                    "config_id": cfg_id,
                    "r_eq_median_over_time": float(np.nanmedian(tab["r_eq_median"].to_numpy(float))),
                }
            )
        if plotted:
            ax.set_title(f"NIS gate equivalent radius over time | dx={float(dx):g} | file={file_stem}")
            ax.set_xlabel("Time")
            ax.set_ylabel("r_eq_median")
            ax.set_ylim(bottom=0.0)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = out_root / f"nis_gate_r_eq_over_time__dx{_fmt_token(float(dx))}.png"
            fig.savefig(p, dpi=300)
            generated.append(p.name)
        plt.close(fig)

    if per_run_summary_rows:
        summary = pd.DataFrame(per_run_summary_rows)
        summary.to_csv(out_root / "nis_gate_r_eq_summary.csv", index=False)
        agg_dx = (
            summary.groupby("dx_scale", as_index=False)["r_eq_median_over_time"]
            .agg(median="median", mean="mean", std="std", n="count")
            .sort_values("dx_scale")
        )
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        x = agg_dx["dx_scale"].to_numpy(float)
        y = agg_dx["median"].to_numpy(float)
        s = pd.to_numeric(agg_dx["std"], errors="coerce").fillna(0.0).to_numpy(float)
        ax.plot(x, y, marker="o", label="median r_eq")
        ax.fill_between(x, y - s, y + s, alpha=0.15)
        ax.set_title("NIS gate equivalent radius vs dx")
        ax.set_xlabel("dx_scale")
        ax.set_ylabel("r_eq_median")
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_root / "nis_gate_r_eq_vs_dx.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        generated.append(p.name)
    return sorted(set(generated))


def _plot_nis_overlay_r_eq_vs_dx(dx_based_root: Path, nn_based_root: Path, out_path: Path) -> bool:
    """Overlay r_eq-vs-dx curves for dx-based and nn-based NIS runs."""
    a_path = dx_based_root / "nis_gate_r_eq_summary.csv"
    b_path = nn_based_root / "nis_gate_r_eq_summary.csv"
    if not a_path.exists() or not b_path.exists():
        return False
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    if a.empty or b.empty:
        return False
    for df in [a, b]:
        df["dx_scale"] = pd.to_numeric(df.get("dx_scale", np.nan), errors="coerce")
        df["r_eq_median_over_time"] = pd.to_numeric(df.get("r_eq_median_over_time", np.nan), errors="coerce")

    aa = a.dropna(subset=["dx_scale", "r_eq_median_over_time"]).groupby("dx_scale", as_index=False)["r_eq_median_over_time"].median()
    bb = b.dropna(subset=["dx_scale", "r_eq_median_over_time"]).groupby("dx_scale", as_index=False)["r_eq_median_over_time"].median()
    if aa.empty or bb.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    aa = aa.sort_values("dx_scale")
    bb = bb.sort_values("dx_scale")
    ax.plot(aa["dx_scale"].to_numpy(float), aa["r_eq_median_over_time"].to_numpy(float), marker="o", label="NIS dx-based init")
    ax.plot(bb["dx_scale"].to_numpy(float), bb["r_eq_median_over_time"].to_numpy(float), marker="o", label="NIS nn-based init")
    ax.set_title("NIS equivalent gate radius vs dx: dx-init vs nn-init")
    ax.set_xlabel("dx_scale")
    ax.set_ylabel("median r_eq")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
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


def _birthdeath_stats_for_file(data_root: Path, regime: str, file_stem: str) -> dict[str, float]:
    """Load birth/death summary metrics for one synthetic file when available."""
    out: dict[str, float] = {
        "active_tracks_mean": float("nan"),
        "birth_events_total": float("nan"),
        "death_events_total": float("nan"),
        "birth_near_fraction": float("nan"),
        "birth_near_dist_mean": float("nan"),
    }
    gt_root = data_root / "gt" / str(regime) / "_groundtruth"
    trk_candidates = [
        gt_root / "track_birthdeath_summary.csv",
        gt_root / f"{file_stem}_track_birthdeath_summary.csv",
    ]
    act_candidates = [
        gt_root / "active_tracks_per_time.csv",
        gt_root / f"{file_stem}_active_tracks_per_time.csv",
    ]

    trk = None
    for p in trk_candidates:
        if p.exists():
            trk = pd.read_csv(p)
            break
    act = None
    for p in act_candidates:
        if p.exists():
            act = pd.read_csv(p)
            break

    if trk is not None and trk.shape[0] > 0:
        if "File" in trk.columns:
            trk = trk[trk["File"].astype(str) == str(file_stem)].copy()
        if trk.shape[0] > 0 and {"t_birth_time", "t_death_time", "birth_mode"}.issubset(trk.columns):
            b = pd.to_numeric(trk["t_birth_time"], errors="coerce")
            d = pd.to_numeric(trk["t_death_time"], errors="coerce")
            t0 = float(np.nanmin(b.to_numpy(float))) if b.notna().any() else 1.0
            t1 = float(np.nanmax(d.to_numpy(float))) if d.notna().any() else float("nan")
            out["birth_events_total"] = float(np.sum(b.to_numpy(float) > t0)) if b.notna().any() else float("nan")
            out["death_events_total"] = float(np.sum(d.to_numpy(float) < t1)) if d.notna().any() else float("nan")
            near_mask = trk["birth_mode"].astype(str) == "near"
            out["birth_near_fraction"] = float(np.mean(near_mask.to_numpy(float))) if near_mask.shape[0] > 0 else float("nan")
            if "init_dist_to_parent" in trk.columns:
                near_dist = pd.to_numeric(trk.loc[near_mask, "init_dist_to_parent"], errors="coerce").dropna().to_numpy(float)
                out["birth_near_dist_mean"] = float(np.mean(near_dist)) if near_dist.size else float("nan")

    if act is not None and act.shape[0] > 0:
        if "File" in act.columns:
            act = act[act["File"].astype(str) == str(file_stem)].copy()
        if act.shape[0] > 0:
            if "active_tracks" in act.columns:
                v = pd.to_numeric(act["active_tracks"], errors="coerce").dropna().to_numpy(float)
            else:
                v = pd.to_numeric(act.iloc[:, -1], errors="coerce").dropna().to_numpy(float)
            if v.size:
                out["active_tracks_mean"] = float(np.mean(v))

    return out


def _drift_stats_for_file(data_root: Path, regime: str, file_stem: str) -> dict[str, float]:
    """Load drift schedule summary metrics for one synthetic file when available."""
    out: dict[str, float] = {
        "drift_trans_median": float("nan"),
        "drift_trans_q90": float("nan"),
        "drift_rot_median_deg": float("nan"),
        "drift_rot_q90_deg": float("nan"),
    }
    gt_root = data_root / "gt" / str(regime) / "_groundtruth"
    candidates = [
        gt_root / f"{file_stem}_drift_schedule.csv",
        gt_root / "drift_schedule.csv",
    ]
    tab = None
    for p in candidates:
        if p.exists():
            tab = pd.read_csv(p)
            break
    if tab is None or tab.shape[0] == 0:
        return out
    if "File" in tab.columns:
        tab = tab[tab["File"].astype(str) == str(file_stem)].copy()
    if tab.shape[0] == 0:
        return out

    dtn = pd.to_numeric(tab.get("drift_trans_norm", np.nan), errors="coerce").dropna().to_numpy(float)
    drot = pd.to_numeric(tab.get("drift_rot_deg", np.nan), errors="coerce").dropna().to_numpy(float)
    if dtn.size:
        out["drift_trans_median"] = float(np.median(dtn))
        out["drift_trans_q90"] = float(np.quantile(dtn, 0.90))
    if drot.size:
        abs_rot = np.abs(drot)
        out["drift_rot_median_deg"] = float(np.median(abs_rot))
        out["drift_rot_q90_deg"] = float(np.quantile(abs_rot, 0.90))
    return out


def _stationary_stats_from_dx_schedule(dx_schedule_path: Path) -> dict[str, float]:
    """Derive stationary-track stats from dx schedule (all dx==0 across time)."""
    out = {
        "stationary_fraction_actual": float("nan"),
        "stationary_tracks_count": float("nan"),
    }
    if not dx_schedule_path.exists():
        return out
    try:
        dx_tab = pd.read_csv(dx_schedule_path)
    except Exception:
        return out
    if not {"TrackID", "dx_scale_um"}.issubset(dx_tab.columns):
        return out
    g = (
        dx_tab.groupby("TrackID", as_index=False)["dx_scale_um"]
        .agg(dx_abs_max=lambda x: float(np.max(np.abs(pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(float)))))
    )
    if g.shape[0] == 0:
        return out
    sta = g["dx_abs_max"].to_numpy(float) <= 1e-12
    out["stationary_tracks_count"] = float(np.sum(sta))
    out["stationary_fraction_actual"] = float(np.mean(sta))
    return out


def _write_density_stats_for_file(
    data_root: Path,
    regime: str,
    file_stem: str,
    volume_size: tuple[float, float, float],
) -> dict[str, float]:
    """Compute and persist density diagnostics for one synthetic file."""
    out = {
        "active_tracks_mean": float("nan"),
        "volume": float(volume_size[0] * volume_size[1] * volume_size[2]),
        "density_mean": float("nan"),
        "mean_nn_distance": float("nan"),
    }
    gt_root = data_root / "gt" / str(regime) / "_groundtruth"
    gt_path = gt_root / f"{file_stem}_groundtruth.csv"
    blind_path = data_root / "blind" / str(regime) / f"{file_stem}_blind.csv"
    if not gt_path.exists():
        return out
    gt = pd.read_csv(gt_path)
    if gt.empty or "Time" not in gt.columns or "TrackID" not in gt.columns:
        return out
    gt["Time"] = pd.to_numeric(gt["Time"], errors="coerce")
    gt = gt.dropna(subset=["Time", "TrackID"])
    if gt.empty:
        return out
    active = gt.groupby("Time", as_index=False)["TrackID"].nunique().rename(columns={"TrackID": "active_tracks"})
    if not active.empty:
        out["active_tracks_mean"] = float(np.mean(pd.to_numeric(active["active_tracks"], errors="coerce").dropna().to_numpy(float)))
        if out["volume"] > 0.0 and np.isfinite(out["active_tracks_mean"]):
            out["density_mean"] = float(out["active_tracks_mean"] / out["volume"])

    if blind_path.exists():
        try:
            blind_frames = load_blind_frames(blind_path)
            nn_df = compute_nn_distance(blind_frames)
            nn_vals = pd.to_numeric(nn_df.get("nn_mean", np.nan), errors="coerce").dropna().to_numpy(float)
            if nn_vals.size == 0:
                nn_vals = pd.to_numeric(nn_df.get("nn_median", np.nan), errors="coerce").dropna().to_numpy(float)
            if nn_vals.size:
                out["mean_nn_distance"] = float(np.nanmedian(nn_vals))
        except Exception:
            pass

    pd.DataFrame([{"File": str(file_stem), **out}]).to_csv(gt_root / f"{file_stem}_density_stats.csv", index=False)
    return out


def _density_stats_for_file(data_root: Path, regime: str, file_stem: str) -> dict[str, float]:
    """Load density diagnostics for one synthetic file."""
    out = {
        "active_tracks_mean": float("nan"),
        "volume": float("nan"),
        "density_mean": float("nan"),
        "mean_nn_distance": float("nan"),
    }
    p = data_root / "gt" / str(regime) / "_groundtruth" / f"{file_stem}_density_stats.csv"
    if not p.exists():
        return out
    try:
        tab = pd.read_csv(p)
    except Exception:
        return out
    if tab.empty:
        return out
    row = tab.iloc[0]
    for k in out.keys():
        out[k] = float(pd.to_numeric(pd.Series([row.get(k, np.nan)]), errors="coerce").iloc[0])
    return out


def _plot_density_sweep_summary(out_root: Path, agg: pd.DataFrame) -> list[str]:
    """Create density-dependent performance plots."""
    generated: list[str] = []
    if agg.empty or "density_scale" not in agg.columns or "method" not in agg.columns:
        return generated

    d = agg.copy()
    d["density_scale"] = pd.to_numeric(d["density_scale"], errors="coerce")
    d = d.dropna(subset=["density_scale", "method"])
    if d.empty:
        return generated

    metric_specs = [
        ("link_f1_mean", True),
        ("id_switches_mean_per_track", False),
        ("fragmentation_rate", False),
    ]
    for metric, higher_is_better in metric_specs:
        if metric not in d.columns:
            continue
        tmp = d.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric])
        if tmp.empty:
            continue
        tmp = tmp.sort_values(
            ["method", "density_scale", metric],
            ascending=[True, True, not higher_is_better],
        )
        best = tmp.groupby(["method", "density_scale"], as_index=False).first()
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for method in sorted(best["method"].astype(str).unique().tolist()):
            sm = best[best["method"].astype(str) == str(method)].sort_values("density_scale")
            ax.plot(sm["density_scale"].to_numpy(float), sm[metric].to_numpy(float), marker="o", label=method)
        ax.set_xlabel("density_scale")
        ax.set_ylabel(metric)
        ax.set_title(f"Best-by-config {metric} vs density")
        if metric == "link_f1_mean":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_root / f"plot_density_{metric}_vs_density_scale.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        generated.append(p.name)

    if "link_f1_mean" in d.columns and "mean_nn_distance" in d.columns:
        rr = d.copy()
        rr["link_f1_mean"] = pd.to_numeric(rr["link_f1_mean"], errors="coerce")
        rr["mean_nn_distance"] = pd.to_numeric(rr["mean_nn_distance"], errors="coerce")
        rr["median_r_eq"] = pd.to_numeric(rr.get("median_r_eq", np.nan), errors="coerce")
        rr["gate_radius_nominal"] = pd.to_numeric(rr.get("gate_radius_nominal", np.nan), errors="coerce")
        rr["gate_radius"] = rr["median_r_eq"]
        mask = rr["gate_radius"].isna() | (~np.isfinite(rr["gate_radius"]))
        rr.loc[mask, "gate_radius"] = rr.loc[mask, "gate_radius_nominal"]
        rr["gate_ratio"] = rr["gate_radius"] / rr["mean_nn_distance"].replace(0.0, np.nan)
        rr = rr.dropna(subset=["link_f1_mean", "gate_ratio", "method"])
        if not rr.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            for method in sorted(rr["method"].astype(str).unique().tolist()):
                sm = rr[rr["method"].astype(str) == str(method)].copy()
                sm = sm.sort_values("gate_ratio")
                ax.plot(sm["gate_ratio"].to_numpy(float), sm["link_f1_mean"].to_numpy(float), marker="o", linestyle="-", alpha=0.7, label=method)
            ax.set_xlabel("gate_radius / mean_nn_distance")
            ax.set_ylabel("link_f1_mean")
            ax.set_title("Association quality vs normalized gate radius")
            ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = out_root / "plot_link_f1_vs_gate_ratio.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            generated.append(p.name)
    return generated


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
    diag_summary = _compute_and_save_diagnostics(out_dir, blind_frames, gt_tracks_df)

    tracks_df.to_csv(out_dir / f"{file_stem}_tracks_inferred_{method_name}.csv", index=False)
    links_df.to_csv(out_dir / f"{file_stem}_links_{method_name}.csv", index=False)
    frame_stats_df.to_csv(out_dir / f"{file_stem}_frame_stats_{method_name}.csv", index=False)
    per_time.to_csv(out_dir / f"{file_stem}_accuracy_over_time_{method_name}.csv", index=False)
    mapping_table.to_csv(out_dir / f"{file_stem}_gt_inferred_mapping_{method_name}.csv", index=False)
    nis_rows = assoc_result.params.get("nis_gate_diagnostics_rows")
    if isinstance(nis_rows, list) and len(nis_rows) > 0:
        nis_df = pd.DataFrame(nis_rows)
        nis_df.to_csv(out_dir / f"{file_stem}_{method_name}_nis_gate_diagnostics.csv", index=False)

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
    row.update({k: float(v) for k, v in diag_summary.items()})
    # Include selected method params explicitly for downstream analysis tables.
    for pkey in ["sigma_init_mode", "nn_scale_factor", "initial_sigma_estimate", "median_sigma_over_time"]:
        if pkey in assoc_result.params:
            pv = assoc_result.params.get(pkey)
            if isinstance(pv, str):
                row[pkey] = pv
            else:
                pvn = pd.to_numeric(pd.Series([pv]), errors="coerce").iloc[0]
                row[pkey] = float(pvn) if np.isfinite(pvn) else np.nan
    if isinstance(nis_rows, list) and len(nis_rows) > 0:
        r_eq_vals = pd.to_numeric(pd.DataFrame(nis_rows).get("r_eq_median", np.nan), errors="coerce").dropna().to_numpy(float)
        if r_eq_vals.size:
            row["median_r_eq"] = float(np.nanmedian(r_eq_vals))
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

    # Generate a heteroscedastic no-drift dataset with varying dx per track/time.
    varying_root = DATA_ROOT_VARYING_DX / NODRIFT_VARYING_DX_TAG
    synthetic_generate_blind_pair(
        synthetic_dir=varying_root,
        seed=BASE_SEED,
        n_files=N_FILES,
        n_tracks=N_TRACKS,
        n_times=N_TIMES,
        regimes=[NODRIFT_VARYING_DX_REGIME],
        drift_params={},
        motion_params={
            "stationary_fraction": 0.10,
            "migrating_fraction": 0.85,
            "temporal_smoothness": 0.9,
            "dx_scale_um": float(NODRIFT_VARYING_DX_NOMINAL_SCALE),
            "dx_distribution": "gaussian",
            "dx_track_smoothness": 0.7,
            "dx_mode": "varying_per_track_time",
            "dx_min_um": 0.2,
            "dx_max_um": 4.0,
            "dx_varying_distribution": "uniform",
        },
        out_blind_dir=varying_root / "blind",
        out_gt_dir=varying_root / "gt",
        unblinding_path=varying_root / "unblinding.csv",
        overwrite=bool(OVERWRITE_DATASETS),
    )

    # Generate heteroscedastic no-drift dataset with births and deaths.
    varying_bd_root = DATA_ROOT_VARYING_DX_BIRTHDEATH / NODRIFT_VARYING_DX_BIRTHDEATH_TAG
    synthetic_generate_blind_pair(
        synthetic_dir=varying_bd_root,
        seed=BASE_SEED,
        n_files=N_FILES,
        n_tracks=N_TRACKS,
        n_times=N_TIMES,
        regimes=[NODRIFT_VARYING_DX_BIRTHDEATH_REGIME],
        drift_params={},
        motion_params={
            "stationary_fraction": 0.10,
            "migrating_fraction": 0.85,
            "temporal_smoothness": 0.9,
            "dx_scale_um": float(NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE),
            "dx_distribution": "gaussian",
            "dx_track_smoothness": 0.7,
            "dx_mode": "varying_per_track_time",
            "dx_min_um": 0.2,
            "dx_max_um": 4.0,
            "dx_varying_distribution": "uniform",
            "birthdeath_enable": True,
            "p_birth_late": 0.35,
            "p_death_early": 0.35,
            "min_life_frames": 5,
            "p_birth_random": 0.5,
            "birth_near_dist_min_um": 0.2,
            "birth_near_dist_max_um": 4.0,
            "birth_near_pick": "uniform_over_active_tracks",
        },
        out_blind_dir=varying_bd_root / "blind",
        out_gt_dir=varying_bd_root / "gt",
        unblinding_path=varying_bd_root / "unblinding.csv",
        overwrite=bool(OVERWRITE_DATASETS),
    )

    # Generate varying-dx + birth/death + stationary subset (no drift).
    varying_sta_root = DATA_ROOT_VARYING_DX_STATIONARY / NODRIFT_VARYING_DX_STATIONARY_TAG
    synthetic_generate_blind_pair(
        synthetic_dir=varying_sta_root,
        seed=BASE_SEED,
        n_files=N_FILES,
        n_tracks=N_TRACKS,
        n_times=N_TIMES,
        regimes=[NODRIFT_VARYING_DX_STATIONARY_REGIME],
        drift_params={},
        motion_params={
            "stationary_fraction": float(NODRIFT_VARYING_DX_STATIONARY_STATIONARY_FRACTION),
            "migrating_fraction": 0.75,
            "temporal_smoothness": 0.9,
            "dx_scale_um": float(NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE),
            "dx_distribution": "gaussian",
            "dx_track_smoothness": 0.7,
            "dx_mode": "varying_per_track_time",
            "dx_min_um": 0.2,
            "dx_max_um": 4.0,
            "dx_varying_distribution": "uniform",
            "birthdeath_enable": True,
            "p_birth_late": 0.35,
            "p_death_early": 0.35,
            "min_life_frames": 5,
            "p_birth_random": 0.5,
            "birth_near_dist_min_um": 0.2,
            "birth_near_dist_max_um": 4.0,
            "birth_near_pick": "uniform_over_active_tracks",
            "stationary_mode": "hard_zero",
            "stationary_apply_to": "all_tracks",
        },
        out_blind_dir=varying_sta_root / "blind",
        out_gt_dir=varying_sta_root / "gt",
        unblinding_path=varying_sta_root / "unblinding.csv",
        overwrite=bool(OVERWRITE_DATASETS),
    )
    # Unit sanity printout for achieved stationary fraction.
    gt_sta_root = varying_sta_root / "gt" / NODRIFT_VARYING_DX_STATIONARY_REGIME / "_groundtruth"
    for p in sorted(gt_sta_root.glob("*_dx_schedule.csv")):
        st = _stationary_stats_from_dx_schedule(p)
        print(
            "[stationary-sanity]",
            p.name,
            "stationary_tracks_count=",
            int(st["stationary_tracks_count"]) if np.isfinite(st["stationary_tracks_count"]) else "nan",
            "stationary_fraction_actual=",
            round(float(st["stationary_fraction_actual"]), 4) if np.isfinite(st["stationary_fraction_actual"]) else "nan",
        )

    # Generate density-sweep datasets: increasing track count in fixed volume.
    for density_scale in DENSITY_SCALES:
        ds_token = _fmt_token(float(density_scale))
        tag = f"{NODRIFT_VARYING_DX_DENSITY_TAG}_ds{ds_token}"
        varying_density_root = DATA_ROOT_VARYING_DX_DENSITY / tag
        n_tracks_density = int(max(1, round(float(DENSITY_INITIAL_TRACKS) * float(density_scale))))
        p_birth_base = 0.35
        p_death_base = 0.35
        synthetic_generate_blind_pair(
            synthetic_dir=varying_density_root,
            seed=BASE_SEED,
            n_files=N_FILES,
            n_tracks=n_tracks_density,
            n_times=N_TIMES,
            regimes=[NODRIFT_VARYING_DX_DENSITY_REGIME],
            drift_params={},
            motion_params={
                "initial_tracks": int(DENSITY_INITIAL_TRACKS),
                "density_scale": float(density_scale),
                "scene_bounds_um": tuple(VOLUME_SIZE),
                "stationary_fraction": float(NODRIFT_VARYING_DX_STATIONARY_STATIONARY_FRACTION),
                "migrating_fraction": 0.75,
                "temporal_smoothness": 0.9,
                "dx_scale_um": 1.0,
                "dx_distribution": "gaussian",
                "dx_track_smoothness": 0.7,
                "dx_mode": "varying_per_track_time",
                "dx_min_um": 0.2,
                "dx_max_um": 4.0,
                "dx_varying_distribution": "uniform",
                "birthdeath_enable": True,
                "p_birth_late": p_birth_base,
                "p_death_early": p_death_base,
                "min_life_frames": 5,
                "p_birth_random": 0.5,
                "birth_near_dist_min_um": 0.2,
                "birth_near_dist_max_um": 4.0,
                "birth_near_pick": "uniform_over_active_tracks",
                "stationary_mode": "hard_zero",
                "stationary_apply_to": "all_tracks",
            },
            out_blind_dir=varying_density_root / "blind",
            out_gt_dir=varying_density_root / "gt",
            unblinding_path=varying_density_root / "unblinding.csv",
            overwrite=bool(OVERWRITE_DATASETS),
        )
        blind_density_dir = varying_density_root / "blind" / NODRIFT_VARYING_DX_DENSITY_REGIME
        for blind_file in sorted(blind_density_dir.glob("*_blind.csv")):
            file_stem = blind_file.name.replace("_blind.csv", "")
            ds = _write_density_stats_for_file(
                varying_density_root,
                NODRIFT_VARYING_DX_DENSITY_REGIME,
                file_stem,
                tuple(float(v) for v in VOLUME_SIZE),
            )
            print(
                "[density-sanity]",
                f"ds={float(density_scale):.2f}",
                file_stem,
                "active_tracks_mean=",
                round(float(ds["active_tracks_mean"]), 3) if np.isfinite(ds["active_tracks_mean"]) else "nan",
                "density_mean=",
                f"{float(ds['density_mean']):.6g}" if np.isfinite(ds["density_mean"]) else "nan",
                "mean_nn_distance=",
                round(float(ds["mean_nn_distance"]), 3) if np.isfinite(ds["mean_nn_distance"]) else "nan",
            )

    # Generate heteroscedastic varying-dx + birth/death + global-drift dataset.
    varying_bd_drift_root = DATA_ROOT_VARYING_DX_BIRTHDEATH_DRIFT / NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_TAG
    synthetic_generate_blind_pair_varying_dx_birthdeath_drift(
        synthetic_dir=varying_bd_drift_root,
        seed=BASE_SEED,
        n_files=N_FILES,
        n_tracks=N_TRACKS,
        n_times=N_TIMES,
        drift_params=dict(DRIFT_PARAMS_BIRTHDEATH),
        motion_params={
            "stationary_fraction": float(MOTION_MIX_PARAMS["stationary_fraction"]),
            "migrating_fraction": float(MOTION_MIX_PARAMS["migrating_fraction"]),
            "locally_coherent_fraction": float(MOTION_MIX_PARAMS["locally_coherent_fraction"]),
            "temporal_smoothness": 0.9,
            "dx_scale_um": float(NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE),
            "dx_distribution": "gaussian",
            "dx_track_smoothness": 0.7,
            "dx_mode": "varying_per_track_time",
            "dx_min_um": 0.2,
            "dx_max_um": 4.0,
            "dx_varying_distribution": "uniform",
            "birthdeath_enable": True,
            "p_birth_late": 0.35,
            "p_death_early": 0.35,
            "min_life_frames": 5,
            "p_birth_random": 0.5,
            "birth_near_dist_min_um": 0.2,
            "birth_near_dist_max_um": 4.0,
            "birth_near_pick": "uniform_over_active_tracks",
        },
        out_blind_dir=varying_bd_drift_root / "blind",
        out_gt_dir=varying_bd_drift_root / "gt",
        unblinding_path=varying_bd_drift_root / "unblinding.csv",
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

""" 
out_root = _results_root("nodrift_dx_sweep")
rows: list[dict[str, float | str]] = []
diag_file_rows: list[dict[str, float | str]] = []

dx =  DX_SCALES[-1]
tag = f"nodrift_dx_{_fmt_token(dx)}"
data_root = DATA_ROOT / tag
blind_reg_dir = data_root / "blind" / "nodrift_dx_sweep"
gt_reg_dir = data_root / "gt" / "nodrift_dx_sweep"
map_path = data_root / "unblinding.csv"

mapping_all = pd.read_csv(map_path)
blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))
blind_file = blind_files[0]
file_stem = blind_file.name.replace("_blind.csv", "")
map_sub = mapping_all[
    (mapping_all["Regime"].astype(str) == "nodrift_dx_sweep")
    & (mapping_all["File"].astype(str) == file_stem)
].copy()


blind_frames = load_blind_frames(blind_file)
gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", "nodrift_dx_sweep", file_stem, mapping_all)
file_diag_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / "_diagnostics"
file_diag = _compute_and_save_diagnostics(file_diag_dir, blind_frames, gt_tracks)
diag_file_rows.append(
    {
        "dx_scale": float(dx),
        "file_stem": str(file_stem),
        **{k: float(v) for k, v in file_diag.items()},
    }
)
        
m= NODRIFT_METHODS[0]
assoc = _dispatch_association_method(blind_frames, m, dx_scale=float(dx))
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
) """


if RUN_NODRIFT_DX_SWEEP:
    out_root = _results_root("nodrift_dx_sweep")
    rows: list[dict[str, float | str]] = []
    diag_file_rows: list[dict[str, float | str]] = []

    for dx in DX_SCALES:
        # dx =  DX_SCALES[-1]
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
            # blind_file = blind_files[0]
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == "nodrift_dx_sweep")
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", "nodrift_dx_sweep", file_stem, mapping_all)
            file_diag_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / "_diagnostics"
            file_diag = _compute_and_save_diagnostics(file_diag_dir, blind_frames, gt_tracks)
            diag_file_rows.append(
                {
                    "dx_scale": float(dx),
                    "file_stem": str(file_stem),
                    **{k: float(v) for k, v in file_diag.items()},
                }
            )

            for m in NODRIFT_METHODS:
                # m= NODRIFT_METHODS[0]
                assoc = _dispatch_association_method(blind_frames, m, dx_scale=float(dx))
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
            p1 = out_root / f"plot_dx_sweep_{metric}_lines.pdf"
            if _plot_metric_vs_dx(agg, metric=metric, out_path=p1, title=f"No-drift dx sweep: {metric}"):
                generated.append(p1.name)
            p2 = out_root / f"plot_dx_sweep_{metric}_heatmap.pdf"
            if _plot_metric_heatmap_dx_method(agg, metric=metric, out_path=p2, 
                                              title=f"{metric}", 
                                              figsize=(9.0*CM, 5.1*CM), clims=None, title_on=False):
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

    if diag_file_rows:
        diag = pd.DataFrame(diag_file_rows)
        diag.to_csv(out_root / "aggregated_diagnostics_by_file.csv", index=False)
        diag_summary = (
            diag.groupby("dx_scale", as_index=False)[["diag_nn_median", "diag_disp_median", "diag_rho_median"]]
            .agg(["mean", "std", "count"])
        )
        # Flatten multi-index columns.
        diag_summary.columns = [
            "dx_scale"
            if c[0] == "dx_scale"
            else f"{c[0]}_{c[1]}"
            for c in diag_summary.columns.to_flat_index()
        ]
        diag_summary.to_csv(out_root / "aggregated_diagnostics_by_dx.csv", index=False)
        _plot_simple_series(
            diag.groupby("dx_scale", as_index=False)["diag_nn_median"].mean(),
            x_col="dx_scale",
            y_col="diag_nn_median",
            title="Median nearest-neighbor distance vs dx",
            y_label="median d_NN (um)",
            out_path=out_root / "plot_diagnostics_median_dnn_vs_dx.png",
        )
        _plot_simple_series(
            diag.groupby("dx_scale", as_index=False)["diag_disp_median"].mean(),
            x_col="dx_scale",
            y_col="diag_disp_median",
            title="Median displacement vs dx",
            y_label="median Δx (um)",
            out_path=out_root / "plot_diagnostics_median_dx_vs_dx.png",
        )
        _plot_simple_series(
            diag.groupby("dx_scale", as_index=False)["diag_rho_median"].mean(),
            x_col="dx_scale",
            y_col="diag_rho_median",
            title="Median ambiguity ratio vs dx",
            y_label="median ρ",
            out_path=out_root / "plot_diagnostics_median_rho_vs_dx.png",
        )

# %% NO-DRIFT PARAMETER SWEEP
"""Sweep gating and refinement parameters on a reduced no-drift subset.

What is tested:
- max-distance gating factor,
- percentile gating,
- motion refinement iterations (for methods supporting it).

Why it matters:
- Gives interpretable settings tradeoffs without rerunning full benchmarks.
"""

if RUN_NODRIFT_PARAM_SWEEP:
    out_root = _results_root("nodrift_param_sweep")
    rows: list[dict[str, float | str]] = []

    for dx in NODRIFT_PARAM_SWEEP_DX:
        tag = f"nodrift_dx_{_fmt_token(dx)}"
        data_root = DATA_ROOT / tag
        blind_reg_dir = data_root / "blind" / "nodrift_dx_sweep"
        gt_reg_dir = data_root / "gt" / "nodrift_dx_sweep"
        map_path = data_root / "unblinding.csv"
        if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
            print(f"[WARN] Missing dataset for param sweep dx={dx}, skip: {data_root}")
            continue
        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, NODRIFT_PARAM_SWEEP_MAX_FILES))]

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
                kind = str(m.get("kind", "nodrift"))
                iter_list = NODRIFT_PARAM_SWEEP_REFINE_ITERS if kind == "nodrift" else [0]
                for md_factor in NODRIFT_PARAM_SWEEP_MAX_DIST_FACTORS:
                    for kp in NODRIFT_PARAM_SWEEP_KEEP_PERCENTILES:
                        for iters in iter_list:
                            gating_override = {
                                "max_dist": float(max(0.5, float(md_factor) * float(dx))),
                                "keep_percentile": float(kp),
                            }
                            params_override = {"motion_refine_iters": int(iters)}
                            assoc = _dispatch_association_method(
                                blind_frames,
                                m,
                                dx_scale=float(dx),
                                gating_override=gating_override,
                                params_override=params_override,
                            )
                            cfg_id = f"md{_fmt_token(float(md_factor))}_kp{int(kp)}_it{int(iters)}"
                            run_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / str(m["name"]) / cfg_id
                            row = _evaluate_and_save_run(
                                out_dir=run_dir,
                                assoc_result=assoc,
                                gt_tracks_df=gt_tracks,
                                mapping_df=map_sub,
                                blind_frames=blind_frames,
                                regime="nodrift_dx_sweep",
                                file_stem=file_stem,
                                method_name=str(m["name"]),
                                extras={
                                    "dx_scale": float(dx),
                                    "config_id": cfg_id,
                                    "max_dist_factor": float(md_factor),
                                    "keep_percentile": float(kp),
                                    "motion_refine_iters": int(iters),
                                },
                            )
                            rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg.to_csv(out_root / "aggregated_param_sweep.csv", index=False)

        metric_cols = [
            c
            for c in [
                "link_precision_mean",
                "link_recall_mean",
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=[
                    "method",
                    "dx_scale",
                    "max_dist_factor",
                    "keep_percentile",
                    "motion_refine_iters",
                    "config_id",
                ],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    ["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "metric"],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_scale", "motion_refine_iters", "max_dist_factor", "keep_percentile"])
                .reset_index(drop=True)
            )
            stats.to_csv(out_root / "aggregated_param_sweep_stats.csv", index=False)

        generated: list[str] = []
        out_root_single_heatmap = out_root / "heatmaps"
        out_root_single_heatmap.mkdir(parents=True, exist_ok=True)
        for metric in [m for m in ["rms_mean", "link_f1_mean"] if m in agg.columns]:
            for method in sorted(agg["method"].astype(str).unique().tolist()):
                for dxv in sorted(pd.to_numeric(agg["dx_scale"], errors="coerce").dropna().unique().tolist()):
                    sub = agg[(agg["method"].astype(str) == str(method)) & (pd.to_numeric(agg["dx_scale"], errors="coerce") == float(dxv))]
                    if sub.empty:
                        continue
                    for it in sorted(pd.to_numeric(sub["motion_refine_iters"], errors="coerce").dropna().astype(int).unique().tolist()):
                        outp = out_root_single_heatmap / f"plot_param_sweep_{metric}__{method}__dx{_fmt_token(float(dxv))}__it{int(it)}.png"
                        if _plot_param_heatmap(
                            agg,
                            metric=metric,
                            method_name=method,
                            dx_scale=float(dxv),
                            motion_refine_iters=int(it),
                            out_path=outp,
                        ):
                            generated.append(outp.name)

        # Overall compact publication-style overview from aggregated CSV outputs.
        generated.extend(_plot_param_sweep_overview(out_root))
        generated = sorted(set(generated))


# %% HUNGARIAN CV VARIANTS SWEEP
"""Focused sweep: baseline Hungarian-CV versus four new Hungarian-CV variants.

What is tested:
- baseline: `hungarian_gating_cv`
- new variants:
  - adaptive max-distance
  - innovation-normalized percentile gating
  - switch-penalty in link cost
  - two-stage Hungarian

Why it matters:
- isolates improvements around the current strongest method with controlled
  runtime and fixed percentile gating.
"""

if RUN_HUNGARIAN_CV_VARIANTS_SWEEP:
    out_root = _results_root("hungarian_cv_variants_sweep")
    rows: list[dict[str, float | str]] = []

    for dx in HCV_DX_SCALES:
        tag = f"nodrift_dx_{_fmt_token(dx)}"
        data_root = DATA_ROOT / tag
        blind_reg_dir = data_root / "blind" / "nodrift_dx_sweep"
        gt_reg_dir = data_root / "gt" / "nodrift_dx_sweep"
        map_path = data_root / "unblinding.csv"
        if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
            print(f"[WARN] Missing dataset for HCV sweep dx={dx}, skip: {data_root}")
            continue

        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES))]

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == "nodrift_dx_sweep")
                & (mapping_all["File"].astype(str) == file_stem)].copy()
            if map_sub.empty:
                continue

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", "nodrift_dx_sweep", file_stem, mapping_all)

            for m in HUNGARIAN_CV_VARIANTS:
                for md_factor in HCV_MAX_DIST_FACTORS:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(md_factor) * float(dx))),
                            "keep_percentile": float(HCV_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(md_factor),
                            "keep_percentile": float(HCV_KEEP_PERCENTILE),
                            "dx_scale": float(dx),
                            "sigma_init": float(dx),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=float(dx),
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = f"md{_fmt_token(float(md_factor))}_kp{int(HCV_KEEP_PERCENTILE)}_it{int(iters)}"
                        run_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / str(m["name"]) / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime="nodrift_dx_sweep",
                            file_stem=file_stem,
                            method_name=str(m["name"]),
                            extras={
                                "dx_scale": float(dx),
                                "config_id": cfg_id,
                                "max_dist_factor": float(md_factor),
                                "keep_percentile": float(HCV_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                            },
                        )
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_variants.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "config_id"],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "metric"], as_index=False)["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_scale", "max_dist_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_variants_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_variants_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dx_scales: {HCV_DX_SCALES}",
                    f"max_dist_factors: {HCV_MAX_DIST_FACTORS}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"keep_percentile_fixed: {HCV_KEEP_PERCENTILE}",
                    f"methods: {[m['name'] for m in HUNGARIAN_CV_VARIANTS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "plots:",
                    *[f"- {p}" for p in sorted(set(generated))],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% HCV NIS GATING COMPARISON/SWEEP
"""Compare Euclidean-gated vs NIS-gated Hungarian CV variants.

This sweep keeps `keep_percentile` fixed and does not sweep max-distance.
For Euclidean baselines, a single fixed `max_dist_factor` is used.
"""

if RUN_HCV_NIS_GATING_COMPARISON:
    #out_root = _results_root("hungarian_cv_variants_sweep") / "nis_gating_comparison"
    out_root = ASSOC_ROOT / "hungarian_cv_variants_sweep_nis"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for dx in HCV_DX_SCALES:
        tag = f"nodrift_dx_{_fmt_token(dx)}"
        data_root = DATA_ROOT / tag
        blind_reg_dir = data_root / "blind" / "nodrift_dx_sweep"
        gt_reg_dir = data_root / "gt" / "nodrift_dx_sweep"
        map_path = data_root / "unblinding.csv"
        if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
            print(f"[WARN] Missing dataset for HCV NIS comparison dx={dx}, skip: {data_root}")
            continue

        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES))]

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

            for m in HCV_NIS_METHODS:
                for iters in HCV_REFINE_ITERS:
                    gating_override = {
                        "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * float(dx))),
                        "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                    }
                    params_override = {
                        "motion_refine_iters": int(iters),
                        "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                        "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        "dx_scale": float(dx),
                        "sigma_init": float(dx),
                    }
                    assoc = _dispatch_association_method(
                        blind_frames,
                        m,
                        dx_scale=float(dx),
                        gating_override=gating_override,
                        params_override=params_override,
                    )
                    cfg_id = f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}_kp{int(HCV_NIS_KEEP_PERCENTILE)}_it{int(iters)}"
                    run_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / str(m["name"]) / cfg_id
                    row = _evaluate_and_save_run(
                        out_dir=run_dir,
                        assoc_result=assoc,
                        gt_tracks_df=gt_tracks,
                        mapping_df=map_sub,
                        blind_frames=blind_frames,
                        regime="nodrift_dx_sweep",
                        file_stem=file_stem,
                        method_name=str(m["name"]),
                        extras={
                            "dx_scale": float(dx),
                            "config_id": cfg_id,
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "motion_refine_iters": int(iters),
                        },
                    )
                    rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_nis_gating_comparison.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "config_id"],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "metric"], as_index=False)["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_scale", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_nis_gating_comparison_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_nis_gating_comparison_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dx_scales: {HCV_DX_SCALES}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "nis_diag_csv_pattern: <file_stem>_<method>_nis_gate_diagnostics.csv",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% HCV_NIS_DX_BLIND_SWEEP
"""Duplicate NIS comparison with dx-blind (nn-based) sigma initialization.

This sweep keeps percentile fixed and does not sweep max-distance or sigma_init.
For NIS methods, sigma is initialized from nearest-neighbor spacing in frame 1.
"""

if RUN_HCV_NIS_DX_BLIND_SWEEP:
    out_root = ASSOC_ROOT / "hungarian_cv_variants_sweep_nis_dx_blind"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for dx in HCV_DX_SCALES:
        tag = f"nodrift_dx_{_fmt_token(dx)}"
        data_root = DATA_ROOT / tag
        blind_reg_dir = data_root / "blind" / "nodrift_dx_sweep"
        gt_reg_dir = data_root / "gt" / "nodrift_dx_sweep"
        map_path = data_root / "unblinding.csv"
        if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
            print(f"[WARN] Missing dataset for HCV nn-based sweep dx={dx}, skip: {data_root}")
            continue

        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES))]

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

            for m in HCV_NIS_METHODS:
                method_name = str(m.get("name", ""))
                nn_scales = HCV_NIS_DX_BLIND_NN_SCALE_FACTORS if "_nis_" in method_name else [0.3]
                for nn_scale in nn_scales:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * float(dx))),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "dx_scale": float(dx),
                            "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            "nn_scale_factor": float(nn_scale),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=float(dx),
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}_kp{int(HCV_NIS_KEEP_PERCENTILE)}_it{int(iters)}_nn{_fmt_token(float(nn_scale))}"
                        run_dir = out_root / f"dx_{_fmt_token(dx)}" / file_stem / method_name / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime="nodrift_dx_sweep",
                            file_stem=file_stem,
                            method_name=method_name,
                            extras={
                                "dx_scale": float(dx),
                                "config_id": cfg_id,
                                "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                                "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                                "nn_scale_factor": float(nn_scale),
                                "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            },
                        )
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_nis_dx_blind.csv"
        agg.to_csv(agg_path, index=False)
        # reload agg from saved CSV to ensure consistent dtypes for plotting/stats
        agg = pd.read_csv(agg_path)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "config_id", "nn_scale_factor", "sigma_init_mode"],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    ["method", "dx_scale", "max_dist_factor", "keep_percentile", "motion_refine_iters", "nn_scale_factor", "sigma_init_mode", "metric"],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_scale", "nn_scale_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))

        dx_based_root = _results_root("hungarian_cv_variants_sweep") / "nis_gating_comparison"
        overlay_path = out_root / "nis_gate_r_eq_vs_dx_overlay_dx_vs_nn.png"
        if _plot_nis_overlay_r_eq_vs_dx(dx_based_root, out_root, overlay_path):
            generated.append(overlay_path.name)
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dx_scales: {HCV_DX_SCALES}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"nn_scale_factors: {HCV_NIS_DX_BLIND_NN_SCALE_FACTORS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "nis_diag_csv_pattern: <file_stem>_<method>_nis_gate_diagnostics.csv",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )


# %% HCV_NIS_DX_BLIND_SWEEP_VARYING_DX
"""Run NIS dx-blind sweep on the heteroscedastic varying-dx no-drift dataset.

What is tested:
- Identical HCV NIS method family as in `HCV_NIS_DX_BLIND_SWEEP`
  but on data where dx varies by track and time.

Why it matters:
- Evaluates robustness of association/gating when motion amplitudes are not
  globally homogeneous.
"""

if RUN_HCV_NIS_DX_BLIND_SWEEP_VARYING_DX:
    out_root = _results_root("hungarian_cv_variants_sweep_nis_dx_blind_varying_dx")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    data_root = DATA_ROOT_VARYING_DX / NODRIFT_VARYING_DX_TAG
    blind_reg_dir = data_root / "blind" / NODRIFT_VARYING_DX_REGIME
    gt_reg_dir = data_root / "gt" / NODRIFT_VARYING_DX_REGIME
    map_path = data_root / "unblinding.csv"
    if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
        print(f"[WARN] Missing varying-dx dataset, skip: {data_root}")
    else:
        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES_VARYING_DX))]

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == NODRIFT_VARYING_DX_REGIME)
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            dx_schedule_path = data_root / "gt" / NODRIFT_VARYING_DX_REGIME / "_groundtruth" / f"{file_stem}_dx_schedule.csv"
            dx_stats: dict[str, float] = {}
            if dx_schedule_path.exists():
                try:
                    dx_tab = pd.read_csv(dx_schedule_path)
                    dx_vals = pd.to_numeric(dx_tab.get("dx_scale_um", np.nan), errors="coerce").dropna().to_numpy(float)
                    if dx_vals.size:
                        dx_stats = {
                            "dx_actual_q10": float(np.quantile(dx_vals, 0.10)),
                            "dx_actual_q50": float(np.quantile(dx_vals, 0.50)),
                            "dx_actual_q90": float(np.quantile(dx_vals, 0.90)),
                            "dx_actual_mean": float(np.mean(dx_vals)),
                            "dx_actual_std": float(np.std(dx_vals)),
                        }
                except Exception as exc:
                    print(f"[WARN] Could not read dx schedule stats: {dx_schedule_path} ({exc})")

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", NODRIFT_VARYING_DX_REGIME, file_stem, mapping_all)

            for m in HCV_NIS_METHODS:
                method_name = str(m.get("name", ""))
                nn_scales = HCV_NIS_DX_BLIND_NN_SCALE_FACTORS if "_nis_" in method_name else [0.3]
                for nn_scale in nn_scales:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * float(NODRIFT_VARYING_DX_NOMINAL_SCALE))),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "dx_scale": float(NODRIFT_VARYING_DX_NOMINAL_SCALE),
                            "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            "nn_scale_factor": float(nn_scale),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=float(NODRIFT_VARYING_DX_NOMINAL_SCALE),
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = (
                            f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}"
                            f"_kp{int(HCV_NIS_KEEP_PERCENTILE)}"
                            f"_it{int(iters)}"
                            f"_nn{_fmt_token(float(nn_scale))}"
                        )
                        run_dir = out_root / "dx_varying" / file_stem / method_name / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime=NODRIFT_VARYING_DX_REGIME,
                            file_stem=file_stem,
                            method_name=method_name,
                            extras={
                                "dx_scale": float(NODRIFT_VARYING_DX_NOMINAL_SCALE),
                                "dx_regime": str(NODRIFT_VARYING_DX_TAG),
                                "config_id": cfg_id,
                                "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                                "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                                "nn_scale_factor": float(nn_scale),
                                "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                                **dx_stats,
                            },
                        )
                        if dx_schedule_path.exists():
                            try:
                                shutil.copy2(dx_schedule_path, run_dir / dx_schedule_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy dx schedule CSV to run dir: {exc}")
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_nis_dx_blind_varying_dx.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=[
                    "method",
                    "dx_scale",
                    "dx_regime",
                    "max_dist_factor",
                    "keep_percentile",
                    "motion_refine_iters",
                    "config_id",
                    "nn_scale_factor",
                    "sigma_init_mode",
                ],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    [
                        "method",
                        "dx_scale",
                        "dx_regime",
                        "max_dist_factor",
                        "keep_percentile",
                        "motion_refine_iters",
                        "nn_scale_factor",
                        "sigma_init_mode",
                        "metric",
                    ],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_regime", "nn_scale_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_varying_dx_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_varying_dx_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dataset_root: {data_root}",
                    f"regime: {NODRIFT_VARYING_DX_REGIME}",
                    f"dx_regime_tag: {NODRIFT_VARYING_DX_TAG}",
                    f"dx_nominal_scale_for_dispatch: {NODRIFT_VARYING_DX_NOMINAL_SCALE}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"nn_scale_factors: {HCV_NIS_DX_BLIND_NN_SCALE_FACTORS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "dx_schedule_csv_pattern: <file_stem>_dx_schedule.csv (copied into each run dir when present)",
                    "nis_diag_csv_pattern: <file_stem>_<method>_nis_gate_diagnostics.csv",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH
"""Run NIS dx-blind sweep on varying-dx dataset with births/deaths.

What is tested:
- Same method family as HCV_NIS_DX_BLIND sweeps, but on a harder regime with
  variable step scales and dynamic track count due to births/deaths.
"""

if RUN_HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH:
    out_root = _results_root("hungarian_cv_variants_sweep_nis_dx_blind_birthdeath")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    data_root = DATA_ROOT_VARYING_DX_BIRTHDEATH / NODRIFT_VARYING_DX_BIRTHDEATH_TAG
    blind_reg_dir = data_root / "blind" / NODRIFT_VARYING_DX_BIRTHDEATH_REGIME
    gt_reg_dir = data_root / "gt" / NODRIFT_VARYING_DX_BIRTHDEATH_REGIME
    map_path = data_root / "unblinding.csv"
    if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
        print(f"[WARN] Missing varying-dx birthdeath dataset, skip: {data_root}")
    else:
        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES_BIRTHDEATH))]

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == NODRIFT_VARYING_DX_BIRTHDEATH_REGIME)
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            dx_schedule_path = data_root / "gt" / NODRIFT_VARYING_DX_BIRTHDEATH_REGIME / "_groundtruth" / f"{file_stem}_dx_schedule.csv"
            dx_stats: dict[str, float] = {}
            if dx_schedule_path.exists():
                try:
                    dx_tab = pd.read_csv(dx_schedule_path)
                    dx_vals = pd.to_numeric(dx_tab.get("dx_scale_um", np.nan), errors="coerce").dropna().to_numpy(float)
                    if dx_vals.size:
                        dx_stats = {
                            "dx_actual_q10": float(np.quantile(dx_vals, 0.10)),
                            "dx_actual_q50": float(np.quantile(dx_vals, 0.50)),
                            "dx_actual_q90": float(np.quantile(dx_vals, 0.90)),
                            "dx_actual_mean": float(np.mean(dx_vals)),
                            "dx_actual_std": float(np.std(dx_vals)),
                        }
                except Exception as exc:
                    print(f"[WARN] Could not read dx schedule stats: {dx_schedule_path} ({exc})")

            bd_stats = _birthdeath_stats_for_file(data_root, NODRIFT_VARYING_DX_BIRTHDEATH_REGIME, file_stem)

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", NODRIFT_VARYING_DX_BIRTHDEATH_REGIME, file_stem, mapping_all)

            for m in HCV_NIS_METHODS:
                method_name = str(m.get("name", ""))
                nn_scales = HCV_NIS_DX_BLIND_NN_SCALE_FACTORS if "_nis_" in method_name else [0.3]
                for nn_scale in nn_scales:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * float(NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE))),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "dx_scale": float(NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE),
                            "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            "nn_scale_factor": float(nn_scale),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=float(NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE),
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = (
                            f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}"
                            f"_kp{int(HCV_NIS_KEEP_PERCENTILE)}"
                            f"_it{int(iters)}"
                            f"_nn{_fmt_token(float(nn_scale))}"
                        )
                        run_dir = out_root / "dx_birthdeath" / file_stem / method_name / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime=NODRIFT_VARYING_DX_BIRTHDEATH_REGIME,
                            file_stem=file_stem,
                            method_name=method_name,
                            extras={
                                "dx_scale": float(NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE),
                                "dx_regime": str(NODRIFT_VARYING_DX_BIRTHDEATH_TAG),
                                "config_id": cfg_id,
                                "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                                "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                                "nn_scale_factor": float(nn_scale),
                                "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                                **dx_stats,
                                **bd_stats,
                            },
                        )
                        if dx_schedule_path.exists():
                            try:
                                shutil.copy2(dx_schedule_path, run_dir / dx_schedule_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy dx schedule CSV to run dir: {exc}")
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=[
                    "method",
                    "dx_scale",
                    "dx_regime",
                    "max_dist_factor",
                    "keep_percentile",
                    "motion_refine_iters",
                    "config_id",
                    "nn_scale_factor",
                    "sigma_init_mode",
                ],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    [
                        "method",
                        "dx_scale",
                        "dx_regime",
                        "max_dist_factor",
                        "keep_percentile",
                        "motion_refine_iters",
                        "nn_scale_factor",
                        "sigma_init_mode",
                        "metric",
                    ],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_regime", "nn_scale_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dataset_root: {data_root}",
                    f"regime: {NODRIFT_VARYING_DX_BIRTHDEATH_REGIME}",
                    f"dx_regime_tag: {NODRIFT_VARYING_DX_BIRTHDEATH_TAG}",
                    f"dx_nominal_scale_for_dispatch: {NODRIFT_VARYING_DX_BIRTHDEATH_NOMINAL_SCALE}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"nn_scale_factors: {HCV_NIS_DX_BLIND_NN_SCALE_FACTORS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "extra_aggregated_columns: active_tracks_mean,birth_events_total,death_events_total,birth_near_fraction,birth_near_dist_mean",
                    "dx_schedule_csv_pattern: <file_stem>_dx_schedule.csv (copied into each run dir when present)",
                    "nis_diag_csv_pattern: <file_stem>_<method>_nis_gate_diagnostics.csv",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH_DRIFT
"""Run NIS dx-blind sweep on varying-dx + birth/death + drift dataset."""

if RUN_HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH_DRIFT:
    out_root = _results_root("hungarian_cv_variants_sweep_nis_dx_blind_birthdeath_drift")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    data_root = DATA_ROOT_VARYING_DX_BIRTHDEATH_DRIFT / NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_TAG
    blind_reg_dir = data_root / "blind" / NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME
    gt_reg_dir = data_root / "gt" / NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME
    map_path = data_root / "unblinding.csv"
    if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
        print(f"[WARN] Missing varying-dx birthdeath+drift dataset, skip: {data_root}")
    else:
        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES_BIRTHDEATH_DRIFT))]

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME)
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            dx_schedule_path = data_root / "gt" / NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME / "_groundtruth" / f"{file_stem}_dx_schedule.csv"
            drift_schedule_path = data_root / "gt" / NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME / "_groundtruth" / f"{file_stem}_drift_schedule.csv"
            dx_stats: dict[str, float] = {}
            if dx_schedule_path.exists():
                try:
                    dx_tab = pd.read_csv(dx_schedule_path)
                    dx_vals = pd.to_numeric(dx_tab.get("dx_scale_um", np.nan), errors="coerce").dropna().to_numpy(float)
                    if dx_vals.size:
                        dx_stats = {
                            "dx_actual_q10": float(np.quantile(dx_vals, 0.10)),
                            "dx_actual_q50": float(np.quantile(dx_vals, 0.50)),
                            "dx_actual_q90": float(np.quantile(dx_vals, 0.90)),
                            "dx_actual_mean": float(np.mean(dx_vals)),
                            "dx_actual_std": float(np.std(dx_vals)),
                        }
                except Exception as exc:
                    print(f"[WARN] Could not read dx schedule stats: {dx_schedule_path} ({exc})")

            bd_stats = _birthdeath_stats_for_file(data_root, NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME, file_stem)
            drift_stats = _drift_stats_for_file(data_root, NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME, file_stem)

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME, file_stem, mapping_all)

            for m in HCV_NIS_METHODS:
                method_name = str(m.get("name", ""))
                nn_scales = HCV_NIS_DX_BLIND_NN_SCALE_FACTORS if "_nis_" in method_name else [0.3]
                for nn_scale in nn_scales:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * float(NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE))),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "dx_scale": float(NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE),
                            "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            "nn_scale_factor": float(nn_scale),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=float(NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE),
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = (
                            f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}"
                            f"_kp{int(HCV_NIS_KEEP_PERCENTILE)}"
                            f"_it{int(iters)}"
                            f"_nn{_fmt_token(float(nn_scale))}"
                        )
                        run_dir = out_root / "dx_birthdeath_drift" / file_stem / method_name / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime=NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME,
                            file_stem=file_stem,
                            method_name=method_name,
                            extras={
                                "dx_scale": float(NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE),
                                "dx_regime": str(NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_TAG),
                                "config_id": cfg_id,
                                "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                                "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                                "nn_scale_factor": float(nn_scale),
                                "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                                **dx_stats,
                                **bd_stats,
                                **drift_stats,
                            },
                        )
                        if dx_schedule_path.exists():
                            try:
                                shutil.copy2(dx_schedule_path, run_dir / dx_schedule_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy dx schedule CSV to run dir: {exc}")
                        if drift_schedule_path.exists():
                            try:
                                shutil.copy2(drift_schedule_path, run_dir / drift_schedule_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy drift schedule CSV to run dir: {exc}")
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_drift.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=[
                    "method",
                    "dx_scale",
                    "dx_regime",
                    "max_dist_factor",
                    "keep_percentile",
                    "motion_refine_iters",
                    "config_id",
                    "nn_scale_factor",
                    "sigma_init_mode",
                ],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    [
                        "method",
                        "dx_scale",
                        "dx_regime",
                        "max_dist_factor",
                        "keep_percentile",
                        "motion_refine_iters",
                        "nn_scale_factor",
                        "sigma_init_mode",
                        "metric",
                    ],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_regime", "nn_scale_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_drift_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_drift_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dataset_root: {data_root}",
                    f"regime: {NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_REGIME}",
                    f"dx_regime_tag: {NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_TAG}",
                    f"dx_nominal_scale_for_dispatch: {NODRIFT_VARYING_DX_BIRTHDEATH_DRIFT_NOMINAL_SCALE}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"nn_scale_factors: {HCV_NIS_DX_BLIND_NN_SCALE_FACTORS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "extra_aggregated_columns: active_tracks_mean,birth_events_total,death_events_total,birth_near_fraction,birth_near_dist_mean,drift_trans_median,drift_trans_q90,drift_rot_median_deg,drift_rot_q90_deg",
                    "dx_schedule_csv_pattern: <file_stem>_dx_schedule.csv (copied into each run dir when present)",
                    "drift_schedule_csv_pattern: <file_stem>_drift_schedule.csv (copied into each run dir when present)",
                    "nis_diag_csv_pattern: <file_stem>_<method>_nis_gate_diagnostics.csv",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH_STATIONARY
"""Run NIS dx-blind sweep on varying-dx + birth/death + stationary subset dataset.

What is tested:
- Same method family as HCV_NIS_DX_BLIND sweeps, with explicit stationary tracks
  (dx=0) mixed with varying-dx moving tracks and births/deaths.
"""

if RUN_HCV_NIS_DX_BLIND_SWEEP_BIRTHDEATH_STATIONARY:
    out_root = _results_root("hungarian_cv_variants_sweep_nis_dx_blind_birthdeath_stationary")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    data_root = DATA_ROOT_VARYING_DX_STATIONARY / NODRIFT_VARYING_DX_STATIONARY_TAG
    blind_reg_dir = data_root / "blind" / NODRIFT_VARYING_DX_STATIONARY_REGIME
    gt_reg_dir = data_root / "gt" / NODRIFT_VARYING_DX_STATIONARY_REGIME
    map_path = data_root / "unblinding.csv"
    if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
        print(f"[WARN] Missing varying-dx birthdeath+stationary dataset, skip: {data_root}")
    else:
        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES_BIRTHDEATH_STATIONARY))]

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == NODRIFT_VARYING_DX_STATIONARY_REGIME)
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            dx_schedule_path = data_root / "gt" / NODRIFT_VARYING_DX_STATIONARY_REGIME / "_groundtruth" / f"{file_stem}_dx_schedule.csv"
            dx_stats: dict[str, float] = {}
            if dx_schedule_path.exists():
                try:
                    dx_tab = pd.read_csv(dx_schedule_path)
                    dx_vals = pd.to_numeric(dx_tab.get("dx_scale_um", np.nan), errors="coerce").dropna().to_numpy(float)
                    if dx_vals.size:
                        dx_stats = {
                            "dx_actual_q10": float(np.quantile(dx_vals, 0.10)),
                            "dx_actual_q50": float(np.quantile(dx_vals, 0.50)),
                            "dx_actual_q90": float(np.quantile(dx_vals, 0.90)),
                            "dx_actual_mean": float(np.mean(dx_vals)),
                            "dx_actual_std": float(np.std(dx_vals)),
                        }
                except Exception as exc:
                    print(f"[WARN] Could not read dx schedule stats: {dx_schedule_path} ({exc})")

            bd_stats = _birthdeath_stats_for_file(data_root, NODRIFT_VARYING_DX_STATIONARY_REGIME, file_stem)
            stationary_stats = _stationary_stats_from_dx_schedule(dx_schedule_path)

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(data_root / "gt", NODRIFT_VARYING_DX_STATIONARY_REGIME, file_stem, mapping_all)

            for m in HCV_NIS_METHODS:
                method_name = str(m.get("name", ""))
                nn_scales = HCV_NIS_DX_BLIND_NN_SCALE_FACTORS if "_nis_" in method_name else [0.3]
                for nn_scale in nn_scales:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * float(NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE))),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "dx_scale": float(NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE),
                            "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            "nn_scale_factor": float(nn_scale),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=float(NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE),
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = (
                            f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}"
                            f"_kp{int(HCV_NIS_KEEP_PERCENTILE)}"
                            f"_it{int(iters)}"
                            f"_nn{_fmt_token(float(nn_scale))}"
                        )
                        run_dir = out_root / "dx_birthdeath_stationary" / file_stem / method_name / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime=NODRIFT_VARYING_DX_STATIONARY_REGIME,
                            file_stem=file_stem,
                            method_name=method_name,
                            extras={
                                "dx_scale": float(NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE),
                                "dx_regime": str(NODRIFT_VARYING_DX_STATIONARY_TAG),
                                "config_id": cfg_id,
                                "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                                "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                                "nn_scale_factor": float(nn_scale),
                                "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                                **dx_stats,
                                **bd_stats,
                                **stationary_stats,
                            },
                        )
                        if dx_schedule_path.exists():
                            try:
                                shutil.copy2(dx_schedule_path, run_dir / dx_schedule_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy dx schedule CSV to run dir: {exc}")
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_stationary.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=[
                    "method",
                    "dx_scale",
                    "dx_regime",
                    "max_dist_factor",
                    "keep_percentile",
                    "motion_refine_iters",
                    "config_id",
                    "nn_scale_factor",
                    "sigma_init_mode",
                ],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    [
                        "method",
                        "dx_scale",
                        "dx_regime",
                        "max_dist_factor",
                        "keep_percentile",
                        "motion_refine_iters",
                        "nn_scale_factor",
                        "sigma_init_mode",
                        "metric",
                    ],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "dx_regime", "nn_scale_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_stationary_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_hcv_nis_dx_blind_birthdeath_stationary_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dataset_root: {data_root}",
                    f"regime: {NODRIFT_VARYING_DX_STATIONARY_REGIME}",
                    f"dx_regime_tag: {NODRIFT_VARYING_DX_STATIONARY_TAG}",
                    f"dx_nominal_scale_for_dispatch: {NODRIFT_VARYING_DX_STATIONARY_NOMINAL_SCALE}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"nn_scale_factors: {HCV_NIS_DX_BLIND_NN_SCALE_FACTORS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "extra_aggregated_columns: active_tracks_mean,birth_events_total,death_events_total,birth_near_fraction,birth_near_dist_mean,stationary_fraction_actual,stationary_tracks_count",
                    "dx_schedule_csv_pattern: <file_stem>_dx_schedule.csv (copied into each run dir when present)",
                    "nis_diag_csv_pattern: <file_stem>_<method>_nis_gate_diagnostics.csv",
                    "plots:",
                    *[f"- {p}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

# %% HCV_NIS_DX_BLIND_SWEEP_DENSITY
"""Run NIS dx-blind sweep on density-scaled datasets.

What is tested:
- How association robustness changes as track density increases in fixed volume,
  while retaining varying dx, births/deaths, and stationary subset.
"""

if RUN_HCV_NIS_DX_BLIND_SWEEP_DENSITY:
    out_root = _results_root("hungarian_cv_variants_sweep_nis_dx_blind_density")
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for density_scale in DENSITY_SCALES:
        ds_token = _fmt_token(float(density_scale))
        data_root = DATA_ROOT_VARYING_DX_DENSITY / f"{NODRIFT_VARYING_DX_DENSITY_TAG}_ds{ds_token}"
        blind_reg_dir = data_root / "blind" / NODRIFT_VARYING_DX_DENSITY_REGIME
        gt_reg_dir = data_root / "gt" / NODRIFT_VARYING_DX_DENSITY_REGIME
        map_path = data_root / "unblinding.csv"
        if not map_path.exists() or not blind_reg_dir.exists() or not gt_reg_dir.exists():
            print(f"[WARN] Missing density dataset {density_scale}: {data_root}")
            continue

        mapping_all = pd.read_csv(map_path)
        blind_files = sorted(blind_reg_dir.glob("*_blind.csv"))[: int(max(1, HCV_MAX_FILES_DENSITY))]

        for blind_file in blind_files:
            file_stem = blind_file.name.replace("_blind.csv", "")
            map_sub = mapping_all[
                (mapping_all["Regime"].astype(str) == NODRIFT_VARYING_DX_DENSITY_REGIME)
                & (mapping_all["File"].astype(str) == file_stem)
            ].copy()
            if map_sub.empty:
                continue

            blind_frames = load_blind_frames(blind_file)
            gt_tracks = _gt_tracks_with_pointindex(
                data_root / "gt",
                NODRIFT_VARYING_DX_DENSITY_REGIME,
                file_stem,
                mapping_all,
            )
            density_stats = _density_stats_for_file(data_root, NODRIFT_VARYING_DX_DENSITY_REGIME, file_stem)
            dx_schedule_path = data_root / "gt" / NODRIFT_VARYING_DX_DENSITY_REGIME / "_groundtruth" / f"{file_stem}_dx_schedule.csv"

            for m in HCV_NIS_METHODS:
                method_name = str(m.get("name", ""))
                nn_scales = HCV_NIS_DX_BLIND_NN_SCALE_FACTORS if "_nis_" in method_name else [0.3]
                for nn_scale in nn_scales:
                    for iters in HCV_REFINE_ITERS:
                        gating_override = {
                            "max_dist": float(max(0.5, float(HCV_NIS_FIXED_MAX_DIST_FACTOR) * 1.0)),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                        }
                        params_override = {
                            "motion_refine_iters": int(iters),
                            "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                            "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                            "dx_scale": 1.0,
                            "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                            "nn_scale_factor": float(nn_scale),
                        }
                        assoc = _dispatch_association_method(
                            blind_frames,
                            m,
                            dx_scale=1.0,
                            gating_override=gating_override,
                            params_override=params_override,
                        )
                        cfg_id = (
                            f"md{_fmt_token(float(HCV_NIS_FIXED_MAX_DIST_FACTOR))}"
                            f"_kp{int(HCV_NIS_KEEP_PERCENTILE)}"
                            f"_it{int(iters)}"
                            f"_nn{_fmt_token(float(nn_scale))}"
                        )
                        run_dir = out_root / f"density_{ds_token}" / file_stem / method_name / cfg_id
                        row = _evaluate_and_save_run(
                            out_dir=run_dir,
                            assoc_result=assoc,
                            gt_tracks_df=gt_tracks,
                            mapping_df=map_sub,
                            blind_frames=blind_frames,
                            regime=NODRIFT_VARYING_DX_DENSITY_REGIME,
                            file_stem=file_stem,
                            method_name=method_name,
                            extras={
                                "density_scale": float(density_scale),
                                "dx_scale": float(density_scale),
                                "dx_regime": f"density_{ds_token}",
                                "config_id": cfg_id,
                                "max_dist_factor": float(HCV_NIS_FIXED_MAX_DIST_FACTOR),
                                "keep_percentile": float(HCV_NIS_KEEP_PERCENTILE),
                                "motion_refine_iters": int(iters),
                                "nn_scale_factor": float(nn_scale),
                                "sigma_init_mode": "nn_based" if "_nis_" in method_name else "dx",
                                "gate_radius_nominal": float(gating_override["max_dist"]),
                                **density_stats,
                            },
                        )
                        if dx_schedule_path.exists():
                            try:
                                shutil.copy2(dx_schedule_path, run_dir / dx_schedule_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy dx schedule CSV to run dir: {exc}")
                        density_stats_path = data_root / "gt" / NODRIFT_VARYING_DX_DENSITY_REGIME / "_groundtruth" / f"{file_stem}_density_stats.csv"
                        if density_stats_path.exists():
                            try:
                                shutil.copy2(density_stats_path, run_dir / density_stats_path.name)
                            except Exception as exc:
                                print(f"[WARN] Could not copy density stats CSV to run dir: {exc}")
                        rows.append(row)

    if rows:
        agg = pd.DataFrame(rows)
        agg_path = out_root / "aggregated_density.csv"
        agg.to_csv(agg_path, index=False)

        metric_cols = [
            c
            for c in [
                "link_f1_mean",
                "fragmentation_rate",
                "id_switches_mean_per_track",
                "rms_mean",
                "unmatched_rate_mean",
            ]
            if c in agg.columns
        ]
        if metric_cols:
            long = agg.melt(
                id_vars=[
                    "method",
                    "density_scale",
                    "dx_scale",
                    "dx_regime",
                    "max_dist_factor",
                    "keep_percentile",
                    "motion_refine_iters",
                    "config_id",
                    "nn_scale_factor",
                    "sigma_init_mode",
                ],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            stats = (
                long.groupby(
                    [
                        "method",
                        "density_scale",
                        "dx_scale",
                        "dx_regime",
                        "max_dist_factor",
                        "keep_percentile",
                        "motion_refine_iters",
                        "nn_scale_factor",
                        "sigma_init_mode",
                        "metric",
                    ],
                    as_index=False,
                )["value"]
                .agg(mean="mean", std="std", n="count")
                .sort_values(["metric", "method", "density_scale", "nn_scale_factor", "motion_refine_iters"])
                .reset_index(drop=True)
            )
            stats_path = out_root / "aggregated_density_stats.csv"
            stats.to_csv(stats_path, index=False)
        else:
            stats_path = out_root / "aggregated_density_stats.csv"
            pd.DataFrame().to_csv(stats_path, index=False)

        generated = _plot_hcv_variants_overview(out_root, agg)
        generated.extend(_plot_nis_gate_diagnostics(out_root, agg))
        generated.extend(_plot_density_sweep_summary(out_root, agg))
        generated = sorted(set(generated))

        (out_root / "README.txt").write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"dataset_root: {DATA_ROOT_VARYING_DX_DENSITY}",
                    f"regime: {NODRIFT_VARYING_DX_DENSITY_REGIME}",
                    f"density_scales: {DENSITY_SCALES}",
                    f"density_initial_tracks: {DENSITY_INITIAL_TRACKS}",
                    f"volume_size_um: {VOLUME_SIZE}",
                    f"max_dist_factor_fixed: {HCV_NIS_FIXED_MAX_DIST_FACTOR}",
                    f"keep_percentile_fixed: {HCV_NIS_KEEP_PERCENTILE}",
                    f"motion_refine_iters: {HCV_REFINE_ITERS}",
                    f"nn_scale_factors: {HCV_NIS_DX_BLIND_NN_SCALE_FACTORS}",
                    f"methods: {[m['name'] for m in HCV_NIS_METHODS]}",
                    f"aggregated_csv: {agg_path.name}",
                    f"aggregated_stats_csv: {stats_path.name}",
                    "extra_aggregated_columns: density_scale,active_tracks_mean,volume,density_mean,mean_nn_distance,median_r_eq,gate_radius_nominal",
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
