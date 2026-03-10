"""Parameter sweeps and benchmark utilities on reproducible synthetic data.

This script is intentionally organized in notebook-like cells (`# %% ...`) so users can
run it in VS Code/PyCharm cell mode or as a normal Python script. All synthetic parameter
study outputs are centralized under one root folder:

`Synthetic files parameter sweep`

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations
"""Import pipeline modules and provide fallback imports from the local repository."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re
import shutil
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
    from driftcorr.benchmarks import paper_regression, synthetic_generate, synthetic_regime_benchmark
    from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
    from driftcorr.method_local_communities import run_local_drift_correction_for_file
    from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
    from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks, build_step_transforms
    from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
    from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
    from driftcorr.method_spacetime_rbf_moe import apply_spacetime_rbf_moe_correction, fit_spacetime_rbf_moe_model
    from driftcorr.stationary import find_stationary_tracks
    from driftcorr.synthetic_experiments import (
        load_drift_field_for_imaris_file,
        load_groundtruth_for_imaris_file,
        save_groundtruth_evaluation_outputs,
    )
    from driftcorr.visualization import save_visual_inspection_plots
except ModuleNotFoundError as exc:
    # Fallback for cloned repo usage without installing driftcorr via pip.
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
        from driftcorr.benchmarks import paper_regression, synthetic_generate, synthetic_regime_benchmark
        from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
        from driftcorr.method_local_communities import run_local_drift_correction_for_file
        from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
        from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks, build_step_transforms
        from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
        from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
        from driftcorr.method_spacetime_rbf_moe import apply_spacetime_rbf_moe_correction, fit_spacetime_rbf_moe_model
        from driftcorr.stationary import find_stationary_tracks
        from driftcorr.synthetic_experiments import (
            load_drift_field_for_imaris_file,
            load_groundtruth_for_imaris_file,
            save_groundtruth_evaluation_outputs,
        )
        from driftcorr.visualization import save_visual_inspection_plots
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import 'driftcorr'. Install the package or run this script from inside the cloned project."
        ) from exc
# %% GLOBAL SETTINGS
"""Set shared paths, default options, and sweep constants used across all cells.

Directory contract
------------------
- `Synthetic files parameter sweep/_groundtruth`: baseline synthetic datasets (3 modes x 5 files).
- `Synthetic files parameter sweep/_groundtruth_large`: large regime datasets (3 regimes x 20 seeds).
- `Synthetic files parameter sweep/<study_tag>`: all benchmark and evaluation outputs.
"""

CURRENT_ANALYSIS_ROOT = PROJECT_ROOT / "Synthetic files parameter sweep"

GROUNDTRUTH_ROOT = CURRENT_ANALYSIS_ROOT / "_groundtruth"
GROUNDTRUTH_LARGE_ROOT = CURRENT_ANALYSIS_ROOT / "_groundtruth_large"
RESULTS_ROOT = CURRENT_ANALYSIS_ROOT
BASELINE_MODES = ["smooth", "piecewise_smooth", "piecewise_abrupt"]
BASELINE_FILES_PER_MODE = 5
LARGE_SEED_START = 42
LARGE_N_SEEDS = 20
LARGE_FOLDER_MODE_TOKEN = {
    "smooth": "smooth_spacetime",
    "piecewise_smooth": "piecewise_smooth",
    "piecewise_abrupt": "piecewise_abrupt",
}

DEFAULT_METHODS = ["rigid", "local", "rbf", "st_rbf"]

# Core method options in one place.
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
    "st_robust_mode": "sample_trim",
    "st_track_trim_iters": 4,
    "st_per_axis_scaling": 1,
    "st_linear_interactions": 0,
    "st_moe_k": 3,
    "st_moe_k_list": [2, 3, 5],
    "st_moe_blend_sigma_um": 140.0,
    "st_moe_cluster_mode": "track_anchor",
    "st_moe_drift_cap_factor": 3.0,
}

# ST-RBF grid/stationary subset benchmark parameters.
ES_VALUES = [1.0, 2.0, 4.0]
ET_VALUES = [0.4, 0.8, 1.6]
STATIONARY_AB_CANDIDATES = [
    {"name": "fast", "ell_space": 4.0, "ell_time": 0.4},
    {"name": "smooth", "ell_space": 1.0, "ell_time": 1.6},
]
GRID_DIR_PATTERN = re.compile(r"^st_rbf_grid_es(?P<es>[-+]?\d*\.?\d+)_et(?P<et>[-+]?\d*\.?\d+)$")
STATIONARY_AB_DIR_PATTERN = re.compile(
    r"^st_rbf_stationary_ab_es(?P<es>[-+]?\d*\.?\d+)_et(?P<et>[-+]?\d*\.?\d+)_subset(?P<subset>[01])$"
)

# %% FUNCTIONS
"""Define reusable functions for running methods, postprocessing, aggregation, and plotting."""

def run_methods_for_dataframe(
    df: pd.DataFrame,
    out_prefix: Path,
    methods: list[str],
    opts: dict,
    use_stationary_subset: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run selected methods on one dataframe and write method-specific artifacts."""
    df = ensure_numeric_tracks(df)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    stationary_ids = None
    if use_stationary_subset:
        sel = find_stationary_tracks(df)
        sel.table.to_csv(str(out_prefix) + "_stationary_subset.csv", index=False)
        if len(sel.track_ids) >= int(opts["min_pairs"]):
            stationary_ids = sorted(sel.track_ids)

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
        pd.DataFrame([st_info]).to_csv(str(out_prefix) + "_st_rbf_fit_info.csv", index=False)
        df_st.to_csv(str(out_prefix) + "_positions_corrected_st_rbf.csv", index=False)
        corr_by_method["st_rbf"] = df_st

    if "st_rbf_moe" in methods:
        k_list = opts.get("st_moe_k_list", [opts.get("st_moe_k", 3)])
        if not isinstance(k_list, list):
            k_list = [int(k_list)]
        for i, k in enumerate([int(v) for v in k_list if int(v) > 0]):
            moe_model, expert_table = fit_spacetime_rbf_moe_model(
                df,
                n_experts=int(k),
                blend_sigma_um=opts["st_moe_blend_sigma_um"],
                cluster_mode=opts["st_moe_cluster_mode"],
                seed=int(opts["seed"] + i * 13),
                reference_time=opts["reference_time"],
                fit_track_ids=stationary_ids,
                m_centers=opts["st_m_centers"],
                ell_space=opts["st_ell_space"],
                ell_time=opts["st_ell_time"],
                ell_x=opts["st_ell_x"],
                ell_y=opts["st_ell_y"],
                ell_z=opts["st_ell_z"],
                lam=opts.get("st_lam", opts["lam"]),
                robust_keep_fraction=opts["st_robust_keep_fraction"],
                robust_iters=opts["st_robust_iters"],
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
                drift_cap_factor=opts["st_moe_drift_cap_factor"],
            )
            df_moe = apply_spacetime_rbf_moe_correction(df, moe_model)
            name = f"st_rbf_moe_k{int(k)}"
            expert_table.to_csv(str(out_prefix) + f"_{name}_experts.csv", index=False)
            df_moe.to_csv(str(out_prefix) + f"_positions_corrected_{name}.csv", index=False)
            corr_by_method[name] = df_moe

    return corr_by_method

def postprocess_outputs(
    df_orig: pd.DataFrame,
    imaris_file: Path,
    out_prefix: Path,
    corr_by_method: dict[str, pd.DataFrame]) -> None:
    """
    Create standard evaluation + visual plots + optional groundtruth evaluation.
    
    DEBUg: 
    df_orig = df_syn
    imaris_file = file_path
    out_prefix = out_prefix
    corr_by_method = corr
    
    """
    summary_df, details = run_all_methods_and_compare(df_orig, corr_by_method)
    save_evaluation_outputs(out_prefix, summary_df, details)

    gt = load_groundtruth_for_imaris_file(imaris_file)
    drift_gt = load_drift_field_for_imaris_file(imaris_file)
    for method_name, df_corr in corr_by_method.items():
        save_visual_inspection_plots(df_orig, df_corr, out_prefix, method_name, df_groundtruth=gt)

    if gt is not None:
        save_groundtruth_evaluation_outputs(out_prefix, gt, corr_by_method, drift_field_groundtruth=drift_gt)


# PARAMETER SWEEP DATASET + OUTPUT HELPERS:
"""Provide dataset/output-root helpers and generic utility functions for sweep result handling."""

FAIRNESS_SYNTHETIC_N_SEEDS = 10
RIGID_KEEP_FRACTIONS = [0.5, 0.7, 0.85, 0.95]
RIGID_INLIER_ITERS = [5, 15, 30]
LOCAL_K_NEIGHBORS = [8, 12, 20, 30]
LOCAL_ELL_FIELD = [80.0, 120.0, 180.0, 240.0]

def results_root(tag: str) -> Path:
    """Return centralized study output root under the main analysis folder."""
    safe = str(tag).strip().replace(" ", "_").replace("-", "_")
    return RESULTS_ROOT / safe

def parse_mode_from_filename(stem: str) -> str | None:
    """Parse synthetic mode token from a mode-prefixed file stem."""
    s = str(stem)
    for mode in BASELINE_MODES:
        prefix = f"{mode}_"
        if s.startswith(prefix):
            return mode
    return None

def enumerate_groundtruth_files(root: Path, modes: list[str] | None = None) -> dict[str, list[Path]]:
    """Return synthetic files grouped by mode from a shared groundtruth root."""
    mode_list = BASELINE_MODES if modes is None else list(modes)
    grouped: dict[str, list[Path]] = {m: [] for m in mode_list}
    for p in list_imaris_xls_files(root):
        mode = parse_mode_from_filename(p.stem)
        if mode in grouped:
            grouped[mode].append(p)
    for m in grouped:
        grouped[m] = sorted(grouped[m])
    return grouped

def list_baseline_imaris_files() -> list[Path]:
    """List all baseline synthetic files across all required modes."""
    ensure_baseline_groundtruth()
    grouped = enumerate_groundtruth_files(GROUNDTRUTH_ROOT)
    out: list[Path] = []
    for mode in BASELINE_MODES:
        out.extend(grouped.get(mode, []))
    return sorted(out)

def _ensure_mode_generated(mode: str) -> None:
    """Generate one missing baseline mode and move data+sidecars with mode prefix."""
    tmp_dir = GROUNDTRUTH_ROOT / f"__tmp_generate_{mode}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        synthetic_generate(
            synthetic_dir=str(tmp_dir),
            synthetic_seed=42,
            synthetic_n_files=5,
            synthetic_n_tracks=80,
            synthetic_n_times=20,
            synthetic_stationary_fraction=0.45,
            synthetic_migrating_fraction=0.35,
            synthetic_temporal_smoothness=0.9,
            synthetic_local_mode=mode,
        )
        generated = sorted(list_imaris_xls_files(tmp_dir))
        tmp_gt_dir = tmp_dir / "_groundtruth"
        tgt_gt_dir = GROUNDTRUTH_ROOT / "_groundtruth"
        tgt_gt_dir.mkdir(parents=True, exist_ok=True)
        for src in generated:
            pref_stem = f"{mode}_{src.stem}"
            dst = GROUNDTRUTH_ROOT / f"{pref_stem}{src.suffix}"
            if not dst.exists():
                shutil.move(str(src), str(dst))
            for suf in ["_groundtruth.csv", "_drift_field_groundtruth.csv", "_meta.csv"]:
                src_side = tmp_gt_dir / f"{src.stem}{suf}"
                dst_side = tgt_gt_dir / f"{pref_stem}{suf}"
                if src_side.exists() and not dst_side.exists():
                    shutil.move(str(src_side), str(dst_side))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _has_required_sidecars(imaris_file: Path) -> bool:
    """Check whether required GT sidecars exist for one synthetic IMARIS file."""
    gt_dir = imaris_file.parent / "_groundtruth"
    stem = imaris_file.stem
    req = [
        gt_dir / f"{stem}_groundtruth.csv",
        gt_dir / f"{stem}_drift_field_groundtruth.csv",
    ]
    return all(p.exists() for p in req)

def ensure_baseline_groundtruth() -> Path:
    """Ensure `_groundtruth` contains baseline synthetic files for all required modes."""
    GROUNDTRUTH_ROOT.mkdir(parents=True, exist_ok=True)
    grouped = enumerate_groundtruth_files(GROUNDTRUTH_ROOT)
    missing_modes: list[str] = []
    for mode in BASELINE_MODES:
        files = grouped.get(mode, [])
        if len(files) < BASELINE_FILES_PER_MODE:
            missing_modes.append(mode)
            continue
        if not all(_has_required_sidecars(p) for p in files):
            missing_modes.append(mode)
    for mode in missing_modes:
        print(f"[INFO] Generating missing baseline mode '{mode}' in {GROUNDTRUTH_ROOT}")
        _ensure_mode_generated(mode)
    # Recheck and fail fast if incomplete.
    grouped_after = enumerate_groundtruth_files(GROUNDTRUTH_ROOT)
    for mode in BASELINE_MODES:
        files = grouped_after.get(mode, [])
        n = len(files)
        if n < BASELINE_FILES_PER_MODE:
            raise RuntimeError(
                f"Incomplete baseline groundtruth for mode '{mode}': found {n}, expected >= {BASELINE_FILES_PER_MODE}."
            )
        missing_sidecars = [p.name for p in files if not _has_required_sidecars(p)]
        if missing_sidecars:
            raise RuntimeError(
                f"Missing GT sidecars for mode '{mode}' files: {missing_sidecars[:5]}"
            )
    return GROUNDTRUTH_ROOT

def _expected_large_folder_names() -> list[str]:
    names: list[str] = []
    for mode in BASELINE_MODES:
        token = LARGE_FOLDER_MODE_TOKEN[mode]
        for seed in range(LARGE_SEED_START, LARGE_SEED_START + LARGE_N_SEEDS):
            names.append(f"{token}_seed_{seed:03d}")
    return sorted(names)

def list_large_regime_folders() -> list[Path]:
    """List existing large regime folders under `_groundtruth_large`."""
    if not GROUNDTRUTH_LARGE_ROOT.exists():
        return []
    return sorted([p for p in GROUNDTRUTH_LARGE_ROOT.iterdir() if p.is_dir()])

def ensure_large_groundtruth_regimes() -> Path:
    """Ensure `_groundtruth_large` contains deterministic 3-regime x 20-seed datasets."""
    GROUNDTRUTH_LARGE_ROOT.mkdir(parents=True, exist_ok=True)
    expected = _expected_large_folder_names()
    existing = {p.name for p in list_large_regime_folders()}
    missing = [name for name in expected if name not in existing]

    for folder_name in missing:
        mode = None
        for m, token in LARGE_FOLDER_MODE_TOKEN.items():
            if folder_name.startswith(f"{token}_seed_"):
                mode = m
                break
        if mode is None:
            continue
        seed = int(folder_name.rsplit("_", 1)[-1])
        out_dir = GROUNDTRUTH_LARGE_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Generating large groundtruth folder: {folder_name}")
        synthetic_generate(
            synthetic_dir=str(out_dir),
            synthetic_seed=seed,
            synthetic_n_files=5,
            synthetic_n_tracks=80,
            synthetic_n_times=20,
            synthetic_stationary_fraction=0.45,
            synthetic_migrating_fraction=0.35,
            synthetic_temporal_smoothness=0.9,
            synthetic_local_mode=mode,
        )

    # Strict completeness check.
    existing_after = {p.name for p in list_large_regime_folders()}
    missing_after = [name for name in expected if name not in existing_after]
    if missing_after:
        raise RuntimeError(f"Missing large-groundtruth folders after generation: {missing_after[:5]}")
    return GROUNDTRUTH_LARGE_ROOT

def _discover_method_timeseries_files(experiment_root: Path, method: str) -> dict[str, Path]:
    """Find `*_gt_rms_over_time_<method>.csv` files and map them by file stem."""
    out: dict[str, Path] = {}
    if not experiment_root.exists():
        return out

    suffix = f"_gt_rms_over_time_{method}.csv"

    # Preferred layout: <experiment_root>/<file_stem>/<file_stem>_gt_rms_over_time_<method>.csv
    for d in sorted([p for p in experiment_root.iterdir() if p.is_dir()]):
        cand = d / f"{d.name}{suffix}"
        if cand.exists():
            out[d.name] = cand

    # One nested level fallback: <experiment_root>/<run_id>/<file_stem>/<file_stem>_gt_rms_over_time_<method>.csv
    for d in sorted([p for p in experiment_root.iterdir() if p.is_dir()]):
        for sd in sorted([p for p in d.iterdir() if p.is_dir()]):
            cand = sd / f"{sd.name}{suffix}"
            if cand.exists() and sd.name not in out:
                out[sd.name] = cand

    if out:
        return out

    # Recursive fallback for arbitrary layouts.
    tmp: dict[str, Path] = {}
    for p in experiment_root.rglob(f"*{suffix}"):
        name = p.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        prev = tmp.get(stem)
        if prev is None or len(p.parts) < len(prev.parts):
            tmp[stem] = p
    return tmp

def _load_rms_timeseries(path: Path) -> pd.DataFrame | None:
    """Load one RMS-over-time CSV and standardize columns to `Time`, `rms_error`."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None
    if df.empty:
        return None

    lower = {c.lower(): c for c in df.columns}
    time_col = lower.get("time")
    if time_col is None:
        time_candidates = [c for c in df.columns if "time" in c.lower()]
        time_col = time_candidates[0] if time_candidates else None
    if time_col is None:
        print(f"[WARN] No time column found in {path}")
        return None

    preferred_vals = ["rms_error", "rms", "rms error"]
    val_col = None
    for key in preferred_vals:
        if key in lower:
            val_col = lower[key]
            break
    if val_col is None:
        numeric_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            print(f"[WARN] No RMS value column found in {path}")
            return None
        val_col = numeric_cols[0]

    out = pd.DataFrame(
        {
            "Time": pd.to_numeric(df[time_col], errors="coerce"),
            "rms_error": pd.to_numeric(df[val_col], errors="coerce"),
        }
    ).dropna(subset=["Time", "rms_error"])
    if out.empty:
        return None
    return out.sort_values("Time").reset_index(drop=True)

def _load_method_timeseries(experiment_root: Path, method: str) -> dict[str, pd.DataFrame]:
    files = _discover_method_timeseries_files(experiment_root, method)
    out: dict[str, pd.DataFrame] = {}
    for stem, path in files.items():
        ts = _load_rms_timeseries(path)
        if ts is not None and not ts.empty:
            out[stem] = ts
    return out

def _aggregate_timeseries(series_by_file: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for stem, ts in series_by_file.items():
        tmp = ts.copy()
        tmp["file_stem"] = stem
        rows.append(tmp[["Time", "rms_error", "file_stem"]])
    if not rows:
        return pd.DataFrame(columns=["Time", "mean", "std", "n"])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (
        all_df.groupby("Time", as_index=False)["rms_error"]
        .agg(mean="mean", std="std", n="count")
        .sort_values("Time")
        .reset_index(drop=True)
    )
    agg["std"] = pd.to_numeric(agg["std"], errors="coerce").fillna(0.0)
    return agg

def _plot_timeseries_comparison(
    label_to_series: dict[str, dict[str, pd.DataFrame]],
    out_path: Path,
    title: str,
) -> bool:
    """Plot aggregated mean±std RMS-over-time curves for multiple labels."""
    agg_by_label: dict[str, pd.DataFrame] = {}
    for label, m in label_to_series.items():
        agg = _aggregate_timeseries(m)
        if not agg.empty:
            agg_by_label[label] = agg
    if not agg_by_label:
        print(f"[WARN] No timeseries data available for {out_path.name}")
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for label, agg in agg_by_label.items():
        x = agg["Time"].to_numpy(float)
        y = agg["mean"].to_numpy(float)
        s = agg["std"].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
        ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_xlabel("Time")
    ax.set_ylabel("RMS error (um)")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _plot_timeseries_per_file(
    label_to_series: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
    filename_prefix: str,
) -> list[Path]:
    """Plot per-file RMS-over-time curves for labels where data exists."""
    generated: list[Path] = []
    stems: set[str] = set()
    for m in label_to_series.values():
        stems |= set(m.keys())
    for stem in sorted(stems):
        present = {label: m[stem] for label, m in label_to_series.items() if stem in m}
        if len(present) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for label, ts in present.items():
            ax.plot(ts["Time"].to_numpy(float), ts["rms_error"].to_numpy(float), marker="o", linewidth=1.8, label=label)
        ax.set_xlabel("Time")
        ax.set_ylabel("RMS error (um)")
        ax.set_title(f"{filename_prefix}: {stem}")
        ax.set_ylim(bottom=0.0)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"{filename_prefix}__{stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        generated.append(out_path)
    return generated

def _discover_gt_metrics_summary_files(experiment_root: Path) -> dict[str, Path]:
    """Find `*_gt_metrics_summary.csv` files and map them by file stem."""
    out: dict[str, Path] = {}
    if not experiment_root.exists():
        return out
    suffix = "_gt_metrics_summary.csv"
    for d in sorted([p for p in experiment_root.iterdir() if p.is_dir()]):
        cand = d / f"{d.name}{suffix}"
        if cand.exists():
            out[d.name] = cand
    for d in sorted([p for p in experiment_root.iterdir() if p.is_dir()]):
        for sd in sorted([p for p in d.iterdir() if p.is_dir()]):
            cand = sd / f"{sd.name}{suffix}"
            if cand.exists() and sd.name not in out:
                out[sd.name] = cand
    if out:
        return out
    tmp: dict[str, Path] = {}
    for p in experiment_root.rglob(f"*{suffix}"):
        name = p.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        prev = tmp.get(stem)
        if prev is None or len(p.parts) < len(prev.parts):
            tmp[stem] = p
    return tmp

def _collect_scalar_metric(experiment_root: Path, method: str, metric_col: str = "overall_rms_error") -> np.ndarray:
    """Collect a scalar metric for one method from `*_gt_metrics_summary.csv` files."""
    vals: list[float] = []
    for _, path in _discover_gt_metrics_summary_files(experiment_root).items():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if metric_col not in df.columns:
            continue
        if "method" in df.columns:
            sub = df[df["method"].astype(str) == method]
            if sub.empty:
                continue
            v = pd.to_numeric(sub.iloc[0][metric_col], errors="coerce")
        else:
            v = pd.to_numeric(df.iloc[0][metric_col], errors="coerce")
        if np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, dtype=float)

def _plot_scalar_summary_bar(
    label_to_values: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    y_label: str = "overall_rms_error",
) -> bool:
    plot_labels = [k for k, v in label_to_values.items() if np.asarray(v).size > 0]
    if not plot_labels:
        print(f"[WARN] Scalar summary bar skipped ({out_path.name}): no values available.")
        return False

    means = [float(np.mean(label_to_values[k])) for k in plot_labels]
    stds = [float(np.std(label_to_values[k], ddof=0)) for k in plot_labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(plot_labels), dtype=float)
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=20, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("nan"), float("nan")
    vv = vals[mask]
    return float(np.mean(vv)), float(np.std(vv, ddof=0))

def _find_regime_metric_csvs(out_dir: Path) -> dict[str, Path]:
    """Map regime to a per-seed metrics CSV for one benchmark output directory."""
    mapping: dict[str, Path] = {}
    if not out_dir.exists():
        return mapping

    long_files = sorted(out_dir.glob("*_seed_metrics_long.csv"))
    for path in long_files:
        regime = path.name.replace("_seed_metrics_long.csv", "")
        if regime:
            mapping[regime] = path
    if mapping:
        return mapping

    runs_csv = out_dir / "synthetic_regime_runs.csv"
    if runs_csv.exists():
        try:
            df = pd.read_csv(runs_csv)
        except Exception:
            return mapping
        if "regime" in df.columns:
            for regime in sorted(df["regime"].dropna().astype(str).unique().tolist()):
                mapping[regime] = runs_csv
    return mapping

def _load_and_standardize_seed_metrics(csv_path: Path) -> pd.DataFrame:
    """Load a metrics CSV and standardize to one row per (regime, seed, method)."""
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return pd.DataFrame(columns=["regime", "seed", "method", "rms_all", "rms_stationary", "rms_migrating", "drift_bias_abs"])

    if {"metric_name", "metric_value"}.issubset(raw.columns):
        wide = (
            raw.pivot_table(
                index=["regime", "seed", "method"],
                columns="metric_name",
                values="metric_value",
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        src = wide
    else:
        src = raw.copy()

    needed = {"regime", "seed", "method"}
    if not needed.issubset(src.columns):
        return pd.DataFrame(columns=["regime", "seed", "method", "rms_all", "rms_stationary", "rms_migrating", "drift_bias_abs"])

    def _pick(cands: list[str]) -> str | None:
        for c in cands:
            if c in src.columns:
                return c
        return None

    col_rms_all = _pick(["overall_rms_error", "rms_error_overall", "rms_all", "overall_rms"])
    col_rms_stat = _pick(["overall_rms_error_stationary", "rms_error_stationary", "rms_stationary"])
    col_rms_migr = _pick(["overall_rms_error_migrating", "rms_error_migrating", "rms_migrating"])
    col_bias = _pick(["drift_bias_abs", "drift_bias_abs_mean_over_time", "drift_bias_mean_abs", "drift_bias_mag_abs", "drift_bias"])
    if col_rms_all is None:
        return pd.DataFrame(columns=["regime", "seed", "method", "rms_all", "rms_stationary", "rms_migrating", "drift_bias_abs"])

    out = src[["regime", "seed", "method"]].copy()
    out["rms_all"] = pd.to_numeric(src[col_rms_all], errors="coerce")
    out["rms_stationary"] = pd.to_numeric(src[col_rms_stat], errors="coerce") if col_rms_stat is not None else np.nan
    out["rms_migrating"] = pd.to_numeric(src[col_rms_migr], errors="coerce") if col_rms_migr is not None else np.nan
    out["drift_bias_abs"] = pd.to_numeric(src[col_bias], errors="coerce") if col_bias is not None else np.nan
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    return out

def _mean_drift_bias_from_per_seed_files(run_dir: Path, regime: str, seed: int, method: str) -> float:
    """Fallback drift-bias scalar from per-time bias files for one seed and regime."""
    reg_dir = run_dir / f"{regime}_seed_{int(seed):03d}"
    if not reg_dir.exists():
        return float("nan")
    files = sorted(reg_dir.glob(f"*/*_gt_drift_bias_over_time_{method}.csv"))
    if not files:
        return float("nan")
    per_file: list[float] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "drift_bias_norm" not in df.columns:
            continue
        m, _ = _safe_mean_std(pd.to_numeric(df["drift_bias_norm"], errors="coerce").to_numpy(float))
        if np.isfinite(m):
            per_file.append(float(m))
    if not per_file:
        return float("nan")
    m, _ = _safe_mean_std(np.asarray(per_file, dtype=float))
    return m

def _has_completed_regime_run(out_dir: Path, method: str) -> bool:
    runs_csv = out_dir / "synthetic_regime_runs.csv"
    if not runs_csv.exists():
        return False
    try:
        df = pd.read_csv(runs_csv)
    except Exception:
        return False
    if df.empty:
        return False
    if "method" in df.columns:
        return method in df["method"].astype(str).unique().tolist()
    return True

def _fmt_float_token(val: float, decimals: int = 2) -> str:
    return f"{float(val):.{decimals}f}".replace(".", "p")

def _parse_rigid_config_name(name: str) -> tuple[float, int] | None:
    m = re.match(r"^kf(?P<kf>[0-9]+p[0-9]+)_inlier(?P<it>[0-9]+)$", name)
    if m is None:
        return None
    keep_fraction = float(m.group("kf").replace("p", "."))
    inlier_iters = int(m.group("it"))
    return keep_fraction, inlier_iters

def _parse_local_config_name(name: str) -> tuple[int, float] | None:
    m = re.match(r"^k(?P<k>[0-9]+)_field(?P<f>[0-9]+(?:p[0-9]+)?)$", name)
    if m is None:
        return None
    k_neighbors = int(m.group("k"))
    field_tok = m.group("f").replace("p", ".")
    return k_neighbors, float(field_tok)

def _aggregate_fairness_sweep(sweep_root: Path, method: str, sweep_kind: str) -> pd.DataFrame:
    """Aggregate one fairness sweep folder into tidy regime-level summary rows."""
    rows: list[dict[str, float | str | int]] = []
    if not sweep_root.exists():
        return pd.DataFrame()

    config_dirs = sorted([p for p in sweep_root.iterdir() if p.is_dir()])
    for cfg_dir in config_dirs:
        if sweep_kind == "rigid":
            parsed = _parse_rigid_config_name(cfg_dir.name)
            if parsed is None:
                continue
            keep_fraction, inlier_iters = parsed
        else:
            parsed2 = _parse_local_config_name(cfg_dir.name)
            if parsed2 is None:
                continue
            k_neighbors, ell_um_field = parsed2

        regime_map = _find_regime_metric_csvs(cfg_dir)
        for regime, csv_path in regime_map.items():
            metrics = _load_and_standardize_seed_metrics(csv_path)
            if metrics.empty:
                continue
            subset = metrics[
                (metrics["method"].astype(str) == str(method))
                & (metrics["regime"].astype(str) == str(regime))
            ].copy()
            if subset.empty:
                continue

            need_bias = ~np.isfinite(pd.to_numeric(subset["drift_bias_abs"], errors="coerce").to_numpy(float))
            if bool(np.any(need_bias)):
                for idx in subset.index[need_bias]:
                    seed_val = pd.to_numeric(subset.loc[idx, "seed"], errors="coerce")
                    if not np.isfinite(seed_val):
                        continue
                    subset.loc[idx, "drift_bias_abs"] = _mean_drift_bias_from_per_seed_files(
                        run_dir=cfg_dir,
                        regime=str(regime),
                        seed=int(seed_val),
                        method=method,
                    )

            n_seeds = int(pd.to_numeric(subset["seed"], errors="coerce").dropna().nunique())
            mean_all, std_all = _safe_mean_std(subset["rms_all"].to_numpy(float))
            mean_stat, std_stat = _safe_mean_std(subset["rms_stationary"].to_numpy(float))
            mean_migr, std_migr = _safe_mean_std(subset["rms_migrating"].to_numpy(float))
            mean_bias, std_bias = _safe_mean_std(subset["drift_bias_abs"].to_numpy(float))

            row: dict[str, float | str | int] = {
                "regime": str(regime),
                "method": str(method),
                "n_seeds_found": n_seeds,
                "mean_rms_all": mean_all,
                "std_rms_all": std_all,
                "mean_rms_stationary": mean_stat,
                "std_rms_stationary": std_stat,
                "mean_rms_migrating": mean_migr,
                "std_rms_migrating": std_migr,
                "mean_drift_bias_abs": mean_bias,
                "std_drift_bias_abs": std_bias,
            }
            if sweep_kind == "rigid":
                row["keep_fraction"] = float(keep_fraction)
                row["inlier_reestimate_iters"] = int(inlier_iters)
            else:
                row["k_neighbors"] = int(k_neighbors)
                row["ell_um_field"] = float(ell_um_field)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if sweep_kind == "rigid":
        cols = [
            "regime",
            "method",
            "keep_fraction",
            "inlier_reestimate_iters",
            "n_seeds_found",
            "mean_rms_all",
            "std_rms_all",
            "mean_rms_stationary",
            "std_rms_stationary",
            "mean_rms_migrating",
            "std_rms_migrating",
            "mean_drift_bias_abs",
            "std_drift_bias_abs",
        ]
        out = out.drop_duplicates(subset=["regime", "method", "keep_fraction", "inlier_reestimate_iters"], keep="last")
        out = out[cols].sort_values(["regime", "keep_fraction", "inlier_reestimate_iters"]).reset_index(drop=True)
    else:
        cols = [
            "regime",
            "method",
            "k_neighbors",
            "ell_um_field",
            "n_seeds_found",
            "mean_rms_all",
            "std_rms_all",
            "mean_rms_stationary",
            "std_rms_stationary",
            "mean_rms_migrating",
            "std_rms_migrating",
            "mean_drift_bias_abs",
            "std_drift_bias_abs",
        ]
        out = out.drop_duplicates(subset=["regime", "method", "k_neighbors", "ell_um_field"], keep="last")
        out = out[cols].sort_values(["regime", "k_neighbors", "ell_um_field"]).reset_index(drop=True)
    return out

def _plot_lines_by_regime(
    df: pd.DataFrame,
    x_col: str,
    line_col: str,
    y_col: str,
    out_path_base: Path,
    title_prefix: str,
    x_label: str,
    legend_label: str,
) -> list[Path]:
    generated: list[Path] = []
    if df.empty:
        return generated
    regimes = sorted(df["regime"].astype(str).unique().tolist())
    if not regimes:
        return generated
    n = len(regimes)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
    for i, regime in enumerate(regimes):
        ax = axes[i // ncols][i % ncols]
        cur = df[df["regime"].astype(str) == regime].copy()
        groups = sorted(cur[line_col].dropna().unique().tolist())
        for gv in groups:
            gg = cur[cur[line_col] == gv].sort_values(x_col)
            ax.plot(
                pd.to_numeric(gg[x_col], errors="coerce").to_numpy(float),
                pd.to_numeric(gg[y_col], errors="coerce").to_numpy(float),
                marker="o",
                linewidth=1.8,
                label=f"{legend_label}={gv:g}",
            )
        ax.set_title(regime)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_col)
        ax.set_ylim(bottom=0.0)
        ax.legend()
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    png = out_path_base.with_suffix(".png")
    pdf = out_path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    generated.extend([png, pdf])
    return generated

def _plot_heatmaps_by_regime(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    out_path_base: Path,
    title_prefix: str,
    x_label: str,
    y_label: str,
) -> list[Path]:
    generated: list[Path] = []
    if df.empty:
        return generated
    regimes = sorted(df["regime"].astype(str).unique().tolist())
    if not regimes:
        return generated
    n = len(regimes)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    for i, regime in enumerate(regimes):
        ax = axes[i // ncols][i % ncols]
        cur = df[df["regime"].astype(str) == regime].copy()
        x_vals = sorted(pd.to_numeric(cur[x_col], errors="coerce").dropna().unique().tolist())
        y_vals = sorted(pd.to_numeric(cur[y_col], errors="coerce").dropna().unique().tolist())
        if not x_vals or not y_vals:
            ax.axis("off")
            continue
        mat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
        x_lookup = {float(v): idx for idx, v in enumerate(x_vals)}
        y_lookup = {float(v): idx for idx, v in enumerate(y_vals)}
        for _, r in cur.iterrows():
            xv = float(pd.to_numeric(r[x_col], errors="coerce"))
            yv = float(pd.to_numeric(r[y_col], errors="coerce"))
            vv = float(pd.to_numeric(r[value_col], errors="coerce"))
            if np.isfinite(xv) and np.isfinite(yv) and xv in x_lookup and yv in y_lookup:
                mat[y_lookup[yv], x_lookup[xv]] = vv
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#f0f0f0")
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels([f"{v:g}" for v in x_vals])
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_yticklabels([f"{v:g}" for v in y_vals])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(regime)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    png = out_path_base.with_suffix(".png")
    pdf = out_path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    generated.extend([png, pdf])
    return generated

def _write_fairness_sweep_readme(
    sweep_root: Path,
    synthetic_dir: Path,
    aggregated_csv: Path,
    generated_plots: list[Path],
    sweep_desc: str,
    grid_desc: str,
    n_regimes: int,
) -> Path:
    readme = sweep_root / "README.txt"
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"synthetic_dataset: {synthetic_dir}",
        f"sweep: {sweep_desc}",
        f"grid: {grid_desc}",
        f"regimes_aggregated: {n_regimes}",
        f"aggregated_csv: {aggregated_csv}",
        "plots:",
    ]
    lines.extend([f"- {p.name}" for p in generated_plots])
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


# ST-RBF GRID + STATIONARY A/B HELPERS:
"""Provide ST-RBF-specific helper functions for grid aggregation, A/B tests, and summary plotting."""

def _aggregate_st_rbf_grid(sweep_root: Path) -> pd.DataFrame:
    """Aggregate ST-RBF grid outputs into one tidy table."""
    if not sweep_root.exists():
        raise FileNotFoundError(f"Output root not found: {sweep_root}")

    rows: list[dict[str, float | str | int]] = []
    grid_dirs = sorted([p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("st_rbf_grid_es")])
    expected_names = {f"st_rbf_grid_es{es}_et{et}" for es in ES_VALUES for et in ET_VALUES}
    existing_names = {p.name for p in grid_dirs}
    for name in sorted(expected_names - existing_names):
        print(f"[WARN] Expected grid folder not found (skip): {sweep_root / name}")
    if not grid_dirs:
        raise FileNotFoundError(f"No grid folders found in {sweep_root}")

    for grid_dir in grid_dirs:
        match = GRID_DIR_PATTERN.match(grid_dir.name)
        if match is None:
            print(f"[WARN] Skip unexpected folder name: {grid_dir.name}")
            continue
        ell_space = float(match.group("es"))
        ell_time = float(match.group("et"))

        regime_to_csv = _find_regime_metric_csvs(grid_dir)
        for regime, csv_path in regime_to_csv.items():
            metrics = _load_and_standardize_seed_metrics(csv_path)
            if metrics.empty:
                continue
            subset = metrics[
                (metrics["method"].astype(str) == "st_rbf")
                & (metrics["regime"].astype(str) == str(regime))
            ].copy()
            if subset.empty:
                continue

            need_bias = ~np.isfinite(pd.to_numeric(subset["drift_bias_abs"], errors="coerce").to_numpy(float))
            if bool(np.any(need_bias)):
                for idx in subset.index[need_bias]:
                    seed_val = pd.to_numeric(subset.loc[idx, "seed"], errors="coerce")
                    if not np.isfinite(seed_val):
                        continue
                    subset.loc[idx, "drift_bias_abs"] = _mean_drift_bias_from_per_seed_files(
                        run_dir=grid_dir,
                        regime=str(regime),
                        seed=int(seed_val),
                        method="st_rbf",
                    )

            n_seeds = int(pd.to_numeric(subset["seed"], errors="coerce").dropna().nunique())
            mean_all, std_all = _safe_mean_std(subset["rms_all"].to_numpy(float))
            mean_stat, std_stat = _safe_mean_std(subset["rms_stationary"].to_numpy(float))
            mean_migr, std_migr = _safe_mean_std(subset["rms_migrating"].to_numpy(float))
            mean_bias, std_bias = _safe_mean_std(subset["drift_bias_abs"].to_numpy(float))

            rows.append(
                {
                    "regime": str(regime),
                    "ell_space": ell_space,
                    "ell_time": ell_time,
                    "n_seeds_found": n_seeds,
                    "mean_rms_all": mean_all,
                    "std_rms_all": std_all,
                    "mean_rms_stationary": mean_stat,
                    "std_rms_stationary": std_stat,
                    "mean_rms_migrating": mean_migr,
                    "std_rms_migrating": std_migr,
                    "mean_drift_bias_abs": mean_bias,
                    "std_drift_bias_abs": std_bias,
                }
            )

    if not rows:
        raise RuntimeError("No ST-RBF grid results could be aggregated.")

    out = pd.DataFrame(rows).drop_duplicates(subset=["regime", "ell_space", "ell_time"], keep="last")
    cols = [
        "regime",
        "ell_space",
        "ell_time",
        "n_seeds_found",
        "mean_rms_all",
        "std_rms_all",
        "mean_rms_stationary",
        "std_rms_stationary",
        "mean_rms_migrating",
        "std_rms_migrating",
        "mean_drift_bias_abs",
        "std_drift_bias_abs",
    ]
    return out[cols].sort_values(["regime", "ell_space", "ell_time"]).reset_index(drop=True)

def _run_stationary_ab_tests(synthetic_dir: Path, out_root: Path) -> None:
    """Run ST-RBF stationary subset A/B tests for two fixed candidate settings."""
    for cand in STATIONARY_AB_CANDIDATES:
        es = float(cand["ell_space"])
        et = float(cand["ell_time"])
        for use_subset in [0, 1]:
            out_dir = out_root / f"st_rbf_stationary_ab_es{es}_et{et}_subset{use_subset}"
            print(
                "[INFO] Running stationary A/B config: "
                f"ell_space={es}, ell_time={et}, use_stationary_subset={use_subset}"
            )
            synthetic_regime_benchmark(
                synthetic_dir=str(synthetic_dir),
                out=str(out_dir),
                reuse_existing=1,
                methods="st_rbf",
                synthetic_n_seeds=10,
                st_ell_space=es,
                st_ell_time=et,
                use_stationary_subset=bool(use_subset),
            )

def _aggregate_stationary_ab(out_root: Path) -> pd.DataFrame:
    """Aggregate stationary subset A/B experiment outputs into one summary table."""
    rows: list[dict[str, float | str | int]] = []
    run_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("st_rbf_stationary_ab_es")])
    if not run_dirs:
        raise FileNotFoundError(f"No stationary A/B folders found in {out_root}")

    for run_dir in run_dirs:
        match = STATIONARY_AB_DIR_PATTERN.match(run_dir.name)
        if match is None:
            print(f"[WARN] Skip unexpected stationary A/B folder: {run_dir.name}")
            continue
        ell_space = float(match.group("es"))
        ell_time = float(match.group("et"))
        use_subset = int(match.group("subset"))

        regime_to_csv = _find_regime_metric_csvs(run_dir)
        for regime, csv_path in regime_to_csv.items():
            metrics = _load_and_standardize_seed_metrics(csv_path)
            if metrics.empty:
                continue
            subset = metrics[
                (metrics["method"].astype(str) == "st_rbf")
                & (metrics["regime"].astype(str) == str(regime))
            ].copy()
            if subset.empty:
                continue

            need_bias = ~np.isfinite(pd.to_numeric(subset["drift_bias_abs"], errors="coerce").to_numpy(float))
            if bool(np.any(need_bias)):
                for idx in subset.index[need_bias]:
                    seed_val = pd.to_numeric(subset.loc[idx, "seed"], errors="coerce")
                    if not np.isfinite(seed_val):
                        continue
                    subset.loc[idx, "drift_bias_abs"] = _mean_drift_bias_from_per_seed_files(
                        run_dir=run_dir,
                        regime=str(regime),
                        seed=int(seed_val),
                        method="st_rbf",
                    )

            n_seeds = int(pd.to_numeric(subset["seed"], errors="coerce").dropna().nunique())
            mean_all, std_all = _safe_mean_std(subset["rms_all"].to_numpy(float))
            mean_stat, std_stat = _safe_mean_std(subset["rms_stationary"].to_numpy(float))
            mean_migr, std_migr = _safe_mean_std(subset["rms_migrating"].to_numpy(float))
            mean_bias, std_bias = _safe_mean_std(subset["drift_bias_abs"].to_numpy(float))

            rows.append(
                {
                    "regime": str(regime),
                    "ell_space": ell_space,
                    "ell_time": ell_time,
                    "use_stationary_subset": use_subset,
                    "n_seeds_found": n_seeds,
                    "mean_rms_all": mean_all,
                    "std_rms_all": std_all,
                    "mean_rms_stationary": mean_stat,
                    "std_rms_stationary": std_stat,
                    "mean_rms_migrating": mean_migr,
                    "std_rms_migrating": std_migr,
                    "mean_drift_bias_abs": mean_bias,
                    "std_drift_bias_abs": std_bias,
                }
            )

    if not rows:
        raise RuntimeError("No stationary A/B results could be aggregated.")
    out = pd.DataFrame(rows).drop_duplicates(subset=["regime", "ell_space", "ell_time", "use_stationary_subset"], keep="last")
    cols = [
        "regime",
        "ell_space",
        "ell_time",
        "use_stationary_subset",
        "n_seeds_found",
        "mean_rms_all",
        "std_rms_all",
        "mean_rms_stationary",
        "std_rms_stationary",
        "mean_rms_migrating",
        "std_rms_migrating",
        "mean_drift_bias_abs",
        "std_drift_bias_abs",
    ]
    return out[cols].sort_values(["regime", "ell_space", "ell_time", "use_stationary_subset"]).reset_index(drop=True)

def _write_stationary_ab_report(summary: pd.DataFrame, out_md_path: Path) -> None:
    """Write a markdown ranking report for stationary subset A/B configs."""
    lines: list[str] = []
    lines.append("# ST-RBF Stationary Subset A/B Report")
    lines.append("")
    lines.append(f"- timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- rows: {int(summary.shape[0])}")
    lines.append("")

    for regime in sorted(summary["regime"].astype(str).unique().tolist()):
        cur = summary[summary["regime"].astype(str) == regime].copy()
        if cur.empty:
            continue
        lines.append(f"## Regime: {regime}")
        lines.append("")
        cur["cfg"] = (
            "es="
            + cur["ell_space"].map(lambda v: f"{float(v):.1f}")
            + ", et="
            + cur["ell_time"].map(lambda v: f"{float(v):.1f}")
            + ", subset="
            + cur["use_stationary_subset"].astype(int).astype(str)
        )
        rms_rank = cur.assign(_sort=cur["mean_rms_all"].where(np.isfinite(cur["mean_rms_all"]), np.inf)).sort_values("_sort")
        bias_rank = cur.assign(
            _sort=cur["mean_drift_bias_abs"].where(np.isfinite(cur["mean_drift_bias_abs"]), np.inf)
        ).sort_values("_sort")

        lines.append("### Rank by mean_rms_all (lower is better)")
        lines.append("")
        lines.append("| rank | config | n_seeds_found | mean_rms_all | std_rms_all |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, (_, row) in enumerate(rms_rank.iterrows(), start=1):
            lines.append(
                f"| {i} | {row['cfg']} | {int(row['n_seeds_found'])} | "
                f"{float(row['mean_rms_all']):.6f} | {float(row['std_rms_all']):.6f} |"
            )
        lines.append("")
        lines.append("### Rank by mean_drift_bias_abs (lower is better)")
        lines.append("")
        lines.append("| rank | config | n_seeds_found | mean_drift_bias_abs | std_drift_bias_abs |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, (_, row) in enumerate(bias_rank.iterrows(), start=1):
            mean_bias = float(row["mean_drift_bias_abs"]) if np.isfinite(row["mean_drift_bias_abs"]) else float("nan")
            std_bias = float(row["std_drift_bias_abs"]) if np.isfinite(row["std_drift_bias_abs"]) else float("nan")
            lines.append(
                f"| {i} | {row['cfg']} | {int(row['n_seeds_found'])} | "
                f"{mean_bias:.6f} | {std_bias:.6f} |"
            )
        lines.append("")

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def _plot_grid_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    highlight: tuple[float, float] = (4.0, 0.4),
) -> bool:
    """Plot grid heatmap (x=ell_time, y=ell_space) from aggregated CSV rows."""
    required = {"ell_space", "ell_time", value_col}
    if not required.issubset(df.columns):
        print(f"[WARN] Heatmap skipped ({out_path.name}): missing columns {sorted(required - set(df.columns))}")
        return False
    cur = df[["ell_space", "ell_time", value_col]].copy()
    cur["ell_space"] = pd.to_numeric(cur["ell_space"], errors="coerce")
    cur["ell_time"] = pd.to_numeric(cur["ell_time"], errors="coerce")
    cur[value_col] = pd.to_numeric(cur[value_col], errors="coerce")
    cur = cur.dropna(subset=["ell_space", "ell_time"])
    if cur.empty:
        print(f"[WARN] Heatmap skipped ({out_path.name}): no valid rows.")
        return False

    y_vals = sorted(cur["ell_space"].unique().tolist())
    x_vals = sorted(cur["ell_time"].unique().tolist())
    mat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    agg = cur.groupby(["ell_space", "ell_time"], as_index=False)[value_col].mean()
    x_lookup = {float(v): i for i, v in enumerate(x_vals)}
    y_lookup = {float(v): i for i, v in enumerate(y_vals)}
    for _, row in agg.iterrows():
        xv = float(row["ell_time"])
        yv = float(row["ell_space"])
        if xv in x_lookup and yv in y_lookup:
            mat[y_lookup[yv], x_lookup[xv]] = float(row[value_col])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_col)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:.2g}" for v in x_vals])
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{v:.2g}" for v in y_vals])
    ax.set_xlabel("ell_time")
    ax.set_ylabel("ell_space")
    ax.set_title(title)

    hs, ht = highlight
    ix = next((i for i, x in enumerate(x_vals) if np.isclose(float(x), float(ht))), None)
    iy = next((i for i, y in enumerate(y_vals) if np.isclose(float(y), float(hs))), None)
    if ix is not None and iy is not None:
        ax.scatter([ix], [iy], s=130, marker="o", facecolors="none", edgecolors="red", linewidths=1.8, zorder=5)
        ax.scatter([ix], [iy], s=20, color="red", zorder=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _plot_stationary_subset_ab(
    df: pd.DataFrame,
    regime: str | None,
    out_path: Path,
    highlight: tuple[float, float] = (4.0, 0.4),
) -> bool:
    """Plot stationary subset A/B bars (per regime) or combined line plot (across regimes)."""
    req = {"regime", "ell_space", "ell_time", "use_stationary_subset", "mean_rms_all", "std_rms_all"}
    if not req.issubset(df.columns):
        print(f"[WARN] Stationary A/B plot skipped ({out_path.name}): missing columns {sorted(req - set(df.columns))}")
        return False
    cur = df.copy()
    for c in ["ell_space", "ell_time", "use_stationary_subset", "mean_rms_all", "std_rms_all"]:
        cur[c] = pd.to_numeric(cur[c], errors="coerce")
    cur = cur.dropna(subset=["use_stationary_subset"])
    cur["use_stationary_subset"] = cur["use_stationary_subset"].astype(int)
    cur = cur[cur["use_stationary_subset"].isin([0, 1])]
    if cur.empty:
        return False

    if regime is not None:
        cur = cur[cur["regime"].astype(str) == str(regime)].copy()
        if cur.empty:
            return False
        cur["cfg"] = cur.apply(lambda row: f"es={float(row['ell_space']):.1f}, et={float(row['ell_time']):.1f}", axis=1)
        cfg_order = (
            cur[["cfg", "ell_space", "ell_time"]].drop_duplicates().sort_values(["ell_space", "ell_time"])["cfg"].tolist()
        )
        piv_mean = cur.pivot_table(index="cfg", columns="use_stationary_subset", values="mean_rms_all", aggfunc="mean").reindex(cfg_order)
        piv_std = cur.pivot_table(index="cfg", columns="use_stationary_subset", values="std_rms_all", aggfunc="mean").reindex(cfg_order)
        x = np.arange(len(cfg_order), dtype=float)
        w = 0.36
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        y0 = piv_mean.get(0, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        y1 = piv_mean.get(1, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        e0 = piv_std.get(0, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        e1 = piv_std.get(1, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        ax.bar(x - w / 2.0, y0, width=w, yerr=e0, capsize=3, label="subset=0")
        ax.bar(x + w / 2.0, y1, width=w, yerr=e1, capsize=3, label="subset=1")
        ax.set_xticks(x)
        ax.set_xticklabels(cfg_order, rotation=20, ha="right")
        ax.set_ylabel("mean_rms_all")
        ax.set_title(f"Stationary subset A/B: {regime}")
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return True

    hs, ht = highlight
    pick = cur[np.isclose(cur["ell_space"].to_numpy(float), hs) & np.isclose(cur["ell_time"].to_numpy(float), ht)].copy()
    if pick.empty:
        pick = cur.groupby(["regime", "use_stationary_subset"], as_index=False).agg(
            mean_rms_all=("mean_rms_all", "mean"),
            std_rms_all=("std_rms_all", "mean"),
        )
    regimes = sorted(pick["regime"].astype(str).unique().tolist())
    if not regimes:
        return False
    p0 = pick[pick["use_stationary_subset"] == 0].set_index("regime")
    p1 = pick[pick["use_stationary_subset"] == 1].set_index("regime")
    x = np.arange(len(regimes), dtype=float)
    y0 = np.array([float(p0.loc[r, "mean_rms_all"]) if r in p0.index else np.nan for r in regimes], dtype=float)
    y1 = np.array([float(p1.loc[r, "mean_rms_all"]) if r in p1.index else np.nan for r in regimes], dtype=float)
    e0 = np.array([float(p0.loc[r, "std_rms_all"]) if r in p0.index else np.nan for r in regimes], dtype=float)
    e1 = np.array([float(p1.loc[r, "std_rms_all"]) if r in p1.index else np.nan for r in regimes], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(x, y0, yerr=e0, marker="o", capsize=3, label="subset=0")
    ax.errorbar(x, y1, yerr=e1, marker="o", capsize=3, label="subset=1")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylabel("mean_rms_all")
    ax.set_title("Stationary subset A/B across regimes")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True

def _write_st_rbf_plot_readme(out_root: Path, generated_paths: list[Path], grid_csv: Path, ab_csv: Path) -> Path:
    """Write README for ST-RBF grid/stationary subset selection plots."""
    readme_path = out_root / "st_rbf_parameter_selection_plots_README.txt"
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"input_grid_csv: {grid_csv}",
        f"input_stationary_ab_csv: {ab_csv}",
        "selected_default_point: ell_space=4.0, ell_time=0.4",
        "selected_stationary_default: use_stationary_subset=0 (off)",
        "generated_plots:",
    ]
    lines.extend([f"- {p.name}" for p in sorted(generated_paths)])
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


# STUDY AGGREGATION HELPERS:
"""Build tidy and summary aggregation tables from per-file groundtruth metrics outputs.

This helper layer exists so each sweep cell can emit canonical study artifacts:
- `<study_tag>_aggregated.csv` (long/tidy)
- `<study_tag>_summary.csv` (grouped mean/std/n)
"""

def _collect_metrics_from_roots(config_roots: dict[str, Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for config, root in config_roots.items():
        if not root.exists():
            continue
        for p in sorted(root.rglob("*_gt_metrics_summary.csv")):
            file_stem = p.name.replace("_gt_metrics_summary.csv", "")
            mode = parse_mode_from_filename(file_stem) or "unknown"
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                print(f"[WARN] Failed to read {p}: {exc}")
                continue
            if df.empty:
                continue
            if "method" not in df.columns:
                df = df.copy()
                df["method"] = "unknown"
            metric_cols = [
                c for c in df.columns
                if c not in {"method"} and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not metric_cols:
                continue
            tidy = df[["method"] + metric_cols].melt(
                id_vars=["method"], var_name="metric_name", value_name="metric_value"
            )
            tidy["file_stem"] = file_stem
            tidy["mode"] = mode
            tidy["config"] = config
            rows.append(tidy[["file_stem", "mode", "config", "method", "metric_name", "metric_value"]])
    if not rows:
        return pd.DataFrame(columns=["file_stem", "mode", "config", "method", "metric_name", "metric_value"])
    out = pd.concat(rows, ignore_index=True)
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    return out

def _write_study_aggregates(study_tag: str, config_roots: dict[str, Path]) -> tuple[Path, Path]:
    """Write tidy and grouped summary CSVs for a study across multiple configs."""
    out_dir = results_root(study_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy = _collect_metrics_from_roots(config_roots)
    tidy_path = out_dir / f"{study_tag}_aggregated.csv"
    tidy.to_csv(tidy_path, index=False)

    metrics_of_interest = ["overall_rms_error", "drift_bias_abs"]
    sub = tidy[tidy["metric_name"].isin(metrics_of_interest)].copy()
    if sub.empty:
        summary = pd.DataFrame(
            columns=["mode", "config", "method", "metric_name", "mean", "std", "n"]
        )
    else:
        summary = (
            sub.groupby(["mode", "config", "method", "metric_name"], as_index=False)["metric_value"]
            .agg(mean="mean", std="std", n="count")
            .sort_values(["mode", "config", "method", "metric_name"])
            .reset_index(drop=True)
        )
    summary_path = out_dir / f"{study_tag}_summary.csv"
    summary.to_csv(summary_path, index=False)
    return tidy_path, summary_path


def _ordered_modes(values: list[str]) -> list[str]:
    """Return mode labels in canonical order with unknown extras appended."""
    known = [m for m in BASELINE_MODES if m in values]
    extras = sorted([m for m in values if m not in BASELINE_MODES])
    return known + extras


def _ordered_methods(values: list[str]) -> list[str]:
    """Return method labels in canonical order with unknown extras appended."""
    known = [m for m in DEFAULT_METHODS if m in values]
    extras = sorted([m for m in values if m not in DEFAULT_METHODS])
    return known + extras


def _plot_summary_grouped_bars(
    summary_df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
    title: str,
) -> bool:
    """Plot grouped mean±std bars per mode with one bar per method."""
    sub = summary_df[summary_df["metric_name"].astype(str) == str(metric_name)].copy()
    if sub.empty:
        print(f"[WARN] No rows for metric '{metric_name}' in baseline summary.")
        return False
    modes = _ordered_modes(sorted(sub["mode"].astype(str).unique().tolist()))
    methods = _ordered_methods(sorted(sub["method"].astype(str).unique().tolist()))
    if not modes or not methods:
        return False

    piv_mean = sub.pivot_table(index="mode", columns="method", values="mean", aggfunc="mean")
    piv_std = sub.pivot_table(index="mode", columns="method", values="std", aggfunc="mean")
    piv_mean = piv_mean.reindex(index=modes, columns=methods)
    piv_std = piv_std.reindex(index=modes, columns=methods)

    x = np.arange(len(modes), dtype=float)
    width = min(0.8 / max(len(methods), 1), 0.22)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for i, method in enumerate(methods):
        xi = x + (i - (len(methods) - 1) / 2.0) * width
        yy = pd.to_numeric(piv_mean[method], errors="coerce").to_numpy(float)
        ee = pd.to_numeric(piv_std[method], errors="coerce").to_numpy(float)
        ax.bar(xi, yy, width=width, yerr=ee, capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _plot_summary_heatmap(
    summary_df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (6.0, 4.0)
) -> bool:
    """Plot method x mode heatmap of mean metric values."""
    sub = summary_df[summary_df["metric_name"].astype(str) == str(metric_name)].copy()
    if sub.empty:
        return False
    modes = _ordered_modes(sorted(sub["mode"].astype(str).unique().tolist()))
    methods = _ordered_methods(sorted(sub["method"].astype(str).unique().tolist()))
    piv = sub.pivot_table(index="method", columns="mode", values="mean", aggfunc="mean")
    piv = piv.reindex(index=methods, columns=modes)
    if piv.shape[0] == 0 or piv.shape[1] == 0:
        return False

    mat = piv.to_numpy(float)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # remove frame around colorbar:
    cbar.outline.set_visible(False)
    # replace all "_" with " ":
    cbar.set_label(metric_name.replace("_", " "))
    ax.set_xticks(np.arange(len(modes)))
    # replace all "_" with " " in mode labels:
    modes_names = [m.replace("_", " ") for m in modes]
    ax.set_xticklabels(modes_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(title)
    ax.set_xlabel("mode")
    ax.set_ylabel("method")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    return True


def _plot_config_grouped_bars(
    summary_df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
    title: str,
    method_name: str = "rbf",
) -> bool:
    """Plot grouped mean±std bars per mode with one bar per config for a single method."""
    sub = summary_df[
        (summary_df["metric_name"].astype(str) == str(metric_name))
        & (summary_df["method"].astype(str) == str(method_name))
    ].copy()
    if sub.empty:
        print(f"[WARN] No rows for metric='{metric_name}', method='{method_name}'.")
        return False

    modes = _ordered_modes(sorted(sub["mode"].astype(str).unique().tolist()))
    configs = sorted(sub["config"].astype(str).unique().tolist())
    if not modes or not configs:
        return False

    piv_mean = sub.pivot_table(index="mode", columns="config", values="mean", aggfunc="mean").reindex(index=modes, columns=configs)
    piv_std = sub.pivot_table(index="mode", columns="config", values="std", aggfunc="mean").reindex(index=modes, columns=configs)

    x = np.arange(len(modes), dtype=float)
    width = min(0.8 / max(len(configs), 1), 0.25)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for i, cfg in enumerate(configs):
        xi = x + (i - (len(configs) - 1) / 2.0) * width
        yy = pd.to_numeric(piv_mean[cfg], errors="coerce").to_numpy(float)
        ee = pd.to_numeric(piv_std[cfg], errors="coerce").to_numpy(float)
        ax.bar(xi, yy, width=width, yerr=ee, capsize=3, label=cfg)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend(title="config")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _plot_config_heatmap(
    summary_df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
    title: str,
    method_name: str = "rbf",
) -> bool:
    """Plot config x mode heatmap of mean metric values for a single method."""
    sub = summary_df[
        (summary_df["metric_name"].astype(str) == str(metric_name))
        & (summary_df["method"].astype(str) == str(method_name))
    ].copy()
    if sub.empty:
        return False
    modes = _ordered_modes(sorted(sub["mode"].astype(str).unique().tolist()))
    configs = sorted(sub["config"].astype(str).unique().tolist())
    piv = sub.pivot_table(index="config", columns="mode", values="mean", aggfunc="mean").reindex(index=configs, columns=modes)
    if piv.shape[0] == 0 or piv.shape[1] == 0:
        return False

    mat = piv.to_numpy(float)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_title(title)
    ax.set_xlabel("mode")
    ax.set_ylabel("config")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _regime_from_run_folder(run_folder: str) -> str:
    """Extract regime name from `<regime>_seed_<NNN>` run folder labels."""
    s = str(run_folder)
    if "_seed_" in s:
        return s.split("_seed_")[0]
    return s


def _plot_large_regime_benchmark_from_aggregates(out_root: Path) -> list[Path]:
    """Create interpretable overview plots from large-regime aggregated CSV outputs."""
    out_root = Path(out_root)
    agg_path = out_root / "regime_benchmark_large_20seeds_aggregated.csv"
    if not agg_path.exists():
        print(f"[WARN] Large-regime plots skipped: missing {agg_path}")
        return []
    try:
        agg = pd.read_csv(agg_path)
    except Exception as exc:
        print(f"[WARN] Large-regime plots skipped: failed reading {agg_path}: {exc}")
        return []
    required = {"run_folder", "method", "metric_name", "metric_value"}
    if not required.issubset(agg.columns):
        print(f"[WARN] Large-regime plots skipped: missing columns {sorted(required - set(agg.columns))}")
        return []
    if agg.empty:
        return []

    tab = agg.copy()
    tab["mode"] = tab["run_folder"].astype(str).map(_regime_from_run_folder)
    tab["metric_value"] = pd.to_numeric(tab["metric_value"], errors="coerce")
    tab = tab.dropna(subset=["metric_value"])
    if tab.empty:
        return []

    summary_regime = (
        tab.groupby(["mode", "method", "metric_name"], as_index=False)["metric_value"]
        .agg(mean="mean", std="std", n="count")
        .sort_values(["mode", "method", "metric_name"])
        .reset_index(drop=True)
    )
    summary_regime_path = out_root / "regime_benchmark_large_20seeds_regime_summary.csv"
    summary_regime.to_csv(summary_regime_path, index=False)

    generated: list[Path] = []
    plot_specs = [
        ("overall_rms_error", "Large regime benchmark: overall RMS by regime and method"),
        ("drift_bias_abs", "Large regime benchmark: drift bias by regime and method"),
        ("overall_rms_error_stationary", "Large regime benchmark: stationary RMS by regime and method"),
        ("overall_rms_error_migrating", "Large regime benchmark: migrating RMS by regime and method"),
    ]
    for metric_name, title in plot_specs:
        p_bar = out_root / f"regime_benchmark_large_20seeds_{metric_name}_grouped_bar_plot.png"
        if _plot_summary_grouped_bars(summary_regime, metric_name=metric_name, out_path=p_bar, title=title):
            generated.append(p_bar)
        p_hm = out_root / f"regime_benchmark_large_20seeds_{metric_name}_heatmap_plot.png"
        if _plot_summary_heatmap(summary_regime, metric_name=metric_name, out_path=p_hm, title=title):
            generated.append(p_hm)

    if generated:
        readme = out_root / "regime_benchmark_large_20seeds_plots_README.txt"
        readme.write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"input_aggregated_csv: {agg_path}",
                    f"regime_summary_csv: {summary_regime_path}",
                    "description: Regime-level mean/std aggregation across all seeds and files.",
                    "plots:",
                    *[f"- {p.name}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return generated

# %% GENERATE SYNTHETIC BASELINE DATASET
"""Generate/validate baseline dataset across three modes in `_groundtruth`.

Question answered:
- Are all baseline modes (smooth, piecewise_smooth, piecewise_abrupt) present with stable names?

Datasets:
- Uses `_groundtruth` only.

Artifacts:
- Mode-prefixed synthetic files in `_groundtruth`.

Success:
- Exactly/at least 5 files per mode exist and are discoverable by mode prefix.
"""

RUN_GENERATE_BASELINE_GROUNDTRUTH = True
if RUN_GENERATE_BASELINE_GROUNDTRUTH:
    ensure_baseline_groundtruth()
    grouped = enumerate_groundtruth_files(GROUNDTRUTH_ROOT)
    for mode in BASELINE_MODES:
        print(f"[OK] {mode}: {len(grouped.get(mode, []))} files in {GROUNDTRUTH_ROOT}")
# %% GENERATE LARGE REGIME DATASET
"""Generate/validate 3x20 seeded regime datasets in `_groundtruth_large`.

Question answered:
- Is the canonical large dataset for regime benchmarking complete and reproducible?

Datasets:
- Uses `_groundtruth_large` only.

Artifacts:
- 60 deterministic folders: `<mode_token>_seed_<NNN>`.

Success:
- All expected folders exist; missing folders are generated without touching existing ones.
"""

RUN_GENERATE_LARGE_GROUNDTRUTH = True
if RUN_GENERATE_LARGE_GROUNDTRUTH:
    ensure_large_groundtruth_regimes()
    print(f"[OK] large dataset folders: {len(list_large_regime_folders())}")
# %% BASELINE METHOD COMPARISON (All Baseline Modes; "baseline_suite")
"""Run baseline methods on all baseline files across all three modes.

Question answered:
- How do core methods compare on per-file synthetic groundtruth across mode diversity?

Datasets:
- `_groundtruth` (smooth + piecewise_smooth + piecewise_abrupt).

Artifacts:
- Per-file method outputs under `Synthetic files parameter sweep/baseline_suite/<file_stem>/...`.

Success:
- Full outputs are generated for every baseline file and method.
"""

""" ensure_baseline_groundtruth()
baseline_root = results_root("baseline_suite")
baseline_methods = ["rigid", "local", "rbf", "st_rbf", "st_rbf_moe"]
baseline_opts = dict(OPTIONS)
baseline_opts["st_moe_k_list"] = [2, 3, 5]


file_path = list_baseline_imaris_files()[14]
df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
out_prefix = baseline_root / file_path.stem / file_path.stem
corr = run_methods_for_dataframe(
    df=df_syn,
    out_prefix=out_prefix,
    methods=baseline_methods,
    opts=baseline_opts,
    use_stationary_subset=False)
postprocess_outputs(df_syn, file_path, out_prefix, corr) """


RUN_BASELINE_METHOD_COMPARISON = True
if RUN_BASELINE_METHOD_COMPARISON:
    ensure_baseline_groundtruth()
    baseline_root = results_root("baseline_suite")
    baseline_methods = ["rigid", "local", "rbf", "st_rbf", "st_rbf_moe"]
    baseline_opts = dict(OPTIONS)
    baseline_opts["st_moe_k_list"] = [2, 3, 5]

    for file_path in list_baseline_imaris_files():
        # file_path = list_baseline_imaris_files()[14]
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = baseline_root / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(
            df=df_syn,
            out_prefix=out_prefix,
            methods=baseline_methods,
            opts=baseline_opts,
            use_stationary_subset=False)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# Baseline Suite Aggregation
"""Aggregate baseline suite metrics into canonical tidy and summary CSVs.

Question answered:
- What is the method-level performance distribution by mode over baseline files?

Datasets:
- Reads outputs from `Synthetic files parameter sweep/baseline_suite`.

Artifacts:
- `Synthetic files parameter sweep/baseline_suite/baseline_suite_aggregated.csv`
- `Synthetic files parameter sweep/baseline_suite/baseline_suite_summary.csv`
- `Synthetic files parameter sweep/baseline_suite/baseline_suite_*_plot.png` (mode-vs-method summaries)

Success:
- Summary tables plus overview plots make method differences across smooth/piecewise modes directly visible.
"""

RUN_BASELINE_SUITE_AGGREGATION = True
if RUN_BASELINE_SUITE_AGGREGATION:
    tidy_path, summary_path = _write_study_aggregates(
        "baseline_suite",
        {"baseline_suite": results_root("baseline_suite")})
    print(f"[OK] {tidy_path}")
    print(f"[OK] {summary_path}")

    out_dir = results_root("baseline_suite")
    summary = pd.read_csv(summary_path)
    tidy = pd.read_csv(tidy_path)
    metrics_extended = ["overall_rms_error", "median_error_norm", "p95_error_norm", "drift_bias_abs"]
    sub_ext = tidy[tidy["metric_name"].astype(str).isin(metrics_extended)].copy()
    if sub_ext.empty:
        summary_ext = pd.DataFrame(columns=["mode", "config", "method", "metric_name", "mean", "std", "n"])
    else:
        summary_ext = (
            sub_ext.groupby(["mode", "config", "method", "metric_name"], as_index=False)["metric_value"]
            .agg(mean="mean", std="std", n="count")
            .sort_values(["mode", "config", "method", "metric_name"])
            .reset_index(drop=True)
        )
    summary_ext_path = out_dir / "baseline_suite_summary_extended.csv"
    summary_ext.to_csv(summary_ext_path, index=False)
    print(f"[OK] {summary_ext_path}")

    generated: list[Path] = []
    plot_specs = [
        ("overall_rms_error", "Overall RMS by mode and method"),
        ("median_error_norm", "Median error by mode and method"),
        ("p95_error_norm", "P95 error by mode and method"),
        ("drift_bias_abs", "Drift bias by mode and method"),
    ]
    for metric_name, title in plot_specs:
        p_bar = out_dir / f"baseline_suite_{metric_name}_grouped_bar_plot.pdf"
        if _plot_summary_grouped_bars(summary_ext, metric_name=metric_name, out_path=p_bar, title=title):
            generated.append(p_bar)
        p_hm = out_dir / f"baseline_suite_{metric_name}_heatmap_plot.pdf"
        CM = 1/2.54  # centimeters in inches
        if _plot_summary_heatmap(summary_ext, metric_name=metric_name, out_path=p_hm, title=title, 
                                 figsize=(8*CM, 6.0*CM)):
            generated.append(p_hm)

    # Reviewer figure: RMS vs P95 scatter (points = method x regime).
    rms_tab = summary_ext[summary_ext["metric_name"].astype(str) == "overall_rms_error"][
        ["mode", "method", "mean"]
    ].rename(columns={"mean": "rms_mean"})
    p95_tab = summary_ext[summary_ext["metric_name"].astype(str) == "p95_error_norm"][
        ["mode", "method", "mean"]
    ].rename(columns={"mean": "p95_mean"})
    scatter_df = rms_tab.merge(p95_tab, on=["mode", "method"], how="inner")
    if scatter_df.shape[0] > 0:
        modes = _ordered_modes(sorted(scatter_df["mode"].astype(str).unique().tolist()))
        methods = _ordered_methods(sorted(scatter_df["method"].astype(str).unique().tolist()))
        marker_map = {m: mk for m, mk in zip(modes, ["o", "s", "^", "D", "v", "P", "X", "*"])}
        color_map = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods)}
        #fig, ax = plt.subplots(figsize=(6.0, 4.8))
        fig, ax = plt.subplots(figsize=(9*CM, 6*CM))
        # add some dummy plots for legend to plot a) colors of methods and b) markers of modes:
        for m in methods:
            ax.scatter([], [], color=color_map.get(m, "C0"), s=30,
                       label=m, alpha=0.8, edgecolor="black", linewidth=0.0)
        for m in modes:
            ax.scatter([], [], marker=marker_map.get(m, "o"), color="gray", label=m, s=30,
                       alpha=0.8, edgecolor="black", linewidth=0.0)
        
        for _, r in scatter_df.iterrows():
            mode = str(r["mode"])
            method = str(r["method"])
            x = float(r["rms_mean"])
            y = float(r["p95_mean"])
            ax.scatter(
                x,
                y,
                s=30,
                marker=marker_map.get(mode, "o"),
                color=color_map.get(method, "C0"),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.0,
            )
            #ax.annotate(f"{method} | {mode}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax.set_xlabel("RMS (mean)")
        ax.set_ylabel("P95 error norm (mean)")
        ax.set_title("RMS vs P95 (method x mode)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        plt.legend(title="method / mode", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        p_sc = out_dir / "baseline_suite_rms_vs_p95_scatter_plot.pdf"
        fig.savefig(p_sc, dpi=300)
        plt.close(fig)
        generated.append(p_sc)

    if generated:
        print(f"[OK] Wrote baseline suite plots: {len(generated)}")
# %% Canonical Large Regime Benchmark (3 Modes x 20 Seeds) FOR SUPPLEMENTARY MATERIALS (!)
"""Run the canonical large benchmark over `_groundtruth_large`.

Question answered:
- How stable are method rankings across 3 regimes and 20 seeds each?

Datasets:
- `_groundtruth_large` only.

Artifacts:
- Per-run outputs in `Synthetic files parameter sweep/regime_benchmark_large_20seeds/...`
- Aggregated tables in the same root.
"""

RUN_LARGE_REGIME_BENCHMARK = True
if RUN_LARGE_REGIME_BENCHMARK:
    ensure_large_groundtruth_regimes()
    out_root = results_root("regime_benchmark_large_20seeds")
    methods = ["rigid", "local", "rbf", "st_rbf"]
    opts = dict(OPTIONS)

    for regime_dir in list_large_regime_folders():
        for file_path in list_imaris_xls_files(regime_dir):
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = out_root / regime_dir.name / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=methods, opts=opts)
            postprocess_outputs(df_syn, file_path, out_prefix, corr)

    # Aggregate per-file metric summaries.
    rows: list[pd.DataFrame] = []
    for p in sorted(out_root.rglob("*_gt_metrics_summary.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "method" not in df.columns:
            continue
        run_folder = p.parents[1].name  # <regime_folder>/<file_stem>/<file_stem>_gt_metrics_summary.csv
        metrics = [c for c in df.columns if c != "method" and pd.api.types.is_numeric_dtype(df[c])]
        if not metrics:
            continue
        tidy = df[["method"] + metrics].melt(id_vars=["method"], var_name="metric_name", value_name="metric_value")
        tidy["run_folder"] = run_folder
        tidy["file_stem"] = p.name.replace("_gt_metrics_summary.csv", "")
        rows.append(tidy[["run_folder", "file_stem", "method", "metric_name", "metric_value"]])
    if rows:
        agg = pd.concat(rows, ignore_index=True)
        agg.to_csv(out_root / "regime_benchmark_large_20seeds_aggregated.csv", index=False)
        summary = (
            agg.groupby(["run_folder", "method", "metric_name"], as_index=False)["metric_value"]
            .agg(mean="mean", std="std", n="count")
            .sort_values(["run_folder", "method", "metric_name"])
            .reset_index(drop=True)
        )
        summary.to_csv(out_root / "regime_benchmark_large_20seeds_summary.csv", index=False)

RUN_LARGE_REGIME_BENCHMARK_PLOTS = True
if RUN_LARGE_REGIME_BENCHMARK and RUN_LARGE_REGIME_BENCHMARK_PLOTS:
    large_root = results_root("regime_benchmark_large_20seeds")
    generated = _plot_large_regime_benchmark_from_aggregates(large_root)
    if generated:
        print(f"[OK] Wrote large-regime evaluation plots: {len(generated)}")
# %% RBF Robust Mode Sweep (All Baseline Modes)
"""Run RBF robust-mode sweep across all baseline files and modes.

Question answered:
- Does IRLS-Huber improve or stabilize RBF versus sample trimming across modes?

Datasets:
- `_groundtruth` all files.

Artifacts:
- Per-config outputs under `Synthetic files parameter sweep/rbf_robust_*`.
"""

RUN_RBF_ROBUST_MODE_SWEEP = True
if RUN_RBF_ROBUST_MODE_SWEEP:
    ensure_baseline_groundtruth()
    runs = [("sample_trim", "rbf_robust_sample_trim"), ("irls_huber", "rbf_robust_irls_huber")]
    for robust_mode, tag in runs:
        opts_rbf = dict(OPTIONS)
        opts_rbf["robust_mode"] = robust_mode
        out_root = results_root(tag)
        for file_path in list_baseline_imaris_files():
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = out_root / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=["rbf"], opts=opts_rbf)
            postprocess_outputs(df_syn, file_path, out_prefix, corr)

# RBF Robust Mode Aggregation:
"""Aggregate RBF robust sweep into tidy and summary CSVs.

Question answered:
- Which robust configuration yields lower RMS and drift-bias by mode?

Datasets:
- Reads `Synthetic files parameter sweep/rbf_robust_sample_trim` and `Synthetic files parameter sweep/rbf_robust_irls_huber`.

Artifacts:
- `Synthetic files parameter sweep/rbf_robust_sweep/rbf_robust_sweep_aggregated.csv`
- `Synthetic files parameter sweep/rbf_robust_sweep/rbf_robust_sweep_summary.csv`
- `Synthetic files parameter sweep/rbf_robust_sweep/rbf_robust_*_plot.png` (config-vs-mode summaries)

Success:
- Plots clearly show whether `sample_trim` or `irls_huber` is better per mode and in aggregate.
"""

RUN_RBF_ROBUST_MODE_AGGREGATION = True
if RUN_RBF_ROBUST_MODE_AGGREGATION:
    tidy_path, summary_path = _write_study_aggregates(
        "rbf_robust_sweep",
        {
            "sample_trim": results_root("rbf_robust_sample_trim"),
            "irls_huber": results_root("rbf_robust_irls_huber"),
        },
    )
    print(f"[OK] {tidy_path}")
    print(f"[OK] {summary_path}")

    summary = pd.read_csv(summary_path)
    out_dir = results_root("rbf_robust_sweep")
    generated: list[Path] = []
    plot_specs = [
        ("overall_rms_error", "RBF robust sweep: overall RMS by mode and config"),
        ("drift_bias_abs", "RBF robust sweep: drift bias by mode and config"),
    ]
    for metric_name, title in plot_specs:
        p_bar = out_dir / f"rbf_robust_{metric_name}_grouped_bar_plot.png"
        if _plot_config_grouped_bars(summary, metric_name=metric_name, out_path=p_bar, title=title, method_name="rbf"):
            generated.append(p_bar)
        p_hm = out_dir / f"rbf_robust_{metric_name}_heatmap_plot.png"
        if _plot_config_heatmap(summary, metric_name=metric_name, out_path=p_hm, title=title, method_name="rbf"):
            generated.append(p_hm)

    if generated:
        readme = out_dir / "rbf_robust_plots_README.txt"
        readme.write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"summary_csv: {summary_path}",
                    "description: Means and std over baseline files, grouped by mode and robust config.",
                    "plots:",
                    *[f"- {p.name}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] Wrote RBF robust plots: {len(generated)}")
# %% ST-RBF Robust Mode Sweep (All Baseline Modes)
"""Run ST-RBF robust-mode sweep across all baseline files and modes.

Question answered:
- How do sample_trim, irls_huber, and hybrid_track_irls compare under identical data coverage?

Datasets:
- `_groundtruth` all files.

Artifacts:
- Per-config outputs under `Synthetic files parameter sweep/st_rbf_robust_*`.
"""

RUN_ST_RBF_ROBUST_MODE_SWEEP = True
if RUN_ST_RBF_ROBUST_MODE_SWEEP:
    ensure_baseline_groundtruth()
    runs = [
        ("sample_trim", "st_rbf_robust_sample_trim"),
        ("irls_huber", "st_rbf_robust_irls_huber"),
        ("hybrid_track_irls", "st_rbf_robust_hybrid_track_irls"),
    ]
    for robust_mode, tag in runs:
        opts_st = dict(OPTIONS)
        opts_st["st_robust_mode"] = robust_mode
        out_root = results_root(tag)
        for file_path in list_baseline_imaris_files():
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = out_root / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=["st_rbf"], opts=opts_st)
            postprocess_outputs(df_syn, file_path, out_prefix, corr)


# ST-RBF Robust Mode Aggregation:
"""Aggregate ST-RBF robust-mode sweep into canonical tidy and summary CSVs.

Question answered:
- Which robust mode is best overall and by baseline mode.

Datasets:
- Reads `Synthetic files parameter sweep/st_rbf_robust_*`.

Artifacts:
- `Synthetic files parameter sweep/st_rbf_robust_sweep/st_rbf_robust_sweep_aggregated.csv`
- `Synthetic files parameter sweep/st_rbf_robust_sweep/st_rbf_robust_sweep_summary.csv`
- `Synthetic files parameter sweep/st_rbf_robust_sweep/st_rbf_robust_*_plot.png` (config-vs-mode summaries)
"""

RUN_ST_RBF_ROBUST_MODE_AGGREGATION = True
if RUN_ST_RBF_ROBUST_MODE_AGGREGATION:
    tidy_path, summary_path = _write_study_aggregates(
        "st_rbf_robust_sweep",
        {
            "sample_trim": results_root("st_rbf_robust_sample_trim"),
            "irls_huber": results_root("st_rbf_robust_irls_huber"),
            "hybrid_track_irls": results_root("st_rbf_robust_hybrid_track_irls"),
        },
    )
    print(f"[OK] {tidy_path}")
    print(f"[OK] {summary_path}")
    summary = pd.read_csv(summary_path)
    out_dir = results_root("st_rbf_robust_sweep")
    generated: list[Path] = []
    plot_specs = [
        ("overall_rms_error", "ST-RBF robust sweep: overall RMS by mode and config"),
        ("drift_bias_abs", "ST-RBF robust sweep: drift bias by mode and config"),
    ]
    for metric_name, title in plot_specs:
        p_bar = out_dir / f"st_rbf_robust_{metric_name}_grouped_bar_plot.png"
        if _plot_config_grouped_bars(
            summary,
            metric_name=metric_name,
            out_path=p_bar,
            title=title,
            method_name="st_rbf",
        ):
            generated.append(p_bar)
        p_hm = out_dir / f"st_rbf_robust_{metric_name}_heatmap_plot.png"
        if _plot_config_heatmap(
            summary,
            metric_name=metric_name,
            out_path=p_hm,
            title=title,
            method_name="st_rbf",
        ):
            generated.append(p_hm)

    if generated:
        readme = out_dir / "st_rbf_robust_plots_README.txt"
        readme.write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"summary_csv: {summary_path}",
                    "description: Means and std over baseline files, grouped by mode and robust config.",
                    "plots:",
                    *[f"- {p.name}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] Wrote ST-RBF robust plots: {len(generated)}")
# %% ST-RBF Expressivity Sweep (All Baseline Modes)
"""Run ST-RBF expressivity settings across all baseline files.

Question answered:
- Do anisotropic scales and/or linear interaction terms improve fit quality across modes?

Datasets:
- `_groundtruth` all files.

Artifacts:
- Per-config outputs under `Synthetic files parameter sweep/st_rbf_expressivity_*`.
"""

RUN_ST_RBF_EXPRESSIVITY_SWEEP = True
if RUN_ST_RBF_EXPRESSIVITY_SWEEP:
    ensure_baseline_groundtruth()
    runs = [
        (
            "st_rbf_expressivity_anisotropic_off",
            {
                "st_ell_space": 2.0,
                "st_ell_time": 0.8,
                "st_ell_x": None,
                "st_ell_y": None,
                "st_ell_z": None,
                "st_linear_interactions": 0,
                "rbf_linear_terms": 0,
            },
        ),
        (
            "st_rbf_expressivity_anisotropic_on",
            {
                "st_ell_space": 2.0,
                "st_ell_time": 0.8,
                "st_ell_x": 1.6,
                "st_ell_y": 2.3,
                "st_ell_z": 1.2,
                "st_linear_interactions": 0,
                "rbf_linear_terms": 0,
            },
        ),
        (
            "st_rbf_expressivity_linear_terms_off",
            {
                "st_ell_space": 2.0,
                "st_ell_time": 0.8,
                "st_ell_x": None,
                "st_ell_y": None,
                "st_ell_z": None,
                "st_linear_interactions": 0,
                "rbf_linear_terms": 0,
            },
        ),
        (
            "st_rbf_expressivity_linear_terms_on",
            {
                "st_ell_space": 2.0,
                "st_ell_time": 0.8,
                "st_ell_x": None,
                "st_ell_y": None,
                "st_ell_z": None,
                "st_linear_interactions": 1,
                "rbf_linear_terms": 1,
            },
        ),
    ]
    for tag, cfg in runs:
        out_root = results_root(tag)
        opts_st = dict(OPTIONS)
        opts_st.update(cfg)
        for file_path in list_baseline_imaris_files():
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = out_root / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=["st_rbf"], opts=opts_st)
            postprocess_outputs(df_syn, file_path, out_prefix, corr)


# ST-RBF Expressivity Aggregation:
"""Aggregate ST-RBF expressivity sweep into tidy and summary CSVs.

Question answered:
- Which expressivity toggle wins by mode and overall.

Datasets:
- Reads `Synthetic files parameter sweep/st_rbf_expressivity_*`.

Artifacts:
- `Synthetic files parameter sweep/st_rbf_expressivity_sweep/st_rbf_expressivity_sweep_aggregated.csv`
- `Synthetic files parameter sweep/st_rbf_expressivity_sweep/st_rbf_expressivity_sweep_summary.csv`
- `Synthetic files parameter sweep/st_rbf_expressivity_sweep/st_rbf_expressivity_*_plot.png` (config-vs-mode summaries)
"""

RUN_ST_RBF_EXPRESSIVITY_AGGREGATION = True
if RUN_ST_RBF_EXPRESSIVITY_AGGREGATION:
    tidy_path, summary_path = _write_study_aggregates(
        "st_rbf_expressivity_sweep",
        {
            "anisotropic_off": results_root("st_rbf_expressivity_anisotropic_off"),
            "anisotropic_on": results_root("st_rbf_expressivity_anisotropic_on"),
            "linear_terms_off": results_root("st_rbf_expressivity_linear_terms_off"),
            "linear_terms_on": results_root("st_rbf_expressivity_linear_terms_on"),
        },
    )
    print(f"[OK] {tidy_path}")
    print(f"[OK] {summary_path}")
    summary = pd.read_csv(summary_path)
    out_dir = results_root("st_rbf_expressivity_sweep")
    generated: list[Path] = []
    plot_specs = [
        ("overall_rms_error", "ST-RBF expressivity sweep: overall RMS by mode and config"),
        ("drift_bias_abs", "ST-RBF expressivity sweep: drift bias by mode and config"),
    ]
    for metric_name, title in plot_specs:
        p_bar = out_dir / f"st_rbf_expressivity_{metric_name}_grouped_bar_plot.png"
        if _plot_config_grouped_bars(
            summary,
            metric_name=metric_name,
            out_path=p_bar,
            title=title,
            method_name="st_rbf",
        ):
            generated.append(p_bar)
        p_hm = out_dir / f"st_rbf_expressivity_{metric_name}_heatmap_plot.png"
        if _plot_config_heatmap(
            summary,
            metric_name=metric_name,
            out_path=p_hm,
            title=title,
            method_name="st_rbf",
        ):
            generated.append(p_hm)

    if generated:
        readme = out_dir / "st_rbf_expressivity_plots_README.txt"
        readme.write_text(
            "\n".join(
                [
                    f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                    f"summary_csv: {summary_path}",
                    "description: Means and std over baseline files, grouped by mode and expressivity config.",
                    "plots:",
                    *[f"- {p.name}" for p in generated],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] Wrote ST-RBF expressivity plots: {len(generated)}")

# %% ST-RBF Parameter Grid Sweep (3x3)
"""Run ST-RBF parameter grid sweep using baseline datasets across all modes.

Question answered:
- Which (ell_space, ell_time) pair minimizes RMS/bias across mode-diverse baseline data?

Datasets:
- `_groundtruth` as source for reuse-existing benchmark inputs.

Artifacts:
- Grid run folders and aggregated tables under `Synthetic files parameter sweep/st_rbf_parameter_selection`.
"""

RUN_ST_RBF_GRID_SWEEP = True
if RUN_ST_RBF_GRID_SWEEP:
    baseline_dir = ensure_baseline_groundtruth()
    grid_root = results_root("st_rbf_parameter_selection")
    grid_root.mkdir(parents=True, exist_ok=True)
    for es in ES_VALUES:
        for et in ET_VALUES:
            synthetic_regime_benchmark(
                synthetic_dir=str(baseline_dir),
                out=str(grid_root / f"st_rbf_grid_es{es}_et{et}"),
                reuse_existing=1,
                methods="st_rbf",
                synthetic_n_seeds=10,
                st_ell_space=float(es),
                st_ell_time=float(et))

# ST-RBF Grid Aggregation:
"""Aggregate ST-RBF grid sweep into canonical CSV artifacts.

Question answered:
- Which ST-RBF grid points are robust across regimes.

Datasets:
- Reads `Synthetic files parameter sweep/st_rbf_parameter_selection/st_rbf_grid_es*_et*`.

Artifacts:
- `st_rbf_grid_aggregated.csv` + README in the same result root.
- `st_rbf_grid_aggregation_*_plot.png` (per-regime and across-regime heatmaps for RMS and drift bias).
"""

RUN_ST_RBF_GRID_AGGREGATION = True
if RUN_ST_RBF_GRID_AGGREGATION:
    grid_root = results_root("st_rbf_parameter_selection")
    grid_root.mkdir(parents=True, exist_ok=True)
    aggregated = _aggregate_st_rbf_grid(grid_root)
    aggregated_csv = grid_root / "st_rbf_grid_aggregated.csv"
    aggregated.to_csv(aggregated_csv, index=False)
    generated: list[Path] = []
    if "regime" in aggregated.columns:
        for regime in sorted(aggregated["regime"].dropna().astype(str).unique().tolist()):
            reg = aggregated[aggregated["regime"].astype(str) == regime].copy()
            p_rms = grid_root / f"st_rbf_grid_aggregation_mean_rms_all__regime_{regime}.png"
            if _plot_grid_heatmap(reg, "mean_rms_all", f"{regime}: mean_rms_all", p_rms):
                generated.append(p_rms)
            p_bias = grid_root / f"st_rbf_grid_aggregation_mean_drift_bias_abs__regime_{regime}.png"
            if _plot_grid_heatmap(reg, "mean_drift_bias_abs", f"{regime}: mean_drift_bias_abs", p_bias):
                generated.append(p_bias)
    p_all_rms = grid_root / "st_rbf_grid_aggregation_mean_rms_all__across_regimes.png"
    if _plot_grid_heatmap(
        aggregated,
        "mean_rms_all",
        "ST-RBF grid: mean_rms_all across regimes",
        p_all_rms,
    ):
        generated.append(p_all_rms)
    p_all_bias = grid_root / "st_rbf_grid_aggregation_mean_drift_bias_abs__across_regimes.png"
    if _plot_grid_heatmap(
        aggregated,
        "mean_drift_bias_abs",
        "ST-RBF grid: mean_drift_bias_abs across regimes",
        p_all_bias,
    ):
        generated.append(p_all_bias)

    (grid_root / "st_rbf_grid_aggregated_README.txt").write_text(
        "\n".join(
            [
                f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                f"baseline_groundtruth: {GROUNDTRUTH_ROOT}",
                f"ell_space_values: {ES_VALUES}",
                f"ell_time_values: {ET_VALUES}",
                f"aggregated_csv: {aggregated_csv}",
                "plots:",
                *[f"- {p.name}" for p in generated],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if generated:
        print(f"[OK] Wrote ST-RBF grid aggregation plots: {len(generated)}")
# %% ST-RBF Stationary Subset A/B
"""Run and aggregate stationary subset A/B tests for ST-RBF parameter selection.

Question answered:
- Should stationary-only fitting be enabled by default for ST-RBF parameter selection setup?

Datasets:
- `_groundtruth` reused for benchmark runs.

Artifacts:
- `st_rbf_stationary_ab_summary.csv`, `st_rbf_stationary_ab_report.md`.
"""

RUN_ST_RBF_STATIONARY_AB_TESTS = True
if RUN_ST_RBF_STATIONARY_AB_TESTS:
    baseline_dir = ensure_baseline_groundtruth()
    selection_root = results_root("st_rbf_parameter_selection")
    selection_root.mkdir(parents=True, exist_ok=True)
    _run_stationary_ab_tests(synthetic_dir=baseline_dir, out_root=selection_root)
    summary = _aggregate_stationary_ab(out_root=selection_root)
    summary_csv = selection_root / "st_rbf_stationary_ab_summary.csv"
    summary.to_csv(summary_csv, index=False)
    _write_stationary_ab_report(summary, selection_root / "st_rbf_stationary_ab_report.md")
# %% ST-RBF Parameter Selection Plots
"""Generate compact parameter-selection plots from canonical ST-RBF aggregate CSVs.

Question answered:
- Are selected defaults visually justified across regimes and stationary subset A/B settings?

Datasets:
- Reads aggregated CSVs from `Synthetic files parameter sweep/st_rbf_parameter_selection`.

Artifacts:
- Heatmaps and A/B plots plus plotting README in selection root.
"""

RUN_ST_RBF_SELECTION_PLOTS = True
if RUN_ST_RBF_SELECTION_PLOTS:
    selection_root = results_root("st_rbf_parameter_selection")
    generated: list[Path] = []
    grid_csv = selection_root / "st_rbf_grid_aggregated.csv"
    ab_csv = selection_root / "st_rbf_stationary_ab_summary.csv"
    if grid_csv.exists():
        grid = pd.read_csv(grid_csv)
        if "regime" in grid.columns:
            for regime in sorted(grid["regime"].dropna().astype(str).unique().tolist()):
                reg = grid[grid["regime"].astype(str) == regime].copy()
                p1 = selection_root / f"st_rbf_grid_heatmap_mean_rms_all__regime_{regime}.png"
                if _plot_grid_heatmap(reg, "mean_rms_all", f"{regime}: mean_rms_all", p1):
                    generated.append(p1)
                p2 = selection_root / f"st_rbf_grid_heatmap_mean_drift_bias_abs__regime_{regime}.png"
                if _plot_grid_heatmap(reg, "mean_drift_bias_abs", f"{regime}: mean_drift_bias_abs", p2):
                    generated.append(p2)
    if ab_csv.exists():
        ab = pd.read_csv(ab_csv)
        if "regime" in ab.columns:
            for regime in sorted(ab["regime"].dropna().astype(str).unique().tolist()):
                p = selection_root / f"st_rbf_stationary_subset_ab__mean_rms_all__regime_{regime}.png"
                if _plot_stationary_subset_ab(ab, regime=regime, out_path=p):
                    generated.append(p)
            p_all = selection_root / "st_rbf_stationary_subset_ab__mean_rms_all__all_regimes.png"
            if _plot_stationary_subset_ab(ab, regime=None, out_path=p_all):
                generated.append(p_all)
    _write_st_rbf_plot_readme(selection_root, generated, grid_csv, ab_csv)
# %% Rigid/Local Fairness Sweeps
"""Run bounded fairness sweeps and aggregate outputs from canonical benchmark folders.

Question answered:
- Are rigid/local baselines competitive because of defaults or due to genuine method behavior?

Datasets:
- Baseline `_groundtruth` reused for synthetic benchmark entry point.

Artifacts:
- Sweep folders + aggregated CSV + line/heatmap plots under `Synthetic files parameter sweep/rigid_fairness_sweep` and `Synthetic files parameter sweep/local_fairness_sweep`.
"""

RUN_RIGID_FAIRNESS_SWEEP = True
RUN_LOCAL_FAIRNESS_SWEEP = True

if RUN_RIGID_FAIRNESS_SWEEP:
    baseline_dir = ensure_baseline_groundtruth()
    rigid_root = results_root("rigid_fairness_sweep")
    rigid_root.mkdir(parents=True, exist_ok=True)
    for keep_fraction in RIGID_KEEP_FRACTIONS:
        for inlier_iters in RIGID_INLIER_ITERS:
            tag = f"kf{_fmt_float_token(keep_fraction, 2)}_inlier{int(inlier_iters)}"
            out_dir = rigid_root / tag
            if _has_completed_regime_run(out_dir, method="rigid"):
                continue
            synthetic_regime_benchmark(
                synthetic_dir=str(baseline_dir),
                out=str(out_dir),
                reuse_existing=1,
                methods="rigid",
                synthetic_n_seeds=int(FAIRNESS_SYNTHETIC_N_SEEDS),
                keep_fraction=float(keep_fraction),
                inlier_reestimate_iters=int(inlier_iters),
            )
    rigid_agg = _aggregate_fairness_sweep(rigid_root, method="rigid", sweep_kind="rigid")
    if not rigid_agg.empty:
        rigid_csv = rigid_root / "rigid_fairness_aggregated.csv"
        rigid_agg.to_csv(rigid_csv, index=False)
        rigid_plots: list[Path] = []
        rigid_plots.extend(
            _plot_lines_by_regime(
                rigid_agg,
                x_col="keep_fraction",
                line_col="inlier_reestimate_iters",
                y_col="mean_rms_all",
                out_path_base=rigid_root / "plot_rigid_lines_by_regime_mean_rms_all",
                title_prefix="Rigid fairness sweep: mean_rms_all by regime",
                x_label="keep_fraction",
                legend_label="inlier_reestimate_iters",
            )
        )
        rigid_plots.extend(
            _plot_heatmaps_by_regime(
                rigid_agg,
                x_col="keep_fraction",
                y_col="inlier_reestimate_iters",
                value_col="mean_rms_all",
                out_path_base=rigid_root / "plot_rigid_heatmaps_by_regime_mean_rms_all",
                title_prefix="Rigid fairness sweep heatmaps: mean_rms_all",
                x_label="keep_fraction",
                y_label="inlier_reestimate_iters",
            )
        )
        _write_fairness_sweep_readme(
            sweep_root=rigid_root,
            synthetic_dir=baseline_dir,
            aggregated_csv=rigid_csv,
            generated_plots=rigid_plots,
            sweep_desc="rigid fairness sweep",
            grid_desc=f"keep_fraction={RIGID_KEEP_FRACTIONS}, inlier_reestimate_iters={RIGID_INLIER_ITERS}, n_seeds={FAIRNESS_SYNTHETIC_N_SEEDS}",
            n_regimes=int(rigid_agg["regime"].nunique()),
        )

if RUN_LOCAL_FAIRNESS_SWEEP:
    baseline_dir = ensure_baseline_groundtruth()
    local_root = results_root("local_fairness_sweep")
    local_root.mkdir(parents=True, exist_ok=True)
    for k_neighbors in LOCAL_K_NEIGHBORS:
        for ell_field in LOCAL_ELL_FIELD:
            field_token = _fmt_float_token(float(ell_field), 0) if float(ell_field).is_integer() else _fmt_float_token(float(ell_field), 2)
            tag = f"k{int(k_neighbors)}_field{field_token}"
            out_dir = local_root / tag
            if _has_completed_regime_run(out_dir, method="local"):
                continue
            synthetic_regime_benchmark(
                synthetic_dir=str(baseline_dir),
                out=str(out_dir),
                reuse_existing=1,
                methods="local",
                synthetic_n_seeds=int(FAIRNESS_SYNTHETIC_N_SEEDS),
                k_neighbors=int(k_neighbors),
                ell_um_field=float(ell_field),
            )
    local_agg = _aggregate_fairness_sweep(local_root, method="local", sweep_kind="local")
    if not local_agg.empty:
        local_csv = local_root / "local_fairness_aggregated.csv"
        local_agg.to_csv(local_csv, index=False)
        local_plots: list[Path] = []
        local_plots.extend(
            _plot_lines_by_regime(
                local_agg,
                x_col="k_neighbors",
                line_col="ell_um_field",
                y_col="mean_rms_all",
                out_path_base=local_root / "plot_local_lines_by_regime_mean_rms_all",
                title_prefix="Local fairness sweep: mean_rms_all by regime",
                x_label="k_neighbors",
                legend_label="ell_um_field",
            )
        )
        local_plots.extend(
            _plot_heatmaps_by_regime(
                local_agg,
                x_col="k_neighbors",
                y_col="ell_um_field",
                value_col="mean_rms_all",
                out_path_base=local_root / "plot_local_heatmaps_by_regime_mean_rms_all",
                title_prefix="Local fairness sweep heatmaps: mean_rms_all",
                x_label="k_neighbors",
                y_label="ell_um_field",
            )
        )
        _write_fairness_sweep_readme(
            sweep_root=local_root,
            synthetic_dir=baseline_dir,
            aggregated_csv=local_csv,
            generated_plots=local_plots,
            sweep_desc="local fairness sweep",
            grid_desc=f"k_neighbors={LOCAL_K_NEIGHBORS}, ell_um_field={LOCAL_ELL_FIELD}, n_seeds={FAIRNESS_SYNTHETIC_N_SEEDS}",
            n_regimes=int(local_agg["regime"].nunique()),
        )

# %% Paper Regression Harness (Optional)
"""Run optional paper regression harness for output-manifest verification.

Question answered:
- Are canonical paper artifacts still present after refactors?

Datasets:
- Uses `_groundtruth` as reusable synthetic input source.

Artifacts:
- Regression report folder under `Synthetic files parameter sweep/paper_regression`.
"""

RUN_PAPER_REGRESSION_CHECK = True
if RUN_PAPER_REGRESSION_CHECK:
    baseline_dir = ensure_baseline_groundtruth()
    paper_regression(
        synthetic_dir=str(baseline_dir),
        out=str(results_root("paper_regression")),
        manifest=str(PROJECT_ROOT / "paper_outputs_manifest.txt"),
        columns_manifest=str(PROJECT_ROOT / "paper_metrics_columns.json"),
        methods="rbf,rigid,local,st_rbf",
        reuse_existing=1,
    )


# %% END
