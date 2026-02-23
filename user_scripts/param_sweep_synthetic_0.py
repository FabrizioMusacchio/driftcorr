"""Tutorial script that demonstrates how to run driftcorr from Python.

This script is intentionally organized in notebook-like cells (`# %% ...`) so users can
run it in VS Code/PyCharm cell mode or as a normal Python script.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re
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
        generate_synthetic_imaris_folder,
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
            generate_synthetic_imaris_folder,
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
IMARIS_DIR = PROJECT_ROOT / "IMARIS xls files"
RESULTS_ROOT = IMARIS_DIR / "driftcorr_results_from_python"

DEFAULT_METHODS = ["rbf", "rigid", "local", "st_rbf"]

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

# %% FUNCTIONS
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
    corr_by_method: dict[str, pd.DataFrame],
) -> None:
    """Create standard evaluation + visual plots + optional groundtruth evaluation."""
    summary_df, details = run_all_methods_and_compare(df_orig, corr_by_method)
    save_evaluation_outputs(out_prefix, summary_df, details)

    gt = load_groundtruth_for_imaris_file(imaris_file)
    drift_gt = load_drift_field_for_imaris_file(imaris_file)
    for method_name, df_corr in corr_by_method.items():
        save_visual_inspection_plots(df_orig, df_corr, out_prefix, method_name, df_groundtruth=gt)

    if gt is not None:
        save_groundtruth_evaluation_outputs(out_prefix, gt, corr_by_method, drift_field_groundtruth=drift_gt)


# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: DEFAULT SETTINGS
RUN_SYNTHETIC = False
if RUN_SYNTHETIC:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files tutorial run 2"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)

    synthetic_results = synthetic_root / "driftcorr_results"

    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix_syn = synthetic_results / file_path.stem / file_path.stem

        corr_syn = run_methods_for_dataframe(
            df=df_syn,
            out_prefix=out_prefix_syn,
            methods=DEFAULT_METHODS,
            opts=OPTIONS,
            use_stationary_subset=False)
        postprocess_outputs(df_syn, file_path, out_prefix_syn, corr_syn)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: RBF center-mode + bias + coverage diagnostics

RUN_SYNTHETIC_JOB_A_EXAMPLE = False
if RUN_SYNTHETIC_JOB_A_EXAMPLE:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files jobA"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts_a = dict(OPTIONS)
    opts_a["center_mode"] = "kmeans_pooled"
    opts_a["center_pooled_time_stride"] = 1
    opts_a["two_stage"] = 1
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts_a)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: RBF robust-mode track_trim
RUN_SYNTHETIC_JOB_B_EXAMPLE = False
if RUN_SYNTHETIC_JOB_B_EXAMPLE:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files jobB"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts_b = dict(OPTIONS)
    opts_b["robust_mode"] = "track_trim"
    opts_b["track_trim_iters"] = 4
    opts_b["center_mode"] = "pooled"
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts_b)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: stationary-only fitting for RBF/ST-RBF
RUN_SYNTHETIC_JOB_C_EXAMPLE = False
if RUN_SYNTHETIC_JOB_C_EXAMPLE:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files jobC"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts_c = dict(OPTIONS)
    opts_c["robust_mode"] = "track_trim"
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(
            df=df_syn,
            out_prefix=out_prefix,
            methods=DEFAULT_METHODS,
            opts=opts_c,
            use_stationary_subset=True,
        )
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: ST-RBF anchor mode + normalization + refinement
RUN_SYNTHETIC_JOB_D_EXAMPLE = False
if RUN_SYNTHETIC_JOB_D_EXAMPLE:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files jobD"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts_d = dict(OPTIONS)
    opts_d["rbf_space_input_mode"] = "track_anchor"
    opts_d["space_input_mode"] = "ref_position"
    opts_d["st_refine_iters"] = 2
    opts_d["two_stage"] = 1
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts_d)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: two-stage translation + residual field
RUN_SYNTHETIC_JOB_E_EXAMPLE = False
if RUN_SYNTHETIC_JOB_E_EXAMPLE:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files jobE"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts_e = dict(OPTIONS)
    opts_e["two_stage"] = 1
    opts_e["center_mode"] = "pooled"
    opts_e["rbf_space_input_mode"] = "track_anchor"
    opts_e["space_input_mode"] = "track_anchor"
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts_e)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: Task 1 (RBF anchor coordinates)
RUN_SYNTHETIC_TASK1_RBF_ANCHOR = False
if RUN_SYNTHETIC_TASK1_RBF_ANCHOR:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files task1_rbf_anchor"
    #generate_synthetic_imaris_folder(out_dir=synthetic_root, n_files=2, n_tracks=80, n_times=20, seed=42)
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=5,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts = dict(OPTIONS)
    opts["rbf_space_input_mode"] = "track_anchor"
    opts["space_input_mode"] = "ref_position"
    opts["center_mode"] = "ref"
    opts["temporal_gamma"] = 0.6
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: Task 2+3 (ST-RBF residual refinement + one-time two-stage)
RUN_SYNTHETIC_TASK23_ST_REFINEMENT = False
if RUN_SYNTHETIC_TASK23_ST_REFINEMENT:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files task23_st_refine"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=5,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts = dict(OPTIONS)
    opts["rbf_space_input_mode"] = "track_anchor"
    opts["space_input_mode"] = "ref_position"
    opts["st_refine_iters"] = 3
    opts["two_stage"] = 1
    opts["st_robust_mode"] = "sample_trim"
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: Task 4 (ST-RBF track-level robust fitting)
RUN_SYNTHETIC_TASK4_ST_TRACK_TRIM = False
if RUN_SYNTHETIC_TASK4_ST_TRACK_TRIM:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files task4_st_tracktrim"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=5,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts = dict(OPTIONS)
    opts["st_robust_mode"] = "track_trim"
    opts["st_track_trim_iters"] = 4
    opts["st_robust_keep_fraction"] = 0.9
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: Task 5 (bias + optional linear terms)
RUN_SYNTHETIC_TASK5_LINEAR_TERMS = False
if RUN_SYNTHETIC_TASK5_LINEAR_TERMS:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files task5_linear_terms"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=5,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts = dict(OPTIONS)
    opts["rbf_linear_terms"] = 1
    opts["two_stage"] = 1
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: Task 6 (RBF row_position fixed-point correction)
RUN_SYNTHETIC_TASK6_RBF_FIXED_POINT = False
if RUN_SYNTHETIC_TASK6_RBF_FIXED_POINT:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files task6_rbf_fixed_point"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=5,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9)
    opts = dict(OPTIONS)
    opts["rbf_space_input_mode"] = "row_position"
    opts["space_input_mode"] = "ref_position"
    opts["row_position_iters"] = 5
    opts["row_position_tol"] = 1e-4
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=DEFAULT_METHODS, opts=opts)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: ST-RBF MoE comparison (K=2,3,5)
RUN_SYNTHETIC_ST_MOE = False
if RUN_SYNTHETIC_ST_MOE:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files st_moe"
    generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=2,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9,
        local_mode="piecewise_smooth",
    )
    opts = dict(OPTIONS)
    opts["st_moe_k_list"] = [2, 3, 5]
    methods = ["rigid", "local", "st_rbf", "st_rbf_moe"]
    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_root / "driftcorr_results" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix, methods=methods, opts=opts)
        postprocess_outputs(df_syn, file_path, out_prefix, corr)

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: Regime suite (3 regimes x 5 seeds)
RUN_SYNTHETIC_REGIME_SUITE = False
if RUN_SYNTHETIC_REGIME_SUITE:
    synthetic_regime_benchmark(
        synthetic_dir=str(PROJECT_ROOT / "synthetic_regime_suite"),
        methods="rbf,rigid,local,st_rbf,st_rbf_moe",
        synthetic_n_seeds=5,
        st_moe_k_list="2,3,5",
    )

# %% HOW TO GENERATE SYNTHETIC DATA ONLY (Python API synthetic_generate)
RUN_SYNTHETIC_GENERATE_ONLY = False
if RUN_SYNTHETIC_GENERATE_ONLY:
    synthetic_generate(
        synthetic_dir=str(PROJECT_ROOT / "Synthetic IMARIS files generated_only"),
        synthetic_seed=123,
        synthetic_n_files=2,
    )

# %% HOW TO RUN PAPER OUTPUT REGRESSION HARNESS (just checks whether outputs plots still match expected)
RUN_PAPER_OUTPUT_REGRESSION = False
if RUN_PAPER_OUTPUT_REGRESSION:
    paper_regression(
        synthetic_dir=str(PROJECT_ROOT / "Synthetic IMARIS files paper_regression"),
        manifest=str(PROJECT_ROOT / "paper_outputs_manifest.txt"),
        columns_manifest=str(PROJECT_ROOT / "paper_metrics_columns.json"),
        methods="rbf,rigid,local,st_rbf",
        reuse_existing=1,
    )

# %% HOW TO GENERATE AND RUN SYNTHETIC DATA: ST-RBF per-axis scaling A/B comparison
RUN_SYNTHETIC_ST_SCALING_AB = False
if RUN_SYNTHETIC_ST_SCALING_AB:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files st_scaling_ab"
    generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=1,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9,
        local_mode="smooth",
    )
    file_path = list_imaris_xls_files(synthetic_root)[0]
    df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))

    opts_on = dict(OPTIONS)
    opts_on["st_per_axis_scaling"] = 1
    out_prefix_on = synthetic_root / "driftcorr_results" / "scaling_on" / file_path.stem
    corr_on = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix_on, methods=["st_rbf"], opts=opts_on)
    postprocess_outputs(df_syn, file_path, out_prefix_on, corr_on)

    opts_off = dict(OPTIONS)
    opts_off["st_per_axis_scaling"] = 0
    out_prefix_off = synthetic_root / "driftcorr_results" / "scaling_off" / file_path.stem
    corr_off = run_methods_for_dataframe(df=df_syn, out_prefix=out_prefix_off, methods=["st_rbf"], opts=opts_off)
    postprocess_outputs(df_syn, file_path, out_prefix_off, corr_off)

# %% ## Synthetic benchmark: reproducible dataset + new option A/B tests (helpers)
SYNTH_REPRO_ROOT = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark 2"
SYNTH_REPRO_RESULTS = SYNTH_REPRO_ROOT / "driftcorr_results"
FAIRNESS_SYNTHETIC_N_SEEDS = 10
RIGID_KEEP_FRACTIONS = [0.5, 0.7, 0.85, 0.95]
RIGID_INLIER_ITERS = [5, 15, 30]
LOCAL_K_NEIGHBORS = [8, 12, 20, 30]
LOCAL_ELL_FIELD = [80.0, 120.0, 180.0, 240.0]


def _ensure_repro_dataset_exists() -> Path:
    """API-first equivalent of --reuse-existing: generate once, then reuse."""
    if list_imaris_xls_files(SYNTH_REPRO_ROOT):
        return SYNTH_REPRO_ROOT
    generate_synthetic_imaris_folder(
        out_dir=SYNTH_REPRO_ROOT,
        n_files=5,
        n_tracks=80,
        n_times=20,
        stationary_fraction=0.45,
        migrating_fraction=0.35,
        seed=42,
        temporal_smoothness=0.9,
        local_mode="smooth",
    )
    return SYNTH_REPRO_ROOT


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


# %% Synthetic benchmark: Cell 1 / generate dataset once (Python API)
RUN_SYNTH_NEW_CELL1_GENERATE_ONCE = False
if RUN_SYNTH_NEW_CELL1_GENERATE_ONCE:
    synthetic_generate(
        synthetic_dir=str(SYNTH_REPRO_ROOT),
        synthetic_seed=42,
        synthetic_n_files=5,
        synthetic_n_tracks=80,
        synthetic_n_times=20,
        synthetic_stationary_fraction=0.45,
        synthetic_migrating_fraction=0.35,
        synthetic_temporal_smoothness=0.9,
        synthetic_local_mode="smooth",
    )
    print(f"Generated synthetic dataset at: {SYNTH_REPRO_ROOT}")


# %% Synthetic benchmark: Cell 2 / baseline run on SAME dataset (rbf, rigid, local, st_rbf, st_rbf_moe)
RUN_SYNTH_NEW_CELL2_BASELINE_API = False
if RUN_SYNTH_NEW_CELL2_BASELINE_API:
    _ensure_repro_dataset_exists()

    baseline_methods = ["rbf", "rigid", "local", "st_rbf", "st_rbf_moe"]
    baseline_opts = dict(OPTIONS)
    baseline_opts["st_moe_k_list"] = [2, 3, 5]

    for file_path in list_imaris_xls_files(SYNTH_REPRO_ROOT):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = SYNTH_REPRO_RESULTS / "baseline_suite" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(
            df=df_syn,
            out_prefix=out_prefix,
            methods=baseline_methods,
            opts=baseline_opts,
            use_stationary_subset=False,
        )
        postprocess_outputs(df_syn, file_path, out_prefix, corr)


# %% Synthetic benchmark: Cell 2b / regime benchmark on fixed data (Python API, reuse_existing)
RUN_SYNTH_NEW_CELL2B_BASELINE_REGIME_API = False
if RUN_SYNTH_NEW_CELL2B_BASELINE_REGIME_API:
    _ensure_repro_dataset_exists()
    synthetic_regime_benchmark(
        synthetic_dir=str(SYNTH_REPRO_ROOT),
        out=str(SYNTH_REPRO_RESULTS / "baseline_suite"),
        reuse_existing=1,
        methods="rbf,rigid,local,st_rbf,st_rbf_moe",
        synthetic_n_seeds=1,
        st_moe_k_list="2,3,5",
    )


# %% Synthetic benchmark: Cell 3 / RBF robust-mode A/B (sample_trim vs irls_huber)
RUN_SYNTH_NEW_CELL3_RBF_ROBUST_AB = False
if RUN_SYNTH_NEW_CELL3_RBF_ROBUST_AB:
    _ensure_repro_dataset_exists()
    for robust_mode, out_tag in [("sample_trim", "rbf_ab_sampletrim"), ("irls_huber", "rbf_ab_irls")]:
        opts_rbf = dict(OPTIONS)
        opts_rbf["robust_mode"] = robust_mode
        for file_path in list_imaris_xls_files(SYNTH_REPRO_ROOT):
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = SYNTH_REPRO_RESULTS / out_tag / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(
                df=df_syn,
                out_prefix=out_prefix,
                methods=["rbf"],
                opts=opts_rbf,
                use_stationary_subset=False,
            )
            postprocess_outputs(df_syn, file_path, out_prefix, corr)


# %% Synthetic benchmark: Cell 4 / ST-RBF robust-mode A/B/C
RUN_SYNTH_NEW_CELL4_ST_ROBUST_ABC = False
if RUN_SYNTH_NEW_CELL4_ST_ROBUST_ABC:
    _ensure_repro_dataset_exists()
    robust_runs = [
        ("sample_trim", "st_rbf_ab_sampletrim"),
        ("irls_huber", "st_rbf_ab_irls"),
        ("hybrid_track_irls", "st_rbf_ab_hybrid_track_irls"),
    ]
    for robust_mode, out_tag in robust_runs:
        opts_st = dict(OPTIONS)
        opts_st["st_robust_mode"] = robust_mode
        for file_path in list_imaris_xls_files(SYNTH_REPRO_ROOT):
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = SYNTH_REPRO_RESULTS / out_tag / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(
                df=df_syn,
                out_prefix=out_prefix,
                methods=["st_rbf"],
                opts=opts_st,
                use_stationary_subset=False,
            )
            postprocess_outputs(df_syn, file_path, out_prefix, corr)


# %% Synthetic benchmark: Cell 5 / ST-RBF expressivity A/B (anisotropic on/off, linear interactions on/off)
RUN_SYNTH_NEW_CELL5_ST_EXPRESSIVITY_AB = False
if RUN_SYNTH_NEW_CELL5_ST_EXPRESSIVITY_AB:
    _ensure_repro_dataset_exists()

    runs = [
        (
            "st_rbf_anisotropic_off",
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
            "st_rbf_anisotropic_on",
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
            "st_rbf_linear_interactions_off",
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
            "st_rbf_linear_interactions_on",
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

    for out_tag, cfg in runs:
        opts_st = dict(OPTIONS)
        opts_st.update(cfg)
        for file_path in list_imaris_xls_files(SYNTH_REPRO_ROOT):
            df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
            out_prefix = SYNTH_REPRO_RESULTS / out_tag / file_path.stem / file_path.stem
            corr = run_methods_for_dataframe(
                df=df_syn,
                out_prefix=out_prefix,
                methods=["st_rbf"],
                opts=opts_st,
                use_stationary_subset=False,
            )
            postprocess_outputs(df_syn, file_path, out_prefix, corr)


# %% Synthetic benchmark: Cell 6 (optional) / paper-regression check on SAME dataset (Python API)
RUN_SYNTH_NEW_CELL6_PAPER_REGRESSION = False
if RUN_SYNTH_NEW_CELL6_PAPER_REGRESSION:
    _ensure_repro_dataset_exists()
    paper_regression(
        synthetic_dir=str(SYNTH_REPRO_ROOT),
        out=str(SYNTH_REPRO_RESULTS / "paper_regression"),
        manifest=str(PROJECT_ROOT / "paper_outputs_manifest.txt"),
        columns_manifest=str(PROJECT_ROOT / "paper_metrics_columns.json"),
        methods="rbf,rigid,local,st_rbf",
        reuse_existing=1,
    )


# %% FAIRNESS SWEEPS: Rigid and Local (bounded grids)
RUN_RIGID_FAIRNESS_SWEEP = False
RUN_LOCAL_FAIRNESS_SWEEP = False

if RUN_RIGID_FAIRNESS_SWEEP:
    _ensure_repro_dataset_exists()
    rigid_root = SYNTH_REPRO_RESULTS / "rigid_fairness_sweep"
    rigid_root.mkdir(parents=True, exist_ok=True)

    for keep_fraction in RIGID_KEEP_FRACTIONS:
        for inlier_iters in RIGID_INLIER_ITERS:
            tag = f"kf{_fmt_float_token(keep_fraction, 2)}_inlier{int(inlier_iters)}"
            out_dir = rigid_root / tag
            if _has_completed_regime_run(out_dir, method="rigid"):
                print(f"[INFO] Reusing rigid sweep run: {out_dir}")
                continue
            print(f"[INFO] Running rigid sweep config: keep_fraction={keep_fraction}, inlier_reestimate_iters={inlier_iters}")
            synthetic_regime_benchmark(
                synthetic_dir=str(SYNTH_REPRO_ROOT),
                out=str(out_dir),
                reuse_existing=1,
                methods="rigid",
                synthetic_n_seeds=int(FAIRNESS_SYNTHETIC_N_SEEDS),
                keep_fraction=float(keep_fraction),
                inlier_reestimate_iters=int(inlier_iters),
            )

    rigid_agg = _aggregate_fairness_sweep(rigid_root, method="rigid", sweep_kind="rigid")
    if rigid_agg.empty:
        print("[WARN] Rigid fairness sweep aggregation produced no rows.")
    else:
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
        rigid_readme = _write_fairness_sweep_readme(
            sweep_root=rigid_root,
            synthetic_dir=SYNTH_REPRO_ROOT,
            aggregated_csv=rigid_csv,
            generated_plots=rigid_plots,
            sweep_desc="rigid fairness sweep",
            grid_desc=f"keep_fraction={RIGID_KEEP_FRACTIONS}, inlier_reestimate_iters={RIGID_INLIER_ITERS}, n_seeds={FAIRNESS_SYNTHETIC_N_SEEDS}",
            n_regimes=int(rigid_agg["regime"].nunique()),
        )
        print(f"[OK] Wrote rigid fairness aggregated CSV: {rigid_csv}")
        print(f"[OK] Wrote rigid fairness README: {rigid_readme}")

if RUN_LOCAL_FAIRNESS_SWEEP:
    _ensure_repro_dataset_exists()
    local_root = SYNTH_REPRO_RESULTS / "local_fairness_sweep"
    local_root.mkdir(parents=True, exist_ok=True)

    for k_neighbors in LOCAL_K_NEIGHBORS:
        for ell_field in LOCAL_ELL_FIELD:
            field_token = _fmt_float_token(float(ell_field), 0) if float(ell_field).is_integer() else _fmt_float_token(float(ell_field), 2)
            tag = f"k{int(k_neighbors)}_field{field_token}"
            out_dir = local_root / tag
            if _has_completed_regime_run(out_dir, method="local"):
                print(f"[INFO] Reusing local sweep run: {out_dir}")
                continue
            print(f"[INFO] Running local sweep config: k_neighbors={k_neighbors}, ell_um_field={ell_field}")
            synthetic_regime_benchmark(
                synthetic_dir=str(SYNTH_REPRO_ROOT),
                out=str(out_dir),
                reuse_existing=1,
                methods="local",
                synthetic_n_seeds=int(FAIRNESS_SYNTHETIC_N_SEEDS),
                k_neighbors=int(k_neighbors),
                ell_um_field=float(ell_field),
            )

    local_agg = _aggregate_fairness_sweep(local_root, method="local", sweep_kind="local")
    if local_agg.empty:
        print("[WARN] Local fairness sweep aggregation produced no rows.")
    else:
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
        local_readme = _write_fairness_sweep_readme(
            sweep_root=local_root,
            synthetic_dir=SYNTH_REPRO_ROOT,
            aggregated_csv=local_csv,
            generated_plots=local_plots,
            sweep_desc="local fairness sweep",
            grid_desc=f"k_neighbors={LOCAL_K_NEIGHBORS}, ell_um_field={LOCAL_ELL_FIELD}, n_seeds={FAIRNESS_SYNTHETIC_N_SEEDS}",
            n_regimes=int(local_agg["regime"].nunique()),
        )
        print(f"[OK] Wrote local fairness aggregated CSV: {local_csv}")
        print(f"[OK] Wrote local fairness README: {local_readme}")

# %% PAPER SUMMARY PLOTS (MoE + robust modes, no rigid/local)
RUN_PAPER_SUMMARY_PLOTS = False

# we need to point in SYNTH_REPRO_RESULTS_1 to the folder, where we have stored 
# above the MoE and robust mode runs (baseline_suite, rbf_ab_sampletrim, rbf_ab_irls, 
# st_rbf_ab_sampletrim, st_rbf_ab_irls, st_rbf_ab_hybrid_track_irls)
#SYNTH_REPRO_RESULTS_1 = Path(" ? ")

if RUN_PAPER_SUMMARY_PLOTS:
    paper_out = SYNTH_REPRO_RESULTS / "paper_summary_plots"
    paper_out.mkdir(parents=True, exist_ok=True)

    # A) ST-RBF MoE k comparison (vs plain ST-RBF)
    moe_root_candidates = [
        SYNTH_REPRO_RESULTS / "baseline_suite",
        PROJECT_ROOT / "Synthetic IMARIS files st_moe" / "driftcorr_results",
    ]
    moe_methods = {
        "st_rbf": "st_rbf",
        "st_rbf_moe_k2": "st_rbf_moe_k2",
        "st_rbf_moe_k3": "st_rbf_moe_k3",
        "st_rbf_moe_k5": "st_rbf_moe_k5",
    }
    moe_label_to_series: dict[str, dict[str, pd.DataFrame]] = {}
    for label, method_name in moe_methods.items():
        best_map: dict[str, pd.DataFrame] = {}
        for root in moe_root_candidates:
            series_map = _load_method_timeseries(root, method_name)
            if len(series_map) > len(best_map):
                best_map = series_map
        if best_map:
            moe_label_to_series[label] = best_map
        else:
            print(f"[WARN] Missing MoE timeseries for {label}")

    _plot_timeseries_comparison(
        moe_label_to_series,
        paper_out / "moe_k_comparison__aggregate.png",
        title="ST-RBF MoE k comparison (aggregate)",
    )
    _plot_timeseries_per_file(moe_label_to_series, paper_out, "moe_k_comparison")

    moe_label_to_vals: dict[str, np.ndarray] = {}
    for label, method_name in moe_methods.items():
        best_vals = np.array([], dtype=float)
        for root in moe_root_candidates:
            vals = _collect_scalar_metric(root, method_name, metric_col="overall_rms_error")
            if vals.size > best_vals.size:
                best_vals = vals
        if best_vals.size > 0:
            moe_label_to_vals[label] = best_vals
    _plot_scalar_summary_bar(
        moe_label_to_vals,
        paper_out / "moe_k_summary_bar__overall_rms.png",
        title="ST-RBF MoE k: overall RMS summary",
        y_label="overall_rms_error",
    )

    # B) Robust mode comparison (RBF): sample_trim vs irls_huber
    rbf_robust_roots = {
        "sample_trim": SYNTH_REPRO_RESULTS / "rbf_ab_sampletrim",
        "irls_huber": SYNTH_REPRO_RESULTS / "rbf_ab_irls",
    }
    rbf_label_to_series = {label: _load_method_timeseries(root, "rbf") for label, root in rbf_robust_roots.items()}
    for label, series_map in rbf_label_to_series.items():
        if not series_map:
            print(f"[WARN] Missing RBF robust timeseries for {label}")
    _plot_timeseries_comparison(
        rbf_label_to_series,
        paper_out / "rbf_robust_mode_ab__aggregate.png",
        title="RBF robust mode A/B (aggregate)",
    )
    _plot_timeseries_per_file(rbf_label_to_series, paper_out, "rbf_robust_mode_ab")

    rbf_label_to_vals = {
        label: _collect_scalar_metric(root, "rbf", metric_col="overall_rms_error")
        for label, root in rbf_robust_roots.items()
    }
    _plot_scalar_summary_bar(
        rbf_label_to_vals,
        paper_out / "rbf_robust_summary_bar__overall_rms.png",
        title="RBF robust mode: overall RMS summary",
        y_label="overall_rms_error",
    )

    # C) Robust mode comparison (ST-RBF): sample_trim vs irls_huber vs hybrid_track_irls
    st_robust_roots = {
        "sample_trim": SYNTH_REPRO_RESULTS / "st_rbf_ab_sampletrim",
        "irls_huber": SYNTH_REPRO_RESULTS / "st_rbf_ab_irls",
        "hybrid_track_irls": SYNTH_REPRO_RESULTS / "st_rbf_ab_hybrid_track_irls",
    }
    st_label_to_series = {label: _load_method_timeseries(root, "st_rbf") for label, root in st_robust_roots.items()}
    for label, series_map in st_label_to_series.items():
        if not series_map:
            print(f"[WARN] Missing ST-RBF robust timeseries for {label}")
    _plot_timeseries_comparison(
        st_label_to_series,
        paper_out / "st_rbf_robust_mode_abc__aggregate.png",
        title="ST-RBF robust mode A/B/C (aggregate)",
    )
    _plot_timeseries_per_file(st_label_to_series, paper_out, "st_rbf_robust_mode_abc")

    st_label_to_vals = {
        label: _collect_scalar_metric(root, "st_rbf", metric_col="overall_rms_error")
        for label, root in st_robust_roots.items()
    }
    _plot_scalar_summary_bar(
        st_label_to_vals,
        paper_out / "st_rbf_robust_summary_bar__overall_rms.png",
        title="ST-RBF robust mode: overall RMS summary",
        y_label="overall_rms_error",
    )

    print(f"[OK] Paper summary plots written to: {paper_out}")


# %% END
