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
SYNTH_REPRO_ROOT = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark"
SYNTH_REPRO_RESULTS = SYNTH_REPRO_ROOT / "driftcorr_results"


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


# %% PAPER SUMMARY PLOTS (MoE + robust modes, no rigid/local)
RUN_PAPER_SUMMARY_PLOTS = False
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
