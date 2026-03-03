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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
    from driftcorr.benchmarks import paper_regression, synthetic_generate, synthetic_regime_benchmark
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
    "st_robust_mode": "hybrid_track_irls",
    "st_track_trim_iters": 4,
    "st_per_axis_scaling": 1,
    "st_linear_interactions": 0,
    "st_moe_k": 3,
    "st_moe_k_list": [2, 3, 5],
    "st_moe_blend_sigma_um": 140.0,
    "st_moe_cluster_mode": "track_anchor",
    "st_moe_drift_cap_factor": 3.0,
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

# %% HOW TO RUN A SINGLE FILE

RUN_SINGLE_FILE = False
if RUN_SINGLE_FILE:
    file_path = IMARIS_DIR / "registered_stack_5D_251008_P2_565um_v1.xls"
    df_single = ensure_numeric_tracks(read_imaris_position_xls(file_path))

    out_prefix_single = RESULTS_ROOT / file_path.stem / file_path.stem

    methods_single = ["rbf", "rigid", "local", "st_rbf"]
    corr_single = run_methods_for_dataframe(
        df=df_single,
        out_prefix=out_prefix_single,
        methods=methods_single,
        opts=OPTIONS,
        use_stationary_subset=False,
    )
    postprocess_outputs(df_single, file_path, out_prefix_single, corr_single)

    RUN_SINGLE_FILE_WITH_SMOOTHING = False
    if RUN_SINGLE_FILE_WITH_SMOOTHING:
        opts_single_smooth = dict(OPTIONS)
        opts_single_smooth["rigid_smooth_enable"] = 1
        opts_single_smooth["local_smooth_enable"] = 1
        out_prefix_single_smooth = RESULTS_ROOT / file_path.stem / "with_smoothing" / file_path.stem
        corr_single_smooth = run_methods_for_dataframe(
            df=df_single,
            out_prefix=out_prefix_single_smooth,
            methods=methods_single,
            opts=opts_single_smooth,
            use_stationary_subset=False,
        )
        postprocess_outputs(df_single, file_path, out_prefix_single_smooth, corr_single_smooth)
# %% HOW TO BATCH RUN MULTIPLE FILES

RUN_BATCH = True
if RUN_BATCH:
    methods_batch = ["rbf", "rigid", "local", "st_rbf"]
    RUN_BATCH_SMOOTHED = True
    opts_batch = dict(OPTIONS)
    batch_root = RESULTS_ROOT / "batch_default"
    if RUN_BATCH_SMOOTHED:
        opts_batch["rigid_smooth_enable"] = 1
        opts_batch["local_smooth_enable"] = 1
        batch_root = RESULTS_ROOT / "batch_with_smoothing"

    for file_path in list_imaris_xls_files(IMARIS_DIR):
        df_batch = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix_batch = batch_root / file_path.stem / file_path.stem

        corr_batch = run_methods_for_dataframe(
            df=df_batch,
            out_prefix=out_prefix_batch,
            methods=methods_batch,
            opts=opts_batch,
            use_stationary_subset=False,
        )
        postprocess_outputs(df_batch, file_path, out_prefix_batch, corr_batch)


# %% HOW TO PARAMETER SWEEP ST RBF FOR SMOOTHER RESULTS

RUN_ST_RBF_SWEEP = True
if RUN_ST_RBF_SWEEP:
    ST_RBF_SWEEP_FILES: list[str] = []
    st_ell_time_list = [0.3, 0.4, 0.6, 0.8, 1.0]
    st_lam_list = [5e-4, 8.5e-4, 1.2e-3]
    st_robust_keep_fraction_list = [0.80, 0.83, 0.87]

    if ST_RBF_SWEEP_FILES:
        sweep_files = [IMARIS_DIR / f for f in ST_RBF_SWEEP_FILES]
    else:
        sweep_files = list_imaris_xls_files(IMARIS_DIR)

    sweep_rows: list[dict[str, object]] = []
    for file_path in sweep_files:
        if not file_path.exists():
            continue
        df_sw = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        for ell_t in st_ell_time_list:
            for lam in st_lam_list:
                for keep in st_robust_keep_fraction_list:
                    opts_sw = dict(OPTIONS)
                    opts_sw["st_ell_time"] = float(ell_t)
                    opts_sw["st_lam"] = float(lam)
                    opts_sw["st_robust_keep_fraction"] = float(keep)
                    tag = f"st_sweep_ellT{str(ell_t).replace('.', 'p')}_lam{str(lam).replace('.', 'p')}_keep{str(keep).replace('.', 'p')}"
                    out_prefix_sw = RESULTS_ROOT / "st_rbf_sweep" / file_path.stem / tag / file_path.stem
                    corr_sw = run_methods_for_dataframe(
                        df=df_sw,
                        out_prefix=out_prefix_sw,
                        methods=["st_rbf"],
                        opts=opts_sw,
                        use_stationary_subset=False,
                    )
                    postprocess_outputs(df_sw, file_path, out_prefix_sw, corr_sw)
                    summary_sw, _ = run_all_methods_and_compare(df_sw, corr_sw)
                    if summary_sw.shape[0] == 0:
                        continue
                    row = summary_sw.iloc[0].to_dict()
                    row["file"] = file_path.name
                    row["st_ell_time"] = float(ell_t)
                    row["st_lam"] = float(lam)
                    row["st_robust_keep_fraction"] = float(keep)
                    sweep_rows.append(row)

    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        out_csv = RESULTS_ROOT / "st_rbf_sweep" / "st_rbf_sweep_summary.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(out_csv, index=False)


# %% HOW TO PARAMETER SWEEP RIGID AND LOCAL METHODS

RUN_RIGID_LOCAL_SWEEP = True
if RUN_RIGID_LOCAL_SWEEP:
    RIGID_LOCAL_SWEEP_FILES: list[str] = []
    rigid_keep_fraction_list = [0.50, 0.70, 0.85, 0.95]
    rigid_inlier_reestimate_iters_list = [5, 15, 30]
    local_k_neighbors_list = [8, 12, 20, 30]
    local_ell_um_field_list = [80.0, 120.0, 180.0, 240.0]

    if RIGID_LOCAL_SWEEP_FILES:
        sweep_files = [IMARIS_DIR / f for f in RIGID_LOCAL_SWEEP_FILES]
    else:
        sweep_files = list_imaris_xls_files(IMARIS_DIR)

    sweep_rows: list[dict[str, object]] = []
    for file_path in sweep_files:
        if not file_path.exists():
            continue
        df_sw = ensure_numeric_tracks(read_imaris_position_xls(file_path))

        for keep in rigid_keep_fraction_list:
            for iters in rigid_inlier_reestimate_iters_list:
                opts_sw = dict(OPTIONS)
                opts_sw["keep_fraction"] = float(keep)
                opts_sw["inlier_reestimate_iters"] = int(iters)
                tag = f"rigid_kf{str(keep).replace('.', 'p')}_iters{iters}"
                out_prefix_sw = RESULTS_ROOT / "rigid_local_sweep" / file_path.stem / tag / file_path.stem
                corr_sw = run_methods_for_dataframe(
                    df=df_sw,
                    out_prefix=out_prefix_sw,
                    methods=["rigid"],
                    opts=opts_sw,
                    use_stationary_subset=False,
                )
                postprocess_outputs(df_sw, file_path, out_prefix_sw, corr_sw)
                summary_sw, _ = run_all_methods_and_compare(df_sw, corr_sw)
                if summary_sw.shape[0] == 0:
                    continue
                row = summary_sw.iloc[0].to_dict()
                row["file"] = file_path.name
                row["sweep_method"] = "rigid"
                row["keep_fraction"] = float(keep)
                row["inlier_reestimate_iters"] = int(iters)
                sweep_rows.append(row)

        for k_neighbors in local_k_neighbors_list:
            for ell_field in local_ell_um_field_list:
                opts_sw = dict(OPTIONS)
                opts_sw["k_neighbors"] = int(k_neighbors)
                opts_sw["ell_um_field"] = float(ell_field)
                tag = f"local_k{k_neighbors}_field{str(ell_field).replace('.', 'p')}"
                out_prefix_sw = RESULTS_ROOT / "rigid_local_sweep" / file_path.stem / tag / file_path.stem
                corr_sw = run_methods_for_dataframe(
                    df=df_sw,
                    out_prefix=out_prefix_sw,
                    methods=["local"],
                    opts=opts_sw,
                    use_stationary_subset=False,
                )
                postprocess_outputs(df_sw, file_path, out_prefix_sw, corr_sw)
                summary_sw, _ = run_all_methods_and_compare(df_sw, corr_sw)
                if summary_sw.shape[0] == 0:
                    continue
                row = summary_sw.iloc[0].to_dict()
                row["file"] = file_path.name
                row["sweep_method"] = "local"
                row["k_neighbors"] = int(k_neighbors)
                row["ell_um_field"] = float(ell_field)
                sweep_rows.append(row)

    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        out_root = RESULTS_ROOT / "rigid_local_sweep"
        out_root.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(out_root / "rigid_local_sweep_summary.csv", index=False)
        if "sweep_method" in sweep_df.columns:
            sweep_df[sweep_df["sweep_method"] == "rigid"].to_csv(
                out_root / "rigid_sweep_summary.csv", index=False
            )
            sweep_df[sweep_df["sweep_method"] == "local"].to_csv(
                out_root / "local_sweep_summary.csv", index=False
            )
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
# %% HOW TO RUN SYNTHETIC PAPER BASELINE
RUN_SYNTHETIC_PAPER_BASELINE_DEFAULTS = False
if RUN_SYNTHETIC_PAPER_BASELINE_DEFAULTS:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark"
    if not list_imaris_xls_files(synthetic_root):
        generate_synthetic_imaris_folder(
            out_dir=synthetic_root,
            n_files=5,
            n_tracks=80,
            n_times=20,
            stationary_fraction=0.45,
            migrating_fraction=0.35,
            seed=42,
            temporal_smoothness=0.9,
            local_mode="smooth",
        )

    methods_paper = ["rigid", "local", "rbf", "st_rbf"]
    opts_paper = dict(OPTIONS)
    opts_paper["st_robust_mode"] = "sample_trim"
    opts_paper["st_ell_space"] = 4.0
    opts_paper["st_ell_time"] = 0.4
    synthetic_results = synthetic_root / "driftcorr_results"

    for file_path in list_imaris_xls_files(synthetic_root):
        df_syn = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix = synthetic_results / "paper_baseline_defaults" / file_path.stem / file_path.stem
        corr = run_methods_for_dataframe(
            df=df_syn,
            out_prefix=out_prefix,
            methods=methods_paper,
            opts=opts_paper,
            use_stationary_subset=False,
        )
        postprocess_outputs(df_syn, file_path, out_prefix, corr)
# %% END
