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
    from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
    from driftcorr.method_local_communities import run_local_drift_correction_for_file
    from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
    from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks, build_step_transforms
    from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
    from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
    from driftcorr.stationary import find_stationary_tracks
    from driftcorr.synthetic_experiments import (
        generate_synthetic_imaris_folder,
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
        from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
        from driftcorr.method_local_communities import run_local_drift_correction_for_file
        from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference, make_rbf_diagnostic_plots
        from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks, build_step_transforms
        from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
        from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
        from driftcorr.stationary import find_stationary_tracks
        from driftcorr.synthetic_experiments import (
            generate_synthetic_imaris_folder,
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
    "lam": 1e-2,
    "keep_fraction": 0.7,
    "min_pairs": 15,
    "m_centers": 25,
    "sigma": None,
    "temporal_gamma": 0.0,
    "rigid_mode": "rigid",
    "inlier_reestimate_iters": 30,
    "k_neighbors": 12,
    "radius_um": None,
    "ell_um_graph": 80.0,
    "ell_um_field": 120.0,
    "min_edge_weight": 0.25,
    "max_pairs_per_node": 50,
    "st_m_centers": 40,
    "st_ell_space": 120.0,
    "st_ell_time": 5.0,
    "st_robust_keep_fraction": 0.7,
    "st_robust_iters": 4,
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
            lam=opts["lam"],
            robust_keep_fraction=opts["st_robust_keep_fraction"],
            robust_iters=opts["st_robust_iters"],
            seed=opts["seed"],
            fit_track_ids=stationary_ids,
        )
        df_st = apply_spacetime_correction(df, st_model)
        pd.DataFrame([st_info]).to_csv(str(out_prefix) + "_st_rbf_fit_info.csv", index=False)
        df_st.to_csv(str(out_prefix) + "_positions_corrected_st_rbf.csv", index=False)
        corr_by_method["st_rbf"] = df_st

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
    for method_name, df_corr in corr_by_method.items():
        save_visual_inspection_plots(df_orig, df_corr, out_prefix, method_name, df_groundtruth=gt)

    if gt is not None:
        save_groundtruth_evaluation_outputs(out_prefix, gt, corr_by_method)


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


# %% HOW TO BATCH RUN MULTIPLE FILES

RUN_BATCH = False
if RUN_BATCH:
    methods_batch = ["rbf", "rigid", "local", "st_rbf"]

    for file_path in list_imaris_xls_files(IMARIS_DIR):
        df_batch = ensure_numeric_tracks(read_imaris_position_xls(file_path))
        out_prefix_batch = RESULTS_ROOT / file_path.stem / file_path.stem

        corr_batch = run_methods_for_dataframe(
            df=df_batch,
            out_prefix=out_prefix_batch,
            methods=methods_batch,
            opts=OPTIONS,
            use_stationary_subset=False,
        )
        postprocess_outputs(df_batch, file_path, out_prefix_batch, corr_batch)


# %% HOW TO GENERATE AND RUN SYNTHETIC DATA
RUN_SYNTHETIC = True
if RUN_SYNTHETIC:
    synthetic_root = PROJECT_ROOT / "Synthetic IMARIS files tutorial"
    synthetic = generate_synthetic_imaris_folder(
        out_dir=synthetic_root,
        n_files=5,
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

# %%
