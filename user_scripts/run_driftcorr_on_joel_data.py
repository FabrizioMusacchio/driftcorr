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

import pandas as pd

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
        tf_table.to_csv(str(out_prefix) + "_rbf_fit_table.csv", index=False)
        make_rbf_diagnostic_plots(tf_table, out_prefix)
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
        pd.DataFrame([st_info]).to_csv(str(out_prefix) + "_st_rbf_fit_info.csv", index=False)
        df_st.to_csv(str(out_prefix) + "_positions_corrected_st_rbf.csv", index=False)
        corr_by_method["st_rbf"] = df_st

    return corr_by_method


def postprocess_outputs(
    df_orig: pd.DataFrame,
    out_prefix: Path,
    corr_by_method: dict[str, pd.DataFrame],
) -> None:
    """Create standard evaluation summaries and qualitative inspection plots."""
    summary_df, details = run_all_methods_and_compare(df_orig, corr_by_method)
    save_evaluation_outputs(out_prefix, summary_df, details)

    for method_name, df_corr in corr_by_method.items():
        save_visual_inspection_plots(df_orig, df_corr, out_prefix, method_name)
# %% HOW TO RUN A SINGLE FILE
RUN_SINGLE_FILE = False
if RUN_SINGLE_FILE:
    SINGLE_FILE_NAME = "registered_stack_5D_251008_P2_565um_v1.xls"
    METHODS_SINGLE = list(DEFAULT_METHODS)
    USE_STATIONARY_SUBSET_SINGLE = False

    file_path = JOEL_DIR / SINGLE_FILE_NAME
    df_single = _read_joel_imaris_file(file_path)

    out_prefix_single = RESULTS_ROOT / "single_file" / file_path.stem / file_path.stem
    _write_input_summary(df_single, out_prefix_single)

    corr_single = run_methods_for_dataframe(
        df=df_single,
        out_prefix=out_prefix_single,
        methods=METHODS_SINGLE,
        opts=OPTIONS,
        use_stationary_subset=USE_STATIONARY_SUBSET_SINGLE,
    )
    postprocess_outputs(df_single, out_prefix_single, corr_single)
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
