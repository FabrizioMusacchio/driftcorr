"""Command line interface for driftcorr pipelines.

This module orchestrates end-to-end workflows:
- loading IMARIS-like tracking tables,
- running one or more correction methods,
- writing corrected outputs and method diagnostics,
- creating standard evaluation plots and visual inspection plots,
- optionally running synthetic benchmark generation and ground-truth evaluation.

Mathematical role:
- No new estimator is defined here.
- The module composes estimators from method modules:
  `rbf`, `rigid`, `local`, `st_rbf`.

Main controllable parameters and effects:
- `--keep-fraction`: robust trimming strength; lower means stronger outlier rejection.
- `--temporal-gamma`: temporal smoothness for RBF weights; higher enforces smoother
  time evolution of drift fields.
- `--rigid-spatial-groups`: number of rigid partitions; >1 enables piecewise rigid
  modeling with spatial blending.
- `--st-ell-space`, `--st-ell-time`: space-time kernel length scales; larger values
  yield smoother drift fields in space/time.
- `--use-stationary-subset`: fit selected methods on likely stationary tracks only.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import run_all_methods_and_compare, save_evaluation_outputs
from .io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
from .metrics import msd_per_track, pivot_tracks
from .method_local_communities import run_local_drift_correction_for_file
from .method_rbf_ref import (
    correct_positions_rbf_to_reference,
    make_rbf_diagnostic_plots,
    plot_rbf_residuals_by_space_input_mode,
)
from .method_rigid_step import (
    apply_piecewise_rigid_correction_to_tracks,
    apply_rigid_correction_to_tracks,
    build_piecewise_rigid_model,
    build_step_transforms,
)
from .method_spacetime_rbf import apply_correction as apply_spacetime_correction
from .method_spacetime_rbf import fit_spacetime_rbf_model
from .method_spacetime_rbf import make_spacetime_diagnostic_plots
from .method_spacetime_rbf_moe import (
    apply_spacetime_rbf_moe_correction,
    fit_spacetime_rbf_moe_model,
    make_st_moe_diagnostic_plots,
)
from .stationary import evaluate_stationary_selection_impact, find_stationary_tracks, plot_stationary_spatial_distribution
from .synthetic_experiments import (
    generate_synthetic_imaris_folder,
    load_drift_field_for_imaris_file,
    load_groundtruth_for_imaris_file,
    save_groundtruth_evaluation_outputs,
)
from .visualization import save_visual_inspection_plots


def _base_from_path(path: Path) -> str:
    return path.stem


def _result_prefix_for_input_file(out_dir: Path, input_file: Path) -> Path:
    """Return per-file output prefix inside a dedicated file subfolder."""
    base = _base_from_path(input_file)
    file_dir = out_dir / base
    file_dir.mkdir(parents=True, exist_ok=True)
    return file_dir / base


def _resolve_output_dir_for_single(args: argparse.Namespace, in_path: Path) -> Path:
    if args.out is not None:
        return Path(args.out)
    return in_path.parent / "driftcorr_results"


def _resolve_output_dir_for_batch(args: argparse.Namespace, imaris_dir: Path) -> Path:
    if args.out is not None:
        return Path(args.out)
    return imaris_dir / "driftcorr_results"


def _prepare_stationary_subset(
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame | None, set[float] | None]:
    use_stationary = bool(args.use_stationary_subset) or bool(getattr(args, "use_stationary_only", 0))
    if not use_stationary:
        return None, None

    sel = find_stationary_tracks(
        df,
        reference_time=args.reference_time,
        rms_threshold_um=args.stationary_rms_threshold_um,
        neighbor_relstd_threshold=args.stationary_neighbor_threshold,
        msd_tau1_threshold_um2=args.stationary_msd_tau1_threshold_um2,
        msd_tau2_threshold_um2=args.stationary_msd_tau2_threshold_um2,
        corr_outlier_threshold=args.stationary_corr_outlier_threshold,
        rms_quantile=args.stationary_rms_quantile,
        neighbor_quantile=args.stationary_neighbor_quantile,
        msd_tau1_quantile=args.stationary_msd_tau1_quantile,
        msd_tau2_quantile=args.stationary_msd_tau2_quantile,
        corr_outlier_quantile=args.stationary_corr_outlier_quantile,
        k_neighbors=args.stationary_k_neighbors,
        use_corr_outlier=bool(args.stationary_use_corr_outlier),
    )
    diagnostics = dict(sel.diagnostics)
    use_subset = len(sel.track_ids) >= int(args.min_pairs)
    diagnostics["used_for_fit"] = bool(use_subset)
    diagnostics["min_pairs_required"] = int(args.min_pairs)
    diagnostics["selected_track_count"] = int(len(sel.track_ids))

    sel.table.to_csv(str(out_prefix) + "_stationary_subset.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(str(out_prefix) + "_stationary_diagnostics.csv", index=False)
    plot_stationary_spatial_distribution(sel.table, str(out_prefix) + "_stationary_spatial_distribution.png")

    if use_subset:
        return sel.table, sel.track_ids
    return sel.table, None


def _run_rbf(
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
    fit_track_ids: set[float] | None = None,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    fit_ids = None if fit_track_ids is None else list(sorted(fit_track_ids))
    df_corr, tf_table, _, _, _ = correct_positions_rbf_to_reference(
        df,
        reference_time=args.reference_time,
        m=args.m_centers,
        keep_fraction=args.keep_fraction,
        lam=args.lam,
        sigma=args.sigma,
        min_pairs=args.min_pairs,
        seed=args.seed,
        fit_track_ids=fit_ids,
        temporal_gamma=args.temporal_gamma,
        center_mode=args.center_mode,
        pooled_time_stride=args.center_pooled_time_stride,
        robust_mode=args.robust_mode,
        track_trim_iters=args.track_trim_iters,
        two_stage=bool(args.two_stage),
        space_input_mode=args.rbf_space_input_mode,
        rbf_linear_terms=bool(args.rbf_linear_terms),
        row_position_iters=args.row_position_iters,
        row_position_tol=args.row_position_tol,
        weight_abort_threshold=args.weight_abort_threshold,
    )
    if save_outputs:
        df_corr.to_csv(str(out_prefix) + "_positions_corrected_rbf.csv", index=False)
        tf_table.to_csv(str(out_prefix) + "_rbf_fit_table.csv", index=False)
        make_rbf_diagnostic_plots(tf_table, out_prefix)
        if bool(getattr(args, "rbf_compare_space_modes", 0)):
            tf_by_mode: dict[str, pd.DataFrame] = {str(args.rbf_space_input_mode): tf_table}
            for mode in ["row_position", "ref_position", "track_anchor"]:
                if mode == str(args.rbf_space_input_mode):
                    continue
                _, tf_mode, _, _, _ = correct_positions_rbf_to_reference(
                    df,
                    reference_time=args.reference_time,
                    m=args.m_centers,
                    keep_fraction=args.keep_fraction,
                    lam=args.lam,
                    sigma=args.sigma,
                    min_pairs=args.min_pairs,
                    seed=args.seed,
                    fit_track_ids=fit_ids,
                    temporal_gamma=args.temporal_gamma,
                    center_mode=args.center_mode,
                    pooled_time_stride=args.center_pooled_time_stride,
                    robust_mode=args.robust_mode,
                    track_trim_iters=args.track_trim_iters,
                    two_stage=bool(args.two_stage),
                    space_input_mode=mode,
                    rbf_linear_terms=bool(args.rbf_linear_terms),
                    row_position_iters=args.row_position_iters,
                    row_position_tol=args.row_position_tol,
                    weight_abort_threshold=args.weight_abort_threshold,
                )
                tf_by_mode[mode] = tf_mode
            plot_rbf_residuals_by_space_input_mode(tf_by_mode, str(out_prefix) + "_rbf_residuals_by_space_input_mode.png")
    return {"rbf": df_corr}


def _run_rigid(
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    d = ensure_numeric_tracks(df)
    out: dict[str, pd.DataFrame] = {}

    step, tf_rigid, times = build_step_transforms(
        d,
        keep_fraction=args.keep_fraction,
        min_pairs=args.min_pairs,
        mode=args.rigid_mode,
        inlier_reestimate_iters=args.inlier_reestimate_iters,
    )
    ref_time = float(times.min()) if args.reference_time is None else float(args.reference_time)
    df_global = apply_rigid_correction_to_tracks(d, step, times, ref_time=ref_time)

    if args.rigid_spatial_groups > 1:
        model, piece_table = build_piecewise_rigid_model(
            d,
            n_groups=args.rigid_spatial_groups,
            keep_fraction=args.keep_fraction,
            min_pairs=args.min_pairs,
            mode=args.rigid_mode,
            inlier_reestimate_iters=args.inlier_reestimate_iters,
            blend_sigma_um=args.rigid_blend_sigma_um,
            seed=args.rigid_cluster_seed,
        )
        df_piece = apply_piecewise_rigid_correction_to_tracks(d, model)

        if save_outputs:
            piece_table.to_csv(str(out_prefix) + "_rigid_piecewise_transforms.csv", index=False)
            df_piece.to_csv(str(out_prefix) + "_positions_corrected_rigid_piecewise.csv", index=False)
            tf_rigid.to_csv(str(out_prefix) + "_rigid_global_transforms.csv", index=False)
            if args.rigid_compare_piecewise:
                df_global.to_csv(str(out_prefix) + "_positions_corrected_rigid_global.csv", index=False)

        if args.rigid_compare_piecewise:
            out["rigid_global"] = df_global
            out["rigid_piecewise"] = df_piece
        else:
            out["rigid"] = df_piece
    else:
        if save_outputs:
            tf_rigid.to_csv(str(out_prefix) + "_rigid_transforms.csv", index=False)
            df_global.to_csv(str(out_prefix) + "_positions_corrected_rigid.csv", index=False)
        out["rigid"] = df_global

    return out


def _run_local(
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    df_corr, _, _, _, _ = run_local_drift_correction_for_file(
        df,
        out_prefix=out_prefix,
        reference_time=args.reference_time,
        k_neighbors=args.k_neighbors,
        radius_um=args.radius_um,
        ell_um_graph=args.ell_um_graph,
        ell_um_field=args.ell_um_field,
        min_edge_weight=args.min_edge_weight,
        max_pairs_per_node=args.max_pairs_per_node,
        use_track_mean_position=not args.use_row_position,
    )
    return {"local": df_corr}


def _run_st_rbf(
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
    fit_track_ids: set[float] | None = None,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    fit_ids = None if fit_track_ids is None else list(sorted(fit_track_ids))
    model, info = fit_spacetime_rbf_model(
        df,
        reference_time=args.reference_time,
        m_centers=args.st_m_centers,
        ell_space=args.st_ell_space,
        ell_time=args.st_ell_time,
        lam=args.st_lam if args.st_lam is not None else args.lam,
        robust_keep_fraction=args.st_robust_keep_fraction,
        robust_iters=args.st_robust_iters,
        seed=args.seed,
        fit_track_ids=fit_ids,
        space_input_mode=args.space_input_mode,
        n_refine=args.st_refine_iters,
        two_stage=bool(args.two_stage),
        robust_mode=args.st_robust_mode,
        track_trim_iters=args.st_track_trim_iters,
        rbf_linear_terms=bool(args.rbf_linear_terms),
        weight_abort_threshold=args.weight_abort_threshold,
        per_axis_scaling=bool(args.st_per_axis_scaling),
    )
    df_corr = apply_spacetime_correction(df, model)
    if save_outputs:
        pd.DataFrame([info]).to_csv(str(out_prefix) + "_st_rbf_fit_info.csv", index=False)
        df_corr.to_csv(str(out_prefix) + "_positions_corrected_st_rbf.csv", index=False)
        make_spacetime_diagnostic_plots(model, out_prefix)
    return {"st_rbf": df_corr}


def _run_st_rbf_moe(
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
    fit_track_ids: set[float] | None = None,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    fit_ids = None if fit_track_ids is None else list(sorted(fit_track_ids))
    k_values: list[int] = []
    raw = str(getattr(args, "st_moe_k_list", "") or "").strip()
    if raw:
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                kv = int(tok)
            except ValueError:
                continue
            if kv > 0 and kv not in k_values:
                k_values.append(kv)
    if not k_values:
        k_values = [int(args.st_moe_k)]

    out: dict[str, pd.DataFrame] = {}
    for i, k in enumerate(k_values):
        model, expert_table = fit_spacetime_rbf_moe_model(
            df,
            n_experts=int(k),
            blend_sigma_um=args.st_moe_blend_sigma_um,
            cluster_mode=args.st_moe_cluster_mode,
            seed=int(args.seed + i * 13),
            reference_time=args.reference_time,
            fit_track_ids=fit_ids,
            m_centers=args.st_m_centers,
            ell_space=args.st_ell_space,
            ell_time=args.st_ell_time,
            lam=args.st_lam if args.st_lam is not None else args.lam,
            robust_keep_fraction=args.st_robust_keep_fraction,
            robust_iters=args.st_robust_iters,
            space_input_mode=args.space_input_mode,
            n_refine=args.st_refine_iters,
            two_stage=bool(args.two_stage),
            robust_mode=args.st_robust_mode,
            track_trim_iters=args.st_track_trim_iters,
            rbf_linear_terms=bool(args.rbf_linear_terms),
            weight_abort_threshold=args.weight_abort_threshold,
            per_axis_scaling=bool(args.st_per_axis_scaling),
        )
        df_corr = apply_spacetime_rbf_moe_correction(df, model)
        name = f"st_rbf_moe_k{int(k)}"
        out[name] = df_corr
        if save_outputs:
            expert_table.to_csv(str(out_prefix) + f"_{name}_experts.csv", index=False)
            df_corr.to_csv(str(out_prefix) + f"_positions_corrected_{name}.csv", index=False)
            make_st_moe_diagnostic_plots(expert_table, str(out_prefix) + f"_{name}")
    return out


def _run_method(
    method: str,
    df: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
    fit_track_ids: set[float] | None = None,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    if method == "rbf":
        return _run_rbf(df, out_prefix, args, fit_track_ids=fit_track_ids, save_outputs=save_outputs)
    if method == "rigid":
        return _run_rigid(df, out_prefix, args, save_outputs=save_outputs)
    if method == "local":
        return _run_local(df, out_prefix, args)
    if method == "st_rbf":
        return _run_st_rbf(df, out_prefix, args, fit_track_ids=fit_track_ids, save_outputs=save_outputs)
    if method == "st_rbf_moe":
        return _run_st_rbf_moe(df, out_prefix, args, fit_track_ids=fit_track_ids, save_outputs=save_outputs)
    raise ValueError(f"Unknown method: {method}")


def _maybe_save_stationary_impact(
    df_orig: pd.DataFrame,
    out_prefix: Path,
    args: argparse.Namespace,
    method: str,
    with_subset: dict[str, pd.DataFrame],
    groundtruth_df: pd.DataFrame | None = None,
) -> None:
    use_stationary = bool(args.use_stationary_subset) or bool(getattr(args, "use_stationary_only", 0))
    if not use_stationary:
        return
    if method not in {"rbf", "st_rbf"}:
        return

    baseline = _run_method(method, df_orig, out_prefix, args, fit_track_ids=None, save_outputs=False)
    name = next(iter(with_subset.keys()))
    base_name = next(iter(baseline.keys()))
    summary_no, _ = run_all_methods_and_compare(df_orig, [(base_name, baseline[base_name])])
    summary_yes, _ = run_all_methods_and_compare(df_orig, [(name, with_subset[name])])

    if summary_no.empty or summary_yes.empty:
        return

    m_no = summary_no.iloc[0].to_dict()
    m_yes = summary_yes.iloc[0].to_dict()
    deltas = evaluate_stationary_selection_impact(m_no, m_yes)
    rows = [
        {
            **{f"without_subset_{k}": v for k, v in m_no.items()},
            **{f"with_subset_{k}": v for k, v in m_yes.items()},
            **deltas,
        }
    ]

    if groundtruth_df is not None and "CellType" in groundtruth_df.columns:
        gt_migr = set(groundtruth_df.loc[groundtruth_df["CellType"] == "migrating", "TrackID"].astype(float).unique().tolist())
        if gt_migr:
            def _migrating_msd_tau1(df_corr: pd.DataFrame) -> float:
                dcm = df_corr[df_corr["TrackID"].astype(float).isin(gt_migr)].copy()
                if dcm.shape[0] == 0:
                    return float("nan")
                _, _, x = pivot_tracks(dcm, "Position X_corr", "Position Y_corr", "Position Z_corr")
                _, msd = msd_per_track(x)
                if msd.size == 0:
                    return float("nan")
                with np.errstate(invalid="ignore"):
                    return float(np.nanmedian(msd[:, 0])) if msd.shape[1] >= 1 else float("nan")

            def _migrating_rms_error(df_corr: pd.DataFrame) -> float:
                mc = df_corr[["TrackID", "Time", "Position X_corr", "Position Y_corr", "Position Z_corr"]].copy()
                mg = groundtruth_df[["TrackID", "Time", "Position X", "Position Y", "Position Z", "CellType"]].copy()
                mm = mc.merge(mg, on=["TrackID", "Time"], how="inner")
                mm = mm[mm["CellType"] == "migrating"].copy()
                if mm.shape[0] == 0:
                    return float("nan")
                err = np.sqrt(
                    (mm["Position X_corr"].to_numpy(float) - mm["Position X"].to_numpy(float)) ** 2
                    + (mm["Position Y_corr"].to_numpy(float) - mm["Position Y"].to_numpy(float)) ** 2
                    + (mm["Position Z_corr"].to_numpy(float) - mm["Position Z"].to_numpy(float)) ** 2
                )
                return float(np.sqrt(np.mean(err ** 2)))

            without_df = baseline[base_name]
            with_df = with_subset[name]
            rows[0]["without_subset_migrating_msd_tau1"] = _migrating_msd_tau1(without_df)
            rows[0]["with_subset_migrating_msd_tau1"] = _migrating_msd_tau1(with_df)
            rows[0]["delta_migrating_msd_tau1"] = float(rows[0]["with_subset_migrating_msd_tau1"]) - float(rows[0]["without_subset_migrating_msd_tau1"])
            rows[0]["without_subset_migrating_rms_error"] = _migrating_rms_error(without_df)
            rows[0]["with_subset_migrating_rms_error"] = _migrating_rms_error(with_df)
            rows[0]["delta_migrating_rms_error"] = float(rows[0]["with_subset_migrating_rms_error"]) - float(rows[0]["without_subset_migrating_rms_error"])

    pd.DataFrame(rows).to_csv(str(out_prefix) + f"_stationary_impact_{method}.csv", index=False)


def _maybe_groundtruth_eval(
    imaris_file: Path,
    out_prefix: Path,
    corrected_by_method: dict[str, pd.DataFrame],
    groundtruth_df: pd.DataFrame | None = None,
) -> None:
    gt = groundtruth_df if groundtruth_df is not None else load_groundtruth_for_imaris_file(imaris_file)
    if gt is None:
        return
    drift_gt = load_drift_field_for_imaris_file(imaris_file)
    save_groundtruth_evaluation_outputs(out_prefix, gt, corrected_by_method, drift_field_groundtruth=drift_gt)


def _run_batch_for_folder(
    imaris_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    methods: list[str],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_imaris_xls_files(imaris_dir)
    if not files:
        raise FileNotFoundError(f"No IMARIS .xls/.xlsx/.csv files found in {imaris_dir}")

    for f in files:
        df = ensure_numeric_tracks(read_imaris_position_xls(f))
        out_prefix = _result_prefix_for_input_file(out_dir, f)
        groundtruth_df = load_groundtruth_for_imaris_file(f)

        _, stationary_ids = _prepare_stationary_subset(df, out_prefix, args)

        corr_by_method: dict[str, pd.DataFrame] = {}
        for method in methods:
            use_stationary = (
                stationary_ids
                if (bool(args.use_stationary_subset) or bool(getattr(args, "use_stationary_only", 0)))
                and method in {"rbf", "st_rbf", "st_rbf_moe"}
                else None
            )
            corr = _run_method(method, df, out_prefix, args, fit_track_ids=use_stationary)
            corr_by_method.update(corr)
            if use_stationary is not None:
                _maybe_save_stationary_impact(df, out_prefix, args, method, corr, groundtruth_df=groundtruth_df)

        if not args.skip_visual_plots:
            for method_name, df_corr in corr_by_method.items():
                save_visual_inspection_plots(df, df_corr, out_prefix, method_name, df_groundtruth=groundtruth_df)

        summary_df, details = run_all_methods_and_compare(df, corr_by_method)
        save_evaluation_outputs(out_prefix, summary_df, details)
        _maybe_groundtruth_eval(f, out_prefix, corr_by_method, groundtruth_df=groundtruth_df)

        print(f"Done: {f.name} methods={methods} -> {out_prefix.parent}")

    return 0


def run_single(args: argparse.Namespace) -> int:
    in_path = Path(args.imaris_xls)
    out_dir = _resolve_output_dir_for_single(args, in_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_numeric_tracks(read_imaris_position_xls(in_path))
    groundtruth_df = load_groundtruth_for_imaris_file(in_path)

    out_prefix = _result_prefix_for_input_file(out_dir, in_path)

    _, stationary_ids = _prepare_stationary_subset(df, out_prefix, args)
    use_stationary = (
        stationary_ids
        if (bool(args.use_stationary_subset) or bool(getattr(args, "use_stationary_only", 0)))
        and args.method in {"rbf", "st_rbf", "st_rbf_moe"}
        else None
    )

    corr_dict = _run_method(args.method, df, out_prefix, args, fit_track_ids=use_stationary)
    if use_stationary is not None:
        _maybe_save_stationary_impact(df, out_prefix, args, args.method, corr_dict, groundtruth_df=groundtruth_df)

    if not args.skip_visual_plots:
        for method_name, df_corr in corr_dict.items():
            save_visual_inspection_plots(df, df_corr, out_prefix, method_name, df_groundtruth=groundtruth_df)

    summary_df, details = run_all_methods_and_compare(df, corr_dict)
    save_evaluation_outputs(out_prefix, summary_df, details)
    _maybe_groundtruth_eval(in_path, out_prefix, corr_dict, groundtruth_df=groundtruth_df)

    print(f"Done: {in_path.name} with method={args.method} -> {out_prefix.parent}")
    return 0


def run_batch(args: argparse.Namespace) -> int:
    imaris_dir = Path(args.imaris_dir)
    out_dir = _resolve_output_dir_for_batch(args, imaris_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    return _run_batch_for_folder(imaris_dir, out_dir, args, methods)


def run_synthetic_benchmark(args: argparse.Namespace) -> int:
    synthetic_dir = Path(args.synthetic_dir)
    generate_synthetic_imaris_folder(
        out_dir=synthetic_dir,
        n_files=args.synthetic_n_files,
        n_tracks=args.synthetic_n_tracks,
        n_times=args.synthetic_n_times,
        stationary_fraction=args.synthetic_stationary_fraction,
        migrating_fraction=args.synthetic_migrating_fraction,
        seed=args.synthetic_seed,
        temporal_smoothness=args.synthetic_temporal_smoothness,
        local_mode=args.synthetic_local_mode,
        piecewise_regions=args.synthetic_piecewise_regions,
        piecewise_transition_um=args.synthetic_piecewise_transition_um,
        local_lengthscale_um=args.synthetic_local_lengthscale_um,
        local_drift_scale_um=args.synthetic_local_drift_scale_um,
        n_rbf_centers=args.synthetic_n_rbf_centers,
        measurement_noise_um=args.synthetic_measurement_noise_um,
    )

    out_dir = Path(args.out) if args.out is not None else synthetic_dir / "driftcorr_results"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"Synthetic IMARIS folder generated: {synthetic_dir}")
    return _run_batch_for_folder(synthetic_dir, out_dir, args, methods)


def _regime_presets() -> dict[str, dict[str, object]]:
    """Fixed synthetic regimes to compare where each correction family should excel."""
    return {
        "smooth_spacetime": {
            "local_mode": "smooth",
            "piecewise_regions": 1,
            "piecewise_transition_um": 120.0,
            "local_lengthscale_um": 280.0,
            "local_drift_scale_um": 2.1,
            "measurement_noise_um": 0.08,
        },
        "piecewise_smooth": {
            "local_mode": "piecewise_smooth",
            "piecewise_regions": 3,
            "piecewise_transition_um": 130.0,
            "local_lengthscale_um": 220.0,
            "local_drift_scale_um": 2.4,
            "measurement_noise_um": 0.12,
        },
        "piecewise_abrupt": {
            "local_mode": "piecewise_abrupt",
            "piecewise_regions": 4,
            "piecewise_transition_um": 6.0,
            "local_lengthscale_um": 180.0,
            "local_drift_scale_um": 2.6,
            "measurement_noise_um": 0.15,
        },
    }


def run_synthetic_regime_benchmark(args: argparse.Namespace) -> int:
    """Run fixed 3-regime synthetic suite over multiple seeds and aggregate RMS summaries."""
    synthetic_root = Path(args.synthetic_dir)
    out_root = Path(args.out) if args.out is not None else synthetic_root / "driftcorr_results_regimes"
    out_root.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seed_values = [int(args.synthetic_seed + i) for i in range(max(1, int(args.synthetic_n_seeds)))]
    rows_runs: list[dict[str, object]] = []

    for regime_name, regime_cfg in _regime_presets().items():
        for seed in seed_values:
            reg_dir = synthetic_root / f"{regime_name}_seed_{seed:03d}"
            generate_synthetic_imaris_folder(
                out_dir=reg_dir,
                n_files=1,
                n_tracks=args.synthetic_n_tracks,
                n_times=args.synthetic_n_times,
                stationary_fraction=args.synthetic_stationary_fraction,
                migrating_fraction=args.synthetic_migrating_fraction,
                seed=seed,
                temporal_smoothness=args.synthetic_temporal_smoothness,
                local_mode=str(regime_cfg["local_mode"]),
                piecewise_regions=int(regime_cfg["piecewise_regions"]),
                piecewise_transition_um=float(regime_cfg["piecewise_transition_um"]),
                local_lengthscale_um=float(regime_cfg["local_lengthscale_um"]),
                local_drift_scale_um=float(regime_cfg["local_drift_scale_um"]),
                n_rbf_centers=args.synthetic_n_rbf_centers,
                measurement_noise_um=float(regime_cfg["measurement_noise_um"]),
            )
            out_dir_run = out_root / f"{regime_name}_seed_{seed:03d}"
            _run_batch_for_folder(reg_dir, out_dir_run, args, methods)

            files = list_imaris_xls_files(reg_dir)
            if not files:
                continue
            f = files[0]
            out_prefix = _result_prefix_for_input_file(out_dir_run, f)
            gt_summary_path = out_prefix.with_name(out_prefix.name + "_gt_metrics_summary.csv")
            if not gt_summary_path.exists():
                continue
            gts = pd.read_csv(gt_summary_path)
            gts["regime"] = regime_name
            gts["seed"] = int(seed)
            rows_runs.extend(gts.to_dict(orient="records"))
            print(f"Regime run done: regime={regime_name} seed={seed} -> {out_prefix.parent}")

    if not rows_runs:
        raise RuntimeError("No regime benchmark rows were produced.")

    runs_df = pd.DataFrame(rows_runs)
    out_runs = out_root / "synthetic_regime_runs.csv"
    runs_df.to_csv(out_runs, index=False)

    metrics = [c for c in ["overall_rms_error", "overall_rms_error_stationary", "overall_rms_error_migrating"] if c in runs_df.columns]
    agg_rows: list[dict[str, object]] = []
    for (regime, method), grp in runs_df.groupby(["regime", "method"], dropna=False):
        row: dict[str, object] = {
            "regime": regime,
            "method": method,
            "n_runs": int(grp.shape[0]),
        }
        for m in metrics:
            vv = pd.to_numeric(grp[m], errors="coerce").to_numpy(float)
            row[f"{m}_mean"] = float(np.nanmean(vv)) if vv.size else float("nan")
            row[f"{m}_std"] = float(np.nanstd(vv, ddof=0)) if vv.size else float("nan")
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).sort_values(["regime", "method"]).reset_index(drop=True)
    out_agg = out_root / "synthetic_regime_summary.csv"
    agg_df.to_csv(out_agg, index=False)
    print(f"Regime benchmark summary -> {out_agg}")
    return 0


def _add_common_method_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--reference-time", type=float, default=None)

    p.add_argument("--keep-fraction", type=float, default=0.7)
    p.add_argument("--min-pairs", type=int, default=15)

    p.add_argument("--m-centers", type=int, default=45)
    p.add_argument("--lam", type=float, default=3e-4)
    p.add_argument("--sigma", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--temporal-gamma", type=float, default=1.0)
    p.add_argument("--center-mode", choices=["ref", "pooled", "kmeans_pooled"], default="ref")
    p.add_argument("--center-pooled-time-stride", type=int, default=1)
    p.add_argument("--robust-mode", choices=["none", "sample_trim", "track_trim"], default="sample_trim")
    p.add_argument("--track-trim-iters", type=int, default=4)
    p.add_argument("--two-stage", type=int, choices=[0, 1], default=0)
    p.add_argument("--rbf-linear-terms", type=int, choices=[0, 1], default=0)
    p.add_argument("--rbf-space-input-mode", choices=["row_position", "track_anchor", "ref_position"], default="track_anchor")
    p.add_argument("--rbf-compare-space-modes", type=int, choices=[0, 1], default=0)
    p.add_argument("--row-position-iters", type=int, default=3)
    p.add_argument("--row-position-tol", type=float, default=1e-3)
    p.add_argument("--weight-abort-threshold", type=float, default=1e4)

    p.add_argument("--rigid-mode", choices=["rigid", "rotz"], default="rigid")
    p.add_argument("--inlier-reestimate-iters", type=int, default=30)
    p.add_argument("--rigid-spatial-groups", type=int, default=1)
    p.add_argument("--rigid-blend-sigma-um", type=float, default=120.0)
    p.add_argument("--rigid-cluster-seed", type=int, default=0)
    p.add_argument("--rigid-compare-piecewise", action="store_true")

    p.add_argument("--k-neighbors", type=int, default=12)
    p.add_argument("--radius-um", type=float, default=None)
    p.add_argument("--ell-um-graph", type=float, default=80.0)
    p.add_argument("--ell-um-field", type=float, default=120.0)
    p.add_argument("--min-edge-weight", type=float, default=0.25)
    p.add_argument("--max-pairs-per-node", type=int, default=50)
    p.add_argument(
        "--use-row-position",
        action="store_true",
        help="Use per-row position instead of per-track mean for local method assignment",
    )

    p.add_argument("--st-m-centers", type=int, default=70)
    p.add_argument("--st-lam", type=float, default=8.5e-4)
    p.add_argument("--st-ell-space", type=float, default=2.0)
    p.add_argument("--st-ell-time", type=float, default=0.8)
    p.add_argument("--st-robust-keep-fraction", type=float, default=0.83)
    p.add_argument("--st-robust-iters", type=int, default=2)
    p.add_argument("--space-input-mode", choices=["row_position", "track_anchor", "ref_position"], default="ref_position")
    p.add_argument("--st-refine-iters", type=int, default=3)
    p.add_argument("--st-robust-mode", choices=["none", "sample_trim", "track_trim"], default="sample_trim")
    p.add_argument("--st-track-trim-iters", type=int, default=4)
    p.add_argument("--st-moe-k", type=int, default=3)
    p.add_argument("--st-moe-k-list", default="3")
    p.add_argument("--st-moe-blend-sigma-um", type=float, default=140.0)
    p.add_argument("--st-moe-cluster-mode", choices=["track_anchor", "ref_position"], default="track_anchor")
    p.add_argument("--st-per-axis-scaling", type=int, choices=[0, 1], default=1)

    p.add_argument("--use-stationary-subset", action="store_true")
    p.add_argument("--use-stationary-only", type=int, choices=[0, 1], default=0)
    p.add_argument("--stationary-rms-threshold-um", type=float, default=None)
    p.add_argument("--stationary-neighbor-threshold", type=float, default=None)
    p.add_argument("--stationary-msd-tau1-threshold-um2", type=float, default=None)
    p.add_argument("--stationary-msd-tau2-threshold-um2", type=float, default=None)
    p.add_argument("--stationary-corr-outlier-threshold", type=float, default=None)
    p.add_argument("--stationary-rms-quantile", type=float, default=0.4)
    p.add_argument("--stationary-neighbor-quantile", type=float, default=0.5)
    p.add_argument("--stationary-msd-tau1-quantile", type=float, default=0.45)
    p.add_argument("--stationary-msd-tau2-quantile", type=float, default=0.45)
    p.add_argument("--stationary-corr-outlier-quantile", type=float, default=0.6)
    p.add_argument("--stationary-k-neighbors", type=int, default=6)
    p.add_argument("--stationary-use-corr-outlier", type=int, choices=[0, 1], default=1)

    p.add_argument("--skip-visual-plots", action="store_true")
    p.add_argument("--synthetic-n-seeds", type=int, default=5)
    p.add_argument("--synthetic-local-mode", choices=["smooth", "piecewise_smooth", "piecewise_abrupt"], default="smooth")
    p.add_argument("--synthetic-piecewise-regions", type=int, default=3)
    p.add_argument("--synthetic-piecewise-transition-um", type=float, default=90.0)
    p.add_argument("--synthetic-local-lengthscale-um", type=float, default=220.0)
    p.add_argument("--synthetic-local-drift-scale-um", type=float, default=2.4)
    p.add_argument("--synthetic-n-rbf-centers", type=int, default=10)
    p.add_argument("--synthetic-measurement-noise-um", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftcorr", description="Drift correction for IMARIS tracking exports")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one correction method on one IMARIS XLS file")
    run.add_argument("--method", choices=["rbf", "rigid", "local", "st_rbf", "st_rbf_moe"], required=True)
    run.add_argument("--imaris-xls", required=True)
    run.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <directory of --imaris-xls>/driftcorr_results",
    )

    run_all = sub.add_parser("run-all", help="Batch process IMARIS files")
    run_all.add_argument("--imaris-dir", required=True)
    run_all.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <imaris-dir>/driftcorr_results",
    )
    run_all.add_argument("--methods", default="rbf,rigid,local,st_rbf", help="Comma-separated method names")

    synth = sub.add_parser("synthetic-benchmark", help="Generate synthetic IMARIS folder and run full pipeline")
    synth.add_argument("--synthetic-dir", default="synthetic_imaris_files")
    synth.add_argument("--out", default=None, help="Default: <synthetic-dir>/driftcorr_results")
    synth.add_argument("--methods", default="rbf,rigid,local,st_rbf,st_rbf_moe", help="Comma-separated method names")
    synth.add_argument("--synthetic-n-files", type=int, default=5)
    synth.add_argument("--synthetic-n-tracks", type=int, default=80)
    synth.add_argument("--synthetic-n-times", type=int, default=20)
    synth.add_argument("--synthetic-stationary-fraction", type=float, default=0.45)
    synth.add_argument("--synthetic-migrating-fraction", type=float, default=0.35)
    synth.add_argument("--synthetic-seed", type=int, default=42)
    synth.add_argument("--synthetic-temporal-smoothness", type=float, default=0.9)

    synth_reg = sub.add_parser(
        "synthetic-regime-benchmark",
        help="Run synthetic regimes (smooth/piecewise_smooth/piecewise_abrupt) across seeds",
    )
    synth_reg.add_argument("--synthetic-dir", default="synthetic_regime_suite")
    synth_reg.add_argument("--out", default=None, help="Default: <synthetic-dir>/driftcorr_results_regimes")
    synth_reg.add_argument("--methods", default="rbf,rigid,local,st_rbf,st_rbf_moe", help="Comma-separated method names")
    synth_reg.add_argument("--synthetic-n-tracks", type=int, default=80)
    synth_reg.add_argument("--synthetic-n-times", type=int, default=20)
    synth_reg.add_argument("--synthetic-stationary-fraction", type=float, default=0.45)
    synth_reg.add_argument("--synthetic-migrating-fraction", type=float, default=0.35)
    synth_reg.add_argument("--synthetic-seed", type=int, default=42)
    synth_reg.add_argument("--synthetic-temporal-smoothness", type=float, default=0.9)
    synth_reg.set_defaults(st_moe_k_list="2,3,5")

    for p in (run, run_all, synth, synth_reg):
        _add_common_method_options(p)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_single(args)
    if args.command == "run-all":
        return run_batch(args)
    if args.command == "synthetic-benchmark":
        return run_synthetic_benchmark(args)
    if args.command == "synthetic-regime-benchmark":
        return run_synthetic_regime_benchmark(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
