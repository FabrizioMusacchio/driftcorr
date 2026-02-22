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
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    load_meta_for_imaris_file,
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


def _count_synthetic_celltypes(groundtruth_df: pd.DataFrame | None) -> tuple[int | None, int | None]:
    if groundtruth_df is None or "CellType" not in groundtruth_df.columns:
        return None, None
    tab = groundtruth_df[["TrackID", "CellType"]].drop_duplicates("TrackID")
    n_stat = int(np.sum(tab["CellType"] == "stationary"))
    n_migr = int(np.sum(tab["CellType"] == "migrating"))
    return n_stat, n_migr


def _snapshot_method_config(args: argparse.Namespace, method: str) -> dict[str, object]:
    cfg = {
        "method": str(method),
        "reference_time": args.reference_time,
        "seed": args.seed,
        "keep_fraction": args.keep_fraction,
        "min_pairs": args.min_pairs,
    }
    if method == "rbf":
        cfg.update(
            {
                "m_centers": args.m_centers,
                "lam": args.lam,
                "sigma": args.sigma,
                "temporal_gamma": args.temporal_gamma,
                "center_mode": args.center_mode,
                "robust_mode": args.robust_mode,
                "track_trim_iters": args.track_trim_iters,
                "two_stage": int(args.two_stage),
                "space_input_mode": args.rbf_space_input_mode,
                "rbf_linear_terms": int(args.rbf_linear_terms),
            }
        )
    if method == "st_rbf":
        cfg.update(
            {
                "st_m_centers": args.st_m_centers,
                "st_lam": args.st_lam if args.st_lam is not None else args.lam,
                "st_ell_space": args.st_ell_space,
                "st_ell_time": args.st_ell_time,
                "st_ell_x": args.st_ell_x,
                "st_ell_y": args.st_ell_y,
                "st_ell_z": args.st_ell_z,
                "st_robust_mode": args.st_robust_mode,
                "st_track_trim_iters": args.st_track_trim_iters,
                "st_refine_iters": args.st_refine_iters,
                "st_refine_tol": args.st_refine_tol,
                "space_input_mode": args.space_input_mode,
                "st_per_axis_scaling": args.st_per_axis_scaling,
                "rbf_linear_terms": int(args.rbf_linear_terms),
                "st_linear_interactions": int(args.st_linear_interactions),
                "two_stage": int(args.two_stage),
            }
        )
    if method == "st_rbf_moe":
        cfg.update(
            {
                "st_moe_k": args.st_moe_k,
                "st_moe_k_list": args.st_moe_k_list,
                "st_moe_blend_sigma_um": args.st_moe_blend_sigma_um,
                "st_moe_cluster_mode": args.st_moe_cluster_mode,
                "st_moe_drift_cap_factor": args.st_moe_drift_cap_factor,
                "st_moe_group_coupling_alpha": args.st_moe_group_coupling_alpha,
            }
        )
    return cfg


def _print_run_config(
    method: str,
    args: argparse.Namespace,
    imaris_file: Path,
    groundtruth_df: pd.DataFrame | None = None,
    regime_label: str | None = None,
) -> None:
    meta = load_meta_for_imaris_file(imaris_file)
    n_stat, n_migr = _count_synthetic_celltypes(groundtruth_df)
    payload: dict[str, object] = {
        "input_file": str(imaris_file),
        "regime": regime_label,
        "dataset_seed": meta.get("seed") if isinstance(meta, dict) else None,
        "n_stationary_tracks": n_stat,
        "n_migrating_tracks": n_migr,
        "method_config": _snapshot_method_config(args, method),
    }
    print("[RUN_CONFIG] " + json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _assert_no_nan_corrected(df_orig: pd.DataFrame, df_corr: pd.DataFrame, method_name: str) -> None:
    req = ["Position X_corr", "Position Y_corr", "Position Z_corr"]
    if not set(req).issubset(df_corr.columns):
        return
    valid_in = np.all(
        np.isfinite(df_orig[["Position X", "Position Y", "Position Z"]].to_numpy(float)),
        axis=1,
    )
    xyzc = df_corr[req].to_numpy(float)
    bad = valid_in & (~np.all(np.isfinite(xyzc), axis=1))
    if np.any(bad):
        raise ValueError(f"{method_name}: corrected coordinates contain NaN/Inf on {int(np.sum(bad))} valid rows.")


def _save_pred_drift_sanity(df_orig: pd.DataFrame, df_corr: pd.DataFrame, out_prefix: Path, method_name: str) -> None:
    req = ["Position X_corr", "Position Y_corr", "Position Z_corr"]
    if not set(req).issubset(df_corr.columns):
        return
    mm = df_orig[["Time", "Position X", "Position Y", "Position Z"]].copy()
    mm["dx"] = mm["Position X"].to_numpy(float) - df_corr["Position X_corr"].to_numpy(float)
    mm["dy"] = mm["Position Y"].to_numpy(float) - df_corr["Position Y_corr"].to_numpy(float)
    mm["dz"] = mm["Position Z"].to_numpy(float) - df_corr["Position Z_corr"].to_numpy(float)
    mm["dmag"] = np.sqrt(mm["dx"].to_numpy(float) ** 2 + mm["dy"].to_numpy(float) ** 2 + mm["dz"].to_numpy(float) ** 2)
    tab = (
        mm.groupby("Time", as_index=False)
        .agg(
            pred_drift_median=("dmag", "median"),
            pred_drift_p95=("dmag", lambda x: float(np.percentile(np.asarray(x, dtype=float), 95))),
        )
        .sort_values("Time")
        .reset_index(drop=True)
    )
    tab.to_csv(str(out_prefix) + f"_pred_drift_sanity_{method_name}.csv", index=False)


def _collect_output_manifest(out_root: Path) -> list[str]:
    files = [p for p in out_root.rglob("*") if p.is_file()]
    rel = [str(p.relative_to(out_root)) for p in files]
    return sorted(rel)


def _write_manifest(manifest_path: Path, entries: list[str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for x in sorted(entries):
            f.write(str(x) + "\n")


def _read_manifest(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return sorted([ln for ln in lines if ln])


def _validate_manifest(out_root: Path, manifest_path: Path) -> None:
    expected = set(_read_manifest(manifest_path))
    produced = set(_collect_output_manifest(out_root))
    missing = sorted(expected - produced)
    if missing:
        preview = ", ".join(missing[:12])
        raise RuntimeError(f"Regression manifest mismatch: {len(missing)} expected artifacts missing. First: {preview}")


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
        if {"diag_type", "center_nn_median", "center_nn_p90"}.issubset(tf_table.columns):
            tt = tf_table[tf_table["diag_type"] == "timepoint"].copy()
            if tt.shape[0] > 0:
                cmed = pd.to_numeric(tt["center_nn_median"], errors="coerce").to_numpy(float)
                cp90 = pd.to_numeric(tt["center_nn_p90"], errors="coerce").to_numpy(float)
                finite_cov = np.isfinite(cmed) & np.isfinite(cp90)
                if not np.any(finite_cov):
                    raise RuntimeError("RBF coverage diagnostics invalid: no finite center distance values.")
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
        ell_x=args.st_ell_x,
        ell_y=args.st_ell_y,
        ell_z=args.st_ell_z,
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
        st_linear_interactions=bool(args.st_linear_interactions),
        weight_abort_threshold=args.weight_abort_threshold,
        per_axis_scaling=bool(args.st_per_axis_scaling),
        refine_tol=args.st_refine_tol,
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
            ell_x=args.st_ell_x,
            ell_y=args.st_ell_y,
            ell_z=args.st_ell_z,
            lam=args.st_lam if args.st_lam is not None else args.lam,
            robust_keep_fraction=args.st_robust_keep_fraction,
            robust_iters=args.st_robust_iters,
            space_input_mode=args.space_input_mode,
            n_refine=args.st_refine_iters,
            two_stage=bool(args.two_stage),
            robust_mode=args.st_robust_mode,
            track_trim_iters=args.st_track_trim_iters,
            rbf_linear_terms=bool(args.rbf_linear_terms),
            st_linear_interactions=bool(args.st_linear_interactions),
            weight_abort_threshold=args.weight_abort_threshold,
            per_axis_scaling=bool(args.st_per_axis_scaling),
            refine_tol=args.st_refine_tol,
            drift_cap_factor=float(args.st_moe_drift_cap_factor),
            group_coupling_alpha=float(args.st_moe_group_coupling_alpha),
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
    regime_label: str | None = None,
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
            _print_run_config(method, args, f, groundtruth_df=groundtruth_df, regime_label=regime_label)
            use_stationary = (
                stationary_ids
                if (bool(args.use_stationary_subset) or bool(getattr(args, "use_stationary_only", 0)))
                and method in {"rbf", "st_rbf", "st_rbf_moe"}
                else None
            )
            corr = _run_method(method, df, out_prefix, args, fit_track_ids=use_stationary)
            corr_by_method.update(corr)
            for mname, dfc in corr.items():
                _assert_no_nan_corrected(df, dfc, mname)
                _save_pred_drift_sanity(df, dfc, out_prefix, mname)
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

    _print_run_config(args.method, args, in_path, groundtruth_df=groundtruth_df, regime_label=None)
    corr_dict = _run_method(args.method, df, out_prefix, args, fit_track_ids=use_stationary)
    for mname, dfc in corr_dict.items():
        _assert_no_nan_corrected(df, dfc, mname)
        _save_pred_drift_sanity(df, dfc, out_prefix, mname)
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
    if not (bool(getattr(args, "reuse_existing", 0)) and list_imaris_xls_files(synthetic_dir)):
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
    else:
        print(f"Reusing existing synthetic dataset in {synthetic_dir}")

    out_dir = Path(args.out) if args.out is not None else synthetic_dir / "driftcorr_results"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"Synthetic IMARIS folder generated: {synthetic_dir}")
    return _run_batch_for_folder(synthetic_dir, out_dir, args, methods)


def run_synthetic_generate(args: argparse.Namespace) -> int:
    """Generate synthetic IMARIS data only (no correction run)."""
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
    print(f"Synthetic dataset generated: {synthetic_dir}")
    return 0


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
    rows_time: list[dict[str, object]] = []

    for regime_name, regime_cfg in _regime_presets().items():
        for seed in seed_values:
            reg_dir = synthetic_root / f"{regime_name}_seed_{seed:03d}"
            if not (bool(getattr(args, "reuse_existing", 0)) and list_imaris_xls_files(reg_dir)):
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
            else:
                print(f"Reusing existing dataset: {reg_dir}")
            out_dir_run = out_root / f"{regime_name}_seed_{seed:03d}"
            _run_batch_for_folder(reg_dir, out_dir_run, args, methods, regime_label=regime_name)

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
            for method_name in gts["method"].astype(str).tolist():
                p_time = out_prefix.with_name(out_prefix.name + f"_gt_rms_over_time_{method_name}.csv")
                if not p_time.exists():
                    continue
                tt = pd.read_csv(p_time)
                if {"Time", "rms_error"}.issubset(tt.columns):
                    for row in tt.to_dict(orient="records"):
                        rows_time.append(
                            {
                                "regime": regime_name,
                                "seed": int(seed),
                                "method": str(method_name),
                                "Time": float(row["Time"]),
                                "rms_error": float(row["rms_error"]),
                            }
                        )
            print(f"Regime run done: regime={regime_name} seed={seed} -> {out_prefix.parent}")

    if not rows_runs:
        raise RuntimeError("No regime benchmark rows were produced.")

    runs_df = pd.DataFrame(rows_runs)
    out_runs = out_root / "synthetic_regime_runs.csv"
    runs_df.to_csv(out_runs, index=False)

    # One long CSV per regime: method, seed, metric_name, metric_value
    metric_cols = [c for c in runs_df.columns if c not in {"regime", "seed", "method"}]
    long_rows: list[dict[str, object]] = []
    for _, r in runs_df.iterrows():
        for mc in metric_cols:
            long_rows.append(
                {
                    "regime": str(r["regime"]),
                    "method": str(r["method"]),
                    "seed": int(r["seed"]),
                    "metric_name": str(mc),
                    "metric_value": float(r[mc]) if np.isfinite(pd.to_numeric(r[mc], errors="coerce")) else np.nan,
                }
            )
    long_df = pd.DataFrame(long_rows)
    for regime_name in sorted(long_df["regime"].unique()):
        tab = long_df[long_df["regime"] == regime_name].copy()
        tab.to_csv(out_root / f"{regime_name}_seed_metrics_long.csv", index=False)

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

    # Mean/std band over seeds for RMS over time.
    if rows_time:
        tdf = pd.DataFrame(rows_time)
        tdf.to_csv(out_root / "synthetic_regime_rms_over_time_long.csv", index=False)
        for regime_name, trg in tdf.groupby("regime"):
            fig = plt.figure(figsize=(9, 5))
            for method_name, mm in trg.groupby("method"):
                cur = (
                    mm.groupby("Time", as_index=False)
                    .agg(mean=("rms_error", "mean"), std=("rms_error", "std"))
                    .sort_values("Time")
                    .reset_index(drop=True)
                )
                x = cur["Time"].to_numpy(float)
                y = cur["mean"].to_numpy(float)
                s = np.nan_to_num(cur["std"].to_numpy(float), nan=0.0)
                plt.plot(x, y, marker="o", label=str(method_name))
                plt.fill_between(x, y - s, y + s, alpha=0.15)
            plt.xlabel("Time")
            plt.ylabel("RMS error (um)")
            plt.title(f"{regime_name}: RMS over time (mean±std across seeds)")
            plt.ylim(bottom=0.0)
            handles, _ = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend()
            plt.tight_layout()
            plt.savefig(out_root / f"{regime_name}_rms_over_time_mean_std.png", dpi=300)
            plt.close(fig)

        # Rank plots by final-time RMS and avg RMS.
        rank_rows: list[dict[str, object]] = []
        for regime_name, trg in tdf.groupby("regime"):
            final_t = float(np.nanmax(trg["Time"].to_numpy(float)))
            for method_name, mm in trg.groupby("method"):
                v_final = mm[mm["Time"] == final_t]["rms_error"].to_numpy(float)
                v_avg = mm.groupby("seed", as_index=False)["rms_error"].mean()["rms_error"].to_numpy(float)
                rank_rows.append(
                    {
                        "regime": regime_name,
                        "method": method_name,
                        "final_rms_mean": float(np.nanmean(v_final)) if v_final.size else float("nan"),
                        "avg_rms_mean": float(np.nanmean(v_avg)) if v_avg.size else float("nan"),
                    }
                )
        rdf = pd.DataFrame(rank_rows)
        rdf.to_csv(out_root / "synthetic_regime_rank_metrics.csv", index=False)
        for metric_name, ylabel in [("final_rms_mean", "Final-time RMS (um)"), ("avg_rms_mean", "Average RMS (um)")]:
            fig = plt.figure(figsize=(10, 5))
            xlabels: list[str] = []
            xpos: list[float] = []
            vals: list[float] = []
            c = 0
            for regime_name in sorted(rdf["regime"].unique()):
                rr = rdf[rdf["regime"] == regime_name].sort_values(metric_name)
                for _, r in rr.iterrows():
                    xlabels.append(f"{regime_name}\n{r['method']}")
                    xpos.append(float(c))
                    vals.append(float(r[metric_name]))
                    c += 1
            plt.bar(xpos, vals)
            plt.xticks(xpos, xlabels, rotation=60, ha="right")
            plt.ylabel(ylabel)
            plt.title(f"Method ranking by {metric_name}")
            plt.ylim(bottom=0.0)
            plt.tight_layout()
            plt.savefig(out_root / f"synthetic_regime_rank_{metric_name}.png", dpi=300)
            plt.close(fig)

    print(f"Regime benchmark summary -> {out_agg}")
    return 0


def run_paper_output_regression(args: argparse.Namespace) -> int:
    """Run one fixed synthetic benchmark and validate output artifacts against a manifest."""
    synthetic_dir = Path(args.synthetic_dir)
    out_dir = Path(args.out) if args.out is not None else synthetic_dir / "paper_regression_results"
    manifest_path = Path(args.manifest)
    cols_manifest_path = Path(args.columns_manifest)

    if not (bool(getattr(args, "reuse_existing", 0)) and list_imaris_xls_files(synthetic_dir)):
        generate_synthetic_imaris_folder(
            out_dir=synthetic_dir,
            n_files=int(args.synthetic_n_files),
            n_tracks=int(args.synthetic_n_tracks),
            n_times=int(args.synthetic_n_times),
            stationary_fraction=float(args.synthetic_stationary_fraction),
            migrating_fraction=float(args.synthetic_migrating_fraction),
            seed=int(args.synthetic_seed),
            temporal_smoothness=float(args.synthetic_temporal_smoothness),
            local_mode=str(args.synthetic_local_mode),
            piecewise_regions=int(args.synthetic_piecewise_regions),
            piecewise_transition_um=float(args.synthetic_piecewise_transition_um),
            local_lengthscale_um=float(args.synthetic_local_lengthscale_um),
            local_drift_scale_um=float(args.synthetic_local_drift_scale_um),
            n_rbf_centers=int(args.synthetic_n_rbf_centers),
            measurement_noise_um=float(args.synthetic_measurement_noise_um),
        )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    _run_batch_for_folder(synthetic_dir, out_dir, args, methods, regime_label="paper_regression")

    manifest_entries = _collect_output_manifest(out_dir)
    files = list_imaris_xls_files(synthetic_dir)
    if not files:
        raise RuntimeError("No synthetic input files found for regression harness.")
    base = files[0].stem
    out_prefix = out_dir / base / base
    metrics_csv = out_prefix.with_name(out_prefix.name + "_metrics_summary.csv")
    gt_metrics_csv = out_prefix.with_name(out_prefix.name + "_gt_metrics_summary.csv")
    if not metrics_csv.exists() or not gt_metrics_csv.exists():
        raise RuntimeError("Regression harness failed: key metrics CSV missing.")

    cols_payload = {
        "metrics_summary_columns": pd.read_csv(metrics_csv, nrows=1).columns.tolist(),
        "gt_metrics_summary_columns": pd.read_csv(gt_metrics_csv, nrows=1).columns.tolist(),
    }

    if bool(getattr(args, "create_manifest", 0)):
        _write_manifest(manifest_path, manifest_entries)
        cols_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with cols_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(cols_payload, f, indent=2, sort_keys=True)
        print(f"Created manifest: {manifest_path}")
        print(f"Created metrics columns manifest: {cols_manifest_path}")
        return 0

    _validate_manifest(out_dir, manifest_path)
    with cols_manifest_path.open("r", encoding="utf-8") as f:
        expected_cols = json.load(f)
    for k in ["metrics_summary_columns", "gt_metrics_summary_columns"]:
        if list(expected_cols.get(k, [])) != list(cols_payload.get(k, [])):
            raise RuntimeError(f"Regression harness failed: columns changed for {k}.")

    print("Paper output regression passed.")
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
    p.add_argument("--robust-mode", choices=["none", "sample_trim", "track_trim", "irls_huber"], default="sample_trim")
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
    p.add_argument("--st-ell-space", type=float, default=4.0)
    p.add_argument("--st-ell-time", type=float, default=0.4)
    p.add_argument("--st-robust-keep-fraction", type=float, default=0.83)
    p.add_argument("--st-robust-iters", type=int, default=2)
    p.add_argument("--space-input-mode", choices=["row_position", "track_anchor", "ref_position"], default="ref_position")
    p.add_argument("--st-refine-iters", type=int, default=3)
    p.add_argument(
        "--st-robust-mode",
        choices=["none", "sample_trim", "track_trim", "irls_huber", "hybrid_track_irls"],
        default="sample_trim",
    )
    p.add_argument("--st-track-trim-iters", type=int, default=4)
    p.add_argument("--st-moe-k", type=int, default=3)
    p.add_argument("--st-moe-k-list", default="3")
    p.add_argument("--st-moe-blend-sigma-um", type=float, default=140.0)
    p.add_argument("--st-moe-cluster-mode", choices=["track_anchor", "ref_position"], default="track_anchor")
    p.add_argument("--st-moe-drift-cap-factor", type=float, default=3.0)
    p.add_argument("--st-moe-group-coupling-alpha", type=float, default=0.0)
    p.add_argument("--st-per-axis-scaling", type=int, choices=[0, 1], default=1)
    p.add_argument("--st-ell-x", type=float, default=None)
    p.add_argument("--st-ell-y", type=float, default=None)
    p.add_argument("--st-ell-z", type=float, default=None)
    p.add_argument("--st-linear-interactions", type=int, choices=[0, 1], default=0)
    p.add_argument("--st-refine-tol", type=float, default=0.0)

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
    p.add_argument("--reuse-existing", type=int, choices=[0, 1], default=0)


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
    synth.add_argument("--methods", default="rbf,rigid,local,st_rbf", help="Comma-separated method names")
    synth.add_argument("--synthetic-n-files", type=int, default=5)
    synth.add_argument("--synthetic-n-tracks", type=int, default=80)
    synth.add_argument("--synthetic-n-times", type=int, default=20)
    synth.add_argument("--synthetic-stationary-fraction", type=float, default=0.45)
    synth.add_argument("--synthetic-migrating-fraction", type=float, default=0.35)
    synth.add_argument("--synthetic-seed", type=int, default=42)
    synth.add_argument("--synthetic-temporal-smoothness", type=float, default=0.9)

    synth_gen = sub.add_parser("synthetic-generate", help="Generate synthetic IMARIS folder only (no correction)")
    synth_gen.add_argument("--synthetic-dir", default="synthetic_imaris_files")
    synth_gen.add_argument("--synthetic-n-files", type=int, default=5)
    synth_gen.add_argument("--synthetic-n-tracks", type=int, default=80)
    synth_gen.add_argument("--synthetic-n-times", type=int, default=20)
    synth_gen.add_argument("--synthetic-stationary-fraction", type=float, default=0.45)
    synth_gen.add_argument("--synthetic-migrating-fraction", type=float, default=0.35)
    synth_gen.add_argument("--synthetic-seed", type=int, default=42)
    synth_gen.add_argument("--synthetic-temporal-smoothness", type=float, default=0.9)

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

    regression = sub.add_parser("paper-regression", help="Run synthetic output regression harness against manifest")
    regression.add_argument("--synthetic-dir", default="Synthetic IMARIS files paper_regression")
    regression.add_argument("--out", default=None, help="Default: <synthetic-dir>/paper_regression_results")
    regression.add_argument("--methods", default="rbf,rigid,local,st_rbf", help="Comma-separated method names")
    regression.add_argument("--manifest", default="paper_outputs_manifest.txt")
    regression.add_argument("--columns-manifest", default="paper_metrics_columns.json")
    regression.add_argument("--create-manifest", type=int, choices=[0, 1], default=0)
    regression.add_argument("--synthetic-n-files", type=int, default=1)
    regression.add_argument("--synthetic-n-tracks", type=int, default=60)
    regression.add_argument("--synthetic-n-times", type=int, default=16)
    regression.add_argument("--synthetic-stationary-fraction", type=float, default=0.45)
    regression.add_argument("--synthetic-migrating-fraction", type=float, default=0.35)
    regression.add_argument("--synthetic-seed", type=int, default=42)
    regression.add_argument("--synthetic-temporal-smoothness", type=float, default=0.9)

    for p in (run, run_all, synth, synth_gen, synth_reg, regression):
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
    if args.command == "synthetic-generate":
        return run_synthetic_generate(args)
    if args.command == "synthetic-regime-benchmark":
        return run_synthetic_regime_benchmark(args)
    if args.command == "paper-regression":
        return run_paper_output_regression(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
