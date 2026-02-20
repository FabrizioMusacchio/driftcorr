"""Parameter sweep for RBF and ST-RBF on controlled synthetic drift datasets.

This script generates synthetic datasets in memory, runs all four correction methods,
and performs two sweeps:
1) RBF parameter sweep (while ST-RBF stays at default).
2) ST-RBF parameter sweep (while RBF stays at default).

For each sweep configuration, the script reports ground-truth RMS errors aggregated
across synthetic datasets and identifies the best configuration for the target method.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftcorr.io_imaris import ensure_numeric_tracks
from driftcorr.method_local_communities import (
    _compute_displacements_to_ref,
    apply_drift_correction,
    build_drift_model,
    build_similarity_graph,
)
from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference
from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks, build_step_transforms
from driftcorr.method_spacetime_rbf import apply_correction as apply_spacetime_correction
from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model
from driftcorr.metrics import pivot_tracks
from driftcorr.synthetic_experiments import evaluate_corrected_vs_groundtruth, generate_controlled_drift_dataset


@dataclass(frozen=True)
class DatasetPair:
    name: str
    df_orig: pd.DataFrame
    df_gt: pd.DataFrame


def correct_local_in_memory(
    df: pd.DataFrame,
    reference_time: float | None = None,
    k_neighbors: int = 12,
    radius_um: float | None = None,
    ell_um_graph: float = 80.0,
    ell_um_field: float = 120.0,
    min_edge_weight: float = 0.25,
    max_pairs_per_node: int | None = 50,
) -> pd.DataFrame:
    """Apply local-community correction without writing files."""
    d = ensure_numeric_tracks(df)
    times, track_ids, x = pivot_tracks(d)
    if reference_time is None:
        reference_time = float(times.min())

    keep_ids, p0, disp = _compute_displacements_to_ref(times, track_ids, x, reference_time=float(reference_time))
    comm = build_similarity_graph(
        p0=p0,
        d=disp,
        track_ids=keep_ids,
        k_neighbors=k_neighbors,
        radius_um=radius_um,
        ell_um=ell_um_graph,
        min_edge_weight=min_edge_weight,
        max_pairs_per_node=max_pairs_per_node,
    )
    model = build_drift_model(
        times=times,
        reference_time=float(reference_time),
        p0=p0,
        d=disp,
        comm_res=comm,
        ell_um=ell_um_field,
    )
    return apply_drift_correction(d, model, use_track_mean_position=True)


def eval_method(df_corr: pd.DataFrame, df_gt: pd.DataFrame, method: str) -> dict[str, float | str]:
    summary, _, _, _ = evaluate_corrected_vs_groundtruth(df_corr, df_gt)
    return {
        "method": method,
        "overall_rms_error": float(summary["overall_rms_error"]),
        "mean_time_rms_error": float(summary["mean_time_rms_error"]),
        "median_track_rms_error": float(summary["median_track_rms_error"]),
        "p90_track_rms_error": float(summary["p90_track_rms_error"]),
    }


def make_datasets(n_datasets: int, n_tracks: int, n_times: int, base_seed: int) -> list[DatasetPair]:
    out: list[DatasetPair] = []
    for i in range(n_datasets):
        seed = base_seed + 100 * i
        df_orig, df_gt, _ = generate_controlled_drift_dataset(
            n_tracks=n_tracks,
            n_times=n_times,
            seed=seed,
            stationary_fraction=0.45,
            migrating_fraction=0.35,
            temporal_smoothness=0.9,
        )
        out.append(DatasetPair(name=f"synthetic_{i+1:02d}", df_orig=ensure_numeric_tracks(df_orig), df_gt=ensure_numeric_tracks(df_gt)))
    return out


def build_static_baselines(datasets: list[DatasetPair]) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute rigid/local defaults once per dataset."""
    baselines: dict[str, dict[str, pd.DataFrame]] = {}
    for ds in datasets:
        df = ds.df_orig
        step, _, times = build_step_transforms(
            df,
            keep_fraction=0.7,
            min_pairs=15,
            mode="rigid",
            inlier_reestimate_iters=30,
        )
        ref_time = float(times.min())
        df_rigid = apply_rigid_correction_to_tracks(df, step, times, ref_time=ref_time)
        df_local = correct_local_in_memory(df)

        df_rbf_default, _, _, _, _ = correct_positions_rbf_to_reference(
            df,
            reference_time=None,
            m=25,
            keep_fraction=0.7,
            lam=1e-2,
            sigma=None,
            min_pairs=15,
            seed=0,
            temporal_gamma=0.0,
        )
        st_model, _ = fit_spacetime_rbf_model(
            df,
            reference_time=None,
            m_centers=40,
            ell_space=120.0,
            ell_time=5.0,
            lam=1e-2,
            robust_keep_fraction=0.7,
            robust_iters=4,
            seed=0,
        )
        df_st_default = apply_spacetime_correction(df, st_model)

        baselines[ds.name] = {
            "rigid": df_rigid,
            "local": df_local,
            "rbf_default": df_rbf_default,
            "st_default": df_st_default,
        }
    return baselines


def sweep_rbf(
    datasets: list[DatasetPair],
    baselines: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grid = list(
        product(
            [25, 45],  # m
            [3e-4, 1e-2, 3e-1],  # lam
            [0.0, 1.0, 5.0],  # temporal_gamma
            [None, 180.0],  # sigma
        )
    )
    for config_id, (m, lam, gamma, sigma) in enumerate(grid, start=1):
        for ds in datasets:
            df_rbf, _, _, _, _ = correct_positions_rbf_to_reference(
                ds.df_orig,
                reference_time=None,
                m=int(m),
                keep_fraction=0.7,
                lam=float(lam),
                sigma=None if sigma is None else float(sigma),
                min_pairs=15,
                seed=0,
                temporal_gamma=float(gamma),
            )
            method_tables = {
                "rbf": df_rbf,
                "rigid": baselines[ds.name]["rigid"],
                "local": baselines[ds.name]["local"],
                "st_rbf": baselines[ds.name]["st_default"],
            }
            for method, df_corr in method_tables.items():
                ev = eval_method(df_corr, ds.df_gt, method)
                rows.append(
                    {
                        "sweep": "rbf",
                        "config_id": int(config_id),
                        "dataset": ds.name,
                        "method": method,
                        "rbf_m": int(m),
                        "rbf_lam": float(lam),
                        "rbf_temporal_gamma": float(gamma),
                        "rbf_sigma": "auto" if sigma is None else float(sigma),
                        "st_m_centers": 40,
                        "st_lam": 1e-2,
                        "st_ell_space": 120.0,
                        "st_ell_time": 5.0,
                        "st_robust_keep_fraction": 0.7,
                        **ev,
                    }
                )
    return pd.DataFrame(rows)


def sweep_st_rbf(
    datasets: list[DatasetPair],
    baselines: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grid = list(
        product(
            [40, 70],  # m_centers
            [3e-4, 1e-2, 3e-1],  # lam
            [90.0, 160.0, 260.0],  # ell_space
            [2.5, 6.0],  # ell_time
            [0.6, 0.8],  # robust_keep_fraction
        )
    )
    for config_id, (m_centers, lam, ell_space, ell_time, keep_frac) in enumerate(grid, start=1):
        for ds in datasets:
            model, _ = fit_spacetime_rbf_model(
                ds.df_orig,
                reference_time=None,
                m_centers=int(m_centers),
                ell_space=float(ell_space),
                ell_time=float(ell_time),
                lam=float(lam),
                robust_keep_fraction=float(keep_frac),
                robust_iters=4,
                seed=0,
            )
            df_st = apply_spacetime_correction(ds.df_orig, model)
            method_tables = {
                "st_rbf": df_st,
                "rigid": baselines[ds.name]["rigid"],
                "local": baselines[ds.name]["local"],
                "rbf": baselines[ds.name]["rbf_default"],
            }
            for method, df_corr in method_tables.items():
                ev = eval_method(df_corr, ds.df_gt, method)
                rows.append(
                    {
                        "sweep": "st_rbf",
                        "config_id": int(config_id),
                        "dataset": ds.name,
                        "method": method,
                        "rbf_m": 25,
                        "rbf_lam": 1e-2,
                        "rbf_temporal_gamma": 0.0,
                        "rbf_sigma": "auto",
                        "st_m_centers": int(m_centers),
                        "st_lam": float(lam),
                        "st_ell_space": float(ell_space),
                        "st_ell_time": float(ell_time),
                        "st_robust_keep_fraction": float(keep_frac),
                        **ev,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "sweep",
        "config_id",
        "method",
        "rbf_m",
        "rbf_lam",
        "rbf_temporal_gamma",
        "rbf_sigma",
        "st_m_centers",
        "st_lam",
        "st_ell_space",
        "st_ell_time",
        "st_robust_keep_fraction",
    ]
    agg = (
        df_raw.groupby(group_cols, as_index=False)
        .agg(
            overall_rms_mean=("overall_rms_error", "mean"),
            overall_rms_std=("overall_rms_error", "std"),
            mean_time_rms_mean=("mean_time_rms_error", "mean"),
            median_track_rms_mean=("median_track_rms_error", "mean"),
            p90_track_rms_mean=("p90_track_rms_error", "mean"),
        )
        .sort_values(["sweep", "config_id", "method"])
        .reset_index(drop=True)
    )
    return agg


def best_target_configs(df_agg: pd.DataFrame) -> pd.DataFrame:
    targets = []
    for sweep_name, target_method in [("rbf", "rbf"), ("st_rbf", "st_rbf")]:
        sub = df_agg[(df_agg["sweep"] == sweep_name) & (df_agg["method"] == target_method)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["overall_rms_mean", "mean_time_rms_mean"], ascending=True)
        best = sub.head(1).copy()
        best.insert(0, "target_method", target_method)
        targets.append(best)
    if not targets:
        return pd.DataFrame()
    return pd.concat(targets, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep RBF and ST-RBF parameters on synthetic datasets.")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "Synthetic IMARIS parameter sweep"))
    p.add_argument("--n-datasets", type=int, default=3)
    p.add_argument("--n-tracks", type=int, default=70)
    p.add_argument("--n-times", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic datasets...")
    datasets = make_datasets(
        n_datasets=int(args.n_datasets),
        n_tracks=int(args.n_tracks),
        n_times=int(args.n_times),
        base_seed=int(args.seed),
    )
    print("Computing static baselines (rigid/local/default-rbf/default-st_rbf)...")
    baselines = build_static_baselines(datasets)

    print("Running RBF sweep...")
    df_rbf = sweep_rbf(datasets, baselines)
    print("Running ST-RBF sweep...")
    df_st = sweep_st_rbf(datasets, baselines)

    df_raw = pd.concat([df_rbf, df_st], ignore_index=True)
    df_agg = aggregate_results(df_raw)
    df_best = best_target_configs(df_agg)

    raw_path = out_dir / "sweep_raw.csv"
    agg_path = out_dir / "sweep_aggregated.csv"
    best_path = out_dir / "sweep_best_configs.csv"
    df_raw.to_csv(raw_path, index=False)
    df_agg.to_csv(agg_path, index=False)
    df_best.to_csv(best_path, index=False)

    print(f"Saved raw sweep table: {raw_path}")
    print(f"Saved aggregated sweep table: {agg_path}")
    print(f"Saved best configs: {best_path}")
    if not df_best.empty:
        print("\nBest target configurations:")
        print(df_best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

