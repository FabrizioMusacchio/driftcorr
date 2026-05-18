"""Unified quantitative evaluation for drift correction results.

This module computes method-agnostic comparison metrics between original tracks
and corrected tracks and exports standardized summary plots/tables.

Core mathematical quantities:
- Displacements relative to anchor:
  `d_i(t) = x_i(t) - x_i(t_anchor)`.
- Pairwise trace correlation uses correlation of `||d_i(t)||` across tracks.
- Neighbor stability evaluates temporal variation of nearest-neighbor distances.
- Mean-squared displacement (MSD):
  `MSD_i(Δt) = E_t ||x_i(t+Δt) - x_i(t)||^2`.

Controlling parameters:
- `k_neighbors`: locality in neighbor-distance stability metrics.
- `min_overlap_corr`: minimum overlapping timepoints for correlation estimation.
- `near_stationary_threshold_um`: RMS threshold for near-stationary fraction.

Effects:
- Increasing `k_neighbors` broadens geometric context for stability checks.
- Increasing `min_overlap_corr` yields stricter, potentially fewer correlations.
- Increasing `near_stationary_threshold_um` raises the stationary-track fraction.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import (
    align_orig_and_corr,
    displacements_relative_to_anchor,
    fraction_near_stationary,
    msd_per_track,
    neighbor_distance_stats,
    pairwise_trace_correlation,
    rms_displacement_per_track,
    scalarize_metrics,
    summarize_msd,
)


def _nanmedian_no_warning(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """NaN-median that avoids all-NaN RuntimeWarning noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def evaluate_method(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
    k_neighbors: int = 8,
    min_overlap_corr: int = 4,
    near_stationary_threshold_um: float = 2.0,
) -> dict[str, Any]:
    """Evaluate one corrected dataframe against original tracks."""
    times, track_ids, xo, xc = align_orig_and_corr(df_orig, df_corr)

    do = displacements_relative_to_anchor(xo)
    dc = displacements_relative_to_anchor(xc)

    corr_orig = pairwise_trace_correlation(do, min_overlap=min_overlap_corr)
    corr_corr = pairwise_trace_correlation(dc, min_overlap=min_overlap_corr)

    ns_o = neighbor_distance_stats(xo, k_neighbors=k_neighbors)
    ns_c = neighbor_distance_stats(xc, k_neighbors=k_neighbors)

    tau, msd_o = msd_per_track(xo)
    _, msd_c = msd_per_track(xc)

    msd_med_o = _nanmedian_no_warning(msd_o, axis=0)
    msd_med_c = _nanmedian_no_warning(msd_c, axis=0)

    rms_o = rms_displacement_per_track(xo)
    rms_c = rms_displacement_per_track(xc)

    metrics: dict[str, Any] = {
        "n_common_tracks": int(track_ids.size),
        "n_common_times": int(times.size),
        "corr_mean_orig": float(np.nanmean(corr_orig)) if corr_orig.size else np.nan,
        "corr_median_orig": float(np.nanmedian(corr_orig)) if corr_orig.size else np.nan,
        "corr_mean_corr": float(np.nanmean(corr_corr)) if corr_corr.size else np.nan,
        "corr_median_corr": float(np.nanmedian(corr_corr)) if corr_corr.size else np.nan,
        "neighbor_relstd_median_orig": float(ns_o.get("rel_std_median", np.nan)),
        "neighbor_relstd_p90_orig": float(ns_o.get("rel_std_p90", np.nan)),
        "neighbor_relstd_median_corr": float(ns_c.get("rel_std_median", np.nan)),
        "neighbor_relstd_p90_corr": float(ns_c.get("rel_std_p90", np.nan)),
        "rms_disp_mean_orig": float(np.nanmean(rms_o)),
        "rms_disp_median_orig": float(np.nanmedian(rms_o)),
        "rms_disp_mean_corr": float(np.nanmean(rms_c)),
        "rms_disp_median_corr": float(np.nanmedian(rms_c)),
        "near_stationary_fraction_orig": fraction_near_stationary(rms_o, threshold_um=near_stationary_threshold_um),
        "near_stationary_fraction_corr": fraction_near_stationary(rms_c, threshold_um=near_stationary_threshold_um),
        "near_stationary_threshold_um": float(near_stationary_threshold_um),
        "msd_tau": tau,
        "msd_median_orig": msd_med_o,
        "msd_median_corr": msd_med_c,
        "corr_values_orig": corr_orig,
        "corr_values_corr": corr_corr,
        "rms_per_track_orig": rms_o,
        "rms_per_track_corr": rms_c,
        "msd_summary_orig": summarize_msd(msd_o),
        "msd_summary_corr": summarize_msd(msd_c),
    }
    return metrics


def run_all_methods_and_compare(
    df_orig: pd.DataFrame,
    methods: (
        dict[str, pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame]]
        | list[tuple[str, pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame]]]
    ),
    **eval_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate a set of methods and return tidy summary + detailed arrays."""
    if isinstance(methods, dict):
        method_items = list(methods.items())
    else:
        method_items = list(methods)

    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}

    for name, method_value in method_items:
        if callable(method_value):
            df_corr = method_value(df_orig)
        else:
            df_corr = method_value

        result = evaluate_method(df_orig, df_corr, **eval_kwargs)
        result["method"] = name
        details[name] = result

        row = scalarize_metrics(result)
        row["method"] = name

        msd_corr_summary = result.get("msd_summary_corr", {})
        if isinstance(msd_corr_summary, dict):
            for k, v in msd_corr_summary.items():
                row[f"corr_{k}"] = v
        msd_orig_summary = result.get("msd_summary_orig", {})
        if isinstance(msd_orig_summary, dict):
            for k, v in msd_orig_summary.items():
                row[f"orig_{k}"] = v

        rows.append(row)

    summary = pd.DataFrame(rows)
    if "method" in summary.columns:
        summary = summary.sort_values("method").reset_index(drop=True)
    return summary, details


def _plot_corr_hist(details: dict[str, dict[str, Any]], out_path: str | Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    first = next(iter(details.values()))
    co = np.asarray(first.get("corr_values_orig", []), dtype=float)
    if co.size:
        plt.hist(co, bins=40, alpha=0.35, label="original")
    for name, res in details.items():
        cc = np.asarray(res.get("corr_values_corr", []), dtype=float)
        if cc.size:
            plt.hist(cc, bins=40, alpha=0.35, label=name)
    plt.xlabel("pairwise corr of ||displacement(t)||")
    plt.ylabel("count")
    plt.title("Correlation histogram: original vs corrected")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_msd(details: dict[str, dict[str, Any]], out_path: str | Path) -> None:
    def _aligned_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = min(int(x.size), int(y.size))
        if n <= 0:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
        return x[:n], y[:n]

    fig = plt.figure(figsize=(8, 5))
    first = next(iter(details.values()))
    tau = np.asarray(first.get("msd_tau", []), dtype=float)
    m0 = np.asarray(first.get("msd_median_orig", []), dtype=float)
    x0, y0 = _aligned_xy(tau, m0)
    if x0.size and y0.size:
        plt.plot(x0, y0, marker="o", label="original")
    for name, res in details.items():
        tau_i = np.asarray(res.get("msd_tau", tau), dtype=float)
        mc = np.asarray(res.get("msd_median_corr", []), dtype=float)
        xi, yi = _aligned_xy(tau_i, mc)
        if xi.size and yi.size:
            plt.plot(xi, yi, marker="o", label=name)
    plt.xlabel("lag (frames)")
    plt.ylabel("MSD (um^2)")
    plt.title("MSD median curve: original vs corrected")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_neighbor_bars(details: dict[str, dict[str, Any]], out_path: str | Path) -> None:
    methods = list(details.keys())
    first = details[methods[0]]

    labels = ["median rel std", "p90 rel std"]
    orig_vals = [
        float(first.get("neighbor_relstd_median_orig", np.nan)),
        float(first.get("neighbor_relstd_p90_orig", np.nan)),
    ]

    x = np.arange(len(labels), dtype=float)
    n_bars = len(methods) + 1
    width = 0.8 / max(n_bars, 1)

    fig = plt.figure(figsize=(9, 5))
    plt.bar(x - 0.4 + width / 2, orig_vals, width=width, label="original")
    for i, name in enumerate(methods):
        vals = [
            float(details[name].get("neighbor_relstd_median_corr", np.nan)),
            float(details[name].get("neighbor_relstd_p90_corr", np.nan)),
        ]
        xpos = x - 0.4 + width * (i + 1) + width / 2
        plt.bar(xpos, vals, width=width, label=name)

    plt.xticks(x, labels)
    plt.ylabel("relative distance variability")
    plt.title("Neighbor-distance stability: original vs corrected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def save_evaluation_outputs(
    base_out_prefix: str | Path,
    summary_df: pd.DataFrame,
    details: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Save standard CSV + plot outputs for one file."""
    base_out_prefix = Path(base_out_prefix)
    base_out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_summary = base_out_prefix.with_name(base_out_prefix.name + "_metrics_summary.csv")
    out_corr = base_out_prefix.with_name(base_out_prefix.name + "_plots_corr_hist.png")
    out_msd = base_out_prefix.with_name(base_out_prefix.name + "_plots_msd_curve.png")
    out_neighbor = base_out_prefix.with_name(base_out_prefix.name + "_plots_neighbor_stats.png")

    summary_df.to_csv(out_summary, index=False)
    if details:
        _plot_corr_hist(details, out_corr)
        _plot_msd(details, out_msd)
        _plot_neighbor_bars(details, out_neighbor)

    return {
        "metrics_summary": out_summary,
        "plot_corr_hist": out_corr,
        "plot_msd_curve": out_msd,
        "plot_neighbor_stats": out_neighbor,
    }
