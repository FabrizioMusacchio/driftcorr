import numpy as np

from driftcorr.metrics import (
    displacements_relative_to_anchor,
    fraction_near_stationary,
    msd_per_track,
    neighbor_distance_stats,
    per_time_groundtruth_metrics,
    pairwise_trace_correlation,
    pivot_tracks,
    rms_displacement_per_track,
)
from driftcorr.synthetic import make_synthetic_tracks
from driftcorr.synthetic_experiments import generate_controlled_drift_dataset


def test_metrics_smoke():
    df = make_synthetic_tracks(n_tracks=18, n_times=12, seed=1)
    _, _, x = pivot_tracks(df)

    d = displacements_relative_to_anchor(x)
    corr = pairwise_trace_correlation(d)
    assert corr.ndim == 1

    ns = neighbor_distance_stats(x, k_neighbors=4, min_pairs=5)
    assert "rel_std_median" in ns

    tau, msd = msd_per_track(x)
    assert tau.size == x.shape[1] - 1
    assert msd.shape[0] == x.shape[0]

    rms = rms_displacement_per_track(x)
    frac = fraction_near_stationary(rms, threshold_um=10.0)
    assert np.isfinite(frac)


def test_per_time_groundtruth_metrics_smoke():
    df_obs, df_gt, df_drift = generate_controlled_drift_dataset(n_tracks=24, n_times=8, seed=4)
    df_corr = df_obs.copy()
    df_corr["Position X_corr"] = df_corr["Position X"]
    df_corr["Position Y_corr"] = df_corr["Position Y"]
    df_corr["Position Z_corr"] = df_corr["Position Z"]
    tab = per_time_groundtruth_metrics(df_corr, df_gt, drift_field_groundtruth=df_drift)
    assert tab.shape[0] > 0
    assert {"Time", "group", "rms_error", "near_stationary_fraction"}.issubset(tab.columns)
