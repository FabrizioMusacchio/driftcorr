import numpy as np

from driftcorr.metrics import (
    displacements_relative_to_anchor,
    fraction_near_stationary,
    msd_per_track,
    neighbor_distance_stats,
    pairwise_trace_correlation,
    pivot_tracks,
    rms_displacement_per_track,
)
from driftcorr.synthetic import make_synthetic_tracks


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
