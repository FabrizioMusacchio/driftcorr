# %% IMPORTS
import numpy as np

from driftcorr.method_rigid_step import (
    apply_rigid_correction_to_tracks,
    apply_rigid_correction_with_optional_smoothing,
    build_step_transforms)
from driftcorr.synthetic import make_synthetic_tracks
# %% TEST FUNCTIONS
def test_rigid_optional_smoothing_disabled_matches_original():
    df = make_synthetic_tracks(n_tracks=24, n_times=10, seed=11)
    step, _, times = build_step_transforms(df, min_pairs=8, keep_fraction=0.7, mode="rigid")
    ref_time = float(times.min())

    df_base = apply_rigid_correction_to_tracks(df, step, times, ref_time)
    df_wrap = apply_rigid_correction_with_optional_smoothing(
        df=df,
        step=step,
        times=times,
        ref_time=ref_time,
        smoothing_enabled=False,
    )

    a = df_base[["Position X_corr", "Position Y_corr", "Position Z_corr"]].to_numpy(float)
    b = df_wrap[["Position X_corr", "Position Y_corr", "Position Z_corr"]].to_numpy(float)
    assert np.allclose(a, b, rtol=0.0, atol=0.0)
# %% END