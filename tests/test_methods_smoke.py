import numpy as np

from driftcorr.method_local_communities import run_local_drift_correction_for_file
from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference
from driftcorr.method_rigid_step import (
    apply_piecewise_rigid_correction_to_tracks,
    apply_rigid_correction_to_tracks,
    build_piecewise_rigid_model,
    build_step_transforms,
)
from driftcorr.method_spacetime_rbf import _choose_centers, apply_correction, fit_spacetime_rbf_model
from driftcorr.stationary import find_stationary_tracks
from driftcorr.synthetic import make_synthetic_tracks


def test_spacetime_choose_centers_returns_subset_rows():
    u = np.arange(60, dtype=float).reshape(15, 4)
    c = _choose_centers(u, m=6, seed=7)
    assert c.shape == (6, 4)
    assert c is not u

    # Every selected center row must correspond to one row from input u.
    u_rows = {tuple(row.tolist()) for row in u}
    c_rows = {tuple(row.tolist()) for row in c}
    assert c_rows.issubset(u_rows)


def test_rbf_rigid_local_smoke(tmp_path):
    df = make_synthetic_tracks(n_tracks=22, n_times=10, seed=2)

    df_rbf, tf, _, times, ref = correct_positions_rbf_to_reference(df, m=12, min_pairs=8)
    assert "Position X_corr" in df_rbf.columns
    assert not tf.empty

    df_rbf_temporal, tf_temporal, *_ = correct_positions_rbf_to_reference(df, m=12, min_pairs=8, temporal_gamma=0.2)
    assert "Position X_corr" in df_rbf_temporal.columns
    assert "temporal_gamma" in tf_temporal.columns

    step, _, times = build_step_transforms(df, min_pairs=8)
    df_rigid = apply_rigid_correction_to_tracks(df, step, times=times, ref_time=float(times.min()))
    assert "Position Y_corr" in df_rigid.columns

    model_piecewise, _ = build_piecewise_rigid_model(df, n_groups=2, min_pairs=8, blend_sigma_um=80.0)
    df_piecewise = apply_piecewise_rigid_correction_to_tracks(df, model_piecewise)
    assert "Position Z_corr" in df_piecewise.columns

    out_prefix = tmp_path / "synthetic"
    df_local, *_ = run_local_drift_correction_for_file(df, out_prefix=out_prefix, k_neighbors=5)
    assert "Position Z_corr" in df_local.columns

    st_model, st_info = fit_spacetime_rbf_model(df, m_centers=20, robust_iters=2)
    df_st = apply_correction(df, st_model)
    assert "Position X_corr" in df_st.columns
    assert st_info["n_samples"] > 0

    sel = find_stationary_tracks(df, rms_quantile=0.5, neighbor_quantile=0.6)
    assert len(sel.track_ids) > 0
