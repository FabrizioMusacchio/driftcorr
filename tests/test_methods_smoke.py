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
from driftcorr.method_spacetime_rbf_moe import apply_spacetime_rbf_moe_correction, fit_spacetime_rbf_moe_model
from driftcorr.stationary import find_stationary_tracks
from driftcorr.synthetic import make_synthetic_tracks
from driftcorr.synthetic_experiments import evaluate_corrected_vs_groundtruth, generate_controlled_drift_dataset


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

    df_rbf_tracktrim, tf_tracktrim, *_ = correct_positions_rbf_to_reference(
        df,
        m=12,
        min_pairs=8,
        center_mode="pooled",
        robust_mode="track_trim",
        track_trim_iters=3,
        two_stage=True,
        space_input_mode="track_anchor",
    )
    assert "Position X_corr" in df_rbf_tracktrim.columns
    assert "center_mode" in tf_tracktrim.columns

    df_rbf_rowpos, tf_rowpos, *_ = correct_positions_rbf_to_reference(
        df,
        m=12,
        min_pairs=8,
        space_input_mode="row_position",
        row_position_iters=4,
        rbf_linear_terms=True,
    )
    assert "Position X_corr" in df_rbf_rowpos.columns
    assert "weight_max_abs" in tf_rowpos.columns

    step, _, times = build_step_transforms(df, min_pairs=8)
    df_rigid = apply_rigid_correction_to_tracks(df, step, times=times, ref_time=float(times.min()))
    assert "Position Y_corr" in df_rigid.columns

    model_piecewise, _ = build_piecewise_rigid_model(df, n_groups=2, min_pairs=8, blend_sigma_um=80.0)
    df_piecewise = apply_piecewise_rigid_correction_to_tracks(df, model_piecewise)
    assert "Position Z_corr" in df_piecewise.columns

    out_prefix = tmp_path / "synthetic"
    df_local, *_ = run_local_drift_correction_for_file(df, out_prefix=out_prefix, k_neighbors=5)
    assert "Position Z_corr" in df_local.columns

    st_model, st_info = fit_spacetime_rbf_model(
        df,
        m_centers=20,
        robust_iters=2,
        space_input_mode="ref_position",
        n_refine=2,
        two_stage=True,
        robust_mode="track_trim",
        track_trim_iters=3,
        rbf_linear_terms=True,
    )
    df_st = apply_correction(df, st_model)
    assert "Position X_corr" in df_st.columns
    assert st_info["n_samples"] > 0

    sel = find_stationary_tracks(df, rms_quantile=0.5, neighbor_quantile=0.6)
    assert len(sel.track_ids) > 0


def test_st_refinement_changes_output_for_ref_position():
    df = make_synthetic_tracks(n_tracks=28, n_times=12, seed=5)
    model0, _ = fit_spacetime_rbf_model(
        df,
        m_centers=18,
        robust_iters=2,
        space_input_mode="ref_position",
        n_refine=1,
        two_stage=False,
        robust_mode="sample_trim",
    )
    model2, _ = fit_spacetime_rbf_model(
        df,
        m_centers=18,
        robust_iters=2,
        space_input_mode="ref_position",
        n_refine=3,
        two_stage=False,
        robust_mode="sample_trim",
    )
    out0 = apply_correction(df, model0)
    out2 = apply_correction(df, model2)

    p0 = out0[["Position X_corr", "Position Y_corr", "Position Z_corr"]].to_numpy(float)
    p2 = out2[["Position X_corr", "Position Y_corr", "Position Z_corr"]].to_numpy(float)
    diff = np.linalg.norm(p0 - p2, axis=1)
    assert float(np.nanmedian(diff)) > 1e-6


def test_st_moe_smoke():
    df = make_synthetic_tracks(n_tracks=30, n_times=10, seed=9)
    model, expert_table = fit_spacetime_rbf_moe_model(
        df,
        n_experts=3,
        blend_sigma_um=100.0,
        space_input_mode="ref_position",
        n_refine=2,
        robust_mode="track_trim",
        track_trim_iters=3,
    )
    out = apply_spacetime_rbf_moe_correction(df, model)
    assert out.shape[0] == df.shape[0]
    assert {"Position X_corr", "Position Y_corr", "Position Z_corr"}.issubset(out.columns)
    assert expert_table.shape[0] >= 1


def test_groundtruth_eval_reports_stationary_and_migrating_rms():
    df_obs, df_gt, _ = generate_controlled_drift_dataset(n_tracks=30, n_times=10, seed=3)
    df_corr = df_obs.copy()
    df_corr["Position X_corr"] = df_corr["Position X"]
    df_corr["Position Y_corr"] = df_corr["Position Y"]
    df_corr["Position Z_corr"] = df_corr["Position Z"]
    summary, *_ = evaluate_corrected_vs_groundtruth(df_corr, df_gt)
    assert "overall_rms_error_stationary" in summary
    assert "overall_rms_error_migrating" in summary
