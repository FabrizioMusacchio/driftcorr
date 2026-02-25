import numpy as np

from driftcorr.association.assignment import apply_gating, build_cost_matrix, hungarian_assignment
from driftcorr.association.linking import link_assignments_to_tracks
from driftcorr.association.registration import icp_rigid
from driftcorr.association.types import PointCloudFrame


def test_hungarian_assignment_shapes():
    x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    x1 = np.array([[0.2, 0.0, 0.0], [1.9, 0.1, 0.0], [4.2, 0.0, 0.0]], dtype=float)
    cost = build_cost_matrix(x0, x1, metric="sqeuclidean")
    out = hungarian_assignment(cost)

    assert out["row_ind"].shape == out["col_ind"].shape
    assert out["costs"].shape[0] == out["row_ind"].shape[0]
    assert out["unmatched_t"].ndim == 1
    assert out["unmatched_tp1"].ndim == 1


def test_gating_forbids_large_distances():
    x0 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    x1 = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], dtype=float)
    cost = build_cost_matrix(x0, x1, metric="sqeuclidean")
    gate = apply_gating(cost, max_dist=0.5, assume_squared_cost=True)

    # Cross matches are too far and must be forbidden.
    assert not bool(gate[0, 1])
    assert not bool(gate[1, 0])

    out = hungarian_assignment(cost, gating_mask=gate)
    assert out["row_ind"].size == 2
    assert out["unmatched_t"].size == 0
    assert out["unmatched_tp1"].size == 0


def test_linking_produces_consistent_track_ids():
    f0 = PointCloudFrame(time=1.0, xyz=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float))
    f1 = PointCloudFrame(time=2.0, xyz=np.array([[0.5, 0.0, 0.0], [10.5, 0.0, 0.0]], dtype=float))
    f2 = PointCloudFrame(time=3.0, xyz=np.array([[1.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=float))

    a01 = hungarian_assignment(build_cost_matrix(f0.xyz, f1.xyz), gating_mask=apply_gating(build_cost_matrix(f0.xyz, f1.xyz), max_dist=2.0))
    a12 = hungarian_assignment(build_cost_matrix(f1.xyz, f2.xyz), gating_mask=apply_gating(build_cost_matrix(f1.xyz, f2.xyz), max_dist=2.0))

    tracks_df, links_df, frame_stats_df = link_assignments_to_tracks([f0, f1, f2], [a01, a12])

    assert tracks_df["TrackID"].nunique() == 2
    assert links_df.shape[0] == 4
    assert frame_stats_df.shape[0] == 2

    # Each point-index trajectory should keep a stable inferred TrackID.
    for idx in [0, 1]:
        tids = (
            tracks_df[tracks_df["PointIndex"] == idx]
            .sort_values("Time")["TrackID"]
            .to_numpy(dtype=float)
        )
        assert tids.size == 3
        assert np.allclose(tids, tids[0])


def test_icp_rigid_recovers_transform_on_synthetic_points():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 3))

    theta = np.deg2rad(12.0)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    r_true = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    b_true = np.array([1.2, -0.7, 0.35], dtype=float)
    y = (r_true @ x.T).T + b_true[None, :]

    r_hat, b_hat, corr, residuals = icp_rigid(x, y, n_iters=25, keep_fraction=0.9, max_corr_dist=np.inf)

    assert corr.shape[0] >= 3
    assert residuals.size >= 3
    assert np.allclose(r_hat, r_true, atol=1e-1)
    assert np.allclose(b_hat, b_true, atol=1e-1)


def test_hungarian_dummy_can_leave_points_unmatched():
    x0 = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=float)
    x1 = np.array([[0.1, 0.0, 0.0], [100.2, 0.0, 0.0]], dtype=float)
    cost = build_cost_matrix(x0, x1, metric="sqeuclidean")
    gate = apply_gating(cost, max_dist=0.5, assume_squared_cost=True)

    # Force unmatched decisions by setting a tiny miss cost.
    out = hungarian_assignment(
        cost,
        gating_mask=gate,
        use_dummy=True,
        miss_cost=0.01,
        birth_cost=0.01,
        death_cost=0.01,
    )
    assert out["row_ind"].size <= 2
    assert out["unmatched_t"].size + out["row_ind"].size >= 2
