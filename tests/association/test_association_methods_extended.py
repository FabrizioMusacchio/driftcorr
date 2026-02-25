import numpy as np

from driftcorr.association.kalman import associate_kalman_cv
from driftcorr.association.min_cost_flow import associate_min_cost_flow
from driftcorr.association.tracklets import associate_tracklet_stitching
from driftcorr.association.types import PointCloudFrame


def _make_frames(n_times: int = 4) -> list[PointCloudFrame]:
    frames: list[PointCloudFrame] = []
    for t in range(n_times):
        xyz = np.array(
            [
                [0.5 * t, 0.0, 0.0],
                [10.0 + 0.5 * t, 2.0, 0.0],
                [20.0 + 0.5 * t, -2.0, 1.0],
            ],
            dtype=float,
        )
        frames.append(PointCloudFrame(time=float(t + 1), xyz=xyz))
    return frames


def test_associate_min_cost_flow_runs_and_returns_tables():
    frames = _make_frames()
    res = associate_min_cost_flow(
        frames,
        gating={"max_dist": 3.0, "keep_percentile": 99.0},
        motion_model="constant_position",
        params={"metric": "sqeuclidean", "birth_cost": 4.0, "death_cost": 4.0},
    )
    assert not res.tracks_df.empty
    assert "TrackID" in res.tracks_df.columns
    assert "Time" in res.tracks_df.columns


def test_associate_tracklet_stitching_runs_and_returns_tables():
    frames = _make_frames()
    res = associate_tracklet_stitching(
        frames,
        gating_strict={"max_dist": 2.0, "keep_percentile": 95.0},
        gating_loose={"max_dist": 4.0, "keep_percentile": 99.0},
        motion_model="constant_velocity",
        params={"metric": "sqeuclidean", "strict_factor": 1.0, "loose_factor": 1.0, "use_dummy": True},
    )
    assert not res.tracks_df.empty
    assert "TrackID" in res.tracks_df.columns
    assert "Time" in res.tracks_df.columns


def test_associate_kalman_cv_runs_and_returns_tables():
    frames = _make_frames()
    res = associate_kalman_cv(
        frames,
        gating={"max_dist": 4.0, "keep_percentile": 99.0},
        params={"q_scale": 0.5, "r_scale": 0.5, "gate_chi2": 16.0, "use_dummy_unmatched": True},
    )
    assert not res.tracks_df.empty
    assert "TrackID" in res.tracks_df.columns
    assert "Time" in res.tracks_df.columns
