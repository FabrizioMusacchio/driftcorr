import numpy as np

from driftcorr.association.hungarian_cv_variants import (
    associate_hungarian_adaptive_maxdist_cv,
    associate_hungarian_innovation_percentile_cv,
    associate_hungarian_switch_penalty_cv,
    associate_hungarian_two_stage_cv,
)
from driftcorr.association.types import PointCloudFrame


def _make_frames() -> list[PointCloudFrame]:
    frames: list[PointCloudFrame] = []
    for t in range(4):
        xyz = np.array(
            [
                [1.0 + 0.6 * t, 0.0, 0.0],
                [10.0 + 0.5 * t, 1.0, 0.5],
                [20.0 + 0.55 * t, -1.0, 0.2],
            ],
            dtype=float,
        )
        frames.append(PointCloudFrame(time=float(t + 1), xyz=xyz))
    return frames


def _assert_basic_result(res) -> None:
    assert hasattr(res, "tracks_df")
    assert hasattr(res, "links_df")
    assert hasattr(res, "frame_stats_df")
    assert not res.tracks_df.empty
    assert {"TrackID", "Time", "PointIndex"}.issubset(set(res.tracks_df.columns))


def test_hungarian_adaptive_maxdist_cv_smoke():
    frames = _make_frames()
    res = associate_hungarian_adaptive_maxdist_cv(
        frames,
        gating={"max_dist": 3.0, "keep_percentile": 99.0},
        params={"motion_refine_iters": 2, "dx_scale": 1.0, "max_dist_factor": 3.0},
    )
    _assert_basic_result(res)


def test_hungarian_innovation_percentile_cv_smoke():
    frames = _make_frames()
    res = associate_hungarian_innovation_percentile_cv(
        frames,
        gating={"max_dist": 3.0, "keep_percentile": 99.0},
        params={"motion_refine_iters": 2, "dx_scale": 1.0, "max_dist_factor": 3.0},
    )
    _assert_basic_result(res)


def test_hungarian_switch_penalty_cv_smoke():
    frames = _make_frames()
    res = associate_hungarian_switch_penalty_cv(
        frames,
        gating={"max_dist": 3.0, "keep_percentile": 99.0},
        params={"motion_refine_iters": 2, "dx_scale": 1.0, "max_dist_factor": 3.0, "lambda_switch": 0.2},
    )
    _assert_basic_result(res)


def test_hungarian_two_stage_cv_smoke():
    frames = _make_frames()
    res = associate_hungarian_two_stage_cv(
        frames,
        gating={"max_dist": 3.0, "keep_percentile": 99.0},
        params={
            "motion_refine_iters": 2,
            "dx_scale": 1.0,
            "max_dist_factor": 3.0,
            "stage1_factor": 0.5,
            "stage1_keep_percentile": 90.0,
            "stage2_keep_percentile": 99.0,
        },
    )
    _assert_basic_result(res)
