import numpy as np

from driftcorr.drift_smoothing import detect_stage_jumps, smooth_drift_timeseries


def test_smoothing_reduces_high_frequency_component_and_keeps_trend():
    rng = np.random.default_rng(0)
    times = np.arange(120, dtype=float)
    trend = np.stack(
        [
            0.05 * times,
            -0.03 * times + 1.5,
            0.02 * times - 0.8,
        ],
        axis=1,
    )
    hi = 0.6 * np.sin(2.0 * np.pi * times / 3.0)[:, None]
    noise = 0.1 * rng.normal(size=(times.size, 3))
    drift = trend + hi + noise

    sm = smooth_drift_timeseries(
        times,
        drift,
        method="savgol",
        params={"window_length": 11, "polyorder": 2},
    )

    hf_raw = np.std(np.diff(drift - trend, axis=0), axis=0)
    hf_sm = np.std(np.diff(sm - trend, axis=0), axis=0)
    assert float(np.nanmean(hf_sm)) < float(np.nanmean(hf_raw))

    mae_raw = float(np.nanmean(np.abs(drift - trend)))
    mae_sm = float(np.nanmean(np.abs(sm - trend)))
    assert mae_sm < mae_raw


def test_jump_detection_and_piecewise_smoothing_preserves_jump():
    rng = np.random.default_rng(1)
    times = np.arange(80, dtype=float)
    drift = np.zeros((times.size, 3), dtype=float)
    drift[:40, :] = np.array([0.2, -0.1, 0.05], dtype=float)
    drift[40:, :] = np.array([4.0, -3.5, 2.2], dtype=float)
    drift += 0.08 * rng.normal(size=drift.shape)

    jumps = detect_stage_jumps(times, drift, params={"jump_thresh_mad": 5.0})
    jump_idx = np.where(jumps)[0]
    assert jump_idx.size >= 1
    assert np.any((jump_idx >= 38) & (jump_idx <= 42))

    left = smooth_drift_timeseries(times[:40], drift[:40], method="moving_median", params={"window": 7})
    right = smooth_drift_timeseries(times[40:], drift[40:], method="moving_median", params={"window": 7})
    sm = np.vstack([left, right])

    left_mean = np.nanmean(sm[:40], axis=0)
    right_mean = np.nanmean(sm[40:], axis=0)
    assert float(np.linalg.norm(right_mean - left_mean)) > 2.0

