"""Optional temporal smoothing utilities for estimated drift trajectories.

This module provides a lightweight, method-agnostic smoothing layer for
time-indexed drift vectors ``d(t) = [dx(t), dy(t), dz(t)]``.

Intended use:
- Compute a raw drift time series from any correction method.
- Optionally detect abrupt stage jumps.
- Smooth within segments to suppress high-frequency noise while preserving
  slow trends and optional discontinuities.

Available smoothing methods
---------------------------
- ``savgol``:
  Savitzky-Golay polynomial filter (SciPy).
  Parameters: ``window_length`` (odd), ``polyorder``.
- ``lowess``:
  Locally weighted regression (statsmodels, optional dependency).
  Parameters: ``frac``, ``it``.
- ``moving_median``:
  Centered rolling median (pandas).
  Parameter: ``window``.

Jump detection
--------------
``detect_stage_jumps`` identifies abrupt changes via robust thresholding on
``||d(t)-d(t-1)||`` using MAD scaling.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
except Exception:  # pragma: no cover
    _sm_lowess = None


def _ensure_xyz_array(drift_xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(drift_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("drift_xyz must have shape (n_times, 3).")
    return arr


def _resolve_odd_window(n: int, win: int) -> int:
    w = int(max(1, min(int(win), int(n))))
    if w % 2 == 0:
        w = max(1, w - 1)
    if w < 3 and n >= 3:
        w = 3
    if w > n:
        w = n if n % 2 == 1 else max(1, n - 1)
    return int(max(1, w))


def _smooth_1d(values: np.ndarray, times: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    x = np.asarray(times, dtype=float)
    n = y.size
    if n <= 2:
        return y.copy()
    if not np.all(np.isfinite(y)):
        # Linear interpolation for internal NaN handling.
        s = pd.Series(y)
        y = s.interpolate(limit_direction="both").to_numpy(float)
    m = str(method).lower()
    if m == "savgol":
        if savgol_filter is None:
            raise RuntimeError("savgol smoothing requested but scipy.signal.savgol_filter is unavailable.")
        w = _resolve_odd_window(n, int(params.get("window_length", params.get("window", 9))))
        p = int(params.get("polyorder", 2))
        p = int(max(0, min(p, max(0, w - 1))))
        return np.asarray(savgol_filter(y, window_length=w, polyorder=p, mode="interp"), dtype=float)
    if m == "lowess":
        if _sm_lowess is None:
            raise RuntimeError("lowess smoothing requested but statsmodels is unavailable.")
        frac = float(params.get("frac", 0.25))
        it = int(params.get("it", 1))
        ys = _sm_lowess(endog=y, exog=x, frac=frac, it=it, return_sorted=False)
        return np.asarray(ys, dtype=float)
    if m == "moving_median":
        w = _resolve_odd_window(n, int(params.get("window", 9)))
        s = pd.Series(y)
        return s.rolling(window=w, center=True, min_periods=1).median().to_numpy(float)
    if m in {"none", "identity"}:
        return y.copy()
    raise ValueError(f"Unsupported smoothing method: {method}")


def smooth_drift_timeseries(
    times: np.ndarray,
    drift_xyz: np.ndarray,
    method: str = "savgol",
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Smooth a drift time series ``d(t)`` axis-wise.

    Parameters
    ----------
    times:
        Array-like shape ``(n_times,)``.
    drift_xyz:
        Drift vectors shape ``(n_times, 3)``.
    method:
        One of ``{"savgol", "lowess", "moving_median", "none"}``.
    params:
        Method-specific parameter dictionary.

    Returns
    -------
    np.ndarray
        Smoothed drift vectors with shape ``(n_times, 3)``.
    """

    p = {} if params is None else dict(params)
    t = np.asarray(times, dtype=float).reshape(-1)
    d = _ensure_xyz_array(drift_xyz)
    if t.size != d.shape[0]:
        raise ValueError("times length must match drift_xyz.shape[0].")
    out = np.zeros_like(d, dtype=float)
    for ax in range(3):
        out[:, ax] = _smooth_1d(d[:, ax], t, method=method, params=p)
    return out


def detect_stage_jumps(
    times: np.ndarray,
    drift_xyz: np.ndarray,
    method: str = "mad_threshold",
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Detect abrupt stage jumps in drift trajectory.

    Returns a boolean mask with shape ``(n_times,)`` where ``True`` marks the
    first sample after a detected jump.
    """

    _ = np.asarray(times, dtype=float).reshape(-1)
    d = _ensure_xyz_array(drift_xyz)
    p = {} if params is None else dict(params)
    m = str(method).lower()
    if d.shape[0] <= 1:
        return np.zeros((d.shape[0],), dtype=bool)
    if m not in {"mad_threshold", "mad", "threshold"}:
        raise ValueError(f"Unsupported jump detection method: {method}")

    step_norm = np.linalg.norm(np.diff(d, axis=0), axis=1)
    med = float(np.nanmedian(step_norm))
    mad = float(np.nanmedian(np.abs(step_norm - med)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
        robust_sigma = float(np.nanstd(step_norm))
    if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
        robust_sigma = 1e-9

    jump_thresh_mad = float(p.get("jump_thresh_mad", p.get("thresh_mad", 6.0)))
    thresh = med + jump_thresh_mad * robust_sigma

    jump_idx = np.where(step_norm > thresh)[0] + 1
    mask = np.zeros((d.shape[0],), dtype=bool)
    mask[jump_idx] = True
    return mask


def plot_drift_timeseries(
    times: np.ndarray,
    drift_xyz: np.ndarray,
    drift_xyz_smooth: np.ndarray,
    out_prefix: str | Path,
    tag: str,
) -> Path:
    """Plot raw vs smoothed drift traces and save PNG.

    Returns the saved figure path.
    """

    t = np.asarray(times, dtype=float).reshape(-1)
    raw = _ensure_xyz_array(drift_xyz)
    sm = _ensure_xyz_array(drift_xyz_smooth)
    if raw.shape != sm.shape or raw.shape[0] != t.size:
        raise ValueError("times/raw/smooth dimensions do not match.")

    out_prefix = Path(out_prefix)
    out_path = Path(str(out_prefix) + f"_{tag}_drift_timeseries.png")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ["dx", "dy", "dz"]
    for ax, label, col in zip(axes, labels, range(3)):
        ax.plot(t, raw[:, col], label=f"{label}_raw", alpha=0.8, linewidth=1.5)
        ax.plot(t, sm[:, col], label=f"{label}_smooth", alpha=0.9, linewidth=2.0)
        ax.set_ylabel(f"{label} (um)")
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Drift timeseries ({tag})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

