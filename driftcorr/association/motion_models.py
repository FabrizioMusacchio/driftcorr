"""Lightweight motion models for association prediction.

These models provide optional stateful predictions to improve matching when
per-frame motion is non-negligible.

Implemented models:
- ConstantPositionModel: x_hat(t+1) = x(t)
- ConstantVelocityModel: x_hat(t+1) = x(t) + v(t)*dt

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
# %% CLASSES

class ConstantPositionModel:
    """State-free motion model with zero velocity assumption."""

    name = "constant_position"

    def predict(self, positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray | None]:
        p = np.asarray(positions, dtype=float)
        return p.copy(), None

    def update(self, *_: Any, **__: Any) -> None:
        return None

@dataclass(slots=True)
class ConstantVelocityModel:
    """Per-track constant velocity predictor.

    Velocity is estimated from consecutive linked positions and smoothed with
    an exponential factor `alpha`.
    """

    alpha: float = 0.7
    _vel_by_track: dict[int, np.ndarray] = field(default_factory=dict)
    _last_pos_by_track: dict[int, np.ndarray] = field(default_factory=dict)
    _last_time_by_track: dict[int, float] = field(default_factory=dict)

    name: str = "constant_velocity"

    def predict_for_track_ids(self, track_ids: np.ndarray, positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray | None]:
        tids = np.asarray(track_ids, dtype=int)
        p = np.asarray(positions, dtype=float)
        out = p.copy()
        for i, tid in enumerate(tids):
            v = self._vel_by_track.get(int(tid))
            if v is not None:
                out[i] = p[i] + float(dt) * v
        return out, None

    def predict(self, positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray | None]:
        p = np.asarray(positions, dtype=float)
        return p.copy(), None

    def update_from_links(
        self,
        prev_track_ids: np.ndarray,
        prev_positions: np.ndarray,
        next_track_ids: np.ndarray,
        next_positions: np.ndarray,
        t_prev: float,
        t_next: float,
    ) -> None:
        """Update internal velocity estimates from matched links."""
        dt = float(t_next) - float(t_prev)
        if not np.isfinite(dt) or dt <= 0.0:
            return
        prev_tids = np.asarray(prev_track_ids, dtype=int)
        prev_pos = np.asarray(prev_positions, dtype=float)
        next_tids = np.asarray(next_track_ids, dtype=int)
        next_pos = np.asarray(next_positions, dtype=float)

        # Match by track id where available in both lists.
        next_lookup = {int(tid): i for i, tid in enumerate(next_tids.tolist())}
        for i, tid in enumerate(prev_tids.tolist()):
            j = next_lookup.get(int(tid))
            if j is None:
                continue
            v_new = (next_pos[j] - prev_pos[i]) / dt
            v_old = self._vel_by_track.get(int(tid))
            if v_old is None:
                v = v_new
            else:
                v = float(self.alpha) * v_old + (1.0 - float(self.alpha)) * v_new
            self._vel_by_track[int(tid)] = np.asarray(v, dtype=float)
            self._last_pos_by_track[int(tid)] = np.asarray(next_pos[j], dtype=float)
            self._last_time_by_track[int(tid)] = float(t_next)

class SimpleKalmanFilter3D:
    """Optional placeholder for a 3D Kalman model.

    This prototype intentionally does not introduce a full Kalman implementation
    to keep dependencies and complexity low. Use ConstantVelocityModel for
    stateful prediction in current benchmarks.
    """

    name = "kalman_3d"

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError("SimpleKalmanFilter3D is not implemented in this prototype.")

# %% END
