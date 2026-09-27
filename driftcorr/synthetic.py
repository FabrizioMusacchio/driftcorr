"""Compact synthetic track generator used mainly for test smoke checks.

This module creates lightweight IMARIS-like trajectories with mixed motion:
- baseline track positions,
- simple global and local drift patterns,
- optional migration subset.

It is intentionally simple and fast compared to `synthetic_experiments.py`,
which is the full controlled benchmark generator.

Model sketch:
- `x_obs = x0 + d_global(t) + d_local(track,t) + m(track,t) + noise`.

Controlling parameters:
- `n_tracks`, `n_times`: data size.
- `migration_fraction`: proportion of non-stationary tracks.
- `seed`: reproducibility.

Effects:
- Increasing `migration_fraction` reduces dominance of common drift components.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

import numpy as np
import pandas as pd
# %% FUNCTIONS
def make_synthetic_tracks(
    n_tracks: int = 40,
    n_times: int = 20,
    seed: int = 0,
    migration_fraction: float = 0.2,
) -> pd.DataFrame:
    """Generate synthetic IMARIS-like tracks with drift + optional migration."""
    rng = np.random.default_rng(seed)

    track_ids = np.arange(1000, 1000 + n_tracks, dtype=float)
    times = np.arange(1, n_times + 1, dtype=float)

    base_pos = rng.uniform([100, 100, 30], [600, 600, 300], size=(n_tracks, 3))

    drift_global = np.stack(
        [
            2.0 * np.sin(times / 6.0),
            1.8 * np.cos(times / 7.0),
            0.9 * np.sin(times / 8.0),
        ],
        axis=1,
    )

    spatial_coef = rng.normal(scale=0.08, size=(n_tracks, 3))
    local_drift = np.array(
        [
            spatial_coef * np.sin(t / 10.0) + 0.5 * spatial_coef * np.cos(t / 13.0)
            for t in times
        ]
    )
    local_drift = np.transpose(local_drift, (1, 0, 2))

    migrating = rng.random(n_tracks) < migration_fraction
    mig_vel = rng.normal(loc=0.4, scale=0.15, size=(n_tracks, 3))
    mig_vel[~migrating] = 0.0

    rows = []
    for i, tid in enumerate(track_ids):
        for ti, t in enumerate(times):
            bio = mig_vel[i] * ti
            pos = base_pos[i] + drift_global[ti] + local_drift[i, ti] + bio + rng.normal(scale=0.15, size=3)
            rows.append(
                {
                    "TrackID": float(tid),
                    "Time": float(t),
                    "Position X": float(pos[0]),
                    "Position Y": float(pos[1]),
                    "Position Z": float(pos[2]),
                }
            )

    return pd.DataFrame(rows)
# %% END