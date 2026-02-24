"""Link frame-to-frame assignments into pseudo tracks.

Given assignments for each transition t -> t+1, this module constructs a
consistent `TrackID` labeling over all observations.

Birth/death behavior:
- unmatched destination points create new tracks,
- unmatched source points terminate tracks by default,
- `max_gap` is accepted for API compatibility; current prototype uses direct
  consecutive linking and therefore supports only `max_gap=0` semantics.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import PointCloudFrame


def link_assignments_to_tracks(
    frames: list[PointCloudFrame],
    assignments_by_t: list[dict[str, Any]],
    allow_birth_death: bool = True,
    max_gap: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert pairwise assignment outputs into pseudo tracks.

    Parameters
    ----------
    frames:
        Ordered point-cloud frames.
    assignments_by_t:
        List with length `len(frames)-1`, each element returned by assignment
        routines (`row_ind`, `col_ind`, `costs`, unmatched arrays).

    Returns
    -------
    tracks_df, links_df, frame_stats_df
    """
    if max_gap > 0:
        # Explicitly accepted but not fully modeled in this prototype.
        max_gap = 0

    if not frames:
        empty = pd.DataFrame()
        return empty, empty, empty

    n_frames = len(frames)
    if len(assignments_by_t) != max(0, n_frames - 1):
        raise ValueError("assignments_by_t length must be len(frames)-1")

    point_track_ids: list[np.ndarray] = [np.full((f.xyz.shape[0],), -1, dtype=int) for f in frames]
    next_track_id = 0

    # Initialize first frame with one track per point.
    n0 = frames[0].xyz.shape[0]
    point_track_ids[0] = np.arange(next_track_id, next_track_id + n0, dtype=int)
    next_track_id += n0

    links_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for t in range(n_frames - 1):
        a = assignments_by_t[t]
        n_src = frames[t].xyz.shape[0]
        n_dst = frames[t + 1].xyz.shape[0]

        row_ind = np.asarray(a.get("row_ind", []), dtype=int)
        col_ind = np.asarray(a.get("col_ind", []), dtype=int)
        costs = np.asarray(a.get("costs", np.full((row_ind.size,), np.nan)), dtype=float)
        unmatched_src = np.asarray(a.get("unmatched_t", []), dtype=int)
        unmatched_dst = np.asarray(a.get("unmatched_tp1", []), dtype=int)

        # Propagate track IDs along matches.
        for k in range(row_ind.size):
            i = int(row_ind[k])
            j = int(col_ind[k])
            tid = int(point_track_ids[t][i])
            if tid < 0 and bool(allow_birth_death):
                tid = int(next_track_id)
                next_track_id += 1
                point_track_ids[t][i] = tid
            point_track_ids[t + 1][j] = tid
            links_rows.append(
                {
                    "t": float(frames[t].time),
                    "src_index": i,
                    "dst_index": j,
                    "track_id": tid,
                    "cost": float(costs[k]) if k < costs.size else np.nan,
                    "gated": True,
                    "matched": True,
                }
            )

        # Births for unmatched destination points.
        for j in unmatched_dst.tolist():
            if point_track_ids[t + 1][int(j)] >= 0:
                continue
            if bool(allow_birth_death):
                point_track_ids[t + 1][int(j)] = int(next_track_id)
                next_track_id += 1

        stats_rows.append(
            {
                "t": float(frames[t].time),
                "t_next": float(frames[t + 1].time),
                "n_points_t": int(n_src),
                "n_points_t_next": int(n_dst),
                "n_matched": int(row_ind.size),
                "n_unmatched_t": int(unmatched_src.size),
                "n_unmatched_t_next": int(unmatched_dst.size),
            }
        )

    # Emit track rows.
    track_rows: list[dict[str, Any]] = []
    for t, frame in enumerate(frames):
        xyz = np.asarray(frame.xyz, dtype=float)
        tids = point_track_ids[t]
        for i in range(xyz.shape[0]):
            tid = int(tids[i])
            if tid < 0 and bool(allow_birth_death):
                tid = int(next_track_id)
                next_track_id += 1
            track_rows.append(
                {
                    "TrackID": float(tid),
                    "Time": float(frame.time),
                    "PointIndex": int(i),
                    "x": float(xyz[i, 0]),
                    "y": float(xyz[i, 1]),
                    "z": float(xyz[i, 2]),
                    "Position X": float(xyz[i, 0]),
                    "Position Y": float(xyz[i, 1]),
                    "Position Z": float(xyz[i, 2]),
                }
            )

    tracks_df = pd.DataFrame(track_rows)
    links_df = pd.DataFrame(links_rows)
    frame_stats_df = pd.DataFrame(stats_rows)
    if not tracks_df.empty:
        tracks_df = tracks_df.sort_values(["TrackID", "Time", "PointIndex"]).reset_index(drop=True)
    if not links_df.empty:
        links_df = links_df.sort_values(["t", "src_index", "dst_index"]).reset_index(drop=True)
    return tracks_df, links_df, frame_stats_df
