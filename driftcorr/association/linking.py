"""Link frame-to-frame assignments into pseudo tracks.

Given assignments for each transition t -> t+1, this module constructs a
consistent `TrackID` labeling over all observations.

Birth/death behavior:
- unmatched destination points create new tracks,
- unmatched source points terminate tracks by default,
- optional post-hoc gap stitching can merge short track breaks when
  `max_gap > 0` without changing the underlying pairwise assignment core.

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


def _track_xyz_columns(df: pd.DataFrame) -> list[str]:
    """Return xyz column names present in a track table."""
    if {"x", "y", "z"}.issubset(df.columns):
        return ["x", "y", "z"]
    return ["Position X", "Position Y", "Position Z"]


def _estimate_global_step_scale(tracks_df: pd.DataFrame) -> float:
    """Estimate a robust per-frame displacement scale from existing tracklets."""
    if tracks_df.empty:
        return 1.0
    cols = _track_xyz_columns(tracks_df)
    d = tracks_df.copy()
    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    step_scales: list[float] = []
    for _, grp in d.groupby("TrackID", sort=True):
        g = grp.sort_values("Time")
        xyz = g[cols].to_numpy(float)
        tt = g["Time"].to_numpy(float)
        if xyz.shape[0] < 2:
            continue
        diffs = np.diff(xyz, axis=0)
        dt = np.diff(tt)
        ok = np.isfinite(dt) & (dt > 0.0)
        if not np.any(ok):
            continue
        step_norm = np.linalg.norm(diffs[ok], axis=1) / np.maximum(dt[ok], 1e-9)
        step_scales.extend(step_norm[np.isfinite(step_norm)].tolist())
    if not step_scales:
        return 1.0
    return max(float(np.median(np.asarray(step_scales, dtype=float))), 1.0)


def _summarize_tracklets(tracks_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize each current tracklet by endpoints and end velocity."""
    if tracks_df.empty:
        return []
    cols = _track_xyz_columns(tracks_df)
    d = tracks_df.copy()
    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    summaries: list[dict[str, Any]] = []
    for track_id, grp in d.groupby("TrackID", sort=True):
        g = grp.sort_values("Time")
        xyz = g[cols].to_numpy(float)
        tt = g["Time"].to_numpy(float)
        if xyz.shape[0] >= 2:
            dt = float(tt[-1] - tt[-2])
            if np.isfinite(dt) and dt > 0.0:
                v_end = (xyz[-1] - xyz[-2]) / dt
            else:
                v_end = np.zeros(3, dtype=float)
        else:
            v_end = np.zeros(3, dtype=float)
        summaries.append(
            {
                "track_id": int(float(track_id)),
                "t_start": float(tt[0]),
                "t_end": float(tt[-1]),
                "p_start": np.asarray(xyz[0], dtype=float),
                "p_end": np.asarray(xyz[-1], dtype=float),
                "v_end": np.asarray(v_end, dtype=float),
            }
        )
    return summaries


def _build_track_id_stitch_map(tracks_df: pd.DataFrame, max_gap: int) -> dict[int, int]:
    """Build a TrackID remapping that stitches short track gaps post hoc."""
    summaries = _summarize_tracklets(tracks_df)
    if len(summaries) <= 1 or int(max_gap) <= 0:
        return {int(s["track_id"]): int(s["track_id"]) for s in summaries}

    step_scale = _estimate_global_step_scale(tracks_df)
    gate_base = max(3.0 * float(step_scale), 5.0)
    candidates: list[tuple[float, int, int]] = []
    for pred in summaries:
        for succ in summaries:
            pred_tid = int(pred["track_id"])
            succ_tid = int(succ["track_id"])
            if pred_tid == succ_tid:
                continue
            dt = float(succ["t_start"] - pred["t_end"])
            if dt <= 0.0:
                continue
            if dt > float(int(max_gap) + 1):
                continue
            pred_pos = np.asarray(pred["p_end"], dtype=float) + float(dt) * np.asarray(pred["v_end"], dtype=float)
            res = np.asarray(pred_pos - np.asarray(succ["p_start"], dtype=float), dtype=float)
            dist = float(np.linalg.norm(res))
            gate = float(gate_base) * float(dt)
            if dist > gate:
                continue
            gap_penalty = 0.35 * float(gate_base) * max(float(dt) - 1.0, 0.0)
            candidates.append((float(dist + gap_penalty), pred_tid, succ_tid))

    if not candidates:
        return {int(s["track_id"]): int(s["track_id"]) for s in summaries}

    pred_used: set[int] = set()
    succ_used: set[int] = set()
    succ_map: dict[int, int] = {}
    pred_map: dict[int, int] = {}
    for cost, pred_tid, succ_tid in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
        if pred_tid in pred_used or succ_tid in succ_used:
            continue
        pred_used.add(pred_tid)
        succ_used.add(succ_tid)
        succ_map[pred_tid] = succ_tid
        pred_map[succ_tid] = pred_tid

    tids = sorted(int(s["track_id"]) for s in summaries)
    final_map: dict[int, int] = {}
    next_tid = 0
    starts = [tid for tid in tids if tid not in pred_map]
    for start_tid in starts:
        cur = int(start_tid)
        while cur not in final_map:
            final_map[cur] = int(next_tid)
            if cur not in succ_map:
                break
            cur = int(succ_map[cur])
        next_tid += 1
    for tid in tids:
        if tid not in final_map:
            final_map[tid] = int(next_tid)
            next_tid += 1
    return final_map


def _remap_track_ids_after_gap_stitch(
    tracks_df: pd.DataFrame,
    links_df: pd.DataFrame,
    max_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally stitch short track breaks and remap TrackIDs in outputs."""
    if tracks_df.empty or int(max_gap) <= 0:
        return tracks_df, links_df
    final_map = _build_track_id_stitch_map(tracks_df, max_gap=int(max_gap))
    if not final_map:
        return tracks_df, links_df

    out_tracks = tracks_df.copy()
    out_tracks["TrackID"] = pd.to_numeric(out_tracks["TrackID"], errors="coerce").map(
        lambda v: float(final_map.get(int(v), int(v))) if np.isfinite(v) else np.nan
    )
    out_tracks = out_tracks.sort_values(["TrackID", "Time", "PointIndex"]).reset_index(drop=True)

    out_links = links_df.copy()
    if not out_links.empty and "track_id" in out_links.columns:
        out_links["track_id"] = pd.to_numeric(out_links["track_id"], errors="coerce").map(
            lambda v: float(final_map.get(int(v), int(v))) if np.isfinite(v) else np.nan
        )
        out_links = out_links.sort_values(["t", "src_index", "dst_index"]).reset_index(drop=True)
    return out_tracks, out_links


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
    tracks_df, links_df = _remap_track_ids_after_gap_stitch(tracks_df, links_df, max_gap=int(max_gap))
    return tracks_df, links_df, frame_stats_df
