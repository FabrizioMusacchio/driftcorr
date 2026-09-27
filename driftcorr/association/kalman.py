"""Kalman-based CV association with innovation gating.

This module implements association for unlabeled 3D point clouds using a
constant-velocity Kalman filter per track. Candidate links are scored by the
Mahalanobis innovation distance and solved with one-to-one assignment.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .assignment import apply_gating, hungarian_assignment
from .scoring import global_association_metrics, score_links_df
from .types import AssociationResult, PointCloudFrame

# %% FUNCTIONS
@dataclass(slots=True)
class _TrackState:
    track_id: int
    x: np.ndarray  # (6,)
    p: np.ndarray  # (6,6)
    last_point_index: int
    last_time: float
    missed: int = 0


def _transition_mats(dt: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.eye(6, dtype=float)
    for i in range(3):
        f[i, i + 3] = float(dt)
    h = np.zeros((3, 6), dtype=float)
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    h[2, 2] = 1.0
    return f, h


def _process_noise(dt: float, q_scale: float) -> np.ndarray:
    q = max(float(q_scale), 1e-12)
    qmat = np.eye(6, dtype=float) * q
    for i in range(3):
        qmat[i, i] = q * max(dt * dt, 1e-9)
        qmat[i + 3, i + 3] = q
    return qmat


def _measurement_noise(r_scale: float) -> np.ndarray:
    r = max(float(r_scale), 1e-12)
    return np.eye(3, dtype=float) * r


def associate_kalman_cv(
    frames: list[PointCloudFrame],
    gating: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Associate point clouds via constant-velocity Kalman filtering.

    Parameters
    ----------
    frames:
        Sequence of frames with xyz arrays.
    gating:
        Optional gating settings (`max_dist`, `keep_percentile`).
    params:
        Parameters controlling filter/assignment:
        - dt (default 1.0)
        - q_scale, r_scale
        - gate_chi2 (default 16.0)
        - max_missed (default 0)
        - use_dummy_unmatched (default True)
        - miss_cost, birth_cost, death_cost
    """
    fr = sorted(frames, key=lambda f: float(f.time))
    g = dict(gating or {})
    p = dict(params or {})

    if len(fr) == 0:
        return AssociationResult(
            tracks_df=pd.DataFrame(),
            links_df=pd.DataFrame(),
            frame_stats_df=pd.DataFrame(),
            quality={},
            params={"method": "kalman_cv"},
        )

    dt_default = float(p.get("dt", 1.0))
    q_scale = float(p.get("q_scale", 1.0))
    r_scale = float(p.get("r_scale", 1.0))
    gate_chi2 = float(p.get("gate_chi2", 16.0))
    max_missed = int(p.get("max_missed", 0))
    use_dummy = bool(p.get("use_dummy_unmatched", True))

    point_track_ids: list[np.ndarray] = [np.full((f.xyz.shape[0],), -1, dtype=int) for f in fr]
    links_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    next_tid = 0
    active_states: dict[int, _TrackState] = {}

    # Initialize from first frame.
    x0 = np.asarray(fr[0].xyz, dtype=float)
    for i in range(x0.shape[0]):
        tid = next_tid
        next_tid += 1
        xx = np.zeros((6,), dtype=float)
        xx[:3] = x0[i]
        pp = np.eye(6, dtype=float)
        st = _TrackState(track_id=tid, x=xx, p=pp, last_point_index=int(i), last_time=float(fr[0].time), missed=0)
        active_states[tid] = st
        point_track_ids[0][i] = tid

    for t in range(len(fr) - 1):
        t0 = float(fr[t].time)
        t1 = float(fr[t + 1].time)
        dt = float(t1 - t0) if np.isfinite(t1 - t0) and (t1 - t0) > 0 else dt_default
        fmat, hmat = _transition_mats(dt)
        qmat = _process_noise(dt, q_scale)
        rmat = _measurement_noise(r_scale)

        x1 = np.asarray(fr[t + 1].xyz, dtype=float)
        n_dst = int(x1.shape[0])

        # Candidate source detections are points observed at frame t.
        src_indices = np.flatnonzero(point_track_ids[t] >= 0).astype(int)
        src_tids = point_track_ids[t][src_indices].astype(int)
        n_src = int(src_indices.size)

        if n_src == 0 or n_dst == 0:
            # Only births or deaths.
            for tid in src_tids.tolist():
                st = active_states.get(int(tid))
                if st is not None:
                    st.missed += 1
            for j in range(n_dst):
                tid = next_tid
                next_tid += 1
                xx = np.zeros((6,), dtype=float)
                xx[:3] = x1[j]
                pp = np.eye(6, dtype=float)
                active_states[tid] = _TrackState(tid, xx, pp, int(j), t1, missed=0)
                point_track_ids[t + 1][j] = tid
            stats_rows.append(
                {
                    "t": t0,
                    "t_next": t1,
                    "n_points_t": int(fr[t].xyz.shape[0]),
                    "n_points_t_next": int(fr[t + 1].xyz.shape[0]),
                    "n_matched": 0,
                    "n_unmatched_t": int(n_src),
                    "n_unmatched_t_next": int(n_dst),
                }
            )
            continue

        # Predict all source track states to t+1.
        x_pred = np.zeros((n_src, 3), dtype=float)
        p_pred_list: list[np.ndarray] = []
        for k, tid in enumerate(src_tids.tolist()):
            st = active_states[int(tid)]
            xk = fmat @ st.x
            pk = fmat @ st.p @ fmat.T + qmat
            x_pred[k] = xk[:3]
            p_pred_list.append(pk)
            st.x = xk
            st.p = pk

        # Build innovation distance matrix.
        cost = np.full((n_src, n_dst), np.inf, dtype=float)
        for i in range(n_src):
            s = hmat @ p_pred_list[i] @ hmat.T + rmat
            s_inv = np.linalg.pinv(s)
            innov = x1[None, :, :] - x_pred[i][None, :]
            cost[i, :] = np.einsum("...i,ij,...j->...", innov, s_inv, innov)

        gate = apply_gating(
            cost,
            max_dist=g.get("max_dist"),
            max_cost=float(gate_chi2),
            keep_percentile=g.get("keep_percentile"),
            assume_squared_cost=False,
        )
        assn = hungarian_assignment(
            cost,
            gating_mask=gate,
            use_dummy=use_dummy,
            miss_cost=p.get("miss_cost", gate_chi2),
            birth_cost=p.get("birth_cost", p.get("miss_cost", gate_chi2)),
            death_cost=p.get("death_cost", p.get("miss_cost", gate_chi2)),
        )

        matched_src_local = assn["row_ind"].astype(int)
        matched_dst = assn["col_ind"].astype(int)

        used_src_local = np.zeros((n_src,), dtype=bool)
        used_dst = np.zeros((n_dst,), dtype=bool)

        # Matched Kalman updates.
        for idx in range(matched_src_local.size):
            i_local = int(matched_src_local[idx])
            j = int(matched_dst[idx])
            tid = int(src_tids[i_local])
            z = x1[j]
            st = active_states[tid]
            s = hmat @ st.p @ hmat.T + rmat
            k_gain = st.p @ hmat.T @ np.linalg.pinv(s)
            y = z - (hmat @ st.x)
            st.x = st.x + k_gain @ y
            st.p = (np.eye(6, dtype=float) - k_gain @ hmat) @ st.p
            st.last_point_index = int(j)
            st.last_time = t1
            st.missed = 0

            src_idx = int(src_indices[i_local])
            point_track_ids[t + 1][j] = tid
            links_rows.append(
                {
                    "t": t0,
                    "src_index": src_idx,
                    "dst_index": j,
                    "track_id": tid,
                    "cost": float(cost[i_local, j]),
                    "gated": bool(gate[i_local, j]),
                    "matched": True,
                }
            )
            used_src_local[i_local] = True
            used_dst[j] = True

        # Unmatched source tracks.
        for i_local in np.flatnonzero(~used_src_local).tolist():
            tid = int(src_tids[int(i_local)])
            st = active_states.get(tid)
            if st is not None:
                st.missed += 1

        # New births for unmatched detections.
        for j in np.flatnonzero(~used_dst).tolist():
            tid = next_tid
            next_tid += 1
            xx = np.zeros((6,), dtype=float)
            xx[:3] = x1[int(j)]
            pp = np.eye(6, dtype=float)
            active_states[tid] = _TrackState(tid, xx, pp, int(j), t1, missed=0)
            point_track_ids[t + 1][int(j)] = tid

        # Prune tracks missed too long.
        if max_missed >= 0:
            drop = [tid for tid, st in active_states.items() if int(st.missed) > int(max_missed)]
            for tid in drop:
                active_states.pop(int(tid), None)

        stats_rows.append(
            {
                "t": t0,
                "t_next": t1,
                "n_points_t": int(fr[t].xyz.shape[0]),
                "n_points_t_next": int(fr[t + 1].xyz.shape[0]),
                "n_matched": int(matched_src_local.size),
                "n_unmatched_t": int(np.sum(~used_src_local)),
                "n_unmatched_t_next": int(np.sum(~used_dst)),
            }
        )

    # Convert track labels to table.
    track_rows: list[dict[str, float]] = []
    for t, f in enumerate(fr):
        xyz = np.asarray(f.xyz, dtype=float)
        tids = point_track_ids[t]
        for i in range(xyz.shape[0]):
            tid = int(tids[i])
            if tid < 0:
                continue
            track_rows.append(
                {
                    "TrackID": float(tid),
                    "Time": float(f.time),
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
    if not tracks_df.empty:
        tracks_df = tracks_df.sort_values(["TrackID", "Time", "PointIndex"]).reset_index(drop=True)
    links_df = pd.DataFrame(links_rows)
    if not links_df.empty:
        links_df = score_links_df(links_df.sort_values(["t", "src_index", "dst_index"]).reset_index(drop=True))
    frame_stats_df = pd.DataFrame(stats_rows)

    quality = global_association_metrics(tracks_df, links_df, frame_stats_df)
    quality["n_tracks"] = float(tracks_df["TrackID"].nunique()) if not tracks_df.empty else 0.0
    quality["mean_track_length"] = float(tracks_df.groupby("TrackID").size().mean()) if not tracks_df.empty else 0.0
    quality["total_cost"] = float(np.nansum(pd.to_numeric(links_df.get("cost", np.nan), errors="coerce"))) if not links_df.empty else 0.0

    return AssociationResult(
        tracks_df=tracks_df,
        links_df=links_df,
        frame_stats_df=frame_stats_df,
        quality=quality,
        params={"method": "kalman_cv_innovation_gating", **p, **{f"gating_{k}": v for k, v in g.items()}},
    )
# %% END