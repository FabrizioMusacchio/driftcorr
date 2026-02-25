"""Global multi-frame association via path-cover style optimization.

This module provides a global linker that considers candidate links across the
whole sequence (not only greedy frame-local decisions). The formulation is a
minimum-cost track path cover with birth/death penalties, implemented via a
maximum-weight bipartite matching equivalent.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .assignment import apply_gating, build_cost_matrix
from .scoring import global_association_metrics, score_links_df
from .types import AssociationResult, PointCloudFrame

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None


@dataclass(slots=True)
class _Det:
    gid: int
    t_idx: int
    time: float
    point_idx: int
    xyz: np.ndarray


@dataclass(slots=True)
class _Cand:
    src_gid: int
    dst_gid: int
    t: float
    t_next: float
    src_index: int
    dst_index: int
    cost: float
    benefit: float


def _flatten_detections(frames: list[PointCloudFrame]) -> list[_Det]:
    dets: list[_Det] = []
    gid = 0
    for ti, f in enumerate(sorted(frames, key=lambda v: float(v.time))):
        x = np.asarray(f.xyz, dtype=float)
        for j in range(x.shape[0]):
            dets.append(_Det(gid=gid, t_idx=int(ti), time=float(f.time), point_idx=int(j), xyz=x[j].copy()))
            gid += 1
    return dets


def _candidate_edges(
    frames: list[PointCloudFrame],
    dets_by_t: dict[int, list[_Det]],
    metric: str,
    gating: dict[str, Any],
    birth_cost: float,
    death_cost: float,
    allow_skip: bool,
    skip_cost_factor: float,
) -> list[_Cand]:
    fr = sorted(frames, key=lambda v: float(v.time))
    out: list[_Cand] = []
    for ti in range(len(fr) - 1):
        src_list = dets_by_t.get(int(ti), [])
        if not src_list:
            continue
        max_step = 2 if bool(allow_skip) else 1
        for step in range(1, max_step + 1):
            tj = ti + step
            if tj >= len(fr):
                continue
            dst_list = dets_by_t.get(int(tj), [])
            if not dst_list:
                continue
            xs = np.asarray([d.xyz for d in src_list], dtype=float)
            xd = np.asarray([d.xyz for d in dst_list], dtype=float)
            cm = build_cost_matrix(
                xs,
                xd,
                metric=metric,
                cov=gating.get("cov"),
                huber_delta=float(gating.get("huber_delta", 1.0)),
                mahalanobis_scales=gating.get("mahalanobis_scales"),
            )
            gate = apply_gating(
                cm,
                max_dist=gating.get("max_dist"),
                max_cost=gating.get("max_cost"),
                keep_percentile=gating.get("keep_percentile"),
                assume_squared_cost=(str(metric).lower() == "sqeuclidean"),
            )
            gap_penalty = 1.0 + (float(skip_cost_factor) * max(step - 1, 0))
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if not bool(gate[i, j]):
                        continue
                    c = float(cm[i, j]) * gap_penalty
                    b = float(birth_cost + death_cost - c)
                    out.append(
                        _Cand(
                            src_gid=int(src_list[i].gid),
                            dst_gid=int(dst_list[j].gid),
                            t=float(src_list[i].time),
                            t_next=float(dst_list[j].time),
                            src_index=int(src_list[i].point_idx),
                            dst_index=int(dst_list[j].point_idx),
                            cost=c,
                            benefit=b,
                        )
                    )
    return out


def _solve_global_matching(cands: list[_Cand]) -> list[_Cand]:
    if not cands:
        return []
    if nx is None:
        # Deterministic greedy fallback if networkx is unavailable.
        selected: list[_Cand] = []
        used_src: set[int] = set()
        used_dst: set[int] = set()
        for c in sorted(cands, key=lambda x: (-x.benefit, x.cost, x.src_gid, x.dst_gid)):
            if c.benefit <= 0.0:
                continue
            if c.src_gid in used_src or c.dst_gid in used_dst:
                continue
            used_src.add(c.src_gid)
            used_dst.add(c.dst_gid)
            selected.append(c)
        return selected

    g = nx.Graph()
    for c in cands:
        if c.benefit <= 0.0:
            continue
        ls = f"L_{c.src_gid}"
        rd = f"R_{c.dst_gid}"
        g.add_node(ls, bipartite=0)
        g.add_node(rd, bipartite=1)
        g.add_edge(ls, rd, weight=float(c.benefit), cand=c)

    if g.number_of_edges() == 0:
        return []

    match = nx.algorithms.matching.max_weight_matching(g, maxcardinality=False, weight="weight")
    out: list[_Cand] = []
    for a, b in match:
        ed = g.get_edge_data(a, b, default={})
        cand = ed.get("cand")
        if isinstance(cand, _Cand):
            out.append(cand)
    out.sort(key=lambda x: (x.t, x.src_index, x.dst_index))
    return out


def _build_tracks_from_global_links(frames: list[PointCloudFrame], selected: list[_Cand]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dets = _flatten_detections(frames)
    det_by_gid = {int(d.gid): d for d in dets}
    pred: dict[int, int] = {}
    succ: dict[int, int] = {}
    link_rows: list[dict[str, Any]] = []
    for c in selected:
        if int(c.src_gid) in succ or int(c.dst_gid) in pred:
            continue
        succ[int(c.src_gid)] = int(c.dst_gid)
        pred[int(c.dst_gid)] = int(c.src_gid)
        link_rows.append(
            {
                "t": float(c.t),
                "src_index": int(c.src_index),
                "dst_index": int(c.dst_index),
                "cost": float(c.cost),
                "gated": True,
                "matched": True,
            }
        )

    # Build track IDs by chaining DAG links.
    starts = [d.gid for d in dets if int(d.gid) not in pred]
    tid_map: dict[int, int] = {}
    next_tid = 0
    for st in sorted(starts):
        cur = int(st)
        while True:
            if cur in tid_map:
                break
            tid_map[cur] = next_tid
            if cur not in succ:
                break
            cur = int(succ[cur])
        next_tid += 1
    # Any isolated cycles should not happen in DAG, but guard for completeness.
    for d in dets:
        if int(d.gid) not in tid_map:
            tid_map[int(d.gid)] = next_tid
            next_tid += 1

    tracks_rows: list[dict[str, float]] = []
    for d in dets:
        tid = int(tid_map[int(d.gid)])
        tracks_rows.append(
            {
                "TrackID": float(tid),
                "Time": float(d.time),
                "PointIndex": int(d.point_idx),
                "x": float(d.xyz[0]),
                "y": float(d.xyz[1]),
                "z": float(d.xyz[2]),
                "Position X": float(d.xyz[0]),
                "Position Y": float(d.xyz[1]),
                "Position Z": float(d.xyz[2]),
            }
        )

    tracks_df = pd.DataFrame(tracks_rows)
    if not tracks_df.empty:
        tracks_df = tracks_df.sort_values(["TrackID", "Time", "PointIndex"]).reset_index(drop=True)

    links_df = pd.DataFrame(link_rows)
    if not links_df.empty:
        links_df = score_links_df(links_df.sort_values(["t", "src_index", "dst_index"]).reset_index(drop=True))

    # Transition stats for consecutive frame pairs.
    fr = sorted(frames, key=lambda v: float(v.time))
    stat_rows: list[dict[str, Any]] = []
    for ti in range(len(fr) - 1):
        t0 = float(fr[ti].time)
        t1 = float(fr[ti + 1].time)
        m = links_df[(pd.to_numeric(links_df.get("t", np.nan), errors="coerce") == t0)] if not links_df.empty else pd.DataFrame()
        if not m.empty:
            m = m[pd.to_numeric(m.get("t", np.nan), errors="coerce") == t0]
        n_matched = int(m.shape[0]) if not m.empty else 0
        n0 = int(fr[ti].xyz.shape[0])
        n1 = int(fr[ti + 1].xyz.shape[0])
        stat_rows.append(
            {
                "t": t0,
                "t_next": t1,
                "n_points_t": n0,
                "n_points_t_next": n1,
                "n_matched": n_matched,
                "n_unmatched_t": int(max(n0 - n_matched, 0)),
                "n_unmatched_t_next": int(max(n1 - n_matched, 0)),
            }
        )
    frame_stats_df = pd.DataFrame(stat_rows)
    return tracks_df, links_df, frame_stats_df


def associate_min_cost_flow(
    frames: list[PointCloudFrame],
    gating: dict[str, Any] | None = None,
    motion_model: str = "constant_position",
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Associate point clouds via a global path-cover optimization.

    Parameters
    ----------
    frames:
        List of point-cloud frames.
    gating:
        Gating configuration (`max_dist`, `keep_percentile`, ...).
    motion_model:
        `constant_position` or `constant_velocity` (currently both use geometric
        linking cost; CV can be approximated by tuning costs and skip edges).
    params:
        Optimization and penalties:
        - `metric`: sqeuclidean/euclidean/huber/mahalanobis_diag
        - `birth_cost`, `death_cost`
        - `allow_skip` (bool), `skip_cost_factor`
    """
    fr = sorted(frames, key=lambda f: float(f.time))
    if len(fr) == 0:
        return AssociationResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {"method": "min_cost_flow"})

    g = dict(gating or {})
    p = dict(params or {})
    metric = str(p.get("metric", "sqeuclidean"))
    birth_cost = float(p.get("birth_cost", p.get("miss_cost", 5.0)))
    death_cost = float(p.get("death_cost", p.get("miss_cost", 5.0)))
    allow_skip = bool(p.get("allow_skip", False))
    skip_cost_factor = float(p.get("skip_cost_factor", 2.0))

    dets = _flatten_detections(fr)
    dets_by_t: dict[int, list[_Det]] = {}
    for d in dets:
        dets_by_t.setdefault(int(d.t_idx), []).append(d)

    cands = _candidate_edges(
        fr,
        dets_by_t,
        metric=metric,
        gating=g,
        birth_cost=birth_cost,
        death_cost=death_cost,
        allow_skip=allow_skip,
        skip_cost_factor=skip_cost_factor,
    )
    sel = _solve_global_matching(cands)
    tracks_df, links_df, frame_stats_df = _build_tracks_from_global_links(fr, sel)

    quality = global_association_metrics(tracks_df, links_df, frame_stats_df)
    quality["total_cost"] = float(np.nansum(pd.to_numeric(links_df.get("cost", np.nan), errors="coerce"))) if not links_df.empty else 0.0
    quality["n_tracks"] = float(tracks_df["TrackID"].nunique()) if not tracks_df.empty else 0.0
    quality["mean_track_length"] = float(tracks_df.groupby("TrackID").size().mean()) if not tracks_df.empty else 0.0
    quality["solver"] = 1.0 if nx is not None else 0.0

    return AssociationResult(
        tracks_df=tracks_df,
        links_df=links_df,
        frame_stats_df=frame_stats_df,
        quality=quality,
        params={
            "method": "min_cost_flow",
            "motion_model": str(motion_model),
            **p,
            **{f"gating_{k}": v for k, v in g.items()},
            "solver": "networkx_max_weight_matching" if nx is not None else "greedy_fallback",
        },
    )
