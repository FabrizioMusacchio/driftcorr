"""Tracklet-first association with stitching.

Two-stage strategy:
1) conservative frame-to-frame linking builds short, high-confidence tracklets,
2) a second stitching stage merges compatible tracklets using motion priors.

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

from .assignment import apply_gating, build_cost_matrix, hungarian_assignment, nearest_neighbor_assignment
from .linking import link_assignments_to_tracks
from .scoring import global_association_metrics, score_links_df
from .types import AssociationResult, PointCloudFrame


@dataclass(slots=True)
class _TrackletSummary:
    track_id: int
    t_start: float
    t_end: float
    p_start: np.ndarray
    p_end: np.ndarray
    v_end: np.ndarray


def _build_tracklets(
    frames: list[PointCloudFrame],
    gating: dict[str, Any],
    metric: str,
    method: str,
    use_dummy: bool,
    miss_cost: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fr = sorted(frames, key=lambda f: float(f.time))
    assigns: list[dict[str, Any]] = []
    for ti in range(len(fr) - 1):
        f0 = fr[ti]
        f1 = fr[ti + 1]
        x0 = np.asarray(f0.xyz, dtype=float)
        x1 = np.asarray(f1.xyz, dtype=float)
        cm = build_cost_matrix(
            x0,
            x1,
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
        if str(method).lower() == "nearest":
            a = nearest_neighbor_assignment(cm, gating_mask=gate, use_dummy=use_dummy, miss_cost=miss_cost)
        else:
            a = hungarian_assignment(
                cm,
                gating_mask=gate,
                use_dummy=use_dummy,
                miss_cost=miss_cost,
                birth_cost=miss_cost,
                death_cost=miss_cost,
            )
        a["t"] = float(f0.time)
        a["t_next"] = float(f1.time)
        assigns.append(a)

    return link_assignments_to_tracks(fr, assigns, allow_birth_death=True, max_gap=0)


def _summarize_tracklets(tracklets_df: pd.DataFrame, k_endpoints: int = 2) -> list[_TrackletSummary]:
    if tracklets_df.empty:
        return []
    out: list[_TrackletSummary] = []
    d = tracklets_df.copy()
    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    for tid, grp in d.groupby("TrackID"):
        g = grp.sort_values("Time")
        xyz = g[["x", "y", "z"]].to_numpy(float) if {"x", "y", "z"}.issubset(g.columns) else g[["Position X", "Position Y", "Position Z"]].to_numpy(float)
        tt = g["Time"].to_numpy(float)
        if xyz.shape[0] < 1:
            continue
        p0 = xyz[0]
        p1 = xyz[-1]
        if xyz.shape[0] >= 2:
            k = int(max(2, k_endpoints))
            k = min(k, xyz.shape[0])
            xa = xyz[-k:]
            ta = tt[-k:]
            dt = np.diff(ta)
            vv = np.zeros((max(len(dt), 1), 3), dtype=float)
            if dt.size > 0:
                for i in range(dt.size):
                    dti = float(dt[i]) if np.isfinite(dt[i]) and dt[i] > 0 else 1.0
                    vv[i] = (xa[i + 1] - xa[i]) / dti
                v = np.nanmean(vv, axis=0)
            else:
                v = np.zeros((3,), dtype=float)
        else:
            v = np.zeros((3,), dtype=float)
        out.append(
            _TrackletSummary(
                track_id=int(float(tid)),
                t_start=float(tt[0]),
                t_end=float(tt[-1]),
                p_start=np.asarray(p0, dtype=float),
                p_end=np.asarray(p1, dtype=float),
                v_end=np.asarray(v, dtype=float),
            )
        )
    return out


def _stitch_tracklets(
    tracklets_df: pd.DataFrame,
    summaries: list[_TrackletSummary],
    gating_loose: dict[str, Any],
    allow_gap: bool,
    max_gap: int,
    stitch_solver: str,
    miss_cost: float,
    stitch_metric: str,
    huber_delta: float,
    mahalanobis_scales: tuple[float, float, float] | list[float] | np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tracklets_df.empty or len(summaries) <= 1:
        return tracklets_df.copy(), pd.DataFrame(columns=["pred_tid", "succ_tid", "cost"])

    max_dt = int(max_gap) + 1 if bool(allow_gap) else 1
    ends = summaries
    starts = summaries

    cmat = np.full((len(ends), len(starts)), np.inf, dtype=float)
    candidate = np.zeros_like(cmat, dtype=bool)

    max_dist = gating_loose.get("max_dist")
    for i, a in enumerate(ends):
        for j, b in enumerate(starts):
            if int(a.track_id) == int(b.track_id):
                continue
            dt = float(b.t_start - a.t_end)
            if dt <= 0:
                continue
            if dt > float(max_dt):
                continue
            pred = a.p_end + dt * a.v_end
            res = np.asarray(pred - b.p_start, dtype=float)
            dist = float(np.linalg.norm(res))
            if max_dist is not None and dist > float(max_dist):
                continue
            m = str(stitch_metric).lower()
            if m == "huber":
                dd = max(float(huber_delta), 1e-9)
                c = float(0.5 * dist * dist) if dist <= dd else float(dd * (dist - 0.5 * dd))
            elif m == "mahalanobis_diag":
                s = np.asarray(mahalanobis_scales if mahalanobis_scales is not None else [1.0, 1.0, 1.0], dtype=float).reshape(-1)
                if s.size != 3:
                    s = np.array([1.0, 1.0, 1.0], dtype=float)
                s = np.maximum(np.abs(s), 1e-9)
                c = float(np.sum((res / s) ** 2))
            elif m == "euclidean":
                c = float(dist)
            else:
                c = float(dist * dist)
            cmat[i, j] = c
            candidate[i, j] = True

    if not np.any(candidate):
        return tracklets_df.copy(), pd.DataFrame(columns=["pred_tid", "succ_tid", "cost"])

    assn = hungarian_assignment(
        cmat,
        gating_mask=candidate,
        use_dummy=True,
        miss_cost=float(miss_cost),
        birth_cost=float(miss_cost),
        death_cost=float(miss_cost),
    )

    edges: list[tuple[int, int, float]] = []
    for i_local, j_local, c in zip(assn["row_ind"].tolist(), assn["col_ind"].tolist(), assn["costs"].tolist()):
        if not np.isfinite(float(c)):
            continue
        if str(stitch_solver).lower() not in {"hungarian", "min_cost_flow"}:
            continue
        pred_tid = int(ends[int(i_local)].track_id)
        succ_tid = int(starts[int(j_local)].track_id)
        if pred_tid == succ_tid:
            continue
        edges.append((pred_tid, succ_tid, float(c)))

    # Enforce one predecessor and one successor max in deterministic order.
    edges_sorted = sorted(edges, key=lambda x: (x[2], x[0], x[1]))
    pred_used: set[int] = set()
    succ_used: set[int] = set()
    keep: list[tuple[int, int, float]] = []
    for a, b, c in edges_sorted:
        if a in pred_used or b in succ_used:
            continue
        pred_used.add(a)
        succ_used.add(b)
        keep.append((a, b, c))

    succ_map = {a: b for a, b, _ in keep}
    pred_map = {b: a for a, b, _ in keep}

    # Build final track-id mapping through stitched chains.
    tids = sorted([int(float(t)) for t in tracklets_df["TrackID"].unique().tolist()])
    starts_chain = [tid for tid in tids if tid not in pred_map]
    final_map: dict[int, int] = {}
    next_tid = 0
    for st in starts_chain:
        cur = int(st)
        while True:
            if cur in final_map:
                break
            final_map[cur] = next_tid
            if cur not in succ_map:
                break
            cur = int(succ_map[cur])
        next_tid += 1
    for tid in tids:
        if tid not in final_map:
            final_map[tid] = next_tid
            next_tid += 1

    out = tracklets_df.copy()
    out["TrackID"] = pd.to_numeric(out["TrackID"], errors="coerce").map(lambda v: float(final_map.get(int(v), int(v))) if np.isfinite(v) else np.nan)
    out = out.sort_values(["TrackID", "Time", "PointIndex"]).reset_index(drop=True)

    stitches_df = pd.DataFrame(
        [{"pred_tid": int(a), "succ_tid": int(b), "cost": float(c)} for a, b, c in keep]
    )
    return out, stitches_df


def _links_from_tracks(tracks_df: pd.DataFrame) -> pd.DataFrame:
    if tracks_df.empty:
        return pd.DataFrame(columns=["t", "src_index", "dst_index", "track_id", "cost", "gated", "matched"])
    rows: list[dict[str, Any]] = []
    d = tracks_df.copy()
    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    d["PointIndex"] = pd.to_numeric(d["PointIndex"], errors="coerce")
    for tid, grp in d.groupby("TrackID"):
        g = grp.sort_values("Time")
        if g.shape[0] < 2:
            continue
        tt = g["Time"].to_numpy(float)
        pi = g["PointIndex"].to_numpy(int)
        xyz = g[["x", "y", "z"]].to_numpy(float) if {"x", "y", "z"}.issubset(g.columns) else g[["Position X", "Position Y", "Position Z"]].to_numpy(float)
        for i in range(len(tt) - 1):
            dt = float(tt[i + 1] - tt[i])
            if not np.isfinite(dt) or dt <= 0:
                continue
            cost = float(np.sum((xyz[i + 1] - xyz[i]) ** 2))
            rows.append(
                {
                    "t": float(tt[i]),
                    "src_index": int(pi[i]),
                    "dst_index": int(pi[i + 1]),
                    "track_id": float(tid),
                    "cost": cost,
                    "gated": True,
                    "matched": True,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["t", "src_index", "dst_index"]).reset_index(drop=True)
    return out


def _frame_stats_from_links(frames: list[PointCloudFrame], links_df: pd.DataFrame) -> pd.DataFrame:
    fr = sorted(frames, key=lambda f: float(f.time))
    rows: list[dict[str, Any]] = []
    for ti in range(len(fr) - 1):
        t0 = float(fr[ti].time)
        n0 = int(fr[ti].xyz.shape[0])
        n1 = int(fr[ti + 1].xyz.shape[0])
        if links_df.empty:
            nm = 0
        else:
            nm = int((pd.to_numeric(links_df.get("t", np.nan), errors="coerce") == t0).sum())
        rows.append(
            {
                "t": t0,
                "t_next": float(fr[ti + 1].time),
                "n_points_t": n0,
                "n_points_t_next": n1,
                "n_matched": nm,
                "n_unmatched_t": int(max(n0 - nm, 0)),
                "n_unmatched_t_next": int(max(n1 - nm, 0)),
            }
        )
    return pd.DataFrame(rows)


def associate_tracklet_stitching(
    frames: list[PointCloudFrame],
    gating_strict: dict[str, Any] | None = None,
    gating_loose: dict[str, Any] | None = None,
    motion_model: str = "constant_velocity",
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Associate point clouds via conservative tracklets plus stitching.

    Parameters
    ----------
    gating_strict:
        Stage-1 gating (strict; low false links).
    gating_loose:
        Stage-2 gating for stitching.
    params:
        - strict_factor, loose_factor
        - allow_gap, max_gap
        - k_endpoints
        - stitch_solver ("hungarian" or "min_cost_flow")
        - metric
    """
    fr = sorted(frames, key=lambda f: float(f.time))
    p = dict(params or {})
    g_strict = dict(gating_strict or {})
    g_loose = dict(gating_loose or {})

    metric = str(p.get("metric", "sqeuclidean"))
    strict_factor = float(p.get("strict_factor", 1.0))
    loose_factor = float(p.get("loose_factor", 1.8))
    allow_gap = bool(p.get("allow_gap", False))
    max_gap = int(p.get("max_gap", 0))
    k_end = int(p.get("K_endpoints", 2))
    stitch_solver = str(p.get("stitch_solver", "hungarian"))
    use_dummy = bool(p.get("use_dummy", True))
    miss_cost = float(p.get("miss_cost", 5.0))

    # Scale strict/loose max_dist from provided base value if present.
    if g_strict.get("max_dist") is not None:
        g_strict["max_dist"] = float(g_strict["max_dist"]) * strict_factor
    if g_loose.get("max_dist") is not None:
        g_loose["max_dist"] = float(g_loose["max_dist"]) * loose_factor

    tracklets_df, links_stage1_df, _ = _build_tracklets(
        fr,
        gating=g_strict,
        metric=metric,
        method="hungarian" if str(motion_model).lower() != "nearest" else "nearest",
        use_dummy=use_dummy,
        miss_cost=miss_cost,
    )

    summaries = _summarize_tracklets(tracklets_df, k_endpoints=k_end)
    tracks_df, stitches_df = _stitch_tracklets(
        tracklets_df,
        summaries,
        gating_loose=g_loose,
        allow_gap=allow_gap,
        max_gap=max_gap,
        stitch_solver=stitch_solver,
        miss_cost=miss_cost,
        stitch_metric=str(p.get("stitch_metric", metric)),
        huber_delta=float(p.get("huber_delta", 1.0)),
        mahalanobis_scales=p.get("mahalanobis_scales"),
    )

    links_df = _links_from_tracks(tracks_df)
    links_df = score_links_df(links_df)
    frame_stats_df = _frame_stats_from_links(fr, links_df)
    quality = global_association_metrics(tracks_df, links_df, frame_stats_df)
    quality["n_tracklets_stage1"] = float(tracklets_df["TrackID"].nunique()) if not tracklets_df.empty else 0.0
    quality["n_tracks_final"] = float(tracks_df["TrackID"].nunique()) if not tracks_df.empty else 0.0
    quality["n_stitches"] = float(stitches_df.shape[0])
    quality["total_cost"] = float(np.nansum(pd.to_numeric(links_df.get("cost", np.nan), errors="coerce"))) if not links_df.empty else 0.0
    quality["mean_track_length"] = float(tracks_df.groupby("TrackID").size().mean()) if not tracks_df.empty else 0.0

    params_out: dict[str, Any] = {
        "method": "tracklet_stitching",
        "motion_model": str(motion_model),
        "metric": metric,
        "strict_factor": strict_factor,
        "loose_factor": loose_factor,
        "allow_gap": allow_gap,
        "max_gap": max_gap,
        "K_endpoints": k_end,
        "stitch_solver": stitch_solver,
        "use_dummy": use_dummy,
    }
    params_out.update({f"gating_strict_{k}": v for k, v in g_strict.items()})
    params_out.update({f"gating_loose_{k}": v for k, v in g_loose.items()})

    # Keep stage-1 links for optional diagnostics without changing output schema.
    if not links_stage1_df.empty:
        quality["stage1_link_count"] = float(links_stage1_df.shape[0])

    return AssociationResult(
        tracks_df=tracks_df,
        links_df=links_df,
        frame_stats_df=frame_stats_df,
        quality=quality,
        params=params_out,
    )
