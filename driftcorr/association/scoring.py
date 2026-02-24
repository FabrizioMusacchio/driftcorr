"""Quality metrics for point-cloud association.

This module provides:
- confidence scoring for links and tracks,
- global diagnostics without ground truth,
- ground-truth-aware association metrics for synthetic benchmarks.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import warnings


def link_confidence(cost: np.ndarray | float, max_cost: float | None = None, scale: float | None = None) -> np.ndarray:
    """Convert assignment cost to [0,1] confidence.

    Confidence decays exponentially with cost. If `max_cost` is provided,
    confidence is approximately 0 near/above this threshold.
    """
    c = np.asarray(cost, dtype=float)
    if scale is None:
        if max_cost is not None and np.isfinite(max_cost):
            scale = max(float(max_cost), 1e-9) / 3.0
        else:
            finite = c[np.isfinite(c)]
            med = float(np.median(finite)) if finite.size else 1.0
            scale = max(med, 1e-9)
    s = max(float(scale), 1e-9)
    conf = np.exp(-np.maximum(c, 0.0) / s)
    conf[~np.isfinite(c)] = 0.0
    return np.clip(conf, 0.0, 1.0)


def score_links_df(links_df: pd.DataFrame, max_cost: float | None = None) -> pd.DataFrame:
    """Add per-link confidence scores to a link table."""
    if links_df.empty:
        out = links_df.copy()
        out["confidence"] = []
        return out
    out = links_df.copy()
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out["confidence"] = link_confidence(out["cost"].to_numpy(float), max_cost=max_cost)
    return out


def score_tracks_df(tracks_df: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track quality scores.

    Output columns include track length, mean confidence, number of gaps and a
    smoothness proxy based on consecutive displacement magnitudes.
    """
    if tracks_df.empty:
        return pd.DataFrame(columns=["TrackID", "length", "mean_confidence", "n_gaps", "smoothness"])

    t = tracks_df.copy()
    t["TrackID"] = pd.to_numeric(t["TrackID"], errors="coerce")
    t["Time"] = pd.to_numeric(t["Time"], errors="coerce")

    l = score_links_df(links_df)
    if not l.empty:
        l["track_id"] = pd.to_numeric(l["track_id"], errors="coerce")
        conf_mean = l.groupby("track_id", as_index=False)["confidence"].mean().rename(columns={"track_id": "TrackID", "confidence": "mean_confidence"})
    else:
        conf_mean = pd.DataFrame(columns=["TrackID", "mean_confidence"])

    rows: list[dict[str, Any]] = []
    for tid, grp in t.groupby("TrackID"):
        g = grp.sort_values("Time")
        x = g[["x", "y", "z"]].to_numpy(float) if {"x", "y", "z"}.issubset(g.columns) else g[["Position X", "Position Y", "Position Z"]].to_numpy(float)
        tt = g["Time"].to_numpy(float)
        d = np.linalg.norm(np.diff(x, axis=0), axis=1) if x.shape[0] > 1 else np.array([], dtype=float)
        dt = np.diff(tt) if tt.size > 1 else np.array([], dtype=float)
        n_gaps = int(np.sum(dt > 1.5 * np.nanmedian(dt))) if dt.size else 0
        smooth = float(np.nanmedian(np.abs(np.diff(d)))) if d.size > 1 else 0.0
        rows.append({"TrackID": float(tid), "length": int(g.shape[0]), "n_gaps": n_gaps, "smoothness": smooth})

    out = pd.DataFrame(rows)
    out = out.merge(conf_mean, on="TrackID", how="left")
    out["mean_confidence"] = pd.to_numeric(out["mean_confidence"], errors="coerce").fillna(0.0)
    return out


def global_association_metrics(tracks_df: pd.DataFrame, links_df: pd.DataFrame, frame_stats_df: pd.DataFrame) -> dict[str, float]:
    """Compute global no-GT quality metrics.

    Metrics are proxies when GT is unavailable and should be interpreted as
    diagnostics rather than absolute correctness scores.
    """
    metrics: dict[str, float] = {}
    if frame_stats_df.empty:
        return {
            "unmatched_rate": float("nan"),
            "mean_cost": float("nan"),
            "fragmentation_rate": float("nan"),
            "track_switch_rate": float("nan"),
            "mean_confidence": float("nan"),
        }

    fs = frame_stats_df.copy()
    nm = pd.to_numeric(fs.get("n_matched", np.nan), errors="coerce")
    nu = pd.to_numeric(fs.get("n_unmatched_t_next", np.nan), errors="coerce")
    denom = nm + nu
    unmatched_rate = float(np.nanmean(np.where(denom > 0, nu / denom, np.nan)))

    l = score_links_df(links_df)
    mean_cost = float(np.nanmean(pd.to_numeric(l.get("cost", np.nan), errors="coerce"))) if not l.empty else float("nan")
    mean_conf = float(np.nanmean(pd.to_numeric(l.get("confidence", np.nan), errors="coerce"))) if not l.empty else float("nan")

    n_tracks = int(tracks_df["TrackID"].nunique()) if "TrackID" in tracks_df.columns and not tracks_df.empty else 0
    n_points = int(tracks_df.shape[0]) if not tracks_df.empty else 0
    fragmentation_rate = float(n_tracks / max(n_points, 1))

    # Switch-like proxy: large kinematic jumps within inferred tracks.
    switch_events = 0
    step_count = 0
    if not tracks_df.empty:
        for _, grp in tracks_df.groupby("TrackID"):
            g = grp.sort_values("Time")
            x = g[["x", "y", "z"]].to_numpy(float) if {"x", "y", "z"}.issubset(g.columns) else g[["Position X", "Position Y", "Position Z"]].to_numpy(float)
            if x.shape[0] < 3:
                continue
            d = np.linalg.norm(np.diff(x, axis=0), axis=1)
            med = float(np.nanmedian(d)) if d.size else 0.0
            mad = float(np.nanmedian(np.abs(d - med))) if d.size else 0.0
            thr = med + 3.0 * max(mad, 1e-9)
            switch_events += int(np.sum(d > thr))
            step_count += int(d.size)
    track_switch_rate = float(switch_events / max(step_count, 1))

    metrics["unmatched_rate"] = unmatched_rate
    metrics["mean_cost"] = mean_cost
    metrics["fragmentation_rate"] = fragmentation_rate
    metrics["track_switch_rate"] = track_switch_rate
    metrics["mean_confidence"] = mean_conf
    return metrics


def evaluate_against_groundtruth(
    inferred_tracks_df: pd.DataFrame,
    point_to_gt_df: pd.DataFrame,
    links_df: pd.DataFrame | None = None,
    gt_tracks_df: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compute GT-aware association metrics.

    Parameters
    ----------
    inferred_tracks_df:
        Inferred pseudo tracks with `TrackID`, `Time`, `PointIndex`.
    point_to_gt_df:
        Unblinding table with `Time`, `PointIndex`, `GT_TrackID`.
    links_df:
        Optional inferred link table (`t`, `src_index`, `dst_index`).

    gt_tracks_df:
        Optional GT coordinates with columns `Time`, `TrackID` and xyz
        (`Position X/Y/Z` or `x/y/z`). If provided, RMS is computed as
        GT-identity-consistent positional error over matched GT IDs per time.

    Returns
    -------
    summary, per_time, mapping_table
    where `per_time` includes:
    - Time
    - link_precision
    - link_recall
    - link_f1
    - n_links_inferred
    - n_links_correct
    - n_links_gt_expected
    - rms
    - n_gt_tracks
    - n_matched_tracks
    - n_unmatched_tracks
    """
    req_inf = {"TrackID", "Time", "PointIndex"}
    req_gt = {"Time", "PointIndex", "GT_TrackID"}
    if not req_inf.issubset(inferred_tracks_df.columns):
        raise ValueError(f"inferred_tracks_df missing columns: {sorted(req_inf - set(inferred_tracks_df.columns))}")
    if not req_gt.issubset(point_to_gt_df.columns):
        raise ValueError(f"point_to_gt_df missing columns: {sorted(req_gt - set(point_to_gt_df.columns))}")

    inf = inferred_tracks_df[["TrackID", "Time", "PointIndex"]].copy()
    gt = point_to_gt_df[["Time", "PointIndex", "GT_TrackID"]].copy()
    inf["Time"] = pd.to_numeric(inf["Time"], errors="coerce")
    inf["PointIndex"] = pd.to_numeric(inf["PointIndex"], errors="coerce")
    gt["Time"] = pd.to_numeric(gt["Time"], errors="coerce")
    gt["PointIndex"] = pd.to_numeric(gt["PointIndex"], errors="coerce")
    merged = inf.merge(gt, on=["Time", "PointIndex"], how="inner")

    coverage = float(merged.shape[0] / max(gt.shape[0], 1))

    # Track purity (weighted by track length).
    purity_num = 0.0
    purity_den = 0.0
    for _, grp in merged.groupby("TrackID"):
        n = float(grp.shape[0])
        if n <= 0:
            continue
        top = float(grp["GT_TrackID"].value_counts(dropna=False).iloc[0])
        purity_num += top
        purity_den += n
    track_purity = float(purity_num / max(purity_den, 1.0))

    # Fragmentation and GT-wise ID switches.
    gt_frag = merged.groupby("GT_TrackID")["TrackID"].nunique()
    frag_counts = gt_frag.to_numpy(dtype=float)
    fragmentation_rate = float(np.mean(frag_counts > 1.0)) if frag_counts.size else float("nan")
    mean_fragments_per_gt = float(np.mean(frag_counts)) if frag_counts.size else float("nan")

    id_switches_total = 0
    gt_tracks_for_switch = 0
    for _, grp in merged.groupby("GT_TrackID"):
        g = grp.sort_values("Time")
        tids = g["TrackID"].to_numpy()
        if tids.size < 2:
            continue
        gt_tracks_for_switch += 1
        id_switches_total += int(np.sum(tids[1:] != tids[:-1]))
    id_switches_mean_per_track = float(id_switches_total / max(gt_tracks_for_switch, 1))

    # Link precision/recall/F1 over time.
    per_time_lookup: dict[float, dict[str, Any]] = {}
    times_sorted = np.sort(gt["Time"].dropna().unique())
    t_next_lookup = {float(times_sorted[i]): float(times_sorted[i + 1]) for i in range(max(0, len(times_sorted) - 1))}
    gt_ids_by_time = {
        float(t): set(gt.loc[gt["Time"] == float(t), "GT_TrackID"].astype(float).tolist())
        for t in times_sorted
    }
    for i in range(max(0, len(times_sorted) - 1)):
        t = float(times_sorted[i])
        t_next = float(times_sorted[i + 1])
        gt_links = len(gt_ids_by_time.get(t, set()) & gt_ids_by_time.get(t_next, set()))
        per_time_lookup[t] = {
            "Time": t,
            "link_precision": np.nan,
            "link_recall": np.nan,
            "link_f1": np.nan,
            "n_links_inferred": 0,
            "n_links_correct": 0,
            "n_links_gt_expected": int(gt_links),
            "mean_link_cost": np.nan,
        }

    link_precision_mean = float("nan")
    link_recall_mean = float("nan")
    link_f1_mean = float("nan")
    mean_link_cost = float("nan")
    if links_df is not None and not links_df.empty:
        l = links_df.copy()
        l["t"] = pd.to_numeric(l["t"], errors="coerce")
        l["src_index"] = pd.to_numeric(l["src_index"], errors="coerce")
        l["dst_index"] = pd.to_numeric(l["dst_index"], errors="coerce")
        g_src = gt.rename(columns={"Time": "t", "PointIndex": "src_index", "GT_TrackID": "GT_src"})
        g_dst = gt.rename(columns={"Time": "t_next", "PointIndex": "dst_index", "GT_TrackID": "GT_dst"})
        inf_src = inf.rename(columns={"Time": "t", "PointIndex": "src_index", "TrackID": "INF_src_track"})
        inf_dst = inf.rename(columns={"Time": "t_next", "PointIndex": "dst_index", "TrackID": "INF_dst_track"})
        # infer next time from sorted unique frame times
        l["t_next"] = l["t"].map(t_next_lookup)
        lm = l.merge(g_src, on=["t", "src_index"], how="left")
        lm = lm.merge(g_dst, on=["t_next", "dst_index"], how="left")
        lm = lm.merge(inf_src[["t", "src_index", "INF_src_track"]], on=["t", "src_index"], how="left")
        lm = lm.merge(inf_dst[["t_next", "dst_index", "INF_dst_track"]], on=["t_next", "dst_index"], how="left")
        # A link is correct only if it preserves GT identity and remains within
        # the same inferred TrackID across the transition.
        same_gt = (lm["GT_src"].astype(str) == lm["GT_dst"].astype(str)) & lm["GT_src"].notna() & lm["GT_dst"].notna()
        same_inferred_track = (
            lm["INF_src_track"].notna()
            & lm["INF_dst_track"].notna()
            & (pd.to_numeric(lm["INF_src_track"], errors="coerce") == pd.to_numeric(lm["INF_dst_track"], errors="coerce"))
        )
        lm["link_correct"] = same_gt & same_inferred_track
        if lm.shape[0] > 0:
            mean_link_cost = float(np.nanmean(pd.to_numeric(lm.get("cost", np.nan), errors="coerce")))
        for t, gtm in lm.groupby("t"):
            tt = float(t)
            n_inf = int(gtm.shape[0])
            n_correct = int(np.nansum(gtm["link_correct"].to_numpy(dtype=float)))
            n_gt = int(per_time_lookup.get(tt, {}).get("n_links_gt_expected", 0))
            precision = float(n_correct / n_inf) if n_inf > 0 else 0.0
            recall = float(n_correct / n_gt) if n_gt > 0 else 0.0
            f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_time_lookup.setdefault(
                tt,
                {
                    "Time": tt,
                    "n_links_gt_expected": n_gt,
                },
            )
            per_time_lookup[tt].update(
                {
                    "link_precision": precision,
                    "link_recall": recall,
                    "link_f1": f1,
                    "n_links_inferred": n_inf,
                    "n_links_correct": n_correct,
                    "mean_link_cost": float(np.nanmean(pd.to_numeric(gtm.get("cost", np.nan), errors="coerce"))),
                }
            )

    # Fill missing per-time entries for times without inferred links.
    for tt, row in per_time_lookup.items():
        n_inf = int(row.get("n_links_inferred", 0))
        n_corr = int(row.get("n_links_correct", 0))
        n_gt = int(row.get("n_links_gt_expected", 0))
        if not np.isfinite(pd.to_numeric(row.get("link_precision"), errors="coerce")):
            row["link_precision"] = float(n_corr / n_inf) if n_inf > 0 else 0.0
        if not np.isfinite(pd.to_numeric(row.get("link_recall"), errors="coerce")):
            row["link_recall"] = float(n_corr / n_gt) if n_gt > 0 else 0.0
        p = float(row["link_precision"])
        r = float(row["link_recall"])
        if not np.isfinite(pd.to_numeric(row.get("link_f1"), errors="coerce")):
            row["link_f1"] = float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if "mean_link_cost" not in row:
            row["mean_link_cost"] = np.nan

    # ID-consistent RMS per time.
    gt_pos_by_time: dict[float, pd.DataFrame] = {}
    if gt_tracks_df is not None and not gt_tracks_df.empty:
        gtp = gt_tracks_df.copy()
        if {"Position X", "Position Y", "Position Z"}.issubset(gtp.columns):
            gtp["x_gt"] = pd.to_numeric(gtp["Position X"], errors="coerce")
            gtp["y_gt"] = pd.to_numeric(gtp["Position Y"], errors="coerce")
            gtp["z_gt"] = pd.to_numeric(gtp["Position Z"], errors="coerce")
        elif {"x", "y", "z"}.issubset(gtp.columns):
            gtp["x_gt"] = pd.to_numeric(gtp["x"], errors="coerce")
            gtp["y_gt"] = pd.to_numeric(gtp["y"], errors="coerce")
            gtp["z_gt"] = pd.to_numeric(gtp["z"], errors="coerce")
        else:
            gtp = pd.DataFrame()
        if not gtp.empty and {"Time", "TrackID", "x_gt", "y_gt", "z_gt"}.issubset(gtp.columns):
            gtp["Time"] = pd.to_numeric(gtp["Time"], errors="coerce")
            gtp["TrackID"] = pd.to_numeric(gtp["TrackID"], errors="coerce")
            gtp = gtp.dropna(subset=["Time", "TrackID", "x_gt", "y_gt", "z_gt"])
            for t, grp in gtp.groupby("Time"):
                gt_pos_by_time[float(t)] = grp.rename(columns={"TrackID": "GT_TrackID"})[
                    ["GT_TrackID", "x_gt", "y_gt", "z_gt"]
                ].copy()

    inf_pos = inferred_tracks_df.copy()
    if {"x", "y", "z"}.issubset(inf_pos.columns):
        inf_pos["x_inf"] = pd.to_numeric(inf_pos["x"], errors="coerce")
        inf_pos["y_inf"] = pd.to_numeric(inf_pos["y"], errors="coerce")
        inf_pos["z_inf"] = pd.to_numeric(inf_pos["z"], errors="coerce")
    elif {"Position X", "Position Y", "Position Z"}.issubset(inf_pos.columns):
        inf_pos["x_inf"] = pd.to_numeric(inf_pos["Position X"], errors="coerce")
        inf_pos["y_inf"] = pd.to_numeric(inf_pos["Position Y"], errors="coerce")
        inf_pos["z_inf"] = pd.to_numeric(inf_pos["Position Z"], errors="coerce")
    else:
        inf_pos["x_inf"] = np.nan
        inf_pos["y_inf"] = np.nan
        inf_pos["z_inf"] = np.nan
    inf_pos["Time"] = pd.to_numeric(inf_pos["Time"], errors="coerce")
    inf_pos["TrackID"] = pd.to_numeric(inf_pos["TrackID"], errors="coerce")
    inf_pos = inf_pos[["Time", "TrackID", "x_inf", "y_inf", "z_inf"]].copy()
    inf_pos = inf_pos.dropna(subset=["Time", "TrackID"]).sort_values(["Time", "TrackID"]).drop_duplicates(["Time", "TrackID"], keep="first")

    # Dominant inferred TrackID for each GT TrackID for identity-consistent RMS.
    gt_to_inf = (
        merged.groupby(["GT_TrackID", "TrackID"], as_index=False)
        .size()
        .sort_values(["GT_TrackID", "size", "TrackID"], ascending=[True, False, True])
        .drop_duplicates(["GT_TrackID"], keep="first")
    )
    gt_to_inf_map = {
        float(r["GT_TrackID"]): float(r["TrackID"])
        for _, r in gt_to_inf.iterrows()
        if np.isfinite(pd.to_numeric(r["GT_TrackID"], errors="coerce")) and np.isfinite(pd.to_numeric(r["TrackID"], errors="coerce"))
    }
    inf_pos_map = {
        (float(r["Time"]), float(r["TrackID"])): (
            float(r["x_inf"]),
            float(r["y_inf"]),
            float(r["z_inf"]),
        )
        for _, r in inf_pos.iterrows()
        if np.isfinite(pd.to_numeric(r["Time"], errors="coerce")) and np.isfinite(pd.to_numeric(r["TrackID"], errors="coerce"))
    }

    for t in times_sorted:
        tt = float(t)
        row = per_time_lookup.setdefault(
            tt,
            {
                "Time": tt,
                "link_precision": 0.0,
                "link_recall": 0.0,
                "link_f1": 0.0,
                "n_links_inferred": 0,
                "n_links_correct": 0,
                "n_links_gt_expected": int(len(gt_ids_by_time.get(tt, set()) & gt_ids_by_time.get(float(t_next_lookup.get(tt, np.nan)), set())))
                if tt in t_next_lookup
                else 0,
                "mean_link_cost": np.nan,
            },
        )
        gt_t = gt_pos_by_time.get(tt, pd.DataFrame(columns=["GT_TrackID", "x_gt", "y_gt", "z_gt"])).copy()
        if gt_t.empty:
            # Fallback on mapping table for counts if GT coordinates are unavailable.
            n_gt = int(gt.loc[gt["Time"] == tt, "GT_TrackID"].nunique())
            row.update({"rms": np.nan, "n_gt_tracks": n_gt, "n_matched_tracks": 0, "n_unmatched_tracks": n_gt})
            continue

        n_gt = int(gt_t["GT_TrackID"].nunique())
        errs_sq: list[float] = []
        n_matched = 0
        for _, gr in gt_t.iterrows():
            gtid = float(gr["GT_TrackID"])
            inf_tid = gt_to_inf_map.get(gtid)
            if inf_tid is None:
                continue
            inf_xyz = inf_pos_map.get((tt, float(inf_tid)))
            if inf_xyz is None:
                continue
            if not np.all(np.isfinite(np.asarray(inf_xyz, dtype=float))):
                continue
            gt_xyz = np.asarray([gr["x_gt"], gr["y_gt"], gr["z_gt"]], dtype=float)
            if not np.all(np.isfinite(gt_xyz)):
                continue
            e = np.asarray(inf_xyz, dtype=float) - gt_xyz
            errs_sq.append(float(np.sum(e * e)))
            n_matched += 1

        rms_t = float(np.sqrt(np.mean(np.asarray(errs_sq, dtype=float)))) if errs_sq else np.nan
        n_unmatched = int(max(n_gt - n_matched, 0))
        row.update(
            {
                "rms": rms_t,
                "n_gt_tracks": n_gt,
                "n_matched_tracks": n_matched,
                "n_unmatched_tracks": n_unmatched,
            }
        )

    # Dominant mapping heatmap table.
    mapping = (
        merged.groupby(["GT_TrackID", "TrackID"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    per_time = pd.DataFrame(list(per_time_lookup.values()))
    if not per_time.empty:
        per_time = per_time.sort_values("Time").reset_index(drop=True)
        per_time["link_precision"] = pd.to_numeric(per_time["link_precision"], errors="coerce").fillna(0.0)
        per_time["link_recall"] = pd.to_numeric(per_time["link_recall"], errors="coerce").fillna(0.0)
        per_time["link_f1"] = pd.to_numeric(per_time["link_f1"], errors="coerce").fillna(0.0)
        per_time["n_links_inferred"] = pd.to_numeric(per_time["n_links_inferred"], errors="coerce").fillna(0).astype(int)
        per_time["n_links_correct"] = pd.to_numeric(per_time["n_links_correct"], errors="coerce").fillna(0).astype(int)
        per_time["n_links_gt_expected"] = pd.to_numeric(per_time["n_links_gt_expected"], errors="coerce").fillna(0).astype(int)
        per_time["mean_link_cost"] = pd.to_numeric(per_time["mean_link_cost"], errors="coerce")
        per_time["rms"] = pd.to_numeric(per_time.get("rms", np.nan), errors="coerce")
        per_time["n_gt_tracks"] = pd.to_numeric(per_time.get("n_gt_tracks", 0), errors="coerce").fillna(0).astype(int)
        per_time["n_matched_tracks"] = pd.to_numeric(per_time.get("n_matched_tracks", 0), errors="coerce").fillna(0).astype(int)
        per_time["n_unmatched_tracks"] = pd.to_numeric(per_time.get("n_unmatched_tracks", 0), errors="coerce").fillna(0).astype(int)
        per_time = per_time[
            [
                "Time",
                "link_precision",
                "link_recall",
                "link_f1",
                "n_links_inferred",
                "n_links_correct",
                "n_links_gt_expected",
                "rms",
                "n_gt_tracks",
                "n_matched_tracks",
                "n_unmatched_tracks",
                "mean_link_cost",
            ]
        ]
        trans_mask = (per_time["n_links_gt_expected"].to_numpy(int) > 0) | (per_time["n_links_inferred"].to_numpy(int) > 0)
        if bool(np.any(trans_mask)):
            link_precision_mean = float(np.nanmean(per_time.loc[trans_mask, "link_precision"].to_numpy(float)))
            link_recall_mean = float(np.nanmean(per_time.loc[trans_mask, "link_recall"].to_numpy(float)))
            link_f1_mean = float(np.nanmean(per_time.loc[trans_mask, "link_f1"].to_numpy(float)))
        else:
            link_precision_mean = float("nan")
            link_recall_mean = float("nan")
            link_f1_mean = float("nan")
        rms_mean = float(np.nanmean(per_time["rms"].to_numpy(float)))
        rms_std = float(np.nanstd(per_time["rms"].to_numpy(float)))
        unmatched_rate_mean = float(
            np.nanmean(
                np.where(
                    per_time["n_gt_tracks"].to_numpy(int) > 0,
                    per_time["n_unmatched_tracks"].to_numpy(float) / np.maximum(per_time["n_gt_tracks"].to_numpy(float), 1.0),
                    np.nan,
                )
            )
        )
    else:
        per_time = pd.DataFrame(
            columns=[
                "Time",
                "link_precision",
                "link_recall",
                "link_f1",
                "n_links_inferred",
                "n_links_correct",
                "n_links_gt_expected",
                "rms",
                "n_gt_tracks",
                "n_matched_tracks",
                "n_unmatched_tracks",
                "mean_link_cost",
            ]
        )
        link_precision_mean = float("nan")
        link_recall_mean = float("nan")
        link_f1_mean = float("nan")
        rms_mean = float("nan")
        rms_std = float("nan")
        unmatched_rate_mean = float("nan")

    summary = {
        "coverage": coverage,
        "track_purity": track_purity,
        "fragmentation_rate": fragmentation_rate,
        "mean_fragments_per_gt": mean_fragments_per_gt,
        "id_switches_total": float(id_switches_total),
        "id_switches_mean_per_track": id_switches_mean_per_track,
        "link_precision_mean": link_precision_mean,
        "link_recall_mean": link_recall_mean,
        "link_f1_mean": link_f1_mean,
        "mean_link_cost": mean_link_cost,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "unmatched_rate_mean": unmatched_rate_mean,
    }

    eps = 1e-6
    if np.isfinite(link_f1_mean) and np.isclose(link_f1_mean, 1.0, atol=1e-12) and np.isfinite(fragmentation_rate) and fragmentation_rate > 0:
        warnings.warn(
            "Perfect link_f1_mean but fragmentation_rate > 0 detected. Check association/scoring consistency.",
            RuntimeWarning,
        )
    if np.isfinite(link_f1_mean) and np.isclose(link_f1_mean, 1.0, atol=1e-12) and np.isfinite(rms_mean) and rms_mean > eps:
        warnings.warn(
            "Perfect association (link_f1_mean=1) but non-zero RMS detected. Check scoring logic or coordinate alignment.",
            RuntimeWarning,
        )

    return summary, per_time, mapping
