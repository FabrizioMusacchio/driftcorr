"""Hungarian constant-velocity variants for point-cloud association.

This module implements four focused Hungarian+CV variants:
- adaptive max-distance gating from innovation scale,
- innovation-normalized percentile gating,
- switch-penalty augmented cost,
- two-stage Hungarian (strict then residual pass).

All functions return `AssociationResult` compatible with the existing module-1
pipeline and evaluation code.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .assignment import apply_gating, build_cost_matrix, hungarian_assignment
from .linking import link_assignments_to_tracks
from .scoring import global_association_metrics, score_links_df
from .types import AssociationResult, PointCloudFrame


def _estimate_velocity_by_observation(tracks_df: pd.DataFrame) -> dict[tuple[float, int], np.ndarray]:
    """Estimate per-observation velocity from inferred track trajectories."""
    if tracks_df.empty:
        return {}
    d = tracks_df.copy()
    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    if "PointIndex" not in d.columns:
        d["PointIndex"] = d.groupby("Time").cumcount().astype(int)
    if {"x", "y", "z"}.issubset(d.columns):
        cols = ["x", "y", "z"]
    else:
        cols = ["Position X", "Position Y", "Position Z"]

    vel: dict[tuple[float, int], np.ndarray] = {}
    for _, grp in d.groupby("TrackID"):
        g = grp.sort_values("Time")
        tt = g["Time"].to_numpy(float)
        xyz = g[cols].to_numpy(float)
        pi = g["PointIndex"].to_numpy(int)
        if xyz.shape[0] < 2:
            continue
        for k in range(1, xyz.shape[0]):
            dt = float(tt[k] - tt[k - 1])
            if not np.isfinite(dt) or dt <= 0.0:
                continue
            v = (xyz[k] - xyz[k - 1]) / dt
            vel[(float(tt[k]), int(pi[k]))] = np.asarray(v, dtype=float)
    return vel


def _build_result(
    frames: list[PointCloudFrame],
    assignments_by_t: list[dict[str, Any]],
    params: dict[str, Any],
) -> AssociationResult:
    tracks_df, links_df, frame_stats_df = link_assignments_to_tracks(frames, assignments_by_t, allow_birth_death=True, max_gap=0)
    links_df = score_links_df(links_df)
    quality = global_association_metrics(tracks_df, links_df, frame_stats_df)
    quality["n_tracks"] = float(tracks_df["TrackID"].nunique()) if "TrackID" in tracks_df.columns and not tracks_df.empty else 0.0
    quality["n_links"] = float(links_df.shape[0]) if not links_df.empty else 0.0
    quality["mean_track_length"] = float(tracks_df.groupby("TrackID").size().mean()) if not tracks_df.empty else 0.0
    quality["total_cost"] = float(np.nansum(pd.to_numeric(links_df.get("cost", np.nan), errors="coerce"))) if not links_df.empty else 0.0
    return AssociationResult(
        tracks_df=tracks_df,
        links_df=links_df,
        frame_stats_df=frame_stats_df,
        quality=quality,
        params=params,
    )


def _robust_sigma_from_residuals(
    residuals: np.ndarray,
    sigma_prev: np.ndarray,
    sigma_init: float,
    sigma_mad_clip: float,
) -> np.ndarray:
    """Estimate per-axis innovation sigma via MAD with clipping."""
    if residuals.ndim != 2 or residuals.shape[1] != 3 or residuals.shape[0] < 3:
        return np.asarray(sigma_prev, dtype=float)
    med = np.nanmedian(residuals, axis=0)
    mad = np.nanmedian(np.abs(residuals - med[None, :]), axis=0)
    sigma = mad / 0.67448975
    sigma = np.where(np.isfinite(sigma), sigma, sigma_prev)
    clip = max(float(sigma_mad_clip), 1.0)
    lo = max(float(sigma_init) / clip, 1e-6)
    hi = max(float(sigma_init) * clip, lo)
    sigma = np.clip(sigma, lo, hi)
    sigma = np.where(np.isfinite(sigma), sigma, sigma_prev)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    return sigma


def _combine_masks(mask_a: np.ndarray, mask_b: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(mask_a, dtype=bool)
    if mask_b is None:
        return a
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} vs {b.shape}")
    return a & b


def _run_variant_pairwise(
    frames: list[PointCloudFrame],
    variant: str,
    gating: dict[str, Any],
    params: dict[str, Any],
    velocity_map: dict[tuple[float, int], np.ndarray] | None,
    dx_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    fr = sorted(frames, key=lambda f: float(f.time))
    out: list[dict[str, Any]] = []

    sigma_init = float(params.get("sigma_init", dx_scale if np.isfinite(dx_scale) and dx_scale > 0 else 1.0))
    sigma_mad_clip = float(params.get("sigma_mad_clip", 3.0))
    sigma_vec = np.array([sigma_init, sigma_init, sigma_init], dtype=float)
    sigma_hist: list[float] = []

    max_dist_factor = float(params.get("max_dist_factor", 1.0))
    keep_percentile = gating.get("keep_percentile")
    use_dummy = bool(params.get("use_dummy", True))
    miss_cost = params.get("miss_cost")
    birth_cost = params.get("birth_cost")
    death_cost = params.get("death_cost")

    for ti in range(len(fr) - 1):
        f0 = fr[ti]
        f1 = fr[ti + 1]
        x0 = np.asarray(f0.xyz, dtype=float)
        x1 = np.asarray(f1.xyz, dtype=float)
        dt = float(f1.time - f0.time)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0

        x_pred = x0.copy()
        v_src = np.zeros_like(x0)
        if velocity_map is not None:
            for i in range(x0.shape[0]):
                v = velocity_map.get((float(f0.time), int(i)))
                if v is not None:
                    vv = np.asarray(v, dtype=float).reshape(-1)
                    if vv.size == 3 and np.all(np.isfinite(vv)):
                        v_src[i] = vv
                        x_pred[i] = x0[i] + dt * vv

        # Residual tensor: src x dst x 3
        res = x1[None, :, :] - x_pred[:, None, :]
        dist = np.sqrt(np.maximum(np.sum(res * res, axis=2), 0.0))
        sigma_safe = np.maximum(sigma_vec, 1e-6)
        d_norm = np.sqrt(
            np.maximum(
                (res[:, :, 0] / sigma_safe[0]) ** 2
                + (res[:, :, 1] / sigma_safe[1]) ** 2
                + (res[:, :, 2] / sigma_safe[2]) ** 2,
                0.0,
            )
        )

        if variant == "adaptive_maxdist":
            # Max distance in micron units, adapted from innovation scale.
            adaptive_max_dist = max(0.5, float(max_dist_factor) * float(np.nanmean(sigma_safe)))
            cm = build_cost_matrix(
                x_pred,
                x1,
                metric=str(params.get("metric", "sqeuclidean")),
                cov=gating.get("cov"),
                huber_delta=float(gating.get("huber_delta", 1.0)),
                mahalanobis_scales=gating.get("mahalanobis_scales"),
            )
            mask_dist = np.isfinite(dist) & (dist <= adaptive_max_dist)
            mask_pct = apply_gating(
                dist,
                max_dist=None,
                max_cost=None,
                keep_percentile=keep_percentile,
                assume_squared_cost=False,
            )
            gate = _combine_masks(mask_dist, mask_pct)
            a = hungarian_assignment(
                cm,
                gating_mask=gate,
                use_dummy=use_dummy,
                miss_cost=miss_cost,
                birth_cost=birth_cost,
                death_cost=death_cost,
            )
            matched_res = res[a["row_ind"], a["col_ind"], :] if a["row_ind"].size else np.empty((0, 3), dtype=float)
            sigma_vec = _robust_sigma_from_residuals(matched_res, sigma_vec, sigma_init, sigma_mad_clip)
            sigma_hist.append(float(np.nanmean(sigma_vec)))

        elif variant == "innovation_percentile":
            # max_dist interpreted in normalized innovation units.
            cm = d_norm * d_norm
            norm_max = float(max_dist_factor)
            gate = apply_gating(
                d_norm,
                max_dist=norm_max,
                max_cost=None,
                keep_percentile=keep_percentile,
                assume_squared_cost=False,
            )
            a = hungarian_assignment(
                cm,
                gating_mask=gate,
                use_dummy=use_dummy,
                miss_cost=miss_cost,
                birth_cost=birth_cost,
                death_cost=death_cost,
            )
            matched_res = res[a["row_ind"], a["col_ind"], :] if a["row_ind"].size else np.empty((0, 3), dtype=float)
            sigma_vec = _robust_sigma_from_residuals(matched_res, sigma_vec, sigma_init, sigma_mad_clip)
            sigma_hist.append(float(np.nanmean(sigma_vec)))

        elif variant == "switch_penalty":
            use_norm = bool(params.get("use_normalized", True))
            lam = float(params.get("lambda_switch", 0.2))
            if use_norm:
                cm = d_norm * d_norm
            else:
                cm = dist * dist
            eps = 1e-9
            v_norm = np.sqrt(np.sum(v_src * v_src, axis=1))
            for i in range(cm.shape[0]):
                if v_norm[i] <= eps:
                    continue
                vi = v_src[i]
                r = res[i]  # (M,3)
                r_norm = np.sqrt(np.maximum(np.sum(r * r, axis=1), 0.0))
                denom = np.maximum(v_norm[i] * r_norm, eps)
                cosang = np.sum(r * vi[None, :], axis=1) / denom
                penalty = lam * np.maximum(0.0, -cosang)
                cm[i, :] = cm[i, :] + penalty
            base_max = float(gating.get("max_dist", max(0.5, max_dist_factor * dx_scale)))
            gate = apply_gating(
                dist,
                max_dist=base_max,
                max_cost=None,
                keep_percentile=keep_percentile,
                assume_squared_cost=False,
            )
            a = hungarian_assignment(
                cm,
                gating_mask=gate,
                use_dummy=use_dummy,
                miss_cost=miss_cost,
                birth_cost=birth_cost,
                death_cost=death_cost,
            )

        elif variant == "two_stage":
            base_max = float(gating.get("max_dist", max(0.5, max_dist_factor * dx_scale)))
            cm = build_cost_matrix(
                x_pred,
                x1,
                metric=str(params.get("metric", "sqeuclidean")),
                cov=gating.get("cov"),
                huber_delta=float(gating.get("huber_delta", 1.0)),
                mahalanobis_scales=gating.get("mahalanobis_scales"),
            )
            stage1_factor = float(params.get("stage1_factor", 0.5))
            stage1_keep = float(params.get("stage1_keep_percentile", 90.0))
            stage2_keep = float(params.get("stage2_keep_percentile", 99.0))
            stage2_allow_unmatched = bool(params.get("stage2_allow_unmatched", True))

            gate1 = apply_gating(
                dist,
                max_dist=max(0.25, stage1_factor * base_max),
                max_cost=None,
                keep_percentile=stage1_keep,
                assume_squared_cost=False,
            )
            a1 = hungarian_assignment(
                cm,
                gating_mask=gate1,
                use_dummy=True,
                miss_cost=miss_cost,
                birth_cost=birth_cost,
                death_cost=death_cost,
            )
            matched_r1 = np.asarray(a1["row_ind"], dtype=int)
            matched_c1 = np.asarray(a1["col_ind"], dtype=int)

            rows_all = np.arange(cm.shape[0], dtype=int)
            cols_all = np.arange(cm.shape[1], dtype=int)
            rem_r = np.setdiff1d(rows_all, matched_r1, assume_unique=False)
            rem_c = np.setdiff1d(cols_all, matched_c1, assume_unique=False)

            matched_r = matched_r1.tolist()
            matched_c = matched_c1.tolist()
            matched_costs = cm[matched_r1, matched_c1].tolist() if matched_r1.size else []

            if rem_r.size and rem_c.size:
                cm2 = cm[np.ix_(rem_r, rem_c)]
                dist2 = dist[np.ix_(rem_r, rem_c)]
                gate2 = apply_gating(
                    dist2,
                    max_dist=base_max,
                    max_cost=None,
                    keep_percentile=stage2_keep,
                    assume_squared_cost=False,
                )
                a2 = hungarian_assignment(
                    cm2,
                    gating_mask=gate2,
                    use_dummy=stage2_allow_unmatched,
                    miss_cost=miss_cost,
                    birth_cost=birth_cost,
                    death_cost=death_cost,
                )
                if a2["row_ind"].size:
                    rr2 = rem_r[np.asarray(a2["row_ind"], dtype=int)]
                    cc2 = rem_c[np.asarray(a2["col_ind"], dtype=int)]
                    matched_r.extend(rr2.tolist())
                    matched_c.extend(cc2.tolist())
                    matched_costs.extend(cm[rr2, cc2].tolist())

            if matched_r:
                rr = np.asarray(matched_r, dtype=int)
                cc = np.asarray(matched_c, dtype=int)
                used_r = np.zeros((cm.shape[0],), dtype=bool)
                used_c = np.zeros((cm.shape[1],), dtype=bool)
                used_r[rr] = True
                used_c[cc] = True
                a = {
                    "row_ind": rr,
                    "col_ind": cc,
                    "costs": np.asarray(matched_costs, dtype=float),
                    "matched_mask": np.ones((rr.size,), dtype=bool),
                    "unmatched_t": np.flatnonzero(~used_r).astype(int),
                    "unmatched_tp1": np.flatnonzero(~used_c).astype(int),
                    "total_cost": float(np.nansum(np.asarray(matched_costs, dtype=float))),
                }
            else:
                a = {
                    "row_ind": np.array([], dtype=int),
                    "col_ind": np.array([], dtype=int),
                    "costs": np.array([], dtype=float),
                    "matched_mask": np.array([], dtype=bool),
                    "unmatched_t": np.arange(cm.shape[0], dtype=int),
                    "unmatched_tp1": np.arange(cm.shape[1], dtype=int),
                    "total_cost": 0.0,
                }

        else:
            raise ValueError(f"Unknown Hungarian CV variant '{variant}'")

        a["t"] = float(f0.time)
        a["t_next"] = float(f1.time)
        out.append(a)

    diag = {
        "sigma_median": float(np.nanmedian(np.asarray(sigma_hist, dtype=float))) if sigma_hist else float("nan"),
        "sigma_mean": float(np.nanmean(np.asarray(sigma_hist, dtype=float))) if sigma_hist else float("nan"),
    }
    return out, diag


def _associate_variant(
    frames: list[PointCloudFrame],
    variant: str,
    gating: dict[str, Any] | None,
    params: dict[str, Any] | None,
) -> AssociationResult:
    fr = sorted(frames, key=lambda f: float(f.time))
    if len(fr) < 2:
        empty_tracks = pd.DataFrame(columns=["TrackID", "Time", "PointIndex", "x", "y", "z", "Position X", "Position Y", "Position Z"])
        return AssociationResult(
            tracks_df=empty_tracks,
            links_df=pd.DataFrame(columns=["t", "src_index", "dst_index", "cost", "confidence"]),
            frame_stats_df=pd.DataFrame(columns=["t", "t_next", "n_points_t", "n_points_t_next", "n_matched"]),
            quality={"unmatched_rate": float("nan"), "mean_cost": float("nan")},
            params={"variant": variant},
        )

    g = dict(gating or {})
    p = dict(params or {})
    n_refine = int(p.get("motion_refine_iters", 2))
    dx_scale = float(p.get("dx_scale", g.get("dx_scale", 1.0)))

    vel_map: dict[tuple[float, int], np.ndarray] | None = None
    final_result: AssociationResult | None = None
    final_diag: dict[str, float] = {}
    for _ in range(max(1, n_refine)):
        assigns, diag = _run_variant_pairwise(
            fr,
            variant=variant,
            gating=g,
            params=p,
            velocity_map=vel_map,
            dx_scale=dx_scale,
        )
        result = _build_result(fr, assigns, params={"variant": variant, **p, **{f"gating_{k}": v for k, v in g.items()}})
        vel_map = _estimate_velocity_by_observation(result.tracks_df)
        final_result = result
        final_diag = diag

    assert final_result is not None
    for k, v in final_diag.items():
        if np.isfinite(v):
            final_result.quality[f"variant_{k}"] = float(v)
    return final_result


def associate_hungarian_adaptive_maxdist_cv(
    frames: list[PointCloudFrame],
    gating: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Hungarian CV with adaptive max-distance from innovation scale.

    Gating threshold is set to:
    ``max_dist = max_dist_factor * sigma_pred``,
    where ``sigma_pred`` is estimated robustly from recent matched innovations.
    """
    return _associate_variant(frames, variant="adaptive_maxdist", gating=gating, params=params)


def associate_hungarian_innovation_percentile_cv(
    frames: list[PointCloudFrame],
    gating: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Hungarian CV with percentile gating on normalized innovation.

    `gating.max_dist` is interpreted in normalized innovation units.
    """
    return _associate_variant(frames, variant="innovation_percentile", gating=gating, params=params)


def associate_hungarian_switch_penalty_cv(
    frames: list[PointCloudFrame],
    gating: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Hungarian CV with directional switch penalty in the link cost.

    Adds a penalty when candidate residual direction opposes the current track
    velocity direction.
    """
    return _associate_variant(frames, variant="switch_penalty", gating=gating, params=params)


def associate_hungarian_two_stage_cv(
    frames: list[PointCloudFrame],
    gating: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Hungarian CV in two stages: strict pass then residual pass.

    Stage 1 accepts high-confidence links using tighter gates.
    Stage 2 links remaining unmatched detections with looser gates.
    """
    return _associate_variant(frames, variant="two_stage", gating=gating, params=params)
