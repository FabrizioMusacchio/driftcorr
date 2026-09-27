"""High-level association runners and optional EM coupling with drift fitters.

This module exposes three entry points:
- `associate_nodrift`: direct assignment on raw point clouds,
- `associate_with_registration`: registration-assisted assignment,
- `em_associate_and_driftfit`: alternating association/drift-fit loop.

The module remains independently usable. Driftcorr correction methods are only
imported lazily inside adapter classes when explicitly requested.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .assignment import apply_gating, build_cost_matrix, hungarian_assignment, nearest_neighbor_assignment
from .linking import link_assignments_to_tracks
from .registration import icp_rigid
from .scoring import global_association_metrics, score_links_df
from .types import AssociationResult, PointCloudFrame
# %% FRAME/POINT CONVERSION FUNCTIONS
def frames_to_points_df(frames: list[PointCloudFrame]) -> pd.DataFrame:
    """Convert frames to point table with Time/PointIndex and xyz columns."""
    rows: list[dict[str, float]] = []
    for f in sorted(frames, key=lambda v: float(v.time)):
        xyz = np.asarray(f.xyz, dtype=float)
        for i in range(xyz.shape[0]):
            rows.append(
                {
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
    return pd.DataFrame(rows)


def frames_to_tracks_df(frames: list[PointCloudFrame], temporary_ids_per_frame: bool = True) -> pd.DataFrame:
    """Convert frames to a pseudo track table.

    This helper is mainly for adapters requiring track-table input.
    """
    rows: list[dict[str, float]] = []
    for fi, f in enumerate(sorted(frames, key=lambda v: float(v.time))):
        xyz = np.asarray(f.xyz, dtype=float)
        for i in range(xyz.shape[0]):
            if temporary_ids_per_frame:
                tid = float(fi * 1_000_000 + i)
            else:
                tid = float(i)
            rows.append(
                {
                    "TrackID": tid,
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
    return pd.DataFrame(rows)


def tracks_df_to_frames(
    tracks_df: pd.DataFrame,
    xyz_cols: tuple[str, str, str] = ("x", "y", "z"),
) -> list[PointCloudFrame]:
    """Convert track table back to frame list, preserving PointIndex order."""
    if tracks_df.empty:
        return []
    d = tracks_df.copy()
    if not set(xyz_cols).issubset(d.columns):
        alt = ("Position X", "Position Y", "Position Z")
        if set(alt).issubset(d.columns):
            xyz_cols = alt
        else:
            raise ValueError(f"tracks_df missing xyz columns {xyz_cols} and Position X/Y/Z")

    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    if "PointIndex" not in d.columns:
        d["PointIndex"] = d.groupby("Time").cumcount().astype(int)
    frames: list[PointCloudFrame] = []
    for t, grp in d.groupby("Time"):
        g = grp.sort_values("PointIndex")
        xyz = g[[xyz_cols[0], xyz_cols[1], xyz_cols[2]]].to_numpy(float)
        frames.append(PointCloudFrame(time=float(t), xyz=xyz, meta={"n_points": int(xyz.shape[0])}))
    return sorted(frames, key=lambda f: float(f.time))


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
            vel[(float(tt[k]), int(pi[k]))] = v
    return vel


def _run_pairwise_assignments(
    frames: list[PointCloudFrame],
    assignment_method: str,
    metric: str,
    gating: dict[str, Any] | None,
    velocity_map: dict[tuple[float, int], np.ndarray] | None = None,
    assignment_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute assignments for all consecutive frame pairs."""
    fr = sorted(frames, key=lambda f: float(f.time))
    gating = dict(gating or {})
    aopts = dict(assignment_options or {})
    out: list[dict[str, Any]] = []

    for ti in range(len(fr) - 1):
        f0 = fr[ti]
        f1 = fr[ti + 1]
        x0 = np.asarray(f0.xyz, dtype=float)
        x1 = np.asarray(f1.xyz, dtype=float)
        dt = float(f1.time - f0.time)

        x_pred = x0.copy()
        if velocity_map is not None:
            for i in range(x0.shape[0]):
                v = velocity_map.get((float(f0.time), int(i)))
                if v is not None:
                    x_pred[i] = x0[i] + dt * v

        cost = build_cost_matrix(
            x_pred,
            x1,
            metric=metric,
            cov=gating.get("cov"),
            huber_delta=float(gating.get("huber_delta", 1.0)),
            mahalanobis_scales=gating.get("mahalanobis_scales"),
        )
        gate_mask = apply_gating(
            cost,
            max_dist=gating.get("max_dist"),
            max_cost=gating.get("max_cost"),
            keep_percentile=gating.get("keep_percentile"),
            assume_squared_cost=(str(metric).lower() == "sqeuclidean"),
        )

        if str(assignment_method).lower() == "nearest":
            a = nearest_neighbor_assignment(
                cost,
                gating_mask=gate_mask,
                use_dummy=bool(aopts.get("use_dummy", False)),
                miss_cost=aopts.get("miss_cost"),
            )
        else:
            a = hungarian_assignment(
                cost,
                gating_mask=gate_mask,
                use_dummy=bool(aopts.get("use_dummy", False)),
                miss_cost=aopts.get("miss_cost"),
                birth_cost=aopts.get("birth_cost"),
                death_cost=aopts.get("death_cost"),
            )

        a["t"] = float(f0.time)
        a["t_next"] = float(f1.time)
        a["gate_keep_fraction"] = float(np.mean(gate_mask)) if gate_mask.size else np.nan
        out.append(a)

    return out


def _build_result(
    frames: list[PointCloudFrame],
    assignments_by_t: list[dict[str, Any]],
    params: dict[str, Any],
) -> AssociationResult:
    """Build `AssociationResult` from assignment list."""
    max_gap = int(params.get("max_gap", params.get("max_frame_gap", 0)))
    tracks_df, links_df, frame_stats_df = link_assignments_to_tracks(
        frames,
        assignments_by_t,
        allow_birth_death=True,
        max_gap=max_gap,
    )
    if not links_df.empty:
        reg_rows: list[dict[str, float]] = []
        for a in assignments_by_t:
            t = float(a.get("t", np.nan))
            rr = np.asarray(a.get("row_ind", []), dtype=int)
            cc = np.asarray(a.get("col_ind", []), dtype=int)
            vv = np.asarray(a.get("costs", []), dtype=float)
            for k in range(rr.size):
                reg_rows.append(
                    {
                        "t": t,
                        "src_index": int(rr[k]),
                        "dst_index": int(cc[k]),
                        "cost_new": float(vv[k]) if k < vv.size else np.nan,
                    }
                )
        if reg_rows:
            cdf = pd.DataFrame(reg_rows)
            links_df = links_df.merge(cdf, on=["t", "src_index", "dst_index"], how="left")
            if "cost_new" in links_df.columns:
                old_cost = pd.to_numeric(links_df.get("cost", np.nan), errors="coerce")
                new_cost = pd.to_numeric(links_df["cost_new"], errors="coerce")
                links_df["cost"] = np.where(np.isfinite(new_cost), new_cost, old_cost)
                links_df = links_df.drop(columns=["cost_new"])
    links_df = score_links_df(links_df)

    quality = global_association_metrics(tracks_df, links_df, frame_stats_df)
    quality["n_tracks"] = float(tracks_df["TrackID"].nunique()) if "TrackID" in tracks_df.columns and not tracks_df.empty else 0.0
    quality["n_links"] = float(links_df.shape[0]) if not links_df.empty else 0.0

    return AssociationResult(
        tracks_df=tracks_df,
        links_df=links_df,
        frame_stats_df=frame_stats_df,
        quality=quality,
        params=params,
    )


def associate_nodrift(
    frames: list[PointCloudFrame],
    method: str = "hungarian",
    gating: dict[str, Any] | None = None,
    motion_model: str = "constant_position",
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Associate unlabeled point clouds without explicit drift pre-registration.

    Parameters
    ----------
    method:
        `hungarian` or `nearest`.
    motion_model:
        `constant_position` or `constant_velocity`.
    """
    fr = sorted(frames, key=lambda f: float(f.time))
    if len(fr) < 2:
        return AssociationResult(
            tracks_df=frames_to_tracks_df(fr, temporary_ids_per_frame=False),
            links_df=pd.DataFrame(columns=["t", "src_index", "dst_index", "cost", "confidence"]),
            frame_stats_df=pd.DataFrame(columns=["t", "t_next", "n_points_t", "n_points_t_next", "n_matched"]),
            quality={"unmatched_rate": float("nan"), "mean_cost": float("nan")},
            params={"method": method, "motion_model": motion_model},
        )

    p = dict(params or {})
    metric = str(p.get("metric", "sqeuclidean"))
    n_refine = int(p.get("motion_refine_iters", 2 if str(motion_model) == "constant_velocity" else 1))
    assignment_options = {
        "use_dummy": bool(p.get("use_dummy", False)),
        "miss_cost": p.get("miss_cost"),
        "birth_cost": p.get("birth_cost"),
        "death_cost": p.get("death_cost"),
    }

    vel_map: dict[tuple[float, int], np.ndarray] | None = None
    assignments: list[dict[str, Any]] = []
    result: AssociationResult | None = None
    for _ in range(max(1, n_refine)):
        assignments = _run_pairwise_assignments(
            fr,
            assignment_method=method,
            metric=metric,
            gating=gating,
            velocity_map=vel_map if str(motion_model) == "constant_velocity" else None,
            assignment_options=assignment_options,
        )
        result = _build_result(fr, assignments, params={"method": method, "motion_model": motion_model, **p})
        if str(motion_model) == "constant_velocity":
            vel_map = _estimate_velocity_by_observation(result.tracks_df)
    assert result is not None
    return result


def associate_with_registration(
    frames: list[PointCloudFrame],
    registration: str = "icp_rigid",
    then_assignment: str = "hungarian",
    params: dict[str, Any] | None = None,
) -> AssociationResult:
    """Associate with registration-assisted correspondences.

    For each pair (t, t+1), destination cloud is aligned to source cloud before
    assignment.
    """
    fr = sorted(frames, key=lambda f: float(f.time))
    p = dict(params or {})
    metric = str(p.get("metric", "sqeuclidean"))
    gating = dict(p.get("gating", {}))

    assignments: list[dict[str, Any]] = []
    for ti in range(len(fr) - 1):
        f0 = fr[ti]
        f1 = fr[ti + 1]
        x0 = np.asarray(f0.xyz, dtype=float)
        x1 = np.asarray(f1.xyz, dtype=float)

        if str(registration) == "icp_rigid":
            r, b, corr, res = icp_rigid(
                x=x1,
                y=x0,
                n_iters=int(p.get("icp_n_iters", 20)),
                max_corr_dist=float(p.get("icp_max_corr_dist", np.inf)),
                keep_fraction=float(p.get("icp_keep_fraction", 0.8)),
            )
            x1_aligned = (r @ x1.T).T + b[None, :]
            reg_res_med = float(np.nanmedian(res)) if res.size else np.nan
            reg_n = int(corr.shape[0])
        else:
            raise ValueError(f"Unsupported registration '{registration}'")

        cost = build_cost_matrix(x0, x1_aligned, metric=metric, cov=gating.get("cov"), huber_delta=float(gating.get("huber_delta", 1.0)))
        gate_mask = apply_gating(
            cost,
            max_dist=gating.get("max_dist"),
            max_cost=gating.get("max_cost"),
            keep_percentile=gating.get("keep_percentile"),
            assume_squared_cost=(str(metric).lower() == "sqeuclidean"),
        )
        if str(then_assignment).lower() == "nearest":
            a = nearest_neighbor_assignment(
                cost,
                gating_mask=gate_mask,
                use_dummy=bool(p.get("use_dummy", False)),
                miss_cost=p.get("miss_cost"),
            )
        else:
            a = hungarian_assignment(
                cost,
                gating_mask=gate_mask,
                use_dummy=bool(p.get("use_dummy", False)),
                miss_cost=p.get("miss_cost"),
                birth_cost=p.get("birth_cost"),
                death_cost=p.get("death_cost"),
            )

        a["t"] = float(f0.time)
        a["t_next"] = float(f1.time)
        a["registration"] = str(registration)
        a["registration_residual_median"] = reg_res_med
        a["registration_n_corr"] = reg_n
        assignments.append(a)

    res = _build_result(fr, assignments, params={"registration": registration, "assignment": then_assignment, **p})
    if assignments:
        med = [float(a.get("registration_residual_median", np.nan)) for a in assignments]
        res.quality["registration_residual_median"] = float(np.nanmedian(np.asarray(med, dtype=float)))
    return res


class DriftFitAdapter(Protocol):
    """Protocol for EM drift-model adapters."""

    def fit(self, tracks_df: pd.DataFrame, options: dict[str, Any] | None = None) -> Any:
        ...

    def apply(self, tracks_df: pd.DataFrame, model: Any, options: dict[str, Any] | None = None) -> pd.DataFrame:
        ...


@dataclass(slots=True)
class DriftModelAdapterRigidStep:
    """Adapter for `method_rigid_step` drift correction."""

    def fit(self, tracks_df: pd.DataFrame, options: dict[str, Any] | None = None) -> Any:
        from driftcorr.method_rigid_step import build_step_transforms

        opts = dict(options or {})
        step, _, times = build_step_transforms(
            tracks_df,
            keep_fraction=float(opts.get("keep_fraction", 0.7)),
            min_pairs=int(opts.get("min_pairs", 8)),
            mode=str(opts.get("rigid_mode", "rigid")),
            inlier_reestimate_iters=int(opts.get("inlier_reestimate_iters", 10)),
        )
        ref_time = float(np.min(times)) if opts.get("reference_time") is None else float(opts.get("reference_time"))
        return {"step": step, "times": times, "ref_time": ref_time}

    def apply(self, tracks_df: pd.DataFrame, model: Any, options: dict[str, Any] | None = None) -> pd.DataFrame:
        from driftcorr.method_rigid_step import apply_rigid_correction_to_tracks

        return apply_rigid_correction_to_tracks(tracks_df, model["step"], model["times"], model["ref_time"])


@dataclass(slots=True)
class DriftModelAdapterRbfToReference:
    """Adapter for `method_rbf_ref` drift correction.

    Note: this adapter is stateless in `fit`; model parameters are estimated
    inside `apply` for each call.
    """

    def fit(self, tracks_df: pd.DataFrame, options: dict[str, Any] | None = None) -> Any:
        return {"options": dict(options or {})}

    def apply(self, tracks_df: pd.DataFrame, model: Any, options: dict[str, Any] | None = None) -> pd.DataFrame:
        from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference

        opts = dict(model.get("options", {}))
        opts.update(dict(options or {}))
        df_corr, *_ = correct_positions_rbf_to_reference(
            tracks_df,
            reference_time=opts.get("reference_time"),
            m=int(opts.get("m", 30)),
            keep_fraction=float(opts.get("keep_fraction", 0.7)),
            lam=float(opts.get("lam", 1e-3)),
            sigma=float(opts.get("sigma", 180.0)),
            min_pairs=int(opts.get("min_pairs", 8)),
            seed=int(opts.get("seed", 0)),
            temporal_gamma=float(opts.get("temporal_gamma", 0.0)),
        )
        return df_corr


@dataclass(slots=True)
class DriftModelAdapterSpacetimeRbf:
    """Adapter for `method_spacetime_rbf` drift correction."""

    def fit(self, tracks_df: pd.DataFrame, options: dict[str, Any] | None = None) -> Any:
        from driftcorr.method_spacetime_rbf import fit_spacetime_rbf_model

        opts = dict(options or {})
        model, _ = fit_spacetime_rbf_model(
            tracks_df,
            reference_time=opts.get("reference_time"),
            m_centers=int(opts.get("m_centers", 50)),
            ell_space=float(opts.get("ell_space", 2.0)),
            ell_time=float(opts.get("ell_time", 0.8)),
            lam=float(opts.get("lam", 1e-3)),
            robust_keep_fraction=float(opts.get("robust_keep_fraction", 0.85)),
            robust_iters=int(opts.get("robust_iters", 2)),
            seed=int(opts.get("seed", 0)),
            space_input_mode=str(opts.get("space_input_mode", "ref_position")),
            n_refine=int(opts.get("n_refine", 2)),
            two_stage=bool(opts.get("two_stage", False)),
        )
        return model

    def apply(self, tracks_df: pd.DataFrame, model: Any, options: dict[str, Any] | None = None) -> pd.DataFrame:
        from driftcorr.method_spacetime_rbf import apply_correction

        _ = options
        return apply_correction(tracks_df, model)


def _edge_set(links_df: pd.DataFrame) -> set[tuple[float, int, int]]:
    if links_df.empty:
        return set()
    rows = links_df[["t", "src_index", "dst_index"]].copy()
    rows["t"] = pd.to_numeric(rows["t"], errors="coerce")
    rows["src_index"] = pd.to_numeric(rows["src_index"], errors="coerce")
    rows["dst_index"] = pd.to_numeric(rows["dst_index"], errors="coerce")
    out: set[tuple[float, int, int]] = set()
    for _, r in rows.dropna().iterrows():
        out.add((float(r["t"]), int(r["src_index"]), int(r["dst_index"])))
    return out


def em_associate_and_driftfit(
    frames: list[PointCloudFrame],
    drift_fit_adapter: DriftFitAdapter,
    n_iters: int = 4,
    init: AssociationResult | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[AssociationResult, list[Any]]:
    """Alternating E/M loop: association (E) and drift fit (M).

    E-step:
    - infer correspondences on drift-corrected coordinates.

    M-step:
    - fit drift model on pseudo tracks (raw coordinates with inferred IDs),
      then produce corrected coordinates for next E-step.

    Convergence stops when assignment-change fraction falls below threshold or
    mean assignment cost no longer improves.
    """
    p = dict(params or {})
    assoc_method = str(p.get("association_method", "hungarian"))
    gating = dict(p.get("gating", {}))
    motion_model = str(p.get("motion_model", "constant_position"))
    drift_options = dict(p.get("drift_options", {}))
    tol = float(p.get("assignment_change_tol", 1e-3))

    original_points = frames_to_points_df(frames)
    current_assoc = init if init is not None else associate_nodrift(
        frames,
        method=assoc_method,
        gating=gating,
        motion_model=motion_model,
        params={"metric": p.get("metric", "sqeuclidean")},
    )
    drift_models: list[Any] = []
    cost_hist: list[float] = []
    change_hist: list[float] = []

    prev_edges = _edge_set(current_assoc.links_df)
    for it in range(max(1, int(n_iters))):
        # Build raw tracks with current inferred IDs.
        ids = current_assoc.tracks_df[["Time", "PointIndex", "TrackID"]].copy()
        tracks_raw = ids.merge(original_points, on=["Time", "PointIndex"], how="left")

        model = drift_fit_adapter.fit(tracks_raw, options=drift_options)
        drift_models.append(model)
        tracks_corr = drift_fit_adapter.apply(tracks_raw, model, options=drift_options)

        if {"Position X_corr", "Position Y_corr", "Position Z_corr"}.issubset(tracks_corr.columns):
            corr_cols = ("Position X_corr", "Position Y_corr", "Position Z_corr")
            tracks_corr = tracks_corr.rename(
                columns={
                    "Position X_corr": "x_corr",
                    "Position Y_corr": "y_corr",
                    "Position Z_corr": "z_corr",
                }
            )
            corr_cols = ("x_corr", "y_corr", "z_corr")
        elif {"x_corr", "y_corr", "z_corr"}.issubset(tracks_corr.columns):
            corr_cols = ("x_corr", "y_corr", "z_corr")
        else:
            # If adapter did not generate corrected columns, reuse raw xyz.
            corr_cols = ("x", "y", "z")

        frames_corr = tracks_df_to_frames(tracks_corr[["Time", "PointIndex", *corr_cols]].rename(
            columns={corr_cols[0]: "x", corr_cols[1]: "y", corr_cols[2]: "z"}
        ))

        assoc_corr = associate_nodrift(
            frames_corr,
            method=assoc_method,
            gating=gating,
            motion_model=motion_model,
            params={"metric": p.get("metric", "sqeuclidean")},
        )

        new_edges = _edge_set(assoc_corr.links_df)
        denom = max(len(prev_edges | new_edges), 1)
        change_frac = float(len(prev_edges.symmetric_difference(new_edges)) / denom)
        change_hist.append(change_frac)

        mean_cost = float(assoc_corr.quality.get("mean_cost", np.nan))
        cost_hist.append(mean_cost)

        # Prepare next iteration on raw coordinates but updated IDs.
        ids_new = assoc_corr.tracks_df[["Time", "PointIndex", "TrackID"]].copy()
        tracks_next = ids_new.merge(original_points, on=["Time", "PointIndex"], how="left")

        # Attach corrected coords from latest model for final outputs.
        if {"Time", "PointIndex"}.issubset(tracks_corr.columns):
            cmerge_cols = ["Time", "PointIndex"]
            for c in ["x_corr", "y_corr", "z_corr", "Position X_corr", "Position Y_corr", "Position Z_corr"]:
                if c in tracks_corr.columns:
                    cmerge_cols.append(c)
            tracks_next = tracks_next.merge(tracks_corr[cmerge_cols].drop_duplicates(["Time", "PointIndex"]), on=["Time", "PointIndex"], how="left")

        current_assoc = AssociationResult(
            tracks_df=tracks_next,
            links_df=assoc_corr.links_df,
            frame_stats_df=assoc_corr.frame_stats_df,
            quality=dict(assoc_corr.quality),
            params={"em_iter": it + 1, **assoc_corr.params},
        )

        prev_edges = new_edges
        if change_frac <= tol:
            break
        if len(cost_hist) >= 2 and np.isfinite(cost_hist[-1]) and np.isfinite(cost_hist[-2]):
            if cost_hist[-1] > cost_hist[-2] and abs(cost_hist[-1] - cost_hist[-2]) < 1e-9:
                break

    current_assoc.quality["em_n_iters"] = float(len(cost_hist))
    current_assoc.quality["em_final_assignment_change"] = float(change_hist[-1]) if change_hist else float("nan")
    current_assoc.quality["em_final_mean_cost"] = float(cost_hist[-1]) if cost_hist else float("nan")
    current_assoc.params = {**current_assoc.params, "em_cost_history": cost_hist, "em_change_history": change_hist}
    return current_assoc, drift_models
# %% END