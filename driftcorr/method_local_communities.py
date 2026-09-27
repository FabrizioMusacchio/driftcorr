"""Local community-based drift correction with spatially smooth mixing.

This method estimates drift from groups of tracks that move similarly over time.
The workflow is:
1. Compute per-track displacements relative to a reference time.
2. Build a spatial neighbor graph.
3. Weight edges by motion similarity and spatial proximity.
4. Detect communities.
5. Estimate one drift trace per community (component-wise robust median).
6. Blend community drifts in space using Gaussian weighting.
7. Correct with `x_corr(t) = x(t) - d_hat(x, t)`.

Key formulas:
- Similarity (per axis Pearson, positive part):
  `s_ij = max(0, r_x) + max(0, r_y) + max(0, r_z)`.
- Spatial edge weight:
  `w_ij = s_ij * exp(-||p_i - p_j||^2 / (2 * ell_graph^2))`.
- Field blending:
  `d_hat(p,t) = Σ_k α_k(p) d_k(t)`, with
  `α_k(p) ∝ exp(-||p-c_k||^2 / (2 * ell_field^2))`.

Controlling parameters:
- `k_neighbors` / `radius_um`: local graph connectivity.
- `ell_um_graph`: spatial penalization in graph edge weights.
- `min_edge_weight`: sparsity threshold for graph edges.
- `ell_um_field`: smoothness of spatial drift blending.

Effects:
- Larger `ell_um_field` yields smoother, less localized correction.
- Higher `min_edge_weight` produces cleaner but potentially fragmented communities.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from .drift_smoothing import detect_stage_jumps, plot_drift_timeseries, smooth_drift_timeseries
from .io_imaris import ensure_numeric_tracks
from .metrics import pivot_tracks
# %% DATA CLASSES
@dataclass
class CommunityResult:
    """Graph community partition for tracks."""

    communities: list[list[int]]
    graph: nx.Graph
    node_index_to_track_id: dict[int, float]
    track_id_to_node_index: dict[float, int]

@dataclass
class DriftModel:
    """Local drift model built from community drift traces."""

    reference_time: float
    times: np.ndarray
    cluster_centers: np.ndarray
    cluster_drift: np.ndarray
    ell_um: float
    cluster_sizes: np.ndarray
    node_to_cluster: dict[int, int]
# %% FUNCTIONS
def _compute_displacements_to_ref(
    times: np.ndarray,
    track_ids: np.ndarray,
    x: np.ndarray,
    reference_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract track displacements relative to reference time."""
    ref_time = float(reference_time)
    ref_idx = int(np.where(times == ref_time)[0][0])

    xref = x[:, ref_idx, :]
    has_ref = np.all(np.isfinite(xref), axis=1)

    finite_counts = np.sum(np.all(np.isfinite(x), axis=2), axis=1)
    ok = has_ref & (finite_counts >= 2)

    keep_ids = track_ids[ok]
    xk = x[ok]
    xrefk = xk[:, ref_idx : ref_idx + 1, :]

    d = xk - xrefk
    p0 = np.nanmean(xk, axis=1)
    return keep_ids, p0, d

def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return 0.0
    aa = a[m]
    bb = b[m]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return 0.0
    r = float(pearsonr(aa, bb)[0])
    if not np.isfinite(r):
        return 0.0
    return r

def displacement_similarity(di: np.ndarray, dj: np.ndarray) -> float:
    """Sum of positive Pearson correlations along x,y,z displacement traces."""
    rx = _corr_1d(di[:, 0], dj[:, 0])
    ry = _corr_1d(di[:, 1], dj[:, 1])
    rz = _corr_1d(di[:, 2], dj[:, 2])
    s = max(0.0, rx) + max(0.0, ry) + max(0.0, rz)
    return float(s)

def spatial_weight(pi: np.ndarray, pj: np.ndarray, ell_um: float) -> float:
    """Gaussian spatial weighting for graph edges."""
    d2 = float(np.sum((pi - pj) ** 2))
    if ell_um <= 0:
        return 1.0
    return float(np.exp(-d2 / (2.0 * ell_um * ell_um)))

def build_similarity_graph(
    p0: np.ndarray,
    d: np.ndarray,
    track_ids: np.ndarray,
    k_neighbors: int = 12,
    radius_um: Optional[float] = None,
    ell_um: float = 80.0,
    min_edge_weight: float = 0.25,
    max_pairs_per_node: Optional[int] = None,
) -> CommunityResult:
    """Construct graph over tracks with weights from similarity x proximity."""
    n = p0.shape[0]
    g = nx.Graph()

    node_index_to_track_id = {i: float(track_ids[i]) for i in range(n)}
    track_id_to_node_index = {float(track_ids[i]): i for i in range(n)}

    for i in range(n):
        g.add_node(i)

    tree = cKDTree(p0)

    if radius_um is not None:
        neighbor_lists = tree.query_ball_point(p0, r=float(radius_um))
        for i in range(n):
            neigh = neighbor_lists[i]
            if i in neigh:
                neigh.remove(i)
            if max_pairs_per_node is not None and len(neigh) > max_pairs_per_node:
                neigh = neigh[:max_pairs_per_node]
            for j in neigh:
                if j <= i:
                    continue
                sim = displacement_similarity(d[i], d[j])
                if sim <= 0:
                    continue
                w = spatial_weight(p0[i], p0[j], ell_um=ell_um) * sim
                if w >= min_edge_weight:
                    g.add_edge(i, j, weight=float(w))
    else:
        k = int(max(1, k_neighbors))
        _, idxs = tree.query(p0, k=min(k + 1, n))
        for i in range(n):
            neigh = idxs[i].tolist()
            if i in neigh:
                neigh.remove(i)
            if max_pairs_per_node is not None and len(neigh) > max_pairs_per_node:
                neigh = neigh[:max_pairs_per_node]
            for j in neigh:
                if j <= i:
                    continue
                sim = displacement_similarity(d[i], d[j])
                if sim <= 0:
                    continue
                w = spatial_weight(p0[i], p0[j], ell_um=ell_um) * sim
                if w >= min_edge_weight:
                    g.add_edge(i, j, weight=float(w))

    communities = None

    if hasattr(nx.algorithms.community, "louvain_communities"):
        try:
            communities = list(
                nx.algorithms.community.louvain_communities(g, weight="weight", resolution=1.0, seed=0)
            )
        except Exception:
            communities = None

    if communities is None:
        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))
        except Exception:
            communities = [set([i]) for i in range(n)]

    comm_lists = [sorted(list(c)) for c in communities if len(c) > 0]
    comm_lists = sorted(comm_lists, key=lambda x: (-len(x), x[0]))

    return CommunityResult(
        communities=comm_lists,
        graph=g,
        node_index_to_track_id=node_index_to_track_id,
        track_id_to_node_index=track_id_to_node_index,
    )

def cluster_drift_traces(
    p0: np.ndarray,
    d: np.ndarray,
    communities: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Compute community centers and median drift traces."""
    k = len(communities)
    t = d.shape[1]
    centers = np.zeros((k, 3), dtype=float)
    drift = np.zeros((k, t, 3), dtype=float)
    sizes = np.zeros((k,), dtype=int)
    node_to_cluster: dict[int, int] = {}

    for ci, nodes in enumerate(communities):
        nodes = list(nodes)
        sizes[ci] = len(nodes)
        for n in nodes:
            node_to_cluster[int(n)] = ci

        centers[ci] = np.mean(p0[nodes], axis=0)

        for ti in range(t):
            for ax in range(3):
                v = d[nodes, ti, ax]
                vv = v[np.isfinite(v)]
                drift[ci, ti, ax] = float(np.median(vv)) if vv.size else 0.0

    return centers, drift, sizes, node_to_cluster

def predict_drift_at_positions(
    p: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_drift_t: np.ndarray,
    ell_um: float,
) -> np.ndarray:
    """Predict local drift by Gaussian blending of cluster drift vectors."""
    n = p.shape[0]
    k = cluster_centers.shape[0]
    out = np.zeros((n, 3), dtype=float)

    if k == 0:
        return out

    if ell_um <= 0:
        out[:] = np.mean(cluster_drift_t, axis=0)
        return out

    diff = p[:, None, :] - cluster_centers[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    w = np.exp(-d2 / (2.0 * ell_um * ell_um))
    wsum = np.sum(w, axis=1, keepdims=True)
    wsum[wsum <= 1e-12] = 1.0
    wn = w / wsum

    out = wn @ cluster_drift_t
    return out

def build_drift_model(
    times: np.ndarray,
    reference_time: float,
    p0: np.ndarray,
    d: np.ndarray,
    comm_res: CommunityResult,
    ell_um: float,
) -> DriftModel:
    """Build a local drift model from communities."""
    centers, drift, sizes, node_to_cluster = cluster_drift_traces(p0, d, comm_res.communities)
    return DriftModel(
        reference_time=float(reference_time),
        times=np.array(times, dtype=float),
        cluster_centers=centers,
        cluster_drift=drift,
        ell_um=float(ell_um),
        cluster_sizes=sizes,
        node_to_cluster=node_to_cluster,
    )

def apply_drift_correction(
    df: pd.DataFrame,
    model: DriftModel,
    use_track_mean_position: bool = True,
) -> pd.DataFrame:
    """Apply local drift correction to all rows in a dataframe."""
    dfc = df.copy()

    times = model.times
    t_to_idx = {float(t): i for i, t in enumerate(times)}

    if use_track_mean_position:
        ptrack = dfc.groupby("TrackID")[["Position X", "Position Y", "Position Z"]].mean().reset_index()
        track_to_p = {
            float(r.TrackID): np.array([r["Position X"], r["Position Y"], r["Position Z"]], dtype=float)
            for _, r in ptrack.iterrows()
        }
        p = np.vstack([track_to_p.get(float(tid), np.array([np.nan, np.nan, np.nan])) for tid in dfc["TrackID"].to_numpy(float)])
    else:
        p = dfc[["Position X", "Position Y", "Position Z"]].to_numpy(float)

    x = dfc[["Position X", "Position Y", "Position Z"]].to_numpy(float)
    trow = dfc["Time"].to_numpy(float)

    xcorr = np.full_like(x, np.nan, dtype=float)

    for tval in times:
        mask = trow == float(tval)
        if not np.any(mask):
            continue
        idx = int(t_to_idx[float(tval)])
        drift_t = model.cluster_drift[:, idx, :]

        pm = p[mask]
        xm = x[mask]
        drift_pred = predict_drift_at_positions(pm, model.cluster_centers, drift_t, model.ell_um)
        xcorr[mask] = xm - drift_pred

    dfc["Position X_corr"] = xcorr[:, 0]
    dfc["Position Y_corr"] = xcorr[:, 1]
    dfc["Position Z_corr"] = xcorr[:, 2]
    return dfc

def summarize_communities(
    p0: np.ndarray,
    d: np.ndarray,
    comm_res: CommunityResult,
) -> pd.DataFrame:
    """Generate a summary table per community."""
    rows: list[dict[str, float | int]] = []
    for k, nodes in enumerate(comm_res.communities):
        nodes = list(nodes)
        if len(nodes) < 2:
            med_sim = 0.0
        else:
            sims = []
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    i = nodes[a]
                    j = nodes[b]
                    sims.append(displacement_similarity(d[i], d[j]))
            med_sim = float(np.median(sims)) if len(sims) else 0.0

        pk = p0[nodes]
        center = np.mean(pk, axis=0)
        spread = float(np.median(np.linalg.norm(pk - center[None, :], axis=1)))

        rows.append(
            {
                "cluster": k,
                "size": int(len(nodes)),
                "median_similarity_sumcorr": med_sim,
                "median_spatial_spread_um": spread,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
            }
        )

    return pd.DataFrame(rows).sort_values(["size", "median_similarity_sumcorr"], ascending=[False, False]).reset_index(
        drop=True
    )

def plot_cluster_drifts(model: DriftModel, out_prefix: str | Path) -> None:
    """Save per-axis community drift traces."""
    out_prefix = str(out_prefix)
    k = model.cluster_drift.shape[0]
    t = model.times

    for ax, name in enumerate(["x", "y", "z"]):
        fig = plt.figure()
        for ci in range(k):
            plt.plot(t, model.cluster_drift[ci, :, ax], alpha=0.8)
        plt.xlabel("Time")
        plt.ylabel(f"drift {name} (um)")
        plt.title(f"Cluster drift traces, axis {name}")
        plt.tight_layout()
        plt.savefig(out_prefix + f"_cluster_drifts_{name}.png", dpi=300)
        plt.close(fig)

def run_local_drift_correction_for_file(
    df: pd.DataFrame,
    out_prefix: str | Path,
    reference_time: Optional[float] = None,
    k_neighbors: int = 12,
    radius_um: Optional[float] = None,
    ell_um_graph: float = 80.0,
    ell_um_field: float = 120.0,
    min_edge_weight: float = 0.25,
    max_pairs_per_node: Optional[int] = 50,
    use_track_mean_position: bool = True,
) -> tuple[pd.DataFrame, DriftModel, CommunityResult, pd.DataFrame, pd.DataFrame]:
    """Run local community-based correction and save outputs."""
    out_prefix = str(out_prefix)
    df = ensure_numeric_tracks(df)
    times, track_ids, x = pivot_tracks(df)

    if reference_time is None:
        reference_time = float(times.min())
    reference_time = float(reference_time)

    keep_ids, p0, d = _compute_displacements_to_ref(times, track_ids, x, reference_time=reference_time)

    if keep_ids.size < 5:
        raise RuntimeError("Too few usable tracks with reference_time present. Cannot build drift communities.")

    comm_res = build_similarity_graph(
        p0=p0,
        d=d,
        track_ids=keep_ids,
        k_neighbors=k_neighbors,
        radius_um=radius_um,
        ell_um=ell_um_graph,
        min_edge_weight=min_edge_weight,
        max_pairs_per_node=max_pairs_per_node,
    )

    comm_table = summarize_communities(p0, d, comm_res)

    model = build_drift_model(
        times=times,
        reference_time=reference_time,
        p0=p0,
        d=d,
        comm_res=comm_res,
        ell_um=ell_um_field,
    )

    df_corr = apply_drift_correction(df, model, use_track_mean_position=use_track_mean_position)

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(out_prefix + "_positions_corrected_localdrift.csv", index=False)
    comm_table.to_csv(out_prefix + "_localdrift_cluster_summary.csv", index=False)

    rows: list[dict[str, float | int]] = []
    for k in range(model.cluster_drift.shape[0]):
        for ti, tval in enumerate(model.times):
            rows.append(
                {
                    "cluster": int(k),
                    "time": float(tval),
                    "drift_x": float(model.cluster_drift[k, ti, 0]),
                    "drift_y": float(model.cluster_drift[k, ti, 1]),
                    "drift_z": float(model.cluster_drift[k, ti, 2]),
                    "center_x": float(model.cluster_centers[k, 0]),
                    "center_y": float(model.cluster_centers[k, 1]),
                    "center_z": float(model.cluster_centers[k, 2]),
                    "cluster_size": int(model.cluster_sizes[k]),
                }
            )
    drift_table = pd.DataFrame(rows)
    drift_table.to_csv(out_prefix + "_localdrift_cluster_drifts.csv", index=False)

    plot_cluster_drifts(model, out_prefix)

    return df_corr, model, comm_res, comm_table, drift_table

def _estimate_time_drift_from_corrected(
    df_orig: pd.DataFrame,
    df_corr: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-time drift vectors from median(original - corrected)."""
    d0 = ensure_numeric_tracks(df_orig)
    dc = ensure_numeric_tracks(df_corr)
    if d0.shape[0] != dc.shape[0]:
        merged = d0.merge(dc, on=["TrackID", "Time"], how="inner", suffixes=("_orig", "_corr"))
        merged["dx"] = merged["Position X_orig"].to_numpy(float) - merged["Position X_corr"].to_numpy(float)
        merged["dy"] = merged["Position Y_orig"].to_numpy(float) - merged["Position Y_corr"].to_numpy(float)
        merged["dz"] = merged["Position Z_orig"].to_numpy(float) - merged["Position Z_corr"].to_numpy(float)
        tab = (
            merged.groupby("Time", as_index=False)
            .agg(dx=("dx", "median"), dy=("dy", "median"), dz=("dz", "median"))
            .sort_values("Time")
            .reset_index(drop=True)
        )
    else:
        tab = pd.DataFrame(
            {
                "Time": d0["Time"].to_numpy(float),
                "dx": d0["Position X"].to_numpy(float) - dc["Position X_corr"].to_numpy(float),
                "dy": d0["Position Y"].to_numpy(float) - dc["Position Y_corr"].to_numpy(float),
                "dz": d0["Position Z"].to_numpy(float) - dc["Position Z_corr"].to_numpy(float),
            }
        )
        tab = (
            tab.groupby("Time", as_index=False)
            .agg(dx=("dx", "median"), dy=("dy", "median"), dz=("dz", "median"))
            .sort_values("Time")
            .reset_index(drop=True)
        )
    return tab["Time"].to_numpy(float), tab[["dx", "dy", "dz"]].to_numpy(float)

def _smooth_drift_piecewise(
    times: np.ndarray,
    drift_raw: np.ndarray,
    method: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    detect_jumps = bool(params.get("detect_stage_jumps", True))
    if not detect_jumps or drift_raw.shape[0] <= 2:
        return smooth_drift_timeseries(times, drift_raw, method=method, params=params), np.zeros((drift_raw.shape[0],), dtype=bool)

    jump_mask = detect_stage_jumps(times, drift_raw, method="mad_threshold", params=params)
    if not np.any(jump_mask):
        return smooth_drift_timeseries(times, drift_raw, method=method, params=params), jump_mask

    out = np.array(drift_raw, dtype=float, copy=True)
    split_idx = [0] + [int(i) for i in np.where(jump_mask)[0].tolist()] + [int(drift_raw.shape[0])]
    for a, b in zip(split_idx[:-1], split_idx[1:]):
        if b - a <= 2:
            continue
        out[a:b] = smooth_drift_timeseries(times[a:b], drift_raw[a:b], method=method, params=params)
    return out, jump_mask

def run_local_drift_correction_for_file_with_optional_smoothing(
    df: pd.DataFrame,
    out_prefix: str | Path,
    reference_time: Optional[float] = None,
    k_neighbors: int = 12,
    radius_um: Optional[float] = None,
    ell_um_graph: float = 80.0,
    ell_um_field: float = 120.0,
    min_edge_weight: float = 0.25,
    max_pairs_per_node: Optional[int] = 50,
    use_track_mean_position: bool = True,
    smoothing_enabled: bool = False,
    smoothing_method: str = "savgol",
    smoothing_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, DriftModel, CommunityResult, pd.DataFrame, pd.DataFrame]:
    """Run local correction with optional posthoc smoothing of global drift trace.

    When smoothing is disabled this is a strict passthrough to
    ``run_local_drift_correction_for_file``.
    """

    df_corr, model, comm_res, comm_table, drift_table = run_local_drift_correction_for_file(
        df=df,
        out_prefix=out_prefix,
        reference_time=reference_time,
        k_neighbors=k_neighbors,
        radius_um=radius_um,
        ell_um_graph=ell_um_graph,
        ell_um_field=ell_um_field,
        min_edge_weight=min_edge_weight,
        max_pairs_per_node=max_pairs_per_node,
        use_track_mean_position=use_track_mean_position,
    )
    if not bool(smoothing_enabled):
        return df_corr, model, comm_res, comm_table, drift_table

    params = {} if smoothing_params is None else dict(smoothing_params)
    tvals, drift_raw = _estimate_time_drift_from_corrected(df, df_corr)
    drift_smooth, jump_mask = _smooth_drift_piecewise(
        tvals,
        drift_raw,
        method=smoothing_method,
        params=params,
    )

    # Drift-only smoothing adjustment:
    # x_corr_s = x_corr_raw + (d_raw(t) - d_smooth(t))
    df_sm = df_corr.copy()
    tt = df_sm["Time"].to_numpy(float)
    map_raw = {float(t): drift_raw[i] for i, t in enumerate(tvals)}
    map_sm = {float(t): drift_smooth[i] for i, t in enumerate(tvals)}
    delta = np.vstack([map_raw.get(float(t), np.zeros(3)) - map_sm.get(float(t), np.zeros(3)) for t in tt])
    df_sm["Position X_corr"] = df_sm["Position X_corr"].to_numpy(float) + delta[:, 0]
    df_sm["Position Y_corr"] = df_sm["Position Y_corr"].to_numpy(float) + delta[:, 1]
    df_sm["Position Z_corr"] = df_sm["Position Z_corr"].to_numpy(float) + delta[:, 2]

    out_prefix = Path(out_prefix)
    tab = pd.DataFrame(
        {
            "time": tvals,
            "dx_raw": drift_raw[:, 0],
            "dy_raw": drift_raw[:, 1],
            "dz_raw": drift_raw[:, 2],
            "dx_smooth": drift_smooth[:, 0],
            "dy_smooth": drift_smooth[:, 1],
            "dz_smooth": drift_smooth[:, 2],
            "jump_detected": jump_mask.astype(int),
        }
    )
    tab.to_csv(Path(str(out_prefix) + "_local_smooth_drift_timeseries.csv"), index=False)
    plot_drift_timeseries(tvals, drift_raw, drift_smooth, out_prefix=out_prefix, tag="local_smooth")

    df_sm.to_csv(str(out_prefix) + "_positions_corrected_localdrift_smooth.csv", index=False)
    return df_sm, model, comm_res, comm_table, drift_table
# %% END