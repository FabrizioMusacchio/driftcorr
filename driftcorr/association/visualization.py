"""Visualization utilities for point-cloud association benchmarks.

These plots are independent from the main driftcorr visualization module but
keep a compatible matplotlib style.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import PointCloudFrame
# %% FUNCTIONS
def _frames_to_points_df(frames: list[PointCloudFrame]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for f in frames:
        for i, p in enumerate(np.asarray(f.xyz, dtype=float)):
            rows.append({"Time": float(f.time), "PointIndex": int(i), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    return pd.DataFrame(rows)


def _plot_tracks(ax, df: pd.DataFrame, color_col: str, xyz_cols: tuple[str, str, str], linewidth: float = 1.0, alpha: float = 0.7) -> None:
    if df.empty or color_col not in df.columns:
        return
    ids = sorted(df[color_col].dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    for idx, tid in enumerate(ids):
        g = df[df[color_col] == tid].sort_values("Time")
        if g.shape[0] < 2:
            """ # Avoid dot-like singletons in track panels; these are not trajectories.
            continue """
            ax.plot(g[xyz_cols[0]].to_numpy(float),
                g[xyz_cols[1]].to_numpy(float),
                g[xyz_cols[2]].to_numpy(float),
                linewidth=0, #alpha=alpha,
                marker=".", markersize=2, c="black")
        ax.plot(g[xyz_cols[0]].to_numpy(float),
                g[xyz_cols[1]].to_numpy(float),
                g[xyz_cols[2]].to_numpy(float),
                color=cmap(idx % 20),
                linewidth=linewidth,
                alpha=alpha)


def _canonical_track_df(df: pd.DataFrame | None, prefer_track_col: str = "TrackID") -> pd.DataFrame:
    """Normalize track tables to columns `TrackID`, `Time`, `x`, `y`, `z`."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["TrackID", "Time", "x", "y", "z", "PointIndex"])
    d = df.copy()

    if prefer_track_col in d.columns:
        d["TrackID"] = d[prefer_track_col]
    elif "TrackID" in d.columns:
        pass
    elif "GT_TrackID" in d.columns:
        d["TrackID"] = d["GT_TrackID"]
    elif "INF_TrackID" in d.columns:
        d["TrackID"] = d["INF_TrackID"]
    else:
        return pd.DataFrame(columns=["TrackID", "Time", "x", "y", "z", "PointIndex"])

    if {"Position X", "Position Y", "Position Z"}.issubset(d.columns):
        d["x"] = pd.to_numeric(d["Position X"], errors="coerce")
        d["y"] = pd.to_numeric(d["Position Y"], errors="coerce")
        d["z"] = pd.to_numeric(d["Position Z"], errors="coerce")
    elif {"x", "y", "z"}.issubset(d.columns):
        d["x"] = pd.to_numeric(d["x"], errors="coerce")
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
        d["z"] = pd.to_numeric(d["z"], errors="coerce")
    else:
        return pd.DataFrame(columns=["TrackID", "Time", "x", "y", "z", "PointIndex"])

    d["Time"] = pd.to_numeric(d["Time"], errors="coerce")
    d["TrackID"] = pd.to_numeric(d["TrackID"], errors="coerce")
    if "PointIndex" in d.columns:
        d["PointIndex"] = pd.to_numeric(d["PointIndex"], errors="coerce")
    else:
        d["PointIndex"] = d.groupby("Time").cumcount()
    d = d.dropna(subset=["TrackID", "Time", "x", "y", "z"]).copy()
    return d[["TrackID", "Time", "x", "y", "z", "PointIndex"]]


def _xyz_from_df(df: pd.DataFrame | None) -> np.ndarray:
    """Return Nx3 xyz array from canonical or points dataframes."""
    if df is None or df.empty or not {"x", "y", "z"}.issubset(df.columns):
        return np.empty((0, 3), dtype=float)
    xyz = df[["x", "y", "z"]].to_numpy(float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return xyz


def _set_data_box_aspect(ax, xyz: np.ndarray) -> None:
    """Set 3D box aspect from data spans if supported by the backend."""
    if xyz.size == 0:
        return
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    try:
        ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))
    except Exception:
        pass


def _set_xyz_limits(ax, xyz: np.ndarray, pad_fraction: float = 0.06) -> None:
    """Set padded cubic XYZ limits from data extents."""
    if xyz.size == 0:
        return
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-6)
    half = 0.5 * float(np.max(span)) * (1.0 + float(pad_fraction))
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_zlim(float(center[2] - half), float(center[2] + half))


def _apply_axis_geometry(
    ax,
    xyz: np.ndarray,
    use_data_box_aspect: bool,
    set_data_limits: bool,
    view_elev: float | None,
    view_azim: float | None,
) -> None:
    """Apply optional 3D geometry controls to a panel axis."""
    if set_data_limits:
        _set_xyz_limits(ax, xyz)
    if use_data_box_aspect:
        _set_data_box_aspect(ax, xyz)
    if view_elev is not None or view_azim is not None:
        elev = float(view_elev) if view_elev is not None else None
        azim = float(view_azim) if view_azim is not None else None
        kwargs = {}
        if elev is not None:
            kwargs["elev"] = elev
        if azim is not None:
            kwargs["azim"] = azim
        if kwargs:
            ax.view_init(**kwargs)


def _extract_regime_and_file_stem(title: str, out_path: Path) -> tuple[str, str]:
    """Infer regime and file stem from panel title or output filename."""
    regime = ""
    file_stem = ""
    parts = [p.strip() for p in str(title).split("|")]
    if len(parts) >= 2:
        regime = parts[0]
        file_stem = parts[1]
    stem = out_path.stem
    if not file_stem and "_panel_3d_" in stem:
        file_stem = stem.split("_panel_3d_")[0]
    return str(regime), str(file_stem)


def _find_drift_schedule_path(out_path: Path, file_stem: str) -> Path | None:
    """Find drift schedule CSV near output path, preferring run-local copies."""
    if file_stem:
        local = out_path.parent / f"{file_stem}_drift_schedule.csv"
        if local.exists():
            return local
    for anc in [out_path.parent, *list(out_path.parents)[:6]]:
        p1 = anc / f"{file_stem}_drift_schedule.csv" if file_stem else None
        p2 = anc / "drift_schedule.csv"
        if p1 is not None and p1.exists():
            return p1
        if p2.exists():
            return p2
    return None


def _apply_drift_schedule_to_gt(gt_tracks: pd.DataFrame, drift_tab: pd.DataFrame) -> pd.DataFrame:
    """Apply per-time rigid drift (rotation around z + translation) to GT tracks.

    Expected drift columns:
    - preferred: `drift_tx`, `drift_ty`, `drift_tz`, `drift_rot_deg`
    - fallback aliases: `Tx`, `Ty`, `Tz`, `rot_deg`
    """
    if gt_tracks.empty or drift_tab.empty:
        return gt_tracks
    d = gt_tracks.copy()
    s = drift_tab.copy()
    s["Time"] = pd.to_numeric(s.get("Time", np.nan), errors="coerce")
    s = s.dropna(subset=["Time"]).copy()
    if s.empty:
        return d

    def _col(tab: pd.DataFrame, names: list[str], default: float = 0.0) -> np.ndarray:
        for n in names:
            if n in tab.columns:
                return pd.to_numeric(tab[n], errors="coerce").fillna(default).to_numpy(float)
        return np.full((tab.shape[0],), float(default), dtype=float)

    s = s.sort_values("Time").drop_duplicates("Time", keep="last").reset_index(drop=True)
    tx = _col(s, ["drift_tx", "Tx", "tx"], 0.0)
    ty = _col(s, ["drift_ty", "Ty", "ty"], 0.0)
    tz = _col(s, ["drift_tz", "Tz", "tz"], 0.0)
    rot_deg = _col(s, ["drift_rot_deg", "rot_deg", "rotation_deg"], 0.0)
    sched = pd.DataFrame(
        {
            "Time": s["Time"].to_numpy(float),
            "_tx": tx,
            "_ty": ty,
            "_tz": tz,
            "_rot_deg": rot_deg,
        }
    )

    merged = d.merge(sched, on="Time", how="left")
    merged["_tx"] = pd.to_numeric(merged["_tx"], errors="coerce").fillna(0.0)
    merged["_ty"] = pd.to_numeric(merged["_ty"], errors="coerce").fillna(0.0)
    merged["_tz"] = pd.to_numeric(merged["_tz"], errors="coerce").fillna(0.0)
    merged["_rot_deg"] = pd.to_numeric(merged["_rot_deg"], errors="coerce").fillna(0.0)

    # Use a fixed center from first observed GT time, matching generation intent.
    t0 = float(pd.to_numeric(merged["Time"], errors="coerce").min())
    m0 = merged[pd.to_numeric(merged["Time"], errors="coerce") == t0]
    if m0.shape[0] > 0:
        cx = float(pd.to_numeric(m0["x"], errors="coerce").mean())
        cy = float(pd.to_numeric(m0["y"], errors="coerce").mean())
    else:
        cx = float(pd.to_numeric(merged["x"], errors="coerce").mean())
        cy = float(pd.to_numeric(merged["y"], errors="coerce").mean())

    x = pd.to_numeric(merged["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(merged["y"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(merged["z"], errors="coerce").to_numpy(float)
    ang = np.deg2rad(pd.to_numeric(merged["_rot_deg"], errors="coerce").to_numpy(float))
    c = np.cos(ang)
    sng = np.sin(ang)
    xr = c * (x - cx) - sng * (y - cy) + cx
    yr = sng * (x - cx) + c * (y - cy) + cy
    zr = z

    merged["x"] = xr + pd.to_numeric(merged["_tx"], errors="coerce").to_numpy(float)
    merged["y"] = yr + pd.to_numeric(merged["_ty"], errors="coerce").to_numpy(float)
    merged["z"] = zr + pd.to_numeric(merged["_tz"], errors="coerce").to_numpy(float)
    return merged[["TrackID", "Time", "x", "y", "z", "PointIndex"]]


def plot_3d_frames_comparison(
    frames_gt: list[PointCloudFrame] | None,
    frames_blind: list[PointCloudFrame],
    frames_inferred: list[PointCloudFrame] | None,
    out_path: str | Path,
    gt_tracks_df: pd.DataFrame | None = None,
    inferred_tracks_df: pd.DataFrame | None = None,
    title: str = "",
    use_data_box_aspect: bool = False,
    set_data_limits: bool = False,
    view_elev: float | None = None,
    view_azim: float | None = None,
) -> Path:
    """Create 2x2 qualitative association figure.

    Panels:
    1) GT tracks
    2) blind point clouds (time-colored scatter)
    3) inferred tracks
    4) GT vs inferred overlay
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    gt_pts = _frames_to_points_df(frames_gt) if frames_gt is not None else pd.DataFrame(columns=["Time", "PointIndex", "x", "y", "z"])
    blind_pts = _frames_to_points_df(frames_blind)
    inf_pts = _frames_to_points_df(frames_inferred) if frames_inferred is not None else pd.DataFrame(columns=["Time", "PointIndex", "x", "y", "z"])

    if gt_tracks_df is None and not gt_pts.empty:
        gt_tracks_df = gt_pts.copy()
        gt_tracks_df["TrackID"] = gt_tracks_df.get("PointIndex", 0)
    if inferred_tracks_df is None and not inf_pts.empty:
        inferred_tracks_df = inf_pts.copy()
        inferred_tracks_df["TrackID"] = inferred_tracks_df.get("PointIndex", 0)

    gt_tracks = _canonical_track_df(gt_tracks_df, prefer_track_col="TrackID")
    inf_tracks = _canonical_track_df(inferred_tracks_df, prefer_track_col="TrackID")
    blind_xyz = _xyz_from_df(blind_pts)
    regime_name, file_stem = _extract_regime_and_file_stem(title=title, out_path=out)
    if "with_drift" in regime_name.lower():
        drift_path = _find_drift_schedule_path(out_path=out, file_stem=file_stem)
        if drift_path is not None and drift_path.exists():
            try:
                drift_tab = pd.read_csv(drift_path)
                gt_tracks = _apply_drift_schedule_to_gt(gt_tracks, drift_tab)
            except Exception:
                # Keep backward-compatible behavior if drift schedule cannot be parsed.
                pass
    gt_xyz = _xyz_from_df(gt_tracks)
    inf_xyz = _xyz_from_df(inf_tracks)
    overlay_xyz = np.vstack([arr for arr in [gt_xyz, inf_xyz] if arr.size]) if (gt_xyz.size or inf_xyz.size) else np.empty((0, 3), dtype=float)

    try:
        fig = plt.figure(figsize=(14, 12))
        ax1 = fig.add_subplot(221, projection="3d")
        ax2 = fig.add_subplot(222, projection="3d")
        ax3 = fig.add_subplot(223, projection="3d")
        ax4 = fig.add_subplot(224, projection="3d")
    except Exception:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        ax1, ax2, ax3, ax4 = axs.ravel()

    # Panel 1: GT
    if not gt_tracks.empty:
        _plot_tracks(ax1, gt_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=1.2, alpha=0.75)
    ax1.set_title("GT tracks")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    _apply_axis_geometry(ax1, gt_xyz, use_data_box_aspect, set_data_limits, view_elev, view_azim)

    # Panel 2: blind by time
    if not blind_pts.empty:
        times = blind_pts["Time"].to_numpy(float)
        ax2.scatter(blind_pts["x"], blind_pts["y"], blind_pts["z"], c=times, cmap="viridis", s=6, alpha=0.55)
    ax2.set_title("Blind point clouds")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    _apply_axis_geometry(ax2, blind_xyz, use_data_box_aspect, set_data_limits, view_elev, view_azim)

    # Panel 3: inferred
    if not inf_tracks.empty:
        _plot_tracks(ax3, inf_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=1.4, alpha=0.8)
    ax3.set_title("Inferred pseudo tracks")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    _apply_axis_geometry(ax3, inf_xyz, use_data_box_aspect, set_data_limits, view_elev, view_azim)

    # Panel 4: overlay GT vs inferred
    if not gt_tracks.empty:
        _plot_tracks(ax4, gt_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=1.0, alpha=1.0)
    if not inf_tracks.empty:
        _plot_tracks(ax4, inf_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=2.0, alpha=0.5)
    ax4.set_title("Overlay: GT tracks (thin) vs inferred tracks (thick)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    _apply_axis_geometry(ax4, overlay_xyz, use_data_box_aspect, set_data_limits, view_elev, view_azim)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    
    
    # repeat all 4 plots above, but now each in a separate figure:
    def _safe_name(s: str) -> str:
        s = str(s).strip().lower()
        for old, new in [
            (" ", "_"),
            (":", ""),
            ("/", "_"),
            ("\\", "_"),
            ("(", ""),
            (")", ""),
            (",", ""),
        ]:
            s = s.replace(old, new)
        return s

    panel_specs = [
        (1, "GT tracks", "gt_tracks"),
        (2, "Blind point clouds", "blind_point_clouds"),
        (3, "Inferred pseudo tracks", "inferred_pseudo_tracks"),
        (4, "Overlay GT vs inferred tracks", "overlay_gt_vs_inferred"),
    ]

    CM = 1 / 2.54  # inches to cm
    for i, title_use, name_stub in panel_specs:
        fig, ax = plt.subplots(figsize=(6*CM, 6*CM), subplot_kw={"projection": "3d"})
        #print("..plotting each panel into separate files...")
        if i == 1 and not gt_tracks.empty:
            _plot_tracks(ax, gt_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=1.2, alpha=0.75)
            #print("true for i=1")
        elif i == 2 and not blind_pts.empty:
            times = blind_pts["Time"].to_numpy(float)
            ax.scatter(blind_pts["x"], blind_pts["y"], blind_pts["z"], c=times, cmap="viridis", s=6, alpha=0.55,
                       lw=0)
            # print("true for i=2")
        elif i == 3 and not inf_tracks.empty:
            _plot_tracks(ax, inf_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=1.4, alpha=0.8)
            # print("true for i=3")
        elif i == 4:
            if not gt_tracks.empty:
                _plot_tracks(ax, inf_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=1.0, alpha=1.0)
                # print("true for i=4.1")
            if not inf_tracks.empty:
                _plot_tracks(ax, gt_tracks, color_col="TrackID", xyz_cols=("x", "y", "z"), linewidth=2.0, alpha=0.5)
                # print("true for i=4.2")

        ax.set_title(title_use, pad=-55, fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.set_position([0.1, 0.1, 0.7, 0.8])
        ax.tick_params(axis="both", which="major", pad=-3)
        ax.xaxis.labelpad = -7
        ax.yaxis.labelpad = -7
        ax.zaxis.labelpad = -7
        if i == 1:
            panel_xyz = gt_xyz
        elif i == 2:
            panel_xyz = blind_xyz
        elif i == 3:
            panel_xyz = inf_xyz
        else:
            panel_xyz = overlay_xyz
        _apply_axis_geometry(ax, panel_xyz, use_data_box_aspect, set_data_limits, view_elev, view_azim)
        

        
        fig.tight_layout()

        out_i = out.parent / f"{out.stem}_panel_{i}_{_safe_name(name_stub)}.pdf"
        # print(f"stored in {out_i}")
        fig.savefig(out_i, dpi=300, transparent=True)
        plt.close(fig)
    
    return out


def _infer_dominant_gt_map(gt_tracks_df: pd.DataFrame, inferred_tracks_df: pd.DataFrame) -> dict[float, float]:
    gt = gt_tracks_df[["Time", "PointIndex", "TrackID"]].rename(columns={"TrackID": "GT_TrackID"}).copy()
    inf = inferred_tracks_df[["Time", "PointIndex", "TrackID"]].rename(columns={"TrackID": "INF_TrackID"}).copy()
    m = inf.merge(gt, on=["Time", "PointIndex"], how="inner")
    if m.empty:
        return {}
    mp: dict[float, float] = {}
    for tid, grp in m.groupby("INF_TrackID"):
        mp[float(tid)] = float(grp["GT_TrackID"].value_counts().idxmax())
    return mp


def plot_rms_over_time(gt_tracks_df: pd.DataFrame, inferred_tracks_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot RMS position mismatch over time after dominant-ID remapping."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = gt_tracks_df.copy()
    i = inferred_tracks_df.copy()
    if {"Position X", "Position Y", "Position Z"}.issubset(g.columns):
        g["x"] = g["Position X"]
        g["y"] = g["Position Y"]
        g["z"] = g["Position Z"]
    if {"Position X", "Position Y", "Position Z"}.issubset(i.columns):
        i["x"] = i["Position X"]
        i["y"] = i["Position Y"]
        i["z"] = i["Position Z"]

    mp = _infer_dominant_gt_map(g, i)
    i["GT_TrackID_mapped"] = i["TrackID"].map(mp)
    mg = g[["Time", "TrackID", "x", "y", "z"]].rename(columns={"TrackID": "GT_TrackID"})
    mi = i[["Time", "GT_TrackID_mapped", "x", "y", "z"]].rename(columns={"GT_TrackID_mapped": "GT_TrackID", "x": "x_inf", "y": "y_inf", "z": "z_inf"})
    m = mg.merge(mi, on=["Time", "GT_TrackID"], how="inner")
    if m.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.set_title("RMS over time (no overlap)")
        ax.set_xlabel("Time")
        ax.set_ylabel("RMS")
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    err = np.sqrt((m["x"] - m["x_inf"]) ** 2 + (m["y"] - m["y_inf"]) ** 2 + (m["z"] - m["z_inf"]) ** 2)
    tab = pd.DataFrame({"Time": m["Time"], "err": err})
    rms = tab.groupby("Time", as_index=False)["err"].apply(lambda v: float(np.sqrt(np.mean(np.asarray(v, dtype=float) ** 2))))
    rms = rms.rename(columns={"err": "rms"})

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(rms["Time"], rms["rms"], marker="o")
    ax.set_title("RMS error over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("RMS")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_accuracy_over_time(gt_tracks_df: pd.DataFrame, inferred_tracks_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot per-time GT-ID-consistent association accuracy.

    Accuracy is computed per GT TrackID at each time using a dominant global
    GT->inferred mapping. This penalizes fragmentation and ID switches over
    time rather than only local point overlap.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gt = gt_tracks_df[["Time", "PointIndex", "TrackID"]].rename(columns={"TrackID": "GT_TrackID"}).copy()
    inf = inferred_tracks_df[["Time", "PointIndex", "TrackID"]].rename(columns={"TrackID": "INF_TrackID"}).copy()
    m = inf.merge(gt, on=["Time", "PointIndex"], how="inner")

    acc = pd.DataFrame(columns=["Time", "accuracy"])
    if not m.empty:
        # Dominant inferred TrackID per GT TrackID.
        dom = (
            m.groupby(["GT_TrackID", "INF_TrackID"], as_index=False)
            .size()
            .sort_values(["GT_TrackID", "size", "INF_TrackID"], ascending=[True, False, True])
            .drop_duplicates(["GT_TrackID"], keep="first")
            .rename(columns={"INF_TrackID": "INF_dom"})
        )
        dom_map = {float(r["GT_TrackID"]): float(r["INF_dom"]) for _, r in dom.iterrows()}
        rows: list[dict[str, float]] = []
        for t, gt_t in m.groupby("Time"):
            tt = float(t)
            gtid_present = sorted(set(pd.to_numeric(gt_t["GT_TrackID"], errors="coerce").dropna().to_numpy(float).tolist()))
            if not gtid_present:
                continue
            correct = 0
            for gtid in gtid_present:
                inf_dom = dom_map.get(float(gtid))
                if inf_dom is None:
                    continue
                # Correct at time t if dominant inferred track is assigned to
                # the same GT identity at that time.
                ok = np.any(
                    (pd.to_numeric(gt_t["GT_TrackID"], errors="coerce").to_numpy(float) == float(gtid))
                    & (pd.to_numeric(gt_t["INF_TrackID"], errors="coerce").to_numpy(float) == float(inf_dom))
                )
                if bool(ok):
                    correct += 1
            rows.append({"Time": tt, "accuracy": float(correct / max(len(gtid_present), 1))})
        acc = pd.DataFrame(rows).sort_values("Time").reset_index(drop=True) if rows else acc

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if not acc.empty:
        ax.plot(acc["Time"], acc["accuracy"], marker="o")
    ax.set_title("Association accuracy over time (GT-ID consistent)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
# %% END
