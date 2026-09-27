"""Synthetic helpers for blinded point-cloud association benchmarks.

This module creates paired datasets:
- ground-truth track tables with IDs,
- blinded per-frame point clouds without IDs,
- unblinding mapping for evaluation.

The generator is deterministic under fixed seed and does not require new
external dependencies beyond numpy/pandas.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io_imaris import list_imaris_xls_files, read_imaris_position_xls
from ..synthetic_experiments import generate_synthetic_imaris_folder
from .types import GroundTruthMapping, PointCloudFrame
# %% FUNCTIONS
def synthetic_generate_blind_pair(
    synthetic_dir: str | Path,
    seed: int,
    n_files: int,
    n_tracks: int,
    n_times: int,
    regimes: list[str] | tuple[str, ...] | None,
    drift_params: dict[str, Any] | None,
    motion_params: dict[str, Any] | None,
    out_blind_dir: str | Path,
    out_gt_dir: str | Path,
    unblinding_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate paired GT/blind synthetic datasets for association evaluation.

    Parameters
    ----------
    synthetic_dir:
        Root workspace path for this generation run (kept for compatibility).
    regimes:
        List of local-mode regimes to generate.
    drift_params, motion_params:
        Optional dictionaries passed into the synthetic generator.

    Returns
    -------
    dict with `gt_root`, `blind_root`, `unblinding_path`, `files`.
    """
    _ = synthetic_dir
    rng = np.random.default_rng(int(seed))
    reg_list = list(regimes) if regimes is not None else ["nodrift_dx_sweep"]

    gt_root = Path(out_gt_dir)
    blind_root = Path(out_blind_dir)
    map_path = Path(unblinding_path)
    gt_root.mkdir(parents=True, exist_ok=True)
    blind_root.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    drift_params = dict(drift_params or {})
    motion_params = dict(motion_params or {})

    map_rows: list[dict[str, Any]] = []
    written_blind: list[Path] = []

    initial_tracks_cfg = motion_params.get("initial_tracks", None)
    density_scale_cfg = float(motion_params.get("density_scale", 1.0))
    if initial_tracks_cfg is not None:
        n_tracks_effective = max(1, int(round(float(initial_tracks_cfg) * density_scale_cfg)))
    else:
        n_tracks_effective = max(1, int(round(float(n_tracks) * density_scale_cfg)))

    # Optional fixed scene bounds in micrometers (x, y, z extents).
    scene_bounds_um = motion_params.get("scene_bounds_um", None)
    bounds_xyz = None
    if scene_bounds_um is not None:
        try:
            sx, sy, sz = [float(v) for v in scene_bounds_um]
            bounds_xyz = ((0.0, sx), (0.0, sy), (0.0, sz))
        except Exception:
            bounds_xyz = None

    for ridx, regime in enumerate(reg_list):
        regime_gt = gt_root / str(regime)
        regime_blind = blind_root / str(regime)
        regime_gt.mkdir(parents=True, exist_ok=True)
        regime_blind.mkdir(parents=True, exist_ok=True)

        # Reuse existing generation if requested.
        existing = list_imaris_xls_files(regime_gt)
        if overwrite or len(existing) < int(n_files):
            gen_kwargs = {
                "out_dir": regime_gt,
                "n_files": int(n_files),
                "n_tracks": int(n_tracks_effective),
                "n_times": int(n_times),
                "stationary_fraction": float(motion_params.get("stationary_fraction", 0.45)),
                "migrating_fraction": float(motion_params.get("migrating_fraction", 0.35)),
                "seed": int(seed + ridx * 10_000),
                "temporal_smoothness": float(motion_params.get("temporal_smoothness", 0.9)),
                "local_mode": str(regime),
                "piecewise_regions": int(drift_params.get("piecewise_regions", 3)),
                "piecewise_transition_um": float(drift_params.get("piecewise_transition_um", 90.0)),
                "global_drift_scale_xyz": tuple(drift_params.get("global_drift_scale_xyz", (2.8, 2.2, 1.1))),
                "local_lengthscale_um": float(drift_params.get("local_lengthscale_um", 220.0)),
                "local_drift_scale_um": float(drift_params.get("local_drift_scale_um", 2.4)),
                "n_rbf_centers": int(drift_params.get("n_rbf_centers", 10)),
                "measurement_noise_um": float(drift_params.get("measurement_noise_um", 0.2)),
                "drift_rotation_enable": bool(drift_params.get("drift_rotation_enable", True)),
                "drift_rotation_deg_scale": float(drift_params.get("drift_rotation_deg_scale", 2.0)),
                "dx_scale_um": float(motion_params.get("dx_scale_um", 0.8)),
                "dx_distribution": str(motion_params.get("dx_distribution", "gaussian")),
                "dx_track_smoothness": float(motion_params.get("dx_track_smoothness", 0.7)),
                "dx_mode": str(motion_params.get("dx_mode", "fixed")),
                "dx_min_um": float(motion_params.get("dx_min_um", 0.2)),
                "dx_max_um": float(motion_params.get("dx_max_um", 4.0)),
                "dx_varying_distribution": str(motion_params.get("dx_varying_distribution", "uniform")),
                "birthdeath_enable": bool(motion_params.get("birthdeath_enable", False)),
                "p_birth_late": float(np.clip(float(motion_params.get("p_birth_late", 0.35)) * density_scale_cfg, 0.0, 1.0)),
                "p_death_early": float(np.clip(float(motion_params.get("p_death_early", 0.35)) * density_scale_cfg, 0.0, 1.0)),
                "min_life_frames": int(motion_params.get("min_life_frames", 5)),
                "p_birth_random": float(motion_params.get("p_birth_random", 0.5)),
                "birth_near_dist_min_um": float(motion_params.get("birth_near_dist_min_um", 0.2)),
                "birth_near_dist_max_um": float(motion_params.get("birth_near_dist_max_um", 4.0)),
                "birth_near_pick": str(motion_params.get("birth_near_pick", "uniform_over_active_tracks")),
                "stationary_mode": str(motion_params.get("stationary_mode", "hard_zero")),
                "stationary_apply_to": str(motion_params.get("stationary_apply_to", "all_tracks")),
            }
            if bounds_xyz is not None:
                gen_kwargs["bounds_xyz"] = bounds_xyz
            generate_synthetic_imaris_folder(**gen_kwargs)

        for gt_file in list_imaris_xls_files(regime_gt):
            gt_df = read_imaris_position_xls(gt_file)
            gt_df = gt_df[["Time", "TrackID", "Position X", "Position Y", "Position Z"]].copy()
            gt_df["Time"] = pd.to_numeric(gt_df["Time"], errors="coerce")
            gt_df = gt_df.dropna(subset=["Time", "TrackID", "Position X", "Position Y", "Position Z"]).reset_index(drop=True)

            blind_rows: list[dict[str, Any]] = []
            for t, grp in gt_df.groupby("Time"):
                g = grp.copy().reset_index(drop=True)
                perm = rng.permutation(g.shape[0])
                g = g.iloc[perm].reset_index(drop=True)
                for j, row in g.iterrows():
                    blind_rows.append(
                        {
                            "Time": float(t),
                            "PointIndex": int(j),
                            "x": float(row["Position X"]),
                            "y": float(row["Position Y"]),
                            "z": float(row["Position Z"]),
                        }
                    )
                    map_rows.append(
                        {
                            "Regime": str(regime),
                            "File": str(gt_file.stem),
                            "Time": float(t),
                            "PointIndex": int(j),
                            "GT_TrackID": float(row["TrackID"]),
                        }
                    )
            blind_df = pd.DataFrame(blind_rows)
            blind_path = regime_blind / f"{gt_file.stem}_blind.csv"
            blind_df.to_csv(blind_path, index=False)
            written_blind.append(blind_path)

    map_df = pd.DataFrame(map_rows)
    map_df.to_csv(map_path, index=False)
    return {
        "gt_root": gt_root,
        "blind_root": blind_root,
        "unblinding_path": map_path,
        "files": written_blind,
    }


def synthetic_generate_blind_pair_varying_dx_birthdeath_drift(
    synthetic_dir: str | Path,
    seed: int = 42,
    n_files: int = 3,
    n_tracks: int = 80,
    n_times: int = 20,
    out_blind_dir: str | Path | None = None,
    out_gt_dir: str | Path | None = None,
    unblinding_path: str | Path | None = None,
    overwrite: bool = False,
    drift_params: dict[str, Any] | None = None,
    motion_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate combined varying-dx + birth/death + global-drift blind/GT pairs.

    This is a convenience wrapper around ``synthetic_generate_blind_pair`` that
    sets the dedicated regime key ``varying_dx_birthdeath_with_drift``.
    """
    root = Path(synthetic_dir)
    bdir = Path(out_blind_dir) if out_blind_dir is not None else (root / "blind")
    gdir = Path(out_gt_dir) if out_gt_dir is not None else (root / "gt")
    upath = Path(unblinding_path) if unblinding_path is not None else (root / "unblinding.csv")

    m = dict(motion_params or {})
    m.setdefault("dx_mode", "varying_per_track_time")
    m.setdefault("dx_min_um", 0.2)
    m.setdefault("dx_max_um", 4.0)
    m.setdefault("dx_varying_distribution", "uniform")
    m.setdefault("birthdeath_enable", True)
    m.setdefault("p_birth_late", 0.35)
    m.setdefault("p_death_early", 0.35)
    m.setdefault("min_life_frames", 5)
    m.setdefault("p_birth_random", 0.5)
    m.setdefault("birth_near_dist_min_um", 0.2)
    m.setdefault("birth_near_dist_max_um", 4.0)
    m.setdefault("birth_near_pick", "uniform_over_active_tracks")

    d = dict(drift_params or {})
    d.setdefault("drift_rotation_enable", True)
    d.setdefault("drift_rotation_deg_scale", 2.0)

    return synthetic_generate_blind_pair(
        synthetic_dir=root,
        seed=int(seed),
        n_files=int(n_files),
        n_tracks=int(n_tracks),
        n_times=int(n_times),
        regimes=["varying_dx_birthdeath_with_drift"],
        drift_params=d,
        motion_params=m,
        out_blind_dir=bdir,
        out_gt_dir=gdir,
        unblinding_path=upath,
        overwrite=bool(overwrite),
    )


def synthetic_generate_blind_pair_random_pointcloud(
    synthetic_dir: str | Path,
    seed: int = 42,
    n_files: int = 5,
    n_points: int = 80,
    n_times: int = 20,
    out_blind_dir: str | Path | None = None,
    out_gt_dir: str | Path | None = None,
    unblinding_path: str | Path | None = None,
    regimes: list[str] | tuple[str, ...] | None = None,
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate null-control blind/GT pairs with random i.i.d. point clouds.

    This generator intentionally creates *no temporal identity continuity*:
    every point in every frame receives a unique GT_TrackID. Consequently,
    ground-truth links between consecutive frames are zero by construction.
    """
    _ = synthetic_dir
    rng = np.random.default_rng(int(seed))
    reg_list = list(regimes) if regimes is not None else ["random_pointcloud_null"]

    if bounds_xyz is None:
        bounds_xyz = ((0.0, 700.0), (0.0, 700.0), (0.0, 350.0))
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_xyz

    gt_root = Path(out_gt_dir) if out_gt_dir is not None else (Path(synthetic_dir) / "gt")
    blind_root = Path(out_blind_dir) if out_blind_dir is not None else (Path(synthetic_dir) / "blind")
    map_path = Path(unblinding_path) if unblinding_path is not None else (Path(synthetic_dir) / "unblinding.csv")
    gt_root.mkdir(parents=True, exist_ok=True)
    blind_root.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    map_rows: list[dict[str, Any]] = []
    written_blind: list[Path] = []

    for ridx, regime in enumerate(reg_list):
        regime_gt = gt_root / str(regime)
        regime_gt_truth = regime_gt / "_groundtruth"
        regime_blind = blind_root / str(regime)
        regime_gt.mkdir(parents=True, exist_ok=True)
        regime_gt_truth.mkdir(parents=True, exist_ok=True)
        regime_blind.mkdir(parents=True, exist_ok=True)

        for fidx in range(1, int(n_files) + 1):
            file_stem = f"synthetic_drift_{fidx:03d}"
            blind_path = regime_blind / f"{file_stem}_blind.csv"
            gt_truth_path = regime_gt_truth / f"{file_stem}_groundtruth.csv"
            if (not overwrite) and blind_path.exists() and gt_truth_path.exists():
                continue

            gt_rows: list[dict[str, Any]] = []
            blind_rows: list[dict[str, Any]] = []
            track_counter = 1
            # deterministic but independent per (regime, file)
            frng = np.random.default_rng(int(seed + ridx * 100_000 + fidx * 10_000))
            for t in range(1, int(n_times) + 1):
                x = frng.uniform(float(xmin), float(xmax), size=int(n_points))
                y = frng.uniform(float(ymin), float(ymax), size=int(n_points))
                z = frng.uniform(float(zmin), float(zmax), size=int(n_points))
                # Unique TrackID per (t, point): no true links across time.
                gt_ids = np.arange(track_counter, track_counter + int(n_points), dtype=int)
                track_counter += int(n_points)
                for j in range(int(n_points)):
                    gt_rows.append(
                        {
                            "Time": float(t),
                            "TrackID": float(gt_ids[j]),
                            "Position X": float(x[j]),
                            "Position Y": float(y[j]),
                            "Position Z": float(z[j]),
                        }
                    )
                perm = frng.permutation(int(n_points))
                for pidx, j in enumerate(perm):
                    blind_rows.append(
                        {
                            "Time": float(t),
                            "PointIndex": int(pidx),
                            "x": float(x[j]),
                            "y": float(y[j]),
                            "z": float(z[j]),
                        }
                    )
                    map_rows.append(
                        {
                            "Regime": str(regime),
                            "File": str(file_stem),
                            "Time": float(t),
                            "PointIndex": int(pidx),
                            "GT_TrackID": float(gt_ids[j]),
                        }
                    )

            pd.DataFrame(gt_rows).to_csv(gt_truth_path, index=False)
            pd.DataFrame(blind_rows).to_csv(blind_path, index=False)
            written_blind.append(blind_path)

    pd.DataFrame(map_rows).to_csv(map_path, index=False)
    return {
        "gt_root": gt_root,
        "blind_root": blind_root,
        "unblinding_path": map_path,
        "files": written_blind,
    }


def load_blind_frames(path: str | Path) -> list[PointCloudFrame]:
    """Load blinded point clouds from CSV into `PointCloudFrame` list.

    Expected columns: `Time`, `x`, `y`, `z` (optionally `PointIndex`).
    """
    p = Path(path)
    if p.is_dir():
        candidates = sorted([q for q in p.glob("*.csv") if q.is_file()])
        if len(candidates) != 1:
            raise ValueError(f"Directory {p} must contain exactly one blind CSV, found {len(candidates)}")
        p = candidates[0]

    df = pd.read_csv(p)
    req = {"Time", "x", "y", "z"}
    if not req.issubset(df.columns):
        raise ValueError(f"Blind CSV {p} missing columns: {sorted(req - set(df.columns))}")

    frames: list[PointCloudFrame] = []
    for t, grp in df.groupby("Time"):
        g = grp.copy()
        if "PointIndex" in g.columns:
            g = g.sort_values("PointIndex")
        xyz = g[["x", "y", "z"]].to_numpy(float)
        frames.append(PointCloudFrame(time=float(t), xyz=xyz, meta={"source": str(p)}))
    return sorted(frames, key=lambda f: float(f.time))


def load_groundtruth_mapping(unblinding_path: str | Path, gt_tracks_df: pd.DataFrame | None = None) -> GroundTruthMapping:
    """Load unblinding mapping as `GroundTruthMapping` container."""
    map_df = pd.read_csv(Path(unblinding_path))
    return GroundTruthMapping(point_to_gt_df=map_df, gt_tracks_df=gt_tracks_df, meta={"source": str(unblinding_path)})
# %% END