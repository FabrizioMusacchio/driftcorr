"""Run additional synthetic benchmarks and aggregate ST-RBF grid results.

This script is API-first (no subprocess CLI calls). It runs:
1) a baseline multi-method synthetic regime benchmark,
2) a ST-RBF parameter sweep over spatial/time length scales,
3) aggregation across all sweep outputs into one tidy CSV.

It does not modify existing pipeline outputs or naming conventions.
It only adds new benchmark result folders and one aggregated CSV + README.
"""

# %% IMPORTS
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from driftcorr.benchmarks import synthetic_generate, synthetic_regime_benchmark
    from driftcorr.io_imaris import list_imaris_xls_files
except ModuleNotFoundError as exc:
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from driftcorr.benchmarks import synthetic_generate, synthetic_regime_benchmark
        from driftcorr.io_imaris import list_imaris_xls_files
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import 'driftcorr'. Install the package or run from inside the cloned project."
        ) from exc
# %% PARAMETERS

ES_VALUES = [1.0, 2.0, 4.0]
ET_VALUES = [0.4, 0.8, 1.6]

GRID_DIR_PATTERN = re.compile(r"^st_rbf_grid_es(?P<es>[-+]?\d*\.?\d+)_et(?P<et>[-+]?\d*\.?\d+)$")
# %% FUNCTIONS
def _list_imaris_xls_files(folder: Path) -> list[Path]:
    """Return IMARIS xls/xlsx files in folder (non-recursive), sorted."""
    return list_imaris_xls_files(folder)


def _ensure_synthetic_dataset_exists(synthetic_dir: Path) -> None:
    """Generate a reproducible dataset once if missing; otherwise reuse existing files."""
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    existing = _list_imaris_xls_files(synthetic_dir)
    if existing:
        print(f"[INFO] Reusing existing synthetic dataset: {synthetic_dir} ({len(existing)} file(s))")
        return
    print(f"[INFO] Synthetic dataset missing. Generating once at: {synthetic_dir}")
    synthetic_generate(
        synthetic_dir=str(synthetic_dir),
        synthetic_n_files=5,
        synthetic_n_tracks=80,
        synthetic_n_times=20,
        synthetic_stationary_fraction=0.45,
        synthetic_migrating_fraction=0.35,
        synthetic_seed=42,
        synthetic_temporal_smoothness=0.9,
        synthetic_local_mode="smooth",
    )


def _find_metric_csvs_for_grid(out_dir: Path) -> dict[str, Path]:
    """Map regime -> metric CSV path for one grid output directory.

    Preferred source:
    - `<regime>_seed_metrics_long.csv` files (explicit per-regime seed metrics).
    Fallback:
    - `synthetic_regime_runs.csv` (contains all regimes, all seeds).
    """
    if not out_dir.exists():
        raise FileNotFoundError(f"Grid output directory not found: {out_dir}")

    mapping: dict[str, Path] = {}
    long_files = sorted(out_dir.glob("*_seed_metrics_long.csv"))
    for path in long_files:
        regime = path.name.replace("_seed_metrics_long.csv", "")
        if regime:
            mapping[regime] = path
    if mapping:
        return mapping

    runs_csv = out_dir / "synthetic_regime_runs.csv"
    if runs_csv.exists():
        df = pd.read_csv(runs_csv)
        if "regime" not in df.columns:
            raise ValueError(f"'regime' column missing in fallback metrics CSV: {runs_csv}")
        regimes = sorted(df["regime"].dropna().astype(str).unique().tolist())
        for regime in regimes:
            mapping[regime] = runs_csv
        if mapping:
            return mapping

    raise FileNotFoundError(
        f"No suitable per-seed metrics CSV found in {out_dir}. "
        "Expected '*_seed_metrics_long.csv' or 'synthetic_regime_runs.csv'."
    )


def _pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_and_standardize_metrics(csv_path: Path) -> pd.DataFrame:
    """Load metrics CSV and return standardized per-seed scalar metrics columns.

    Output columns:
    - regime, seed, method
    - rms_all, rms_stationary, rms_migrating, drift_bias_abs
    """
    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"Metrics CSV is empty: {csv_path}")

    # Long format: regime, method, seed, metric_name, metric_value
    if {"metric_name", "metric_value"}.issubset(raw.columns):
        required = {"regime", "method", "seed", "metric_name", "metric_value"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"Missing required long-format columns in {csv_path}: {missing}")
        wide = (
            raw.pivot_table(
                index=["regime", "seed", "method"],
                columns="metric_name",
                values="metric_value",
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        source = wide
    else:
        source = raw.copy()

    required_wide = {"regime", "seed", "method"}
    missing_wide = sorted(required_wide - set(source.columns))
    if missing_wide:
        raise ValueError(f"Missing required wide-format columns in {csv_path}: {missing_wide}")

    col_rms_all = _pick_first_existing_column(
        source,
        ["overall_rms_error", "rms_error_overall", "rms_all", "overall_rms"],
    )
    col_rms_stationary = _pick_first_existing_column(
        source,
        ["overall_rms_error_stationary", "rms_error_stationary", "rms_stationary"],
    )
    col_rms_migrating = _pick_first_existing_column(
        source,
        ["overall_rms_error_migrating", "rms_error_migrating", "rms_migrating"],
    )
    col_bias = _pick_first_existing_column(
        source,
        ["drift_bias_abs", "drift_bias_abs_mean_over_time", "drift_bias_mean_abs", "drift_bias_mag_abs", "drift_bias"],
    )

    if col_rms_all is None:
        raise ValueError(
            f"Could not find RMS-all metric column in {csv_path}. "
            "Expected one of: overall_rms_error, rms_error_overall, rms_all, overall_rms."
        )

    standardized = source[["regime", "seed", "method"]].copy()
    standardized["rms_all"] = pd.to_numeric(source[col_rms_all], errors="coerce")
    standardized["rms_stationary"] = (
        pd.to_numeric(source[col_rms_stationary], errors="coerce")
        if col_rms_stationary is not None
        else np.nan
    )
    standardized["rms_migrating"] = (
        pd.to_numeric(source[col_rms_migrating], errors="coerce")
        if col_rms_migrating is not None
        else np.nan
    )
    standardized["drift_bias_abs"] = (
        pd.to_numeric(source[col_bias], errors="coerce")
        if col_bias is not None
        else np.nan
    )
    standardized["seed"] = pd.to_numeric(standardized["seed"], errors="coerce")
    return standardized


def _aggregate_grid(out_root: Path) -> pd.DataFrame:
    """Aggregate all ST-RBF grid outputs into one tidy table."""
    if not out_root.exists():
        raise FileNotFoundError(f"Output root not found: {out_root}")

    rows: list[dict[str, float | str | int]] = []
    grid_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("st_rbf_grid_es")])
    expected_names = {f"st_rbf_grid_es{es}_et{et}" for es in ES_VALUES for et in ET_VALUES}
    existing_names = {p.name for p in grid_dirs}
    missing_names = sorted(expected_names - existing_names)
    for name in missing_names:
        print(f"[WARN] Expected grid folder not found (will skip): {out_root / name}")

    if not grid_dirs:
        raise FileNotFoundError(f"No grid folders found in {out_root} (expected names like st_rbf_grid_es*_et*).")

    for grid_dir in grid_dirs:
        m = GRID_DIR_PATTERN.match(grid_dir.name)
        if m is None:
            print(f"[WARN] Skipping folder with unexpected name format: {grid_dir.name}")
            continue

        ell_space = float(m.group("es"))
        ell_time = float(m.group("et"))

        try:
            regime_to_csv = _find_metric_csvs_for_grid(grid_dir)
        except Exception as exc:
            print(f"[WARN] Could not locate metric CSVs in {grid_dir}: {exc}")
            continue

        for regime, csv_path in regime_to_csv.items():
            try:
                metrics = _load_and_standardize_metrics(csv_path)
            except Exception as exc:
                print(f"[WARN] Failed to load/standardize metrics from {csv_path}: {exc}")
                continue

            subset = metrics[
                (metrics["method"].astype(str) == "st_rbf")
                & (metrics["regime"].astype(str) == str(regime))
            ].copy()
            if subset.empty:
                print(f"[WARN] No st_rbf rows for regime='{regime}' in {csv_path}")
                continue

            n_seeds = int(pd.to_numeric(subset["seed"], errors="coerce").dropna().nunique())
            row = {
                "regime": str(regime),
                "ell_space": ell_space,
                "ell_time": ell_time,
                "n_seeds_found": n_seeds,
                "mean_rms_all": float(np.nanmean(subset["rms_all"].to_numpy(float))),
                "std_rms_all": float(np.nanstd(subset["rms_all"].to_numpy(float), ddof=0)),
                "mean_rms_stationary": float(np.nanmean(subset["rms_stationary"].to_numpy(float))),
                "std_rms_stationary": float(np.nanstd(subset["rms_stationary"].to_numpy(float), ddof=0)),
                "mean_rms_migrating": float(np.nanmean(subset["rms_migrating"].to_numpy(float))),
                "std_rms_migrating": float(np.nanstd(subset["rms_migrating"].to_numpy(float), ddof=0)),
                "mean_drift_bias_abs": float(np.nanmean(subset["drift_bias_abs"].to_numpy(float))),
                "std_drift_bias_abs": float(np.nanstd(subset["drift_bias_abs"].to_numpy(float), ddof=0)),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No grid results could be aggregated from the available output folders.")

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["regime", "ell_space", "ell_time"], keep="last")
    required_cols = [
        "regime",
        "ell_space",
        "ell_time",
        "n_seeds_found",
        "mean_rms_all",
        "std_rms_all",
        "mean_rms_stationary",
        "std_rms_stationary",
        "mean_rms_migrating",
        "std_rms_migrating",
        "mean_drift_bias_abs",
        "std_drift_bias_abs",
    ]
    out = out[required_cols].sort_values(["regime", "ell_space", "ell_time"]).reset_index(drop=True)
    return out


# %% ADDITIONAL BENCHMARKS (Python API)
RUN_CREATE_SYNTHETIC_DATASET = False
RUN_PARAMETER_SWEEP = False
RUN_AGGREGATION = True

synthetic_dir = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark 2"
out_root = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark 2" / "driftcorr_results"

if RUN_CREATE_SYNTHETIC_DATASET:
    # 0) Ensure dataset exists or generate it once.
    # If synthetic_dir contains no IMARIS XLS files, generate it with:
    # n_files=5, n_tracks=80, n_times=20, stationary_fraction=0.45, migrating_fraction=0.35, seed=42,
    # temporal_smoothness=0.9, local_mode="smooth"
    # Otherwise reuse existing files.
    _ensure_synthetic_dataset_exists(synthetic_dir)

    # 1) Baseline regime benchmark (20 seeds), reusing an existing synthetic dataset.
    """ 
    This should resamble the CLI command:
    python -m driftcorr synthetic-regime-benchmark \
        --synthetic-dir "Synthetic IMARIS files tutorial reproducible benchmark 2" \
        --out "Synthetic IMARIS files tutorial reproducible benchmark/driftcorr_results/regimes_20seeds" \
        --reuse-existing 1 \
        --methods "rbf,rigid,local,st_rbf" \
        --synthetic-n-seeds 20
    """
    synthetic_regime_benchmark(
        synthetic_dir=str(synthetic_dir),
        out=str(out_root / "regimes_20seeds"),
        reuse_existing=1,
        methods="rbf,rigid,local,st_rbf",
        synthetic_n_seeds=20)

if RUN_PARAMETER_SWEEP:
    # 2) Parameter sweep: 3 spatial scales x 3 temporal scales.
    """ 
    This should resemble CLI commands like:
    for es in 1.0 2.0 4.0; do
        for et in 0.4 0.8 1.6; do
            python -m driftcorr synthetic-regime-benchmark \
            --synthetic-dir "Synthetic IMARIS files tutorial reproducible benchmark 2" \
            --out "Synthetic IMARIS files tutorial reproducible benchmark/driftcorr_results/st_rbf_grid_es${es}_et${et}" \
            --reuse-existing 1 \
            --methods "st_rbf" \
            --synthetic-n-seeds 10 \
            --st-ell-space ${es} \
            --st-ell-time ${et}
        done
        done
    """
    for es in [1.0, 2.0, 4.0]:
        print(f"[INFO] Starting grid sweep for ell_space={es}...")
        for et in [0.4, 0.8, 1.6]:
            print(f"[INFO]    Running grid point: ell_space={es}, ell_time={et}...")
            synthetic_regime_benchmark(
                synthetic_dir=str(synthetic_dir),
                out=str(out_root / f"st_rbf_grid_es{es}_et{et}"),
                reuse_existing=1,
                methods="st_rbf",
                synthetic_n_seeds=10,
                st_ell_space=float(es),
                st_ell_time=float(et))

if RUN_AGGREGATION:
    # 3) Aggregation: read ALL grid outputs and write ONE CSV
    """ Output CSV:
    out_root / "st_rbf_grid_aggregated.csv"
    
    It must aggregate across seeds and produce one row per:
    (regime, ell_space, ell_time)
    
    Required columns in the aggregated CSV:
    - regime
    - ell_space
    - ell_time
    - n_seeds_found
    - mean_rms_all
    - std_rms_all
    - mean_rms_stationary
    - std_rms_stationary
    - mean_rms_migrating
    - std_rms_migrating
    - mean_drift_bias_abs
    - std_drift_bias_abs
    
    The script must find the correct per-regime per-seed metric CSVs written by synthetic_regime_benchmark.
    If multiple candidate CSVs exist, choose the one that clearly contains per-seed scalar metrics for each method.
    The aggregator must:
    - only include method == "st_rbf"
    - parse ell_space and ell_time from the folder name "st_rbf_grid_es{es}_et{et}"
    - compute mean/std across seeds for each regime
    - tolerate missing seeds gracefully and report n_seeds_found
    
    Also write a small README-like text file:
    out_root / "st_rbf_grid_aggregated_README.txt"
    containing:
    - timestamp
    - synthetic_dir used
    - list of es, et values
    - where the aggregated CSV is written
    - how many grid points and regimes were successfully aggregated """
    out_root.mkdir(parents=True, exist_ok=True)
    aggregated = _aggregate_grid(out_root)

    aggregated_csv_path = out_root / "st_rbf_grid_aggregated.csv"
    aggregated.to_csv(aggregated_csv_path, index=False)

    n_grid_points = int(aggregated[["ell_space", "ell_time"]].drop_duplicates().shape[0])
    n_regimes = int(aggregated["regime"].nunique())

    readme_path = out_root / "st_rbf_grid_aggregated_README.txt"
    readme_text = "\n".join(
        [
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"synthetic_dir: {synthetic_dir}",
            f"ell_space_values: {ES_VALUES}",
            f"ell_time_values: {ET_VALUES}",
            f"aggregated_csv: {aggregated_csv_path}",
            f"grid_points_aggregated: {n_grid_points}",
            f"regimes_aggregated: {n_regimes}",
        ]
    )
    readme_path.write_text(readme_text + "\n", encoding="utf-8")

    print(f"[OK] Wrote aggregated CSV: {aggregated_csv_path}")
    print(f"[OK] Wrote aggregation README: {readme_path}")

print("[INFO] Additional benchmarks and aggregation complete.")
# %% END