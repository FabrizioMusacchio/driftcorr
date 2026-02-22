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

import matplotlib.pyplot as plt
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
STATIONARY_AB_CANDIDATES = [
    {"name": "fast", "ell_space": 4.0, "ell_time": 0.4},
    {"name": "smooth", "ell_space": 1.0, "ell_time": 1.6},
]

GRID_DIR_PATTERN = re.compile(r"^st_rbf_grid_es(?P<es>[-+]?\d*\.?\d+)_et(?P<et>[-+]?\d*\.?\d+)$")
STATIONARY_AB_DIR_PATTERN = re.compile(
    r"^st_rbf_stationary_ab_es(?P<es>[-+]?\d*\.?\d+)_et(?P<et>[-+]?\d*\.?\d+)_subset(?P<subset>[01])$")

# define paths for synthetic dataset and grid outputs (will be created if missing):
synthetic_dir = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark 2"
out_root = PROJECT_ROOT / "Synthetic IMARIS files tutorial reproducible benchmark 2" / "driftcorr_results"
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


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return mean/std on finite values, or (nan, nan) if none are finite."""
    vals = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("nan"), float("nan")
    cur = vals[mask]
    return float(np.mean(cur)), float(np.std(cur, ddof=0))


def _drift_bias_from_per_seed_files(grid_dir: Path, regime: str, seed: int, method: str = "st_rbf") -> float:
    """Load per-seed drift-bias CSV(s) and return mean drift-bias magnitude.

    Expected files (inside run directory):
    - `*_gt_drift_bias_over_time_<method>.csv`
    """
    run_dir = grid_dir / f"{regime}_seed_{int(seed):03d}"
    if not run_dir.exists():
        return float("nan")
    files = sorted(run_dir.glob(f"*/*_gt_drift_bias_over_time_{method}.csv"))
    if not files:
        return float("nan")

    per_file_vals: list[float] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "drift_bias_norm" not in df.columns:
            continue
        m, _ = _safe_mean_std(pd.to_numeric(df["drift_bias_norm"], errors="coerce").to_numpy(float))
        if np.isfinite(m):
            per_file_vals.append(float(m))

    if not per_file_vals:
        return float("nan")
    m, _ = _safe_mean_std(np.asarray(per_file_vals, dtype=float))
    return float(m)


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

            # Fallback: derive drift-bias scalar from per-seed per-time bias files.
            if "drift_bias_abs" in subset.columns:
                need_bias = ~np.isfinite(pd.to_numeric(subset["drift_bias_abs"], errors="coerce").to_numpy(float))
                if bool(np.any(need_bias)):
                    for idx in subset.index[need_bias]:
                        seed_val = pd.to_numeric(subset.loc[idx, "seed"], errors="coerce")
                        if not np.isfinite(seed_val):
                            continue
                        subset.loc[idx, "drift_bias_abs"] = _drift_bias_from_per_seed_files(
                            grid_dir=grid_dir,
                            regime=str(regime),
                            seed=int(seed_val),
                            method="st_rbf",
                        )

            n_seeds = int(pd.to_numeric(subset["seed"], errors="coerce").dropna().nunique())
            mean_rms_all, std_rms_all = _safe_mean_std(subset["rms_all"].to_numpy(float))
            mean_rms_stationary, std_rms_stationary = _safe_mean_std(subset["rms_stationary"].to_numpy(float))
            mean_rms_migrating, std_rms_migrating = _safe_mean_std(subset["rms_migrating"].to_numpy(float))
            mean_drift_bias_abs, std_drift_bias_abs = _safe_mean_std(subset["drift_bias_abs"].to_numpy(float))
            row = {
                "regime": str(regime),
                "ell_space": ell_space,
                "ell_time": ell_time,
                "n_seeds_found": n_seeds,
                "mean_rms_all": mean_rms_all,
                "std_rms_all": std_rms_all,
                "mean_rms_stationary": mean_rms_stationary,
                "std_rms_stationary": std_rms_stationary,
                "mean_rms_migrating": mean_rms_migrating,
                "std_rms_migrating": std_rms_migrating,
                "mean_drift_bias_abs": mean_drift_bias_abs,
                "std_drift_bias_abs": std_drift_bias_abs,
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


def _run_stationary_ab_tests(synthetic_dir: Path, out_root: Path) -> None:
    """Run ST-RBF stationary subset A/B tests on fixed candidate ell settings."""
    for cand in STATIONARY_AB_CANDIDATES:
        es = float(cand["ell_space"])
        et = float(cand["ell_time"])
        for use_subset in [0, 1]:
            out_dir = out_root / f"st_rbf_stationary_ab_es{es}_et{et}_subset{use_subset}"
            print(
                "[INFO] Running stationary A/B config: "
                f"ell_space={es}, ell_time={et}, use_stationary_subset={use_subset}"
            )
            synthetic_regime_benchmark(
                synthetic_dir=str(synthetic_dir),
                out=str(out_dir),
                reuse_existing=1,
                methods="st_rbf",
                synthetic_n_seeds=10,
                st_ell_space=es,
                st_ell_time=et,
                use_stationary_subset=bool(use_subset),
            )


def _aggregate_stationary_ab(out_root: Path) -> pd.DataFrame:
    """Aggregate stationary A/B experiment outputs into one summary table."""
    rows: list[dict[str, float | str | int]] = []
    ab_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("st_rbf_stationary_ab_es")])
    if not ab_dirs:
        raise FileNotFoundError(
            f"No stationary A/B folders found in {out_root} "
            "(expected names like st_rbf_stationary_ab_es*_et*_subset*)."
        )

    for run_dir in ab_dirs:
        m = STATIONARY_AB_DIR_PATTERN.match(run_dir.name)
        if m is None:
            print(f"[WARN] Skipping unexpected stationary A/B folder name: {run_dir.name}")
            continue
        ell_space = float(m.group("es"))
        ell_time = float(m.group("et"))
        use_subset = int(m.group("subset"))

        try:
            regime_to_csv = _find_metric_csvs_for_grid(run_dir)
        except Exception as exc:
            print(f"[WARN] Could not locate metrics for stationary A/B run {run_dir}: {exc}")
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
                continue

            need_bias = ~np.isfinite(pd.to_numeric(subset["drift_bias_abs"], errors="coerce").to_numpy(float))
            if bool(np.any(need_bias)):
                for idx in subset.index[need_bias]:
                    seed_val = pd.to_numeric(subset.loc[idx, "seed"], errors="coerce")
                    if not np.isfinite(seed_val):
                        continue
                    subset.loc[idx, "drift_bias_abs"] = _drift_bias_from_per_seed_files(
                        grid_dir=run_dir,
                        regime=str(regime),
                        seed=int(seed_val),
                        method="st_rbf",
                    )

            n_seeds = int(pd.to_numeric(subset["seed"], errors="coerce").dropna().nunique())
            mean_rms_all, std_rms_all = _safe_mean_std(subset["rms_all"].to_numpy(float))
            mean_rms_stationary, std_rms_stationary = _safe_mean_std(subset["rms_stationary"].to_numpy(float))
            mean_rms_migrating, std_rms_migrating = _safe_mean_std(subset["rms_migrating"].to_numpy(float))
            mean_drift_bias_abs, std_drift_bias_abs = _safe_mean_std(subset["drift_bias_abs"].to_numpy(float))

            rows.append(
                {
                    "regime": str(regime),
                    "ell_space": ell_space,
                    "ell_time": ell_time,
                    "use_stationary_subset": use_subset,
                    "n_seeds_found": n_seeds,
                    "mean_rms_all": mean_rms_all,
                    "std_rms_all": std_rms_all,
                    "mean_rms_stationary": mean_rms_stationary,
                    "std_rms_stationary": std_rms_stationary,
                    "mean_rms_migrating": mean_rms_migrating,
                    "std_rms_migrating": std_rms_migrating,
                    "mean_drift_bias_abs": mean_drift_bias_abs,
                    "std_drift_bias_abs": std_drift_bias_abs,
                }
            )

    if not rows:
        raise RuntimeError("No stationary A/B results could be aggregated.")

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["regime", "ell_space", "ell_time", "use_stationary_subset"], keep="last")
    required_cols = [
        "regime",
        "ell_space",
        "ell_time",
        "use_stationary_subset",
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
    out = out[required_cols].sort_values(["regime", "ell_space", "ell_time", "use_stationary_subset"]).reset_index(drop=True)
    return out


def _write_stationary_ab_report(summary: pd.DataFrame, out_md_path: Path) -> None:
    """Write a short markdown ranking report for stationary A/B configurations."""
    lines: list[str] = []
    lines.append("# ST-RBF Stationary Subset A/B Report")
    lines.append("")
    lines.append(f"- timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- rows: {int(summary.shape[0])}")
    lines.append("")

    for regime in sorted(summary["regime"].astype(str).unique().tolist()):
        cur = summary[summary["regime"].astype(str) == regime].copy()
        if cur.empty:
            continue
        lines.append(f"## Regime: {regime}")
        lines.append("")

        cur["cfg"] = (
            "es="
            + cur["ell_space"].map(lambda v: f"{float(v):.1f}")
            + ", et="
            + cur["ell_time"].map(lambda v: f"{float(v):.1f}")
            + ", subset="
            + cur["use_stationary_subset"].astype(int).astype(str)
        )

        rms_rank = cur.assign(_sort=cur["mean_rms_all"].where(np.isfinite(cur["mean_rms_all"]), np.inf)).sort_values("_sort")
        bias_rank = cur.assign(
            _sort=cur["mean_drift_bias_abs"].where(np.isfinite(cur["mean_drift_bias_abs"]), np.inf)
        ).sort_values("_sort")

        lines.append("### Rank by mean_rms_all (lower is better)")
        lines.append("")
        lines.append("| rank | config | n_seeds_found | mean_rms_all | std_rms_all |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, (_, r) in enumerate(rms_rank.iterrows(), start=1):
            lines.append(
                f"| {i} | {r['cfg']} | {int(r['n_seeds_found'])} | "
                f"{float(r['mean_rms_all']):.6f} | {float(r['std_rms_all']):.6f} |"
            )
        lines.append("")

        lines.append("### Rank by mean_drift_bias_abs (lower is better)")
        lines.append("")
        lines.append("| rank | config | n_seeds_found | mean_drift_bias_abs | std_drift_bias_abs |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, (_, r) in enumerate(bias_rank.iterrows(), start=1):
            mean_bias = float(r["mean_drift_bias_abs"]) if np.isfinite(r["mean_drift_bias_abs"]) else float("nan")
            std_bias = float(r["std_drift_bias_abs"]) if np.isfinite(r["std_drift_bias_abs"]) else float("nan")
            lines.append(
                f"| {i} | {r['cfg']} | {int(r['n_seeds_found'])} | "
                f"{mean_bias:.6f} | {std_bias:.6f} |"
            )
        lines.append("")

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    highlight: tuple[float, float] = (4.0, 0.4),
) -> bool:
    """Plot a heatmap with x=ell_time, y=ell_space from an aggregated dataframe."""
    required = {"ell_space", "ell_time", value_col}
    if not required.issubset(df.columns):
        print(f"[WARN] Heatmap skipped ({out_path.name}): missing columns {sorted(required - set(df.columns))}")
        return False

    cur = df[["ell_space", "ell_time", value_col]].copy()
    cur["ell_space"] = pd.to_numeric(cur["ell_space"], errors="coerce")
    cur["ell_time"] = pd.to_numeric(cur["ell_time"], errors="coerce")
    cur[value_col] = pd.to_numeric(cur[value_col], errors="coerce")
    cur = cur.dropna(subset=["ell_space", "ell_time"])
    if cur.empty:
        print(f"[WARN] Heatmap skipped ({out_path.name}): no valid rows.")
        return False

    y_vals = sorted(cur["ell_space"].unique().tolist())
    x_vals = sorted(cur["ell_time"].unique().tolist())
    if not y_vals or not x_vals:
        print(f"[WARN] Heatmap skipped ({out_path.name}): empty axis values.")
        return False

    mat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    agg = cur.groupby(["ell_space", "ell_time"], as_index=False)[value_col].mean()
    x_lookup = {float(v): i for i, v in enumerate(x_vals)}
    y_lookup = {float(v): i for i, v in enumerate(y_vals)}
    for _, r in agg.iterrows():
        xv = float(r["ell_time"])
        yv = float(r["ell_space"])
        if xv in x_lookup and yv in y_lookup:
            mat[y_lookup[yv], x_lookup[xv]] = float(r[value_col])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_col)

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:.2g}" for v in x_vals])
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{v:.2g}" for v in y_vals])
    ax.set_xlabel("ell_time")
    ax.set_ylabel("ell_space")
    ax.set_title(title)

    hs, ht = highlight
    ix = next((i for i, x in enumerate(x_vals) if np.isclose(float(x), float(ht))), None)
    iy = next((i for i, y in enumerate(y_vals) if np.isclose(float(y), float(hs))), None)
    if ix is not None and iy is not None:
        ax.scatter([ix], [iy], s=130, marker="o", facecolors="none", edgecolors="red", linewidths=1.8, zorder=5)
        ax.scatter([ix], [iy], s=20, color="red", zorder=6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _plot_stationary_ab(
    df: pd.DataFrame,
    regime: str | None,
    out_path: Path,
    highlight: tuple[float, float] = (4.0, 0.4),
) -> bool:
    """Plot stationary subset A/B comparisons for one regime or across all regimes."""
    req = {"regime", "ell_space", "ell_time", "use_stationary_subset", "mean_rms_all", "std_rms_all"}
    if not req.issubset(df.columns):
        print(f"[WARN] Stationary A/B plot skipped ({out_path.name}): missing columns {sorted(req - set(df.columns))}")
        return False

    cur = df.copy()
    for c in ["ell_space", "ell_time", "use_stationary_subset", "mean_rms_all", "std_rms_all"]:
        cur[c] = pd.to_numeric(cur[c], errors="coerce")
    cur = cur.dropna(subset=["use_stationary_subset"])
    cur["use_stationary_subset"] = cur["use_stationary_subset"].astype(int)
    cur = cur[cur["use_stationary_subset"].isin([0, 1])]
    if cur.empty:
        print(f"[WARN] Stationary A/B plot skipped ({out_path.name}): no valid rows.")
        return False

    if regime is not None:
        cur = cur[cur["regime"].astype(str) == str(regime)].copy()
        if cur.empty:
            print(f"[WARN] Stationary A/B plot skipped ({out_path.name}): regime '{regime}' not found.")
            return False
        cur["cfg"] = cur.apply(lambda r: f"es={float(r['ell_space']):.1f}, et={float(r['ell_time']):.1f}", axis=1)
        cfg_order = (
            cur[["cfg", "ell_space", "ell_time"]]
            .drop_duplicates()
            .sort_values(["ell_space", "ell_time"])["cfg"]
            .tolist()
        )
        piv_mean = cur.pivot_table(index="cfg", columns="use_stationary_subset", values="mean_rms_all", aggfunc="mean")
        piv_std = cur.pivot_table(index="cfg", columns="use_stationary_subset", values="std_rms_all", aggfunc="mean")
        piv_mean = piv_mean.reindex(cfg_order)
        piv_std = piv_std.reindex(cfg_order)

        x = np.arange(len(cfg_order), dtype=float)
        w = 0.36
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        y0 = piv_mean.get(0, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        y1 = piv_mean.get(1, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        e0 = piv_std.get(0, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        e1 = piv_std.get(1, pd.Series(index=cfg_order, dtype=float)).to_numpy(float)
        ax.bar(x - w / 2.0, y0, width=w, yerr=e0, capsize=3, label="subset=0")
        ax.bar(x + w / 2.0, y1, width=w, yerr=e1, capsize=3, label="subset=1")
        ax.set_xticks(x)
        ax.set_xticklabels(cfg_order, rotation=20, ha="right")
        ax.set_ylabel("mean_rms_all")
        ax.set_title(f"Stationary subset A/B: {regime}")
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return True

    hs, ht = highlight
    pick = cur[np.isclose(cur["ell_space"].to_numpy(float), hs) & np.isclose(cur["ell_time"].to_numpy(float), ht)].copy()
    if pick.empty:
        print("[WARN] Combined stationary A/B plot: highlight grid point missing; using regime-wise mean across all configs.")
        pick = cur.groupby(["regime", "use_stationary_subset"], as_index=False).agg(
            mean_rms_all=("mean_rms_all", "mean"),
            std_rms_all=("std_rms_all", "mean"),
        )

    regimes = sorted(pick["regime"].astype(str).unique().tolist())
    if not regimes:
        print(f"[WARN] Stationary A/B plot skipped ({out_path.name}): no regimes available.")
        return False

    p0 = pick[pick["use_stationary_subset"] == 0].set_index("regime")
    p1 = pick[pick["use_stationary_subset"] == 1].set_index("regime")
    x = np.arange(len(regimes), dtype=float)
    y0 = np.array([float(p0.loc[r, "mean_rms_all"]) if r in p0.index else np.nan for r in regimes], dtype=float)
    y1 = np.array([float(p1.loc[r, "mean_rms_all"]) if r in p1.index else np.nan for r in regimes], dtype=float)
    e0 = np.array([float(p0.loc[r, "std_rms_all"]) if r in p0.index else np.nan for r in regimes], dtype=float)
    e1 = np.array([float(p1.loc[r, "std_rms_all"]) if r in p1.index else np.nan for r in regimes], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(x, y0, yerr=e0, marker="o", capsize=3, label="subset=0")
    ax.errorbar(x, y1, yerr=e1, marker="o", capsize=3, label="subset=1")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylabel("mean_rms_all")
    ax.set_title("Stationary subset A/B across regimes")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def _write_plot_readme(out_root: Path, generated_paths: list[Path], grid_csv: Path, ab_csv: Path) -> Path:
    """Write a short README describing plot inputs and outputs."""
    readme_path = out_root / "st_rbf_parameter_selection_plots_README.txt"
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"input_grid_csv: {grid_csv}",
        f"input_stationary_ab_csv: {ab_csv}",
        "selected_default_point: ell_space=4.0, ell_time=0.4",
        "selected_stationary_default: use_stationary_subset=0 (off)",
        "generated_plots:",
    ]
    lines.extend([f"- {p.name}" for p in sorted(generated_paths)])
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


# %% ADDITIONAL BENCHMARKS (Python API)
RUN_CREATE_SYNTHETIC_DATASET = False
RUN_PARAMETER_SWEEP = False
RUN_AGGREGATION = False
RUN_STATIONARY_AB_TESTS = False
RUN_PLOTS = False

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

if RUN_STATIONARY_AB_TESTS:
    _ensure_synthetic_dataset_exists(synthetic_dir)
    _run_stationary_ab_tests(synthetic_dir=synthetic_dir, out_root=out_root)
    stationary_ab_summary = _aggregate_stationary_ab(out_root=out_root)
    stationary_ab_csv_path = out_root / "st_rbf_stationary_ab_summary.csv"
    stationary_ab_summary.to_csv(stationary_ab_csv_path, index=False)
    stationary_ab_report_path = out_root / "st_rbf_stationary_ab_report.md"
    _write_stationary_ab_report(stationary_ab_summary, stationary_ab_report_path)
    print(f"[OK] Wrote stationary A/B summary CSV: {stationary_ab_csv_path}")
    print(f"[OK] Wrote stationary A/B markdown report: {stationary_ab_report_path}")

if RUN_PLOTS:
    generated: list[Path] = []
    grid_csv = out_root / "st_rbf_grid_aggregated.csv"
    ab_csv = out_root / "st_rbf_stationary_ab_summary.csv"

    if grid_csv.exists():
        grid = pd.read_csv(grid_csv)
        if "regime" in grid.columns:
            for regime in sorted(grid["regime"].dropna().astype(str).unique().tolist()):
                reg = grid[grid["regime"].astype(str) == regime].copy()
                p1 = out_root / f"st_rbf_grid_heatmap_mean_rms_all__regime_{regime}.png"
                if _plot_heatmap(reg, "mean_rms_all", f"{regime}: mean_rms_all", p1, highlight=(4.0, 0.4)):
                    generated.append(p1)
                p2 = out_root / f"st_rbf_grid_heatmap_mean_drift_bias_abs__regime_{regime}.png"
                if _plot_heatmap(
                    reg,
                    "mean_drift_bias_abs",
                    f"{regime}: mean_drift_bias_abs",
                    p2,
                    highlight=(4.0, 0.4),
                ):
                    generated.append(p2)

            g_rms = (
                grid.groupby(["ell_space", "ell_time"], as_index=False)
                .agg(mean_rms_all=("mean_rms_all", "mean"))
                .copy()
            )
            p3 = out_root / "st_rbf_grid_heatmap_mean_rms_all__across_regimes.png"
            if _plot_heatmap(g_rms, "mean_rms_all", "Across regimes: mean_rms_all", p3, highlight=(4.0, 0.4)):
                generated.append(p3)

            g_bias = (
                grid.groupby(["ell_space", "ell_time"], as_index=False)
                .agg(mean_drift_bias_abs=("mean_drift_bias_abs", "mean"))
                .copy()
            )
            p4 = out_root / "st_rbf_grid_heatmap_mean_drift_bias_abs__across_regimes.png"
            if _plot_heatmap(
                g_bias,
                "mean_drift_bias_abs",
                "Across regimes: mean_drift_bias_abs",
                p4,
                highlight=(4.0, 0.4),
            ):
                generated.append(p4)
        else:
            print(f"[WARN] Grid plots skipped: missing 'regime' column in {grid_csv}")
    else:
        print(f"[WARN] Grid plots skipped: missing input CSV {grid_csv}")

    if ab_csv.exists():
        ab = pd.read_csv(ab_csv)
        if "regime" in ab.columns:
            for regime in sorted(ab["regime"].dropna().astype(str).unique().tolist()):
                p = out_root / f"st_rbf_stationary_subset_ab__mean_rms_all__regime_{regime}.png"
                if _plot_stationary_ab(ab, regime=regime, out_path=p, highlight=(4.0, 0.4)):
                    generated.append(p)
            p_all = out_root / "st_rbf_stationary_subset_ab__mean_rms_all__all_regimes.png"
            if _plot_stationary_ab(ab, regime=None, out_path=p_all, highlight=(4.0, 0.4)):
                generated.append(p_all)
        else:
            print(f"[WARN] Stationary A/B plots skipped: missing 'regime' column in {ab_csv}")
    else:
        print(f"[WARN] Stationary A/B plots skipped: missing input CSV {ab_csv}")

    readme_plot = _write_plot_readme(out_root=out_root, generated_paths=generated, grid_csv=grid_csv, ab_csv=ab_csv)
    print(f"[OK] Wrote plotting README: {readme_plot}")
    print(f"[OK] Generated {len(generated)} plot(s) in {out_root}")

print("[INFO] Additional benchmarks and aggregation complete.")
# %% END
