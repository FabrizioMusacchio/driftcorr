# driftcorr

`driftcorr` is a Python toolkit for drift correction in IMARIS spot-track exports (`.xls`, `.xlsx`, and synthetic `.csv` tables).
It is designed for 4D microscopy tracking data (`time + x/y/z`) where observed motion contains both true biological migration and imaging-related drift artifacts.

## Problem Setting
Typical trajectories contain three motion components:
1. Global imaging drift (many cells move similarly).
2. Spatially varying but smooth drift (not piecewise constant, not just linear in `z`).
3. Genuine biological migration that must be preserved.

The goal is to remove (1) and (2) while preserving (3).

## Implemented Methods

### `rbf`: RBF-to-reference drift correction
For each timepoint `t` (relative to a reference time `t_ref`), fit a smooth drift field:
- Training target: `D_t = X_t - X_ref`
- Model: `D_t ≈ [K(X_t, C), 1] W_t` (bias-augmented kernel)
- Kernel: Gaussian RBF
- Correction: `X_corr = X - d_hat(X,t)`

Optional temporal consistency (`--temporal-gamma`):
- shared centers across time,
- jointly fit `W(t)` with penalty:
  `sum_t ||K_t W(t)-D_t||^2 + lambda ||W(t)||^2 + gamma ||W(t)-W(t-1)||^2`

Additional controls:
- center selection: `--center-mode {ref,pooled,kmeans_pooled}`
- robust fitting: `--robust-mode {sample_trim,track_trim}`
- two-stage translation + residual field: `--two-stage 0/1`

### `rigid`: rigid step-wise correction
Between consecutive times `t0 -> t1`:
- fit `Y ≈ R X + b` (Kabsch, optional `rotz` mode),
- robust trimmed inlier re-estimation (`--inlier-reestimate-iters`),
- compose transforms back to reference frame.

Optional piecewise rigid (`--rigid-spatial-groups > 1`):
- cluster tracks in space,
- fit per-group rigid transforms,
- blend corrections spatially with Gaussian weights.

### `local`: community-based local drift field
- Build displacement traces relative to reference time.
- Build spatial neighbor graph.
- Edge weight = displacement similarity × spatial proximity.
- Detect communities.
- Estimate one robust drift trace per community.
- Blend community drifts into a smooth local field.

### `st_rbf`: space-time RBF drift model
Jointly fit drift in 4D input space `(x,y,z,t)`:
- `d(u) ≈ K(u, C) W`
- Bias-augmented kernel (`[K,1]`) and optional two-stage decomposition
- Space input modes: `--space-input-mode {row_position,track_anchor,ref_position}`
- Normalized coordinates (space robustly scaled, time mapped to `[0,1]`)
- Optional iterative refinement (`--st-refine-iters`)
- Robust trimming / reweighting to reduce influence of migrating tracks.

### Stationary subset discovery
Optional for `rbf` and `st_rbf` (`--use-stationary-subset`):
- Select likely stationary tracks using displacement and neighborhood-stability criteria.
- Fit drift model using selected subset.
- Save selection diagnostics and metric impact tables.

## Evaluation Outputs
For each file and method:
- pairwise displacement-trace correlation,
- neighbor distance stability,
- MSD comparison,
- RMS displacement statistics,
- near-stationary fractions.

Standard files:
- `<base>_metrics_summary.csv`
- `<base>_plots_corr_hist.png`
- `<base>_plots_msd_curve.png`
- `<base>_plots_neighbor_stats.png`

## Visual Inspection Outputs
Per method:
- trajectory comparison panel:
  - `<base>_tracks_comparison_<method>.png`
- frame-by-frame 3D snapshots:
  - folder: `<base>_timeseries_<method>/`
  - files: `frame_0000_t_...png`, `frame_0001_t_...png`, ...
  - GIF animation: `<base>_timeseries_<method>/timeseries.gif`

For synthetic runs (ground truth available), frame plots include:
- `Ground Truth | Original | Corrected`

(3D frame plots intentionally have no colorbar.)

## Synthetic Benchmark Mode
`synthetic-benchmark` creates controlled synthetic datasets and runs the normal pipeline end-to-end.

Generated synthetic components:
- stationary + migrating subsets,
- global drift,
- smooth spatially varying local drift,
- optional temporal smoothness,
- measurement noise.

Ground-truth sidecar outputs are written to `_groundtruth` and include:
- drift-free trajectories,
- per-row drift vectors (global/local/total).

Additional ground-truth evaluation outputs:
- `<base>_gt_metrics_summary.csv`
- `<base>_gt_rms_per_track_hist.png`
- `<base>_gt_rms_over_time.png`
- `<base>_gt_rms_stationary_over_time.png`
- `<base>_gt_rms_migrating_over_time.png`
- plus per-method CSV tables for per-track/per-time RMS errors.

## Result Folder Structure
If `--out` is not provided:
- `run`: `<input-file-directory>/driftcorr_results`
- `run-all`: `<imaris-dir>/driftcorr_results`

Inside that folder, each input file gets its own subfolder:
- `<results>/<input_stem>/...all outputs for that file...`

## Usage

### 1) Run one method on one file
```bash
conda run -n manual_registration env PYTHONPATH="$PWD" python -m driftcorr.cli run \
  --method rbf \
  --imaris-xls "IMARIS xls files/registered_stack_5D_251008_P2_565um_v1.xls"
```

### 2) Batch run on a folder
```bash
conda run -n manual_registration env PYTHONPATH="$PWD" python -m driftcorr.cli run-all \
  --imaris-dir "IMARIS xls files" \
  --methods rbf,rigid,local,st_rbf
```

### 3) Generate and benchmark synthetic data
```bash
conda run -n manual_registration env PYTHONPATH="$PWD" python -m driftcorr.cli synthetic-benchmark \
  --synthetic-dir "Synthetic IMARIS files" \
  --synthetic-n-files 5 \
  --synthetic-n-tracks 80 \
  --synthetic-n-times 20 \
  --methods rbf,rigid,local,st_rbf
```

## Python Tutorial Script
A notebook-style tutorial script is provided here:
- `user_scripts/tutorial_pipeline_usage.py`
- `user_scripts/param_sweep_synthetic.py` (RBF/ST-RBF parameter sweep on synthetic data)

It contains these sections:
- `# %% HOW TO RUN A SIGNLE FILE`
- `# %% HOW TO BTACH RUN MULTIPLE FILES`
- `# %% HOW TO GENERATE AND RUN SYNTHETIC DATA`

## Commonly Tuned Options
- RBF: `--m-centers`, `--lam`, `--sigma`, `--temporal-gamma`, `--center-mode`, `--robust-mode`, `--two-stage`
- Rigid: `--inlier-reestimate-iters`, `--rigid-spatial-groups`, `--rigid-compare-piecewise`
- Local: `--k-neighbors`, `--ell-um-graph`, `--ell-um-field`
- ST-RBF: `--st-m-centers`, `--st-ell-space`, `--st-ell-time`, `--space-input-mode`, `--st-refine-iters`, `--two-stage`
- Stationary subset: `--use-stationary-subset` or `--use-stationary-only 1`
- Disable visual plots: `--skip-visual-plots`

Current tuned defaults (Feb 2026):
- RBF defaults: `--m-centers 45 --lam 3e-4 --sigma 180 --temporal-gamma 0`
- ST-RBF defaults: `--st-m-centers 70 --st-ell-space 1.2 --st-ell-time 0.3 --st-robust-keep-fraction 0.8 --st-refine-iters 2 --lam 3e-4`
