# DriftCorr: Correcting geometric drift in time-lapse microscopy data

DriftCorr is a Python toolkit for coordinate-based analysis of time-lapse microscopy data. It focuses on two coupled problems that arise after segmentation or spot detection: reconstructing object identities from frame-wise point clouds and correcting geometric drift in reconstructed trajectories.

The package was developed for 3D+t microscopy workflows in which image data are reduced to centroid tables, for example IMARIS spot exports or synthetic IMARIS-like tables. It can also be applied to 2D+t data by embedding planar coordinates into 3D with `Position Z = 0`.

## Motivation
Structural time-lapse microscopy often contains both real biological motion and acquisition-related geometric artifacts. Standard image-registration workflows are not always appropriate once cells move, appear, disappear, or change morphology, because overly flexible image warps can absorb real biology into the correction field.

DriftCorr therefore works directly on exported coordinates. It provides:

- drift-correction models for identity-labeled trajectories,
- point-cloud association methods for recovering identities from unlabeled detections,
- synthetic benchmark generators with known ground truth,
- reproducibility scripts for the benchmark and biological-example analyses reported in the preprint.

## Main features
- Rigid step-wise drift correction with robust pair selection.
- Local community-based drift correction for spatially heterogeneous drift.
- Per-timepoint RBF drift correction.
- Joint space-time RBF drift correction.
- Optional residual iterative drift correction.
- Hungarian, nearest-neighbor, tracklet, Kalman, and HCV point-cloud association variants.
- Synthetic data generators for drift-correction and association benchmarks.
- IMARIS `.xls` / `.xlsx` position-table parsing.
- Diagnostic plots, corrected-coordinate exports, and benchmark summaries.

## Repository structure

```text
driftcorr/                  Core Python package
  association/              Point-cloud association algorithms and scoring
  *.py                      Drift-correction methods, metrics, visualization, CLI
user_scripts/               Reproducibility and example scripts
papers/preprint/            Manuscript source and figure PDFs, if included locally
tests/                      Unit and smoke tests
```

The most relevant user scripts are:

- `user_scripts/tutorial_pipeline_usage.py`: Minimal Python workflow examples.
- `user_scripts/parameter_sweep_on_synthetic_data.py`: Synthetic drift-correction benchmark and sweep scripts.
- `user_scripts/parameter_sweep_on_synthetic_non_rigid_data.py`: Structured non-rigid stress benchmarks.
- `user_scripts/benchmark_point_cloud_association.py`: Synthetic point-cloud association benchmarks.
- `user_scripts/segment_carcinoma_3dt_and_build_gt.py`: Biological 3D+t carcinoma example workflow.
- `user_scripts/segment_b1_04_2_1_dapi_with_cellpose.py`: Cellpose segmentation for the DAPI example.
- `user_scripts/segment_b1_04_2_1_dapi_2dt_and_run_pipeline.py`: DAPI PCA and DriftCorr downstream workflow.

Private or unpublished exploratory scripts should live in `user_scripts_private/`, which is ignored by Git.

## Installation
Clone the repository and install it in editable mode:

```bash
git clone https://github.com/FabrizioMusacchio/driftcorr.git
cd driftcorr
python -m pip install -e .
```

A conda environment similar to the one used during development can be created with:

```bash
conda create -n manual_registration python=3.11 -y
conda activate manual_registration
python -m pip install -e .
```

For the core package and synthetic benchmarks, the required dependencies are listed in `pyproject.toml`. Biological segmentation workflows may require additional packages such as `tifffile`, `scikit-image`, and optionally `cellpose`.

## Quick start
### Run drift correction on one IMARIS export

```bash
python -m driftcorr.cli run \
  --method rigid \
  --imaris-xls path/to/imaris_position_export.xls \
  --out driftcorr_results
```

Available correction methods include:

```text
rigid, local, rbf, st_rbf
```

### Run several methods on a folder

```bash
python -m driftcorr.cli run-all \
  --imaris-dir path/to/imaris_exports \
  --methods rigid,local,rbf,st_rbf \
  --out driftcorr_results
```

### Generate synthetic drift-correction data

```bash
python -m driftcorr.cli synthetic-generate \
  --synthetic-dir synthetic_imaris_files \
  --synthetic-n-files 5 \
  --synthetic-n-tracks 80 \
  --synthetic-n-times 20 \
  --synthetic-seed 42
```

### Run an end-to-end synthetic drift-correction benchmark

```bash
python -m driftcorr.cli synthetic-benchmark \
  --synthetic-dir synthetic_imaris_files \
  --synthetic-n-files 5 \
  --synthetic-n-tracks 80 \
  --synthetic-n-times 20 \
  --methods rigid,local,rbf,st_rbf
```

### Run the point-cloud association benchmark scripts

The association benchmarks used in the paper are implemented as notebook-style Python scripts with `# %%` cells:

```bash
PYTHONPATH="$PWD" python user_scripts/benchmark_point_cloud_association.py
```

The scripts are designed so that synthetic inputs are generated from fixed seeds and configuration parameters. Large output folders are intentionally not tracked in Git.

## Input data format
For real-data drift correction, DriftCorr expects an IMARIS-like position table with at least these columns:

```text
Position X, Position Y, Position Z, Time, TrackID
```

The parser reads the `Position` sheet from IMARIS `.xls` / `.xlsx` exports. Synthetic benchmark tables use the same coordinate convention.

## Output files
For each input file and method, DriftCorr writes corrected coordinate tables and diagnostic plots. Typical outputs include:

- corrected coordinate CSV files,
- method-specific fit summaries,
- trajectory comparison plots,
- per-frame or full-track visualizations,
- metric summary CSV files,
- ground-truth error summaries for synthetic data.

If no output directory is provided, batch workflows write results into a `driftcorr_results/` folder next to the input data.

## Reproducing the preprint analyses
The synthetic benchmark datasets reported in the preprint are generated by the code itself. They do not need to be downloaded separately. The relevant scripts are:

```text
user_scripts/parameter_sweep_on_synthetic_data.py
user_scripts/parameter_sweep_on_synthetic_non_rigid_data.py
user_scripts/benchmark_point_cloud_association.py
```

The biological example datasets are external public datasets and are referenced in the manuscript. Download those datasets from the cited Zenodo records before running the biological-example scripts. The scripts assume the local paths shown in their docstrings unless edited.

Because generated benchmark folders can be large, synthetic outputs, biological raw data, and local result folders are ignored by Git. Reproducibility is intended to come from the scripts, fixed seeds, and documented parameter grids rather than from committing generated data products.

## Testing
Run the test suite with:

```bash
python -m pytest
```

The tests cover command-line parsing, core drift-correction smoke checks, smoothing behavior, and metric calculations.

## License
This project is released under the GNU General Public License v3.0 or later. See `LICENSE` for details.

## How to cite
If you use DriftCorr, please cite the associated preprint

> Fabrizio Musacchio and Martin Fuhrmann, "DriftCorr: Robust geometric drift correction and identity reconstruction for biological 3D time-lapse point clouds," forthcoming.

and the software repository:

```bibtex
@software{musacchio_driftcorr_2026,
  author = {Musacchio, Fabrizio and Fuhrmann, Martin},
  title = {DriftCorr: Robust geometric drift correction and identity reconstruction for biological 3D time-lapse point clouds},
  year = {2026},
  url = {https://github.com/FabrizioMusacchio/driftcorr},
  license = {GPL-3.0-or-later}
}
```

A citation file is provided in `CITATION.cff`.
