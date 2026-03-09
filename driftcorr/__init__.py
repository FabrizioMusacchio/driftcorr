"""Top-level package initialization for driftcorr.

This package implements multiple drift-correction models for IMARIS-exported 3D
cell tracking data over time. The library provides:
- model modules (`rbf`, `rigid`, `local`, `st_rbf`),
- unified quantitative evaluation utilities,
- visualization helpers for qualitative inspection,
- synthetic benchmark generation with ground truth.

No correction model is fit in this module directly. Instead, this module exposes
I/O helpers and configures global plotting aesthetics (spine visibility), which
affect downstream plotting modules consistently.

Controlling behavior:
- Global `matplotlib.rcParams` are set here:
  `axes.spines.right/top/left/bottom = False`.
- This only influences figure style; it does not change numerical results.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

try:
    import matplotlib.pyplot as plt

    # remove spines globally for all plots:
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.bottom"] = False
    
    # set font size globally for all plots:
    plt.rcParams["font.size"] = 8
    
    # set font family globally for all plots to Arial:
    plt.rcParams["font.family"] = "Arial"
except Exception:
    pass

from .io_imaris import list_imaris_xls_files, load_imaris_directory, read_imaris_position_xls
from .benchmarks import paper_regression, synthetic_benchmark, synthetic_generate, synthetic_regime_benchmark

__all__ = [
    "list_imaris_xls_files",
    "load_imaris_directory",
    "read_imaris_position_xls",
    "synthetic_generate",
    "synthetic_benchmark",
    "synthetic_regime_benchmark",
    "paper_regression",
]
