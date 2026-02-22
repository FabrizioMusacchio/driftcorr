"""Python benchmark API wrappers for driftcorr synthetic workflows.

This module provides Python-callable wrappers around the benchmark-oriented
CLI subcommands. It allows research scripts and tutorials to execute the same
pipelines without building terminal argument lists.

Mathematical role:
- No new drift model is introduced here.
- These wrappers delegate to existing orchestration routines in `cli.py`,
  which in turn call correction methods (`rbf`, `rigid`, `local`, `st_rbf`,
  `st_rbf_moe`) and unified evaluation logic.

Controlling parameters:
- Functions accept keyword overrides matching CLI argument names with
  underscores (for example `synthetic_n_seeds`, `st_ell_space`, `st_ell_time`).
- Unspecified values keep the same defaults as the CLI parser.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from __future__ import annotations

import argparse
from typing import Any

from .cli import (
    build_parser,
    run_paper_output_regression,
    run_synthetic_benchmark,
    run_synthetic_generate,
    run_synthetic_regime_benchmark,
)


def _namespace_for_command(command: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args([command])


def _apply_overrides(args: argparse.Namespace, overrides: dict[str, Any]) -> argparse.Namespace:
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown option for command '{args.command}': {key}")
        setattr(args, key, value)
    return args


def synthetic_generate(**kwargs: Any) -> int:
    """Generate synthetic IMARIS-like datasets using Python API options."""
    args = _namespace_for_command("synthetic-generate")
    args = _apply_overrides(args, kwargs)
    return run_synthetic_generate(args)


def synthetic_benchmark(**kwargs: Any) -> int:
    """Run one synthetic benchmark dataset end-to-end (generate + methods + evaluation)."""
    args = _namespace_for_command("synthetic-benchmark")
    args = _apply_overrides(args, kwargs)
    return run_synthetic_benchmark(args)


def synthetic_regime_benchmark(**kwargs: Any) -> int:
    """Run the 3-regime synthetic suite across seeds with fixed dataset reuse support."""
    args = _namespace_for_command("synthetic-regime-benchmark")
    args = _apply_overrides(args, kwargs)
    return run_synthetic_regime_benchmark(args)


def paper_regression(**kwargs: Any) -> int:
    """Run the paper-output regression harness against artifact/column manifests."""
    args = _namespace_for_command("paper-regression")
    args = _apply_overrides(args, kwargs)
    return run_paper_output_regression(args)

