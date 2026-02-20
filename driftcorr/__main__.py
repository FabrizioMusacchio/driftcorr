"""Module entry point for `python -m driftcorr`.

This script forwards execution to the command-line interface dispatcher in
`driftcorr.cli:main`, so all pipeline functionality is available from module
execution without installing console scripts explicitly.

Mathematical model:
- None in this file. Numerical methods are implemented in dedicated modules.

Controlling behavior:
- CLI arguments are parsed in `driftcorr.cli`.
- Method-specific parameters (e.g., `--temporal-gamma`, `--keep-fraction`,
  `--st-ell-space`) are passed through untouched.

Author
-------
author: Fabrizio Musacchio
date: Feb 2026
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
