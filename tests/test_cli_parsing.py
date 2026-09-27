# %% IMPORTS
from driftcorr.cli import build_parser
# %% TEST FUNCTIONS
def test_cli_parses_smooth_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "smooth",
            "--method",
            "rigid",
            "--input",
            "sample.xls",
            "--out",
            "out_dir",
            "--savgol-window",
            "11",
            "--polyorder",
            "3",
            "--detect-jumps",
        ]
    )
    assert args.command == "smooth"
    assert args.method == "rigid"
    assert args.savgol_window == 11
    assert args.polyorder == 3
    assert bool(args.detect_jumps) is True

def test_cli_parses_st_sweep_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "st-sweep",
            "--input-dir",
            "imaris_dir",
            "--out",
            "out_dir",
            "--ell-time",
            "0.3,0.4,0.8",
            "--lam",
            "5e-4,8.5e-4",
        ]
    )
    assert args.command == "st-sweep"
    assert args.input_dir == "imaris_dir"
    assert args.ell_time == "0.3,0.4,0.8"
    assert args.lam == "5e-4,8.5e-4"
# %% END