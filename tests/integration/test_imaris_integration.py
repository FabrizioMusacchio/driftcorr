from pathlib import Path

from driftcorr.evaluate import run_all_methods_and_compare, save_evaluation_outputs
from driftcorr.io_imaris import ensure_numeric_tracks, list_imaris_xls_files, read_imaris_position_xls
from driftcorr.method_rbf_ref import correct_positions_rbf_to_reference


def test_imaris_xls_rbf_integration(tmp_path):
    root = Path(__file__).resolve().parents[2]
    imaris_dir = root / "IMARIS xls files"
    files = list_imaris_xls_files(imaris_dir)
    assert files, f"No IMARIS xls files found in {imaris_dir}"

    in_file = files[0]
    df = ensure_numeric_tracks(read_imaris_position_xls(in_file))

    df_corr, _, _, _, _ = correct_positions_rbf_to_reference(
        df,
        m=20,
        keep_fraction=0.7,
        lam=1e-2,
        min_pairs=15,
        seed=0,
    )

    summary_df, details = run_all_methods_and_compare(df, [("rbf", df_corr)])
    outputs = save_evaluation_outputs(tmp_path / in_file.stem, summary_df, details)

    assert not summary_df.empty
    assert outputs["metrics_summary"].exists()
    assert outputs["plot_corr_hist"].exists()
    assert outputs["plot_msd_curve"].exists()
    assert outputs["plot_neighbor_stats"].exists()
