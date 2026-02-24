import numpy as np
import pandas as pd

from driftcorr.association.scoring import evaluate_against_groundtruth


def _make_gt_mapping_and_tracks() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Two GT tracks over 3 frames.
    rows_map = []
    rows_gt = []
    for t in [1.0, 2.0, 3.0]:
        # GT track 10 -> point 0
        rows_map.append({"Time": t, "PointIndex": 0, "GT_TrackID": 10.0})
        rows_gt.append({"Time": t, "TrackID": 10.0, "Position X": 1.0 + t, "Position Y": 2.0, "Position Z": 3.0})
        # GT track 20 -> point 1
        rows_map.append({"Time": t, "PointIndex": 1, "GT_TrackID": 20.0})
        rows_gt.append({"Time": t, "TrackID": 20.0, "Position X": 5.0 + t, "Position Y": 6.0, "Position Z": 7.0})
    return pd.DataFrame(rows_map), pd.DataFrame(rows_gt)


def test_per_time_metrics_columns_and_summary_fields_present():
    mapping, gt_tracks = _make_gt_mapping_and_tracks()

    # Inferred IDs fragment GT track 10 (id changes between t=1 and t=2).
    inf_rows = [
        {"TrackID": 101.0, "Time": 1.0, "PointIndex": 0, "x": 2.0, "y": 2.0, "z": 3.0},
        {"TrackID": 102.0, "Time": 2.0, "PointIndex": 0, "x": 3.0, "y": 2.0, "z": 3.0},
        {"TrackID": 102.0, "Time": 3.0, "PointIndex": 0, "x": 4.0, "y": 2.0, "z": 3.0},
        {"TrackID": 201.0, "Time": 1.0, "PointIndex": 1, "x": 6.0, "y": 6.0, "z": 7.0},
        {"TrackID": 201.0, "Time": 2.0, "PointIndex": 1, "x": 7.0, "y": 6.0, "z": 7.0},
        {"TrackID": 201.0, "Time": 3.0, "PointIndex": 1, "x": 8.0, "y": 6.0, "z": 7.0},
    ]
    inferred = pd.DataFrame(inf_rows)

    # Deliberately miss one GT link at t=1 by only including one inferred link.
    links = pd.DataFrame(
        [
            {"t": 1.0, "src_index": 0, "dst_index": 0, "cost": 0.1},
            {"t": 2.0, "src_index": 0, "dst_index": 0, "cost": 0.1},
            {"t": 2.0, "src_index": 1, "dst_index": 1, "cost": 0.1},
        ]
    )

    summary, per_time, _ = evaluate_against_groundtruth(inferred, mapping, links_df=links, gt_tracks_df=gt_tracks)

    required_cols = {
        "Time",
        "link_precision",
        "link_recall",
        "link_f1",
        "n_links_inferred",
        "n_links_correct",
        "n_links_gt_expected",
        "rms",
        "n_gt_tracks",
        "n_matched_tracks",
        "n_unmatched_tracks",
    }
    assert required_cols.issubset(set(per_time.columns))

    for key in [
        "link_precision_mean",
        "link_recall_mean",
        "link_f1_mean",
        "fragmentation_rate",
        "mean_fragments_per_gt",
        "id_switches_total",
        "id_switches_mean_per_track",
        "rms_mean",
        "rms_std",
        "unmatched_rate_mean",
    ]:
        assert key in summary


def test_fragmentation_and_switch_metrics_positive_when_ids_split():
    mapping, gt_tracks = _make_gt_mapping_and_tracks()

    inferred = pd.DataFrame(
        [
            {"TrackID": 1.0, "Time": 1.0, "PointIndex": 0, "x": 2.0, "y": 2.0, "z": 3.0},
            {"TrackID": 2.0, "Time": 2.0, "PointIndex": 0, "x": 3.0, "y": 2.0, "z": 3.0},
            {"TrackID": 2.0, "Time": 3.0, "PointIndex": 0, "x": 4.0, "y": 2.0, "z": 3.0},
            {"TrackID": 3.0, "Time": 1.0, "PointIndex": 1, "x": 6.0, "y": 6.0, "z": 7.0},
            {"TrackID": 3.0, "Time": 2.0, "PointIndex": 1, "x": 7.0, "y": 6.0, "z": 7.0},
            {"TrackID": 3.0, "Time": 3.0, "PointIndex": 1, "x": 8.0, "y": 6.0, "z": 7.0},
        ]
    )

    links = pd.DataFrame(
        [
            {"t": 1.0, "src_index": 0, "dst_index": 0, "cost": 0.1},
            {"t": 1.0, "src_index": 1, "dst_index": 1, "cost": 0.1},
            {"t": 2.0, "src_index": 0, "dst_index": 0, "cost": 0.1},
            {"t": 2.0, "src_index": 1, "dst_index": 1, "cost": 0.1},
        ]
    )

    summary, _, _ = evaluate_against_groundtruth(inferred, mapping, links_df=links, gt_tracks_df=gt_tracks)

    assert float(summary["fragmentation_rate"]) > 0.0
    assert float(summary["mean_fragments_per_gt"]) > 1.0
    assert float(summary["id_switches_total"]) > 0.0
    # With GT-track splitting, link F1 must drop below 1.
    assert float(summary["link_f1_mean"]) < 1.0


def test_rms_reports_unmatched_tracks():
    mapping, gt_tracks = _make_gt_mapping_and_tracks()

    # Missing one point at t=2 (GT track 20 unmatched).
    inferred = pd.DataFrame(
        [
            {"TrackID": 1.0, "Time": 1.0, "PointIndex": 0, "x": 2.0, "y": 2.0, "z": 3.0},
            {"TrackID": 1.0, "Time": 2.0, "PointIndex": 0, "x": 3.0, "y": 2.0, "z": 3.0},
            {"TrackID": 1.0, "Time": 3.0, "PointIndex": 0, "x": 4.0, "y": 2.0, "z": 3.0},
            {"TrackID": 2.0, "Time": 1.0, "PointIndex": 1, "x": 6.0, "y": 6.0, "z": 7.0},
            {"TrackID": 2.0, "Time": 3.0, "PointIndex": 1, "x": 8.0, "y": 6.0, "z": 7.0},
        ]
    )
    links = pd.DataFrame([{"t": 1.0, "src_index": 0, "dst_index": 0, "cost": 0.1}])

    summary, per_time, _ = evaluate_against_groundtruth(inferred, mapping, links_df=links, gt_tracks_df=gt_tracks)

    assert "unmatched_rate_mean" in summary
    assert per_time["n_unmatched_tracks"].max() >= 1
    assert np.isfinite(float(summary["rms_mean"])) or np.isnan(float(summary["rms_mean"]))
