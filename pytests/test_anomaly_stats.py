"""骨架异常诊断的合成数据测试。"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vascular_statistics.aggregate_stats import (
    generate_cross_sample_summary,
    write_aggregate_stats,
)
from vascular_statistics.anomaly_stats import write_run_anomaly_stats


def _write_lines(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def _write_summary(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Vascular_Statistics\n"
            "平均直径 (um): 8.0\n"
            "平均长度 (um): 570.0\n"
            "段密度 (seg/mm^3): 30.0\n"
            "平均弯曲度: 1.2\n"
        )


def _make_sample(tmpdir):
    sample_key = "TEST_GROUP/20240101_A001_D0/sample_001"
    sample_dir = os.path.join(tmpdir, *sample_key.split("/"))
    run_dir = os.path.join(sample_dir, "run_20260707_210000")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(sample_dir, "sample_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_key": sample_key,
                "path_parsed": {
                    "group": "TEST_GROUP",
                    "batch_id": "A001",
                    "daypoint": "D0",
                    "study_day": 0,
                    "study_day_label": "D0",
                    "acquisition_date": "2024-01-01",
                    "timepoint_id": "20240101_D0",
                },
                "spatial": {"tissue_volume_mm3": 0.1},
            },
            f,
        )

    _write_summary(os.path.join(run_dir, "statistics_summary.txt"))
    _write_lines(os.path.join(run_dir, "generate_vessel_radius.txt"), [1.0, 1.0, 1.0])
    _write_lines(os.path.join(run_dir, "generate_vessel_path_length.txt"), [10.0, 20.0, 800.0])
    _write_lines(os.path.join(run_dir, "generate_vessel_tortuosity.txt"), [1.1, 1.2, 1.3])
    _write_lines(os.path.join(run_dir, "generate_vessel.txt"), ["1 2 3", "4 5 6", "7 8 9"])

    with open(os.path.join(run_dir, "skeleton_vertices.txt"), "w", encoding="utf-8") as f:
        for nid in range(1, 10):
            f.write(f"{nid} 0 {nid}.0 0.0 0.0 1.0\n")

    _write_lines(
        os.path.join(run_dir, "skeleton_edges.txt"),
        ["1 2", "2 3", "4 5", "5 6", "7 8", "8 9"],
    )

    return sample_dir, run_dir


def test_run_and_aggregate_anomaly_stats():
    tmpdir = tempfile.mkdtemp()
    try:
        sample_dir, run_dir = _make_sample(tmpdir)

        anomaly_path = write_run_anomaly_stats(run_dir, 0.1)
        assert anomaly_path is not None
        assert os.path.exists(anomaly_path)
        assert os.path.exists(os.path.join(run_dir, "skeleton_anomaly_summary.txt"))
        assert os.path.exists(os.path.join(run_dir, "skeleton_segment_anomalies.tsv"))
        assert os.path.exists(os.path.join(run_dir, "skeleton_edge_anomalies.tsv"))

        with open(anomaly_path, encoding="utf-8") as f:
            raw_json = f.read()
        assert "NaN" not in raw_json
        data = json.loads(raw_json)
        assert data["status"] == "FAIL"
        assert data["populations"]["full"]["abs_outliers"] == 1
        assert data["populations"]["d0-10um"]["abs_outliers"] == 1

        with open(os.path.join(run_dir, "skeleton_segment_anomalies.tsv"), encoding="utf-8") as f:
            segment_content = f.read()
        assert "absolute_long" in segment_content
        assert "mad_length" in segment_content
        assert "7 8 9" in segment_content

        with open(os.path.join(run_dir, "skeleton_edge_anomalies.tsv"), encoding="utf-8") as f:
            edge_content = f.read()
        assert "segment_id" in edge_content
        assert "7\t8" in edge_content or "8\t9" in edge_content

        agg_path = write_aggregate_stats(sample_dir)
        assert agg_path is not None
        with open(agg_path, encoding="utf-8") as f:
            agg_content = f.read()
        assert "骨架异常诊断" in agg_content
        assert "QC状态: FAIL" in agg_content

        cross_path = generate_cross_sample_summary(tmpdir)
        assert cross_path is not None
        with open(cross_path, encoding="utf-8") as f:
            cross_content = f.read()
        assert "QC" in cross_content
        assert "FAIL" in cross_content
        assert "33.3333" in cross_content
    finally:
        shutil.rmtree(tmpdir)


def test_edge_topology_qc_flags_single_long_edge():
    tmpdir = tempfile.mkdtemp()
    try:
        sample_dir, run_dir = _make_sample(tmpdir)
        _write_lines(os.path.join(run_dir, "generate_vessel_path_length.txt"), [10.0, 20.0, 30.0])
        with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"stats_unit_mode": "anisotropic", "stats_spacing_um": [1.0, 1.0, 1.0]}, f)
        with open(os.path.join(run_dir, "skeleton_vertices.txt"), "w", encoding="utf-8") as f:
            for nid in range(1, 10):
                y = 150.0 if nid == 9 else float(nid)
                f.write(f"{nid} 0 {y:.1f} 0.0 0.0 1.0\n")
        _write_lines(
            os.path.join(run_dir, "skeleton_edges.txt"),
            ["1 2", "2 3", "4 5", "5 6", "7 8", "1 9"],
        )

        anomaly_path = write_run_anomaly_stats(run_dir, 0.1)
        with open(anomaly_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "FAIL"
        assert data["edge_qc"]["status"] == "FAIL"
        assert data["edge_qc"]["fail_edges"] >= 1
        assert data["edge_qc"]["max_edge_length_um"] == 149.0

        long_edge_path = os.path.join(run_dir, "skeleton_long_edges.tsv")
        assert os.path.exists(long_edge_path)
        with open(long_edge_path, encoding="utf-8") as f:
            long_edge_content = f.read()
        assert "FAIL" in long_edge_content
        assert "edge_length_gt_100um" in long_edge_content
        assert "1\t9" in long_edge_content

        with open(os.path.join(run_dir, "skeleton_anomaly_summary.txt"), encoding="utf-8") as f:
            summary_content = f.read()
        assert "edge topology QC" in summary_content
        assert "all_edges | FAIL" in summary_content

        agg_path = write_aggregate_stats(sample_dir)
        with open(agg_path, encoding="utf-8") as f:
            agg_content = f.read()
        assert "拓扑单边QC" in agg_content
        assert "FAIL边: 1" in agg_content
    finally:
        shutil.rmtree(tmpdir)
