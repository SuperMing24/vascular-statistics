import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vascular_statistics.aggregate_stats import generate_cross_sample_summary
from vascular_statistics.subrange_stats import DIAMETER_SUFFIX, SUFFIX_D10_PLUS


def _write_summary(path, diameter, length, density, tortuosity):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Vascular_Statistics\n"
            f"平均直径 (um): {diameter}\n"
            f"平均长度 (um): {length}\n"
            f"段密度 (seg/mm^3): {density}\n"
            f"平均弯曲度: {tortuosity}\n"
        )


def _write_vessels(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{i} {i + 1} {i + 2}\n")


def test_generate_cross_sample_summary_three_populations():
    tmpdir = tempfile.mkdtemp()
    try:
        sample_dir = os.path.join(
            tmpdir, "TEST_GROUP", "20240101_A001_D0", "sample_001"
        )
        run_dir = os.path.join(sample_dir, "run_20240702_120000")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(sample_dir, "sample_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_key": "TEST_GROUP/20240101_A001_D0/sample_001",
                    "path_parsed": {
                        "group": "TEST_GROUP",
                        "batch_id": "A001",
                        "daypoint": "D0",
                    },
                    "spatial": {"tissue_volume_mm3": 0.1},
                },
                f,
            )

        _write_summary(os.path.join(run_dir, "statistics_summary.txt"), 5.5, 30.0, 100.0, 1.1)
        _write_summary(
            os.path.join(run_dir, f"statistics_summary{DIAMETER_SUFFIX}.txt"),
            4.5,
            25.0,
            80.0,
            1.2,
        )
        _write_summary(
            os.path.join(run_dir, f"statistics_summary{SUFFIX_D10_PLUS}.txt"),
            14.0,
            45.0,
            20.0,
            1.05,
        )

        _write_vessels(os.path.join(run_dir, "generate_vessel.txt"), 10)
        _write_vessels(os.path.join(run_dir, f"generate_vessel{DIAMETER_SUFFIX}.txt"), 8)
        _write_vessels(os.path.join(run_dir, f"generate_vessel{SUFFIX_D10_PLUS}.txt"), 2)

        cases = [
            ("", "cross_sample_summary_full.txt", "全量 full", "5.5000"),
            (DIAMETER_SUFFIX, "cross_sample_summary_d0-10um.txt", "直径 0-10 um", "4.5000"),
            (SUFFIX_D10_PLUS, "cross_sample_summary_d10+um.txt", "直径 >=10 um", "14.0000"),
        ]
        for suffix, filename, label, expected_diameter in cases:
            path = generate_cross_sample_summary(tmpdir, suffix=suffix)
            assert path == os.path.join(tmpdir, filename)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert label in content
            assert "sample_001" in content
            assert expected_diameter in content
    finally:
        shutil.rmtree(tmpdir)
