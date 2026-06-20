"""子范围统计模块的合成数据逻辑测试（0-10 um）。"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vascular_statistics.subrange_stats import (
    write_subrange_stats,
    generate_cross_sample_summary,
)
from vascular_statistics.aggregate_stats import write_aggregate_stats


def test_end_to_end():
    """端到端测试：合成数据 -> per-run 子范围统计 -> 聚合 -> 跨样本汇总。"""
    tmpdir = tempfile.mkdtemp()
    try:
        sample_dir = os.path.join(
            tmpdir, "TEST_GROUP", "20240101_A001_D0", "sample_001"
        )

        for run_id in ["run_20240601_120000", "run_20240601_130000"]:
            run_dir = os.path.join(sample_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            # 5 段：
            #   0: radius=1.0 (< 2.5, 3 nodes) -> subrange
            #   1: radius=3.0 (>= 2.5)         -> NOT subrange
            #   2: radius=1.5 (< 2.5, 3 nodes) -> subrange
            #   3: radius=0.5 (< 2.5, 2 nodes) -> NOT subrange (nodes < 3)
            #   4: radius=2.0 (< 2.5, 3 nodes) -> subrange
            radii = [1.0, 3.0, 1.5, 0.5, 2.0]
            lengths = [10.0, 20.0, 15.0, 8.0, 12.0]
            torts = [1.1, 1.2, 1.3, 1.05, 1.4]
            nodes = ["1 2 3", "4 5 6", "7 8 9", "10 11", "12 13 14"]

            for key, data in [
                ("radius", radii),
                ("path_length", lengths),
                ("tortuosity", torts),
            ]:
                with open(
                    os.path.join(run_dir, f"generate_vessel_{key}_d10+um.txt"), "w"
                ) as f:
                    for v in data:
                        f.write(f"{v}\n")
            with open(os.path.join(run_dir, "generate_vessel_d10+um.txt"), "w") as f:
                for seq in nodes:
                    f.write(f"{seq}\n")

            # C++ 主统计文件（标记 run 已完成）
            with open(
                os.path.join(run_dir, "statistics_summary_d10+um.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(
                    "Vascular_Statistics\n"
                    "平均直径 (um): 20.0\n"
                    "平均长度 (um): 30.0\n"
                    "段密度 (seg/mm^3): 10.0\n"
                    "平均弯曲度: 1.2\n"
                )

        # sample_metadata.json
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, "sample_metadata.json"), "w") as f:
            json.dump(
                {
                    "sample_key": "TEST_GROUP/20240101_A001_D0/sample_001",
                    "path_parsed": {
                        "group": "TEST_GROUP",
                        "batch_id": "A001",
                        "daypoint": "D0",
                    },
                    "spatial": {"tissue_volume_mm3": 0.078},
                },
                f,
            )

        # ---- Test 1: write_subrange_stats (per-run) ----
        for run_id in ["run_20240601_120000", "run_20240601_130000"]:
            run_dir = os.path.join(sample_dir, run_id)
            result = write_subrange_stats(run_dir, 0.078)
            assert result is not None, f"write_subrange_stats returned None for {run_id}"
            assert os.path.exists(
                os.path.join(run_dir, "statistics_summary_d0-10um.txt")
            ), "Missing statistics_summary_d0-10um.txt"

        print("PASS: test_write_subrange_stats")

        # ---- Test 2: aggregate (suffix=_d0-10um) ----
        agg_path = write_aggregate_stats(sample_dir, suffix="_d0-10um")
        assert agg_path is not None, "aggregate returned None"
        assert os.path.exists(agg_path), f"Missing {agg_path}"
        print("PASS: test_aggregate_subrange")

        # ---- Test 3: cross-sample summary ----
        summary_path = generate_cross_sample_summary(tmpdir)
        assert summary_path is not None, "cross-sample summary returned None"
        assert os.path.exists(summary_path), f"Missing {summary_path}"

        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "sample_001" in content, "Sample key not found in summary"
        print("PASS: test_cross_sample_summary")

        # ---- Manual verification of computed values ----
        # Expected subrange segments: [0, 2, 4]
        #   radii: [1.0, 1.5, 2.0] -> avg_radius=1.5, avg_diameter=6.0
        #   lengths: [10, 15, 12] -> avg=12.333... * 2.0 = 24.67 (legacy x2)
        #   torts: [1.1, 1.3, 1.4] -> avg=1.2666...
        #   density: 3/0.078 = 38.4615...
        with open(
            os.path.join(
                sample_dir, "run_20240601_120000", "statistics_summary_d0-10um.txt"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            run_content = f.read()

        assert "6.000000" in run_content, f"Expected avg_diameter=6.0, got: {run_content}"
        print("PASS: avg_diameter = 6.0 (radius 1.5 * 4)")

        print("\nAll 4 tests passed.")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_end_to_end()
