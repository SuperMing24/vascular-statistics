from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vascular_statistics.manualkeep_analysis import (  # noqa: E402
    bh,
    bootstrap_ci,
    descriptive_table,
    rank_biserial,
    parse_summary,
)


class ManualKeepAnalysisTests(unittest.TestCase):
    def test_parse_summary_accepts_full_and_ascii_units(self):
        text = (
            "平均直径 (um): 5.2\n平均长度 (um): 31.0\n"
            "段密度 (seg/mm^3): 20000\n平均弯曲度: 1.08\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "statistics_summary.txt"
            path.write_text(text, encoding="utf-8")
            parsed = parse_summary(path)
        self.assertEqual(parsed["avg_diameter_um"], 5.2)
        self.assertEqual(parsed["segment_density_per_mm3"], 20000)

    def test_rank_biserial_direction(self):
        self.assertEqual(rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)
        self.assertEqual(rank_biserial(np.array([-1.0, -2.0])), -1.0)
        self.assertEqual(rank_biserial(np.zeros(3)), 0.0)

    def test_bh_is_monotone_in_p_order(self):
        adjusted = bh([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)

    def test_bootstrap_is_reproducible(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        first = bootstrap_ci(values, np.random.default_rng(7), 100)
        second = bootstrap_ci(values, np.random.default_rng(7), 100)
        self.assertEqual(first, second)

    def test_descriptive_table_keeps_subgroup(self):
        row = {
            "experimenter": "xiaoqian", "group": "ACTH_angiogram",
            "subgroup": "ACTH_HNK", "study_day_label": "D15",
            "population": "full", "metric": "avg_diameter_um",
            "final_value": 5.2,
        }
        result = descriptive_table([row])
        subgroup = [item for item in result if item["level"] == "subgroup"]
        self.assertEqual(subgroup[0]["subgroup"], "ACTH_HNK")

if __name__ == "__main__":
    unittest.main()
