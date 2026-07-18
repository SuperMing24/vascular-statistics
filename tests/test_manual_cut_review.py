from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import networkx as nx
import numpy as np
from scipy.io import savemat
from matplotlib.ft2font import FT2Font


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vascular_statistics.manual_cut import CutLayer, cut_pajek_graph
from vascular_statistics.manual_cut_review import (
    BUNDLED_CJK_FONT_PATH,
    _layout_review_figure,
    _review_regions,
    _wrap_sample_key,
    discover_manual_cut_metadata,
    read_pajek_geometry,
    render_sample_review,
)


class ManualCutReviewTests(unittest.TestCase):
    def test_bundled_font_covers_all_non_ascii_review_text(self):
        module_path = Path(BUNDLED_CJK_FONT_PATH).parents[2] / "manual_cut_review.py"
        source = module_path.read_text(encoding="utf-8")
        font = FT2Font(BUNDLED_CJK_FONT_PATH)

        missing = sorted({
            character for character in source
            if ord(character) > 127 and font.get_char_index(ord(character)) == 0
        })

        self.assertEqual(missing, [])

    def test_wraps_long_sample_key_without_spaces(self):
        sample_key = (
            "xiaoqian/magraine_angiogram/Saline/"
            "20250911_A85_D9+8_angiogram_crop_70_117"
        )

        wrapped = _wrap_sample_key(sample_key, width=60)

        self.assertIn("\n", wrapped)
        self.assertTrue(all(len(line) <= 60 for line in wrapped.splitlines()))

    def test_compact_multi_panel_layout_has_no_overlaps(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        for rows, columns in ((1, 3), (3, 2)):
            with self.subTest(rows=rows, columns=columns):
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(
                        max(6.0, 4.8 * columns),
                        max(5.6, 4.5 * rows + 1.1),
                    ),
                    squeeze=False,
                )
                try:
                    for axis in axes.flat:
                        axis.set_xlim(-0.5, 511.5)
                        axis.set_ylim(511.5, -0.5)
                        axis.set_aspect("equal", adjustable="box")
                        axis.set_xlabel("x（骨架/图像坐标）", fontsize=8)
                        axis.set_ylabel("y（向下递增）", fontsize=8)
                        axis.set_title(
                            "run_20260629_010053 | p2 | Z 帧 29-45",
                            fontsize=9,
                        )
                    title = figure.text(
                        0.5, 0.985, "人工裁剪坐标核查",
                        ha="center", va="top", fontsize=12,
                        fontweight="bold",
                    )
                    sample = figure.text(
                        0.5, 0.94,
                        _wrap_sample_key(
                            "xiaoqian/magraine_angiogram/Saline/"
                            "20250911_A88_D9+8_angiogram_crop_75_102",
                            width=60,
                        ),
                        ha="center", va="top", fontsize=10,
                        fontweight="bold",
                    )
                    summary = figure.text(
                        0.5, 0.0,
                        "shape [D,H,W]=(57,512,512) | 保留体积=68.99%\n"
                        "结果与计算保留坐标：一致",
                        ha="center", va="bottom", fontsize=9,
                    )
                    legend = figure.legend(
                        handles=[
                            Patch(label="保留 XY 区域"),
                            Patch(label="裁剪 XY 区域（已删除）"),
                            Patch(label="源骨架点"),
                            Patch(label="已删除骨架点"),
                            Patch(label="最终骨架"),
                            Patch(label="多边形与人工选点"),
                        ],
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.012),
                        ncol=3,
                        fontsize=8,
                    )

                    _layout_review_figure(
                        figure, axes, title, sample, summary, legend
                    )
                finally:
                    plt.close(figure)

    def test_splits_overlapping_layers_into_exact_union_regions(self):
        polygon = np.asarray([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=float)
        p1 = CutLayer("p1", polygon, polygon + 0.5, 1, 2)
        p2 = CutLayer("p2", polygon, polygon + 0.5, 2, 4)

        regions = _review_regions((p1, p2), depth=5)

        self.assertEqual(
            [(region.name, region.start_frame, region.end_frame) for region in regions],
            [("p1", 1, 1), ("p1+p2", 2, 2), ("p2", 3, 4)],
        )

    def test_reads_empty_pajek_as_n_by_three_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.pajek")
            nx.write_pajek(nx.Graph(), path)

            geometry = read_pajek_geometry(path)

            self.assertEqual(geometry.positions_xyz.shape, (0, 3))
            self.assertEqual(geometry.edges.shape, (0, 2))

    def test_renders_source_deleted_and_final_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = os.path.join(tmp, "source")
            experiment_root = os.path.join(tmp, "experiment")
            relative = os.path.join("group", "sample")
            run_name = "run_1"
            backup_name = "_badz2_run_1"
            source_run = os.path.join(source_root, relative, run_name)
            source_backup = os.path.join(source_root, relative, backup_name)
            result_sample = os.path.join(experiment_root, "operator", relative)
            result_run = os.path.join(result_sample, run_name)
            result_backup = os.path.join(result_sample, backup_name)
            manual_dir = os.path.join(result_sample, "manual_cut")
            os.makedirs(source_run)
            os.makedirs(source_backup)
            os.makedirs(result_run)
            os.makedirs(result_backup)
            os.makedirs(manual_dir)

            graph = nx.Graph()
            for node, position in enumerate(((0, 0, 0), (1, 1, 0), (3, 3, 0))):
                graph.add_node(str(node), pos=f"[{position[0]} {position[1]} {position[2]}]", r="1")
            graph.add_edges_from((("0", "1"), ("1", "2")))
            source_pajek = os.path.join(source_run, "skeleton.pajek")
            result_pajek = os.path.join(result_run, "skeleton.pajek")
            nx.write_pajek(graph, source_pajek)
            nx.write_pajek(graph, os.path.join(source_backup, "skeleton.pajek"))
            nx.write_pajek(graph, os.path.join(result_backup, "skeleton.pajek"))

            polygon = np.asarray([
                [0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5],
            ])
            save_info = {
                "p1": {
                    "polygonPosition": polygon,
                    "editStartFrame": 1,
                    "editEndFrame": 1,
                    "numFrames": 1,
                    "fileName": "sample.mat",
                },
                "p2": {"polygonPosition": np.empty((0, 2))},
                "p3": {"polygonPosition": np.empty((0, 2))},
                "p4": {"polygonPosition": np.empty((0, 2))},
            }
            annotation_path = os.path.join(manual_dir, "polygonInfo.mat")
            savemat(annotation_path, {"saveInfo": save_info})

            from vascular_statistics.manual_cut import load_cut_annotation
            annotation = load_cut_annotation(annotation_path)
            cut_pajek_graph(source_pajek, annotation.layers, result_pajek)

            with open(os.path.join(result_sample, "sample_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump({"stack_properties": {"shape": [4, 4, 1]}}, handle)
            meta = {
                "schema_version": "1.0",
                "source_experiment_root": source_root,
                "source_sample_relative_path": relative.replace("\\", "/"),
                "destination_sample_key": "operator/group/sample",
                "status": "quality_region_retained",
                "selection_semantics": "polygon_inside_retained",
                "annotation_file": "polygonInfo.mat",
                "retention": {"shape_dhw": [1, 4, 4], "retained_fraction": 0.0625},
            }
            meta_path = os.path.join(manual_dir, "manual_cut_meta.json")
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                summary = render_sample_review(meta_path, dpi=72)
            glyph_warnings = [
                str(warning.message) for warning in caught
                if "Glyph" in str(warning.message)
            ]

            self.assertEqual(glyph_warnings, [])
            self.assertTrue(summary.all_run_positions_match)
            self.assertEqual(summary.run_count, 1)
            self.assertEqual(summary.panel_count, 1)
            self.assertTrue(os.path.isfile(summary.output_path))
            self.assertGreater(os.path.getsize(summary.output_path), 1000)
            self.assertEqual(
                discover_manual_cut_metadata(experiment_root), [meta_path]
            )


if __name__ == "__main__":
    unittest.main()
