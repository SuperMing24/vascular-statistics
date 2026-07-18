from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments"))
import build_manualcut_experiment_20260716 as manualcut_builder

from vascular_statistics.manual_cut import (
    CutLayer,
    POLYGON_INSIDE_DELETED,
    compute_retention,
    cut_pajek_graph,
    deleted_region_node_mask,
    load_cut_annotation,
    points_in_polygon_including_boundary,
    points_strictly_inside_polygon,
)


def layer(name, raw_polygon, start=1, end=1):
    raw = np.asarray(raw_polygon, dtype=float)
    return CutLayer(name, raw - 0.5, raw, start, end)


class ManualCutGeometryTests(unittest.TestCase):
    def test_polygon_boundary_is_deleted(self):
        polygon = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        points = np.asarray([
            [1, 1], [0, 1], [2, 2], [2.1, 1], [-0.1, 0],
        ])
        actual = points_in_polygon_including_boundary(points, polygon)
        np.testing.assert_array_equal(actual, [True, True, True, False, False])

    def test_sample_metadata_hwd_shape_is_converted_to_dhw(self):
        sample = manualcut_builder.SampleRecord(
            experimenter="xiaoqian",
            source_root="source",
            source_dir="source/sample",
            relative_path="group/date/sample",
            sample_key="group/date/sample",
            input_rel_path="group/date/sample.mat",
            metadata={"stack_properties": {"shape": [3, 4, 2]}},
            run_names=("run_1",),
        )
        annotation = type("Annotation", (), {
            "num_frames": 2,
            "layers": (
                layer(
                    "p1",
                    [[0.5, 0.5], [4.5, 0.5], [4.5, 3.5], [0.5, 3.5]],
                    1,
                    2,
                ),
            ),
        })()
        self.assertEqual(
            manualcut_builder.resolve_shape_dhw(sample, annotation), (2, 3, 4)
        )

    def test_retained_volume_uses_union_across_overlapping_layers(self):
        first = layer("p1", [[0, 0], [3, 0], [3, 3], [0, 3]], 1, 1)
        second = layer("p2", [[2, 0], [3, 0], [3, 3], [2, 3]], 1, 2)
        result = compute_retention((2, 3, 3), [first, second])
        self.assertEqual(result.total_voxels, 18)
        self.assertEqual(result.retained_voxels, 12)
        self.assertEqual(result.deleted_voxels, 6)
        self.assertEqual(result.selection_semantics, "polygon_inside_retained")

    def test_unannotated_z_and_strict_polygon_interior_are_retained(self):
        selected = layer(
            "p1", [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]], 1, 1
        )
        result = compute_retention((2, 3, 3), [selected])
        self.assertEqual(result.retained_voxels, 10)
        self.assertEqual(result.deleted_voxels, 8)

        points = np.asarray([[1, 1], [0, 1], [2, 2], [2.1, 1]])
        np.testing.assert_array_equal(
            points_strictly_inside_polygon(points, selected.polygon_xy),
            [True, False, False, False],
        )

    def test_legacy_bad_region_semantics_remain_available_for_audit(self):
        selected = layer(
            "p1", [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]]
        )
        positions = np.asarray([[1, 1, 0], [0, 1, 0], [3, 3, 0]], dtype=float)
        actual = deleted_region_node_mask(
            positions, [selected], selection_semantics=POLYGON_INSIDE_DELETED
        )
        np.testing.assert_array_equal(actual, [True, True, False])

    def test_pajek_cut_removes_incident_edges_and_relabels(self):
        graph = nx.Graph()
        for node, x in enumerate((0.0, 1.0, 2.0)):
            graph.add_node(str(node), pos=f"[{x} 1 0]", r=str(node + 1), type="2")
        graph.add_edges_from([("0", "1"), ("1", "2")])
        selected = layer(
            "p1", [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]]
        )

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.pajek")
            output = os.path.join(tmp, "output.pajek")
            nx.write_pajek(graph, source)
            result = cut_pajek_graph(source, [selected], output)
            cut_graph = nx.read_pajek(output)

        self.assertEqual(result.deleted_nodes, 2)
        self.assertEqual(result.deleted_edges, 2)
        self.assertEqual(sorted(cut_graph.nodes()), ["0"])
        self.assertEqual(cut_graph.number_of_edges(), 0)
        self.assertEqual(cut_graph.nodes["0"]["r"], "2")


class ManualCutMatTests(unittest.TestCase):
    def test_loads_save_info_and_shifts_matlab_coordinates(self):
        polygon = np.asarray([[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]])
        empty = np.empty((0, 2))
        save_info = {
            "p1": {
                "polygonPosition": polygon,
                "editStartFrame": 1,
                "editEndFrame": 2,
                "startFrameValue": 1,
                "endFrameValue": 2,
                "numFrames": 2,
                "fileName": "angiogram_crop_1_2.mat",
                "txtFile": r"C:\data\A1\angiogram_crop_1_2.txt",
            },
            "p2": {
                "polygonPosition": empty,
                "numFrames": 99,
                "fileName": "unrelated_sample.mat",
                "txtFile": r"C:\data\OTHER\unrelated_sample.mat",
            },
            "p3": {"polygonPosition": empty, "numFrames": 2},
            "p4": {"polygonPosition": empty, "numFrames": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "polygonInfo.mat")
            savemat(path, {"saveInfo": save_info})
            annotation = load_cut_annotation(path)

        self.assertEqual(annotation.num_frames, 2)
        self.assertEqual(len(annotation.layers), 1)
        np.testing.assert_allclose(annotation.layers[0].polygon_xy[0], [0.0, 0.0])
        self.assertIn("angiogram_crop_1_2", annotation.reference_stems)
        self.assertNotIn("unrelated_sample", annotation.reference_stems)


if __name__ == "__main__":
    unittest.main()
