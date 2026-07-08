"""GUI stack display axis transform tests."""
import importlib.util
import os

import numpy as np


_HELPER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "python",
    "vascular_statistics",
    "vascgraph",
    "VascGraph",
    "GraphLab",
    "StackDisplayTransform.py",
)
_spec = importlib.util.spec_from_file_location("stack_display_transform", _HELPER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
display_stack_for_graph = _mod.display_stack_for_graph


def test_display_stack_for_graph_swaps_xy_and_flips_z():
    raw = np.zeros((2, 3, 4), dtype=np.int16)
    raw[1, 2, 0] = 7
    raw[0, 1, 3] = 9

    display = display_stack_for_graph(raw)

    assert display.shape == (3, 2, 4)
    assert display[2, 1, 3] == 7
    assert display[1, 0, 0] == 9
    assert display.flags["C_CONTIGUOUS"]


def test_display_stack_for_graph_leaves_non_3d_arrays_unchanged():
    raw = np.zeros((2, 3), dtype=np.uint8)

    display = display_stack_for_graph(raw)

    assert display is raw
