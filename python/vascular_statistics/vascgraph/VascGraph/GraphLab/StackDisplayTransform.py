#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Display-only stack coordinate transform for GraphLab/Mayavi.

Graph nodes are kept in the canonical [x, y, z] coordinate system.  The raw
stack loaded from .mat is not modified; only the ndarray sent to Mayavi is
mapped so the displayed volume overlays the graph in the GUI.
"""

import numpy as np


def display_stack_for_graph(array):
    """Return stack data mapped into GraphPlot's [x, y, z] display space.

    Mapping: display[x, y, z] = raw[y, x, zmax - z].
    This applies the observed GUI alignment correction: swap x/y and reverse z.
    """
    arr = np.asarray(array)
    if arr.ndim != 3:
        return arr
    return np.ascontiguousarray(np.transpose(arr, (1, 0, 2))[:, :, ::-1])
