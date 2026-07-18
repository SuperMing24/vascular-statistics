"""Render coordinate-audit images for manual skeleton cuts.

The renderer is deliberately read-only. It compares each adjusted graph with
the corresponding graph in the source axis-fixed experiment and visualizes
the exact masks used by :mod:`vascular_statistics.manual_cut`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
import textwrap
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

from vascular_statistics.manual_cut import (
    CutLayer,
    _parse_pos,
    bad_region_node_mask,
    load_cut_annotation,
    points_in_polygon_including_boundary,
    polygon_pixel_mask,
)


@dataclass(frozen=True)
class GraphGeometry:
    positions_xyz: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class ReviewSummary:
    sample_key: str
    output_path: str
    run_count: int
    panel_count: int
    all_run_positions_match: bool


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def discover_manual_cut_metadata(experiment_root: str) -> list[str]:
    """Return all ``manual_cut_meta.json`` paths below an experiment root."""
    root = os.path.abspath(os.path.expanduser(experiment_root))
    if not os.path.isdir(root):
        raise ValueError(f"experiment root does not exist: {root}")
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "manual_cut":
            if "manual_cut_meta.json" in filenames:
                paths.append(os.path.join(dirpath, "manual_cut_meta.json"))
            dirnames[:] = []
    return sorted(paths)


def read_pajek_geometry(path: str) -> GraphGeometry:
    graph = nx.read_pajek(path)
    nodes = list(graph.nodes())
    node_index = {node: index for index, node in enumerate(nodes)}
    positions = np.asarray([
        _parse_pos(graph.nodes[node].get("pos", graph.nodes[node].get("Pos")))
        for node in nodes
    ], dtype=float).reshape((-1, 3))
    edge_rows = [
        (node_index[first], node_index[second])
        for first, second in graph.edges()
    ]
    edges = np.asarray(edge_rows, dtype=int).reshape((-1, 2))
    return GraphGeometry(positions_xyz=positions, edges=edges)


def _shape_dhw(sample_dir: str, meta: dict[str, Any]) -> tuple[int, int, int]:
    retention = meta.get("retention")
    if isinstance(retention, dict):
        raw = retention.get("shape_dhw")
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            shape = tuple(int(value) for value in raw)
            if min(shape) > 0:
                return shape

    sample_meta = _load_json(os.path.join(sample_dir, "sample_metadata.json"))
    raw = sample_meta.get("stack_properties", {}).get("shape")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"missing stack_properties.shape=[H,W,D]: {sample_dir}")
    height, width, depth = (int(value) for value in raw)
    if min(depth, height, width) <= 0:
        raise ValueError(f"invalid sample shape: {raw}")
    return depth, height, width


def _find_runs(sample_dir: str) -> list[str]:
    runs: list[str] = []
    for name in sorted(os.listdir(sample_dir)):
        run_dir = os.path.join(sample_dir, name)
        if os.path.isdir(run_dir) and os.path.isfile(
            os.path.join(run_dir, "skeleton.pajek")
        ):
            runs.append(name)
    if not runs:
        raise ValueError(f"no run with skeleton.pajek: {sample_dir}")
    return runs


def _in_frame_range(positions: np.ndarray, start: int, end: int) -> np.ndarray:
    z = positions[:, 2]
    return (z >= start - 1.5 - 1e-9) & (z <= end - 0.5 + 1e-9)


def _position_counter(positions: np.ndarray) -> Counter[tuple[float, float, float]]:
    return Counter(tuple(row) for row in np.round(positions, decimals=9))


def _positions_match(expected: np.ndarray, actual: np.ndarray) -> bool:
    return _position_counter(expected) == _position_counter(actual)


def _wrap_sample_key(sample_key: str, width: int) -> str:
    lines: list[str] = []
    current = ""
    for component in sample_key.split("/"):
        candidate = component if not current else f"{current}/{component}"
        if current and len(candidate) > width:
            lines.append(current + "/")
            current = component
        else:
            current = candidate
    if current:
        lines.append(current)

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""])
    return "\n".join(wrapped)


def _draw_mask(
    axis: Any,
    height: int,
    width: int,
    polygon: Optional[np.ndarray],
) -> np.ndarray:
    mask = (
        np.zeros((height, width), dtype=bool)
        if polygon is None
        else polygon_pixel_mask(height, width, polygon)
    )
    rgba = np.empty((height, width, 4), dtype=float)
    rgba[~mask] = (0.88, 0.96, 0.88, 1.0)
    rgba[mask] = (1.0, 0.78, 0.74, 1.0)
    axis.imshow(
        rgba,
        origin="upper",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        interpolation="nearest",
        zorder=0,
    )
    return mask


def _edge_segments(geometry: GraphGeometry, in_z: np.ndarray) -> np.ndarray:
    if geometry.edges.size == 0:
        return np.empty((0, 2, 2), dtype=float)
    selected = geometry.edges[
        in_z[geometry.edges[:, 0]] & in_z[geometry.edges[:, 1]]
    ]
    return geometry.positions_xyz[selected, :2]


def _render_panel(
    axis: Any,
    source: GraphGeometry,
    result: GraphGeometry,
    layer: Optional[CutLayer],
    depth: int,
    height: int,
    width: int,
    run_name: str,
) -> None:
    if layer is None:
        name = "full volume"
        start, end = 1, depth
        polygon = None
    else:
        name = layer.name
        start, end = layer.start_frame, layer.end_frame
        polygon = layer.polygon_xy

    pixel_mask = _draw_mask(axis, height, width, polygon)
    source_in_z = _in_frame_range(source.positions_xyz, start, end)
    result_in_z = _in_frame_range(result.positions_xyz, start, end)
    if polygon is None:
        deleted_here = np.zeros(source.positions_xyz.shape[0], dtype=bool)
    else:
        deleted_here = np.zeros(source.positions_xyz.shape[0], dtype=bool)
        candidates = np.flatnonzero(source_in_z)
        deleted_here[candidates] = points_in_polygon_including_boundary(
            source.positions_xyz[candidates, :2], polygon
        )

    source_points = source.positions_xyz[source_in_z, :2]
    if source_points.size:
        axis.scatter(
            source_points[:, 0], source_points[:, 1], s=2.0,
            c="#6b7280", alpha=0.35, linewidths=0, zorder=2,
        )
    deleted_points = source.positions_xyz[deleted_here, :2]
    if deleted_points.size:
        axis.scatter(
            deleted_points[:, 0], deleted_points[:, 1], s=8.0,
            c="#c62828", alpha=0.8, marker="x", linewidths=0.45, zorder=4,
        )

    segments = _edge_segments(result, result_in_z)
    if segments.size:
        axis.add_collection(LineCollection(
            segments, colors="#075985", linewidths=0.35, alpha=0.55, zorder=3
        ))
    result_points = result.positions_xyz[result_in_z, :2]
    if result_points.size:
        axis.scatter(
            result_points[:, 0], result_points[:, 1], s=3.0,
            c="#0369a1", alpha=0.8, linewidths=0, zorder=5,
        )

    if polygon is not None:
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="#111827", linewidth=1.2, zorder=6)
        axis.scatter(
            polygon[:, 0], polygon[:, 1], s=22.0,
            c="#f59e0b", edgecolors="#111827", linewidths=0.5, zorder=7,
        )
        for index, (x_value, y_value) in enumerate(polygon, start=1):
            axis.annotate(
                str(index), (x_value, y_value), xytext=(3, 3),
                textcoords="offset points", fontsize=6, color="#111827", zorder=8,
            )

    polygon_bounds = [] if polygon is None else [
        float(np.min(polygon[:, 0])), float(np.max(polygon[:, 0])),
        float(np.min(polygon[:, 1])), float(np.max(polygon[:, 1])),
    ]
    xmin = min([-0.5] + ([polygon_bounds[0] - 0.5] if polygon_bounds else []))
    xmax = max([width - 0.5] + ([polygon_bounds[1] + 0.5] if polygon_bounds else []))
    ymin = min([-0.5] + ([polygon_bounds[2] - 0.5] if polygon_bounds else []))
    ymax = max([height - 0.5] + ([polygon_bounds[3] + 0.5] if polygon_bounds else []))
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymax, ymin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (skeleton/image coordinate)", fontsize=8)
    axis.set_ylabel("y (increases downward)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(color="#111827", alpha=0.12, linewidth=0.4)
    axis.set_title(f"{run_name} | {name} | Z frames {start}-{end}", fontsize=9)

    bad_pixels = int(np.count_nonzero(pixel_mask))
    annotation = (
        f"bad XY pixels: {bad_pixels:,}/{height * width:,}\n"
        f"source nodes in Z: {int(np.count_nonzero(source_in_z)):,}\n"
        f"deleted by this layer: {int(np.count_nonzero(deleted_here)):,}\n"
        f"final nodes in Z: {int(np.count_nonzero(result_in_z)):,}"
    )
    axis.text(
        0.01, 0.01, annotation, transform=axis.transAxes,
        ha="left", va="bottom", fontsize=6.5, color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.88,
              "boxstyle": "square,pad=0.3"},
        zorder=9,
    )


def render_sample_review(
    meta_path: str,
    output_path: Optional[str] = None,
    dpi: int = 150,
    overwrite: bool = False,
) -> ReviewSummary:
    """Render one sample review image and return validation information."""
    meta_path = os.path.abspath(meta_path)
    meta = _load_json(meta_path)
    manual_dir = os.path.dirname(meta_path)
    sample_dir = os.path.dirname(manual_dir)
    sample_key = str(meta.get("destination_sample_key") or os.path.basename(sample_dir))
    source_sample_dir = os.path.join(
        str(meta["source_experiment_root"]), str(meta["source_sample_relative_path"])
    )
    depth, height, width = _shape_dhw(sample_dir, meta)
    status = str(meta.get("status", ""))

    if status == "bad_region_removed":
        annotation_path = os.path.join(manual_dir, str(meta.get("annotation_file", "polygonInfo.mat")))
        annotation = load_cut_annotation(annotation_path)
        layers: Sequence[Optional[CutLayer]] = annotation.layers
        retention_fraction = float(meta["retention"]["retained_fraction"])
    elif status == "no_cut_keep_full":
        annotation = None
        layers = (None,)
        retention_fraction = 1.0
    else:
        raise ValueError(f"unknown manual cut status {status!r}: {meta_path}")

    run_names = _find_runs(sample_dir)
    source_graphs: dict[str, GraphGeometry] = {}
    result_graphs: dict[str, GraphGeometry] = {}
    run_matches: dict[str, bool] = {}
    for run_name in run_names:
        source_path = os.path.join(source_sample_dir, run_name, "skeleton.pajek")
        result_path = os.path.join(sample_dir, run_name, "skeleton.pajek")
        if not os.path.isfile(source_path):
            raise ValueError(f"source skeleton does not exist: {source_path}")
        source = read_pajek_geometry(source_path)
        result = read_pajek_geometry(result_path)
        source_graphs[run_name] = source
        result_graphs[run_name] = result
        if annotation is None:
            expected = source.positions_xyz
        else:
            deleted = bad_region_node_mask(source.positions_xyz, annotation.layers)
            expected = source.positions_xyz[~deleted]
        run_matches[run_name] = _positions_match(expected, result.positions_xyz)

    if output_path is None:
        output_path = os.path.join(manual_dir, "manual_cut_review.png")
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"review image already exists: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = len(run_names)
    columns = len(layers)
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(max(6.0, 4.8 * columns), max(5.6, 4.5 * rows + 1.1)),
        squeeze=False,
    )
    for row, run_name in enumerate(run_names):
        for column, layer in enumerate(layers):
            _render_panel(
                axes[row, column], source_graphs[run_name], result_graphs[run_name],
                layer, depth, height, width, run_name,
            )

    wrapped_sample_key = _wrap_sample_key(sample_key, width=max(60, 72 * columns))
    figure.text(
        0.5, 0.985, "Manual cut coordinate audit",
        ha="center", va="top", fontsize=12, fontweight="bold",
    )
    figure.text(
        0.5, 0.94, wrapped_sample_key,
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    match_label = "MATCH" if all(run_matches.values()) else "MISMATCH"
    figure.text(
        0.5,
        0.125,
        f"shape [D,H,W]={depth,height,width} | retained volume={retention_fraction:.2%}\n"
        f"result vs computed retained positions: {match_label}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    legend = [
        Patch(facecolor="#e0f5e0", edgecolor="none", label="retained XY region"),
        Patch(facecolor="#ffc7bd", edgecolor="none", label="bad XY region (deleted)"),
        Line2D([], [], color="#6b7280", marker=".", linestyle="None", label="source nodes"),
        Line2D([], [], color="#c62828", marker="x", linestyle="None", label="deleted nodes"),
        Line2D([], [], color="#0369a1", marker=".", linestyle="-", label="final skeleton"),
        Line2D([], [], color="#f59e0b", marker="o", markeredgecolor="#111827",
               linestyle="-", label="polygon and selected points"),
    ]
    figure.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3, fontsize=8,
    )
    path_lines = wrapped_sample_key.count("\n") + 1
    top = 0.81 if path_lines > 1 else 0.86
    figure.tight_layout(rect=(0, 0.18, 1, top))

    temporary = output_path + ".tmp"
    try:
        figure.savefig(temporary, dpi=dpi, format="png", facecolor="white")
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        if os.path.exists(temporary):
            os.remove(temporary)

    return ReviewSummary(
        sample_key=sample_key,
        output_path=output_path,
        run_count=len(run_names),
        panel_count=rows * columns,
        all_run_positions_match=all(run_matches.values()),
    )
