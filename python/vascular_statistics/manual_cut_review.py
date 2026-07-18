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
from matplotlib import font_manager

matplotlib.use("Agg")
BUNDLED_CJK_FONT_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "fonts",
    "NotoSansCJKSC-ManualCutSubset.otf",
)
if not os.path.isfile(BUNDLED_CJK_FONT_PATH):
    raise RuntimeError(f"bundled CJK review font is missing: {BUNDLED_CJK_FONT_PATH}")
font_manager.fontManager.addfont(BUNDLED_CJK_FONT_PATH)
BUNDLED_CJK_FONT_FAMILY = font_manager.FontProperties(
    fname=BUNDLED_CJK_FONT_PATH
).get_name()
matplotlib.rcParams["font.family"] = [BUNDLED_CJK_FONT_FAMILY]
matplotlib.rcParams["font.sans-serif"] = [BUNDLED_CJK_FONT_FAMILY]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

from vascular_statistics.manual_cut import (
    CutLayer,
    POLYGON_INSIDE_DELETED,
    POLYGON_INSIDE_RETAINED,
    _parse_pos,
    deleted_region_node_mask,
    load_cut_annotation,
    polygon_pixel_mask,
)


@dataclass(frozen=True)
class GraphGeometry:
    positions_xyz: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class ReviewRegion:
    name: str
    start_frame: int
    end_frame: int
    layers: tuple[CutLayer, ...]


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
        if name.startswith("run_") and os.path.isdir(run_dir) and os.path.isfile(
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


def _review_regions(layers: Sequence[CutLayer], depth: int) -> tuple[ReviewRegion, ...]:
    """Split Z into intervals with a constant union of active polygons."""
    if not layers:
        return (ReviewRegion("全量保留", 1, depth, ()),)
    boundaries = {1, depth + 1}
    for layer in layers:
        boundaries.add(layer.start_frame)
        boundaries.add(layer.end_frame + 1)
    ordered = sorted(boundaries)
    regions: list[ReviewRegion] = []
    for start, end_exclusive in zip(ordered, ordered[1:]):
        end = end_exclusive - 1
        active = tuple(
            layer for layer in layers
            if layer.start_frame <= start and layer.end_frame >= end
        )
        if active:
            regions.append(ReviewRegion(
                name="+".join(layer.name for layer in active),
                start_frame=start,
                end_frame=end,
                layers=active,
            ))
    return tuple(regions)


def _draw_mask(
    axis: Any,
    height: int,
    width: int,
    region: ReviewRegion,
    selection_semantics: str,
) -> np.ndarray:
    if not region.layers:
        retained = np.ones((height, width), dtype=bool)
    else:
        selected = np.zeros((height, width), dtype=bool)
        for layer in region.layers:
            selected |= polygon_pixel_mask(
                height,
                width,
                layer.polygon_xy,
                include_boundary=selection_semantics == POLYGON_INSIDE_DELETED,
            )
        retained = (
            selected
            if selection_semantics == POLYGON_INSIDE_RETAINED
            else ~selected
        )
    rgba = np.empty((height, width, 4), dtype=float)
    rgba[retained] = (0.88, 0.96, 0.88, 1.0)
    rgba[~retained] = (1.0, 0.78, 0.74, 1.0)
    axis.imshow(
        rgba,
        origin="upper",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        interpolation="nearest",
        zorder=0,
    )
    return retained


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
    region: ReviewRegion,
    all_layers: Sequence[CutLayer],
    selection_semantics: str,
    height: int,
    width: int,
    run_name: str,
) -> None:
    start, end = region.start_frame, region.end_frame
    retained_pixels = _draw_mask(
        axis, height, width, region, selection_semantics
    )
    source_in_z = _in_frame_range(source.positions_xyz, start, end)
    result_in_z = _in_frame_range(result.positions_xyz, start, end)
    deleted_here = source_in_z & deleted_region_node_mask(
        source.positions_xyz, all_layers, selection_semantics
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

    for layer in region.layers:
        polygon = layer.polygon_xy
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="#111827", linewidth=1.2, zorder=6)
        axis.scatter(
            polygon[:, 0], polygon[:, 1], s=22.0,
            c="#f59e0b", edgecolors="#111827", linewidths=0.5, zorder=7,
        )
        multiple = len(region.layers) > 1
        for index, (x_value, y_value) in enumerate(polygon, start=1):
            label = f"{layer.name}-{index}" if multiple else str(index)
            axis.annotate(
                label, (x_value, y_value), xytext=(3, 3),
                textcoords="offset points", fontsize=6, color="#111827", zorder=8,
            )

    if region.layers:
        polygons = np.vstack([layer.polygon_xy for layer in region.layers])
        polygon_bounds = [
            float(np.min(polygons[:, 0])), float(np.max(polygons[:, 0])),
            float(np.min(polygons[:, 1])), float(np.max(polygons[:, 1])),
        ]
    else:
        polygon_bounds = []
    xmin = min([-0.5] + ([polygon_bounds[0] - 0.5] if polygon_bounds else []))
    xmax = max([width - 0.5] + ([polygon_bounds[1] + 0.5] if polygon_bounds else []))
    ymin = min([-0.5] + ([polygon_bounds[2] - 0.5] if polygon_bounds else []))
    ymax = max([height - 0.5] + ([polygon_bounds[3] + 0.5] if polygon_bounds else []))
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymax, ymin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x（骨架/图像坐标）", fontsize=8)
    axis.set_ylabel("y（向下递增）", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(color="#111827", alpha=0.12, linewidth=0.4)
    axis.set_title(
        f"{run_name} | {region.name} | Z 帧 {start}-{end}", fontsize=9
    )

    annotation = (
        f"保留 XY 像素：{int(np.count_nonzero(retained_pixels)):,}/{height * width:,}\n"
        f"Z 范围源骨架点：{int(np.count_nonzero(source_in_z)):,}\n"
        f"本范围删除骨架点：{int(np.count_nonzero(deleted_here)):,}\n"
        f"Z 范围最终骨架点：{int(np.count_nonzero(result_in_z)):,}"
    )
    axis.text(
        0.01, 0.01, annotation, transform=axis.transAxes,
        ha="left", va="bottom", fontsize=6.5, color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.88,
              "boxstyle": "square,pad=0.3"},
        zorder=9,
    )


def _boxes_overlap(first: Any, second: Any, padding: float = 0.0) -> bool:
    return (
        first.x0 < second.x1 + padding
        and first.x1 + padding > second.x0
        and first.y0 < second.y1 + padding
        and first.y1 + padding > second.y0
    )


def _layout_review_figure(
    figure: Any,
    axes: np.ndarray,
    title_artist: Any,
    sample_artist: Any,
    summary_artist: Any,
    legend_artist: Any,
) -> None:
    """Place external text bands and reject cross-region overlaps."""
    gap_pixels = figure.dpi * 6.0 / 72.0
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_height = figure.bbox.height

    title_box = title_artist.get_window_extent(renderer)
    sample_artist.set_position((0.5, (title_box.y0 - gap_pixels) / figure_height))

    legend_box = legend_artist.get_window_extent(renderer)
    summary_artist.set_position((
        0.5,
        (legend_box.y1 + gap_pixels) / figure_height,
    ))

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    sample_box = sample_artist.get_window_extent(renderer)
    summary_box = summary_artist.get_window_extent(renderer)
    bottom = (summary_box.y1 + gap_pixels) / figure_height
    top = (sample_box.y0 - gap_pixels) / figure_height
    if bottom >= top:
        raise RuntimeError("review layout has no room for plot panels")

    for _ in range(4):
        figure.tight_layout(rect=(0, bottom, 1, top))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel_boxes = [
            axis.get_tightbbox(renderer)
            for axis in axes.flat
            if axis.get_visible()
        ]
        bottom_shortfall = max(
            0.0,
            summary_artist.get_window_extent(renderer).y1
            + gap_pixels
            - min(box.y0 for box in panel_boxes),
        )
        top_shortfall = max(
            0.0,
            max(box.y1 for box in panel_boxes)
            + gap_pixels
            - sample_artist.get_window_extent(renderer).y0,
        )
        if bottom_shortfall < 0.5 and top_shortfall < 0.5:
            break
        bottom += (bottom_shortfall + 1.0) / figure_height
        top -= (top_shortfall + 1.0) / figure_height
        if bottom >= top:
            raise RuntimeError("review layout has no room for plot panels")

    for _ in range(4):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel_grid = [
            [axis.get_tightbbox(renderer) for axis in row]
            for row in axes
        ]
        vertical_shortfall = max(
            [
                panel_grid[row + 1][column].y1
                + 1.0
                - panel_grid[row][column].y0
                for row in range(axes.shape[0] - 1)
                for column in range(axes.shape[1])
            ]
            or [0.0]
        )
        horizontal_shortfall = max(
            [
                panel_grid[row][column].x1
                + 1.0
                - panel_grid[row][column + 1].x0
                for row in range(axes.shape[0])
                for column in range(axes.shape[1] - 1)
            ]
            or [0.0]
        )
        if vertical_shortfall < 0.5 and horizontal_shortfall < 0.5:
            break
        axis_boxes = [axis.get_window_extent(renderer) for axis in axes.flat]
        adjustments: dict[str, float] = {}
        if vertical_shortfall >= 0.5:
            average_height = float(np.mean([box.height for box in axis_boxes]))
            adjustments["hspace"] = (
                figure.subplotpars.hspace
                + (vertical_shortfall + 1.0) / average_height
            )
        if horizontal_shortfall >= 0.5:
            average_width = float(np.mean([box.width for box in axis_boxes]))
            adjustments["wspace"] = (
                figure.subplotpars.wspace
                + (horizontal_shortfall + 1.0) / average_width
            )
        figure.subplots_adjust(**adjustments)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = {
        "title": title_artist.get_window_extent(renderer),
        "sample": sample_artist.get_window_extent(renderer),
        "summary": summary_artist.get_window_extent(renderer),
        "legend": legend_artist.get_window_extent(renderer),
    }
    panel_boxes = [
        axis.get_tightbbox(renderer)
        for axis in axes.flat
        if axis.get_visible()
    ]
    overlaps: list[str] = []
    for first, second in (("title", "sample"), ("summary", "legend")):
        if _boxes_overlap(boxes[first], boxes[second], padding=1.0):
            overlaps.append(f"{first}/{second}")
    for index, panel_box in enumerate(panel_boxes, start=1):
        for name in ("title", "sample", "summary", "legend"):
            if _boxes_overlap(panel_box, boxes[name], padding=1.0):
                overlaps.append(f"panel-{index}/{name}")
    for first in range(len(panel_boxes)):
        for second in range(first + 1, len(panel_boxes)):
            if _boxes_overlap(
                panel_boxes[first], panel_boxes[second], padding=1.0
            ):
                overlaps.append(f"panel-{first + 1}/panel-{second + 1}")
    if overlaps:
        raise RuntimeError(f"review layout overlap: {', '.join(overlaps)}")


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

    if status in ("bad_region_removed", "quality_region_retained"):
        annotation_path = os.path.join(
            manual_dir, str(meta.get("annotation_file", "polygonInfo.mat"))
        )
        annotation = load_cut_annotation(annotation_path)
        all_layers = annotation.layers
        default_semantics = (
            POLYGON_INSIDE_DELETED
            if status == "bad_region_removed"
            else POLYGON_INSIDE_RETAINED
        )
        selection_semantics = str(
            meta.get("selection_semantics", default_semantics)
        )
        if selection_semantics not in (
            POLYGON_INSIDE_RETAINED, POLYGON_INSIDE_DELETED
        ):
            raise ValueError(
                f"unknown selection semantics {selection_semantics!r}: {meta_path}"
            )
        regions = _review_regions(all_layers, depth)
        retention_fraction = float(meta["retention"]["retained_fraction"])
    elif status == "no_cut_keep_full":
        annotation = None
        all_layers = ()
        selection_semantics = POLYGON_INSIDE_RETAINED
        regions = _review_regions((), depth)
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
        deleted = deleted_region_node_mask(
            source.positions_xyz, all_layers, selection_semantics
        )
        expected = source.positions_xyz[~deleted]
        run_matches[run_name] = _positions_match(expected, result.positions_xyz)

    if output_path is None:
        output_path = os.path.join(manual_dir, "manual_cut_review.png")
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"review image already exists: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = len(run_names)
    columns = len(regions)
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(max(6.0, 4.8 * columns), max(5.6, 4.5 * rows + 1.1)),
        squeeze=False,
    )
    for row, run_name in enumerate(run_names):
        for column, region in enumerate(regions):
            _render_panel(
                axes[row, column],
                source_graphs[run_name],
                result_graphs[run_name],
                region,
                all_layers,
                selection_semantics,
                height,
                width,
                run_name,
            )

    wrapped_sample_key = _wrap_sample_key(sample_key, width=max(60, 72 * columns))
    title_artist = figure.text(
        0.5, 0.985, "人工裁剪坐标核查",
        ha="center", va="top", fontsize=12, fontweight="bold",
    )
    sample_artist = figure.text(
        0.5, 0.94, wrapped_sample_key,
        ha="center", va="top", fontsize=10, fontweight="bold",
    )
    match_label = "一致" if all(run_matches.values()) else "不一致"
    summary_artist = figure.text(
        0.5,
        0.0,
        f"shape [D,H,W]={depth,height,width} | 保留体积={retention_fraction:.2%}\n"
        f"结果与计算保留坐标：{match_label}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    legend_handles = [
        Patch(facecolor="#e0f5e0", edgecolor="none", label="保留 XY 区域"),
        Patch(facecolor="#ffc7bd", edgecolor="none", label="裁剪 XY 区域（已删除）"),
        Line2D([], [], color="#6b7280", marker=".", linestyle="None", label="源骨架点"),
        Line2D([], [], color="#c62828", marker="x", linestyle="None", label="已删除骨架点"),
        Line2D([], [], color="#0369a1", marker=".", linestyle="-", label="最终骨架"),
        Line2D(
            [], [], color="#f59e0b", marker="o", markeredgecolor="#111827",
            linestyle="-", label="多边形与人工选点",
        ),
    ]
    legend_artist = figure.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=3, fontsize=8,
    )
    _layout_review_figure(
        figure,
        axes,
        title_artist,
        sample_artist,
        summary_artist,
        legend_artist,
    )

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
