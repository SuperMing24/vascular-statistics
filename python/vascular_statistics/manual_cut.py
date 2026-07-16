"""Manual bad-region removal for canonical ``[x, y, z]`` skeleton graphs.

MATLAB polygon coordinates use the ROI convention observed in this experiment:
``0.5`` denotes image/skeleton coordinate ``0``.  Polygons are therefore shifted
by ``-0.5`` before testing skeleton nodes or image voxel centres.  Polygon
boundaries are part of the bad region.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.io import loadmat


COORDINATE_SHIFT = 0.5
GEOMETRY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CutLayer:
    name: str
    polygon_xy: np.ndarray
    polygon_xy_matlab: np.ndarray
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_frame_1based_inclusive": self.start_frame,
            "end_frame_1based_inclusive": self.end_frame,
            "polygon_xy_matlab": self.polygon_xy_matlab.tolist(),
            "polygon_xy_skeleton": self.polygon_xy.tolist(),
        }


@dataclass(frozen=True)
class CutAnnotation:
    source_path: str
    layers: tuple[CutLayer, ...]
    num_frames: Optional[int]
    reference_paths: tuple[str, ...]
    sha256: str

    @property
    def reference_stems(self) -> set[str]:
        stems: set[str] = set()
        for value in self.reference_paths:
            base = os.path.basename(value.replace("\\", "/"))
            stem = os.path.splitext(base)[0].strip().casefold()
            if stem:
                stems.add(stem)
        return stems


@dataclass(frozen=True)
class RetentionResult:
    shape_dhw: tuple[int, int, int]
    total_voxels: int
    deleted_voxels: int
    retained_voxels: int
    retained_fraction: float
    layer_deleted_xy_pixels: dict[str, int]


@dataclass(frozen=True)
class GraphCutResult:
    original_nodes: int
    retained_nodes: int
    deleted_nodes: int
    original_edges: int
    retained_edges: int
    deleted_edges: int


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    return getattr(value, name, default)


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return array.reshape(-1)[0]


def _as_text(value: Any) -> str:
    scalar = _scalar(value)
    if scalar is None:
        return ""
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace").strip()
    return str(scalar).strip()


def _as_int(value: Any, field_name: str) -> Optional[int]:
    scalar = _scalar(value)
    if scalar is None:
        return None
    numeric = float(scalar)
    rounded = int(round(numeric))
    if not np.isclose(numeric, rounded, atol=1e-7):
        raise ValueError(f"{field_name} is not an integer: {numeric}")
    return rounded


def _choose_frame(layer: Any, primary: str, fallback: str) -> Optional[int]:
    first = _as_int(_field(layer, primary), primary)
    second = _as_int(_field(layer, fallback), fallback)
    if first is not None and second is not None and first != second:
        raise ValueError(f"{primary}={first} conflicts with {fallback}={second}")
    return first if first is not None else second


def _polygon_array(value: Any, layer_name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    polygon = np.asarray(value, dtype=float)
    if polygon.size == 0:
        return None
    polygon = np.squeeze(polygon)
    if polygon.ndim != 2:
        raise ValueError(f"{layer_name}.polygonPosition must be N x 2, got {polygon.shape}")
    if polygon.shape[1] != 2 and polygon.shape[0] == 2:
        polygon = polygon.T
    if polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError(f"{layer_name}.polygonPosition must contain >=3 XY points")
    if not np.all(np.isfinite(polygon)):
        raise ValueError(f"{layer_name}.polygonPosition contains non-finite values")
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if polygon.shape[0] < 3:
        raise ValueError(f"{layer_name}.polygonPosition is degenerate")
    x = polygon[:, 0]
    y = polygon[:, 1]
    area2 = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if area2 <= GEOMETRY_TOLERANCE:
        raise ValueError(f"{layer_name}.polygonPosition has zero area")
    return polygon


def load_cut_annotation(path: str) -> CutAnnotation:
    """Load and validate one MATLAB ``saveInfo`` annotation."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    save_info = mat.get("saveInfo")
    if save_info is None:
        raise ValueError(f"missing saveInfo: {path}")

    layers: list[CutLayer] = []
    num_frames_values: set[int] = set()
    references: set[str] = set()
    for layer_name in ("p1", "p2", "p3", "p4"):
        layer = _field(save_info, layer_name)
        if layer is None:
            continue
        raw_polygon = _polygon_array(_field(layer, "polygonPosition"), layer_name)
        if raw_polygon is None:
            continue
        # Empty GUI slots retain unrelated files and numFrames values. Only
        # active polygon layers describe this sample and may drive matching.
        for ref_name in ("fileName", "txtFile"):
            text = _as_text(_field(layer, ref_name))
            if text:
                references.add(text)
        layer_num_frames = _as_int(
            _field(layer, "numFrames"), f"{layer_name}.numFrames"
        )
        if layer_num_frames is not None:
            num_frames_values.add(layer_num_frames)
        start = _choose_frame(layer, "editStartFrame", "startFrameValue")
        end = _choose_frame(layer, "editEndFrame", "endFrameValue")
        if start is None or end is None:
            raise ValueError(f"{layer_name} has a polygon but no complete Z frame range")
        if start < 1 or end < start:
            raise ValueError(f"{layer_name} has invalid frame range {start}..{end}")
        layers.append(CutLayer(
            name=layer_name,
            polygon_xy=raw_polygon - COORDINATE_SHIFT,
            polygon_xy_matlab=raw_polygon,
            start_frame=start,
            end_frame=end,
        ))

    if not layers:
        raise ValueError(f"annotation contains no non-empty polygonPosition: {path}")
    if len(num_frames_values) > 1:
        raise ValueError(f"inconsistent numFrames values: {sorted(num_frames_values)}")
    num_frames = next(iter(num_frames_values), None)
    if num_frames is not None:
        for layer in layers:
            if layer.end_frame > num_frames:
                raise ValueError(
                    f"{layer.name} ends at frame {layer.end_frame}, beyond numFrames={num_frames}"
                )

    return CutAnnotation(
        source_path=os.path.abspath(path),
        layers=tuple(layers),
        num_frames=num_frames,
        reference_paths=tuple(sorted(references)),
        sha256=sha256_file(path),
    )


def points_in_polygon_including_boundary(
    points_xy: np.ndarray,
    polygon_xy: np.ndarray,
    tolerance: float = GEOMETRY_TOLERANCE,
) -> np.ndarray:
    """Return whether each point lies inside or on a simple polygon."""
    points = np.asarray(points_xy, dtype=float)
    polygon = np.asarray(polygon_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must be N x 2")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("polygon_xy must be M x 2 with M >= 3")
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    px = points[:, 0]
    py = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    boundary = np.zeros(points.shape[0], dtype=bool)

    x1 = polygon[:, 0]
    y1 = polygon[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        dx = bx - ax
        dy = by - ay
        scale = max(1.0, abs(dx), abs(dy))
        cross = (px - ax) * dy - (py - ay) * dx
        within = (
            (px >= min(ax, bx) - tolerance)
            & (px <= max(ax, bx) + tolerance)
            & (py >= min(ay, by) - tolerance)
            & (py <= max(ay, by) + tolerance)
        )
        boundary |= within & (np.abs(cross) <= tolerance * scale)

        crosses = (ay > py) != (by > py)
        if abs(dy) > tolerance:
            x_intersection = ax + (py - ay) * dx / dy
            inside ^= crosses & (px < x_intersection)

    return inside | boundary


def polygon_pixel_mask(height: int, width: int, polygon_xy: np.ndarray) -> np.ndarray:
    """Rasterize a polygon against pixel centres ``x=0..W-1, y=0..H-1``."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    polygon = np.asarray(polygon_xy, dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    xmin = max(0, int(np.floor(np.min(polygon[:, 0]))))
    xmax = min(width - 1, int(np.ceil(np.max(polygon[:, 0]))))
    ymin = max(0, int(np.floor(np.min(polygon[:, 1]))))
    ymax = min(height - 1, int(np.ceil(np.max(polygon[:, 1]))))
    if xmin > xmax or ymin > ymax:
        return mask
    yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    local = points_in_polygon_including_boundary(points, polygon).reshape(xx.shape)
    mask[ymin:ymax + 1, xmin:xmax + 1] = local
    return mask


def compute_retention(
    shape_dhw: Sequence[int],
    layers: Sequence[CutLayer],
) -> RetentionResult:
    """Count retained image voxels after the union of all bad-region prisms."""
    if len(shape_dhw) != 3:
        raise ValueError(f"shape must be [D,H,W], got {shape_dhw}")
    depth, height, width = (int(v) for v in shape_dhw)
    if min(depth, height, width) <= 0:
        raise ValueError(f"invalid shape: {shape_dhw}")
    deleted = np.zeros((depth, height, width), dtype=bool)
    layer_counts: dict[str, int] = {}
    for layer in layers:
        if layer.start_frame < 1 or layer.end_frame > depth:
            raise ValueError(
                f"{layer.name} frame range {layer.start_frame}..{layer.end_frame} "
                f"is outside depth 1..{depth}"
            )
        mask = polygon_pixel_mask(height, width, layer.polygon_xy)
        layer_counts[layer.name] = int(np.count_nonzero(mask))
        deleted[layer.start_frame - 1:layer.end_frame] |= mask
    deleted_count = int(np.count_nonzero(deleted))
    total = depth * height * width
    retained = total - deleted_count
    if retained <= 0:
        raise ValueError("manual cut deletes the complete sample volume")
    return RetentionResult(
        shape_dhw=(depth, height, width),
        total_voxels=total,
        deleted_voxels=deleted_count,
        retained_voxels=retained,
        retained_fraction=retained / total,
        layer_deleted_xy_pixels=layer_counts,
    )


def _parse_pos(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        cleaned = value.strip().strip('"').strip("[").strip("]").replace(",", " ")
        parts = cleaned.split()
    elif isinstance(value, (tuple, list, np.ndarray)):
        parts = list(value)
    else:
        parts = []
    if len(parts) < 3:
        raise ValueError(f"Pajek node has invalid pos={value!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def bad_region_node_mask(
    positions_xyz: np.ndarray,
    layers: Sequence[CutLayer],
) -> np.ndarray:
    """Classify canonical XYZ nodes against 1-based inclusive Z layers.

    Frame ``f`` has centre ``z=f-1`` and extent ``[f-1.5, f-0.5]``.  This
    continuous slab also handles non-integer graph nodes produced by refinement.
    """
    positions = np.asarray(positions_xyz, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_xyz must be N x 3")
    deleted = np.zeros(positions.shape[0], dtype=bool)
    z = positions[:, 2]
    for layer in layers:
        zmin = layer.start_frame - 1.5
        zmax = layer.end_frame - 0.5
        in_z = (z >= zmin - GEOMETRY_TOLERANCE) & (z <= zmax + GEOMETRY_TOLERANCE)
        candidates = np.flatnonzero(in_z & ~deleted)
        if candidates.size:
            in_xy = points_in_polygon_including_boundary(
                positions[candidates, :2], layer.polygon_xy
            )
            deleted[candidates[in_xy]] = True
    return deleted


def _node_sort_key(node: Any) -> tuple[int, Any]:
    try:
        return 0, int(node)
    except (TypeError, ValueError):
        return 1, str(node)


def cut_pajek_graph(
    pajek_path: str,
    layers: Sequence[CutLayer],
    output_path: Optional[str] = None,
) -> GraphCutResult:
    """Delete bad-region nodes and incident edges, then relabel survivors 0..N-1."""
    graph = nx.read_pajek(pajek_path)
    ordered_nodes = sorted(graph.nodes(), key=_node_sort_key)
    positions = np.asarray([
        _parse_pos(graph.nodes[node].get("pos", graph.nodes[node].get("Pos")))
        for node in ordered_nodes
    ])
    delete_mask = bad_region_node_mask(positions, layers)
    deleted_nodes = {node for node, delete in zip(ordered_nodes, delete_mask) if delete}
    retained_nodes = [node for node in ordered_nodes if node not in deleted_nodes]
    original_edges = graph.number_of_edges()
    retained_graph = graph.subgraph(retained_nodes).copy()
    mapping = {old: str(new) for new, old in enumerate(retained_nodes)}
    retained_graph = nx.relabel_nodes(retained_graph, mapping, copy=True)

    if output_path is not None:
        nx.write_pajek(retained_graph, output_path)

    return GraphCutResult(
        original_nodes=graph.number_of_nodes(),
        retained_nodes=retained_graph.number_of_nodes(),
        deleted_nodes=len(deleted_nodes),
        original_edges=original_edges,
        retained_edges=retained_graph.number_of_edges(),
        deleted_edges=original_edges - retained_graph.number_of_edges(),
    )


def layer_dicts(layers: Iterable[CutLayer]) -> list[dict[str, Any]]:
    return [layer.to_dict() for layer in layers]
