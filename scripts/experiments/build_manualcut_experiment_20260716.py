#!/usr/bin/env python3
"""Build the 149-sample axis-fixed experiment with manual bad-region cuts.

The two source experiments are copied below separate experimenter namespaces;
all paths inside each source root remain unchanged::

    DST/xiaoqian/<relative path from xiaoqian axisfix root>
    DST/huaien/<relative path from huaien axisfix root>

For 105 annotated samples this script copies the original MATLAB file to
``<sample>/manual_cut/polygonInfo.mat``, deletes skeleton nodes inside the bad
XY polygon prisms (including their boundaries), deletes incident edges,
relabels retained nodes to contiguous IDs, recomputes retained tissue volume,
and reruns the existing statistics.  The other 44 samples are copied unchanged
apart from a ``manual_cut/manual_cut_meta.json`` trace record.

Dry run (read-only)::

    python scripts/experiments/build_manualcut_experiment_20260716.py \
      --src xiaoqian=/share/home/sukm/experiments/vs_xiaoqian_croppedz_20260628_axisfix \
      --src huaien=/share/home/sukm/experiments/vs_huaien_croppedz_20260629_axisfix \
      --annotation-root /share/home/sukm/data/manual_cut_annotations_20260716 \
      --dst-root /share/home/sukm/experiments/vs_croppedz149_axisfix_manualcut_20260716 \
      --dry-run

The non-dry run requires an absent destination root.  Source experiments are
never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Optional, Sequence

import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from vascular_statistics.aggregate_stats import (  # noqa: E402
    generate_cross_sample_summary,
    run_aggregation,
)
from vascular_statistics.anomaly_stats import run_anomaly_stats  # noqa: E402
from vascular_statistics.bridge import pajek_to_cpp_input  # noqa: E402
from vascular_statistics.manual_cut import (  # noqa: E402
    CutAnnotation,
    GraphCutResult,
    RetentionResult,
    compute_retention,
    cut_pajek_graph,
    layer_dicts,
    load_cut_annotation,
    sha256_file,
)
from vascular_statistics.subrange_stats import (  # noqa: E402
    SUBRANGE_D0_10,
    SUBRANGE_D10_PLUS,
    run_subrange_stats,
)


EXPECTED_EXPERIMENTERS = ("xiaoqian", "huaien")
DEFAULT_EXPECTED_SAMPLES = 149
DEFAULT_EXPECTED_ANNOTATIONS = 105

STALE_PATTERNS = (
    "skeleton_edges.txt",
    "skeleton_vertices.txt",
    "generate_vessel*.txt",
    "statistics_summary*.txt",
    "skeleton_anomaly_summary.json",
    "skeleton_anomaly_summary.txt",
    "skeleton_segment_anomalies.tsv",
    "skeleton_edge_anomalies.tsv",
)


@dataclass(frozen=True)
class SampleRecord:
    experimenter: str
    source_root: str
    source_dir: str
    relative_path: str
    sample_key: str
    input_rel_path: str
    metadata: dict[str, Any]
    run_names: tuple[str, ...]

    @property
    def unified_key(self) -> str:
        return f"{self.experimenter}/{self.relative_path}".replace("\\", "/")


@dataclass(frozen=True)
class AnnotationRecord:
    experimenter: str
    relative_path: str
    annotation: CutAnnotation


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def normalize_rel(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def parse_sources(values: Sequence[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--src must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip().casefold()
        path = os.path.abspath(os.path.expanduser(path.strip()))
        if name not in EXPECTED_EXPERIMENTERS:
            raise ValueError(
                f"unknown experimenter {name!r}; expected {EXPECTED_EXPERIMENTERS}"
            )
        if name in sources:
            raise ValueError(f"duplicate --src for {name}")
        if not os.path.isdir(path):
            raise ValueError(f"source root does not exist: {path}")
        sources[name] = path
    missing = set(EXPECTED_EXPERIMENTERS) - set(sources)
    if missing:
        raise ValueError(f"missing --src for: {', '.join(sorted(missing))}")
    return sources


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def scan_samples(sources: dict[str, str]) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for experimenter in EXPECTED_EXPERIMENTERS:
        source_root = sources[experimenter]
        for dirpath, dirnames, filenames in os.walk(source_root):
            dirnames.sort()
            if "sample_metadata.json" not in filenames:
                continue
            run_names = tuple(sorted(
                name for name in dirnames
                if name.startswith("run_")
                and os.path.isfile(os.path.join(dirpath, name, "skeleton.pajek"))
            ))
            if not run_names:
                continue
            metadata = load_json(os.path.join(dirpath, "sample_metadata.json"))
            relative = normalize_rel(os.path.relpath(dirpath, source_root))
            records.append(SampleRecord(
                experimenter=experimenter,
                source_root=source_root,
                source_dir=dirpath,
                relative_path=relative,
                sample_key=normalize_rel(str(metadata.get("sample_key", relative))),
                input_rel_path=normalize_rel(str(metadata.get("input_rel_path", ""))),
                metadata=metadata,
                run_names=run_names,
            ))
    keys = [record.unified_key.casefold() for record in records]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate sample paths: {duplicates[:5]}")
    return sorted(records, key=lambda item: item.unified_key.casefold())


def scan_annotations(annotation_root: str) -> list[AnnotationRecord]:
    root = os.path.abspath(annotation_root)
    if not os.path.isdir(root):
        raise ValueError(f"annotation root does not exist: {root}")
    records: list[AnnotationRecord] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.casefold().endswith(".mat"):
                continue
            path = os.path.join(dirpath, filename)
            relative = normalize_rel(os.path.relpath(path, root))
            parts = relative.split("/")
            experimenter = parts[0].casefold() if parts else ""
            if experimenter not in EXPECTED_EXPERIMENTERS:
                raise ValueError(
                    f"annotation must be under xiaoqian/ or huaien/: {relative}"
                )
            records.append(AnnotationRecord(
                experimenter=experimenter,
                relative_path=relative,
                annotation=load_cut_annotation(path),
            ))
    return records


def path_components(*values: str) -> list[str]:
    components: list[str] = []
    for value in values:
        for part in re.split(r"[\\/]+", value.casefold()):
            part = os.path.splitext(part.strip())[0]
            if part:
                components.append(part)
    return components


def sample_reference_stems(sample: SampleRecord) -> set[str]:
    stems: set[str] = set()
    for value in (sample.relative_path, sample.sample_key, sample.input_rel_path):
        if value:
            base = value.replace("\\", "/").split("/")[-1]
            stem = os.path.splitext(base)[0].strip().casefold()
            if stem:
                stems.add(stem)
    return stems


def suffix_component_match(a: Sequence[str], b: Sequence[str]) -> int:
    count = 0
    for left, right in zip(reversed(a), reversed(b)):
        if left != right:
            break
        count += 1
    return count


def annotation_match_score(sample: SampleRecord, record: AnnotationRecord) -> int:
    sample_parts = path_components(
        sample.relative_path, sample.sample_key, sample.input_rel_path
    )
    annotation_parts = path_components(
        record.relative_path, *record.annotation.reference_paths
    )
    common = set(sample_parts) & set(annotation_parts)
    ignored = {
        "polygoninfo", "saveinfo", "mat", "txt", "data", "cut", "manual_cut",
    }
    informative = [part for part in common if part not in ignored and len(part) >= 3]
    score = 10 * len(informative)
    score += 100 * max(
        suffix_component_match(sample_parts, annotation_parts),
        suffix_component_match(
            path_components(sample.input_rel_path),
            path_components(*record.annotation.reference_paths),
        ),
    )
    score += 1000 * len(sample_reference_stems(sample) & record.annotation.reference_stems)
    return score


def match_annotations(
    samples: Sequence[SampleRecord],
    annotations: Sequence[AnnotationRecord],
) -> dict[str, AnnotationRecord]:
    by_experimenter: dict[str, list[SampleRecord]] = {
        name: [sample for sample in samples if sample.experimenter == name]
        for name in EXPECTED_EXPERIMENTERS
    }
    matched: dict[str, AnnotationRecord] = {}
    for annotation in annotations:
        candidates = []
        annotation_stems = annotation.annotation.reference_stems
        for sample in by_experimenter[annotation.experimenter]:
            stem_overlap = sample_reference_stems(sample) & annotation_stems
            if stem_overlap:
                candidates.append(sample)
        if not candidates:
            raise ValueError(
                f"no sample filename matches annotation {annotation.relative_path}; "
                f"references={sorted(annotation_stems)}"
            )
        ranked = sorted(
            ((annotation_match_score(sample, annotation), sample) for sample in candidates),
            key=lambda item: (-item[0], item[1].unified_key.casefold()),
        )
        best_score, best_sample = ranked[0]
        tied = [sample for score, sample in ranked if score == best_score]
        if len(tied) != 1:
            names = [sample.unified_key for sample in tied]
            raise ValueError(
                f"ambiguous annotation {annotation.relative_path} at score {best_score}: {names}"
            )
        if best_sample.unified_key in matched:
            previous = matched[best_sample.unified_key]
            raise ValueError(
                f"two annotations match {best_sample.unified_key}: "
                f"{previous.relative_path}, {annotation.relative_path}"
            )
        matched[best_sample.unified_key] = annotation
    return matched


def resolve_shape_dhw(sample: SampleRecord, annotation: CutAnnotation) -> tuple[int, int, int]:
    raw_shape = sample.metadata.get("stack_properties", {}).get("shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
        raise ValueError(f"{sample.unified_key}: missing stack_properties.shape=[H,W,D]")
    # Metadata extraction reads cropped MATLAB stacks as [H,W,D]. Geometry
    # and rasterization below use the explicit internal order [D,H,W].
    shape_hwd = tuple(int(value) for value in raw_shape)
    if min(shape_hwd) <= 0:
        raise ValueError(f"{sample.unified_key}: invalid shape {shape_hwd}")
    height, width, depth = shape_hwd
    if annotation.num_frames is not None and depth != annotation.num_frames:
        raise ValueError(
            f"{sample.unified_key}: metadata depth {depth} != annotation "
            f"numFrames {annotation.num_frames}; shape order must be confirmed"
        )
    shape_dhw = (depth, height, width)
    for layer in annotation.layers:
        polygon = layer.polygon_xy
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        if xmin < -1e-7 or ymin < -1e-7 or xmax > width + 1e-7 or ymax > height + 1e-7:
            raise ValueError(
                f"{sample.unified_key}/{layer.name}: polygon bounds "
                f"x={xmin:g}..{xmax:g}, y={ymin:g}..{ymax:g} outside W={width}, H={height}"
            )
    return shape_dhw


def volume_values(
    sample: SampleRecord,
    retention: RetentionResult,
) -> tuple[float, float, list[float]]:
    spatial = sample.metadata.get("spatial", {})
    spacing = spatial.get("voxel_spacing_um")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        raise ValueError(f"{sample.unified_key}: missing spatial.voxel_spacing_um")
    spacing_values = [float(value) for value in spacing]
    voxel_volume = float(np.prod(spacing_values)) / 1e9
    computed_original = retention.total_voxels * voxel_volume
    metadata_original = spatial.get("tissue_volume_mm3")
    if metadata_original is None:
        raise ValueError(f"{sample.unified_key}: missing spatial.tissue_volume_mm3")
    metadata_original = float(metadata_original)
    if not np.isclose(computed_original, metadata_original, rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"{sample.unified_key}: metadata volume {metadata_original} != "
            f"shape*spacing volume {computed_original}"
        )
    retained = retention.retained_voxels * voxel_volume
    return metadata_original, retained, spacing_values


def find_executable(explicit: Optional[str]) -> str:
    if explicit:
        path = os.path.abspath(explicit)
        if os.path.isfile(path):
            return path
        raise ValueError(f"--exe does not exist: {path}")
    for relative in (
        "build/vessel_stats", "build/vessel_stats.exe", "build/Release/vessel_stats.exe",
    ):
        path = os.path.join(_REPO_ROOT, relative)
        if os.path.isfile(path):
            return path
    raise ValueError("cannot find C++ vessel_stats; provide --exe")


def resolve_run_spacing(run_dir: str, run_meta: dict[str, Any]) -> Optional[list[float]]:
    sidecar = os.path.join(run_dir, "skeleton_voxel_spacing_um.txt")
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as handle:
            values = [part.strip() for part in handle.read().strip().split(",")]
        if len(values) == 3:
            return [float(value) for value in values]
    for key in ("stats_spacing_um", "effective_spacing_um", "spacing_um"):
        values = run_meta.get(key)
        if isinstance(values, (list, tuple)) and len(values) == 3:
            return [float(value) for value in values]
    return None


def infer_radius_mode(run_dir: str, run_meta: dict[str, Any]) -> str:
    if run_meta.get("radius_mode"):
        return str(run_meta["radius_mode"])
    sidecar = os.path.join(run_dir, "skeleton_radius_unit.txt")
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as handle:
            if handle.read().strip().casefold() == "um":
                return "physical"
    return "legacy"


def clear_stale_statistics(run_dir: str) -> None:
    for pattern in STALE_PATTERNS:
        for path in glob.glob(os.path.join(run_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)


def rerun_statistics(
    run_dir: str,
    executable: str,
    retained_volume_mm3: float,
    graph_result: GraphCutResult,
    annotation_sha256: str,
) -> None:
    meta_path = os.path.join(run_dir, "run_meta.json")
    run_meta = load_json(meta_path) if os.path.isfile(meta_path) else {}
    spacing = resolve_run_spacing(run_dir, run_meta)
    radius_mode = infer_radius_mode(run_dir, run_meta)
    unit_mode = run_meta.get("stats_unit_mode")
    anisotropic = (
        unit_mode == "anisotropic"
        or radius_mode == "physical"
        or (unit_mode is None and spacing is not None)
    )
    if anisotropic and spacing is None:
        raise ValueError(f"{run_dir}: anisotropic statistics has no spacing")

    stem = str(run_meta.get("output_stem", "skeleton"))
    clear_stale_statistics(run_dir)
    pajek_to_cpp_input(
        os.path.join(run_dir, "skeleton.pajek"),
        os.path.join(run_dir, f"{stem}_edges.txt"),
        os.path.join(run_dir, f"{stem}_vertices.txt"),
    )
    command = [executable, f"{stem}_edges", f"{stem}_vertices", str(retained_volume_mm3)]
    if anisotropic:
        command.extend(str(value) for value in spacing)
        if radius_mode == "physical":
            command.append("1")
    result = subprocess.run(command, cwd=run_dir, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise RuntimeError(f"vessel_stats failed in {run_dir} (rc={result.returncode}): {detail}")

    previous_volume = run_meta.get("volume_mm3")
    run_meta.setdefault("output_stem", stem)
    run_meta["volume_mm3"] = retained_volume_mm3
    run_meta.setdefault("radius_mode", radius_mode)
    run_meta.setdefault("stats_unit_mode", "anisotropic" if anisotropic else "legacy")
    if spacing is not None:
        run_meta.setdefault("stats_spacing_um", spacing)
    run_meta.setdefault("manual_cut_history", []).append({
        "operation": "delete bad-region nodes and all incident edges; no re-skeletonization",
        "annotation_sha256": annotation_sha256,
        "previous_volume_mm3": previous_volume,
        "retained_volume_mm3": retained_volume_mm3,
        "graph": asdict(graph_result),
        "applied_at": now_iso(),
    })
    write_json(meta_path, run_meta)


def cut_run(
    run_dir: str,
    annotation: CutAnnotation,
    retained_volume_mm3: float,
    executable: Optional[str],
    dry_run: bool,
) -> GraphCutResult:
    pajek_path = os.path.join(run_dir, "skeleton.pajek")
    if dry_run:
        return cut_pajek_graph(pajek_path, annotation.layers)
    if executable is None:
        raise ValueError("internal error: executable required for non-dry run")
    temporary = pajek_path + ".manualcut.tmp"
    try:
        graph_result = cut_pajek_graph(pajek_path, annotation.layers, temporary)
        os.replace(temporary, pajek_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    rerun_statistics(
        run_dir, executable, retained_volume_mm3, graph_result, annotation.sha256
    )
    return graph_result


def update_sample_metadata(
    destination_sample_dir: str,
    original_volume: float,
    retained_volume: float,
    retention: RetentionResult,
    annotation: CutAnnotation,
) -> None:
    path = os.path.join(destination_sample_dir, "sample_metadata.json")
    metadata = load_json(path)
    metadata.setdefault("spatial", {})["tissue_volume_mm3"] = retained_volume
    metadata["manual_cut"] = {
        "status": "bad_region_removed",
        "coordinate_convention": "polygonPosition - 0.5 -> canonical skeleton/image coordinates",
        "polygon_boundary": "deleted",
        "z_convention": "MATLAB frames are 1-based inclusive; skeleton z frame centres are 0-based",
        "original_tissue_volume_mm3": original_volume,
        "retained_tissue_volume_mm3": retained_volume,
        "retention": asdict(retention),
        "annotation_sha256": annotation.sha256,
        "applied_at": now_iso(),
    }
    write_json(path, metadata)


def annotation_meta(
    sample: SampleRecord,
    record: Optional[AnnotationRecord],
    retention: Optional[RetentionResult],
    original_volume: float,
    retained_volume: float,
    run_results: dict[str, GraphCutResult],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "1.0",
        "experimenter": sample.experimenter,
        "source_experiment_root": sample.source_root,
        "source_sample_relative_path": sample.relative_path,
        "destination_sample_key": sample.unified_key,
        "status": "bad_region_removed" if record else "no_cut_keep_full",
        "skeletonization": "reused from axisfix source; not rerun",
        "original_tissue_volume_mm3": original_volume,
        "retained_tissue_volume_mm3": retained_volume,
        "generated_at": now_iso(),
    }
    if record and retention:
        base.update({
            "source_annotation_relative_path": record.relative_path,
            "annotation_file": "polygonInfo.mat",
            "annotation_sha256": record.annotation.sha256,
            "coordinate_convention": {
                "xy": "stored MATLAB 0.5 means image/skeleton coordinate 0; subtract 0.5",
                "polygon_boundary": "bad region; deleted",
                "z": "1-based inclusive MATLAB frame ranges; frame f centre is skeleton z=f-1",
            },
            "layers": layer_dicts(record.annotation.layers),
            "retention": asdict(retention),
            "runs": {name: asdict(result) for name, result in sorted(run_results.items())},
        })
    return base


def copy_sources(sources: dict[str, str], destination_root: str) -> None:
    if os.path.exists(destination_root):
        raise ValueError(f"destination root already exists: {destination_root}")
    os.makedirs(destination_root)
    try:
        for experimenter in EXPECTED_EXPERIMENTERS:
            destination = os.path.join(destination_root, experimenter)
            print(f"[copy] {sources[experimenter]} -> {destination}")
            shutil.copytree(sources[experimenter], destination)
    except Exception:
        print(
            f"[ERROR] partial destination retained for diagnosis: {destination_root}",
            file=sys.stderr,
        )
        raise


def update_experimenter_catalog(
    experimenter_root: str,
    samples: Sequence[dict[str, Any]],
) -> None:
    path = os.path.join(experimenter_root, "sample_catalog.json")
    if not os.path.isfile(path):
        return
    catalog = load_json(path)
    catalog["dataset_root"] = experimenter_root
    catalog["manual_cut_updated_at"] = now_iso()
    index = catalog.get("samples", {})
    if isinstance(index, dict):
        for row in samples:
            key = row["source_sample_key"]
            if key in index and isinstance(index[key], dict):
                index[key]["tissue_volume_mm3"] = row["retained_volume_mm3"]
                index[key]["manual_cut_status"] = row["status"]
    write_json(path, catalog)


def write_manifest(destination_root: str, rows: Sequence[dict[str, Any]]) -> None:
    payload = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "experiment_root": destination_root,
        "total_samples": len(rows),
        "annotated_samples": sum(row["status"] == "bad_region_removed" for row in rows),
        "full_retained_samples": sum(row["status"] == "no_cut_keep_full" for row in rows),
        "samples": list(rows),
    }
    write_json(os.path.join(destination_root, "manual_cut_manifest.json"), payload)
    columns = (
        "experimenter", "unified_sample_key", "source_sample_key", "status",
        "annotation_relative_path", "annotation_sha256", "original_volume_mm3",
        "retained_volume_mm3", "retained_fraction", "run_count", "deleted_nodes",
        "deleted_edges",
    )
    path = os.path.join(destination_root, "manual_cut_manifest.tsv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def verify_no_cut_samples_unchanged(
    destination_root: str,
    samples: Sequence[SampleRecord],
    cut_keys: set[str],
) -> int:
    def hashes(sample_dir: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for dirpath, _dirnames, filenames in os.walk(sample_dir):
            for filename in filenames:
                keep = (
                    filename == "sample_metadata.json"
                    or filename == "skeleton.pajek"
                    or filename in ("skeleton_edges.txt", "skeleton_vertices.txt")
                    or filename.startswith("generate_vessel")
                    or filename.startswith("statistics_summary")
                    or filename.startswith("skeleton_anomaly")
                    or filename.startswith("skeleton_segment_anomalies")
                    or filename.startswith("skeleton_edge_anomalies")
                )
                if keep:
                    path = os.path.join(dirpath, filename)
                    relative = normalize_rel(os.path.relpath(path, sample_dir))
                    result[relative] = sha256_file(path)
        return result

    verified = 0
    for sample in samples:
        if sample.unified_key in cut_keys:
            continue
        destination = os.path.join(
            destination_root, sample.experimenter, sample.relative_path
        )
        if hashes(sample.source_dir) != hashes(destination):
            raise RuntimeError(f"no-cut scientific artifacts changed: {sample.unified_key}")
        verified += 1
    return verified


def check_stats_result(label: str, result: dict[str, int]) -> None:
    if result.get("failed", 0):
        raise RuntimeError(f"{label} failed for {result['failed']} item(s): {result}")


def refresh_derived_statistics(
    destination_root: str,
    samples: Sequence[SampleRecord],
    cut_keys: Sequence[str],
) -> None:
    if cut_keys:
        check_stats_result(
            "diameter-stats 0-10",
            run_subrange_stats(destination_root, list(cut_keys), SUBRANGE_D0_10),
        )
        check_stats_result(
            "diameter-stats 10+",
            run_subrange_stats(destination_root, list(cut_keys), SUBRANGE_D10_PLUS),
        )
        check_stats_result(
            "anomaly-stats",
            run_anomaly_stats(destination_root, list(cut_keys)),
        )
        for suffix in ("", "_d0-10um", "_d10+um"):
            check_stats_result(
                f"aggregate {suffix or 'full'}",
                run_aggregation(destination_root, list(cut_keys), suffix=suffix),
            )

    all_keys = [sample.unified_key for sample in samples]
    for suffix in ("", "_d0-10um", "_d10+um"):
        output = generate_cross_sample_summary(
            destination_root, suffix=suffix, sample_keys=all_keys
        )
        if output is None:
            raise RuntimeError(f"unified cross-sample summary produced no output: {suffix}")

    for experimenter in EXPECTED_EXPERIMENTERS:
        experimenter_root = os.path.join(destination_root, experimenter)
        keys = [
            sample.relative_path for sample in samples if sample.experimenter == experimenter
        ]
        for suffix in ("", "_d0-10um", "_d10+um"):
            output = generate_cross_sample_summary(
                experimenter_root, suffix=suffix, sample_keys=keys
            )
            if output is None:
                raise RuntimeError(
                    f"{experimenter} cross-sample summary produced no output: {suffix}"
                )


def process(
    sources: dict[str, str],
    annotation_root: str,
    destination_root: str,
    executable: Optional[str],
    dry_run: bool,
    expected_samples: int,
    expected_annotations: int,
    no_aggregate: bool,
) -> None:
    samples = scan_samples(sources)
    annotations = scan_annotations(annotation_root)
    if len(samples) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, found {len(samples)}")
    if len(annotations) != expected_annotations:
        raise ValueError(
            f"expected {expected_annotations} annotations, found {len(annotations)}"
        )
    matched = match_annotations(samples, annotations)
    if len(matched) != expected_annotations:
        raise ValueError(
            f"expected {expected_annotations} unique annotation matches, got {len(matched)}"
        )

    print(
        f"[validated] samples={len(samples)}, annotated={len(matched)}, "
        f"keep-full={len(samples) - len(matched)}"
    )
    if not dry_run:
        copy_sources(sources, destination_root)

    manifest_rows: list[dict[str, Any]] = []
    cut_keys: list[str] = []
    for number, sample in enumerate(samples, start=1):
        record = matched.get(sample.unified_key)
        destination_sample_dir = os.path.join(
            destination_root, sample.experimenter, sample.relative_path
        )
        working_sample_dir = sample.source_dir if dry_run else destination_sample_dir
        if record is None:
            original_volume = float(
                sample.metadata.get("spatial", {}).get("tissue_volume_mm3")
            )
            retained_volume = original_volume
            retention = None
            run_results: dict[str, GraphCutResult] = {}
            status = "no_cut_keep_full"
        else:
            shape = resolve_shape_dhw(sample, record.annotation)
            retention = compute_retention(shape, record.annotation.layers)
            original_volume, retained_volume, _spacing = volume_values(sample, retention)
            run_results = {}
            for run_name in sample.run_names:
                run_dir = os.path.join(working_sample_dir, run_name)
                run_results[run_name] = cut_run(
                    run_dir, record.annotation, retained_volume, executable, dry_run
                )
            status = "bad_region_removed"
            cut_keys.append(sample.unified_key)

        deleted_nodes = sum(result.deleted_nodes for result in run_results.values())
        deleted_edges = sum(result.deleted_edges for result in run_results.values())
        fraction = retention.retained_fraction if retention else 1.0
        annotation_label = record.relative_path if record else "-"
        print(
            f"[{number:03d}/{len(samples)}] {status:18s} {sample.unified_key} "
            f"annotation={annotation_label} retained={fraction:.6f} "
            f"deleted_nodes={deleted_nodes} deleted_edges={deleted_edges}"
        )

        if not dry_run:
            manual_dir = os.path.join(destination_sample_dir, "manual_cut")
            if os.path.exists(manual_dir):
                raise ValueError(f"manual_cut directory already exists: {manual_dir}")
            os.makedirs(manual_dir)
            if record and retention:
                shutil.copy2(
                    record.annotation.source_path,
                    os.path.join(manual_dir, "polygonInfo.mat"),
                )
                update_sample_metadata(
                    destination_sample_dir,
                    original_volume,
                    retained_volume,
                    retention,
                    record.annotation,
                )
            write_json(
                os.path.join(manual_dir, "manual_cut_meta.json"),
                annotation_meta(
                    sample, record, retention, original_volume, retained_volume, run_results
                ),
            )

        manifest_rows.append({
            "experimenter": sample.experimenter,
            "unified_sample_key": sample.unified_key,
            "source_sample_key": sample.sample_key,
            "source_relative_path": sample.relative_path,
            "status": status,
            "annotation_relative_path": record.relative_path if record else "",
            "annotation_sha256": record.annotation.sha256 if record else "",
            "original_volume_mm3": original_volume,
            "retained_volume_mm3": retained_volume,
            "retained_fraction": fraction,
            "run_count": len(sample.run_names),
            "deleted_nodes": deleted_nodes,
            "deleted_edges": deleted_edges,
        })

    cut_rows = [row for row in manifest_rows if row["status"] == "bad_region_removed"]
    if any(row["retained_volume_mm3"] >= row["original_volume_mm3"] for row in cut_rows):
        raise ValueError("at least one annotated sample did not reduce retained volume")
    if dry_run:
        print(
            f"[dry-run complete] {len(cut_rows)} annotated samples validated; "
            f"{len(samples) - len(cut_rows)} samples remain full; no files written"
        )
        return

    for experimenter in EXPECTED_EXPERIMENTERS:
        update_experimenter_catalog(
            os.path.join(destination_root, experimenter),
            [row for row in manifest_rows if row["experimenter"] == experimenter],
        )
    write_manifest(destination_root, manifest_rows)
    write_json(os.path.join(destination_root, "manual_cut_experiment.json"), {
        "schema_version": "1.0",
        "created_at": now_iso(),
        "source_roots": sources,
        "annotation_input_root": os.path.abspath(annotation_root),
        "experimenter_namespaces": list(EXPECTED_EXPERIMENTERS),
        "sample_count": len(samples),
        "annotated_sample_count": len(cut_rows),
        "full_retained_sample_count": len(samples) - len(cut_rows),
        "old_results_preserved": True,
    })
    if not no_aggregate:
        refresh_derived_statistics(destination_root, samples, cut_keys)
    verified = verify_no_cut_samples_unchanged(
        destination_root, samples, set(cut_keys)
    )
    print(f"[verified] no-cut scientific artifacts unchanged: {verified}")
    print(f"[complete] experiment built at {destination_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src", action="append", required=True, metavar="NAME=PATH",
        help="axisfix source root; provide once for xiaoqian and once for huaien",
    )
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--exe", default=None, help="C++ vessel_stats executable")
    parser.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_SAMPLES)
    parser.add_argument(
        "--expected-annotations", type=int, default=DEFAULT_EXPECTED_ANNOTATIONS
    )
    parser.add_argument("--dry-run", action="store_true", help="validate only; write nothing")
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="skip subrange/anomaly/aggregate refresh (diagnostic use only)",
    )
    args = parser.parse_args()
    try:
        sources = parse_sources(args.src)
        destination_root = os.path.abspath(os.path.expanduser(args.dst_root))
        executable = None if args.dry_run else find_executable(args.exe)
        process(
            sources=sources,
            annotation_root=args.annotation_root,
            destination_root=destination_root,
            executable=executable,
            dry_run=args.dry_run,
            expected_samples=args.expected_samples,
            expected_annotations=args.expected_annotations,
            no_aggregate=args.no_aggregate,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
