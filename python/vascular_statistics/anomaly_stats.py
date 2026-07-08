"""骨架异常诊断层。

从每个 run 的全量血管段明细派生 QC 结果，不改变主统计口径：
  - skeleton_anomaly_summary.json / .txt
  - skeleton_segment_anomalies.tsv
  - skeleton_edge_anomalies.tsv

诊断覆盖 full / d0-10um / d10+um 三档，标准复用现有子范围统计的
P99、MAD(k=3.5)、绝对超长段阈值，以及 validation_ranges.json 的生理范围。
另有 edge-level 拓扑 QC，专门检测两个骨架点之间的单边长跳连接。
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vascular_statistics.subrange_stats import (
    LEGACY_MIN_NODES,
    MAD_K,
    SUBRANGE_D0_10,
    SUBRANGE_D10_PLUS,
    _read_caliber,
    parse_segment_data,
)


POPULATIONS = ("full", "d0-10um", "d10+um")
STATUS_RANK = {"OK": 0, "WARN": 1, "FAIL": 2}
EDGE_WARN_LENGTH_UM = 30.0
EDGE_FAIL_LENGTH_UM = 100.0
EDGE_FAIL_MEDIAN_RATIO = 50.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_validation_ranges(species: str = "mouse_brain") -> Dict[str, Any]:
    path = _repo_root() / "configs" / "delta" / "validation_ranges.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(species, {})
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_median(values: List[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _p99_threshold(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx99 = int(99.0 / 100.0 * len(ordered))
    if idx99 >= len(ordered):
        idx99 = len(ordered) - 1
    return ordered[idx99]



def _percentile_threshold(values: List[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(percentile / 100.0 * len(ordered))
    if idx >= len(ordered):
        idx = len(ordered) - 1
    return ordered[idx]

def _mad_flags(values: List[float], k: float = MAD_K) -> List[bool]:
    if len(values) < 2:
        return [False] * len(values)
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    if mad < 1e-10:
        return [False] * len(values)
    return [abs(0.6745 * (x - med) / mad) > k for x in values]


def _node_count(seq: str) -> int:
    return len(seq.split()) if seq else 0


def _population_indices(
    radii: List[float],
    node_sequences: List[str],
    population: str,
    r_scale: float,
) -> List[int]:
    if population == "full":
        return list(range(len(radii)))

    if population == "d0-10um":
        lo_um, hi_um = SUBRANGE_D0_10.lo_um, SUBRANGE_D0_10.hi_um
    elif population == "d10+um":
        lo_um, hi_um = SUBRANGE_D10_PLUS.lo_um, SUBRANGE_D10_PLUS.hi_um
    else:
        raise ValueError(f"未知 population: {population}")

    lo_r = lo_um / (2.0 * r_scale)
    hi_r = hi_um / (2.0 * r_scale)
    return [
        i for i, r in enumerate(radii)
        if lo_r <= r < hi_r and _node_count(node_sequences[i]) >= LEGACY_MIN_NODES
    ]


def _range_violations(metrics: Dict[str, float], ranges: Dict[str, Any]) -> List[str]:
    mapping = {
        "avg_diameter_um": "diameter_um",
        "avg_length_um": "segment_length_um",
        "avg_tortuosity_au": "tortuosity",
        "segment_density_per_mm3": "segment_density_per_mm3",
    }
    violations: List[str] = []
    for metric_key, range_key in mapping.items():
        value = metrics.get(metric_key)
        spec = ranges.get(range_key, {})
        if value is None or value != value or not spec:
            continue
        min_v = spec.get("min")
        max_v = spec.get("max")
        if min_v is not None and value < float(min_v):
            violations.append(f"{metric_key}<{min_v}")
        if max_v is not None and value > float(max_v):
            violations.append(f"{metric_key}>{max_v}")
    min_count = ranges.get("segment_count_min")
    n = metrics.get("n_segments")
    if min_count is not None and n is not None and n < int(min_count):
        violations.append(f"n_segments<{min_count}")
    return violations


def _qc_status(
    *,
    abs_ratio: float,
    mad_ratio: float,
    invalid_tortuosity: int,
    range_violations: List[str],
) -> str:
    if abs_ratio > 1.0:
        return "FAIL"
    if range_violations:
        return "WARN"
    if abs_ratio > 0.0 or mad_ratio > 5.0 or invalid_tortuosity > 0:
        return "WARN"
    return "OK"


def _read_vertices(run_dir: str) -> Dict[int, Tuple[float, float, float]]:
    path = os.path.join(run_dir, "skeleton_vertices.txt")
    if not os.path.exists(path):
        return {}
    vertices: Dict[int, Tuple[float, float, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                nid = int(float(parts[0]))
                vertices[nid] = (float(parts[2]), float(parts[3]), float(parts[4]))
    except (OSError, ValueError):
        return {}
    return vertices



def _read_edges(run_dir: str) -> List[Tuple[int, int]]:
    path = os.path.join(run_dir, "skeleton_edges.txt")
    if not os.path.exists(path):
        return []
    edges: List[Tuple[int, int]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(float(parts[0]))
                v = int(float(parts[1]))
                if u != v:
                    edges.append((u, v))
    except (OSError, ValueError):
        return []
    return edges

def _edge_length_um(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    spacing: Optional[List[float]],
    length_um_factor: float,
) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    if spacing and len(spacing) == 3:
        return math.sqrt((dx * spacing[0]) ** 2 + (dy * spacing[1]) ** 2 + (dz * spacing[2]) ** 2)
    return math.sqrt(dx * dx + dy * dy + dz * dz) * length_um_factor


def _read_spacing_from_meta(run_dir: str) -> Optional[List[float]]:
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("stats_spacing_um", "effective_spacing_um", "spacing_um"):
            sp = meta.get(key)
            if sp and len(sp) == 3:
                return [float(sp[0]), float(sp[1]), float(sp[2])]
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return None



def _compute_edge_qc(
    run_dir: str,
    *,
    length_um_factor: float,
) -> Dict[str, Any]:
    vertices = _read_vertices(run_dir)
    raw_edges = _read_edges(run_dir)
    spacing = _read_spacing_from_meta(run_dir)
    degrees: Dict[int, int] = {}
    rows: List[Dict[str, Any]] = []

    for u, v in raw_edges:
        if u not in vertices or v not in vertices:
            continue
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1

    all_lengths: List[float] = []
    for u, v in raw_edges:
        if u not in vertices or v not in vertices:
            continue
        p1 = vertices[u]
        p2 = vertices[v]
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        if spacing and len(spacing) == 3:
            dx_um = dx * spacing[0]
            dy_um = dy * spacing[1]
            dz_um = dz * spacing[2]
            length_um = math.sqrt(dx_um * dx_um + dy_um * dy_um + dz_um * dz_um)
        else:
            dx_um = dx * length_um_factor
            dy_um = dy * length_um_factor
            dz_um = dz * length_um_factor
            length_um = math.sqrt(dx_um * dx_um + dy_um * dy_um + dz_um * dz_um)
        all_lengths.append(length_um)
        rows.append({
            "edge_u": u,
            "edge_v": v,
            "edge_length_um": length_um,
            "dx_um": dx_um,
            "dy_um": dy_um,
            "dz_um": dz_um,
            "degree_u": degrees.get(u, 0),
            "degree_v": degrees.get(v, 0),
        })

    if not rows:
        return {
            "status": "NA",
            "n_edges": 0,
            "rows": [],
        }

    median_len = _safe_median(all_lengths)
    p99_len = _percentile_threshold(all_lengths, 99.0)
    p999_len = _percentile_threshold(all_lengths, 99.9)

    fail_edges = 0
    warn_edges = 0
    for row in rows:
        length_um = float(row["edge_length_um"])
        median_ratio = length_um / median_len if median_len and median_len == median_len else float("nan")
        reasons: List[str] = []
        severity = "OK"
        if length_um > EDGE_FAIL_LENGTH_UM:
            reasons.append("edge_length_gt_100um")
            severity = "FAIL"
        if median_ratio == median_ratio and median_ratio > EDGE_FAIL_MEDIAN_RATIO:
            reasons.append("edge_length_gt_50x_median")
            severity = "FAIL"
        if severity != "FAIL" and length_um > EDGE_WARN_LENGTH_UM:
            reasons.append("edge_length_gt_30um")
            severity = "WARN"
        if severity == "OK" and length_um > p999_len:
            reasons.append("edge_length_gt_p99.9")
            severity = "INFO"
        row["median_ratio"] = median_ratio
        row["severity"] = severity
        row["reasons"] = ",".join(reasons) if reasons else "-"
        if severity == "FAIL":
            fail_edges += 1
        elif severity == "WARN":
            warn_edges += 1

    status = "FAIL" if fail_edges else ("WARN" if warn_edges else "OK")
    flagged_rows = [r for r in rows if r["severity"] != "OK"]
    flagged_rows.sort(
        key=lambda r: (STATUS_RANK.get(r["severity"], 0), float(r["edge_length_um"])),
        reverse=True,
    )
    rows.sort(key=lambda r: float(r["edge_length_um"]), reverse=True)

    return {
        "status": status,
        "n_edges": len(rows),
        "max_edge_length_um": rows[0]["edge_length_um"],
        "median_edge_length_um": median_len,
        "p99_edge_length_um": p99_len,
        "p999_edge_length_um": p999_len,
        "warn_length_threshold_um": EDGE_WARN_LENGTH_UM,
        "fail_length_threshold_um": EDGE_FAIL_LENGTH_UM,
        "fail_median_ratio_threshold": EDGE_FAIL_MEDIAN_RATIO,
        "fail_edges": fail_edges,
        "warn_edges": warn_edges,
        "flagged_edges": len(flagged_rows),
        "rows": flagged_rows,
    }

def compute_run_anomaly_stats(
    run_dir: str,
    volume: float,
    *,
    species: str = "mouse_brain",
) -> Optional[Dict[str, Any]]:
    seg_data = parse_segment_data(run_dir)
    if seg_data is None:
        return None

    mode, r_scale, length_um_factor, _radius_threshold, abs_thresh_raw = _read_caliber(run_dir)
    ranges_by_pop = load_validation_ranges(species)
    radii = seg_data["radii"]
    lengths_raw = seg_data["path_lengths"]
    torts = seg_data["tortuosities"]
    node_sequences = seg_data["node_sequences"]

    lengths_um = [v * length_um_factor for v in lengths_raw]
    diameters_um = [r * 2.0 * r_scale for r in radii]
    abs_thresh_um = abs_thresh_raw * length_um_factor

    populations: Dict[str, Any] = {}
    segment_reasons: Dict[int, Dict[str, List[str]]] = {}

    for pop in POPULATIONS:
        idxs = _population_indices(radii, node_sequences, pop, r_scale)
        pop_lengths = [lengths_um[i] for i in idxs]
        pop_diams = [diameters_um[i] for i in idxs]
        pop_torts_raw = [torts[i] for i in idxs]
        pop_torts = [t for t in pop_torts_raw if not (math.isnan(t) or math.isinf(t))]

        p99 = _p99_threshold(pop_lengths)
        mad_flags = _mad_flags(pop_lengths, MAD_K)
        abs_flags = [v > abs_thresh_um for v in pop_lengths]
        p99_flags = [v > p99 for v in pop_lengths] if pop_lengths else []

        n = len(idxs)
        mad_outliers = sum(1 for v in mad_flags if v)
        abs_outliers = sum(1 for v in abs_flags if v)
        p99_outliers = sum(1 for v in p99_flags if v)
        invalid_torts = len(pop_torts_raw) - len(pop_torts)

        metrics = {
            "n_segments": n,
            "avg_diameter_um": _safe_mean(pop_diams),
            "median_diameter_um": _safe_median(pop_diams),
            "avg_length_um": _safe_mean(pop_lengths),
            "median_length_um": _safe_median(pop_lengths),
            "avg_tortuosity_au": _safe_mean(pop_torts),
            "median_tortuosity_au": _safe_median(pop_torts),
            "segment_density_per_mm3": n / volume if volume > 0 else float("nan"),
        }
        range_violations = _range_violations(metrics, ranges_by_pop.get(pop, {}))
        mad_ratio = 100.0 * mad_outliers / n if n else 0.0
        abs_ratio = 100.0 * abs_outliers / n if n else 0.0
        p99_ratio = 100.0 * p99_outliers / n if n else 0.0
        status = _qc_status(
            abs_ratio=abs_ratio,
            mad_ratio=mad_ratio,
            invalid_tortuosity=invalid_torts,
            range_violations=range_violations,
        )

        populations[pop] = {
            "status": status,
            "metrics": metrics,
            "p99_length_um": p99,
            "p99_outliers": p99_outliers,
            "p99_ratio": p99_ratio,
            "mad_outliers": mad_outliers,
            "mad_ratio": mad_ratio,
            "abs_length_threshold_um": abs_thresh_um,
            "abs_outliers": abs_outliers,
            "abs_ratio": abs_ratio,
            "invalid_tortuosity": invalid_torts,
            "range_violations": range_violations,
        }

        for local_i, seg_i in enumerate(idxs):
            reasons: List[str] = []
            if p99_flags and p99_flags[local_i]:
                reasons.append("p99_length")
            if mad_flags and mad_flags[local_i]:
                reasons.append("mad_length")
            if abs_flags and abs_flags[local_i]:
                reasons.append("absolute_long")
            tort = torts[seg_i]
            if math.isnan(tort) or math.isinf(tort):
                reasons.append("invalid_tortuosity")
            if reasons:
                entry = segment_reasons.setdefault(seg_i, {})
                entry[pop] = reasons

    edge_qc = _compute_edge_qc(run_dir, length_um_factor=length_um_factor)
    status_candidates = [pop_data["status"] for pop_data in populations.values()]
    if edge_qc.get("status") in STATUS_RANK:
        status_candidates.append(edge_qc["status"])
    worst = max(
        status_candidates,
        key=lambda s: STATUS_RANK.get(s, 0),
    )

    return {
        "version": "1.1",
        "run_dir": os.path.abspath(run_dir),
        "stats_unit_mode": mode,
        "r_scale": r_scale,
        "length_um_factor": length_um_factor,
        "volume_mm3": volume,
        "status": worst,
        "populations": populations,
        "edge_qc": {k: v for k, v in edge_qc.items() if k != "rows"},
        "segments": {
            "radii": radii,
            "diameters_um": diameters_um,
            "lengths_um": lengths_um,
            "tortuosities": torts,
            "node_sequences": node_sequences,
            "reasons": segment_reasons,
        },
        "edge_qc_rows": edge_qc.get("rows", []),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value

def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v) or math.isinf(v):
        return "N/A"
    return f"{v:.{digits}f}"


def write_run_anomaly_stats(
    run_dir: str,
    volume: float,
    *,
    species: str = "mouse_brain",
) -> Optional[str]:
    stats = compute_run_anomaly_stats(run_dir, volume, species=species)
    if stats is None:
        return None

    json_path = os.path.join(run_dir, "skeleton_anomaly_summary.json")
    txt_path = os.path.join(run_dir, "skeleton_anomaly_summary.txt")
    seg_tsv_path = os.path.join(run_dir, "skeleton_segment_anomalies.tsv")
    edge_tsv_path = os.path.join(run_dir, "skeleton_edge_anomalies.tsv")
    long_edge_tsv_path = os.path.join(run_dir, "skeleton_long_edges.tsv")

    public_stats = dict(stats)
    segments = public_stats.pop("segments")
    long_edge_rows = public_stats.pop("edge_qc_rows", [])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(public_stats), f, indent=2, ensure_ascii=False, allow_nan=False)

    lines = [
        "Vascular_Statistics -- 骨架异常诊断",
        "",
        f"QC状态: {stats['status']}",
        f"单位口径: {stats['stats_unit_mode']}",
        f"组织体积: {stats['volume_mm3']} mm^3",
        "",
        "population | QC | 段数 | 超长段% | MAD离群% | P99离群% | inf/NaN弯曲度 | 范围违规",
        "-" * 102,
    ]
    for pop in POPULATIONS:
        pdata = stats["populations"][pop]
        metrics = pdata["metrics"]
        violations = ",".join(pdata["range_violations"]) if pdata["range_violations"] else "-"
        lines.append(
            f"{pop} | {pdata['status']} | {metrics['n_segments']} | "
            f"{pdata['abs_ratio']:.4f} | {pdata['mad_ratio']:.4f} | "
            f"{pdata['p99_ratio']:.4f} | {pdata['invalid_tortuosity']} | {violations}"
        )
    eqc = stats.get("edge_qc", {})
    lines.extend([
        "",
        "edge topology QC | QC | 边数 | 最长单边(um) | 中位单边(um) | P99.9单边(um) | FAIL边 | WARN边",
        "-" * 102,
        (
            f"all_edges | {eqc.get('status', 'NA')} | {eqc.get('n_edges', 0)} | "
            f"{_format_float(eqc.get('max_edge_length_um'), 4)} | "
            f"{_format_float(eqc.get('median_edge_length_um'), 4)} | "
            f"{_format_float(eqc.get('p999_edge_length_um'), 4)} | "
            f"{eqc.get('fail_edges', 0)} | {eqc.get('warn_edges', 0)}"
        ),
        "",
        "判据: 段级QC用于生理统计；单边拓扑QC独立扫描全部 skeleton_edges。",
        "段级: 绝对超长段比例 >1% => FAIL；生理范围违规或存在离群/无效弯曲度 => WARN。",
        "单边: >100um 或 >50x median => FAIL；>30um => WARN。",
        "明细: skeleton_segment_anomalies.tsv / skeleton_edge_anomalies.tsv / skeleton_long_edges.tsv",
        "",
    ])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    reasons = segments["reasons"]
    segment_rows: List[Dict[str, Any]] = []
    for seg_i, pop_reasons in reasons.items():
        populations = sorted(pop_reasons.keys(), key=lambda p: POPULATIONS.index(p))
        reason_text = ";".join(
            f"{pop}:{','.join(pop_reasons[pop])}" for pop in populations
        )
        segment_rows.append({
            "segment_id": seg_i,
            "populations": ",".join(populations),
            "reasons": reason_text,
            "node_count": _node_count(segments["node_sequences"][seg_i]),
            "diameter_um": segments["diameters_um"][seg_i],
            "length_um": segments["lengths_um"][seg_i],
            "tortuosity": segments["tortuosities"][seg_i],
            "node_sequence": segments["node_sequences"][seg_i],
        })
    segment_rows.sort(key=lambda r: float(r["length_um"]), reverse=True)

    with open(seg_tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment_id", "populations", "reasons", "node_count",
                "diameter_um", "length_um", "tortuosity", "node_sequence",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in segment_rows:
            out = dict(row)
            out["diameter_um"] = _format_float(out["diameter_um"], 6)
            out["length_um"] = _format_float(out["length_um"], 6)
            out["tortuosity"] = _format_float(out["tortuosity"], 6)
            writer.writerow(out)

    vertices = _read_vertices(run_dir)
    spacing = _read_spacing_from_meta(run_dir)
    edge_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()
    for row in segment_rows:
        nodes = [int(float(x)) for x in str(row["node_sequence"]).split() if x]
        for u, v in zip(nodes[:-1], nodes[1:]):
            if u not in vertices or v not in vertices:
                continue
            key = (int(row["segment_id"]), min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edge_rows.append({
                "segment_id": row["segment_id"],
                "edge_u": u,
                "edge_v": v,
                "edge_length_um": _edge_length_um(
                    vertices[u], vertices[v], spacing, stats["length_um_factor"]
                ),
                "segment_length_um": row["length_um"],
                "segment_reasons": row["reasons"],
            })
    edge_rows.sort(key=lambda r: float(r["edge_length_um"]), reverse=True)

    with open(edge_tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment_id", "edge_u", "edge_v", "edge_length_um",
                "segment_length_um", "segment_reasons",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in edge_rows:
            out = dict(row)
            out["edge_length_um"] = _format_float(out["edge_length_um"], 6)
            out["segment_length_um"] = _format_float(out["segment_length_um"], 6)
            writer.writerow(out)

    with open(long_edge_tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "edge_u", "edge_v", "severity", "reasons", "edge_length_um",
                "median_ratio", "dx_um", "dy_um", "dz_um", "degree_u", "degree_v",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in long_edge_rows:
            out = dict(row)
            out["edge_length_um"] = _format_float(out["edge_length_um"], 6)
            out["median_ratio"] = _format_float(out["median_ratio"], 6)
            out["dx_um"] = _format_float(out["dx_um"], 6)
            out["dy_um"] = _format_float(out["dy_um"], 6)
            out["dz_um"] = _format_float(out["dz_um"], 6)
            writer.writerow(out)
    return json_path


def _resolve_sample_volume(sample_dir: str) -> Optional[float]:
    meta_path = os.path.join(sample_dir, "sample_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            vol = meta.get("spatial", {}).get("tissue_volume_mm3")
            if vol is not None:
                return float(vol)
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass
    return None


def _run_dirs(sample_dir: str) -> Iterable[str]:
    for entry in sorted(os.listdir(sample_dir)):
        run_dir = os.path.join(sample_dir, entry)
        if entry.startswith("run_") and os.path.isdir(run_dir):
            yield run_dir


def run_anomaly_stats(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
    *,
    species: str = "mouse_brain",
    force: bool = False,
) -> Dict[str, int]:
    from vascular_statistics.aggregate_stats import scan_sample_keys

    if sample_keys is None:
        sample_keys = scan_sample_keys(output_root)

    processed = 0
    skipped = 0
    no_result = 0
    failed = 0

    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)
        if not os.path.isdir(sample_dir):
            failed += 1
            print(f"  [FAIL] {sk} -- 目录不存在")
            continue

        volume = _resolve_sample_volume(sample_dir)
        if volume is None:
            failed += 1
            print(f"  [FAIL] {sk} -- 无法确定组织体积")
            continue

        for run_dir in _run_dirs(sample_dir):
            marker = os.path.join(run_dir, "skeleton_anomaly_summary.json")
            if os.path.exists(marker) and not force:
                skipped += 1
                continue
            try:
                result = write_run_anomaly_stats(run_dir, volume, species=species)
                if result:
                    processed += 1
                    print(f"  [{processed}] {sk}/{os.path.basename(run_dir)}")
                else:
                    no_result += 1
                    print(f"  [NO_RESULT] {sk}/{os.path.basename(run_dir)} -- 无段明细")
            except Exception as exc:
                failed += 1
                print(f"  [FAIL] {sk}/{os.path.basename(run_dir)}: {exc}")

    return {
        "processed": processed,
        "skipped": skipped,
        "no_result": no_result,
        "failed": failed,
    }

