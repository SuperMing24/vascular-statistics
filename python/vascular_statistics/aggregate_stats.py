"""
统计聚合模块 —— 汇总同一样本多次骨架化运行的形态学统计指标。

每个 run_* 目录中的 C++ 引擎输出 4 项指标（直径 / 长度 / 段密度 / 弯曲度）
和每条血管段的明细数据。本模块跨多次运行计算均值 ± 标准差，
写入样本级目录的 statistics_summary.txt（与 sample_metadata.json 同级）。
"""

from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# 解析单次运行的统计输出
# ═══════════════════════════════════════════════════════════════════════

# 子范围文件后缀常量（避免循环导入，与 subrange_stats.py 保持同步）
SUFFIX_D0_10 = "_d0-10um"
SUFFIX_D10_PLUS = "_d10+um"

_SUMMARY_LABELS = {
    "": "全量 full",
    SUFFIX_D0_10: "直径 0-10 um",
    SUFFIX_D10_PLUS: "直径 >=10 um",
}

_SUMMARY_FILENAMES = {
    "": "cross_sample_summary_full.txt",
    SUFFIX_D0_10: "cross_sample_summary_d0-10um.txt",
    SUFFIX_D10_PLUS: "cross_sample_summary_d10+um.txt",
}

_SUFFIX_POPULATION = {
    "": "full",
    SUFFIX_D0_10: "d0-10um",
    SUFFIX_D10_PLUS: "d10+um",
}

_QC_STATUS_RANK = {"OK": 0, "WARN": 1, "FAIL": 2}


def _read_run_anomaly_population(run_dir: str, population: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "skeleton_anomaly_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("populations", {}).get(population)
    except (OSError, json.JSONDecodeError):
        return None



def _read_run_anomaly_edge_qc(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "skeleton_anomaly_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("edge_qc")
    except (OSError, json.JSONDecodeError):
        return None

def _aggregate_anomaly_qc(runs_data: List[Dict[str, Any]], population: str) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    edge_entries: List[Dict[str, Any]] = []
    for run in runs_data:
        run_dir = run.get("run_dir")
        if not run_dir:
            continue
        pop = _read_run_anomaly_population(run_dir, population)
        if pop:
            entries.append(pop)
        edge_qc = _read_run_anomaly_edge_qc(run_dir)
        if edge_qc:
            edge_entries.append(edge_qc)

    if not entries and not edge_entries:
        return {"n_runs_with_qc": 0, "n_runs_with_edge_qc": 0, "status": "NA"}

    status_candidates = [entry.get("status", "OK") for entry in entries]
    status_candidates.extend(
        entry.get("status", "OK") for entry in edge_entries
        if entry.get("status") in _QC_STATUS_RANK
    )
    status = max(status_candidates or ["OK"], key=lambda x: _QC_STATUS_RANK.get(x, 0))
    edge_status = max(
        (entry.get("status", "OK") for entry in edge_entries),
        key=lambda x: _QC_STATUS_RANK.get(x, 0),
    ) if edge_entries else "NA"
    violations = sorted({
        item
        for entry in entries
        for item in entry.get("range_violations", [])
    })

    result = {
        "n_runs_with_qc": len(entries),
        "n_runs_with_edge_qc": len(edge_entries),
        "status": status,
        "edge_status": edge_status,
        "max_edge_length_um": max(
            float(entry.get("max_edge_length_um", 0.0)) for entry in edge_entries
        ) if edge_entries else 0.0,
        "edge_fail_edges": sum(int(entry.get("fail_edges", 0)) for entry in edge_entries),
        "edge_warn_edges": sum(int(entry.get("warn_edges", 0)) for entry in edge_entries),
        "range_violations": violations,
    }
    if entries:
        result.update({
            "max_abs_ratio": max(float(entry.get("abs_ratio", 0.0)) for entry in entries),
            "max_mad_ratio": max(float(entry.get("mad_ratio", 0.0)) for entry in entries),
            "max_p99_ratio": max(float(entry.get("p99_ratio", 0.0)) for entry in entries),
            "abs_outliers": sum(int(entry.get("abs_outliers", 0)) for entry in entries),
            "mad_outliers": sum(int(entry.get("mad_outliers", 0)) for entry in entries),
            "invalid_tortuosity": sum(int(entry.get("invalid_tortuosity", 0)) for entry in entries),
        })
    else:
        result.update({
            "max_abs_ratio": 0.0,
            "max_mad_ratio": 0.0,
            "max_p99_ratio": 0.0,
            "abs_outliers": 0,
            "mad_outliers": 0,
            "invalid_tortuosity": 0,
        })
    return result

def _timing_from_path_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """从 metadata.path_parsed 取标准化时间字段，兼容旧 metadata。"""
    daypoint = parsed.get("daypoint", "?")
    study_day_label = parsed.get("study_day_label") or daypoint
    study_day = parsed.get("study_day")

    if study_day is None and isinstance(study_day_label, str):
        m = re.match(r"^D(\d+(?:\+\d+)*)$", study_day_label, re.IGNORECASE)
        if m:
            study_day = sum(int(part) for part in m.group(1).split("+"))

    return {
        "daypoint": daypoint,
        "study_day": study_day,
        "study_day_label": study_day_label,
        "acquisition_date": parsed.get("acquisition_date", "?"),
        "timepoint_id": parsed.get("timepoint_id", "?"),
        "source_time_token": parsed.get("source_time_token", "?"),
    }


def _sort_day(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.match(r"^D(\d+(?:\+\d+)*)$", value, re.IGNORECASE)
        if m:
            return sum(int(part) for part in m.group(1).split("+"))
    return 10**9


def parse_run_statistics(run_dir: str, suffix: str = "") -> Optional[Dict[str, Any]]:
    """解析单个 run_* 目录中的统计数据。

    参数：
        run_dir: run_YYYYMMDD_HHMMSS 目录路径。
        suffix: 统计文件后缀（默认 "" = 全量；_d10+um = >=10um, _d0-10um = 0-10um 子范围）。

    返回：
        dict 含 4 项指标 + 段数 + 运行元数据，若目录无统计文件则返回 None。
    """
    stats_path = os.path.join(run_dir, f"statistics_summary{suffix}.txt")
    meta_path = os.path.join(run_dir, "run_meta.json")
    # 段数明细：优先与 suffix 对应文件；全量(suffix="")回退旧 _d10+um 名
    # （存量 run 的 generate_vessel_d10+um.txt 内容本就是全量）。
    vessel_path = os.path.join(run_dir, f"generate_vessel{suffix}.txt")
    if not os.path.exists(vessel_path) and suffix == "":
        vessel_path = os.path.join(run_dir, f"generate_vessel{SUFFIX_D10_PLUS}.txt")

    if not os.path.exists(stats_path):
        return None

    # 解析 C++ 统计输出
    metrics: Dict[str, float] = {}
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(
                    r"^(平均直径|平均长度|段密度|平均弯曲度)"
                    r"(?:\s*[\(（].*[\)）])?\s*:\s*"
                    r"(-?[\d.]+|-?inf|-?nan)",
                    line,
                )
                if m:
                    key_map = {
                        "平均直径": "avg_diameter_um",
                        "平均长度": "avg_length_um",
                        "段密度": "segment_density_per_mm3",
                        "平均弯曲度": "avg_tortuosity_au",
                    }
                    metrics[key_map[m.group(1)]] = float(m.group(2))
    except OSError:
        return None

    if len(metrics) < 4:
        return None  # 文件不完整

    # 数段数（generate_vessel_d10+um.txt 每行一段）
    segment_count = 0
    if os.path.exists(vessel_path):
        try:
            with open(vessel_path, "r", encoding="utf-8") as f:
                segment_count = sum(1 for line in f if line.strip())
        except OSError:
            pass

    # 运行元数据
    run_meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                run_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "run_dir": run_dir,
        "run_id": run_meta.get("run_id", os.path.basename(run_dir)),
        "job_id": run_meta.get("job_id", "?"),
        "node": run_meta.get("slurm_node", "?"),
        "sampling": run_meta.get("sampling", "?"),
        "volume_mm3": run_meta.get("volume_mm3", None),
        "segment_count": segment_count,
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# 聚合
# ═══════════════════════════════════════════════════════════════════════

def aggregate_sample_stats(
    sample_dir: str,
    suffix: str = "",
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """跨所有 run_* 目录聚合统计数据。

    参数：
        sample_dir: 样本目录路径（含 run_* 子目录）。
        suffix: 统计文件后缀（"" = 全量, "_d0-10um"/"_d10+um" = 子范围）。

    返回：
        聚合 dict，含 runs 列表 + 均值/标准差 + 样本元数据，
        若无有效 run 则返回 None。
    """
    if not os.path.isdir(sample_dir):
        return None

    # 收集所有 run 的统计数据
    runs_data: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(sample_dir)):
        run_dir = os.path.join(sample_dir, entry)
        if not entry.startswith("run_") or not os.path.isdir(run_dir):
            continue
        parsed = parse_run_statistics(run_dir, suffix=suffix)
        if parsed is not None:
            runs_data.append(parsed)

    if not runs_data:
        return None

    # --- 口径一致性守卫（轨道 B）：检查 run 之间是否存在混口径 ---
    calibers: List[str] = []
    for entry in sorted(os.listdir(sample_dir)):
        run_dir = os.path.join(sample_dir, entry)
        if not entry.startswith("run_") or not os.path.isdir(run_dir):
            continue
        meta_path = os.path.join(run_dir, "run_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    rm = json.load(f)
                cm = rm.get("stats_unit_mode")
                if cm:
                    calibers.append(cm)
            except (json.JSONDecodeError, OSError):
                pass
    if len(set(calibers)) > 1:
        msg = (
            f"口径混用: {sample_dir} 中同时存在 legacy 与 anisotropic 的 run "
            f"(modes={set(calibers)})，聚合均值±标准差物理无意义。"
            f"请统一口径后重跑，或使用 --force 强制聚合。"
        )
        if force:
            import warnings
            warnings.warn(f"FORCED: {msg}")
        else:
            raise ValueError(msg)

    # 样本元数据（来自 sample_metadata.json）
    sample_meta: Dict[str, Any] = {}
    meta_path = os.path.join(sample_dir, "sample_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                sample_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # 计算聚合并替换到摘要文件
    metric_keys = [
        ("avg_diameter_um", "平均直径", "μm"),
        ("avg_length_um", "平均长度", "μm"),
        ("segment_density_per_mm3", "段密度", "seg/mm³"),
        ("avg_tortuosity_au", "平均弯曲度", "a.u."),
        ("segment_count", "血管段数", "段"),
    ]

    aggregates = {}
    for key, _label, _unit in metric_keys:
        values = [r[key] for r in runs_data if key in r]
        if len(values) >= 2:
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values)
            aggregates[key] = {
                "mean": mean_val,
                "stdev": stdev_val,
                "min": min(values),
                "max": max(values),
                "n": len(values),
                "values": values,
            }
        elif len(values) == 1:
            aggregates[key] = {
                "mean": values[0],
                "stdev": None,
                "min": values[0],
                "max": values[0],
                "n": 1,
                "values": values,
            }

    # 确定组织体积（来自 sample_metadata.json）
    volume_mm3 = None
    if sample_meta:
        volume_mm3 = (
            sample_meta.get("spatial", {}).get("tissue_volume_mm3")
        )

    # 兼容旧字段名 (animal_id) 与新字段名 (batch_id)
    parsed = sample_meta.get("path_parsed", {})
    batch_id = parsed.get("batch_id") or parsed.get("animal_id", "?")
    timing = _timing_from_path_parsed(parsed)
    population = _SUFFIX_POPULATION.get(suffix, "full")
    anomaly_qc = _aggregate_anomaly_qc(runs_data, population)

    return {
        "sample_key": (
            sample_meta.get("sample_key", os.path.basename(sample_dir))
        ),
        "group": parsed.get("group", "?"),
        "batch_id": batch_id,
        **timing,
        "volume_mm3": volume_mm3,
        "n_runs": len(runs_data),
        "runs": runs_data,
        "aggregates": aggregates,
        "metric_keys": metric_keys,
        "anomaly_population": population,
        "anomaly_qc": anomaly_qc,
    }


# ═══════════════════════════════════════════════════════════════════════
# 格式化输出
# ═══════════════════════════════════════════════════════════════════════

def format_aggregate_stats(agg: Dict[str, Any], suffix: str = "") -> str:
    """将聚合结果格式化为可读文本。

    参数：
        agg: aggregate_sample_stats 返回的聚合 dict。
        suffix: 统计文件后缀（"" = 主统计, "_d0-10um" = 0-10um 子范围）。
    """
    lines: List[str] = []
    sep = "=" * 78

    _suffix_labels = {"_d0-10um": "直径 0-10 um", "_d10+um": "直径 >=10 um"}
    is_subrange = suffix != ""
    sub_label = _suffix_labels.get(suffix, suffix)
    stat_label = f"子范围统计汇总（{sub_label}）" if is_subrange else "样本统计汇总（全量）"

    lines.append(sep)
    lines.append(f"  Vascular_Statistics — {stat_label}")
    lines.append(sep)
    lines.append(f"样本:    {agg['sample_key']}")
    lines.append(
        f"组别:    {agg['group']}  |  批次: {agg['batch_id']}  "
        f"|  研究日: {agg.get('study_day_label', agg.get('daypoint', '?'))}"
    )
    lines.append(
        f"采集日期: {agg.get('acquisition_date', '?')}  "
        f"|  时间点ID: {agg.get('timepoint_id', '?')}"
    )
    vol = agg.get("volume_mm3")
    if vol is not None:
        lines.append(f"体积:    {vol:.6f} mm³")
    lines.append(sep)
    lines.append("")

    runs = agg["runs"]
    n = len(runs)

    # 逐次运行明细表
    run_header = (
        f"  {'':<10s} | {'采样':>4s} | {'段数':>5s} "
        f"| {'直径(μm)':>10s} | {'长度(μm)':>10s} "
        f"| {'弯曲度':>8s}"
    )
    lines.append(run_header)
    lines.append("  " + "-" * (len(run_header) - 2))
    for i, r in enumerate(runs, start=1):
        label = f"运行 #{i}"
        tort = r.get('avg_tortuosity_au', 0)
        lines.append(
            f"  {label:<10s} | {str(r['sampling']):>4s} "
            f"| {r.get('segment_count', '?'):>5} "
            f"| {r.get('avg_diameter_um', 0):>10.2f} "
            f"| {r.get('avg_length_um', 0):>10.2f} "
            f"| {tort:>8.5f}"
        )
    lines.append("")

    # 统计指标表（n≥2 时含均值/标准差，n=1 时仅展示值）
    aggregates = agg["aggregates"]

    def _fmt(v: float | None) -> str:
        if v is None:
            return "       N/A"
        if v != v:  # NaN
            return "       N/A"
        if v == float("inf") or v == float("-inf"):
            return "       N/A"
        return f"{v:>10.4f}"

    if n >= 2:
        lines.append(f"基于 {n} 次独立骨架化与统计结果的汇总：")
        lines.append("")
        stat_header = (
            f"  {'统计指标':<28s} | {'均值':>10s} | {'标准差':>10s} "
            f"| {'最小':>10s} | {'最大':>10s}"
        )
    else:
        lines.append("统计结果（单次运行）：")
        lines.append("")
        stat_header = f"  {'统计指标':<28s} | {'值':>10s}"

    lines.append(stat_header)
    lines.append("  " + "-" * (len(stat_header) - 2))

    for key, label, unit in agg["metric_keys"]:
        entry = aggregates.get(key, {})
        mean_v = entry.get("mean")
        stdev_v = entry.get("stdev")
        min_v = entry.get("min")
        max_v = entry.get("max")

        if mean_v is None:
            continue

        label_full = f"{label} ({unit})"
        if n >= 2 and stdev_v is not None:
            lines.append(
                f"  {label_full:<28s} | {_fmt(mean_v)} "
                f"| {_fmt(stdev_v)} | {_fmt(min_v)} | {_fmt(max_v)}"
            )
        elif n >= 2:
            lines.append(
                f"  {label_full:<28s} | {_fmt(mean_v)} "
                f"| {'       N/A':>10s} | {_fmt(min_v)} | {_fmt(max_v)}"
            )
        else:
            lines.append(
                f"  {label_full:<28s} | {_fmt(mean_v)}"
            )
    lines.append("")

    qc = agg.get("anomaly_qc", {})
    if qc.get("n_runs_with_qc", 0) > 0 or qc.get("n_runs_with_edge_qc", 0) > 0:
        violations = qc.get("range_violations") or []
        violation_text = ", ".join(violations) if violations else "-"
        lines.append(f"--- 骨架异常诊断（{agg.get('anomaly_population', 'full')}）---")
        lines.append(
            f"QC状态: {qc.get('status', 'NA')}  "
            f"| 含段级QC的run: {qc.get('n_runs_with_qc', 0)}/{agg.get('n_runs', 0)}  "
            f"| 含单边QC的run: {qc.get('n_runs_with_edge_qc', 0)}/{agg.get('n_runs', 0)}"
        )
        lines.append(
            f"最大超绝对阈值段比例: {qc.get('max_abs_ratio', 0.0):.4f}%  "
            f"| 最大MAD离群段比例: {qc.get('max_mad_ratio', 0.0):.4f}%  "
            f"| 最大P99离群段比例: {qc.get('max_p99_ratio', 0.0):.4f}%"
        )
        lines.append(
            f"超长段总数: {qc.get('abs_outliers', 0)}  "
            f"| MAD离群段总数: {qc.get('mad_outliers', 0)}  "
            f"| inf/NaN弯曲度段总数: {qc.get('invalid_tortuosity', 0)}"
        )
        lines.append(
            f"拓扑单边QC: {qc.get('edge_status', 'NA')}  "
            f"| 最长单边: {qc.get('max_edge_length_um', 0.0):.4f} um  "
            f"| FAIL边: {qc.get('edge_fail_edges', 0)}  "
            f"| WARN边: {qc.get('edge_warn_edges', 0)}"
        )
        lines.append(f"范围违规: {violation_text}")
        lines.append(
            "明细: 各 run 的 skeleton_anomaly_summary.*、"
            "skeleton_segment_anomalies.tsv、skeleton_edge_anomalies.tsv"
        )
        lines.append("")

    # 段密度说明
    density_entry = aggregates.get("segment_density_per_mm3", {})
    if density_entry:
        d_mean = density_entry.get("mean")
        if d_mean is not None and d_mean != d_mean:
            # NaN density (volume=0 导致)
            lines.append(
                "注：段密度 = 有效段数 / 组织体积。"
                "当前样本 volume 可能未正确设置，导致密度为 N/A。"
            )
        elif is_subrange:
            lines.append(
                "注：段密度 = 子范围段数 / 组织体积。"
                f"仅统计 {sub_label} 且节点数 >= 3 的血管段。"
            )
        else:
            lines.append(
                "注：段密度 = 全量段数 / 组织体积。"
                "纳入所有完成遍历的血管段（无直径/节点数/P99 过滤）。"
            )
        lines.append("")

    # 脚注
    lines.append(sep)
    if n >= 2:
        lines.append(
            "方法: 各次运行统计值（平均直径 / 平均长度 / 段密度 / "
            "平均弯曲度）取均值 ± 样本标准差（n=" + str(n) + "）。"
        )
    else:
        lines.append(
            "方法: 单次运行统计值。有多次骨架化运行后，重新运行 "
            "aggregate-stats 将自动计算均值 ± 标准差。"
        )
    lines.append(sep)

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════════════════════════════════

def write_aggregate_stats(
    sample_dir: str,
    suffix: str = "",
    force: bool = False,
) -> Optional[str]:
    """汇总样本统计并写入 statistics_summary{suffix}.txt。

    参数：
        sample_dir: 样本目录路径（同一级目录下的 run_* 将被扫描）。
        suffix: 统计文件后缀（"" = 全量, "_d0-10um"/"_d10+um" = 子范围）。

    返回：
        写入的文件路径，若无有效数据返回 None。
    """
    agg = aggregate_sample_stats(sample_dir, suffix=suffix, force=force)
    if agg is None:
        return None

    out_path = os.path.join(sample_dir, f"statistics_summary{suffix}.txt")
    formatted = format_aggregate_stats(agg, suffix=suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# 批量编排
# ═══════════════════════════════════════════════════════════════════════

def scan_sample_keys(output_root: str) -> List[str]:
    """扫描输出根目录，返回含 run_* 子目录的样本键列表。"""
    sample_keys: List[str] = []
    if not os.path.isdir(output_root):
        return sample_keys
    for group in os.listdir(output_root):
        group_dir = os.path.join(output_root, group)
        if not os.path.isdir(group_dir):
            continue
        for date_dir in os.listdir(group_dir):
            date_path = os.path.join(group_dir, date_dir)
            if not os.path.isdir(date_path):
                continue
            for sample in os.listdir(date_path):
                sample_path = os.path.join(date_path, sample)
                if not os.path.isdir(sample_path):
                    continue
                has_runs = any(
                    d.startswith("run_")
                    for d in os.listdir(sample_path)
                    if os.path.isdir(os.path.join(sample_path, d))
                )
                if has_runs:
                    rel = os.path.relpath(
                        sample_path, output_root
                    ).replace("\\", "/")
                    sample_keys.append(rel)
    return sample_keys


def run_aggregation(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
    suffix: str = "",
    force: bool = False,
) -> dict:
    """对所有样本（或指定样本）运行统计聚合。

    参数：
        output_root: 输出根目录。
        sample_keys: 指定样本键列表。None = 全部。
        suffix: 统计文件后缀（"" = 全量, "_d0-10um"/"_d10+um" = 子范围）。

    返回：
        {"processed": N, "skipped": N, "failed": N}
    """
    if sample_keys is None:
        if not os.path.isdir(output_root):
            return {"processed": 0, "skipped": 0, "failed": 0,
                    "error": "output_root 不存在"}
        sample_keys = scan_sample_keys(output_root)

    processed = 0
    skipped = 0
    failed = 0

    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)
        try:
            result = write_aggregate_stats(sample_dir, suffix=suffix, force=force)
            if result:
                processed += 1
                print(f"  [{processed}] {sk}")
            else:
                skipped += 1
                print(f"  [SKIP] {sk} — 无有效统计数据")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {sk}: {e}")

    return {"processed": processed, "skipped": skipped, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════
# 跨样本汇总
# ═══════════════════════════════════════════════════════════════════════

def generate_cross_sample_summary(
    output_root: str,
    *,
    suffix: str = "",
    sample_keys: Optional[List[str]] = None,
    force: bool = False,
) -> Optional[str]:
    """生成跨样本统计汇总（full / d0-10um / d10+um）。

    从每个样本的 run_* 统计重新聚合，不依赖样本级 statistics_summary*.txt
    是否已存在；因此可在 run 级统计刷新后直接生成 root 级汇总。
    """
    if suffix not in _SUMMARY_FILENAMES:
        raise ValueError(f"不支持的统计后缀: {suffix!r}")

    if sample_keys is None:
        sample_keys = scan_sample_keys(output_root)

    rows: List[Dict[str, Any]] = []
    failed = 0
    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)
        try:
            agg = aggregate_sample_stats(sample_dir, suffix=suffix, force=force)
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {sk}: {exc}")
            continue
        if agg is None:
            continue

        aggregates = agg.get("aggregates", {})

        def _get_mean(key: str) -> Optional[float]:
            entry = aggregates.get(key)
            if entry is None:
                return None
            mean_v = entry.get("mean")
            if mean_v is None or mean_v != mean_v:
                return None
            return float(mean_v)

        avg_diameter = _get_mean("avg_diameter_um")
        avg_length = _get_mean("avg_length_um")
        seg_density = _get_mean("segment_density_per_mm3")
        avg_tortuosity = _get_mean("avg_tortuosity_au")
        segment_count = _get_mean("segment_count")

        if avg_diameter is None:
            continue

        rows.append({
            "sample_key": sk,
            "group": agg.get("group", "?"),
            "batch_id": agg.get("batch_id", "?"),
            "daypoint": agg.get("daypoint", "?"),
            "study_day": agg.get("study_day"),
            "study_day_label": agg.get("study_day_label", agg.get("daypoint", "?")),
            "acquisition_date": agg.get("acquisition_date", "?"),
            "timepoint_id": agg.get("timepoint_id", "?"),
            "n_runs": agg.get("n_runs", 0),
            "avg_diameter_um": avg_diameter,
            "avg_length_um": avg_length or 0.0,
            "segment_density_per_mm3": seg_density or 0.0,
            "avg_tortuosity_au": avg_tortuosity or 0.0,
            "segment_count": segment_count or 0.0,
            "qc_status": agg.get("anomaly_qc", {}).get("status", "NA"),
            "qc_abs_ratio": agg.get("anomaly_qc", {}).get("max_abs_ratio", 0.0),
            "qc_mad_ratio": agg.get("anomaly_qc", {}).get("max_mad_ratio", 0.0),
            "qc_edge_max_um": agg.get("anomaly_qc", {}).get("max_edge_length_um", 0.0),
        })

    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            str(row.get("group", "")),
            _sort_day(row.get("study_day")),
            str(row.get("acquisition_date", "")),
            str(row.get("batch_id", "")),
            str(row.get("sample_key", "")),
        )
    )

    label = _SUMMARY_LABELS[suffix]
    out_path = os.path.join(output_root, _SUMMARY_FILENAMES[suffix])
    sep = "=" * 150

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"  Vascular_Statistics -- 跨样本统计汇总（{label}）\n")
        f.write(f"{sep}\n")
        f.write(f"  样本数: {len(rows)}\n")
        f.write(f"  数据源: {output_root}\n")
        f.write(f"  统计文件: statistics_summary{suffix}.txt\n")
        if failed:
            f.write(f"  聚合失败样本数: {failed}\n")
        f.write(f"{sep}\n\n")

        header = (
            f"  {'样本':<50s} | {'组别':>16s} | {'批次':>8s} "
            f"| {'研究日':>6s} | {'采集日期':>10s} | {'时间点ID':>12s} "
            f"| {'run数':>5s} | {'QC':>4s} | {'超长%':>8s} | {'MAD%':>8s} | {'长边um':>8s} "
            f"| {'段数':>9s} | {'直径(um)':>10s} "
            f"| {'长度(um)':>10s} | {'段密度':>10s} | {'弯曲度':>8s}"
        )
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")

        for row in rows:
            key_display = row["sample_key"]
            if len(key_display) > 49:
                key_display = "..." + key_display[-46:]

            f.write(
                f"  {key_display:<50s} | {row['group']:>16s} "
                f"| {row['batch_id']:>8s} "
                f"| {row['study_day_label']:>6s} "
                f"| {row['acquisition_date']:>10s} "
                f"| {row['timepoint_id']:>12s} "
                f"| {row['n_runs']:>5} "
                f"| {row['qc_status']:>4s} "
                f"| {row['qc_abs_ratio']:>8.4f} "
                f"| {row['qc_mad_ratio']:>8.4f} "
                f"| {row['qc_edge_max_um']:>8.2f} "
                f"| {row['segment_count']:>9.1f} "
                f"| {row['avg_diameter_um']:>10.4f} "
                f"| {row['avg_length_um']:>10.4f} "
                f"| {row['segment_density_per_mm3']:>10.4f} "
                f"| {row['avg_tortuosity_au']:>8.5f}\n"
            )

        f.write("\n")
        f.write(f"{sep}\n")
        f.write(
            "方法: 每个样本先聚合其 run_* 统计值，再汇总为跨样本视图。"
            "指标列为样本级聚合均值。\n"
        )
        f.write(
            "元数据: 批次=batch_id（Axxx 成像/实验批次，非动物编号）；"
            "研究日=study_day_label；采集日期=acquisition_date；"
            "时间点ID=YYYYMMDD_Dn，用于显示与追溯。排序按组别、研究日、采集日期、批次。\n"
        )
        f.write(
            "指标: 段数 / 平均直径(um) / 平均长度(um) / 段密度(seg/mm^3) / "
            "平均弯曲度(a.u.)\n"
        )
        f.write(
            "QC: 读取各 run 的 skeleton_anomaly_summary.json；"
            "QC 为最严重状态，超长%/MAD% 为该样本各 run 的最大比例；"
            "长边um为各 run 的最长 skeleton_edges 单边长度。\n"
        )
        f.write(f"{sep}\n")

    return out_path

