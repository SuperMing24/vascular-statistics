"""
统计聚合模块 —— 汇总同一样本多次骨架化运行的形态学统计指标。

每个 run_* 目录中的 C++ 引擎输出 4 项指标（直径 / 长度 / 段密度 / 弯曲度）
和每条血管段的明细数据。本模块跨多次运行计算均值 ± 标准差，
写入样本级目录的 statistics_summary.txt（与 sample_metadata.json 同级）。
"""

import json
import os
import re
import statistics
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# 解析单次运行的统计输出
# ═══════════════════════════════════════════════════════════════════════

def parse_run_statistics(run_dir: str) -> Optional[Dict[str, Any]]:
    """解析单个 run_* 目录中的统计数据。

    参数：
        run_dir: run_YYYYMMDD_HHMMSS 目录路径。

    返回：
        dict 含 4 项指标 + 段数 + 运行元数据，若目录无统计文件则返回 None。
    """
    stats_path = os.path.join(run_dir, "statistics_summary.txt")
    meta_path = os.path.join(run_dir, "run_meta.json")
    vessel_path = os.path.join(run_dir, "generate_vessel.txt")

    if not os.path.exists(stats_path):
        return None

    # 解析 C++ 统计输出
    metrics: Dict[str, float] = {}
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(
                    r"^(平均直径|平均长度|段密度|平均弯曲度)\s*[\(（].*[\)）]\s*:\s*([\d.]+)",
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

    # 数段数（generate_vessel.txt 每行一段）
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
) -> Optional[Dict[str, Any]]:
    """跨所有 run_* 目录聚合统计数据。

    参数：
        sample_dir: 样本目录路径（含 run_* 子目录）。

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
        parsed = parse_run_statistics(run_dir)
        if parsed is not None:
            runs_data.append(parsed)

    if not runs_data:
        return None

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

    # 确定组织体积（优先 sample_metadata，其次 run_meta）
    volume_mm3 = None
    if sample_meta:
        volume_mm3 = (
            sample_meta.get("spatial", {}).get("tissue_volume_mm3")
        )
    if volume_mm3 is None:
        for r in runs_data:
            if r.get("volume_mm3") is not None:
                volume_mm3 = r["volume_mm3"]
                break

    return {
        "sample_key": (
            sample_meta.get("sample_key", os.path.basename(sample_dir))
        ),
        "group": sample_meta.get("path_parsed", {}).get("group", "?"),
        "animal_id": sample_meta.get("path_parsed", {}).get("animal_id", "?"),
        "daypoint": sample_meta.get("path_parsed", {}).get("daypoint", "?"),
        "volume_mm3": volume_mm3,
        "n_runs": len(runs_data),
        "runs": runs_data,
        "aggregates": aggregates,
        "metric_keys": metric_keys,
        "generated_at": datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# 格式化输出
# ═══════════════════════════════════════════════════════════════════════

def format_aggregate_stats(agg: Dict[str, Any]) -> str:
    """将聚合结果格式化为可读文本。"""
    lines: List[str] = []
    sep = "=" * 78

    lines.append(sep)
    lines.append("  Vascular_Statistics — 样本统计汇总")
    lines.append(sep)
    lines.append(f"样本:    {agg['sample_key']}")
    lines.append(
        f"组别:    {agg['group']}  |  动物: {agg['animal_id']}  "
        f"|  时间点: {agg['daypoint']}"
    )
    vol = agg.get("volume_mm3")
    if vol is not None:
        lines.append(f"体积:    {vol:.6f} mm³")
    lines.append(sep)
    lines.append("")

    runs = agg["runs"]
    n = len(runs)

    if n == 1:
        lines.append(f"仅 1 次运行，无聚合标准差。")
        lines.append("")
    else:
        lines.append(f"基于 {n} 次骨架化运行的统计：")
        lines.append("")

    # 逐次运行明细表
    header = (
        f"  {'运行':<32s} | {'采样':>4s} | {'段数':>5s} "
        f"| {'直径(μm)':>10s} | {'长度(μm)':>10s} "
        f"| {'弯曲度':>8s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in runs:
        run_short = r["run_id"][:28] if len(r["run_id"]) > 28 else r["run_id"]
        lines.append(
            f"  {run_short:<32s} | {str(r['sampling']):>4s} "
            f"| {r.get('segment_count', '?'):>5} "
            f"| {r.get('avg_diameter_um', 0):>10.2f} "
            f"| {r.get('avg_length_um', 0):>10.2f} "
            f"| {r.get('avg_tortuosity_au', 0):>8.5f}"
        )
    lines.append("")

    # 聚合统计表
    aggregates = agg["aggregates"]
    if n >= 2:
        lines.append(f"均值和标准差（n={n}）：")
        lines.append("")
        stat_header = (
            f"  {'统计指标':<28s} | {'均值':>10s} | {'标准差':>10s} "
            f"| {'最小':>10s} | {'最大':>10s}"
        )
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
            if stdev_v is not None:
                lines.append(
                    f"  {label_full:<28s} | {mean_v:>10.4f} "
                    f"| {stdev_v:>10.4f} | {min_v:>10.4f} | {max_v:>10.4f}"
                )
            else:
                lines.append(
                    f"  {label_full:<28s} | {mean_v:>10.4f} "
                    f"| {'—':>10s} | {min_v:>10.4f} | {max_v:>10.4f}"
                )
        lines.append("")

    # 段密度单独说明（依赖 volume）
    density_entry = aggregates.get("segment_density_per_mm3", {})
    if density_entry:
        lines.append(
            "注：段密度 = 有效段数 / 组织体积。"
            "直径 < 10 μm 的微血管段不计入。"
        )
        lines.append("")

    # 脚注
    lines.append(sep)
    lines.append(f"生成: {agg['generated_at']}")
    if n >= 2:
        lines.append(
            "方法: 每个 run 的 C++ 统计值（平均直径 / 平均长度 / 段密度 / "
            "平均弯曲度）取均值 ± 样本标准差。"
        )
    lines.append(sep)

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════════════════════════════════

def write_aggregate_stats(
    sample_dir: str,
) -> Optional[str]:
    """汇总样本统计并写入 statistics_summary.txt。

    参数：
        sample_dir: 样本目录路径（同一级目录下的 run_* 将被扫描）。

    返回：
        写入的文件路径，若无有效数据返回 None。
    """
    agg = aggregate_sample_stats(sample_dir)
    if agg is None:
        return None

    out_path = os.path.join(sample_dir, "statistics_summary.txt")
    formatted = format_aggregate_stats(agg)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# 批量编排
# ═══════════════════════════════════════════════════════════════════════

def run_aggregation(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
) -> dict:
    """对所有样本（或指定样本）运行统计聚合。

    参数：
        output_root: 输出根目录。
        sample_keys: 指定样本键列表。None = 全部。

    返回：
        {"processed": N, "skipped": N, "failed": N}
    """
    if sample_keys is None:
        # 扫描全部含 run_* 子目录的样本
        sample_keys = []
        if not os.path.isdir(output_root):
            return {"processed": 0, "skipped": 0, "failed": 0,
                    "error": "output_root 不存在"}
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
                    # 检查是否有 run_* 子目录
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

    processed = 0
    skipped = 0
    failed = 0

    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)
        try:
            result = write_aggregate_stats(sample_dir)
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
