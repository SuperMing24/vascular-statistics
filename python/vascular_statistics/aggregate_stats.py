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
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# 解析单次运行的统计输出
# ═══════════════════════════════════════════════════════════════════════

def parse_run_statistics(run_dir: str, suffix: str = "") -> Optional[Dict[str, Any]]:
    """解析单个 run_* 目录中的统计数据。

    参数：
        run_dir: run_YYYYMMDD_HHMMSS 目录路径。
        suffix: 统计文件后缀（"" = 主统计, "_d0-10um" = 0-10um 子范围）。

    返回：
        dict 含 4 项指标 + 段数 + 运行元数据，若目录无统计文件则返回 None。
    """
    stats_path = os.path.join(run_dir, f"statistics_summary{suffix}.txt")
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
    suffix: str = "",
) -> Optional[Dict[str, Any]]:
    """跨所有 run_* 目录聚合统计数据。

    参数：
        sample_dir: 样本目录路径（含 run_* 子目录）。
        suffix: 统计文件后缀（"" = 主统计, "_d0-10um" = 0-10um 子范围）。

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
        import warnings
        warnings.warn(
            f"口径混用: {sample_dir} 中同时存在 legacy 与 anisotropic 的 run，"
            f"聚合结果的均值口径不一致，建议重跑为新口径后替换。"
        )

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

    return {
        "sample_key": (
            sample_meta.get("sample_key", os.path.basename(sample_dir))
        ),
        "group": parsed.get("group", "?"),
        "batch_id": batch_id,
        "daypoint": parsed.get("daypoint", "?"),
        "volume_mm3": volume_mm3,
        "n_runs": len(runs_data),
        "runs": runs_data,
        "aggregates": aggregates,
        "metric_keys": metric_keys,
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

    is_subrange = suffix == "_d0-10um"
    stat_label = "子范围统计汇总（直径 0-10 um）" if is_subrange else "样本统计汇总"

    lines.append(sep)
    lines.append(f"  Vascular_Statistics — {stat_label}")
    lines.append(sep)
    lines.append(f"样本:    {agg['sample_key']}")
    lines.append(
        f"组别:    {agg['group']}  |  批次: {agg['batch_id']}  "
        f"|  时间点: {agg['daypoint']}"
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
                "仅统计直径 0-10 um 且节点数 >= 3 的血管段。"
            )
        else:
            lines.append(
                "注：段密度 = 有效段数 / 组织体积。"
                "直径 0-10 um 的血管段不计入。"
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
) -> Optional[str]:
    """汇总样本统计并写入 statistics_summary{suffix}.txt。

    参数：
        sample_dir: 样本目录路径（同一级目录下的 run_* 将被扫描）。
        suffix: 统计文件后缀（"" = 主统计, "_d0-10um" = 0-10um 子范围）。

    返回：
        写入的文件路径，若无有效数据返回 None。
    """
    agg = aggregate_sample_stats(sample_dir, suffix=suffix)
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

def run_aggregation(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
    suffix: str = "",
) -> dict:
    """对所有样本（或指定样本）运行统计聚合。

    参数：
        output_root: 输出根目录。
        sample_keys: 指定样本键列表。None = 全部。
        suffix: 统计文件后缀（"" = 主统计, "_d0-10um" = 0-10um 子范围）。

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
            result = write_aggregate_stats(sample_dir, suffix=suffix)
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
