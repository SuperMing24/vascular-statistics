"""
微血管统计模块 —— 从已有骨架化运行结果中提取直径 < 10 μm 的微血管段统计。

背景：
  C++ 引擎输出的 statistics_summary.txt 仅统计直径 ≥ 10 μm 的"有效段"，
  但 generate_vessel_radius.txt / generate_vessel_path_length.txt /
  generate_vessel_tortuosity.txt 中保留了全部血管段的明细数据。
  本模块从这些明细文件中过滤出直径 < 10 μm（半径 < 2.5 μm）的微血管段，
  生成独立的微血管统计文件，不覆盖原有输出。

阈值依据：
  C++ data_io.cpp 中 if (avg_temp >= 2.5) 对应直径 ≥ 10 μm
  （avg_temp 为段平均半径，输出时直径 = avg_temp * 4）。
  因此微血管过滤条件为 avg_radius < 2.5。

输出（per-run）：
  statistics_summary_micro.txt       — 微血管汇总统计
  generate_vessel_micro.txt          — 微血管段节点序列
  generate_vessel_radius_micro.txt   — 微血管段平均半径
  generate_vessel_path_length_micro.txt — 微血管段路径长度
  generate_vessel_tortuosity_micro.txt  — 微血管段弯曲度
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# 阈值常量
# ═══════════════════════════════════════════════════════════════════════

# C++ 引擎中 avg_temp >= 2.5 判定为"有效段"（直径 >= 10 μm）。
# 微血管取反：半径 < 2.5 μm。
MICRO_RADIUS_THRESHOLD = 2.5
MICRO_MIN_NODES = 3  # 与 C++ 一致：k > 2 即至少 3 个节点


# ═══════════════════════════════════════════════════════════════════════
# 解析 per-run 段明细文件
# ═══════════════════════════════════════════════════════════════════════

def parse_segment_data(run_dir: str) -> Optional[Dict[str, Any]]:
    """读取单个 run_* 目录中的全部血管段明细数据。

    参数：
        run_dir: run_YYYYMMDD_HHMMSS 目录路径。

    返回：
        dict 含 4 个列表（一一对应，按行号对齐），若关键文件缺失则返回 None。
        {
            "radii": [float, ...],        # 每段平均半径 (μm)
            "path_lengths": [float, ...], # 每段路径长度 (μm)
            "tortuosities": [float, ...], # 每段弯曲度 (a.u.)
            "node_sequences": [str, ...], # 每段节点序列（原始行文本）
        }
    """
    radius_path = os.path.join(run_dir, "generate_vessel_radius.txt")
    length_path = os.path.join(run_dir, "generate_vessel_path_length.txt")
    tort_path = os.path.join(run_dir, "generate_vessel_tortuosity.txt")
    vessel_path = os.path.join(run_dir, "generate_vessel.txt")

    if not os.path.exists(radius_path):
        return None

    try:
        with open(radius_path, "r", encoding="utf-8") as f:
            radii = [float(line.strip()) for line in f if line.strip()]
    except (OSError, ValueError):
        return None

    n_segments = len(radii)
    if n_segments == 0:
        return None

    # 辅助：读取单列浮点文件
    def _read_floats(path: str, expected: int) -> List[float]:
        if not os.path.exists(path):
            return [0.0] * expected
        try:
            with open(path, "r", encoding="utf-8") as f:
                values = [float(line.strip()) for line in f if line.strip()]
        except (OSError, ValueError):
            return [0.0] * expected
        # 补齐或截断到 expected 长度
        if len(values) < expected:
            values.extend([0.0] * (expected - len(values)))
        return values[:expected]

    path_lengths = _read_floats(length_path, n_segments)
    tortuosities = _read_floats(tort_path, n_segments)

    # 读取节点序列（保留原始文本行）
    node_sequences: List[str] = []
    if os.path.exists(vessel_path):
        try:
            with open(vessel_path, "r", encoding="utf-8") as f:
                node_sequences = [
                    line.strip() for line in f if line.strip()
                ]
        except OSError:
            pass
    # 补齐
    if len(node_sequences) < n_segments:
        node_sequences.extend([""] * (n_segments - len(node_sequences)))
    node_sequences = node_sequences[:n_segments]

    return {
        "radii": radii,
        "path_lengths": path_lengths,
        "tortuosities": tortuosities,
        "node_sequences": node_sequences,
    }


# ═══════════════════════════════════════════════════════════════════════
# 微血管段过滤与统计计算
# ═══════════════════════════════════════════════════════════════════════

def compute_micro_stats(
    run_dir: str,
    volume: float,
) -> Optional[Dict[str, Any]]:
    """从 per-run 段数据中计算微血管（直径 < 10 μm）统计。

    参数：
        run_dir: run_* 目录路径。
        volume: 组织体积 mm^3。

    返回：
        dict 含微血管的 4 项指标 + 段数 + 过滤后的段明细，
        若无段数据或微血管段数为 0 则返回 None。
    """
    seg_data = parse_segment_data(run_dir)
    if seg_data is None:
        return None

    radii = seg_data["radii"]
    path_lengths = seg_data["path_lengths"]
    tortuosities = seg_data["tortuosities"]
    node_sequences = seg_data["node_sequences"]

    # 过滤：半径 < 2.5 μm 且节点数 >= 3
    micro_indices: List[int] = []
    for i, r in enumerate(radii):
        n_nodes = len(node_sequences[i].split()) if node_sequences[i] else 0
        if r < MICRO_RADIUS_THRESHOLD and n_nodes >= MICRO_MIN_NODES:
            micro_indices.append(i)

    if not micro_indices:
        return None

    # 提取微血管段数据
    micro_radii = [radii[i] for i in micro_indices]
    micro_lengths = [path_lengths[i] for i in micro_indices]
    micro_torts = [tortuosities[i] for i in micro_indices]
    micro_nodes = [node_sequences[i] for i in micro_indices]

    n_micro = len(micro_radii)

    # 平均直径 = 平均半径 × 4（与 C++ 输出口径一致）
    avg_radius = sum(micro_radii) / n_micro
    avg_diameter = avg_radius * 4.0

    # 平均长度（C++ 输出 avg_seg_length * 2，但这里的 path_length
    # 是 C++ 已计算的路径长度，直接取均值即可）
    # 注意：C++ 中 avg_seg_length 是 running average，输出时 * 2。
    # 此处直接从明细文件取均值，口径保持一致。
    avg_length = sum(micro_lengths) / n_micro

    # 平均弯曲度（过滤旧 C++ 运行中 path_direct_distance=0 产生的 inf/NaN）
    valid_torts = [t for t in micro_torts if not (math.isnan(t) or math.isinf(t))]
    n_valid_torts = len(valid_torts)
    if n_valid_torts > 0:
        avg_tortuosity = sum(valid_torts) / n_valid_torts
    else:
        avg_tortuosity = float("nan")
    n_invalid_torts = n_micro - n_valid_torts

    # 段密度 = 微血管段数 / 组织体积
    segment_density = n_micro / volume if volume > 0 else float("nan")

    return {
        "n_segments_total": len(radii),
        "n_segments_micro": n_micro,
        "avg_diameter_um": avg_diameter,
        "avg_length_um": avg_length,
        "avg_tortuosity_au": avg_tortuosity,
        "segment_density_per_mm3": segment_density,
        "volume_mm3": volume,
        "n_invalid_tortuosity": n_invalid_torts,
        # 过滤后的明细（供写入）
        "micro_radii": micro_radii,
        "micro_lengths": micro_lengths,
        "micro_tortuosities": micro_torts,
        "micro_node_sequences": micro_nodes,
    }


# ═══════════════════════════════════════════════════════════════════════
# 写入 per-run 微血管统计文件
# ═══════════════════════════════════════════════════════════════════════

def write_micro_stats(
    run_dir: str,
    volume: float,
) -> Optional[str]:
    """计算并写入 per-run 微血管统计文件。

    在 run_dir 中生成 5 个 _micro 后缀的文件，不覆盖原有文件。

    参数：
        run_dir: run_* 目录路径。
        volume: 组织体积 mm^3。

    返回：
        写入的 statistics_summary_micro.txt 路径，若无微血管段则返回 None。
    """
    stats = compute_micro_stats(run_dir, volume)
    if stats is None:
        return None

    # --- 写入过滤后的段明细文件 ---
    def _write_lines(suffix: str, data: List[Any]) -> str:
        path = os.path.join(run_dir, f"generate_vessel_{suffix}_micro.txt")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"{item}\n")
        return path

    _write_lines("radius", stats["micro_radii"])
    _write_lines("path_length", stats["micro_lengths"])
    _write_lines("tortuosity", stats["micro_tortuosities"])

    # 节点序列（每行已是空格分隔的节点编号字符串）
    node_path = os.path.join(run_dir, "generate_vessel_micro.txt")
    with open(node_path, "w", encoding="utf-8") as f:
        for seq in stats["micro_node_sequences"]:
            f.write(f"{seq}\n")

    # --- 写入微血管统计汇总 ---
    summary_path = os.path.join(run_dir, "statistics_summary_micro.txt")
    n_micro = stats["n_segments_micro"]
    n_total = stats["n_segments_total"]

    n_invalid = stats.get("n_invalid_tortuosity", 0)
    invalid_note = ""
    if n_invalid > 0:
        invalid_note = (
            f"\n注：{n_invalid} 个微血管段的弯曲度为 inf/NaN（旧 C++ 运行中 "
            f"path_direct_distance=0 所致），已从弯曲度均值计算中排除。"
        )

    lines = [
        "Vascular_Statistics — 单次运行微血管统计（直径 < 10 μm）",
        "",
        f"平均直径 (μm): {stats['avg_diameter_um']:.6f}",
        f"平均长度 (μm): {stats['avg_length_um']:.6f}",
        f"段密度 (seg/mm³): {stats['segment_density_per_mm3']:.6f}",
        f"平均弯曲度: {stats['avg_tortuosity_au']:.6f}",
        "",
        f"注：仅统计直径 < 10 μm 且节点数 >= 3 的微血管段。",
        f"总段数: {n_total}，微血管段数: {n_micro}",
        f"组织体积: {stats['volume_mm3']} mm³",
        invalid_note,
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary_path


# ═══════════════════════════════════════════════════════════════════════
# 批量编排
# ═══════════════════════════════════════════════════════════════════════

def run_micro_stats(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
) -> dict:
    """对所有已完成骨架化的样本（或指定样本）的所有 run 生成微血管统计。

    遍历每个样本目录下的 run_* 子目录，读取已有的段明细文件，
    计算并写入微血管统计。仅在已有 statistics_summary.txt（C++ 已运行）
    但尚无 statistics_summary_micro.txt 的 run 上执行（幂等）。

    参数：
        output_root: 输出根目录。
        sample_keys: 指定样本键列表。None = 扫描全部含 run_* 的样本。

    返回：
        {"processed": N, "skipped": N, "no_micro": N, "failed": N}
    """
    if sample_keys is None:
        sample_keys = _scan_completed_samples(output_root)

    processed = 0
    skipped = 0
    no_micro = 0
    failed = 0

    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)
        if not os.path.isdir(sample_dir):
            failed += 1
            print(f"  [FAIL] {sk} — 目录不存在")
            continue

        # 获取样本 volume
        volume = _resolve_sample_volume(sample_dir)
        if volume is None:
            failed += 1
            print(f"  [FAIL] {sk} — 无法确定组织体积")
            continue

        # 遍历 run_* 子目录
        run_dirs = sorted([
            d for d in os.listdir(sample_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(sample_dir, d))
        ])

        for run_name in run_dirs:
            run_dir = os.path.join(sample_dir, run_name)

            # 跳过无 C++ 统计输出的 run
            if not os.path.exists(os.path.join(run_dir, "statistics_summary.txt")):
                continue

            # 幂等：已有 micro 统计则跳过
            micro_summary = os.path.join(run_dir, "statistics_summary_micro.txt")
            if os.path.exists(micro_summary):
                skipped += 1
                continue

            try:
                result = write_micro_stats(run_dir, volume)
                if result:
                    processed += 1
                    print(f"  [{processed}] {sk}/{run_name}")
                else:
                    no_micro += 1
                    print(f"  [NO_MICRO] {sk}/{run_name} — 无微血管段")
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {sk}/{run_name}: {e}")

    return {
        "processed": processed,
        "skipped": skipped,
        "no_micro": no_micro,
        "failed": failed,
    }


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _resolve_sample_volume(sample_dir: str) -> Optional[float]:
    """从样本目录的 sample_metadata.json 读取组织体积。"""
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


def _scan_completed_samples(output_root: str) -> List[str]:
    """扫描 output_root 下所有含 run_* 子目录的样本，返回 sample_key 列表。"""
    sample_keys: List[str] = []
    if not os.path.isdir(output_root):
        return sample_keys

    for group in sorted(os.listdir(output_root)):
        group_dir = os.path.join(output_root, group)
        if not os.path.isdir(group_dir):
            continue
        for date_dir in sorted(os.listdir(group_dir)):
            date_path = os.path.join(group_dir, date_dir)
            if not os.path.isdir(date_path):
                continue
            for sample in sorted(os.listdir(date_path)):
                sample_path = os.path.join(date_path, sample)
                if not os.path.isdir(sample_path):
                    continue
                has_runs = any(
                    d.startswith("run_")
                    for d in os.listdir(sample_path)
                    if os.path.isdir(os.path.join(sample_path, d))
                )
                if has_runs:
                    rel = os.path.relpath(sample_path, output_root).replace("\\", "/")
                    sample_keys.append(rel)

    return sample_keys


# ═══════════════════════════════════════════════════════════════════════
# 跨样本微血管汇总
# ═══════════════════════════════════════════════════════════════════════

def generate_cross_sample_summary(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """生成跨样本微血管统计汇总文件。

    扫描所有（或指定）样本目录下的 run_*/statistics_summary_micro.txt，
    聚合每个样本的全部 run 的微血管指标，
    写入 output_root 下的 microvascular_cross_sample_summary.txt。

    参数：
        output_root: 输出根目录。
        sample_keys: 指定样本键列表。None = 扫描全部。

    返回：
        写入的文件路径，若无有效样本则返回 None。
    """
    # 延迟导入，避免循环依赖
    from vascular_statistics.aggregate_stats import aggregate_sample_stats

    if sample_keys is None:
        sample_keys = _scan_completed_samples(output_root)

    # 收集每个样本的微血管聚合数据
    rows: List[Dict[str, Any]] = []

    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)

        # 使用 aggregate_sample_stats 获取结构化聚合结果
        agg = aggregate_sample_stats(sample_dir, suffix="_micro")
        if agg is None:
            continue

        n_runs = agg.get("n_runs", 0)
        aggregates = agg.get("aggregates", {})

        # 提取 4 项指标的均值
        def _get_mean(key: str) -> Optional[float]:
            entry = aggregates.get(key)
            if entry is None:
                return None
            mean_v = entry.get("mean")
            if mean_v is None:
                return None
            if mean_v != mean_v:  # NaN
                return None
            return float(mean_v)

        avg_diameter = _get_mean("avg_diameter_um")
        avg_length = _get_mean("avg_length_um")
        seg_density = _get_mean("segment_density_per_mm3")
        avg_tortuosity = _get_mean("avg_tortuosity_au")

        # 至少需要直径数据
        if avg_diameter is None:
            continue

        rows.append({
            "sample_key": sk,
            "group": agg.get("group", "?"),
            "batch_id": agg.get("batch_id", "?"),
            "daypoint": agg.get("daypoint", "?"),
            "n_runs": n_runs,
            "avg_diameter_um": avg_diameter,
            "avg_length_um": avg_length or 0.0,
            "segment_density_per_mm3": seg_density or 0.0,
            "avg_tortuosity_au": avg_tortuosity or 0.0,
        })

    if not rows:
        return None

    # --- 格式化输出 ---
    out_path = os.path.join(output_root, "microvascular_cross_sample_summary.txt")
    sep = "=" * 100

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("  Vascular_Statistics — 跨样本微血管统计汇总（直径 < 10 μm）\n")
        f.write(f"{sep}\n")
        f.write(f"  样本数: {len(rows)}\n")
        f.write(f"  数据源: {output_root}\n")
        f.write(f"  阈值: 直径 < 10 μm（半径 < 2.5 μm），节点数 >= 3\n")
        f.write(f"{sep}\n\n")

        # 表头
        header = (
            f"  {'样本':<50s} | {'组别':>10s} | {'批次':>6s} | {'时间点':>6s} "
            f"| {'run数':>5s} | {'直径(μm)':>10s} | {'长度(μm)':>10s} "
            f"| {'段密度':>10s} | {'弯曲度':>8s}"
        )
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")

        for row in rows:
            # 截断过长的 sample_key
            key_display = row["sample_key"]
            if len(key_display) > 49:
                key_display = "..." + key_display[-(49 - 3):]

            f.write(
                f"  {key_display:<50s} | {row['group']:>10s} "
                f"| {row['batch_id']:>6s} | {row['daypoint']:>6s} "
                f"| {row['n_runs']:>5} "
                f"| {row['avg_diameter_um']:>10.4f} "
                f"| {row['avg_length_um']:>10.4f} "
                f"| {row['segment_density_per_mm3']:>10.4f} "
                f"| {row['avg_tortuosity_au']:>8.5f}\n"
            )

        f.write("\n")
        f.write(f"{sep}\n")
        f.write(
            "方法: 各样本的 statistics_summary_micro.txt（多次骨架化运行的聚合值）"
            "汇总为单一视图。\n"
        )
        f.write(
            "指标: 平均直径(μm) / 平均长度(μm) / 段密度(seg/mm³) / "
            "平均弯曲度(a.u.)\n"
        )
        f.write(f"{sep}\n")

    return out_path
