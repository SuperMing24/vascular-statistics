"""
子范围统计模块 —— 从已有骨架化运行结果中按直径范围提取血管段统计。

背景：
  C++ 引擎输出全量统计 statistics_summary.txt + 全量段明细
  generate_vessel{,_radius,_path_length,_tortuosity}.txt（无任何过滤）。
  本模块从全量明细中按直径范围 + 节点数 >= 3 + P99 过滤，生成独立的
  子范围统计文件（如 0-10 um、>=10 um），不覆盖全量输出。
  （存量 run 的明细仍叫 _d10+um，内容同为全量，parse_segment_data 自动回退兼容。）

输出后缀: _d0-10um（直径 0-10 um）/ _d10+um（直径 >=10 um）。旧名 _micro 已弃用。

阈值依据（legacy 口径）：
  C++ data_io.cpp 中 if (avg_temp >= 2.5) 对应直径 >= 10 um
  （avg_temp 为段平均半径，输出时直径 = avg_temp * 4）。
  因此 0-10 um 过滤条件为 avg_radius < 2.5。
  （轨道 B 各向异性口径下阈值由 r_scale 动态计算）

输出（per-run）：
  statistics_summary_d0-10um.txt       — 子范围汇总统计
  generate_vessel_d0-10um.txt          — 子范围段节点序列
  generate_vessel_radius_d0-10um.txt   — 子范围段平均半径
  generate_vessel_path_length_d0-10um.txt — 子范围段路径长度
  generate_vessel_tortuosity_d0-10um.txt  — 子范围段弯曲度
"""

import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 子范围常量与预设
# ============================================================================

# 文件后缀与输出标签
DIAMETER_SUFFIX = "_d0-10um"        # 0-10 um 子范围文件后缀
SUFFIX_D10_PLUS = "_d10+um"          # >=10 um 子范围文件后缀（旧默认 C++ 输出同名）
DIAMETER_LABEL = "直径 0-10 um"     # 用于汇总/跨样本报告标题
DIAMETER_LO_UM = 0.0                # 直径下界 (um)
DIAMETER_HI_UM = 10.0               # 直径上界 (um)

# Legacy 口径常量（轨道 B 各向异性下由 _read_caliber 动态覆盖）
LEGACY_RADIUS_THRESHOLD = 2.5       # 半径 < 2.5 体素 (= 直径 < 10um，旧 x4)
LEGACY_MIN_NODES = 3                # 与 C++ 一致：k > 2 即至少 3 个节点


@dataclass(frozen=True)
class SubRange:
    """一个直径子范围档位 [lo_um, hi_um)：后缀 + 标题 + 直径上下界。

    全量统计由 C++ 引擎产出（无后缀）；子范围（如 0-10um、>=10um）由本模块
    从全量明细中按直径范围 + 节点数 >= 3 + P99 过滤派生。
    """
    suffix: str
    label: str
    lo_um: float
    hi_um: float


# 两个预设档位
SUBRANGE_D0_10 = SubRange(DIAMETER_SUFFIX, DIAMETER_LABEL, DIAMETER_LO_UM, DIAMETER_HI_UM)
SUBRANGE_D10_PLUS = SubRange(SUFFIX_D10_PLUS, "直径 >=10 um", 10.0, float("inf"))

# 离群段标记（与 C++ data_io.cpp 一致，轨道 A）。
MAD_K = 3.5  # Iglewicz-Hoaglin 修正 z 分数阈值（相对判据，单位无关）


def _mad_outlier_count(values: List[float], k: float = MAD_K) -> int:
    """MAD（中位数绝对偏差）离群计数：|0.6745*(x-median)/MAD| > k。
    单位无关；MAD 退化为 0（半数以上同值）或样本不足 2 时不标记任何离群。
    """
    if len(values) < 2:
        return 0
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    if mad < 1e-10:
        return 0
    return sum(1 for x in values if abs(0.6745 * (x - med) / mad) > k)


# ============================================================================
# 解析 per-run 段明细文件
# ============================================================================

def parse_segment_data(run_dir: str) -> Optional[Dict[str, Any]]:
    """读取单个 run_* 目录中的全部血管段明细数据。
    优先无后缀全量明细（新 C++ 输出），回退旧 _d10+um 名（存量 run，内容同为全量）。
    """
    def _resolve(name: str) -> str:
        full_path = os.path.join(run_dir, f"{name}.txt")
        if os.path.exists(full_path):
            return full_path
        return os.path.join(run_dir, f"{name}{SUFFIX_D10_PLUS}.txt")

    radius_path = _resolve("generate_vessel_radius")
    length_path = _resolve("generate_vessel_path_length")
    tort_path = _resolve("generate_vessel_tortuosity")
    vessel_path = _resolve("generate_vessel")

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

    def _read_floats(path: str, expected: int) -> List[float]:
        if not os.path.exists(path):
            return [0.0] * expected
        try:
            with open(path, "r", encoding="utf-8") as f:
                values = [float(line.strip()) for line in f if line.strip()]
        except (OSError, ValueError):
            return [0.0] * expected
        if len(values) < expected:
            values.extend([0.0] * (expected - len(values)))
        return values[:expected]

    path_lengths = _read_floats(length_path, n_segments)
    tortuosities = _read_floats(tort_path, n_segments)

    node_sequences: List[str] = []
    if os.path.exists(vessel_path):
        try:
            with open(vessel_path, "r", encoding="utf-8") as f:
                node_sequences = [
                    line.strip() for line in f if line.strip()
                ]
        except OSError:
            pass
    if len(node_sequences) < n_segments:
        node_sequences.extend([""] * (n_segments - len(node_sequences)))
    node_sequences = node_sequences[:n_segments]

    return {
        "radii": radii,
        "path_lengths": path_lengths,
        "tortuosities": tortuosities,
        "node_sequences": node_sequences,
    }


# ============================================================================
# 口径检测
# ============================================================================

def _read_caliber(run_dir: str) -> Tuple[str, float, float, float, float]:
    """从 run_meta.json 读取口径标记。
    返回 (mode, r_scale, length_um_factor, radius_threshold, abs_length_threshold)。
    无 run_meta.json 或未标记 -> legacy 默认。
    """
    meta_path = os.path.join(run_dir, "run_meta.json")
    mode = "legacy"
    r_scale = 2.0
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("stats_unit_mode") == "anisotropic":
                mode = "anisotropic"
                # 与 C++ data_io.cpp 口径一致地推导 r_scale（run_meta 不写 r_scale）：
                #   radius_mode=physical → 1.0（半径已是物理 μm，方案B）
                #   否则各向异性 → (sx+sy)/2（取自 stats_spacing_um）
                #   缺 spacing 时退回 meta.r_scale 或 2.0（保守）
                if meta.get("radius_mode") == "physical":
                    r_scale = 1.0
                else:
                    sp = meta.get("stats_spacing_um")
                    if sp and len(sp) >= 2:
                        r_scale = (float(sp[0]) + float(sp[1])) / 2.0
                    else:
                        r_scale = float(meta.get("r_scale", 2.0))
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            pass

    if mode == "anisotropic":
        return mode, r_scale, 1.0, 5.0 / r_scale, 1200.0
    else:
        return mode, 2.0, 2.0, LEGACY_RADIUS_THRESHOLD, 600.0


# ============================================================================
# 子范围统计计算（默认 0-10 um）
# ============================================================================

def compute_subrange_stats(
    run_dir: str,
    volume: float,
    subrange: SubRange = SUBRANGE_D0_10,
) -> Optional[Dict[str, Any]]:
    """从 per-run 全量段明细中计算指定直径子范围统计。

    参数：
        run_dir: run_* 目录路径。
        volume: 组织体积 mm^3。
        subrange: 直径档位 [lo_um, hi_um)（默认 0-10 um）。

    返回：
        dict 含子范围指标 + 段数 + 过滤后的段明细，
        若无段数据或符合条件的段数为 0 则返回 None。
    """
    mode, r_scale, length_um, _radius_thresh, abs_thresh = _read_caliber(run_dir)

    seg_data = parse_segment_data(run_dir)
    if seg_data is None:
        return None

    radii = seg_data["radii"]
    path_lengths = seg_data["path_lengths"]
    tortuosities = seg_data["tortuosities"]
    node_sequences = seg_data["node_sequences"]

    # 直径范围 [lo_um, hi_um) → 半径范围 [lo_r, hi_r)（直径 = 半径 * 2 * r_scale）。
    lo_r = subrange.lo_um / (2.0 * r_scale)
    hi_r = subrange.hi_um / (2.0 * r_scale)  # hi_um=inf → hi_r=inf

    # 过滤：半径落在 [lo_r, hi_r) 且节点数 >= 3
    sub_indices: List[int] = []
    for i, r in enumerate(radii):
        n_nodes = len(node_sequences[i].split()) if node_sequences[i] else 0
        if lo_r <= r < hi_r and n_nodes >= LEGACY_MIN_NODES:
            sub_indices.append(i)

    if not sub_indices:
        return None

    sub_radii = [radii[i] for i in sub_indices]
    sub_lengths = [path_lengths[i] for i in sub_indices]
    sub_torts = [tortuosities[i] for i in sub_indices]
    n_all = len(sub_radii)

    # --- P99 百分位排除（实际剔除，与 C++ 一致）---
    p99_threshold = 0.0
    n_excluded = 0
    if n_all > 0:
        sorted_lens = sorted(sub_lengths)
        idx99 = int(99.0 / 100.0 * len(sorted_lens))
        p99_threshold = sorted_lens[idx99] if idx99 < len(sorted_lens) else sorted_lens[-1]
        n_excluded = len(sorted_lens) - idx99

    # 排除后子集
    keep = [i for i, L in enumerate(sub_lengths) if L <= p99_threshold]
    radii_f = [sub_radii[i] for i in keep]
    lengths_f = [sub_lengths[i] for i in keep]
    torts_f = [sub_torts[i] for i in keep]
    n_eff = len(radii_f)

    # 排除前均值（参考）
    avg_len_all = sum(sub_lengths) / n_all * length_um if n_all > 0 else 0.0

    # 直径 = 半径 x 2 x r_scale
    avg_radius = sum(radii_f) / n_eff if n_eff > 0 else 0.0
    avg_diameter = avg_radius * 2.0 * r_scale
    median_diameter = statistics.median(radii_f) * 2.0 * r_scale if n_eff > 0 else 0.0

    # 长度
    avg_length = sum(lengths_f) / n_eff * length_um if n_eff > 0 else 0.0
    median_length = statistics.median(lengths_f) * length_um if n_eff > 0 else 0.0

    # 弯曲度
    valid_torts = [t for t in torts_f if not (math.isnan(t) or math.isinf(t))]
    n_valid_torts = len(valid_torts)
    avg_tortuosity = sum(valid_torts) / n_valid_torts if n_valid_torts > 0 else float("nan")
    median_tortuosity = statistics.median(valid_torts) if n_valid_torts > 0 else float("nan")
    n_invalid_torts = n_eff - n_valid_torts

    segment_density = n_eff / volume if volume > 0 else float("nan")

    # 离群诊断（信息项，基于原始子范围段长度）
    mad_outliers = _mad_outlier_count(sub_lengths, MAD_K)
    abs_outliers = sum(1 for x in sub_lengths if x > abs_thresh)
    mad_ratio = 100.0 * mad_outliers / n_all if n_all > 0 else 0.0
    abs_ratio = 100.0 * abs_outliers / n_all if n_all > 0 else 0.0
    excluded_ratio = 100.0 * n_excluded / n_all if n_all > 0 else 0.0

    return {
        "n_segments_total": len(radii),
        "n_segments_subrange": n_eff,
        "n_segments_subrange_all": n_all,
        "avg_diameter_um": avg_diameter,
        "median_diameter_um": median_diameter,
        "avg_length_um": avg_length,
        "median_length_um": median_length,
        "avg_tortuosity_au": avg_tortuosity,
        "median_tortuosity_au": median_tortuosity,
        "segment_density_per_mm3": segment_density,
        "volume_mm3": volume,
        "n_invalid_tortuosity": n_invalid_torts,
        "mad_outliers": mad_outliers,
        "abs_outliers": abs_outliers,
        "mad_ratio": mad_ratio,
        "abs_ratio": abs_ratio,
        "stats_unit_mode": mode,
        "r_scale": r_scale,
        "p99_threshold": p99_threshold,
        "n_excluded": n_excluded,
        "excluded_ratio": excluded_ratio,
        "avg_length_before_exclusion": avg_len_all,
        # 明细（供写入）
        "sub_radii": sub_radii,
        "sub_lengths": sub_lengths,
        "sub_torts": sub_torts,
        "node_sequences_sub": [node_sequences[i] for i in sub_indices],
    }


# ============================================================================
# 写入 per-run 子范围统计文件
# ============================================================================

def write_subrange_stats(
    run_dir: str,
    volume: float,
    subrange: SubRange = SUBRANGE_D0_10,
) -> Optional[str]:
    """计算并写入 per-run 子范围统计文件。

    在 run_dir 中生成 5 个 {subrange.suffix} 后缀的文件，不覆盖全量明细。

    返回：写入的 statistics_summary{suffix}.txt 路径，无符合条件的段则返回 None。
    """
    stats = compute_subrange_stats(run_dir, volume, subrange)
    if stats is None:
        return None

    suffix = subrange.suffix
    label = subrange.label

    # --- 写入过滤后的段明细 ---
    def _write_lines(key: str, data: List[Any]) -> str:
        path = os.path.join(run_dir, f"generate_vessel_{key}{suffix}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"{item}\n")
        return path

    _write_lines("radius", stats["sub_radii"])
    _write_lines("path_length", stats["sub_lengths"])
    _write_lines("tortuosity", stats["sub_torts"])

    node_path = os.path.join(run_dir, f"generate_vessel{suffix}.txt")
    with open(node_path, "w", encoding="utf-8") as f:
        for seq in stats["node_sequences_sub"]:
            f.write(f"{seq}\n")

    # --- 写入统计汇总 ---
    summary_path = os.path.join(run_dir, f"statistics_summary{suffix}.txt")
    n_sub = stats["n_segments_subrange"]
    n_total = stats["n_segments_total"]

    n_invalid = stats.get("n_invalid_tortuosity", 0)
    invalid_note = ""
    if n_invalid > 0:
        invalid_note = (
            f"\n注：{n_invalid} 个子范围段的弯曲度为 inf/NaN（旧 C++ 运行中 "
            f"path_direct_distance=0 所致），已从弯曲度均值计算中排除。"
        )

    warn_note = ""
    if stats["abs_ratio"] > 1.0:
        warn_note = "!! 注意：骨架可能存在过度连接（绝对超长段比例偏高）"

    mode = stats.get("stats_unit_mode", "legacy")
    unit_note = ("（各向异性 spacing 物理单位）" if mode == "anisotropic"
                 else "（legacy 各向同性 2um/体素）")
    abs_unit = "um" if mode == "anisotropic" else "体素"
    abs_thresh = 1200.0 if mode == "anisotropic" else 600.0

    lines = [
        f"Vascular_Statistics -- 单次运行子范围统计（{label}）",
        "",
        f"平均直径 (um): {stats['avg_diameter_um']:.6f}",
        f"中位直径 (um): {stats['median_diameter_um']:.6f}",
        f"平均长度 (um): {stats['avg_length_um']:.6f}",
        f"中位长度 (um): {stats['median_length_um']:.6f}",
        f"段密度 (seg/mm^3): {stats['segment_density_per_mm3']:.6f}",
        f"平均弯曲度: {stats['avg_tortuosity_au']:.6f}",
        f"中位弯曲度: {stats['median_tortuosity_au']:.6f}",
        f"有效段数: {n_sub}",
        "",
        f"--- 百分位排除（P99 = {stats['p99_threshold']:.2f} {abs_unit}）---",
        f"排除段数: {stats['n_excluded']} ({stats['excluded_ratio']:.4f}%)",
        f"排除前有效段数: {stats['n_segments_subrange_all']}",
        f"排除前平均长度: {stats['avg_length_before_exclusion']:.4f} {abs_unit}",
        "",
        f"--- 离群诊断（信息项，已由 P99 排除处理）---",
        f"MAD 离群段 (k={MAD_K}): {stats['mad_outliers']} ({stats['mad_ratio']:.4f}%)",
        f"超绝对阈值段 (>{abs_thresh:.0f} {abs_unit}): "
        f"{stats['abs_outliers']} ({stats['abs_ratio']:.4f}%)",
        warn_note,
        "",
        f"注：统计值基于 P99 排除后的 {n_sub} 个子范围段（{label}，"
        f"节点数 >= 3）；"
        f"P99 排除顶 1% 极端长尾，排除的段写入明细文件但不参与统计；"
        f"单位口径 {unit_note}。",
        f"总段数: {n_total}，子范围段数（排除前）: {stats['n_segments_subrange_all']}",
        f"组织体积: {stats['volume_mm3']} mm^3",
        invalid_note,
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary_path


# ============================================================================
# 批量编排
# ============================================================================

def run_subrange_stats(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
    subrange: SubRange = SUBRANGE_D0_10,
) -> dict:
    """对所有已完成骨架化的样本生成指定档位的子范围统计。

    幂等：已有 statistics_summary{subrange.suffix}.txt 的 run 将跳过。
    """
    if sample_keys is None:
        sample_keys = _scan_completed_samples(output_root)

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

        run_dirs = sorted([
            d for d in os.listdir(sample_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(sample_dir, d))
        ])

        for run_name in run_dirs:
            run_dir = os.path.join(sample_dir, run_name)

            # 检查 C++ 主统计是否已运行（全量无后缀，向后兼容旧 _d10+um 名）
            main_stats = os.path.join(run_dir, "statistics_summary.txt")
            if not os.path.exists(main_stats):
                main_stats = os.path.join(run_dir, f"statistics_summary{SUFFIX_D10_PLUS}.txt")
            if not os.path.exists(main_stats):
                continue

            sub_summary = os.path.join(run_dir, f"statistics_summary{subrange.suffix}.txt")
            if os.path.exists(sub_summary):
                skipped += 1
                continue

            try:
                result = write_subrange_stats(run_dir, volume, subrange)
                if result:
                    processed += 1
                    print(f"  [{processed}] {sk}/{run_name}")
                else:
                    no_result += 1
                    print(f"  [NO_RESULT] {sk}/{run_name} -- 无符合条件的段")
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {sk}/{run_name}: {e}")

    return {
        "processed": processed,
        "skipped": skipped,
        "no_result": no_result,
        "failed": failed,
    }


# ============================================================================
# 辅助函数
# ============================================================================

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


def _scan_completed_samples(output_root: str) -> List[str]:
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


# ============================================================================
# 跨样本次范围汇总
# ============================================================================

def generate_cross_sample_summary(
    output_root: str,
    sample_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """生成跨样本次范围统计汇总。

    扫描所有样本的 statistics_summary_d0-10um.txt，
    写入 output_root 下的 cross_sample_summary_d0-10um.txt。
    """
    from vascular_statistics.aggregate_stats import aggregate_sample_stats

    if sample_keys is None:
        sample_keys = _scan_completed_samples(output_root)

    rows: List[Dict[str, Any]] = []

    for sk in sorted(sample_keys):
        sample_dir = os.path.join(output_root, sk)
        agg = aggregate_sample_stats(sample_dir, suffix=DIAMETER_SUFFIX)
        if agg is None:
            continue

        n_runs = agg.get("n_runs", 0)
        aggregates = agg.get("aggregates", {})

        def _get_mean(key: str) -> Optional[float]:
            entry = aggregates.get(key)
            if entry is None:
                return None
            mean_v = entry.get("mean")
            if mean_v is None or mean_v != mean_v:  # NaN
                return None
            return float(mean_v)

        avg_diameter = _get_mean("avg_diameter_um")
        avg_length = _get_mean("avg_length_um")
        seg_density = _get_mean("segment_density_per_mm3")
        avg_tortuosity = _get_mean("avg_tortuosity_au")

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

    out_path = os.path.join(output_root, f"cross_sample_summary{DIAMETER_SUFFIX}.txt")
    sep = "=" * 100

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"  Vascular_Statistics -- 跨样本次范围统计汇总（{DIAMETER_LABEL}）\n")
        f.write(f"{sep}\n")
        f.write(f"  样本数: {len(rows)}\n")
        f.write(f"  数据源: {output_root}\n")
        f.write(f"  阈值: {DIAMETER_LABEL}，节点数 >= 3\n")
        f.write(f"{sep}\n\n")

        header = (
            f"  {'样本':<50s} | {'组别':>10s} | {'批次':>6s} | {'时间点':>6s} "
            f"| {'run数':>5s} | {'直径(um)':>10s} | {'长度(um)':>10s} "
            f"| {'段密度':>10s} | {'弯曲度':>8s}"
        )
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")

        for row in rows:
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
            f"方法: 各样本的 statistics_summary{DIAMETER_SUFFIX}.txt"
            "（多次骨架化运行的聚合值）汇总为单一视图。\n"
        )
        f.write(
            "指标: 平均直径(um) / 平均长度(um) / 段密度(seg/mm^3) / "
            "平均弯曲度(a.u.)\n"
        )
        f.write(f"{sep}\n")

    return out_path
