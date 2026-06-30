#!/usr/bin/env python
"""存量结果迁移：旧默认（≥10μm）→ 新默认（全量）+ 按需子范围。

背景
----
旧版 C++ 引擎只输出 ≥10μm 子范围汇总 statistics_summary_d10+um.txt，
而 4 个 generate_vessel*_d10+um.txt 明细文件**本就是全量数据**（写入处无任何
直径/节点数过滤）。因此存量结果不必重跑骨架化，可直接从明细重算全量。

本脚本对每个存量 run_* 目录：
  1. 从 generate_vessel*_d10+um.txt（全量明细）重算全量汇总 → statistics_summary.txt
  2. 把 4 个 _d10+um 明细正名为无后缀全量名 generate_vessel*.txt
  3. 用 Python 重新派生两档子范围（0-10um / ≥10um）的统计与明细
     （覆盖旧 statistics_summary_d10+um.txt，使之成为「真·≥10μm 子范围」）

幂等：已有 statistics_summary.txt 的 run 视为已迁移，跳过。
用法：
  python scripts/data/migrate_full_stats.py --output-root <exp_root> [--dry-run]
  python scripts/data/migrate_full_stats.py --output-root <exp_root> --sample-key "BCAS_1st/..."
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from vascular_statistics.subrange_stats import (  # noqa: E402
    SUBRANGE_D0_10,
    SUBRANGE_D10_PLUS,
    SUFFIX_D10_PLUS,
    _read_caliber,
    _resolve_sample_volume,
    _scan_completed_samples,
    parse_segment_data,
    write_subrange_stats,
)

# 旧 _d10+um 明细（实为全量）→ 新无后缀全量名
_DETAIL_STEMS = [
    "generate_vessel",
    "generate_vessel_radius",
    "generate_vessel_path_length",
    "generate_vessel_tortuosity",
]


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_full_summary_text(run_dir: str, volume: float) -> str | None:
    """从全量明细重算全量汇总文本（与 C++ statistics_summary.txt 同口径/标签）。

    无过滤：纳入明细中每一段。单位换算（r_scale / length_um_factor）取自 run_meta.json。
    """
    mode, r_scale, length_um, _radius_thresh, _abs_thresh = _read_caliber(run_dir)
    seg = parse_segment_data(run_dir)
    if seg is None:
        return None

    radii = seg["radii"]
    lengths = seg["path_lengths"]
    torts = [t for t in seg["tortuosities"] if not (math.isnan(t) or math.isinf(t))]
    n = len(radii)
    if n == 0:
        return None

    mean_d = _mean(radii) * 2.0 * r_scale
    median_d = _median(radii) * 2.0 * r_scale
    mean_l = _mean(lengths) * length_um
    median_l = _median(lengths) * length_um
    mean_t = _mean(torts)
    median_t = _median(torts)
    density = n / volume if volume > 0 else float("nan")

    unit_note = ("（各向异性 spacing 物理单位）" if mode == "anisotropic"
                 else "（legacy 各向同性 2μm/体素）")

    lines = [
        "Vascular_Statistics — 单次运行统计（全量，未过滤；由 migrate_full_stats 从明细重算）",
        f"平均直径 (μm): {mean_d:.6g}",
        f"中位直径 (μm): {median_d:.6g}",
        f"平均长度 (μm): {mean_l:.6g}",
        f"中位长度 (μm): {median_l:.6g}",
        f"段密度 (seg/mm³): {density:.6g}",
        f"平均弯曲度: {mean_t:.6g}",
        f"中位弯曲度: {median_t:.6g}",
        f"总段数: {n}",
        "",
        f"注：全量统计，由 migrate_full_stats 从存量明细重算，纳入所有段（无直径/节点数/P99 "
        f"过滤）；单位口径{unit_note}。",
    ]
    return "\n".join(lines) + "\n"


def migrate_run(run_dir: str, volume: float, dry_run: bool) -> str:
    """迁移单个 run。返回状态字符串。"""
    full_summary = os.path.join(run_dir, "statistics_summary.txt")
    if os.path.exists(full_summary):
        return "skip(已迁移)"

    # 旧全量明细以 _d10+um 命名；缺失则该 run 无可迁移产物。
    old_radius = os.path.join(run_dir, f"generate_vessel_radius{SUFFIX_D10_PLUS}.txt")
    if not os.path.exists(old_radius):
        return "skip(无明细)"

    # 1) 重算全量汇总（趁明细仍为 _d10+um，parse_segment_data 自动回退读取）
    text = compute_full_summary_text(run_dir, volume)
    if text is None:
        return "fail(明细为空)"

    if dry_run:
        n = text.split("总段数: ")[-1].split("\n")[0]
        return f"DRY 全量段数={n}"

    # 2) 明细 _d10+um → 无后缀全量名
    for stem in _DETAIL_STEMS:
        src = os.path.join(run_dir, f"{stem}{SUFFIX_D10_PLUS}.txt")
        dst = os.path.join(run_dir, f"{stem}.txt")
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)

    # 3) 写全量汇总
    with open(full_summary, "w", encoding="utf-8") as f:
        f.write(text)

    # 4) 重新派生两档子范围（覆盖旧 _d10+um 汇总，使其成为真·≥10μm 子范围）
    for sub in (SUBRANGE_D0_10, SUBRANGE_D10_PLUS):
        sub_summary = os.path.join(run_dir, f"statistics_summary{sub.suffix}.txt")
        if os.path.exists(sub_summary):
            os.remove(sub_summary)  # 强制重算（旧 _d10+um 汇总口径已变）
        write_subrange_stats(run_dir, volume, sub)

    return "migrated"


def main() -> None:
    ap = argparse.ArgumentParser(description="存量结果迁移：≥10μm 默认 → 全量默认")
    ap.add_argument("--output-root", required=True, help="实验输出根目录")
    ap.add_argument("--sample-key", default=None, help="仅迁移指定样本（缺省全部）")
    ap.add_argument("--dry-run", action="store_true", help="仅预览，不写文件")
    args = ap.parse_args()

    sample_keys = ([args.sample_key] if args.sample_key
                   else _scan_completed_samples(args.output_root))

    counts: dict[str, int] = {}
    for sk in sorted(sample_keys):
        sample_dir = os.path.join(args.output_root, sk)
        volume = _resolve_sample_volume(sample_dir)
        if volume is None:
            print(f"  [FAIL] {sk} — 无法确定组织体积")
            counts["fail"] = counts.get("fail", 0) + 1
            continue
        for run_name in sorted(os.listdir(sample_dir)):
            run_dir = os.path.join(sample_dir, run_name)
            if not run_name.startswith("run_") or not os.path.isdir(run_dir):
                continue
            status = migrate_run(run_dir, volume, args.dry_run)
            key = status.split("(")[0].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            print(f"  [{status}] {sk}/{run_name}")

    print("\n汇总:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
