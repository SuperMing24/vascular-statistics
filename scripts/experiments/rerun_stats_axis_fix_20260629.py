#!/usr/bin/env python3
"""
一次性补救：修正 [D,H,W] 轴序 bug 导致的长度/弯曲度 x↔z spacing 错配，原地重跑 stats。

═══ 背景 ═══
pipeline 从 seg .tif（imread → [D,H,W]=[z,y,x]）骨架化时，VascGraph 直接以**数组轴索引**
赋节点 pos，故 .pajek 里 pos=[z,y,x]。但旧版 bridge.py 把 pos[0]→x、pos[2]→z 直接写入
vertices，下游 C++ 据此把深度 z 配成 sx、宽度 x 配成 sz（x↔z spacing 错配）。
各向异性 spacing（默认 z=2.0 vs x=1.11）下，**长度/弯曲度有偏**；
**半径/直径/段密度不受影响**（半径是逐体素标量，与坐标轴标注解耦）。

本脚本**不重新骨架化**——骨架几何（拓扑+位置+半径）完好。只需：
  1. 用修好的 bridge（axis_order="zyx"）从既有 skeleton.pajek 重新生成 vertices；
  2. 重跑 C++ vessel_stats（spacing 正常序 [sx,sy,sz]）→ 覆盖 d10+um 主统计与明细；
  3. 删除过期 d0-10um 子范围文件（由明细派生，须重算）。
之后在 output_root 级别重跑 diameter-stats + aggregate（见脚本尾部打印的命令）。

═══ 前置条件（集群）═══
  - 已 git pull 含修复后的 bridge.py（含 axis_order 参数）；
  - 已重新 `pip install -e`（或确保 PYTHONPATH 指向本 repo 的 python/）；
  - C++ vessel_stats 已编译（build/vessel_stats）。**无需重编 C++**（bug 在 Python 侧）。

═══ 用法 ═══
  # 预览将处理哪些 run（不写文件）
  python scripts/experiments/rerun_stats_axis_fix_20260629.py \
      --output-root /share/home/sukm/experiments/vs_vascstats --dry-run

  # 实际执行（仅各向异性/物理半径口径的 run；legacy 各向同性不受 bug 影响，自动跳过）
  python scripts/experiments/rerun_stats_axis_fix_20260629.py \
      --output-root /share/home/sukm/experiments/vs_vascstats

  # 限定单个样本
  python scripts/experiments/rerun_stats_axis_fix_20260629.py \
      --output-root ... --sample-key "xiaoqian/.../angiogram_crop_..."
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# 让脚本可独立运行（无需先 pip install）：把 repo 的 python/ 注入 sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from vascular_statistics.bridge import pajek_to_cpp_input  # noqa: E402


# 受 bug 影响、需补救的轴序（pipeline 的 .tif 入口）
AXIS_ORDER = "zyx"
# 过期的 d0-10um 子范围产物（由 C++ 明细派生，重跑主统计后必须删除以触发重算）
SUBRANGE_FILES = [
    "statistics_summary_d0-10um.txt",
    "generate_vessel_d0-10um.txt",
    "generate_vessel_radius_d0-10um.txt",
    "generate_vessel_path_length_d0-10um.txt",
    "generate_vessel_tortuosity_d0-10um.txt",
]


def find_exe(explicit: str | None) -> str:
    """定位 C++ vessel_stats 可执行文件。"""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"指定的 --exe 不存在: {explicit}")
        return explicit
    for c in ("build/vessel_stats", "build/vessel_stats.exe",
              "build/Release/vessel_stats.exe"):
        p = os.path.join(_REPO_ROOT, c)
        if os.path.exists(p):
            return p
    sys.exit("找不到 C++ vessel_stats，请用 --exe 指定（集群通常为 build/vessel_stats）。")


def resolve_spacing(run_dir: str, meta: dict) -> list[float] | None:
    """有效 spacing [sx,sy,sz]：优先本运行旁车（含网格校正），回退 run_meta。"""
    sidecar = os.path.join(run_dir, "skeleton_voxel_spacing_um.txt")
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            parts = [p.strip() for p in f.read().strip().split(",")]
        if len(parts) == 3:
            return [float(v) for v in parts]
    for key in ("stats_spacing_um", "effective_spacing_um"):
        sp = meta.get(key)
        if sp and len(sp) == 3:
            return [float(v) for v in sp]
    return None


def reprocess_run(run_dir: str, exe: str, dry: bool) -> str:
    """重跑单个 run 的 stats。返回状态串：done / skip-legacy / skip-no-pajek / skip-no-meta / skip-no-spacing / skip-no-volume。"""
    pajek = os.path.join(run_dir, "skeleton.pajek")
    if not os.path.exists(pajek):
        return "skip-no-pajek"

    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return "skip-no-meta"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    unit_mode = meta.get("stats_unit_mode", "legacy")
    radius_mode = meta.get("radius_mode", "legacy")
    # legacy 各向同性（ex=ey=ez=1）下轴序无关，长度本就正确 → 无需补救
    if unit_mode != "anisotropic" and radius_mode != "physical":
        return "skip-legacy"

    volume = meta.get("volume_mm3")
    if volume is None:
        return "skip-no-volume"

    spacing = resolve_spacing(run_dir, meta)
    if spacing is None:
        return "skip-no-spacing"

    stem = meta.get("output_stem", "skeleton")
    edges = os.path.join(run_dir, f"{stem}_edges.txt")
    verts = os.path.join(run_dir, f"{stem}_vertices.txt")

    # C++ 入参：stem 相对 cwd；spacing 正常序 [sx,sy,sz]（bridge 已把 pos 重排为物理 xyz）；
    # 物理半径口径追加第 7 参 "1"（r_scale=1.0，半径已是 μm）。
    cmd = [exe, f"{stem}_edges", f"{stem}_vertices", str(volume),
           str(spacing[0]), str(spacing[1]), str(spacing[2])]
    if radius_mode == "physical":
        cmd.append("1")

    if dry:
        print(f"    [dry] convert axis_order={AXIS_ORDER} → {os.path.basename(verts)}")
        print(f"    [dry] {' '.join(cmd)}  (cwd={run_dir})")
        print(f"    [dry] 删除过期 d0-10um 子范围文件")
        return "done"

    # 1) 用修好的 bridge 重新生成 edges/vertices（pos 重排 zyx→物理 xyz）
    pajek_to_cpp_input(pajek, edges, verts, axis_order=AXIS_ORDER)

    # 2) 重跑 C++（cwd=run_dir：exe 在 CWD 读 stem.txt、写 statistics_summary_d10+um.txt 等）
    r = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [ERR] C++ 失败 rc={r.returncode}: {r.stderr.strip()[:300]}")
        return "skip-cpp-fail"

    # 3) 删除过期 d0-10um（由明细派生，须由 diameter-stats 重算）
    for fn in SUBRANGE_FILES:
        fp = os.path.join(run_dir, fn)
        if os.path.exists(fp):
            os.remove(fp)

    # 4) 在 run_meta 留补救痕迹（不重写其余字段，保留 slurm provenance）
    meta.setdefault("axis_fix_history", []).append({
        "fix": f"axis_order {AXIS_ORDER}->xyz applied at vertices stage; stats re-run",
        "spacing_um": spacing,
        "applied_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "半径/直径/段密度不变；长度/弯曲度修正为正确各向异性度量。",
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return "done"


def iter_run_dirs(output_root: str, sample_key: str | None):
    """产出所有 run_* 目录路径。sample_key 给定则限定该样本。"""
    if sample_key:
        sample_dir = os.path.join(output_root, sample_key)
        roots = [sample_dir] if os.path.isdir(sample_dir) else []
    else:
        # 递归找所有含 skeleton.pajek 的 run 目录（不依赖固定层级深度）
        roots = None
    if roots is not None:
        for sd in roots:
            for d in sorted(os.listdir(sd)):
                rd = os.path.join(sd, d)
                if d.startswith("run_") and os.path.isdir(rd):
                    yield rd
        return
    for dirpath, dirnames, filenames in os.walk(output_root):
        if "skeleton.pajek" in filenames and os.path.basename(dirpath).startswith("run_"):
            yield dirpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", required=True, help="实验输出根目录（含样本/run 子目录）")
    ap.add_argument("--sample-key", default=None, help="仅处理指定样本（缺省=全部）")
    ap.add_argument("--exe", default=None, help="C++ vessel_stats 路径（缺省自动在 build/ 查找）")
    ap.add_argument("--dry-run", action="store_true", help="仅预览，不写文件")
    args = ap.parse_args()

    if not os.path.isdir(args.output_root):
        sys.exit(f"output-root 不存在: {args.output_root}")
    exe = find_exe(args.exe)

    print(f"[axis-fix] output_root = {args.output_root}")
    print(f"[axis-fix] exe         = {exe}")
    print(f"[axis-fix] axis_order  = {AXIS_ORDER} (pipeline .tif 入口)")
    print(f"[axis-fix] dry_run     = {args.dry_run}\n")

    counts: dict[str, int] = {}
    for run_dir in iter_run_dirs(args.output_root, args.sample_key):
        status = reprocess_run(run_dir, exe, args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        tag = "OK " if status == "done" else "-- "
        print(f"  [{tag}] {status:16s} {os.path.relpath(run_dir, args.output_root)}")

    print("\n=== 汇总 ===")
    for k in sorted(counts):
        print(f"  {k:18s}: {counts[k]}")

    if not args.dry_run and counts.get("done"):
        print("\n=== 下一步：在 output_root 级刷新聚合（子范围已删，会重算）===")
        print(f"  vascular-stats diameter-stats           --output-root {args.output_root}")
        print(f"  vascular-stats aggregate-stats          --output-root {args.output_root}")
        print(f"  vascular-stats aggregate-diameter-stats --output-root {args.output_root}")
        print(f"  vascular-stats diameter-summary         --output-root {args.output_root}")
        print("  # 注：aggregate-stats 读取各 run 的 d10+um 主统计（已重算），直接刷新。")


if __name__ == "__main__":
    main()
