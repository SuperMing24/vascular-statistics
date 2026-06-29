#!/usr/bin/env python3
"""
轴序补救：由旧实验目录构建「坐标轴序已规范」的新实验目录，并重跑 stats。

═══ 背景 ═══
2026-06-29 之前 pipeline 从 seg .tif（imread→[D,H,W]=[z,y,x]）骨架化时，VascGraph 以
数组轴索引赋节点 pos，故旧 skeleton.pajek 的 pos=[z,y,x]（深度 z 在 slot0）。这使：
  - GUI（ReadPajek 逐列读 x,y,z）显示轴向错乱（与 7 例金标准格式不同）；
  - C++ stats 各向异性下把深度 z 配 sx、宽度 x 配 sz（x↔z spacing 错配）→ 长度/弯曲度偏。
半径/直径/段密度不受影响（半径是逐体素标量，与坐标轴解耦）。

修复（不重新骨架化）：把 pajek 的 pos 列重排为规范物理 [x,y,z]，则 GUI + stats + 金标准
对比 全部自然正确。骨架几何（拓扑+半径）完好，只重排坐标 + 重跑廉价的 stats。

═══ 策略（保留旧目录，建新目录）═══
旧实验目录（如 vs_xiaoqian_croppedz_20260628）作为「有问题但可追溯」的记录**保留不动**；
本脚本复制为新目录（如 vs_xiaoqian_croppedz_20260628_axisfix），在副本上：
  1. 文本级交换每个 skeleton.pajek 的 pos 列（pos_axes→[x,y,z]）；
  2. 删除副本里的过期 stats（edges/vertices/statistics_summary_*/generate_vessel_*）；
  3. 重跑 C++ stats（bridge 默认 xyz，因 pajek 已规范）；
  4. run_meta 记录补救痕迹。
最后在新 root 级刷新 diameter-stats + aggregate。

═══ 前置条件（集群）═══
  - 已 git pull 含本脚本 + bridge.canonicalize_graph_pos；已 pip install -e（或 PYTHONPATH）；
  - C++ build/vessel_stats 就绪。**无需重编 C++**。

═══ 用法 ═══
  conda activate vascstats
  # 预览（不写文件）
  python scripts/experiments/build_axisfix_experiment_20260629.py \
      --src-root /share/home/sukm/experiments/vs_xiaoqian_croppedz_20260628 \
      --dst-root /share/home/sukm/experiments/vs_xiaoqian_croppedz_20260628_axisfix \
      --dry-run

  # 实际执行
  python scripts/experiments/build_axisfix_experiment_20260629.py \
      --src-root /share/home/sukm/experiments/vs_xiaoqian_croppedz_20260628 \
      --dst-root /share/home/sukm/experiments/vs_xiaoqian_croppedz_20260628_axisfix

  # huaien 同理（src/dst 换成 huaien 对应目录）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from vascular_statistics.bridge import pajek_to_cpp_input  # noqa: E402

# 这批 149 例均产自 cli.pipeline 从 .tif（[D,H,W]）骨架化 → pos 数组轴序 = "zyx"
DEFAULT_POS_AXES = "zyx"

# 重跑前需从副本删除的过期产物（C++/convert 会重生成 d10+um；d0-10um 由 diameter-stats 重算）
STALE_FILES = [
    "skeleton_edges.txt", "skeleton_vertices.txt",
    "generate_vessel_d10+um.txt", "generate_vessel_radius_d10+um.txt",
    "generate_vessel_path_length_d10+um.txt", "generate_vessel_tortuosity_d10+um.txt",
    "statistics_summary_d10+um.txt",
    "generate_vessel_d0-10um.txt", "generate_vessel_radius_d0-10um.txt",
    "generate_vessel_path_length_d0-10um.txt", "generate_vessel_tortuosity_d0-10um.txt",
    "statistics_summary_d0-10um.txt",
]

_POS_RE = re.compile(r'(pos\s+")\[([^\]]*)\](")')


def swap_pajek_pos(pajek_path: str, pos_axes: str) -> int:
    """文本级重排 pajek 各 vertices 行的 pos 列为规范 [x,y,z]。返回改动的行数。

    只动含 `pos "[...]"` 的行（顶点行；边行无 pos），其余字节不变。
    """
    order = pos_axes.lower()
    if sorted(order) != ["x", "y", "z"]:
        raise ValueError(f"pos_axes 必须是 'xyz' 的排列: {pos_axes!r}")
    ix, iy, iz = order.index("x"), order.index("y"), order.index("z")

    def _repl(m: re.Match) -> str:
        vals = m.group(2).split()
        if len(vals) != 3:
            return m.group(0)  # 异常行不动
        new = [vals[ix], vals[iy], vals[iz]]
        return f'{m.group(1)}[{new[0]} {new[1]} {new[2]}]{m.group(3)}'

    with open(pajek_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n_changed = 0
    out = []
    for ln in lines:
        new_ln, k = _POS_RE.subn(_repl, ln)
        n_changed += k
        out.append(new_ln)
    with open(pajek_path, "w", encoding="utf-8") as f:
        f.writelines(out)
    return n_changed


def find_exe(explicit: str | None) -> str:
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"--exe 不存在: {explicit}")
        return explicit
    for c in ("build/vessel_stats", "build/vessel_stats.exe",
              "build/Release/vessel_stats.exe"):
        p = os.path.join(_REPO_ROOT, c)
        if os.path.exists(p):
            return p
    sys.exit("找不到 C++ vessel_stats，请用 --exe 指定。")


def resolve_spacing(run_dir: str, meta: dict) -> list[float] | None:
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


def resolve_volume(run_dir: str, meta: dict) -> float | None:
    """volume：run_meta.volume_mm3 优先；否则向上找 sample_metadata.json（兼容
    run_meta.json 功能（86448bb）之前产出的旧 run）。"""
    if meta.get("volume_mm3") is not None:
        return float(meta["volume_mm3"])
    d = run_dir
    for _ in range(4):
        d = os.path.dirname(d)
        smp = os.path.join(d, "sample_metadata.json")
        if os.path.exists(smp):
            with open(smp, "r", encoding="utf-8") as f:
                sm = json.load(f)
            v = sm.get("spatial", {}).get("tissue_volume_mm3")
            if v is not None:
                return float(v)
    return None


def infer_radius_mode(run_dir: str, meta: dict) -> str:
    """radius_mode：run_meta 优先；否则看 skeleton_radius_unit.txt（=um → physical）。"""
    if meta.get("radius_mode"):
        return meta["radius_mode"]
    ru = os.path.join(run_dir, "skeleton_radius_unit.txt")
    if os.path.exists(ru):
        with open(ru, "r", encoding="utf-8") as f:
            if f.read().strip().lower() == "um":
                return "physical"
    return "legacy"


def reprocess_run(run_dir: str, exe: str, pos_axes: str, dry: bool) -> str:
    pajek = os.path.join(run_dir, "skeleton.pajek")
    if not os.path.exists(pajek):
        return "skip-no-pajek"
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 幂等护栏：已修过（run_meta 含 axis_fix_history）→ 跳过，避免二次交换 pajek 把它还原回错的
    if meta.get("axis_fix_history"):
        return "skip-already-fixed"

    volume = resolve_volume(run_dir, meta)
    if volume is None:
        return "skip-no-volume"
    spacing = resolve_spacing(run_dir, meta)
    radius_mode = infer_radius_mode(run_dir, meta)
    unit_mode = meta.get("stats_unit_mode")
    # 各向异性判定：meta 标记 / 物理半径 / 有 sidecar spacing（兼容无 meta 旧 run）
    aniso = (unit_mode == "anisotropic") or (radius_mode == "physical") or \
            (unit_mode is None and spacing is not None)
    if aniso and spacing is None:
        return "skip-no-spacing"

    stem = meta.get("output_stem", "skeleton")

    if dry:
        print(f"    [dry] swap pajek pos ({pos_axes}→xyz)；删除 stale；重跑 stats "
              f"(aniso={aniso}, physical={radius_mode=='physical'}, vol={volume})")
        return "done"

    # 1) pajek pos 重排为规范 [x,y,z]（覆写副本里的 skeleton.pajek）
    n = swap_pajek_pos(pajek, pos_axes)

    # 2) 删除副本里的过期 stats
    for fn in STALE_FILES:
        fp = os.path.join(run_dir, fn)
        if os.path.exists(fp):
            os.remove(fp)

    # 3) 格式转换（默认 xyz，pajek 已规范）+ C++ stats
    pajek_to_cpp_input(pajek, os.path.join(run_dir, f"{stem}_edges.txt"),
                       os.path.join(run_dir, f"{stem}_vertices.txt"))
    cmd = [exe, f"{stem}_edges", f"{stem}_vertices", str(volume)]
    if aniso:
        cmd += [str(spacing[0]), str(spacing[1]), str(spacing[2])]
        if radius_mode == "physical":
            cmd.append("1")
    r = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [ERR] C++ rc={r.returncode}: {r.stderr.strip()[:300]}")
        return "skip-cpp-fail"

    # 4) run_meta 留痕（无 meta 的旧 run 补全基本字段，使 axis_fix_history 落到有效 run_meta）
    meta.setdefault("output_stem", stem)
    meta.setdefault("volume_mm3", volume)
    meta.setdefault("radius_mode", radius_mode)
    meta.setdefault("stats_unit_mode", "anisotropic" if aniso else "legacy")
    if spacing is not None:
        meta.setdefault("stats_spacing_um", spacing)
    meta.setdefault("axis_fix_history", []).append({
        "fix": f"pajek pos canonicalized {pos_axes}->xyz ({n} 顶点); stats re-run",
        "spacing_um": spacing,
        "applied_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "半径/直径/段密度不变；长度/弯曲度修正；pajek 现为规范 [x,y,z]（GUI 正确）。",
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return "done"


def iter_run_dirs(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "skeleton.pajek" in filenames and os.path.basename(dirpath).startswith("run_"):
            yield dirpath


def run_cli_aggregates(dst_root: str) -> None:
    """在新 root 上刷新子范围 + 聚合（diameter-stats 重算 d0-10um）。"""
    steps = [
        ["diameter-stats", "--output-root", dst_root],
        ["aggregate-stats", "--output-root", dst_root],
        ["aggregate-diameter-stats", "--output-root", dst_root],
        ["diameter-summary", "--output-root", dst_root],
    ]
    for s in steps:
        print(f"  $ vascular-stats {' '.join(s)}")
        r = subprocess.run([sys.executable, "-m", "vascular_statistics.cli"] + s,
                           capture_output=True, text=True)
        tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
        for t in tail:
            print(f"      {t}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", required=True, help="旧实验根目录（保留不动，作副本源）")
    ap.add_argument("--dst-root", required=True, help="新实验根目录（_axisfix，不存在则从 src 复制）")
    ap.add_argument("--pos-axes", default=DEFAULT_POS_AXES,
                    help="旧 pajek 的 pos 轴序（这批 .tif pipeline=zyx；默认 zyx）")
    ap.add_argument("--exe", default=None, help="C++ vessel_stats（缺省自动查 build/）")
    ap.add_argument("--dry-run", action="store_true", help="仅预览，不复制/不写文件")
    ap.add_argument("--no-aggregate", action="store_true", help="跳过结尾的 aggregate 刷新")
    args = ap.parse_args()

    if not os.path.isdir(args.src_root):
        sys.exit(f"src-root 不存在: {args.src_root}")
    exe = find_exe(args.exe)

    print(f"[axisfix] src = {args.src_root}")
    print(f"[axisfix] dst = {args.dst_root}")
    print(f"[axisfix] pos_axes = {args.pos_axes} → xyz；exe = {exe}；dry = {args.dry_run}\n")

    # 复制实验树（保留旧目录；dst 已存在则不覆盖，直接在其上幂等重跑）
    if not os.path.exists(args.dst_root):
        if args.dry_run:
            print(f"[dry] 将复制 {args.src_root} → {args.dst_root}")
        else:
            print(f"复制实验树 → {args.dst_root} ...")
            shutil.copytree(args.src_root, args.dst_root)
    else:
        print(f"dst 已存在，在其上幂等重跑（不重新复制）。")

    walk_root = args.src_root if args.dry_run and not os.path.exists(args.dst_root) else args.dst_root
    counts: dict[str, int] = {}
    for run_dir in iter_run_dirs(walk_root):
        status = reprocess_run(run_dir, exe, args.pos_axes, args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        tag = "OK " if status == "done" else "-- "
        print(f"  [{tag}] {status:14s} {os.path.relpath(run_dir, walk_root)}")

    print("\n=== 汇总 ===")
    for k in sorted(counts):
        print(f"  {k:16s}: {counts[k]}")

    if not args.dry_run and counts.get("done") and not args.no_aggregate:
        print("\n=== 刷新 aggregate（新 root）===")
        run_cli_aggregates(args.dst_root)


if __name__ == "__main__":
    main()
