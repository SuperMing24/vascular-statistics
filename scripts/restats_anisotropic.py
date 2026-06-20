"""
对已完成的骨架化 run 目录重新运行 C++ 统计（各向异性口径），并更新 run_meta.json。

用法（在服务器项目目录下执行）:
    conda activate vascstats
    python scripts/restats_anisotropic.py <output_root> [sample_keys...]

    # 全部样本
    python scripts/restats_anisotropic.py /share/home/sukm/experiments/vs_2pfm_skelgt

    # 指定样本
    python scripts/restats_anisotropic.py /share/home/sukm/experiments/vs_vascstats \
        BCAS_1st/20240101_A001_D0/sample_001

原理：
  1. 扫描每个样本目录下所有含 skeleton.pajek 的 run_* 子目录
  2. 从 sample_metadata.json 读取 voxel_spacing_um
  3. 运行格式转换 + C++ 统计（各向异性口径）——覆盖旧 statistics_summary.txt
  4. 更新 run_meta.json（写入 stats_unit_mode / spacing_um / r_scale）
  5. 可选：自动刷新 sample-level aggregate + micro-stats
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta


PROJECT_DIR = "/share/home/sukm/vascular-statistics"
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
EXE_CANDIDATES = [
    os.path.join(BUILD_DIR, "vessel_stats"),
    os.path.join(BUILD_DIR, "vessel_stats.exe"),
]


def find_exe() -> str:
    for c in EXE_CANDIDATES:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(f"找不到 vessel_stats 可执行文件，已尝试: {EXE_CANDIDATES}")


def scan_run_dirs(sample_dir: str) -> list[str]:
    """扫描样本目录下所有含 skeleton.pajek 的 run_* 子目录。"""
    runs = []
    if not os.path.isdir(sample_dir):
        return runs
    for entry in sorted(os.listdir(sample_dir)):
        run_dir = os.path.join(sample_dir, entry)
        if not entry.startswith("run_") or not os.path.isdir(run_dir):
            continue
        if os.path.isfile(os.path.join(run_dir, "skeleton.pajek")):
            runs.append(run_dir)
    return runs


def restats_run(run_dir: str, volume_mm3: float, spacing_um: list[float],
                exe: str) -> bool:
    """对单个 run 目录重新运行统计。"""
    stem = os.path.join(run_dir, "skeleton")
    pajek_path = stem + ".pajek"
    edges_path = stem + "_edges.txt"
    vertices_path = stem + "_vertices.txt"

    # --- 格式转换 ---
    sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))
    from vascular_statistics.bridge import pajek_to_cpp_input
    pajek_to_cpp_input(pajek_path, edges_path, vertices_path)

    # --- C++ 统计（各向异性） ---
    r_scale = (spacing_um[0] + spacing_um[1]) / 2.0
    cmd = [
        exe,
        stem + "_edges",
        stem + "_vertices",
        str(volume_mm3),
        str(spacing_um[0]),
        str(spacing_um[1]),
        str(spacing_um[2]),
    ]
    result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] C++ stats 失败 (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return False

    # --- 更新 run_meta.json ---
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta: dict = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    meta["stats_unit_mode"] = "anisotropic"
    meta["spacing_um"] = spacing_um
    meta["r_scale"] = r_scale
    meta["restats_anisotropic_at"] = datetime.now(
        timezone(timedelta(hours=8))
    ).isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- 读取新统计摘要（用于日志） ---
    summary_path = os.path.join(run_dir, "statistics_summary.txt")
    diam = length = tort = segs = "?"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                if "中位直径" in line:
                    diam = line.split(":")[-1].strip()
                elif "中位长度" in line:
                    length = line.split(":")[-1].strip()
                elif "平均弯曲度" in line:
                    tort = line.split(":")[-1].strip()
                elif "有效段数" in line:
                    segs = line.split(":")[-1].strip()

    print(f"    中位直径={diam} μm, 中位长度={length} μm, "
          f"弯曲度={tort}, 段数={segs}")
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/restats_anisotropic.py <output_root> [sample_key ...]")
        sys.exit(1)

    output_root = sys.argv[1]
    specified_keys = sys.argv[2:] if len(sys.argv) > 2 else []

    # 收集样本目录
    if specified_keys:
        sample_dirs = [os.path.join(output_root, k) for k in specified_keys]
    else:
        sample_dirs = []
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
                    if os.path.isdir(sample_path):
                        sample_dirs.append(sample_path)

    exe = find_exe()
    print(f"C++ 可执行文件: {exe}")

    total_runs = 0
    ok_runs = 0
    fail_runs = 0

    for sample_dir in sample_dirs:
        runs = scan_run_dirs(sample_dir)
        if not runs:
            continue

        sample_key = os.path.relpath(sample_dir, output_root).replace("\\", "/")

        # 读取 metadata
        meta_path = os.path.join(sample_dir, "sample_metadata.json")
        if not os.path.exists(meta_path):
            print(f"[SKIP] {sample_key} — 无 sample_metadata.json")
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            sm = json.load(f)

        volume = sm.get("spatial", {}).get("tissue_volume_mm3")
        spacing = sm.get("spatial", {}).get("voxel_spacing_um")
        if volume is None or not spacing or len(spacing) != 3:
            print(f"[SKIP] {sample_key} — 缺少 volume 或 voxel_spacing_um")
            continue

        spacing_um = [float(v) for v in spacing]
        print(f"\n{sample_key}  volume={volume} mm³  spacing={spacing_um}  runs={len(runs)}")

        for run_dir in runs:
            run_name = os.path.basename(run_dir)
            total_runs += 1
            print(f"  {run_name} ...", end="", flush=True)
            if restats_run(run_dir, float(volume), spacing_um, exe):
                ok_runs += 1
            else:
                fail_runs += 1

    print(f"\n===== 完成 =====")
    print(f"总计: {total_runs} runs, 成功: {ok_runs}, 失败: {fail_runs}")

    if fail_runs > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
