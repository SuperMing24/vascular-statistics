"""
元数据提取 — 原始 angiogram.mat（无裁切信息）

用法：
  python scripts/extract_raw_metadata.py \\
    --data-root /share/home/sukm/datasets/Lab_Data/huaien/raw \\
    --member huaien

输出：data_root/sample_catalog.json
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

# ── 配置 ──────────────────────────────────────────────────
VOXEL_SPACING_UM = [1.1142, 1.0652, 2.0]  # 课题组标准分辨率 (µm)


# ── 扫描 ──────────────────────────────────────────────────
def discover_raw_files(data_root: str) -> List[str]:
    """递归扫描所有 angiogram.mat（排除 cropped 目录）。"""
    import glob as _glob
    pattern = os.path.join(data_root, "**/angiogram.mat")
    files = sorted(_glob.glob(pattern, recursive=True))
    return [f for f in files if "/cropped/" not in f]


# ── 路径解析（适配原始 angiogram.mat） ───────────────────
def parse_raw_path(mat_path: str, data_root: str) -> Dict[str, Any]:
    """从原始 angiogram.mat 路径解析组别 / 动物 / 时间点。"""
    rel = os.path.relpath(mat_path, data_root)
    parts = Path(rel).parts  # e.g. ('BCAS_1st', '20241009_A192_D0', 'angiogram', 'angiogram.mat')

    group = parts[0] if len(parts) > 0 else "_unknown"
    date_dir = parts[1] if len(parts) > 1 else "_unknown"

    animal_id = "_unknown"
    daypoint = "_unknown"
    if date_dir != "_unknown":
        segments = date_dir.split("_")
        if len(segments) >= 3:
            animal_id = segments[1]
            daypoint = segments[2]

    # sample_key: 用于 catalog 索引和 cropped ↔ raw 对齐
    sample_key = f"{group}/{date_dir}/angiogram"

    return {
        "group": group,
        "batch_id": animal_id,
        "daypoint": daypoint,
        "date_dir": date_dir,
        "sample_key": sample_key,
        "input_rel_path": rel.replace("\\", "/"),
    }


# ── .mat 加载 ─────────────────────────────────────────────
def load_mat_properties(mat_path: str) -> Dict[str, Any]:
    """加载 .mat 文件，提取形状 / 前景计数。"""
    raw = sio.loadmat(mat_path)
    stack = None
    for key, val in raw.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 3:
            stack = val
            break
    if stack is None:
        raise ValueError(f"未找到 3D ndarray: {mat_path}")

    shape = tuple(int(s) for s in stack.shape)
    foreground = int((stack > 0).sum())
    return {
        "shape": list(shape),
        "foreground_voxel_count": foreground,
        "dtype": str(stack.dtype),
    }


# ── 体素计算 ──────────────────────────────────────────────
def compute_volume_mm3(shape: List[int], spacing: List[float]) -> float:
    voxel_count = shape[0] * shape[1] * shape[2]
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return voxel_count * voxel_vol / 1e9


def compute_vessel_volume_mm3(fg: int, spacing: List[float]) -> float:
    return fg * spacing[0] * spacing[1] * spacing[2] / 1e9


# ── Catalog ───────────────────────────────────────────────
def build_catalog(samples: List[dict], data_root: str) -> dict:
    by_group: Dict[str, dict] = {}
    samples_index: Dict[str, dict] = {}

    for s in samples:
        g = s["group"]
        sk = s["sample_key"]
        if g not in by_group:
            by_group[g] = {"count": 0, "animals": set(), "sample_keys": []}
        by_group[g]["count"] += 1
        by_group[g]["animals"].add(s["batch_id"])
        by_group[g]["sample_keys"].append(sk)

        samples_index[sk] = {
            "group": g,
            "batch_id": s["batch_id"],
            "daypoint": s["daypoint"],
            "shape": s["shape"],
            "foreground_voxel_count": s["foreground_voxel_count"],
            "voxel_spacing_um": VOXEL_SPACING_UM,
            "tissue_volume_mm3": compute_volume_mm3(s["shape"], VOXEL_SPACING_UM),
            "vessel_volume_mm3": compute_vessel_volume_mm3(s["foreground_voxel_count"], VOXEL_SPACING_UM),
            "input_rel_path": s["input_rel_path"],
        }

    for g in by_group:
        by_group[g]["animals"] = sorted(by_group[g]["animals"])
        by_group[g]["sample_keys"] = sorted(by_group[g]["sample_keys"])

    return {
        "version": "1.0",
        "dataset_root": os.path.abspath(data_root),
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "overview": {
            "total_samples": len(samples),
            "total_groups": len(by_group),
        },
        "by_group": by_group,
        "samples": samples_index,
        "voxel_spacing_config_used": {
            "default": VOXEL_SPACING_UM,
            "units": "micrometers",
        },
    }


# ── 主流程 ────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--member", required=True, help="huaien | xiaoqian")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)

    if args.member == "huaien":
        files = discover_raw_files(data_root)
    else:
        # xiaoqian: flat layout, *_angiogram.mat
        import glob as _glob
        pattern = os.path.join(data_root, "**/*_angiogram.mat")
        files = sorted(_glob.glob(pattern, recursive=True))

    print(f"发现 {len(files)} 个样本\n")

    samples = []
    failed = 0
    for i, f in enumerate(files, 1):
        try:
            if args.member == "huaien":
                parsed = parse_raw_path(f, data_root)
            else:
                # xiaoqian: parent dir = subgroup, grandparent = experiment
                rel = os.path.relpath(f, data_root)
                parts = Path(rel).parts
                exp = parts[0] if len(parts) > 0 else "_unknown"
                subgroup = parts[1] if len(parts) > 1 else "_unknown"
                fname = Path(f).stem
                # fname: YYYYMMDD_AXXX_DNN_angiogram or YYYYMMDD_AXXX_DNN_DN+DN_angiogram
                base = fname.replace("_angiogram", "")
                segs = base.split("_")
                animal_id = segs[1] if len(segs) > 1 else "_unknown"
                daypoint = segs[2] if len(segs) > 2 else "_unknown"
                # 重建 date_dir
                date_dir = "_".join(segs[:3])
                sample_key = f"{exp}/{subgroup}/{date_dir}"

                parsed = {
                    "group": f"{exp}/{subgroup}",
                    "batch_id": animal_id,
                    "daypoint": daypoint,
                    "date_dir": date_dir,
                    "sample_key": sample_key,
                    "input_rel_path": rel.replace("\\", "/"),
                }

            props = load_mat_properties(f)
            entry = {**parsed, "shape": props["shape"], "foreground_voxel_count": props["foreground_voxel_count"]}
            samples.append(entry)
            print(f"  [{i}/{len(files)}] {parsed['sample_key']}  "
                  f"shape={props['shape']}  fg={props['foreground_voxel_count']:,}")
        except Exception as e:
            failed += 1
            print(f"  [FAILED] {f}: {e}")

    catalog = build_catalog(samples, data_root)
    out_path = os.path.join(data_root, "sample_catalog.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(catalog, fp, indent=2, ensure_ascii=False)

    print(f"\n完成: {len(samples)} 成功, {failed} 失败")
    print(f"Catalog → {out_path}")


if __name__ == "__main__":
    main()
