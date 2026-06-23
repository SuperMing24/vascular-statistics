"""
元数据提取 — Lab_Data 各成员 angiogram 数据（raw 原图 / cropped_z 仅Z裁切）

支持 4 种情形：{huaien, xiaoqian} × {raw, cropped}

用法：
  # 原图（无裁切）
  python scripts/extract_raw_metadata.py \\
    --data-root /share/home/sukm/datasets/Lab_Data/huaien/raw \\
    --member huaien --variant raw

  # 仅Z裁切数据（XY 保留 512 全幅）
  python scripts/extract_raw_metadata.py \\
    --data-root /share/home/sukm/datasets/Lab_Data/xiaoqian/cropped_z \\
    --member xiaoqian --variant cropped

输出：data_root/sample_catalog.json
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

# ── 体素物理分辨率配置 ────────────────────────────────────
# 全实验室 2PFM 标准 XY 标定（huaien / xiaoqian 一致，用户确认 2026-06-23）
VOXEL_SPACING_XY = [1.1142, 1.0652]  # µm
Z_SPACING_DEFAULT = 2.0              # µm

# z 步长=1µm 的样本（成像时 z 采样更密，用户提供 2026-06-23）：
#   xiaoqian ACTH_SAL / A161 / Day15   → 20260128_A161_D15
#   xiaoqian magraine NTG / A427 / Day10 → 20250807_A427_D10（仅 raw，无裁切）
# 以 date_dir 子串匹配（"A161_D15" / "A427_D10"）。
Z1_SAMPLES = ["A161_D15", "A427_D10"]


def resolve_spacing(date_dir: str) -> List[float]:
    """按样本 date_dir 解析体素分辨率（z 步长例外处理）。"""
    z = Z_SPACING_DEFAULT
    for ex in Z1_SAMPLES:
        if ex in date_dir:
            z = 1.0
            break
    return VOXEL_SPACING_XY + [z]


# ── glob 模式：(member, variant) → pattern ───────────────
GLOB_PATTERNS = {
    ("huaien", "raw"):       "**/angiogram.mat",
    ("huaien", "cropped"):   "**/angiogram_crop_*.mat",
    ("xiaoqian", "raw"):     "**/*_angiogram.mat",        # 不匹配 *_crop_*
    ("xiaoqian", "cropped"): "**/*_angiogram_crop_*.mat",
}


# ── 扫描 ──────────────────────────────────────────────────
def discover_files(data_root: str, member: str, variant: str) -> List[str]:
    import glob as _glob
    pattern = os.path.join(data_root, GLOB_PATTERNS[(member, variant)])
    return sorted(_glob.glob(pattern, recursive=True))


# ── 文件名 z 裁切范围解析 ─────────────────────────────────
def parse_crop_range(stem: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """从含 _crop_X_Y 的文件名解析 (z_start, z_end, num_slices)。"""
    if "_crop_" not in stem:
        return None, None, None
    tail = stem.split("_crop_")[1]
    segs = tail.split("_")
    if len(segs) == 2:
        try:
            zs, ze = int(segs[0]), int(segs[1])
            return zs, ze, ze - zs + 1
        except ValueError:
            pass
    return None, None, None


# ── date_dir 段解析（兼容 '512_' 等非日期前缀） ──────────
def split_date_segments(base: str) -> List[str]:
    """切分 date_dir 段，剥离前导非日期段（如 '512_' 标注）。

    日期段定义为 8 位纯数字（YYYYMMDD）。返回从日期段起的列表
    [date, animal, daypoint, ...]。
    """
    segs = base.split("_")
    while segs and not (len(segs[0]) == 8 and segs[0].isdigit()):
        segs.pop(0)
    return segs


# ── 路径解析 ──────────────────────────────────────────────
def parse_path(mat_path: str, data_root: str, member: str, variant: str) -> Dict[str, Any]:
    rel = os.path.relpath(mat_path, data_root)
    parts = Path(rel).parts
    stem = Path(mat_path).stem
    zs, ze, nslices = parse_crop_range(stem)

    if member == "huaien":
        # 结构: {group}/{date_dir}/angiogram/<file>.mat
        group = parts[0] if len(parts) > 0 else "_unknown"
        date_dir = parts[1] if len(parts) > 1 else "_unknown"
        segments = split_date_segments(date_dir)
        animal_id = segments[1] if len(segments) > 1 else "_unknown"
        daypoint = segments[2] if len(segments) > 2 else "_unknown"
        # sample_key 与 cropped_xyz catalog 对齐
        leaf = "angiogram_crop_%d_%d" % (zs, ze) if zs is not None else "angiogram"
        sample_key = f"{group}/{date_dir}/{leaf}"
    else:
        # xiaoqian 结构: {exp}/{subgroup}/<file>.mat（扁平）
        exp = parts[0] if len(parts) > 0 else "_unknown"
        subgroup = parts[1] if len(parts) > 1 else "_unknown"
        group = f"{exp}/{subgroup}"
        # 文件名: YYYYMMDD_AXXX_DNN_angiogram[_crop_X_Y]，可能带 '512_' 前缀
        base = stem.split("_angiogram")[0]
        segments = split_date_segments(base)
        animal_id = segments[1] if len(segments) > 1 else "_unknown"
        daypoint = segments[2] if len(segments) > 2 else "_unknown"
        date_dir = "_".join(segments[:3]) if len(segments) >= 3 else base
        sample_key = f"{exp}/{subgroup}/{stem}"

    return {
        "group": group,
        "batch_id": animal_id,
        "daypoint": daypoint,
        "date_dir": date_dir,
        "z_start": zs,
        "z_end": ze,
        "num_slices": nslices,
        "sample_key": sample_key,
        "input_rel_path": rel.replace("\\", "/"),
    }


# ── .mat 加载 ─────────────────────────────────────────────
def load_mat_properties(mat_path: str) -> Dict[str, Any]:
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
    return {
        "shape": list(shape),
        "foreground_voxel_count": int((stack > 0).sum()),
        "dtype": str(stack.dtype),
    }


# ── 体素计算 ──────────────────────────────────────────────
def compute_volume_mm3(shape: List[int], spacing: List[float]) -> float:
    return shape[0] * shape[1] * shape[2] * spacing[0] * spacing[1] * spacing[2] / 1e9


def compute_vessel_volume_mm3(fg: int, spacing: List[float]) -> float:
    return fg * spacing[0] * spacing[1] * spacing[2] / 1e9


# ── Catalog ───────────────────────────────────────────────
def build_catalog(samples: List[dict], data_root: str) -> dict:
    by_group: Dict[str, dict] = {}
    samples_index: Dict[str, dict] = {}

    for s in samples:
        g = s["group"]
        sk = s["sample_key"]
        spacing = resolve_spacing(s["date_dir"])

        if g not in by_group:
            by_group[g] = {"count": 0, "animals": set(), "sample_keys": []}
        by_group[g]["count"] += 1
        by_group[g]["animals"].add(s["batch_id"])
        by_group[g]["sample_keys"].append(sk)

        entry = {
            "group": g,
            "batch_id": s["batch_id"],
            "daypoint": s["daypoint"],
            "shape": s["shape"],
            "foreground_voxel_count": s["foreground_voxel_count"],
            "voxel_spacing_um": spacing,
            "tissue_volume_mm3": compute_volume_mm3(s["shape"], spacing),
            "vessel_volume_mm3": compute_vessel_volume_mm3(s["foreground_voxel_count"], spacing),
            "input_rel_path": s["input_rel_path"],
        }
        # 裁切样本附带 z 范围信息
        if s.get("z_start") is not None:
            entry["z_start"] = s["z_start"]
            entry["z_end"] = s["z_end"]
            entry["num_slices"] = s["num_slices"]
        samples_index[sk] = entry

    for g in by_group:
        by_group[g]["animals"] = sorted(by_group[g]["animals"])
        by_group[g]["sample_keys"] = sorted(by_group[g]["sample_keys"])

    return {
        "version": "1.1",
        "dataset_root": os.path.abspath(data_root),
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "overview": {
            "total_samples": len(samples),
            "total_groups": len(by_group),
        },
        "by_group": by_group,
        "samples": samples_index,
        "voxel_spacing_config_used": {
            "xy_um": VOXEL_SPACING_XY,
            "z_um_default": Z_SPACING_DEFAULT,
            "z_um_1_samples": Z1_SAMPLES,
            "units": "micrometers",
        },
    }


# ── 主流程 ────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--member", required=True, choices=["huaien", "xiaoqian"])
    ap.add_argument("--variant", default="raw", choices=["raw", "cropped"],
                    help="raw=原图 angiogram.mat；cropped=仅Z裁切 *_crop_X_Y.mat")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    files = discover_files(data_root, args.member, args.variant)
    print(f"发现 {len(files)} 个样本 (member={args.member}, variant={args.variant})\n")

    samples = []
    failed = 0
    for i, f in enumerate(files, 1):
        try:
            parsed = parse_path(f, data_root, args.member, args.variant)
            props = load_mat_properties(f)
            entry = {**parsed,
                     "shape": props["shape"],
                     "foreground_voxel_count": props["foreground_voxel_count"]}
            samples.append(entry)
            z = resolve_spacing(parsed["date_dir"])[2]
            ztag = f" z={z}" if z != Z_SPACING_DEFAULT else ""
            print(f"  [{i}/{len(files)}] {parsed['sample_key']}  "
                  f"shape={props['shape']}  fg={props['foreground_voxel_count']:,}{ztag}")
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
