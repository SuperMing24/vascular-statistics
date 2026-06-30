"""
样本元数据提取模块 — 从 .mat 文件与路径中提取每样本元数据，
写入每样本 JSON + 全局导航目录。
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from vascular_statistics.batch import STAGE_DIRS, compute_sample_key

# ═══════════════════════════════════════════════════════════════════════
# 扫描
# ═══════════════════════════════════════════════════════════════════════

def discover_mat_files(
    data_root: str,
    glob_pattern: str = "**/angiogram_crop_*.mat",
) -> List[str]:
    """递归扫描 data_root 下的 .mat 文件。"""
    from glob import glob as _glob
    pattern = os.path.join(data_root, glob_pattern)
    # glob 新式递归支持 **
    try:
        files = _glob(pattern, recursive=True)
    except TypeError:
        # Python < 3.5 fallback: walk manually
        files = []
        for dirpath, _, filenames in os.walk(data_root):
            for fn in filenames:
                if fn.endswith(".mat") and "angiogram_crop_" in fn:
                    files.append(os.path.join(dirpath, fn))
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════
# 路径解析
# ═══════════════════════════════════════════════════════════════════════

def parse_sample_path(
    mat_path: str,
    data_root: str,
) -> Dict[str, Any]:
    """从 .mat 文件路径中解析组别 / 动物 / 时间点 / 裁剪范围。"""
    try:
        rel = os.path.relpath(mat_path, data_root)
    except ValueError:
        rel = mat_path

    p = Path(rel)
    parts = p.parts
    filename = p.stem  # angiogram_crop_X_Y（无扩展名）

    # 文件名解析：angiogram_crop_{z_start}_{z_end}
    z_start: Optional[int] = None
    z_end: Optional[int] = None
    if "_crop_" in filename:
        crop_parts = filename.split("_crop_")
        if len(crop_parts) == 2:
            z_range = crop_parts[1].split("_")
            if len(z_range) == 2:
                try:
                    z_start = int(z_range[0])
                    z_end = int(z_range[1])
                except ValueError:
                    pass

    num_slices = None
    if z_start is not None and z_end is not None:
        num_slices = z_end - z_start + 1

    # 路径层级：group / date_animal_daypoint / [angiogram] / file
    # 过滤 stage dirs（从 batch.py 复用逻辑）
    meaningful = [d for d in parts[:-1] if d.lower() not in STAGE_DIRS]

    group = meaningful[0] if len(meaningful) > 0 else "_unknown"
    date_dir = meaningful[1] if len(meaningful) > 1 else "_unknown"

    # date_dir 格式：YYYYMMDD_AXXX_DNN 或 YYYYMMDD_NT_DNN
    animal_id = "_unknown"
    daypoint = "_unknown"
    if date_dir != "_unknown":
        segments = date_dir.split("_")
        if len(segments) >= 3:
            # segments[0] = YYYYMMDD, segments[1] = AXXX or NT, segments[2] = DNN
            animal_id = segments[1]
            daypoint = segments[2]

    sample_key = compute_sample_key(rel)

    return {
        "group": group,
        "batch_id": animal_id,  # Axxx = 成像实验批次标识，非动物编号
        "daypoint": daypoint,
        "date_dir": date_dir,
        "z_start": z_start,
        "z_end": z_end,
        "num_slices": num_slices,
        "sample_key": sample_key,
        "input_rel_path": rel.replace("\\", "/"),
    }


# ═══════════════════════════════════════════════════════════════════════
# .mat 加载
# ═══════════════════════════════════════════════════════════════════════

def load_mat_metadata(mat_path: str) -> Dict[str, Any]:
    """加载 .mat 文件，提取 stack 形状 / 前景计数 / 裁剪元数据。"""
    raw = sio.loadmat(mat_path)

    # 找到第一个 ndarray（stack）
    stack = None
    rect_position = None
    frame_range = None

    for key, val in raw.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray):
            if val.ndim == 3 and stack is None:
                stack = val
            elif val.size == 4 and rect_position is None:
                rect_position = val.flatten().tolist()
            elif val.size == 2 and frame_range is None:
                frame_range = val.flatten().tolist()

    if stack is None:
        raise ValueError(f"未在 {mat_path} 中找到 3D ndarray (stack)")

    # 注意：.mat stack 原序为 [H, W, D]（面内两轴 + 末轴切片数 Z），非 [D,H,W]。
    # 下游若需对齐 seg .tif 的 [D,H,W] 轴序，须自行转换（见 cli._write_run_spacing_sidecar）。
    shape = tuple(int(s) for s in stack.shape)  # [H, W, D]（D=切片数，在末轴）
    foreground = int((stack > 0).sum())
    total_voxels = int(np.prod(shape))
    grayscale_sum = float(stack.sum())

    # dtype 转字符串
    dtype_name = str(stack.dtype)

    return {
        "shape": list(shape),
        "dtype": dtype_name,
        "voxel_count_total": total_voxels,
        "foreground_voxel_count": foreground,
        "grayscale_sum": grayscale_sum,
        "rect_position": rect_position,
        "frame_range": frame_range if frame_range else [1, shape[0]],
    }


# ═══════════════════════════════════════════════════════════════════════
# 体素分辨率
# ═══════════════════════════════════════════════════════════════════════

def load_voxel_spacing_config(config_path: Optional[str]) -> Optional[dict]:
    """加载体素分辨率层级配置。文件不存在时返回 None。"""
    if config_path is None or not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sample_spacing(
    sample_key: str,
    group: str,
    config: Optional[dict],
) -> Tuple[Optional[List[float]], str]:
    """层级化查找体素分辨率。

    返回：(spacing_um, source)
    source: "per_sample" | "per_group" | "default" | "none"
    """
    if config is None:
        return None, "none"

    per_sample = config.get("per_sample", {})
    # 1) 精确匹配（完整 sample_key）
    if sample_key in per_sample and isinstance(per_sample[sample_key], list):
        return per_sample[sample_key], "per_sample"
    # 2) 子串匹配（key 为 sample_key 的子串，如动物-时间点 token "A161_D15"）——
    #    用于「同一动物的所有 crop 共享采集 spacing」的例外。最长 key 优先（更具体）。
    #    仅取值为 list 的条目（跳过 _note_* 等注释键）。
    substr = [(k, v) for k, v in per_sample.items()
              if isinstance(v, list) and k in sample_key]
    if substr:
        k, v = max(substr, key=lambda kv: len(kv[0]))
        return v, "per_sample(substr)"

    per_group = config.get("per_group", {})
    if group in per_group and per_group[group] is not None:
        return per_group[group], "per_group"

    default = config.get("default")
    if default is not None:
        return default, "default"

    return None, "none"


def compute_volume_mm3(
    shape: List[int],
    spacing_um: List[float],
) -> float:
    """总组织体积 (mm^3) = D*H*W * dx*dy*dz / 1e9。"""
    voxel_count = shape[0] * shape[1] * shape[2]
    voxel_volume_um3 = spacing_um[0] * spacing_um[1] * spacing_um[2]
    return voxel_count * voxel_volume_um3 / 1e9


def compute_vessel_volume_mm3(
    foreground_voxel_count: int,
    spacing_um: List[float],
) -> float:
    """血管体积 (mm^3) = 前景体素数 * dx*dy*dz / 1e9。"""
    voxel_volume_um3 = spacing_um[0] * spacing_um[1] * spacing_um[2]
    return foreground_voxel_count * voxel_volume_um3 / 1e9


# ═══════════════════════════════════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════════════════════════════════

def write_sample_metadata(
    metadata: dict,
    output_root: str,
) -> str:
    """写每样本元数据 JSON 到输出目录。"""
    sample_dir = os.path.join(output_root, metadata["sample_key"])
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, "sample_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return path


def build_sample_catalog(
    all_samples_meta: List[dict],
    voxel_spacing_config: Optional[dict],
    data_root: str,
) -> dict:
    """从所有样本元数据列表构建全局导航目录。"""
    by_group: Dict[str, dict] = {}
    samples_index: Dict[str, dict] = {}

    for meta in all_samples_meta:
        group = meta["path_parsed"]["group"]
        sk = meta["sample_key"]

        if group not in by_group:
            by_group[group] = {"count": 0, "animals": set(), "sample_keys": []}
        by_group[group]["count"] += 1
        # 兼容旧字段名 (animal_id) 与新字段名 (batch_id)
        pp = meta["path_parsed"]
        batch = pp.get("batch_id") or pp.get("animal_id", "?")
        by_group[group]["animals"].add(batch)
        by_group[group]["sample_keys"].append(sk)

        samples_index[sk] = {
            "group": group,
            "batch_id": batch,
            "daypoint": pp["daypoint"],
            "shape": meta["stack_properties"]["shape"],
            "foreground_voxel_count": meta["stack_properties"]["foreground_voxel_count"],
            "voxel_spacing_um": meta["spatial"]["voxel_spacing_um"],
            "tissue_volume_mm3": meta["spatial"]["tissue_volume_mm3"],
            "vessel_volume_mm3": meta["spatial"]["vessel_volume_mm3"],
            "input_rel_path": meta["input_rel_path"],
        }

    # 将 set 转为 list 以便 JSON 序列化
    for g in by_group:
        by_group[g]["animals"] = sorted(by_group[g]["animals"])
        by_group[g]["sample_keys"] = sorted(by_group[g]["sample_keys"])

    spacing_ref = None
    if voxel_spacing_config:
        spacing_ref = {
            "default": voxel_spacing_config.get("default"),
            "units": voxel_spacing_config.get("units", "micrometers"),
        }

    return {
        "version": "1.0",
        "dataset_root": os.path.abspath(data_root),
        "generated_at": datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat(),
        "overview": {
            "total_samples": len(all_samples_meta),
            "total_groups": len(by_group),
        },
        "by_group": by_group,
        "samples": samples_index,
        "voxel_spacing_config_used": spacing_ref,
    }


def write_sample_catalog(catalog: dict, data_root: str) -> str:
    """写全局 navigation catalog 到数据集根目录。"""
    path = os.path.join(data_root, "sample_catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    return path


# ═══════════════════════════════════════════════════════════════════════
# 编排
# ═══════════════════════════════════════════════════════════════════════

def run_extraction(
    data_root: str,
    output_root: str,
    voxel_spacing_config_path: Optional[str] = None,
    regen: bool = False,
    glob_pattern: str = "**/angiogram_crop_*.mat",
) -> dict:
    """编排完整提取流程。

    返回：
        {"processed": N, "skipped": N, "failed": N,
         "catalog_path": str, "samples": [...]}
    """
    spacing_config = load_voxel_spacing_config(voxel_spacing_config_path)
    files = discover_mat_files(data_root, glob_pattern)

    processed = 0
    skipped = 0
    failed = 0
    all_meta: List[dict] = []

    for mat_path in files:
        try:
            parsed = parse_sample_path(mat_path, data_root)
            sample_key = parsed["sample_key"]

            # 已有且非 regen → 跳过
            meta_path = os.path.join(
                output_root, sample_key, "sample_metadata.json"
            )
            if not regen and os.path.exists(meta_path):
                # 仍需要加载已有元数据加入 catalog
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                all_meta.append(existing)
                skipped += 1
                continue

            # 加载 .mat
            stack_meta = load_mat_metadata(mat_path)

            # 查找体素分辨率
            group = parsed["group"]
            spacing, spacing_source = resolve_sample_spacing(
                sample_key, group, spacing_config
            )

            # 计算 volume
            shape = stack_meta["shape"]
            if spacing:
                tissue_vol = compute_volume_mm3(shape, spacing)
                vessel_vol = compute_vessel_volume_mm3(
                    stack_meta["foreground_voxel_count"], spacing
                )
            else:
                tissue_vol = None
                vessel_vol = None
                spacing_source = "none"

            # 组装元数据
            metadata: dict = {
                "version": "1.0",
                "sample_key": sample_key,
                "input_rel_path": parsed["input_rel_path"],
                "path_parsed": {
                    "group": group,
                    "batch_id": parsed["batch_id"],
                    "daypoint": parsed["daypoint"],
                    "date_dir": parsed["date_dir"],
                    "z_start": parsed["z_start"],
                    "z_end": parsed["z_end"],
                    "num_slices": parsed["num_slices"],
                },
                "stack_properties": stack_meta,
                "spatial": {
                    "voxel_spacing_um": spacing,
                    "spacing_source": spacing_source,
                    "tissue_volume_mm3": tissue_vol,
                    "vessel_volume_mm3": vessel_vol,
                },
                "extraction_meta": {
                    "extracted_at": datetime.now(
                        timezone(timedelta(hours=8))
                    ).isoformat(),
                    "extractor_version": "1.0",
                },
            }

            write_sample_metadata(metadata, output_root)
            all_meta.append(metadata)
            processed += 1
            print(f"  [{processed}] {sample_key}")
            print(f"       shape={shape}, fg={stack_meta['foreground_voxel_count']}, "
                  f"spacing={spacing}, vol={tissue_vol}")

        except Exception as e:
            failed += 1
            print(f"  [FAILED] {mat_path}: {e}")

    # 构建并写入全局 catalog
    catalog = build_sample_catalog(all_meta, spacing_config, data_root)
    catalog_path = write_sample_catalog(catalog, data_root)

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "catalog_path": catalog_path,
        "samples": all_meta,
    }
