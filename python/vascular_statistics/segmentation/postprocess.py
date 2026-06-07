"""
后处理：推理输出 → 二值掩码 → 骨架化就绪。

predict.py 输出为多页 TIFF（[D, H, W] uint8, 值域 0/255）。
本模块提供：
  - 加载并转换回 [H, W, D] 二值数组
  - 可选形态学后处理（小连通域去除）
  - 质量报告
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_mask(tif_path: str) -> np.ndarray:
    """加载 predict.py 输出的多页 TIFF 掩码，转为 [H, W, D] bool 数组。

    参数：
        tif_path: predict.py 输出的 .tif/.tiff 文件路径。

    返回：
        [H, W, D] bool 数组（True = 前景/血管）。
    """
    try:
        import tifffile
        pages = tifffile.imread(str(tif_path))
    except ImportError:
        from PIL import Image
        img = Image.open(str(tif_path))
        frames = []
        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        pages = np.stack(frames, axis=0) if frames else np.array(img)

    # pages: [D, H, W] uint8 (0/255) → [H, W, D] bool
    mask = (pages > 127).transpose(1, 2, 0)
    return mask


def mask_stats(mask: np.ndarray) -> dict:
    """计算二值掩码的统计信息。

    返回：
        {shape, total_voxels, foreground_voxels, foreground_ratio,
         num_connected_components, largest_component_size}
    """
    mask = np.asarray(mask, dtype=bool)
    stats = {
        "shape": mask.shape,
        "total_voxels": int(mask.size),
        "foreground_voxels": int(mask.sum()),
        "foreground_ratio": float(mask.mean()),
    }

    # 连通域分析（如 scipy 可用）
    try:
        from scipy import ndimage
        labeled, n_components = ndimage.label(mask)
        stats["num_connected_components"] = int(n_components)
        if n_components > 0:
            sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
            stats["largest_component_size"] = int(sizes.max())
            stats["mean_component_size"] = float(sizes.mean())
    except ImportError:
        pass

    return stats


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """去除小于指定体素数的连通域（去噪）。

    参数：
        mask: [H, W, D] bool 数组。
        min_size: 最小保留体素数（默认 100）。

    返回：
        过滤后的 bool 数组。
    """
    from scipy import ndimage
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    keep_labels = np.where(sizes >= min_size)[0] + 1
    return np.isin(labeled, keep_labels)
