"""
一站式分割管线：.mat → .nii → 推理 → .tif。

此模块是 segment_mat() 的实现。通过 vascular_statistics.segmentation 包的
__getattr__ 延迟加载机制访问，避免本地环境缺少 nibabel/tifffile 时导入失败。
"""

import os
import tempfile
from pathlib import Path


def segment_mat(
    mat_path: str,
    output_dir: str,
    *,
    normalize: str = "passthrough",
    threshold: float = 0.5,
    min_component_size: int = 0,
    keep_nii: bool = False,
    device: str = "cuda",
    timeout_sec: int = 3600,
) -> str:
    """一站式：.mat 原图 → 分割 → 二值掩码 .tif。

    步骤：
        1. .mat → .nii（无损格式转换，float32）
        2. 可选强度归一化（默认 passthrough，依赖 predict.py 内部逐窗口 MinMax）
        3. subprocess 调用 Vascular_Extraction predict.py（nnU-Net 2D seed22）
        4. 可选后处理（小连通域去除）
        5. 返回 .tif 掩码路径

    参数：
        mat_path: 输入 .mat 文件路径（含 'stack' 键的 3D 数组）。
        output_dir: 输出目录。
        normalize: 归一化策略（"passthrough" | "minmax" | "percentile" | "zscore"）。
        threshold: 二值化阈值（默认 0.5）。
        min_component_size: 最小连通域体素数（0 = 不过滤）。
        keep_nii: 是否保留中间 .nii 文件。
        timeout_sec: 推理超时秒数。

    返回：
        二值掩码 .tif 文件的绝对路径。
    """
    from vascular_statistics.segmentation.convert import mat_to_nii
    from vascular_statistics.segmentation.preprocess import normalize_volume
    from vascular_statistics.segmentation.inference import run_segmentation

    mat_path = str(mat_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── 步骤 1: .mat → .nii ──
    stem = Path(mat_path).stem
    if keep_nii:
        nii_path = os.path.join(output_dir, stem + ".nii")
    else:
        # 使用可预测的临时文件名（避免 predict.py 输出带随机后缀）
        nii_path = os.path.join(tempfile.gettempdir(), stem + "_seg_tmp.nii")

    try:
        mat_to_nii(mat_path, nii_path)

        # ── 步骤 2: 可选归一化 ──
        if normalize != "passthrough":
            import nibabel as nib
            import numpy as np
            vol = nib.load(nii_path).get_fdata()
            vol = normalize_volume(vol, strategy=normalize)
            affine = np.eye(4, dtype=np.float64)
            nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), nii_path)

        # ── 步骤 3: 推理 ──
        tif_path = run_segmentation(
            nii_path=nii_path,
            output_dir=output_dir,
            threshold=threshold,
            device=device,
            timeout_sec=timeout_sec,
        )

        # ── 步骤 4: 可选后处理 ──
        if min_component_size > 0:
            from vascular_statistics.segmentation.postprocess import (
                load_mask,
                remove_small_components,
            )
            mask = load_mask(tif_path)
            mask = remove_small_components(mask, min_size=min_component_size)
            _save_mask_tif(mask, tif_path)

        return tif_path

    finally:
        # 清理临时 .nii
        if not keep_nii and os.path.exists(nii_path):
            try:
                if tempfile.gettempdir() in nii_path:
                    os.unlink(nii_path)
            except OSError:
                pass


def _save_mask_tif(mask, tif_path: str) -> None:
    """将 [H, W, D] bool 掩码保存为多页 TIFF [D, H, W] uint8。"""
    import numpy as np
    pages = (mask.transpose(2, 0, 1).astype(np.uint8)) * 255
    try:
        import tifffile
        tifffile.imwrite(str(tif_path), pages, photometric="minisblack")
    except ImportError:
        from PIL import Image
        imgs = [Image.fromarray(pages[i]) for i in range(pages.shape[0])]
        imgs[0].save(str(tif_path), save_all=True, append_images=imgs[1:])
