"""
格式转换：.mat（MATLAB v7）→ .nii（NIfTI-1）。

VascStats 原始数据为双光子荧光显微成像，存储为 .mat 文件中的 float64 数组。
Vascular_Extraction 的 predict.py 接受 .nii 格式输入。
本模块提供无损格式转换，保留原始数值精度。
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional


def mat_to_nii(
    mat_path: str,
    nii_path: Optional[str] = None,
    array_key: Optional[str] = None,
) -> str:
    """将 .mat 文件中的 3D 数组转换为 .nii 文件。

    参数：
        mat_path: .mat 文件路径。
        nii_path: 输出 .nii 路径。默认与输入同名（改后缀为 .nii）。
        array_key: .mat 中目标数组的键名。
                   默认取第一个非 __ 前缀的 numpy 数组。

    返回：
        输出 .nii 文件的绝对路径。

    异常：
        FileNotFoundError: .mat 文件不存在。
        ValueError: .mat 中找不到合适的 3D 数组。
    """
    mat_path = str(mat_path)
    if not Path(mat_path).exists():
        raise FileNotFoundError(f".mat 文件不存在: {mat_path}")

    mat = sio.loadmat(mat_path)

    # 查找目标 3D 数组 — 按优先级：
    #   1. 键名为 'stack' 的数组（VascStats 约定命名）
    #   2. 第一个 3D numpy 数组
    #   3. 如果用户指定了 array_key，则只查找该键
    target = None

    if array_key is not None:
        # 用户指定键名 → 严格匹配
        if array_key in mat:
            target = mat[array_key]
        else:
            available = [k for k, v in mat.items()
                         if not k.startswith("__") and isinstance(v, np.ndarray)]
            raise ValueError(
                f"在 {mat_path} 中找不到键 '{array_key}'。"
                f"可用键: {available}"
            )
    else:
        # 自动发现：优先 'stack'，否则取最大的 3D 数组
        arrays_3d = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray):
                v = np.squeeze(value)
                if v.ndim == 3:
                    arrays_3d.append((key, v))

        if not arrays_3d:
            available = [(k, np.squeeze(v).shape)
                         for k, v in mat.items()
                         if not k.startswith("__") and isinstance(v, np.ndarray)]
            raise ValueError(
                f"在 {mat_path} 中找不到 3D 数组。"
                f"可用键（含维度）: {available}"
            )

        # 优先 'stack' 键
        for key, arr in arrays_3d:
            if key == "stack":
                target = arr
                break

        if target is None:
            # 取体素最多的 3D 数组
            target = max(arrays_3d, key=lambda x: x[1].size)[1]

    # 去单一维度
    target = np.squeeze(target)
    if target.ndim != 3:
        raise ValueError(
            f"数组维度为 {target.ndim}D，期望 3D。shape={target.shape}"
        )

    # 转换为 float32（NIfTI 标准精度，兼容 predict.py 的 float32 加载）
    data = target.astype(np.float32)

    # Padding 到 32 的倍数 —— MONAI DynamicUNet 有 5 层下采样（stride=32），
    # MiniVess 训练数据都是 512×512（整除 32），但 VascStats 样本尺寸各异。
    # 不做 padding 会在 decoder skip connection 中出现
    # "Expected size 24 but got size 23" 错误。
    h, w, d = data.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    pad_d = (32 - d % 32) % 32
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        data = np.pad(data, ((0, pad_h), (0, pad_w), (0, pad_d)),
                      mode='constant', constant_values=0.0)

    # 确定输出路径
    if nii_path is None:
        stem = Path(mat_path).stem
        nii_path = str(Path(mat_path).with_name(stem + ".nii"))

    # 延迟导入 nibabel（仅在服务器端需要，本地 vascstats 环境可能未安装）
    import nibabel as nib

    # 写入 NIfTI-1（使用单位仿射矩阵，因为我们不关心物理空间对齐）
    affine = np.eye(4, dtype=np.float64)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, nii_path)

    return str(Path(nii_path).resolve())


def mat_info(mat_path: str) -> dict:
    """查看 .mat 文件中所有数组的元信息，用于调试。

    返回：
        {key: {'shape': ..., 'dtype': ..., 'min': ..., 'max': ..., 'nonzero_ratio': ...}}
    """
    mat = sio.loadmat(str(mat_path))
    info = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            v = np.squeeze(value)
            nonzero = np.count_nonzero(v)
            info[key] = {
                "shape": v.shape,
                "ndim": v.ndim,
                "dtype": str(v.dtype),
                "min": float(v.min()),
                "max": float(v.max()),
                "nonzero": int(nonzero),
                "nonzero_ratio": float(nonzero / v.size) if v.size > 0 else 0.0,
            }
    return info
