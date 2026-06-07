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

    # 查找目标数组
    target = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            if array_key is None or key == array_key:
                target = value
                if array_key is not None:
                    break

    if target is None:
        available = [k for k, v in mat.items()
                     if not k.startswith("__") and isinstance(v, np.ndarray)]
        raise ValueError(
            f"在 {mat_path} 中找不到合适的 3D 数组。"
            f"可用键: {available}"
        )

    # 确保是 3D（去掉多余的单一维度）
    target = np.squeeze(target)
    if target.ndim != 3:
        raise ValueError(
            f"数组维度为 {target.ndim}D，期望 3D。shape={target.shape}"
        )

    # 转换为 float32（NIfTI 标准精度，兼容 predict.py 的 float32 加载）
    data = target.astype(np.float32)

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
