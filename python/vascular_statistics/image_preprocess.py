"""
图像域预处理（raw 强度图 → 分割就绪）。

与 segmentation/preprocess.py 的区别：
  - 本模块处理 **raw 强度域** 操作（去背景 / 百分位裁剪），在裁切之后、
    送入分割之前应用，结果写盘为 cropped_z。
  - segmentation/preprocess.py 是送入模型前的 **归一化策略**（MinMax/zscore），
    当前分割模型内部已逐栈 MinMax，故那一层默认 passthrough。

设计原则（2026-06-24 决策）：
  - 裁切（几何）与预处理（强度）解耦：先从 raw 按 frameRange 无损 z 切，
    再把预处理作为独立、可换、参数化的一步。
  - 当前分割模型（nnU-Net 2D seed22，cropped_xyz 41 例验证）训练/推理均仅
    逐栈全局 MinMax，无 z-score/percentile/去背景。本模块的候选预处理用于
    A/B 对比，择优后烘焙进 cropped_z。
"""

from typing import Tuple

import numpy as np


# ── Z 裁切（几何，无损）─────────────────────────────────────
def zcrop(stack: np.ndarray, z_start: int, z_end: int) -> np.ndarray:
    """按 1-based 闭区间 [z_start, z_end] 截取 Z 轴。

    与 deepRegionCro.m 的 frameRange=[sFrame,eFrame] 语义一致：
    raw[:, :, z_start-1 : z_end]。
    """
    if z_start < 1 or z_end > stack.shape[2] or z_start > z_end:
        raise ValueError(
            f"非法 z 范围 [{z_start},{z_end}]（stack Z={stack.shape[2]}）"
        )
    return stack[:, :, z_start - 1 : z_end]


# ── 预处理候选 ──────────────────────────────────────────────
def preprocess_none(vol: np.ndarray) -> np.ndarray:
    """不做强度预处理，保留 raw 值（交给模型内部 MinMax）。"""
    return vol


def preprocess_bgsub(vol: np.ndarray, kernel: int = 100) -> np.ndarray:
    """逐层去背景，复刻 deepRegionCro.m 行为。

    bg = kernel×kernel 均值核（零填充，等价 MATLAB filter2('same')）；
    img - bg；负值截 0。

    注意：硬截 0 有损——强度低于局部均值的暗/深血管会被抹掉。
    """
    from scipy.ndimage import uniform_filter

    out = np.zeros_like(vol, dtype=np.float32)
    for k in range(vol.shape[2]):
        img = vol[:, :, k].astype(np.float32)
        # mode='constant', cval=0 复刻 filter2 的零填充（边界仍除以满窗口）
        bg = uniform_filter(img, size=kernel, mode="constant", cval=0.0)
        diff = img - bg
        diff[diff < 0] = 0.0
        out[:, :, k] = diff
    return out


def preprocess_percentile(
    vol: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    """逐体积百分位裁剪（抑制双光子热像素），不做 MinMax。

    裁剪到 [p_low, p_high]；模型内部 MinMax 会把该区间映射到 [0,1]，
    等价于稳健的 percentile 归一化。与官方 nnU-Net 的 0.5/99.5 裁剪哲学一致。
    不硬截 0，保留暗/深血管的相对结构。
    """
    v = vol.astype(np.float32)
    lo = np.percentile(v, low_pct)
    hi = np.percentile(v, high_pct)
    if hi - lo < 1e-8:
        return v
    return np.clip(v, lo, hi)


_METHODS = {
    "none": preprocess_none,
    "bgsub": preprocess_bgsub,
    "percentile": preprocess_percentile,
}


def apply_preprocess(vol: np.ndarray, method: str) -> np.ndarray:
    """按名称分派预处理。method ∈ {none, bgsub, percentile}。"""
    if method not in _METHODS:
        raise ValueError(f"未知预处理: {method}。可选: {list(_METHODS)}")
    return _METHODS[method](vol)


def volume_intensity_summary(vol: np.ndarray) -> dict:
    """强度摘要，用于 A/B 对比诊断。"""
    v = vol.astype(np.float64)
    return {
        "shape": list(vol.shape),
        "dtype": str(vol.dtype),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "zero_frac": float((v == 0).mean()),
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
    }
