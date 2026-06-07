"""
强度预处理策略（可插拔）。

不同的归一化策略适用于不同的领域迁移场景。
当前默认策略为 "passthrough"（不做归一化），
因为 StandardExperimentFactory._infer_25d() 内部已有逐窗口 MinMax 归一化。

如果 Phase C 验证发现效果不佳，可切换策略：
  - percentile:  百分位裁剪后 MinMax → 抑制极端离群值
  - histogram:   直方图匹配到 MiniVess 分布 → 减小领域差异
  - clahe:       自适应直方图均衡 → 增强局部对比度
"""

import numpy as np
from typing import Optional, Tuple, Callable
from enum import Enum


class NormalizeStrategy(Enum):
    """归一化策略枚举。"""
    PASSTHROUGH = "passthrough"    # 不做处理
    MINMAX = "minmax"              # 全局 MinMax 到 [0, 1]
    PERCENTILE = "percentile"      # 百分位裁剪后 MinMax
    Z_SCORE = "zscore"             # Z-score 标准化


def _passthrough(vol: np.ndarray) -> np.ndarray:
    """不做任何归一化（默认策略）。"""
    return vol


def _minmax_normalize(vol: np.ndarray) -> np.ndarray:
    """MinMax 归一化到 [0, 1]。"""
    mn, mx = vol.min(), vol.max()
    if mx - mn < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - mn) / (mx - mn)).astype(np.float32)


def _percentile_normalize(
    vol: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    """百分位裁剪后 MinMax 归一化。

    抑制极端离群值（如双光子成像中的热像素），
    保留 99% 有效信号范围内的对比度。

    参数：
        low_pct: 下界百分位（默认 0.5）。
        high_pct: 上界百分位（默认 99.5）。
    """
    vmin = np.percentile(vol, low_pct)
    vmax = np.percentile(vol, high_pct)
    vol = np.clip(vol, vmin, vmax)
    return _minmax_normalize(vol)


def normalize_volume(
    vol: np.ndarray,
    strategy: str = "passthrough",
) -> np.ndarray:
    """对 3D 体积应用归一化策略。

    参数：
        vol: [H, W, D] float32 数组。
        strategy: 策略名（"passthrough" | "minmax" | "percentile" | "zscore"）。

    返回：
        归一化后的数组（float32）。
    """
    vol = vol.astype(np.float32, copy=False)

    if strategy == "passthrough":
        return _passthrough(vol)
    elif strategy == "minmax":
        return _minmax_normalize(vol)
    elif strategy == "percentile":
        return _percentile_normalize(vol)
    elif strategy == "zscore":
        mean, std = vol.mean(), vol.std()
        if std < 1e-8:
            return np.zeros_like(vol, dtype=np.float32)
        return ((vol - mean) / std).astype(np.float32)
    else:
        raise ValueError(f"未知归一化策略: {strategy}。")


# ══════════════════════════════════════════════════════════════════
# 体素统计工具（用于调试和策略选择）
# ══════════════════════════════════════════════════════════════════

def volume_stats(vol: np.ndarray) -> dict:
    """计算体数据的统计摘要。"""
    v = vol.astype(np.float64)
    nonzero = v[v > 0]
    return {
        "shape": vol.shape,
        "dtype": str(vol.dtype),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "nonzero_count": len(nonzero),
        "nonzero_ratio": float(len(nonzero) / v.size),
        "nonzero_min": float(nonzero.min()) if len(nonzero) > 0 else 0.0,
        "nonzero_max": float(nonzero.max()) if len(nonzero) > 0 else 0.0,
        "nonzero_mean": float(nonzero.mean()) if len(nonzero) > 0 else 0.0,
        "p1": float(np.percentile(v, 1)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
    }
