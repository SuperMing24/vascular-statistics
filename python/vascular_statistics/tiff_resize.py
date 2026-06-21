"""
TIFF 尺寸缩放核心 —— 生成原图的定制尺寸副本（XY 网格对齐用）。

用途：把原图（或任意 3D TIFF）缩放出匹配某重采样骨架/金标准网格的**副本**，
供配准/叠加，无需改动或重生成骨架。

安全保证：只读输入，**绝不删除或覆盖原图**；输出==输入时报错退出。

缩放约定（tifffile [D,H,W]）：默认只缩放 XY，Z 保持（各向异性单独处理）；
灰度原图用线性 order=1，二值/标签图用最近邻 order=0。

CLI: `vascular-stats resize-tiff`；独立脚本: `scripts/resize_tiff.py`。
"""
import os
from typing import Optional


def resize_tiff(
    src: str,
    dst: str,
    target_xy: Optional[int] = None,
    scale: Optional[float] = None,
    target_z: Optional[int] = None,
    order: int = 1,
) -> str:
    """读入 src，按目标缩放，写入 dst（不动 src）。返回 dst。"""
    import tifffile
    from scipy.ndimage import zoom

    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("输出路径与输入相同——拒绝覆盖原图。请指定不同的输出路径。")

    arr = tifffile.imread(src)
    if arr.ndim != 3:
        raise ValueError(f"仅支持 3D TIFF [D,H,W]，实际 ndim={arr.ndim} shape={arr.shape}")

    d, h, w = arr.shape

    if target_xy is not None:
        if h != w:
            print(f"  警告: H={h} != W={w}，按各自比例缩放到 {target_xy}", flush=True)
        zf_h = target_xy / float(h)
        zf_w = target_xy / float(w)
    elif scale is not None:
        zf_h = zf_w = float(scale)
    else:
        raise ValueError("须指定 target_xy 或 scale 之一")

    zf_z = (target_z / float(d)) if target_z is not None else 1.0

    out = zoom(arr, (zf_z, zf_h, zf_w), order=order).astype(arr.dtype)
    tifffile.imwrite(dst, out)

    kind = "最近邻/二值" if order == 0 else "线性" if order == 1 else f"spline-{order}"
    print(f"  {src}", flush=True)
    print(f"    [D,H,W] {arr.shape} (dtype={arr.dtype}) -> {out.shape}  order={order} ({kind})", flush=True)
    print(f"  -> {dst}", flush=True)
    return dst


def default_output(input_path: str, target_xy: Optional[int], scale: Optional[float]) -> str:
    """默认输出名 <stem>_resize<XY>.tiff（与源同目录）。"""
    stem, ext = os.path.splitext(input_path)
    tag = f"resize{target_xy}" if target_xy is not None else f"scale{scale}"
    return f"{stem}_{tag}{ext or '.tiff'}"


def resolve_order(order: Optional[int], binary: bool) -> int:
    """插值阶数：显式 order 优先，其次 binary→0，否则默认 1。"""
    if order is not None:
        return order
    return 0 if binary else 1
