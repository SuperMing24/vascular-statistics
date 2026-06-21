"""
TIFF 尺寸缩放脚本 —— 生成原图的定制尺寸副本（XY 网格对齐用，不动原图）。

核心逻辑在 vascular_statistics.tiff_resize；本脚本是独立 CLI 包装。
等价 CLI 子命令：`vascular-stats resize-tiff`。

用法：
  python scripts/resize_tiff.py img.tiff --xy 382            # 灰度原图 512->382 副本
  python scripts/resize_tiff.py seg.tiff --xy 382 --binary   # 二值图最近邻
  python scripts/resize_tiff.py img.tiff --scale 0.746       # 按比例
  python scripts/resize_tiff.py img.tiff --xy 382 --z 300    # 同时缩放 Z
  python scripts/resize_tiff.py img.tiff --xy 382 -o out.tiff

安全：只读输入，绝不删除/覆盖原图；输出==输入时报错退出。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from vascular_statistics.tiff_resize import resize_tiff, default_output, resolve_order


def main() -> None:
    p = argparse.ArgumentParser(description="生成原图的定制尺寸副本（不动原图）")
    p.add_argument("input", help="输入 TIFF（只读，不会被修改）")
    p.add_argument("--xy", type=int, default=None, help="目标 XY 像素数（如 382）")
    p.add_argument("--scale", type=float, default=None, help="XY 缩放比例（与 --xy 二选一）")
    p.add_argument("--z", type=int, default=None, help="目标 Z 层数（默认不缩放 Z）")
    p.add_argument("--binary", action="store_true",
                   help="二值/标签图：最近邻 order=0（默认 order=1 线性）")
    p.add_argument("--order", type=int, default=None, help="插值阶数（覆盖默认）：0/1/3")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（默认 <stem>_resize<XY>.tiff）")
    args = p.parse_args()

    if args.xy is None and args.scale is None:
        p.error("须指定 --xy 或 --scale")

    order = resolve_order(args.order, args.binary)
    output = args.output or default_output(args.input, args.xy, args.scale)

    try:
        resize_tiff(args.input, output, target_xy=args.xy, scale=args.scale,
                    target_z=args.z, order=order)
    except (ValueError, OSError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
