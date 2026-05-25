"""
批处理工具 —— 对多个输入文件执行批量骨架化或统计。
"""

import os
import glob
from typing import List, Optional


def batch_skeletonize(
    input_dir: str,
    pattern: str = "*.mat",
    output_dir: Optional[str] = None,
    **skeleton_kwargs,
) -> List[str]:
    """对目录中所有匹配的分割文件执行骨架化。

    Args:
        input_dir: 输入目录。
        pattern: 文件匹配模式 (glob)。
        output_dir: 输出目录（默认同输入目录）。
        **skeleton_kwargs: 传递给 Skeleton() 的参数。

    Returns:
        生成的 Pajek 文件路径列表。
    """
    from vascular_statistics.vascgraph import GraphIO, Skeletonize
    from VascGraph.Tools.CalcTools import fixG
    ReadStackMat = GraphIO.ReadStackMat
    WritePajek = GraphIO.WritePajek
    Skeleton = Skeletonize.Skeleton

    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    results = []

    for i, fpath in enumerate(files):
        basename = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(output_dir, basename + ".pajek")
        print(f"[{i+1}/{len(files)}] {basename} ...")

        stack = ReadStackMat(fpath).GetOutput()
        sk = Skeleton(label=stack, **skeleton_kwargs)
        sk.Update()
        graph = fixG(sk.GetOutput())
        WritePajek(path="", name=out_path, graph=graph)
        results.append(out_path)
        print(f"  → {out_path}  ({graph.number_of_nodes()} 节点)")

    return results


def batch_stats(
    pajek_dir: str,
    volume: float,
    pattern: str = "*.pajek",
    exe_path: Optional[str] = None,
) -> None:
    """对目录中所有 Pajek 文件执行统计。

    Args:
        pajek_dir: Pajek 文件所在目录。
        volume: 组织体积 (mm^3)。
        pattern: 文件匹配模式。
        exe_path: C++ 可执行文件路径。
    """
    import subprocess
    from vascular_statistics.bridge import pajek_to_cpp_input

    if exe_path is None:
        candidates = [
            "build/vessel_stats.exe",
            "build/vessel_stats",
        ]
        for c in candidates:
            if os.path.exists(c):
                exe_path = c
                break
        if exe_path is None:
            raise FileNotFoundError("找不到 C++ 可执行文件，请先编译。")

    files = sorted(glob.glob(os.path.join(pajek_dir, pattern)))

    for i, fpath in enumerate(files):
        stem = os.path.splitext(fpath)[0]
        basename = os.path.basename(stem)
        print(f"[{i+1}/{len(files)}] {basename} ...")

        edges_out = stem + "_edges.txt"
        vertices_out = stem + "_vertices.txt"
        pajek_to_cpp_input(fpath, edges_out, vertices_out)

        subprocess.run([exe_path, stem, stem, str(volume)], check=True)
        print(f"  → {stem}_statistics_summary.txt")
