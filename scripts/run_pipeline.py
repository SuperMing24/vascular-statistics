"""
简化的端到端管线脚本 —— 一次性执行骨架化 + 格式转换 + 统计。
"""

import os
import subprocess
import sys


def run(input_path: str, volume: float, output_stem: str = "output") -> None:
    """完整管线：.mat 分割 → Pajek 骨架图 → C++ 统计。

    Args:
        input_path: .mat 或 .tif 分割文件路径。
        volume: 组织体积 (mm^3)。
        output_stem: 输出文件前缀。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "python"))

    from vascular_statistics.bridge import pajek_to_cpp_input
    from vascular_statistics.vascgraph import GraphIO, Skeletonize
    ReadStackMat = GraphIO.ReadStackMat
    WritePajek = GraphIO.WritePajek
    Skeleton = Skeletonize.Skeleton

    print("=== 阶段 1/3: 骨架化 ===")
    if input_path.endswith(".mat"):
        stack = ReadStackMat(input_path).GetOutput()
    elif input_path.endswith((".tif", ".tiff")):
        import skimage.io as skio
        stack = skio.imread(input_path)
        stack = (stack > 0).astype(int)
    else:
        raise ValueError(f"不支持的格式: {input_path}")

    print(f"  体素尺寸: {stack.shape}, 前景: {stack.sum()}")

    sk = Skeleton(label=stack, sampling=1.0, speed_param=0.05, dist_param=0.5, med_param=0.5)
    sk.Update()
    graph = fixG(sk.GetOutput())

    pajek_path = output_stem + ".pajek"
    WritePajek(path="", name=pajek_path, graph=graph)
    print(f"  骨架图: {pajek_path} ({graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边)")

    print("=== 阶段 2/3: 格式转换 ===")
    edges_path = output_stem + "_edges.txt"
    vertices_path = output_stem + "_vertices.txt"
    pajek_to_cpp_input(pajek_path, edges_path, vertices_path)
    print(f"  {edges_path}, {vertices_path}")

    print("=== 阶段 3/3: C++ 统计 ===")
    candidates = [
        os.path.join(project_root, "build/vessel_stats.exe"),
        os.path.join(project_root, "build/vessel_stats"),
        os.path.join(project_root, "build/Release/vessel_stats.exe"),
    ]
    exe = None
    for c in candidates:
        if os.path.exists(c):
            exe = c
            break
    if exe is None:
        # 尝试直接编译
        src_dir = os.path.join(project_root, "src")
        include_dir = os.path.join(project_root, "include")
        build_dir = os.path.join(project_root, "build")
        os.makedirs(build_dir, exist_ok=True)
        exe = os.path.join(build_dir, "vessel_stats.exe")
        compile_cmd = [
            "g++", "-O2",
            os.path.join(src_dir, "main.cpp"),
            os.path.join(src_dir, "data_io.cpp"),
            "-I", include_dir,
            "-o", exe,
        ]
        print(f"  编译 C++: {' '.join(compile_cmd)}")
        subprocess.run(compile_cmd, check=True)

    subprocess.run([exe, output_stem, output_stem, str(volume)], check=True)
    print(f"  完成。汇总: {output_stem}_statistics_summary.txt")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python run_pipeline.py <input.mat> <volume> [output_stem]")
        sys.exit(1)

    input_path = sys.argv[1]
    volume = float(sys.argv[2])
    stem = sys.argv[3] if len(sys.argv) >= 4 else "output"
    run(input_path, volume, stem)
