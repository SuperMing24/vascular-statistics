"""
简化的端到端管线脚本 —— 一次性执行骨架化 + 格式转换 + 统计。

支持两种输出模式：
  1. 传统模式（默认）：产物写入当前目录，使用 --output-stem 作为文件名前缀
  2. 面向样本模式（指定 --output-root）：产物写入面向样本的目录结构

面向样本模式示例：
    python scripts/run_pipeline.py input.tif 0.078 \\
        --output-root ./experiments --data-root ./datasets
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta


def run(
    input_path: str,
    volume: float,
    output_stem: str = "output",
    sampling: float = 1.0,
    phases: str = "all",
    *,
    output_root: str = "",
    data_root: str = "",
    anisotropic: bool = False,
    speed: float = 0.05,
) -> None:
    """完整管线：.mat/.tif 分割 → Pajek 骨架图 → C++ 统计。

    参数：
        input_path: .mat 或 .tif 分割文件路径。
        volume: 组织体积 (mm^3)。仅 phases=all/stats 时使用。
        output_stem: 输出文件前缀。
        sampling: 稀疏采样率 (1.0=最密)。
        phases: "all" | "skeletonize" | "stats"（分阶段执行）。
        output_root: 输出根目录。
        data_root: 数据根目录。
        anisotropic: 启用各向异性 spacing 物理单位（opt-in，从 sample_metadata.json 读取 voxel_spacing_um）。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "python"))

    from vascular_statistics.bridge import pajek_to_cpp_input, canonicalize_graph_pos
    from vascular_statistics.vascgraph import GraphIO, Skeletonize
    from VascGraph.Tools.CalcTools import fixG
    ReadStackMat = GraphIO.ReadStackMat
    WritePajek = GraphIO.WritePajek
    Skeleton = Skeletonize.Skeleton

    # --- 确定输出目录 ---
    actual_output_dir = "."  # CWD（传统模式）
    if output_root:
        # 面向样本模式：计算 sample_key → 生成 run_id → 创建目录
        from vascular_statistics.batch import compute_sample_key

        # 计算样本相对路径
        if data_root and input_path.startswith(data_root):
            rel_input = os.path.relpath(input_path, data_root)
        else:
            rel_input = input_path
        sample_key = compute_sample_key(rel_input)

        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_output_dir = os.path.join(output_root, sample_key, run_id)
        os.makedirs(actual_output_dir, exist_ok=True)

        # 面向样本模式下统一用 "skeleton" 前缀
        output_stem = "skeleton"

        print(f"样本键:  {sample_key}")
        print(f"运行 ID: {run_id}")
        print(f"输出目录: {actual_output_dir}")

    # 从原始目录开始（加载输入文件）
    original_cwd = os.getcwd()

    # ═══════════════════════════════════════════════════════════
    # 阶段 1：骨架化（skeletonize / all）
    # ═══════════════════════════════════════════════════════════
    if phases in ("skeletonize", "all"):
        print("=== 阶段 1/3: 骨架化 ===")
        if input_path.endswith(".mat"):
            stack = ReadStackMat(input_path).GetOutput()
        elif input_path.endswith((".tif", ".tiff")):
            import skimage.io as skio
            import numpy as np
            stack = skio.imread(input_path)
            # TIFF 多页格式为 [D, H, W]，骨架化期望 [H, W, D]
            if stack.ndim == 3:
                stack = np.transpose(stack, (1, 2, 0))
            stack = (stack > 0).astype(int)
        else:
            raise ValueError(f"不支持的格式: {input_path}")

        voxel_count = int(stack.sum())
        print(f"  体素尺寸: {stack.shape}, 前景: {voxel_count}")

        sk = Skeleton(label=stack, sampling=sampling,
                      speed_param=speed, dist_param=0.5, med_param=0.5)
        sk.Update()
        graph = fixG(sk.GetOutput())

        # 坐标轴序规范化：本脚本 .tif 转 [H,W,D]、.mat 也 [H,W,D]，故 pos 恒为 "yxz"，
        # 重排为物理 [x,y,z] 使 .pajek 规范（GUI/stats/金标准对比 全对）。半径不受影响。
        # 详见 docs/skeleton_axis_order_bug_20260629.md。
        canonicalize_graph_pos(graph, "yxz")

        # 切换到输出目录写入产物
        os.chdir(actual_output_dir)

        pajek_path = output_stem + ".pajek"
        WritePajek(path="", name=pajek_path, graph=graph)
        print(f"  骨架图: {pajek_path} ({graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边)")

        # 记录前景体素数 → 供后续体积计算
        voxel_path = output_stem + "_voxel_count.txt"
        with open(voxel_path, "w") as f:
            f.write(f"{voxel_count}\n")
        print(f"  体素计数: {voxel_path}")

    if phases == "skeletonize":
        print("管线停止（--phases skeletonize）。")
        os.chdir(original_cwd)
        return

    # ═══════════════════════════════════════════════════════════
    # 阶段 2：格式转换（stats / all）
    # ═══════════════════════════════════════════════════════════
    os.chdir(actual_output_dir)
    pajek_path = output_stem + ".pajek"
    if not os.path.exists(pajek_path):
        raise FileNotFoundError(
            f"找不到骨架图: {pajek_path}\n"
            f"请先运行 --phases skeletonize 生成骨架图。"
        )

    print("=== 阶段 2/3: 格式转换 ===")
    edges_path = output_stem + "_edges.txt"
    vertices_path = output_stem + "_vertices.txt"
    # .pajek 的 pos 已在骨架化阶段规范化为物理 [x,y,z]，故用 bridge 默认 axis_order="xyz"。
    pajek_to_cpp_input(pajek_path, edges_path, vertices_path)
    print(f"  {edges_path}, {vertices_path}")

    # ═══════════════════════════════════════════════════════════
    # 阶段 3：C++ 统计（stats / all）
    # ═══════════════════════════════════════════════════════════
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

    # --- 单位：各向异性或 legacy ---
    spacing_um = None
    if anisotropic and output_root:
        # 从样本的 sample_metadata.json 读取 voxel_spacing_um
        sample_meta_path = os.path.join(output_root, sample_key, "sample_metadata.json")
        if os.path.exists(sample_meta_path):
            with open(sample_meta_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
            spacing_um = sm.get("spatial", {}).get("voxel_spacing_um")
    if anisotropic and not spacing_um:
        import warnings
        warnings.warn("--anisotropic 但未找到 voxel_spacing_um，回退 legacy。"
                      "请先运行 extract-metadata。")
        anisotropic = False  # 回退

    cmd = [exe, output_stem + "_edges", output_stem + "_vertices", str(volume)]
    if anisotropic and spacing_um:
        cmd += [str(spacing_um[0]), str(spacing_um[1]), str(spacing_um[2])]
        print(f"  口径: 各向异性（spacing {spacing_um}）")
    else:
        print(f"  口径: legacy 各向同性（×2/×4）")
    subprocess.run(cmd, check=True)
    print(f"  完成。汇总: statistics_summary_d10+um.txt")

    # --- 收尾：面向样本模式下写入 run_meta.json ---
    if output_root:
        run_meta = {
            "run_id": run_id,
            "sample_key": sample_key,
            "input_path": os.path.abspath(input_path),
            "volume_mm3": volume,
            "sampling": sampling,
            "phases": phases,
            "start_time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "status": "completed",
            "stats_unit_mode": "anisotropic" if anisotropic else "legacy",
        }
        if spacing_um:
            run_meta["spacing_um"] = spacing_um
            run_meta["r_scale"] = (spacing_um[0] + spacing_um[1]) / 2.0
        meta_path = os.path.join(actual_output_dir, "run_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=False)
        print(f"  运行元数据: {meta_path}")

    # 恢复原始工作目录
    os.chdir(original_cwd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vascular_Statistics 端到端管线（骨架化 + 格式转换 + 统计）"
    )
    parser.add_argument("input", help="输入 .mat / .tif 文件路径")
    parser.add_argument("volume", type=float, help="组织体积 (mm^3)")
    parser.add_argument("output_stem", nargs="?", default="output", help="输出文件前缀（默认 output）")
    parser.add_argument("sampling", nargs="?", type=float, default=1.0, help="稀疏采样率（默认 1.0）")
    parser.add_argument("--phases", default="all",
                        choices=["all", "skeletonize", "stats"],
                        help="执行阶段（默认 all）")
    parser.add_argument("--output-root", "-o", default="", help="输出根目录（启用面向样本的输出结构）")
    parser.add_argument("--data-root", "-d", default="", help="数据根目录（面向样本模式下用于计算 sample_key）")
    parser.add_argument("--anisotropic", action="store_true", default=False,
                        help="启用各向异性 spacing 物理单位口径（从 sample_metadata.json 读取 voxel_spacing_um）")
    parser.add_argument("--speed", type=float, default=0.05,
                        help="收缩速度 speed_param（默认 0.05；大样本可用 0.2 加速）")

    args = parser.parse_args()

    run(
        input_path=args.input,
        volume=args.volume,
        output_stem=args.output_stem,
        sampling=args.sampling,
        phases=args.phases,
        output_root=args.output_root,
        data_root=args.data_root,
        anisotropic=args.anisotropic,
        speed=args.speed,
    )
