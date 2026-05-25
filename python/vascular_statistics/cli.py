"""
Click CLI — Vascular_Statistics 统一命令行入口。

用法示例：
    python -m vascular_statistics.cli convert graph.pajek
    python -m vascular_statistics.cli stats graph -v 0.078
    python -m vascular_statistics.cli skeletonize input.mat -o graph.pajek
    python -m vascular_statistics.cli pipeline input.mat -v 0.078
    python -m vascular_statistics.cli gui
"""

import os
import subprocess
import sys

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Vascular_Statistics — 血管网络骨架化与形态学统计。"""
    pass


@main.command()
@click.argument("pajek", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", default=".",
    help="输出目录（默认当前目录）",
)
def convert(pajek, output_dir):
    """将 Pajek 图转换为 C++ 统计代码所需的平面格式。

    PAJEK: VascGraph 输出的 .pajek / .net 文件。
    """
    from vascular_statistics.bridge import pajek_to_cpp_input

    stem = os.path.splitext(os.path.basename(pajek))[0]
    edges_out = os.path.join(output_dir, stem + "_edges.txt")
    vertices_out = os.path.join(output_dir, stem + "_vertices.txt")

    e_path, v_path = pajek_to_cpp_input(pajek, edges_out, vertices_out)
    click.echo(f"边文件:   {e_path}")
    click.echo(f"节点文件: {v_path}")


@main.command()
@click.argument("stem")
@click.option(
    "--volume", "-v", type=float, required=True,
    help="组织体积 (mm^3)",
)
@click.option(
    "--exe", type=click.Path(exists=True),
    default=None,
    help="C++ vessel_stats.exe 路径（默认在 build/ 中查找）",
)
def stats(stem, volume, exe):
    """对已转换的平面格式文件运行 C++ 统计。

    STEM: 边/节点文件的前缀（不含 _edges/_vertices 和 .txt 后缀）。
    """
    if exe is None:
        # 尝试在 build/ 下查找
        candidates = [
            "build/vessel_stats.exe",
            "build/vessel_stats",
            "build/Release/vessel_stats.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                exe = c
                break
        if exe is None:
            click.echo(
                "找不到 C++ 可执行文件。请用 -O2 编译后通过 --exe 指定路径，"
                "或运行: python scripts/build_cpp.py",
                err=True,
            )
            raise click.Abort()

    cmd = [exe, stem + "_edges", stem + "_vertices", str(volume)]
    click.echo(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        click.echo("C++ 统计执行失败。", err=True)
        raise click.Abort()
    click.echo("统计完成，输出文件已生成于当前目录。")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", default="graph.pajek", help="输出 Pajek 文件路径")
@click.option("--sampling", "-s", type=float, default=1.0, help="稀疏采样率 (1.0=最密)")
@click.option("--speed", type=float, default=0.05, help="收缩速度")
@click.option("--dist", type=float, default=0.5, help="距离项权重 (0-1)")
@click.option("--med", type=float, default=0.5, help="中轴项权重 (0-1)")
def skeletonize(input, output, sampling, speed, dist, med):
    """从 3D 分割图像生成血管骨架图。

    INPUT: .mat 或 .tif 格式的 3D 二值分割。
    """
    from vascular_statistics.vascgraph import GraphIO, Skeletonize
    ReadStackMat = GraphIO.ReadStackMat
    WritePajek = GraphIO.WritePajek
    Skeleton = Skeletonize.Skeleton

    # 加载分割
    if input.endswith(".mat"):
        stack = ReadStackMat(input).GetOutput()
    elif input.endswith((".tif", ".tiff")):
        import skimage.io as skio
        stack = skio.imread(input)
        stack = (stack > 0).astype(int)
    else:
        click.echo("不支持的输入格式。请使用 .mat 或 .tif 文件。", err=True)
        raise click.Abort()

    click.echo(f"已加载分割: {stack.shape}, 前景体素数: {stack.sum()}")

    # 骨架化
    sk = Skeleton(
        label=stack,
        sampling=sampling,
        speed_param=speed,
        dist_param=dist,
        med_param=med,
    )
    sk.Update()
    graph = sk.GetOutput()
    graph = fixG(graph)

    # 输出
    WritePajek(path="", name=output, graph=graph)
    click.echo(f"骨架图已保存: {output}")
    click.echo(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--volume", "-v", type=float, required=True, help="组织体积 (mm^3)")
@click.option("--output-stem", "-o", default=None, help="输出文件前缀")
def pipeline(input, volume, output_stem):
    """完整管线：骨架化 → 格式转换 → 统计。"""
    import tempfile
    from vascular_statistics.bridge import pajek_to_cpp_input

    if output_stem is None:
        output_stem = os.path.splitext(os.path.basename(input))[0]

    # 步骤 1：骨架化
    pajek_path = output_stem + ".pajek"

    # 通过 CLI 自身调用 skeletonize
    click.echo("=== 阶段 1/3: 骨架化 ===")
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(
        skeletonize,
        ["--output", pajek_path, input],
        catch_exceptions=False,
    )
    if result.exit_code != 0:
        click.echo("骨架化失败。", err=True)
        raise click.Abort()

    # 步骤 2：格式转换
    click.echo("=== 阶段 2/3: 格式转换 ===")
    edges_path = output_stem + "_edges.txt"
    vertices_path = output_stem + "_vertices.txt"
    pajek_to_cpp_input(pajek_path, edges_path, vertices_path)
    click.echo(f"已生成: {edges_path}, {vertices_path}")

    # 步骤 3：C++ 统计
    click.echo("=== 阶段 3/3: 统计 ===")
    result = runner.invoke(
        stats,
        ["--volume", str(volume), output_stem],
        catch_exceptions=False,
    )
    if result.exit_code != 0:
        click.echo("统计失败。", err=True)
        raise click.Abort()

    click.echo(f"管线完成。汇总文件: statistics_summary.txt")


@main.command()
def gui():
    """启动 VascGraph 3D 交互式编辑 GUI（需要 mayavi）。"""
    try:
        from vascular_statistics.vascgraph import GraphLab
        MainDialogue = GraphLab.MainDialogue
    except ImportError as e:
        click.echo(
            f"无法导入 GUI 组件: {e}\n"
            "请安装 GUI 依赖: pip install vascular_statistics[gui]\n"
            "或通过 conda: conda install -c conda-forge mayavi traits traitsui",
            err=True,
        )
        raise click.Abort()

    click.echo("启动 VascGraph GUI ...")
    MainDialogue()


if __name__ == "__main__":
    main()
