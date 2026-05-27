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


def _resolve_volume_from_cwd() -> float | None:
    """从 CWD 向上查找 sample_metadata.json，读取 tissue_volume_mm3。"""
    import json as _json
    cwd = os.getcwd()
    # 尝试从当前目录向上最多 3 层查找 sample_metadata.json
    for _ in range(4):
        candidate = os.path.join(cwd, "sample_metadata.json")
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            vol = meta.get("spatial", {}).get("tissue_volume_mm3")
            if vol is not None:
                return float(vol)
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent
    return None


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
    "--volume", "-v", type=float, default=None,
    help="组织体积 (mm^3)。不提供则从 sample_metadata.json 自动查找。",
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
        # 基于项目根目录查找（不依赖 CWD）
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidates = [
            os.path.join(_project_root, "build/vessel_stats.exe"),
            os.path.join(_project_root, "build/vessel_stats"),
            os.path.join(_project_root, "build/Release/vessel_stats.exe"),
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
    from VascGraph.Tools.CalcTools import fixG
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

    click.echo(f"已加载分割: {stack.shape}, 前景体素数: {(stack > 0).sum()}")

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
@click.option("--volume", "-v", type=float, default=None, help="组织体积 (mm^3)。不提供则从样本元数据自动查找。")
@click.option("--output-stem", "-o", default=None, help="输出文件前缀")
@click.option("--sampling", "-s", type=float, default=1.0, help="稀疏采样率 (1.0=最密, 2.0=快速)")
@click.option(
    "--phases", default="all",
    type=click.Choice(["all", "skeletonize", "stats"]),
    help="执行哪些阶段。all=全流程; skeletonize=仅骨架化; stats=跳过骨架化(需已有 skeleton.pajek)")
def pipeline(input, volume, output_stem, phases, sampling):
    """完整管线：骨架化 → 格式转换 → 统计。

    用 --phases 可分阶段执行：

        --phases skeletonize  仅骨架化，输出 skeleton.pajek + voxel_count.txt
        --phases stats        跳过骨架化，从已有 skeleton.pajek 开始
        --phases all          全流程（默认）
    """
    from vascular_statistics.bridge import pajek_to_cpp_input
    from vascular_statistics.vascgraph import GraphIO, Skeletonize
    from VascGraph.Tools.CalcTools import fixG
    ReadStackMat = GraphIO.ReadStackMat
    WritePajek = GraphIO.WritePajek
    Skeleton = Skeletonize.Skeleton

    if output_stem is None:
        output_stem = os.path.splitext(os.path.basename(input))[0]

    # ═════════════════════════════════════════════════════════════
    # 阶段 1：骨架化（skeletonize / all）
    # ═════════════════════════════════════════════════════════════
    if phases in ("skeletonize", "all"):
        click.echo("=== 阶段 1/3: 骨架化 ===")
        if input.endswith(".mat"):
            stack = ReadStackMat(input).GetOutput()
        elif input.endswith((".tif", ".tiff")):
            import skimage.io as skio
            stack = skio.imread(input)
            stack = (stack > 0).astype(int)
        else:
            click.echo("不支持的输入格式。请使用 .mat 或 .tif 文件。", err=True)
            raise click.Abort()

        voxel_count = int((stack > 0).sum())
        click.echo(f"已加载分割: {stack.shape}, 前景体素数: {voxel_count}")

        sk = Skeleton(label=stack, sampling=sampling,
                      speed_param=0.05, dist_param=0.5, med_param=0.5)
        sk.Update()
        graph = fixG(sk.GetOutput())

        pajek_path = output_stem + ".pajek"
        WritePajek(path="", name=pajek_path, graph=graph)
        click.echo(f"骨架图已保存: {pajek_path}")
        click.echo(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")

        # 记录前景体素数 → 供后续体积计算
        voxel_path = output_stem + "_voxel_count.txt"
        with open(voxel_path, "w") as f:
            f.write(f"{voxel_count}\n")
        click.echo(f"体素计数: {voxel_path}")

    if phases == "skeletonize":
        click.echo("管线停止（--phases skeletonize）。")
        return

    # ═════════════════════════════════════════════════════════════
    # 阶段 2：格式转换（stats / all）
    # ═════════════════════════════════════════════════════════════
    click.echo("=== 阶段 2/3: 格式转换 ===")
    pajek_path = output_stem + ".pajek"
    if not os.path.exists(pajek_path):
        click.echo(f"找不到骨架图文件: {pajek_path}", err=True)
        click.echo("请先运行 --phases skeletonize 生成骨架图。", err=True)
        raise click.Abort()

    edges_path = output_stem + "_edges.txt"
    vertices_path = output_stem + "_vertices.txt"
    pajek_to_cpp_input(pajek_path, edges_path, vertices_path)
    click.echo(f"已生成: {edges_path}, {vertices_path}")

    # ═════════════════════════════════════════════════════════════
    # 阶段 3：C++ 统计（stats / all）
    # ═════════════════════════════════════════════════════════════
    click.echo("=== 阶段 3/3: 统计 ===")

    # 基于项目根目录查找 C++ 可执行文件（不依赖 CWD）
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    exe = None
    candidates = [
        os.path.join(_project_root, "build/vessel_stats.exe"),
        os.path.join(_project_root, "build/vessel_stats"),
        os.path.join(_project_root, "build/Release/vessel_stats.exe"),
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

    # --- Volume 自动解析 ---
    if volume is None:
        volume = _resolve_volume_from_cwd()
    if volume is None:
        click.echo(
            "无法确定组织体积。请提供 --volume 或先运行 extract-metadata。",
            err=True,
        )
        raise click.Abort()
    click.echo(f"体积: {volume} mm^3")

    cmd = [exe, output_stem + "_edges", output_stem + "_vertices", str(volume)]
    click.echo(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        click.echo("C++ 统计执行失败。", err=True)
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


# ═══════════════════════════════════════════════════════════════════════
# 输出结构管理命令（S3: update-manifest + status）
# ═══════════════════════════════════════════════════════════════════════

@main.command("update-manifest")
@click.option("--manifest", type=click.Path(), required=True, help="manifest.json 的完整路径")
@click.option("--sample-key", required=True, help="样本键 (compute_sample_key 输出)")
@click.option("--input-path", default="", help="输入文件相对 DATA_ROOT 的路径")
@click.option("--volume-mm3", type=float, default=0.0, help="组织体积 (mm^3)")
@click.option("--run-id", default="", help="运行 ID (如 run_20260526_013245)")
@click.option("--job-id", default="", help="Slurm 作业 ID")
@click.option("--status", default="", help="状态: pending | running | completed | failed")
@click.option("--exit-code", type=int, default=0, help="进程退出码")
@click.option("--start-time", default="", help="ISO 格式开始时间")
@click.option("--end-time", default="", help="ISO 格式结束时间")
@click.option("--node", default="", help="计算节点名")
@click.option("--commit", default="", help="代码 commit hash")
def update_manifest_cmd(
    manifest, sample_key, input_path, volume_mm3,
    run_id, job_id, status, exit_code,
    start_time, end_time, node, commit,
):
    """更新 manifest.json 中的运行记录。

    供 pipeline.slurm 在作业开始 / 结束时调用。
    每次调用至少需提供 --sample-key 与 --run-id / --status。

    示例（作业开始时注册 running）：

        python -m vascular_statistics.cli update-manifest \\\\
            --manifest \$OUTPUT_ROOT/manifest.json \\\\
            --sample-key "BCAS_1st/.../angiogram_crop_97_111" \\\\
            --run-id "run_20260526_013245" \\\\
            --status running \\\\
            --start-time "\$(date -Iseconds)"

    示例（作业结束时更新结果）：

        python -m vascular_statistics.cli update-manifest \\\\
            --manifest \$OUTPUT_ROOT/manifest.json \\\\
            --sample-key "BCAS_1st/.../angiogram_crop_97_111" \\\\
            --run-id "run_20260526_013245" \\\\
            --status completed --exit-code 0 \\\\
            --end-time "\$(date -Iseconds)"
    """
    from vascular_statistics.batch import update_manifest

    update_manifest(
        manifest_path=manifest,
        sample_key=sample_key,
        input_path=input_path,
        volume_mm3=volume_mm3,
        run_id=run_id,
        job_id=job_id,
        status=status,
        exit_code=exit_code,
        start_time=start_time,
        end_time=end_time,
        node=node,
        commit=commit,
    )
    click.echo(f"[{status or 'registered'}] {sample_key} → {manifest}")


@main.command("status")
@click.option("--manifest", type=click.Path(exists=True), required=True,
              help="manifest.json 的完整路径")
@click.option("--failed-only", "-f", is_flag=True,
              help="仅列出失败的样本")
@click.option("--pending-only", "-p", is_flag=True,
              help="仅列出待处理的样本")
def status_cmd(manifest, failed_only, pending_only):
    """查看全局管线处理进度。

    读取 manifest.json 并打印汇总统计。
    可通过 --failed-only / --pending-only 过滤。
    """
    import json

    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data.get("stats", {})
    samples: dict[str, dict] = data.get("samples", {})

    # 汇总统计
    click.echo("Vascular_Statistics Pipeline Status")
    click.echo("=" * 44)
    click.echo(f"  Total samples    : {stats.get('total_samples', len(samples))}")
    click.echo(f"  Completed        : {stats.get('completed_samples', 0)}")
    click.echo(f"  Failed           : {stats.get('failed_samples', 0)}")
    click.echo(f"  Pending          : {stats.get('pending_samples', 0)}")
    if data.get("updated"):
        click.echo(f"  Last update      : {data['updated']}")
    click.echo()

    # 明细列表
    if failed_only:
        click.echo("Failed samples:")
        for key, entry in sorted(samples.items()):
            if entry.get("latest_status") == "failed":
                click.echo(f"  FAIL  {key}")
    elif pending_only:
        click.echo("Pending samples:")
        for key, entry in sorted(samples.items()):
            if entry.get("latest_status") != "completed":
                click.echo(f"  PEND  {key}")
    else:
        # 默认：显示所有非 completed 样本
        pending_or_failed = [
            (k, e) for k, e in sorted(samples.items())
            if e.get("latest_status") != "completed"
        ]
        if pending_or_failed:
            click.echo(f"Non-completed samples ({len(pending_or_failed)}):")
            for key, entry in pending_or_failed:
                st = entry.get("latest_status", "?")
                runs = entry.get("total_runs", 0)
                click.echo(f"  [{st:11s}] {key}  ({runs} runs)")
        else:
            click.echo("All samples completed.")


@main.command("extract-metadata")
@click.option("--data-root", type=click.Path(exists=True), required=True,
              help="数据集根目录（含 .mat 文件）")
@click.option("--output-root", type=click.Path(), required=True,
              help="输出根目录（样本元数据写入此处）")
@click.option("--voxel-spacing-config", type=click.Path(exists=True), default=None,
              help="voxel_spacing.json 路径（可选，无则跳过 volume 计算）")
@click.option("--regen", is_flag=True,
              help="重新提取已存在的样本元数据")
@click.option("--glob", "glob_pattern", default="**/angiogram_crop_*.mat",
              help=".mat 文件匹配模式（默认 **/angiogram_crop_*.mat）")
def extract_metadata_cmd(data_root, output_root, voxel_spacing_config,
                         regen, glob_pattern):
    """从数据集 .mat 文件中提取每个样本的元数据。

    对每个 angiogram_crop_*.mat 文件：
      1. 解析路径 → group / animal_id / daypoint / z_range
      2. 加载 .mat → stack shape / foreground_voxel_count / rect_position
      3. 查找体素间距配置 → 计算 tissue_volume_mm3
      4. 写入 $OUTPUT_ROOT/{sample_key}/sample_metadata.json

    同时生成全局导航目录：
      $DATASET_ROOT/sample_catalog.json
    """
    from vascular_statistics.extract_metadata import run_extraction

    result = run_extraction(
        data_root=data_root,
        output_root=output_root,
        voxel_spacing_config_path=voxel_spacing_config,
        regen=regen,
        glob_pattern=glob_pattern,
    )

    click.echo()
    click.echo(f"已处理: {result['processed']}, "
               f"跳过: {result['skipped']}, "
               f"失败: {result['failed']}")
    click.echo(f"全局目录: {result['catalog_path']}")


if __name__ == "__main__":
    main()
