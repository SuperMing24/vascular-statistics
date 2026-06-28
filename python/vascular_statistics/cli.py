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

import click


def _resolve_volume_from_cwd() -> float | None:
    """从 CWD 向上查找 sample_metadata.json，读取 tissue_volume_mm3。"""
    meta = _read_sample_meta_upwards()
    if meta is None:
        return None
    vol = meta.get("spatial", {}).get("tissue_volume_mm3")
    return float(vol) if vol is not None else None


def _resolve_spacing_from_cwd() -> list[float] | None:
    """从 CWD 向上查找 sample_metadata.json，读取 voxel_spacing_um（native 采集 spacing）。"""
    meta = _read_sample_meta_upwards()
    if meta is None:
        return None
    spacing = meta.get("spatial", {}).get("voxel_spacing_um")
    if spacing is not None and len(spacing) == 3:
        return [float(v) for v in spacing]
    return None


# ───────────────────────────────────────────────────────────────────────
# 方案 A：半径 μm 转换随骨架化网格联动
#
# 背景：血管半径在骨架化阶段以「该网格的像素」为单位计算（各向同性距离变换，
# 见 docs/radius_computation_resampling_analysis_20260628.md）。物理 μm 转换在
# C++ stats 经 r_scale=(sx+sy)/2 完成。**前提是 stats 拿到的 spacing 必须是骨架化
# 实际网格的 μm/体素**，而非 native 采集 spacing——否则半径偏差 (native_size/grid) 倍。
#
# 物理视场 FOV 与采样无关、是不变量：FOV[轴] = native_size × native_spacing。
# 重采样到网格 X 后：effective_spacing[轴] = FOV[轴] / X = native_spacing × native_size / X。
#
# 实现：骨架化阶段（此时实际网格已知）算出 effective spacing，写入**每运行**的旁车文件
# `<stem>_voxel_spacing_um.txt`（不写共享 sample_metadata.json——同一样本可在不同网格
# 骨架化，写共享字段会歧义）。stats 阶段优先读旁车，缺省回退 native（无重采样时二者相等）。
# ───────────────────────────────────────────────────────────────────────

def _run_spacing_path(stem: str) -> str:
    """每运行的有效 spacing 旁车文件路径。"""
    return stem + "_voxel_spacing_um.txt"


def _effective_spacing_from_grid(
    native_spacing: list[float],
    native_shape: list[int],
    current_shape: list[int],
) -> list[float]:
    """由骨架化实际网格反推有效体素 spacing（μm/体素）。

    effective_spacing[轴] = native_spacing[轴] × native_size[轴] / current_size[轴]。

    轴序对齐（关键，易错）：spacing 是 [x, y, z]，shape 是 [D, H, W] = [z, y, x]：
        x ↔ W = shape[2]，y ↔ H = shape[1]，z ↔ D = shape[0]。
    无重采样（current==native）时返回 native_spacing 本身。
    """
    axis_to_dim = (2, 1, 0)  # spacing 索引 [x,y,z] → shape 维度 [W,H,D]
    eff: list[float] = []
    for sp_idx, dim in enumerate(axis_to_dim):
        ratio = native_shape[dim] / current_shape[dim]
        eff.append(round(float(native_spacing[sp_idx]) * ratio, 6))
    return eff


def _write_run_spacing_sidecar(stem: str, current_shape: list[int]) -> list[float] | None:
    """骨架化阶段调用：算出本运行有效 spacing 并写旁车，返回该 spacing（无 metadata 则 None）。"""
    meta = _read_sample_meta_upwards()
    if meta is None:
        return None
    native_sp = meta.get("spatial", {}).get("voxel_spacing_um")
    native_shape = meta.get("stack_properties", {}).get("shape")
    if not native_sp or not native_shape or len(native_sp) != 3 or len(native_shape) != 3:
        return None
    eff = _effective_spacing_from_grid(native_sp, native_shape, list(current_shape))
    with open(_run_spacing_path(stem), "w", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in eff) + "\n")
    return eff


def _resolve_spacing(stem: str | None = None) -> list[float] | None:
    """解析有效 spacing：优先本运行旁车（已含网格校正），回退 native metadata。"""
    if stem is not None:
        sidecar = _run_spacing_path(stem)
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                parts = [p.strip() for p in f.read().strip().split(",")]
            if len(parts) == 3:
                return [float(v) for v in parts]
    return _resolve_spacing_from_cwd()


def _read_sample_meta_upwards() -> dict | None:
    """从 CWD 向上最多 3 层查找 sample_metadata.json。"""
    import json as _json
    cwd = os.getcwd()
    for _ in range(4):
        candidate = os.path.join(cwd, "sample_metadata.json")
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return _json.load(f)
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
@click.option(
    "--spacing", default=None,
    help="体素 spacing 'sx,sy,sz' (μm/体素)，启用各向异性物理单位口径；"
         "缺省走 legacy 各向同性(×2/×4)。",
)
def stats(stem, volume, exe, spacing):
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
    if spacing:
        parts = [p.strip() for p in spacing.split(",")]
        if len(parts) != 3:
            click.echo("--spacing 须为 'sx,sy,sz' 三个值。", err=True)
            raise click.Abort()
        cmd += parts  # sx sy sz → 各向异性模式
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
    from vascular_statistics.vascgraph import GraphIO, Skeletonize, Tools
    ReadStackMat = GraphIO.ReadStackMat
    WritePajek = GraphIO.WritePajek
    Skeleton = Skeletonize.Skeleton

    # 加载分割
    if input.endswith(".mat"):
        stack = ReadStackMat(input).GetOutput()
    elif input.endswith((".tif", ".tiff")):
        import skimage.io as skio
        import numpy as np
        stack = skio.imread(input)
        # TIFF 多页格式为 [D, H, W]，骨架化期望 [H, W, D]
        if stack.ndim == 3:
            stack = np.transpose(stack, (1, 2, 0))
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
    graph = Tools.CalcTools.fixG(graph)

    # 输出
    WritePajek(path="", name=output, graph=graph)
    click.echo(f"骨架图已保存: {output}")
    click.echo(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--volume", "-v", type=float, default=None, help="组织体积 (mm^3)。不提供则从样本元数据自动查找。")
@click.option("--output-stem", "-o", default=None, help="输出文件前缀")
@click.option("--sampling", "-s", type=float, default=1.0, help="稀疏采样率 (1.0=最密, 2.0=快速)")
@click.option("--speed", type=float, default=0.05,
              help="收缩速度 speed_param（Laplacian 位置吸引权重）。值越大收敛越快、迭代越少；"
                   "默认 0.05（保守，与历史一致）；大样本可用 0.2 加速。")
@click.option(
    "--phases", default="all",
    type=click.Choice(["all", "skeletonize", "stats"]),
    help="执行哪些阶段。all=全流程; skeletonize=仅骨架化; stats=跳过骨架化(需已有 skeleton.pajek)")
@click.option(
    "--anisotropic", is_flag=True, default=False,
    help="启用各向异性 spacing 物理单位口径。从 sample_metadata.json 读取 voxel_spacing_um。"
         "未设置则走 legacy 各向同性(×2/×4)，保持与历史输出一致。")
def pipeline(input, volume, output_stem, phases, sampling, speed, anisotropic):
    """完整管线：骨架化 → 格式转换 → 统计。

    用 --phases 可分阶段执行：

        --phases skeletonize  仅骨架化，输出 skeleton.pajek + voxel_count.txt
        --phases stats        跳过骨架化，从已有 skeleton.pajek 开始
        --phases all          全流程（默认）
    """
    from vascular_statistics.bridge import pajek_to_cpp_input
    from vascular_statistics.vascgraph import GraphIO, Skeletonize, Tools
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

        # 方案 A：按骨架化实际网格写有效 spacing 旁车（半径 μm 转换随网格联动）
        eff_sp = _write_run_spacing_sidecar(output_stem, list(stack.shape))
        if eff_sp is not None:
            click.echo(
                f"有效 spacing（网格 {list(stack.shape)} → FOV/grid）: {eff_sp} μm/体素 "
                f"→ {_run_spacing_path(output_stem)}"
            )

        sk = Skeleton(label=stack, sampling=sampling,
                      speed_param=speed, dist_param=0.5, med_param=0.5)
        sk.Update()
        graph = Tools.CalcTools.fixG(sk.GetOutput())

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
    if anisotropic:
        # 优先读骨架化阶段写入的有效 spacing 旁车（含网格校正），缺省回退 native
        spacing = _resolve_spacing(output_stem)
        if spacing is None:
            click.echo(
                "--anisotropic 须有 sample_metadata.json（含 voxel_spacing_um）。"
                "请先运行 extract-metadata。",
                err=True,
            )
            raise click.Abort()
        cmd += [str(spacing[0]), str(spacing[1]), str(spacing[2])]
        click.echo(f"口径: 各向异性（spacing {spacing} μm/体素）")
    else:
        click.echo("口径: legacy 各向同性（×2/×4）")
    click.echo(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        click.echo("C++ 统计执行失败。", err=True)
        raise click.Abort()

    click.echo(f"管线完成。汇总文件: statistics_summary_d10+um.txt")


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
# 血管分割桥接命令（Phase B — 2026-06-07）
# ═══════════════════════════════════════════════════════════════════════

@main.command("segment")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", default=".",
    help="输出目录（默认当前目录）",
)
@click.option(
    "--normalize", "-n",
    type=click.Choice(["passthrough", "minmax", "percentile", "zscore"]),
    default="passthrough",
    help="强度归一化策略（默认 passthrough=不做处理，依赖模型内部逐窗口MinMax）",
)
@click.option(
    "--threshold", "-t", type=float, default=0.5,
    help="二值化阈值（默认 0.5）",
)
@click.option(
    "--min-size", type=int, default=0,
    help="最小连通域体素数（0=不过滤，建议 100-500）",
)
@click.option(
    "--keep-nii", is_flag=True,
    help="保留中间 .nii 文件（默认推理完成后自动删除）",
)
@click.option(
    "--device", "-d",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="推理设备（默认 cuda，需 GPU 节点）",
)
@click.option(
    "--stats", "show_stats", is_flag=True,
    help="推理完成后打印掩码质量统计",
)
def segment_cmd(input, output_dir, normalize, threshold, min_size,
                keep_nii, device, show_stats):
    """对双光子原图 .mat 文件进行血管分割，输出二值掩码 .tif。

    INPUT: .mat 文件（含 'stack' 键的 3D 双光子荧光成像）。

    使用 nnU-Net 2D（MiniVess 验证集最优模型，Dice=0.856 clDice=0.593）
    进行推理。输出可直接用于 skeletonize 命令。

    示例：

        # 基本用法
        vascular-stats segment angiogram_crop_93_163.mat -o ./seg_output

        # 带百分位归一化（抑制极端离群值）
        vascular-stats segment angiogram_crop_93_163.mat -o ./seg_output -n percentile

        # 去除 <200 体素的小连通域
        vascular-stats segment angiogram_crop_93_163.mat -o ./seg_output --min-size 200 --stats
    """
    from vascular_statistics.segmentation import segment_mat, load_mask, mask_stats

    click.echo(f"输入: {input}")
    click.echo(f"归一化策略: {normalize}")
    click.echo(f"模型: nnU-Net 2D (seed22, epoch 41)")

    try:
        tif_path = segment_mat(
            mat_path=input,
            output_dir=output_dir,
            normalize=normalize,
            threshold=threshold,
            min_component_size=min_size,
            keep_nii=keep_nii,
            device=device,
        )
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        click.echo(
            "\n提示：推理需要在服务器端运行（需要 GPU + Vascular_Extraction 环境）。\n"
            "本地开发机不支持此命令。",
            err=True,
        )
        raise click.Abort()
    except Exception as e:
        click.echo(f"分割失败: {e}", err=True)
        raise click.Abort()

    click.echo(f"二值掩码: {tif_path}")

    if show_stats:
        mask = load_mask(tif_path)
        stats = mask_stats(mask)
        click.echo(f"  形状: {stats['shape']}")
        click.echo(f"  前景体素: {stats['foreground_voxels']:,} "
                    f"({stats['foreground_ratio']:.2%})")
        if "num_connected_components" in stats:
            click.echo(f"  连通域数: {stats['num_connected_components']:,}")
            click.echo(f"  最大连通域: {stats['largest_component_size']:,} 体素")
            click.echo(f"  平均连通域: {stats['mean_component_size']:.0f} 体素")


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


@main.command("aggregate-stats")
@click.option("--output-root", type=click.Path(exists=True), required=True,
              help="输出根目录（含样本子目录）")
@click.option("--sample-key", default=None,
              help="仅处理指定样本（缺省则全部）")
def aggregate_stats_cmd(output_root, sample_key):
    """汇总同一样本多次骨架化运行的形态学统计。

    遍历每个样本目录下所有 run_*/statistics_summary_d10+um.txt，
    计算 4 项指标的均值 ± 标准差，
    将结果写入样本目录下的 statistics_summary_d10+um.txt（与 sample_metadata.json 同级）。

    示例：
      vascular-stats aggregate-stats --output-root /share/home/sukm/experiments/vascstats
      vascular-stats aggregate-stats --output-root ... --sample-key "BCAS_1st/20241009_A192_D0/angiogram_crop_97_111"
    """
    from vascular_statistics.aggregate_stats import run_aggregation

    sample_keys = [sample_key] if sample_key else None
    result = run_aggregation(output_root, sample_keys=sample_keys)

    click.echo()
    click.echo(f"已处理: {result['processed']}, "
               f"跳过: {result['skipped']}, "
               f"失败: {result['failed']}")


# ═══════════════════════════════════════════════════════════════════════
# 子范围统计命令（默认 0-10 um）。旧名 micro-* 保留为别名。
# ═══════════════════════════════════════════════════════════════════════

from vascular_statistics.subrange_stats import DIAMETER_SUFFIX

@main.command("diameter-stats")
@click.option("--output-root", type=click.Path(exists=True), required=True,
              help="输出根目录（含样本子目录）")
@click.option("--sample-key", default=None,
              help="仅处理指定样本（缺省则全部已完成骨架化的样本）")
def diameter_stats_cmd(output_root, sample_key):
    """为所有已完成骨架化的 run 生成子范围（0-10 um）统计。

    从已有 generate_vessel_radius_d10+um.txt / generate_vessel_path_length_d10+um.txt /
    generate_vessel_tortuosity_d10+um.txt 中提取直径 < 10 um 的血管段，
    写入新的 _d0-10um 后缀文件，不覆盖原有统计。

    幂等：已有 statistics_summary_d0-10um.txt 的 run 将自动跳过。

    示例：
      vascular-stats diameter-stats --output-root /share/home/sukm/experiments/vascstats
      vascular-stats diameter-stats --output-root ... --sample-key "BCAS_1st/..."
    """
    from vascular_statistics.subrange_stats import run_subrange_stats

    sample_keys = [sample_key] if sample_key else None
    result = run_subrange_stats(output_root, sample_keys=sample_keys)

    click.echo()
    click.echo(f"已处理: {result['processed']}, "
               f"跳过(已有): {result['skipped']}, "
               f"无符合条件的段: {result['no_result']}, "
               f"失败: {result['failed']}")


# 别名：旧名 micro-stats
@main.command("micro-stats", hidden=True)
@click.option("--output-root", type=click.Path(exists=True), required=True)
@click.option("--sample-key", default=None)
def micro_stats_cmd(output_root, sample_key):
    """[已弃用] 请使用 diameter-stats。"""
    return diameter_stats_cmd(output_root, sample_key)


@main.command("aggregate-diameter-stats")
@click.option("--output-root", type=click.Path(exists=True), required=True,
              help="输出根目录（含样本子目录）")
@click.option("--sample-key", default=None,
              help="仅处理指定样本（缺省则全部）")
def aggregate_diameter_stats_cmd(output_root, sample_key):
    """汇总同一样本多次骨架化运行的子范围（0-10 um）统计。

    遍历每个样本目录下所有 run_*/statistics_summary_d0-10um.txt，
    计算 4 项指标的均值 +/- 标准差，
    将结果写入样本目录下的 statistics_summary_d0-10um.txt。

    示例：
      vascular-stats aggregate-diameter-stats --output-root /share/home/sukm/experiments/vascstats
      vascular-stats aggregate-diameter-stats --output-root ... --sample-key "BCAS_1st/..."
    """
    from vascular_statistics.aggregate_stats import run_aggregation

    sample_keys = [sample_key] if sample_key else None
    result = run_aggregation(output_root, sample_keys=sample_keys, suffix=DIAMETER_SUFFIX)

    click.echo()
    click.echo(f"已处理: {result['processed']}, "
               f"跳过: {result['skipped']}, "
               f"失败: {result['failed']}")


# 别名：旧名 aggregate-micro-stats
@main.command("aggregate-micro-stats", hidden=True)
@click.option("--output-root", type=click.Path(exists=True), required=True)
@click.option("--sample-key", default=None)
def aggregate_micro_stats_cmd(output_root, sample_key):
    """[已弃用] 请使用 aggregate-diameter-stats。"""
    return aggregate_diameter_stats_cmd(output_root, sample_key)


@main.command("diameter-summary")
@click.option("--output-root", type=click.Path(exists=True), required=True,
              help="输出根目录（含样本子目录）")
@click.option("--sample-key", default=None,
              help="仅包含指定样本（缺省则全部）")
def diameter_summary_cmd(output_root, sample_key):
    """生成跨样本次范围统计汇总（0-10 um）。

    扫描所有样本的 statistics_summary_d0-10um.txt（需先运行 diameter-stats +
    aggregate-diameter-stats），汇总为单一 cross_sample_summary_d0-10um.txt。

    示例：
      vascular-stats diameter-summary --output-root /share/home/sukm/experiments/vascstats
    """
    from vascular_statistics.subrange_stats import generate_cross_sample_summary

    sample_keys = [sample_key] if sample_key else None
    result = generate_cross_sample_summary(output_root, sample_keys=sample_keys)

    if result:
        click.echo(f"跨样本次范围汇总已写入: {result}")
    else:
        click.echo("无有效子范围统计数据，未生成汇总文件。", err=True)


# 别名：旧名 micro-summary
@main.command("micro-summary", hidden=True)
@click.option("--output-root", type=click.Path(exists=True), required=True)
@click.option("--sample-key", default=None)
def micro_summary_cmd(output_root, sample_key):
    """[已弃用] 请使用 diameter-summary。"""
    return diameter_summary_cmd(output_root, sample_key)


# ═══════════════════════════════════════════════════════════════════════
# TIFF 尺寸缩放（生成原图定制尺寸副本，不动原图）
# ═══════════════════════════════════════════════════════════════════════

@main.command("resize-tiff")
@click.argument("input", type=click.Path(exists=True))
@click.option("--xy", type=int, default=None, help="目标 XY 像素数（如 382）")
@click.option("--scale", type=float, default=None, help="XY 缩放比例（与 --xy 二选一）")
@click.option("--z", "z", type=int, default=None, help="目标 Z 层数（默认不缩放 Z）")
@click.option("--binary", is_flag=True, default=False,
              help="二值/标签图：最近邻 order=0（默认 order=1 线性，适合灰度原图）")
@click.option("--order", type=int, default=None, help="插值阶数（覆盖默认）：0/1/3")
@click.option("-o", "--output", default=None,
              help="输出路径（默认 <stem>_resize<XY>.tiff，与源同目录）")
def resize_tiff_cmd(input, xy, scale, z, binary, order, output):
    """生成 TIFF 的定制尺寸副本（XY 网格对齐用，不动原图）。

    用于把原图缩放到与某重采样骨架/金标准相同的网格，供配准/叠加，
    无需改动或重生成骨架。仅缩放 XY，Z 默认保持。

    示例：
      vascular-stats resize-tiff img.tiff --xy 382
      vascular-stats resize-tiff seg.tiff --xy 382 --binary
    """
    from vascular_statistics.tiff_resize import (
        resize_tiff, default_output, resolve_order,
    )

    if xy is None and scale is None:
        raise click.UsageError("须指定 --xy 或 --scale")

    _order = resolve_order(order, binary)
    out = output or default_output(input, xy, scale)

    try:
        resize_tiff(input, out, target_xy=xy, scale=scale, target_z=z, order=_order)
    except (ValueError, OSError) as e:
        click.echo(f"错误: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
