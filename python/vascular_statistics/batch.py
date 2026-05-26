"""
批处理工具 —— 对多个输入文件执行批量骨架化或统计，
以及输出目录结构管理（sample_key 计算 / manifest 注册 / pending 过滤）。
"""

import json
import os
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# 输出目录结构：阶段指示目录（在 sample_key 计算中被过滤）
#
# 这些目录名表示"数据处理阶段"而非"样本标识"，
# 不应出现在面向样本的输出路径中。列表按需扩展。
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_DIRS: set[str] = {
    "angiogram",      # 血管造影采集阶段
    "angiography",
    "segmentation",   # 分割阶段
    "seg",
    "images",         # 通用图像阶段
    "raw",            # 原始数据阶段
    "processed",      # 处理后阶段
    "interim",        # 中间产物阶段
    "results",        # 结果阶段
}


# ═══════════════════════════════════════════════════════════════════════
# 输出路径映射
# ═══════════════════════════════════════════════════════════════════════

def compute_sample_key(input_rel_path: str) -> str:
    """从相对输入路径计算样本键（去除阶段目录、去除扩展名）。

    算法：
      1. 拆分路径段
      2. 过滤 STAGE_DIRS 中的目录名（大小写不敏感）
      3. 最后一段去除扩展名
      4. 拼合为 POSIX 风格路径

    参数：
        input_rel_path: 相对于 DATA_ROOT 的输入文件路径。

    返回：
        规范化 sample_key（如 ``"BCAS_1st/20241009_A192_D0/angiogram_crop_97_111"``）。
        若过滤后无层级目录，归入 ``"_flat/"`` 前缀。

    示例：
        >>> compute_sample_key("BCAS_1st/20241009_A192_D0/angiogram/angiogram_crop_97_111.tif")
        "BCAS_1st/20241009_A192_D0/angiogram_crop_97_111"
        >>> compute_sample_key("sample2.tif")
        "_flat/sample2"
    """
    p = Path(input_rel_path)
    stem = p.stem  # 文件名去扩展名
    # 目录部分（不含文件名最后一节）
    parts = p.parts
    dirs = parts[:-1]
    # 过滤 STAGE_DIRS（大小写不敏感）
    filtered = [d for d in dirs if d.lower() not in STAGE_DIRS]
    filtered.append(stem)
    # 若过滤后仅剩文件名（无有效目录层级），归入 _flat 命名空间
    if len(filtered) <= 1:
        return "_flat/" + stem
    # 统一使用 POSIX 分隔符，保证 Linux/Windows 跨平台一致
    return "/".join(filtered)


# ═══════════════════════════════════════════════════════════════════════
# manifest.json 管理
# ═══════════════════════════════════════════════════════════════════════

def load_manifest(manifest_path: str) -> dict:
    """加载 manifest.json，若文件不存在则返回空骨架字典。"""
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": "1.0",
        "project": "Vascular_Statistics",
        "samples": {},
        "stats": {
            "total_samples": 0,
            "completed_samples": 0,
            "failed_samples": 0,
            "pending_samples": 0,
        },
    }


def save_manifest(manifest: dict, manifest_path: str) -> None:
    """将 manifest 字典写入磁盘 JSON 文件。"""
    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    manifest["updated"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def update_manifest(
    manifest_path: str,
    sample_key: str,
    *,
    input_path: str = "",
    volume_mm3: float = 0.0,
    run_id: str = "",
    job_id: str = "",
    status: str = "",
    exit_code: int = 0,
    start_time: str = "",
    end_time: str = "",
    node: str = "",
    commit: str = "",
) -> None:
    """更新 manifest.json 中某个样本的运行记录。

    由 ``pipeline.slurm`` 在作业开始/结束时调用，
    或由 ``batch_launch.sh`` 在提交前写入 pending 条目。

    参数：
        manifest_path: manifest.json 的完整路径。
        sample_key: 样本键（compute_sample_key 输出）。
        input_path: 输入文件相对 DATA_ROOT 的路径。
        volume_mm3: 组织体积。
        run_id: 运行 ID（如 ``"run_20260526_013245"``）。
        job_id: Slurm 作业 ID。
        status: ``"pending"`` | ``"running"`` | ``"completed"`` | ``"failed"``。
        exit_code: 进程退出码（结束时填写）。
        start_time: ISO 格式开始时间。
        end_time: ISO 格式结束时间。
        node: 计算节点名。
        commit: 代码 commit hash。
    """
    manifest = load_manifest(manifest_path)
    samples: dict = manifest.setdefault("samples", {})

    # 为新样本创建条目
    if sample_key not in samples:
        samples[sample_key] = {
            "input_path": input_path,
            "first_seen": start_time or datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "latest_run": None,
            "latest_status": "pending",
            "runs": [],
        }

    entry = samples[sample_key]

    # 无 run_id / status 的调用（如仅注册样本）→ 仅确保条目存在
    if not run_id or not status:
        save_manifest(manifest, manifest_path)
        return

    if status == "running":
        # 作业开始时追加新 run 记录
        run_record: dict = {
            "run_id": run_id,
            "job_id": job_id,
            "status": "running",
            "start_time": start_time,
            "exit_code": None,
            "duration_seconds": None,
            "node": node,
            "commit": commit,
        }
        entry["runs"].append(run_record)
        entry["total_runs"] = len(entry["runs"])
        entry["latest_run"] = run_id
        entry["latest_status"] = "running"
    else:
        # 作业结束时更新匹配 run_id 的记录
        for run in reversed(entry["runs"]):
            if run.get("run_id") == run_id:
                run["status"] = status
                run["exit_code"] = exit_code
                run["end_time"] = end_time
                if start_time and end_time:
                    try:
                        st = datetime.fromisoformat(start_time)
                        et = datetime.fromisoformat(end_time)
                        run["duration_seconds"] = round((et - st).total_seconds())
                    except (ValueError, TypeError):
                        pass
                break

        # 更新样本级统计
        entry["latest_status"] = status
        if status == "completed":
            # completed_runs: 避免重复累加（重跑同一 run 完成后多次调用）
            completed = sum(1 for r in entry["runs"] if r.get("status") == "completed")
            entry["completed_runs"] = completed
        elif status == "failed":
            failed = sum(1 for r in entry["runs"] if r.get("status") == "failed")
            entry["failed_runs"] = failed

    # 重算全局统计（遍历全部样本确保准确）
    all_samples = list(samples.values())
    stats: dict = manifest.setdefault("stats", {})
    stats["total_samples"] = len(all_samples)
    stats["completed_samples"] = sum(
        1 for s in all_samples if s.get("latest_status") == "completed"
    )
    stats["failed_samples"] = sum(
        1 for s in all_samples if s.get("latest_status") == "failed"
    )
    stats["pending_samples"] = sum(
        1 for s in all_samples
        if s.get("latest_status") in ("pending", "running", None)
    )

    save_manifest(manifest, manifest_path)


def filter_pending(
    input_files: List[str],
    manifest_path: str,
    data_root: str,
) -> List[str]:
    """过滤出尚未成功处理的输入文件。

    从 manifest.json 读取每个样本的 latest_status，
    仅保留状态不为 ``"completed"`` 的输入文件，
    用于 ``--resume`` 模式下的批量跳过。

    参数：
        input_files: 输入文件的绝对路径列表。
        manifest_path: manifest.json 的完整路径。
        data_root: 数据根目录（用于计算 sample_key）。

    返回：
        尚未 completed 的输入文件列表。
    """
    if not os.path.exists(manifest_path):
        return list(input_files)

    manifest = load_manifest(manifest_path)
    samples: dict = manifest.get("samples", {})

    pending: List[str] = []
    for fpath in input_files:
        try:
            rel = os.path.relpath(fpath, data_root)
        except ValueError:
            # 不同盘符等无法计算相对路径的情况 → 保留
            pending.append(fpath)
            continue
        key = compute_sample_key(rel)
        entry = samples.get(key, {})
        if entry.get("latest_status") != "completed":
            pending.append(fpath)

    return pending


# ═══════════════════════════════════════════════════════════════════════
# 批量处理（保留原有功能）
# ═══════════════════════════════════════════════════════════════════════

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
