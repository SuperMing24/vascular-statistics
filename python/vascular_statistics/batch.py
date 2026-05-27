"""
批处理工具 —— 对多个输入文件执行批量骨架化或统计，
以及输出目录结构管理（sample_key 计算 / manifest 注册 / pending 过滤）。
"""

import json
import os
import fcntl
import tempfile
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _save_manifest_locked(manifest: dict, manifest_path: str) -> None:
    """将 manifest 原子写入磁盘（调用方已持有文件锁）。"""
    parent = os.path.dirname(manifest_path)
    manifest["updated"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=parent or ".", prefix=".manifest_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
            json.dump(manifest, tmp, indent=2, ensure_ascii=False)
        os.replace(tmp_path, manifest_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def save_manifest(manifest: dict, manifest_path: str) -> None:
    """将 manifest 字典原子写入磁盘 JSON 文件（含文件锁防并发损坏）。"""
    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lock_path = manifest_path + ".lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _save_manifest_locked(manifest, manifest_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


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

    并发安全：通过 fcntl 文件锁保护读-改-写全过程，
    锁内重新加载 manifest 避免 TOCTOU 丢失更新。

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
    # 无 run_id / status 且条目已存在 → 快速路径，不加锁
    if (not run_id or not status) and os.path.exists(manifest_path):
        manifest = load_manifest(manifest_path)
        samples: dict = manifest.setdefault("samples", {})
        if sample_key in samples:
            return
        # 新样本需要注册 → 走加锁路径

    # 锁内：重新加载 → 修改 → 原子写入
    lock_path = manifest_path + ".lock"
    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # 锁内重新加载，获取最新状态
        manifest = load_manifest(manifest_path)
        samples = manifest.setdefault("samples", {})

        # 为新样本创建条目
        if sample_key not in samples:
            samples[sample_key] = {
                "input_path": input_path,
                "first_seen": start_time or datetime.now(
                    timezone(timedelta(hours=8))
                ).isoformat(),
                "total_runs": 0,
                "completed_runs": 0,
                "failed_runs": 0,
                "latest_run": None,
                "latest_status": "pending",
                "runs": [],
            }

        entry = samples[sample_key]

        # 仅注册样本存在（无 run_id / status）→ 返回
        if not run_id or not status:
            _save_manifest_locked(manifest, manifest_path)
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
            run_matched = False
            for run in reversed(entry["runs"]):
                if run.get("run_id") == run_id:
                    run["status"] = status
                    run["exit_code"] = exit_code
                    run["end_time"] = end_time
                    if start_time and end_time:
                        try:
                            st = datetime.fromisoformat(start_time)
                            et = datetime.fromisoformat(end_time)
                            run["duration_seconds"] = round(
                                (et - st).total_seconds()
                            )
                        except (ValueError, TypeError):
                            pass
                    run_matched = True
                    break

            if run_matched:
                # 仅在 run 匹配成功时更新样本级统计
                entry["latest_status"] = status
                entry["latest_run"] = run_id
                if status == "completed":
                    completed = sum(
                        1 for r in entry["runs"] if r.get("status") == "completed"
                    )
                    entry["completed_runs"] = completed
                elif status == "failed":
                    failed = sum(
                        1 for r in entry["runs"] if r.get("status") == "failed"
                    )
                    entry["failed_runs"] = failed
            else:
                # run_id 未匹配：可能 start-time 记录丢失，追加新记录
                fallback_record: dict = {
                    "run_id": run_id,
                    "job_id": job_id,
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                    "exit_code": exit_code,
                    "duration_seconds": None,
                    "node": node,
                    "commit": commit,
                }
                if start_time and end_time:
                    try:
                        st = datetime.fromisoformat(start_time)
                        et = datetime.fromisoformat(end_time)
                        fallback_record["duration_seconds"] = round(
                            (et - st).total_seconds()
                        )
                    except (ValueError, TypeError):
                        pass
                entry["runs"].append(fallback_record)
                entry["total_runs"] = len(entry["runs"])
                entry["latest_run"] = run_id
                entry["latest_status"] = status
                if status == "completed":
                    completed = sum(
                        1 for r in entry["runs"] if r.get("status") == "completed"
                    )
                    entry["completed_runs"] = completed
                elif status == "failed":
                    failed = sum(
                        1 for r in entry["runs"] if r.get("status") == "failed"
                    )
                    entry["failed_runs"] = failed

        # 重算全局统计（遍历全部样本确保准确）
        all_samples = list(samples.values())
        stats = manifest.setdefault("stats", {})
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

        _save_manifest_locked(manifest, manifest_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


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
# 骨架计数（从磁盘实际输出统计，不依赖 manifest）
# ═══════════════════════════════════════════════════════════════════════

def count_sample_skeletons(
    sample_key: str,
    output_root: str,
) -> int:
    """扫描输出目录，统计某样本已有多少个有效骨架（pajek 文件）。

    有效骨架 = run_* 目录中含 .pajek 文件。

    参数：
        sample_key: compute_sample_key 输出。
        output_root: 输出根目录。

    返回：
        该样本已有的有效骨架数。
    """
    sample_dir = os.path.join(output_root, sample_key)
    if not os.path.isdir(sample_dir):
        return 0

    count = 0
    for run_name in os.listdir(sample_dir):
        run_path = os.path.join(sample_dir, run_name)
        if not run_name.startswith("run_") or not os.path.isdir(run_path):
            continue
        if any(f.endswith(".pajek") for f in os.listdir(run_path)):
            count += 1
    return count


def filter_by_skeleton_count(
    input_files: List[str],
    data_root: str,
    output_root: str,
    *,
    min_skeletons: int = 0,
    max_skeletons: int = -1,
) -> Tuple[List[str], List[str], dict]:
    """按有效骨架数过滤输入文件。

    不从 manifest 读取（避免依赖损坏数据），直接从磁盘 run_* 目录扫描。

    参数：
        input_files: 输入文件的绝对路径列表。
        data_root: 数据根目录。
        output_root: 输出根目录。
        min_skeletons: 已有 >= 此数量的样本将被跳过（默认 0 = 不过滤）。
        max_skeletons: 已有 >= 此数量的样本将被跳过（-1 = 无上限）。

    返回：
        (to_submit, skipped, counts) —
        to_submit: 需要提交的文件列表。
        skipped: 已跳过的文件列表。
        counts: {sample_key: skeleton_count} 全部样本的骨架计数。
    """
    to_submit: List[str] = []
    skipped: List[str] = []
    counts: dict = {}

    for fpath in input_files:
        try:
            rel = os.path.relpath(fpath, data_root)
        except ValueError:
            # 无法计算相对路径 → 保留
            to_submit.append(fpath)
            continue
        key = compute_sample_key(rel)
        n = count_sample_skeletons(key, output_root)
        counts[key] = n

        if min_skeletons > 0 and n >= min_skeletons:
            skipped.append(fpath)
        elif max_skeletons >= 0 and n >= max_skeletons:
            skipped.append(fpath)
        else:
            to_submit.append(fpath)

    return to_submit, skipped, counts


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
