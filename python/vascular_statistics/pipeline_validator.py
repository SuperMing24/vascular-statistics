"""
管线验证引擎 —— 五维检查体系（S/Se/C/Cs/N）。

每个验证函数返回 (passed: bool, message: str, details: dict)。
passed=False 表示检查未通过，message 为人类可读说明，details 含具体数据。

设计原则：
  - 非阻塞：validation 失败不抛异常，返回 False + 说明（由调用方决定是否 abort）
  - 可配置：生理范围从 configs/delta/validation_ranges.json 读取
  - 无状态：每个函数纯输入→输出，便于测试和组合
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# ───────────────────────────────────────────────────────────────────────
# 工具函数
# ───────────────────────────────────────────────────────────────────────


def _load_ranges(config_path: str | None = None) -> dict[str, Any]:
    """加载生理范围配置。"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "delta", "validation_ranges.json",
        )
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    # 内置默认（鼠脑血管）
    return {
        "mouse_brain": {
            "diameter_um": {"min": 5, "max": 50},
            "segment_length_um": {"min": 10, "max": 200},
            "tortuosity": {"min": 1.0, "max": 1.5},
            "segment_count_min": 1,
        }
    }


def _parse_stats_file(path: str) -> dict[str, float] | None:
    """解析 statistics_summary_d10+um.txt 提取关键指标。"""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    metrics: dict[str, float] = {}
    patterns = {
        "avg_diameter": r"平均直径 \(μm\):\s*([\d.]+)",
        "avg_length": r"平均长度 \(μm\):\s*([\d.]+)",
        "avg_tortuosity": r"平均弯曲度[^:]*:\s*([\d.]+)",
        "segment_density": r"段密度[^:]*:\s*([\d.]+)",
        "segment_count": r"有效段数:\s*(\d+)",
        "volume_mm3": r"体积[^:]*:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics if metrics else None


def _find_run_meta(run_dir: str) -> dict[str, Any] | None:
    """读取 run_meta.json。"""
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ───────────────────────────────────────────────────────────────────────
# S — 结构检查
# ───────────────────────────────────────────────────────────────────────


def validate_structural_run(run_dir: str) -> tuple[bool, str, dict]:
    """S3+S4: 检查单个 run 目录的结构完整性。"""
    details: dict[str, Any] = {"run_dir": run_dir}
    issues: list[str] = []

    # S3: 目录名格式
    dirname = os.path.basename(run_dir.rstrip("/\\"))
    if not re.match(r"^run_\d{8}_\d{6}$", dirname):
        issues.append(f"目录名不符合 run_YYYYMMDD_HHMMSS: {dirname}")

    # S4: 必需文件
    required = ["skeleton.pajek", "run_meta.json"]
    for f in required:
        fp = os.path.join(run_dir, f)
        if not os.path.exists(fp):
            issues.append(f"缺失文件: {f}")
        elif os.path.getsize(fp) == 0:
            issues.append(f"空文件: {f}")

    # skeleton.pajek 非空
    pajek = os.path.join(run_dir, "skeleton.pajek")
    if os.path.exists(pajek):
        details["pajek_size"] = os.path.getsize(pajek)

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "结构检查通过", details


def validate_structural_presubmit(seg_root: str) -> tuple[bool, str, dict]:
    """S1+S2: 提交前检查 seg 目录结构和 metadata 完整性。"""
    issues: list[str] = []
    details: dict[str, Any] = {"seg_root": seg_root}

    # S1: 无重复层级 (.tiff 文件不在同名目录中)
    for dirpath, _dirnames, filenames in os.walk(seg_root):
        for f in filenames:
            if f.endswith(".tiff"):
                parent = os.path.basename(dirpath)
                stem = f.replace(".tiff", "")
                if stem == parent:
                    issues.append(f"S1 重复层级: {os.path.join(dirpath, f)}")

    details["tiff_count"] = sum(
        1 for _d, _n, fs in os.walk(seg_root) for f in fs if f.endswith(".tiff")
    )
    details["duplicated"] = len(issues)

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "结构检查通过", details


# ───────────────────────────────────────────────────────────────────────
# Se — 语义检查（关键）
# ───────────────────────────────────────────────────────────────────────


def validate_semantic(run_dir: str, species: str = "mouse_brain") -> tuple[bool, str, dict]:
    """Se1–Se5: 检查统计输出是否在生理范围内。"""
    ranges = _load_ranges().get(species, {})
    stats_path = os.path.join(run_dir, "statistics_summary_d10+um.txt")
    issues: list[str] = []
    details: dict[str, Any] = {}

    metrics = _parse_stats_file(stats_path)
    if metrics is None:
        return False, f"无法解析统计文件: {stats_path}", details

    details["metrics"] = metrics

    # Se1: 直径
    if "avg_diameter" in metrics:
        r = ranges.get("diameter_um", {})
        lo, hi = r.get("min", 5), r.get("max", 50)
        v = metrics["avg_diameter"]
        if not (lo <= v <= hi):
            issues.append(f"Se1 平均直径 {v:.1f} μm 超出范围 [{lo}, {hi}]")

    # Se2: 段长
    if "avg_length" in metrics:
        r = ranges.get("segment_length_um", {})
        lo, hi = r.get("min", 10), r.get("max", 200)
        v = metrics["avg_length"]
        if not (lo <= v <= hi):
            issues.append(f"Se2 平均段长 {v:.1f} μm 超出范围 [{lo}, {hi}]")

    # Se3: 弯曲度
    if "avg_tortuosity" in metrics:
        r = ranges.get("tortuosity", {})
        lo, hi = r.get("min", 1.0), r.get("max", 1.5)
        v = metrics["avg_tortuosity"]
        if not (lo <= v <= hi):
            issues.append(f"Se3 平均弯曲度 {v:.4f} 超出范围 [{lo}, {hi}]")

    # Se4: 段密度 > 0
    if "segment_density" in metrics:
        if metrics["segment_density"] <= 0:
            issues.append("Se4 段密度 ≤ 0")
        if metrics["segment_density"] > 10000:
            issues.append(f"Se4 段密度异常高: {metrics['segment_density']:.0f}")

    # Se5: Pajek 骨架非空
    pajek = os.path.join(run_dir, "skeleton.pajek")
    if os.path.exists(pajek):
        with open(pajek, encoding="utf-8") as f:
            header = f.readline()
        # *vertices N
        m = re.match(r"\*vertices\s+(\d+)", header)
        if m:
            n_nodes = int(m.group(1))
            details["n_nodes"] = n_nodes
            if n_nodes == 0:
                issues.append("Se5 骨架无节点")
        else:
            issues.append("Se5 Pajek 格式异常：无法解析节点数")

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "语义检查通过", details


# ───────────────────────────────────────────────────────────────────────
# C — 完整性检查
# ───────────────────────────────────────────────────────────────────────


def validate_completeness(exp_root: str) -> tuple[bool, str, dict]:
    """C1+C2: 检查 metadata 与 seg 的对齐，以及 manifest 完整性。"""
    issues: list[str] = []
    details: dict[str, Any] = {}

    manifest_path = os.path.join(exp_root, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        stats = manifest.get("stats", {})
        details["manifest"] = {
            "total": stats.get("total_samples", 0),
            "completed": stats.get("completed_samples", 0),
            "failed": stats.get("failed_samples", 0),
        }
        if stats.get("failed_samples", 0) > 0:
            issues.append(f"C2 {stats['failed_samples']} 样本失败")
        pending = stats.get("total_samples", 0) - stats.get("completed_samples", 0) - stats.get(
            "failed_samples", 0
        )
        if pending > 0:
            issues.append(f"C2 {pending} 样本未完成")
    else:
        issues.append("C2 manifest.json 不存在")

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "完整性检查通过", details


# ───────────────────────────────────────────────────────────────────────
# Cs — 一致性检查（关键）
# ───────────────────────────────────────────────────────────────────────


def validate_consistency(sample_dir: str) -> tuple[bool, str, dict]:
    """Cs1–Cs3: 同一样本所有 run 的参数一致性。"""
    issues: list[str] = []
    details: dict[str, Any] = {"sample_dir": sample_dir, "runs": []}

    runs: list[dict] = []
    for entry in sorted(os.listdir(sample_dir)):
        run_path = os.path.join(sample_dir, entry)
        if not os.path.isdir(run_path) or not entry.startswith("run_"):
            continue
        meta = _find_run_meta(run_path)
        if meta is None:
            runs.append({"run_id": entry, "error": "缺失 run_meta.json"})
            issues.append(f"Cs: {entry} 无 run_meta.json")
            continue
        runs.append({
            "run_id": entry,
            "radius_mode": meta.get("radius_mode"),
            "stats_unit_mode": meta.get("stats_unit_mode"),
            "sampling": meta.get("sampling"),
            "speed": meta.get("speed"),
            "grid_shape": meta.get("grid_shape"),
            "effective_spacing_um": meta.get("effective_spacing_um"),
        })

    details["runs"] = runs
    if len(runs) <= 1:
        return True, "仅一次运行，无需一致性检查", details

    # Cs1: radius_mode 一致
    modes = {r.get("radius_mode") for r in runs if "error" not in r}
    if len(modes) > 1:
        issues.append(f"Cs1 radius_mode 不一致: {modes}")

    # Cs2: stats_unit_mode 一致
    unit_modes = {r.get("stats_unit_mode") for r in runs if "error" not in r}
    if len(unit_modes) > 1:
        issues.append(f"Cs2 stats_unit_mode 不一致: {unit_modes}")

    # Cs3: effective_spacing 自洽（同一网格应相同）
    spacings = {
        tuple(r["effective_spacing_um"])
        for r in runs
        if "error" not in r and r.get("effective_spacing_um")
    }
    if len(spacings) > 1:
        issues.append(f"Cs3 effective_spacing 不一致: {spacings}")

    details["radius_modes"] = list(modes)
    details["unit_modes"] = list(unit_modes)

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "一致性检查通过", details


# ───────────────────────────────────────────────────────────────────────
# N — 数值检查
# ───────────────────────────────────────────────────────────────────────


def validate_numeric(run_dir: str) -> tuple[bool, str, dict]:
    """N1–N3: inf/NaN 和基础数值完整性。"""
    issues: list[str] = []
    details: dict[str, Any] = {}

    stats_path = os.path.join(run_dir, "statistics_summary_d10+um.txt")
    if not os.path.exists(stats_path):
        return False, f"统计文件不存在: {stats_path}", details

    with open(stats_path, encoding="utf-8") as f:
        text = f.read()

    # N1: 检查 inf/NaN
    if "inf" in text.lower() or "nan" in text.lower():
        issues.append("N1 统计输出含 inf 或 NaN")

    # N1: 检查负数（排除弯曲度=1.0 附近的浮点噪声）
    for line in text.split("\n"):
        nums = re.findall(r"(-?\d+\.?\d*)", line)
        for n in nums:
            v = float(n)
            if v < -0.001:  # 允许 -0.000 浮点噪声
                issues.append(f"N1 负数: {v} in '{line.strip()}'")
                break

    # N2: volume > 0
    meta = _find_run_meta(run_dir)
    if meta:
        vol = meta.get("volume_mm3")
        if vol is not None:
            details["volume_mm3"] = vol
            if float(vol) <= 0:
                issues.append(f"N2 volume_mm3 <= 0: {vol}")

    # N3: 关键输出文件非空
    for fname in ["skeleton.pajek", "statistics_summary_d10+um.txt"]:
        fp = os.path.join(run_dir, fname)
        if os.path.exists(fp) and os.path.getsize(fp) == 0:
            issues.append(f"N3 空文件: {fname}")

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "数值检查通过", details


# ───────────────────────────────────────────────────────────────────────
# 组合验证（便利函数）
# ───────────────────────────────────────────────────────────────────────


def validate_run(run_dir: str, species: str = "mouse_brain") -> dict[str, Any]:
    """对单个 run 目录执行所有适用检查。"""
    results: dict[str, Any] = {"run_dir": run_dir, "checks": {}}

    for name, func in [
        ("structural", lambda: validate_structural_run(run_dir)),
        ("semantic", lambda: validate_semantic(run_dir, species)),
        ("numeric", lambda: validate_numeric(run_dir)),
    ]:
        passed, msg, details = func()
        results["checks"][name] = {"passed": passed, "message": msg, "details": details}

    results["all_passed"] = all(c["passed"] for c in results["checks"].values())
    return results


def validate_sample(sample_dir: str, species: str = "mouse_brain") -> dict[str, Any]:
    """对单个样本目录执行所有适用检查（含一致性）。"""
    results: dict[str, Any] = {"sample_dir": sample_dir, "checks": {}}

    # 结构：检查所有 run 子目录
    run_issues: list[str] = []
    for entry in sorted(os.listdir(sample_dir)):
        run_path = os.path.join(sample_dir, entry)
        if os.path.isdir(run_path) and entry.startswith("run_"):
            r = validate_run(run_path, species)
            if not r["all_passed"]:
                run_issues.append(f"{entry}: {r['checks']}")

    results["checks"]["runs"] = {
        "passed": len(run_issues) == 0,
        "message": "; ".join(run_issues) if run_issues else "所有 run 检查通过",
    }

    # 一致性
    passed, msg, details = validate_consistency(sample_dir)
    results["checks"]["consistency"] = {"passed": passed, "message": msg, "details": details}

    results["all_passed"] = all(c["passed"] for c in results["checks"].values())
    return results


def validate_batch(exp_root: str, species: str = "mouse_brain") -> dict[str, Any]:
    """全量 batch 验证：遍历所有样本 + 完整性检查。"""
    results: dict[str, Any] = {"exp_root": exp_root, "samples": {}, "summary": {}}

    total = passed = failed = 0
    for group in sorted(os.listdir(exp_root)):
        gp = os.path.join(exp_root, group)
        if not os.path.isdir(gp):
            continue
        for date_dir in sorted(os.listdir(gp)):
            dp = os.path.join(gp, date_dir)
            if not os.path.isdir(dp):
                continue
            for sample in sorted(os.listdir(dp)):
                sp = os.path.join(dp, sample)
                if not os.path.isdir(sp):
                    continue
                if not os.path.exists(os.path.join(sp, "sample_metadata.json")):
                    continue
                total += 1
                r = validate_sample(sp, species)
                results["samples"][os.path.relpath(sp, exp_root)] = r
                if r["all_passed"]:
                    passed += 1
                else:
                    failed += 1

    # 完整性
    c_passed, c_msg, c_details = validate_completeness(exp_root)
    results["checks"] = {"completeness": {"passed": c_passed, "message": c_msg, "details": c_details}}

    results["summary"] = {"total_samples": total, "passed": passed, "failed": failed}
    results["all_passed"] = c_passed and failed == 0
    return results


def validate_presubmit(
    seg_root: str, exp_root: str
) -> tuple[bool, str, dict]:
    """提交前检查：seg 结构 + metadata 覆盖。"""
    issues: list[str] = []
    details: dict[str, Any] = {}

    # S1+S2: seg 结构
    s_passed, s_msg, s_details = validate_structural_presubmit(seg_root)
    details["structural"] = {"passed": s_passed, "message": s_msg}
    if not s_passed:
        issues.append(s_msg)

    # C1: metadata 与 seg 交叉对比
    meta_samples: set[str] = set()
    for root, _dirs, files in os.walk(exp_root):
        for f in files:
            if f == "sample_metadata.json":
                rel = os.path.relpath(root, exp_root)  # root 即 sample_metadata.json 所在目录
                meta_samples.add(rel)

    seg_samples: set[str] = set()
    for root, _dirs, files in os.walk(seg_root):
        for f in files:
            if f.endswith(".tiff"):
                rel = os.path.relpath(root, seg_root)
                stem = os.path.splitext(f)[0]
                seg_samples.add(os.path.join(rel, stem))

    only_meta = meta_samples - seg_samples
    only_seg = seg_samples - meta_samples
    details["only_in_metadata"] = sorted(only_meta)
    details["only_in_seg"] = sorted(only_seg)

    if only_meta:
        issues.append(f"C1 {len(only_meta)} 样本有 metadata 无 seg .tiff: {sorted(only_meta)[:5]}...")
    if only_seg:
        issues.append(f"C1 {len(only_seg)} 样本有 seg .tiff 无 metadata: {sorted(only_seg)[:5]}...")

    details["meta_count"] = len(meta_samples)
    details["seg_count"] = len(seg_samples)

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "提交前检查通过", details
