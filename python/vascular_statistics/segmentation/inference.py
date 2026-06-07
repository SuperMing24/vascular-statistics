"""
推理编排：通过 subprocess 调用 Vascular_Extraction 的 predict.py。

设计决策（见 docs/model_comparison_minivess_20260607.md）：
  - 首选模型：nnU-Net 2D (seed22, epoch 41) — MiniVess 验证集 6 项指标全第一
  - 推理模式：StandardExperimentFactory._infer_25d() → 逐切片 2.5D 滑窗
  - 归一化：predict.py 内部逐窗口 MinMax，无需外部预处理

调用方式为 subprocess，与现有 C++ vessel_stats.exe 调用模式一致。
不直接 import Extraction 模块以避免依赖树污染 vascstats 环境。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


# ══════════════════════════════════════════════════════════════════
# 服务器端路径（可通过环境变量或配置文件覆盖）
# ══════════════════════════════════════════════════════════════════

# Vascular_Extraction 项目根目录（服务器端）
_EXTRACTION_ROOT = "/share/home/sukm/Vascular_Extraction"

# nnU-Net 2D 实验目录（seed22 — MiniVess 最优 seed）
_NNUNET_EXP_DIR = (
    "/share/home/sukm/experiments/phase0_baseline/"
    "nnunet_2d_bce_dice_slice3_bs4/seed22"
)

# Conda 环境（含 PyTorch + MONAI + nibabel）
_CONDA_ENV = "/share/home/sukm/conda_envs/vesseg"


def _resolve_project_root() -> str:
    """解析当前 Vascular_Statistics 项目的根目录。"""
    return str(Path(__file__).resolve().parent.parent.parent.parent)


def _find_predict_script() -> str:
    """查找 Vascular_Extraction 的 predict.py 脚本路径。"""
    candidates = [
        os.path.join(_EXTRACTION_ROOT, "src", "scripts", "predict.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"找不到 predict.py。请确认 Vascular_Extraction 已部署到 {_EXTRACTION_ROOT}。\n"
        f"如路径不同，请设置环境变量 VASCULAR_EXTRACTION_ROOT。"
    )


def run_segmentation(
    nii_path: str,
    output_dir: str,
    exp_dir: Optional[str] = None,
    threshold: float = 0.5,
    timeout_sec: int = 3600,
) -> str:
    """对单个 .nii 文件运行分割推理，输出二值掩码 .tif。

    参数：
        nii_path: 输入 .nii 文件路径（已从 .mat 转换）。
        output_dir: 输出目录。
        exp_dir: Extraction 实验目录（含 checkpoints/best_model.pth）。
                 默认使用 nnU-Net 2D seed22。
        threshold: 二值化阈值（默认 0.5）。
        timeout_sec: 推理超时秒数（默认 3600 = 1 小时）。

    返回：
        输出 .tif 文件路径。

    异常：
        subprocess.TimeoutExpired: 推理超时。
        subprocess.CalledProcessError: 推理失败。
    """
    if exp_dir is None:
        exp_dir = _NNUNET_EXP_DIR

    nii_path = str(nii_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(nii_path).stem
    # predict.py 输出的 .tif 文件名：{stem}.tiff（不带路径前缀时写入 CWD）
    out_tif = os.path.join(output_dir, stem + ".tiff")

    predict_script = _find_predict_script()

    # 构建命令：
    # conda run -n vesseg python predict.py --exp_dir ... --input ... --out ... --threshold ...
    cmd = [
        "conda", "run", "-n", os.path.basename(_CONDA_ENV),
        "python", predict_script,
        "--exp_dir", exp_dir,
        "--input", nii_path,
        "--out", out_tif,
        "--threshold", str(threshold),
    ]

    # 如果 conda 不可用，回退到直接 python 调用（依赖当前环境）
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        cmd = [
            sys.executable, predict_script,
            "--exp_dir", exp_dir,
            "--input", nii_path,
            "--out", out_tif,
            "--threshold", str(threshold),
        ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=output_dir,
    )

    if result.returncode != 0:
        import sys as _sys
        print(f"[inference] predict.py stderr:", file=_sys.stderr)
        print(result.stderr[-2000:], file=_sys.stderr)
        print(f"[inference] predict.py stdout (last 500 chars):", file=_sys.stderr)
        print(result.stdout[-500:], file=_sys.stderr)
        raise RuntimeError(
            f"predict.py 推理失败 (exit={result.returncode})。"
            f"详情见上方 stderr 输出。"
        )

    if not os.path.exists(out_tif):
        raise FileNotFoundError(
            f"推理完成但未生成输出文件: {out_tif}\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    return out_tif
