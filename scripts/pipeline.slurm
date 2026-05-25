#!/bin/bash
# ============================================================================
#  Slurm 骨架化+统计作业模板 —— Vascular_Statistics
# ============================================================================
#
#  使用方法（在服务器项目目录下执行）:
#
#    ┌─ 基本用法 ──────────────────────────────────────────────────────┐
#    │  sbatch scripts/pipeline.slurm \                                │
#    │    /share/home/sukm/datasets/VascStats/sample.tif 0.078          │
#    │                                                                  │
#    │  # 通过 batch_launch.sh 批量提交：                                  │
#    │  bash scripts/batch_launch.sh --volume 0.078 --dry                 │
#    └─────────────────────────────────────────────────────────────────┘
#
#    ┌─ 监控命令 ──────────────────────────────────────────────────────┐
#    │  squeue -u $USER          # 查看自己的任务（PD=排队, R=运行）    │
#    │  scancel <job_id>         # 取消任务（Slurm 发 SIGTERM→120s→KILL │
#    │  tail -f logs/pipeline_*.out  # 实时查看最新输出                  │
#    │  sacct -j <job_id> --format=Elapsed,MaxRSS,MaxVMSize,State       │
#    │                           # 查看已完成任务的资源使用              │
#    └─────────────────────────────────────────────────────────────────┘
#
#    ┌─ 交互式调试 ────────────────────────────────────────────────────┐
#    │  srun --partition=compute --cpus-per-task=4 --mem=16G \         │
#    │       --time=00:30:00 --pty bash                                 │
#    │  # 进入后手动激活 conda 并运行                                    │
#    └─────────────────────────────────────────────────────────────────┘
#
#  与 Vascular_Extraction train.slurm 的差异:
#   - CPU-only 任务（无 GPU 申请）
#   - partition=compute（更多 CPU 核，避免占用 GPU 节点）
#   - 更短的时间限制（骨架化一条血管通常 < 2h）
#   - 无 CUDA 环境验证
#   - 输出写入 output/<job_name>_<job_id>/ 目录
#
# ============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第一部分：Slurm 资源申请（必须在任何非注释命令之前）                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

#SBATCH --job-name=vs_pipeline

#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --ntasks=1

#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第二部分：用户可调参数                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- 路径配置 ---
PROJECT_DIR="/share/home/sukm/vascular-statistics"
DATA_ROOT="/share/home/sukm/datasets/VascStats"
OUTPUT_ROOT="/share/home/sukm/experiments/vascstats"

# --- 作业参数 ---
# 用法：sbatch scripts/pipeline.slurm <input.tif> <volume> [output_stem] [sampling]
#   input.tif   — 输入 TIFF 堆叠（相对于 DATA_ROOT，或绝对路径）
#   volume      — 组织体积 mm^3
#   output_stem — 输出文件前缀（默认基于输入文件名）
#   sampling    — 稀疏采样率 (1.0=最密, 2.0=快速, 默认 1.0)
INPUT_FILE="${1:?缺少输入文件参数}"
VOLUME="${2:?缺少组织体积参数 (mm^3)}"
OUTPUT_STEM="${3:-}"
SAMPLING="${4:-1.0}"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第三部分：环境初始化 + 作业元数据记录                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

echo "========================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Job Name   : $SLURM_JOB_NAME"
echo "  Node       : $SLURM_NODELIST"
echo "  Partition  : $SLURM_JOB_PARTITION"
echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Input      : $INPUT_FILE"
echo "  Volume     : $VOLUME mm^3"
echo "  Sampling   : $SAMPLING"
echo "  Data Root  : $DATA_ROOT"
echo "  Output Dir : $OUTPUT_ROOT"
echo "========================================"
echo ""

# --- 可复现性：记录代码版本 ---
cd "$PROJECT_DIR"
echo "--- Code Version ---"
echo "Commit  : $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Branch  : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
echo "Dirty   : $(git diff --stat 2>/dev/null | tail -1 || echo 'clean')"
echo "--------------------"
echo ""

# --- 磁盘空间预检 ---
REQUIRED_GB=5
AVAIL_KB=$(df --output=avail "$OUTPUT_ROOT" 2>/dev/null | tail -1 | tr -d ' ')
if [ -z "$AVAIL_KB" ] || [ "$AVAIL_KB" -le 0 ]; then
    echo "WARNING: 无法检测 $OUTPUT_ROOT 磁盘空间，跳过预检"
else
    AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
    if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
        echo "FATAL: $OUTPUT_ROOT 可用空间不足 ($AVAIL_GB GB < $REQUIRED_GB GB)"
        exit 1
    fi
    echo "Disk check: $OUTPUT_ROOT available = $AVAIL_GB GB (need >= $REQUIRED_GB GB)"
fi
echo ""

# --- 输出目录 ---
# 每个作业输出到独立目录，避免覆盖
if [ -z "$OUTPUT_STEM" ]; then
    OUTPUT_STEM="$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')"
fi
JOB_OUTDIR="$OUTPUT_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$JOB_OUTDIR" "$PROJECT_DIR/logs"

echo "Output stem: $OUTPUT_STEM"
echo "Output dir : $JOB_OUTDIR"
echo ""

# --- 初始化 conda 环境 ---
set +u
source ~/.bashrc
set -u

# 尝试 vascstats 环境，若不存在则回退到 vesseg
if conda env list 2>/dev/null | grep -q "vascstats"; then
    conda activate vascstats
else
    conda activate /share/home/sukm/conda_envs/vesseg
fi

# --- 环境验证 ---
echo "--- Environment Check ---"
python -c "
import sys
print(f'Python  : {sys.executable} — {sys.version.split()[0]}')
import numpy;  print(f'numpy   : {numpy.__version__}')
import scipy;  print(f'scipy   : {scipy.__version__}')
import networkx; print(f'networkx: {networkx.__version__}')
import skimage; print(f'skimage : {skimage.__version__}')
import h5py;   print(f'h5py    : {h5py.__version__}')
print('Environment OK')
" || exit 1
echo "-------------------------"
echo ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第四部分：信号捕获与优雅退出                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PIPE_PID=""

on_sigterm() {
    echo ""
    echo "======================================"
    echo "  [$(date '+%H:%M:%S')] SIGTERM received"
    echo "  Forwarding to pipeline (PID=$PIPE_PID)..."
    echo "======================================"
    if [ -n "$PIPE_PID" ] && kill -0 "$PIPE_PID" 2>/dev/null; then
        kill -SIGTERM "$PIPE_PID"
        wait "$PIPE_PID" 2>/dev/null
    fi
    exit 143
}

trap on_sigterm SIGTERM


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第五部分：执行管线                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# 解析输入路径：如果是相对路径则拼接 DATA_ROOT
if [[ "$INPUT_FILE" == /* ]]; then
    FULL_INPUT="$INPUT_FILE"
else
    FULL_INPUT="$DATA_ROOT/$INPUT_FILE"
fi

if [ ! -f "$FULL_INPUT" ]; then
    echo "FATAL: 输入文件不存在: $FULL_INPUT"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Starting pipeline..."
echo "Input  : $FULL_INPUT"
echo "Volume : $VOLUME mm^3"
echo ""

# 切换到输出目录执行（产物集中存放）
cd "$JOB_OUTDIR"

PIPE_CMD=(
    python -m vascular_statistics.cli
    pipeline "$FULL_INPUT"
    --volume "$VOLUME"
    --output-stem "$OUTPUT_STEM"
    --sampling "$SAMPLING"
)

echo "Command: ${PIPE_CMD[*]}"
echo ""

"${PIPE_CMD[@]}" &
PIPE_PID=$!

set +e
wait "$PIPE_PID"
EXIT_CODE=$?
set -e

echo ""
echo "========================================"
echo "  End Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit Code  : $EXIT_CODE"
echo "  Output Dir : $JOB_OUTDIR"
echo "========================================"

# 列出输出文件
if [ -d "$JOB_OUTDIR" ]; then
    echo ""
    echo "Output files:"
    ls -la "$JOB_OUTDIR"/
fi

exit $EXIT_CODE
