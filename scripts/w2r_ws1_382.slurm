#!/bin/bash
# ============================================================================
#  Slurm 作业：W2R_ws1 CSAM 分割 @ 金标准尺寸 (382×382×300) 骨架化
# ============================================================================
#
#   目标：验证 CSAM 分割与手动金标准骨架的对齐
#   输入：resize382 CSAM seg (由 resize-tiff --xy 382 --z 300 --binary 生成)
#   参数：spacing=[1.8362,1.8362,2.0] (金标准有效间距), sampling=2.0, speed=0.2
#
#   用法：
#     sbatch scripts/w2r_ws1_382.slurm
#
# ============================================================================

#SBATCH --job-name=w2r_ws1_382

#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/w2r_ws1_382_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/w2r_ws1_382_%j.err

# ---- 资源申请（匹配 A/B 实验方案 A：~3.3M 前景体素）----
# 骨架化 85% 时间在 scipy.sparse.linalg.lsqr（SpMV 内存带宽敏感）
# 3.3M fg → 稀疏矩阵 ~24-40 GB，64G 含 1.6× 余量
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# 线程控制：锁定 numpy/scipy 底层线性代数线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

set -euo pipefail

# ---- 环境初始化 ----
set +u
source ~/.bashrc
set -u
conda activate vascstats

PROJECT_DIR="/share/home/sukm/vascular-statistics"
cd "$PROJECT_DIR"

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo 'N/A')
COMMIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')

echo "=============================================="
echo "  W2R_ws1 骨架化（金标准尺寸 382×382×300）"
echo "=============================================="
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  Partition  : $SLURM_JOB_PARTITION"
echo "  Commit     : $COMMIT_HASH ($COMMIT_BRANCH)"
echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

# ---- 固定参数 ----
INPUT_TIFF="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_csam_seg_resize382.tiff"
OUTPUT_ROOT="/share/home/sukm/experiments/vs_2pfm_skelgt/w2r_ws1_382"
OUTPUT_STEM="skeleton"
SAMPLING="2.0"
SPEED="0.2"
VOLUME="0.40936"    # 与 A/B 实验一致（整图组织体积）
# 金标准 spacing: [1.8362, 1.8362, 2.0] — 通过 --anisotropic 启用各向异性统计
ANISOTROPIC="--anisotropic"

RUN_ID="run_$(date '+%Y%m%d_%H%M%S')"
JOB_OUTDIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$JOB_OUTDIR" "$PROJECT_DIR/logs"

echo "  Input      : $INPUT_TIFF"
echo "  Output     : $JOB_OUTDIR"
echo "  Sampling   : $SAMPLING"
echo "  Speed      : $SPEED"
echo "  Volume     : $VOLUME mm^3"
echo "  Anisotropic: yes"
echo ""

# 环境验证
echo "--- Environment Check ---"
python -c "
import sys
print(f'Python  : {sys.executable} — {sys.version.split()[0]}')
import numpy;  print(f'numpy   : {numpy.__version__}')
import scipy;  print(f'scipy   : {scipy.__version__}')
import networkx; print(f'networkx: {networkx.__version__}')
print('Environment OK')
" || exit 1
echo "-------------------------"
echo ""

# ---- 信号捕获 ----
PIPE_PID=""
on_sigterm() {
    echo ""
    echo "[$(date '+%H:%M:%S')] SIGTERM received, forwarding..."
    if [ -n "$PIPE_PID" ] && kill -0 "$PIPE_PID" 2>/dev/null; then
        kill -SIGTERM "$PIPE_PID"
        wait "$PIPE_PID" 2>/dev/null || true
    fi
    exit 143
}
trap on_sigterm SIGTERM

# ---- 执行管线 (skeletonize + stats all phases) ----
echo "[$(date '+%H:%M:%S')] Starting pipeline..."
echo ""

cd "$JOB_OUTDIR"

PIPE_CMD=(
    python -m vascular_statistics.cli
    pipeline "$INPUT_TIFF"
    --output-stem "$OUTPUT_STEM"
    --sampling "$SAMPLING"
    --speed "$SPEED"
    --volume "$VOLUME"
    --phases all
    --anisotropic
)

echo "Command: ${PIPE_CMD[*]}"
echo ""

"${PIPE_CMD[@]}" 2>&1 | tee "$JOB_OUTDIR/pipeline.log" &
PIPE_PID=$!
set +e
wait "$PIPE_PID"
EXIT_CODE=$?
set -e

# ---- 收尾 ----
echo ""
echo "=============================================="
echo "  End Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit Code  : $EXIT_CODE"
echo "  Output Dir : $JOB_OUTDIR"
echo "=============================================="

# 列出产物
if [ -d "$JOB_OUTDIR" ]; then
    echo ""
    echo "Output files:"
    ls -la "$JOB_OUTDIR"/
fi

exit $EXIT_CODE
