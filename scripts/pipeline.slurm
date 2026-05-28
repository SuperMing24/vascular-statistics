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
#    │                                                                  │
#    │  # 全局进度查询（新）                                              │
#    │  python -m vascular_statistics.cli status \                      │
#    │    --manifest $OUTPUT_ROOT/manifest.json                          │
#    └─────────────────────────────────────────────────────────────────┘
#
#    ┌─ 交互式调试 ────────────────────────────────────────────────────┐
#    │  srun --partition=compute --cpus-per-task=8 --mem=48G \         │
#    │       --time=00:30:00 --pty bash                                 │
#    │  # 进入后手动激活 conda 并运行                                    │
#    └─────────────────────────────────────────────────────────────────┘
#
#  输出结构（v2 — 面向样本）:
#    每个样本有其专属目录，多次运行以时间戳子目录隔离。
#    全局 manifest.json 注册所有样本处理状态，支持断点续跑。
#    详见 docs/output_structure_blueprint.md。
#
#  与 Vascular_Extraction train.slurm 的差异:
#   - CPU-only 任务（无 GPU 申请）
#   - partition=compute（96 核 × 1004 GB，CPU 任务不占 GPU 槽位）
#   - 8 核 + 48 GB 内存（numpy BLAS 效率甜点 + 大堆栈余量）
#   - 4 小时时限（典型样本 0.5-2h，短时限 = 高回填优先级）
#   - 无 CUDA 环境验证
#   - 输出写入面向样本的目录结构 (v2)
#   - 线程控制：export OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
#
# ============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第一部分：Slurm 资源申请（必须在任何非注释命令之前）                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

#SBATCH --job-name=vs_pipeline

#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# ═══════════════════════════════════════════════════════════════════════════
# 资源申请（2026-05-26 优化，基于 SpMV/稀疏求解器文献调研 + 集群硬件分析）
#
#   计算特征：
#     骨架化 85% 时间在 scipy.sparse.linalg.lsqr（每次迭代 3 次 SpMV）。
#     SpMV 是内存带宽敏感型操作（非 FLOPS 敏感），性能由 DRAM 带宽决定。
#     10% 在 NetworkX 图操作（单线程 Python GIL），其余阶段不计。
#
#   CPU 核心：8
#     SpMV 核心数超过 4-8 后收益递减（受限于内存带宽，非计算能力）。
#     Amdahl 定律：8→16 核仅 +11% 增益但资源翻倍。
#
#   内存：48 GB（典型样本 ~8-16 GB，大 3D 堆栈 ~32 GB，保留 50% 余量）
#
#   时限：1 天（实测采样=1.0 时骨架化需 3-8h，1 天提供充足余量；
#     compute 分区 MaxTime=7 天，1 天不影响回填优先级。）
#
#   分区：compute（优先）
#     推测为双路服务器（96 核无法来自单 CPU）→ 双内存控制器 → 双倍带宽。
#     c003 当前全空闲，为首选节点。
#     备选：tao 分区（40 核 × 376 GB，也可满足，推测为单路）
#
#   [待确认] 各节点 CPU 具体型号（lscpu）、NUMA 拓扑（numactl --hardware）、
#   内存通道数。确认后可按需调整分区选择或添加 --nodelist 偏好。
#
#   线程控制（关键！）：不设 OMP_NUM_THREADS 则 numpy 可能抢占节点全部 96 核，
#   导致同节点多作业严重争抢。以下 4 项强制匹配 --cpus-per-task。
#   OMP_PROC_BIND=close 减少 NUMA 跨片内存访问（对带宽敏感型任务有益）。
# ═══════════════════════════════════════════════════════════════════════════

#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1

#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# 线程控制：锁定 numpy/scipy 的底层线性代数线程数，匹配 Slurm 分配
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NUMA 亲和绑定：将线程固定在相邻核心，减少跨 NUMA 节点的内存访问延迟
# 对内存带宽敏感的稀疏求解器（lsqr SpMV）有正向效果
export OMP_PROC_BIND=close
export OMP_PLACES=cores

set -euo pipefail


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第二部分：用户可调参数                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- 路径配置 ---
PROJECT_DIR="/share/home/sukm/vascular-statistics"
DATA_ROOT="/share/home/sukm/datasets/VascStats"
OUTPUT_ROOT="/share/home/sukm/experiments/vascstats"

# --- 作业参数 ---
# 用法：sbatch scripts/pipeline.slurm <input> <volume> [output_stem] [sampling] [phases]
#   input       — 输入文件（相对于 DATA_ROOT，或绝对路径）
#   volume      — 组织体积 mm^3（仅 phases=all/stats 时使用）
#   output_stem — 输出文件前缀（默认 "skeleton"）
#   sampling    — 稀疏采样率 (1.0=最密, 2.0=快速, 默认 1.0)
#   phases      — 执行阶段: all | skeletonize | stats（默认 all）
INPUT_FILE="${1:?缺少输入文件参数}"
VOLUME="${2:-}"
OUTPUT_STEM="${3:-skeleton}"
SAMPLING="${4:-1.0}"
PHASES="${5:-all}"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第三部分：环境初始化 + 输出目录计算 + manifest 注册 (v2 新结构)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

START_TIME="$(date -Iseconds)"
RUN_ID="run_$(date '+%Y%m%d_%H%M%S')"

echo "========================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Job Name   : $SLURM_JOB_NAME"
echo "  Node       : $SLURM_NODELIST"
echo "  Partition  : $SLURM_JOB_PARTITION"
echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Run ID     : $RUN_ID"
echo "  Input      : $INPUT_FILE"
echo "  Volume     : $VOLUME mm^3"
echo "  Sampling   : $SAMPLING"
echo "  Phases     : $PHASES"
echo "  Data Root  : $DATA_ROOT"
echo "  Output Root: $OUTPUT_ROOT"
echo "========================================"
echo ""

# --- 可复现性：记录代码版本 ---
cd "$PROJECT_DIR"
COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo 'N/A')
COMMIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')
echo "--- Code Version ---"
echo "Commit  : $COMMIT_HASH"
echo "Branch  : $COMMIT_BRANCH"
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

# --- 计算面向样本的输出路径 ---
# 步骤：
#  1. 解析输入文件的相对路径（去 DATA_ROOT 前缀）
#  2. 通过 Python compute_sample_key() 计算样本目录名
#  3. 拼合：OUTPUT_ROOT / sample_key / run_id
if [[ "$INPUT_FILE" == /* ]]; then
    REL_INPUT="${INPUT_FILE#$DATA_ROOT/}"
else
    REL_INPUT="$INPUT_FILE"
fi

SAMPLE_DIR=$(python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/python')
from vascular_statistics.batch import compute_sample_key
# 传入相对路径（含扩展名），compute_sample_key 会自动去扩展名
print(compute_sample_key('$REL_INPUT'))
")

# --- 确定输出目录 ---
# skeletonize / all → 新 run_<ts> 目录；stats → 已有骨架 run 目录（统计从属于骨架）
if [ "$PHASES" = "stats" ]; then
    SAMPLE_ROOT="$OUTPUT_ROOT/$SAMPLE_DIR"
    JOB_OUTDIR=""
    if [ -d "$SAMPLE_ROOT" ]; then
        # 找最新含 skeleton.pajek 的 run_* 目录
        JOB_OUTDIR=$(find "$SAMPLE_ROOT" -name 'skeleton.pajek' -type f 2>/dev/null \
            | sed 's|/skeleton\.pajek$||' | sort | tail -1)
    fi
    if [ -z "$JOB_OUTDIR" ]; then
        echo "FATAL: 该样本无 skeleton.pajek，请先运行骨架化。"
        exit 1
    fi
    echo "Stats target: $JOB_OUTDIR （写入已有骨架目录，不创建新 run）"
else
    JOB_OUTDIR="$OUTPUT_ROOT/$SAMPLE_DIR/$RUN_ID"
    mkdir -p "$JOB_OUTDIR"
fi
mkdir -p "$PROJECT_DIR/logs"

echo "Sample dir : $SAMPLE_DIR"
echo "Output dir : $JOB_OUTDIR"
echo ""

# --- 初始化 conda 环境（必须在任何依赖 vascstats 包的 Python 调用之前） ---
set +u
source ~/.bashrc
set -u

# 尝试 vascstats 环境，若不存在则回退到 vesseg
if conda env list 2>/dev/null | grep -q "vascstats"; then
    conda activate vascstats
else
    conda activate /share/home/sukm/conda_envs/vesseg
fi

# --- 向 manifest.json 注册作业开始（stats 阶段不注册，统计从属于骨架 run） ---
if [ "$PHASES" != "stats" ]; then
    set +o pipefail
    MANIFEST_LOG=$(python -m vascular_statistics.cli update-manifest \
        --manifest "$OUTPUT_ROOT/manifest.json" \
        --sample-key "$SAMPLE_DIR" \
        --input-path "$REL_INPUT" \
        --volume-mm3 "$VOLUME" \
        --run-id "$RUN_ID" \
        --job-id "$SLURM_JOB_ID" \
        --status running \
        --start-time "$START_TIME" \
        --node "$SLURM_NODELIST" \
        --commit "$COMMIT_HASH" \
        2>&1)
    MANIFEST_RC=$?
    set -o pipefail
    echo "$MANIFEST_LOG" | sed 's/^/  [manifest] /'
    if [ $MANIFEST_RC -ne 0 ]; then
        echo "  [manifest] WARNING: update-manifest (start) failed (rc=$MANIFEST_RC), continuing..."
    fi
fi

echo ""

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
    # 尝试更新 manifest 为 killed
    python -m vascular_statistics.cli update-manifest \
        --manifest "$OUTPUT_ROOT/manifest.json" \
        --sample-key "$SAMPLE_DIR" \
        --run-id "$RUN_ID" \
        --status failed \
        --exit-code 143 \
        --end-time "$(date -Iseconds)" \
        2>/dev/null || true

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
    # 写入 manifest 失败状态
    python -m vascular_statistics.cli update-manifest \
        --manifest "$OUTPUT_ROOT/manifest.json" \
        --sample-key "$SAMPLE_DIR" \
        --run-id "$RUN_ID" \
        --status failed \
        --exit-code 1 \
        --end-time "$(date -Iseconds)" \
        2>/dev/null || true
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Starting pipeline..."
echo "Input  : $FULL_INPUT"
echo "Volume : $VOLUME mm^3"
echo ""

# 切换到输出目录执行
cd "$JOB_OUTDIR"

PIPE_CMD=(
    python -m vascular_statistics.cli
    pipeline "$FULL_INPUT"
    --output-stem "$OUTPUT_STEM"
    --sampling "$SAMPLING"
    --phases "$PHASES"
)
# --volume 可选：仅在明确传入时追加，否则由 CLI 从样本元数据自动查找
if [ -n "$VOLUME" ]; then
    PIPE_CMD+=(--volume "$VOLUME")
fi

echo "Command: ${PIPE_CMD[*]}"
echo ""

# 管线输出同时写入 stdout（Slurm 捕获）和日志（run 目录内归档）
# stats 阶段写独立日志，避免覆盖骨架化阶段的 pipeline.log
if [ "$PHASES" = "stats" ]; then
    PIPE_LOG="$JOB_OUTDIR/pipeline_stats.log"
else
    PIPE_LOG="$JOB_OUTDIR/pipeline.log"
fi
"${PIPE_CMD[@]}" 2>&1 | tee "$PIPE_LOG" &
PIPE_PID=$!

set +e
wait "$PIPE_PID"
EXIT_CODE=$?
set -e

END_TIME="$(date -Iseconds)"

echo ""
echo "========================================"
echo "  End Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit Code  : $EXIT_CODE"
echo "  Output Dir : $JOB_OUTDIR"
echo "========================================"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第六部分：收尾 —— manifest 更新 + run_meta.json + 产物清单              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- 更新 manifest.json + run_meta.json（stats 阶段跳过，统计从属于骨架 run） ---
if [ "$PHASES" != "stats" ]; then
    FINAL_STATUS="completed"
    if [ "$EXIT_CODE" -ne 0 ]; then
        FINAL_STATUS="failed"
    fi

    set +o pipefail
    MANIFEST_LOG=$(python -m vascular_statistics.cli update-manifest \
        --manifest "$OUTPUT_ROOT/manifest.json" \
        --sample-key "$SAMPLE_DIR" \
        --run-id "$RUN_ID" \
        --status "$FINAL_STATUS" \
        --exit-code "$EXIT_CODE" \
        --end-time "$END_TIME" \
        --start-time "$START_TIME" \
        2>&1)
    MANIFEST_RC=$?
    set -o pipefail
    echo "$MANIFEST_LOG" | sed 's/^/  [manifest] /'
    if [ $MANIFEST_RC -ne 0 ]; then
        echo "  [manifest] WARNING: update-manifest (end) failed (rc=$MANIFEST_RC)"
    fi
fi

# --- 写入 run_meta.json（单次运行的完整元数据，stats 阶段不单独记录） ---
if [ "$PHASES" != "stats" ]; then
python -c "
import json, os, sys
meta = {
    'run_id': '$RUN_ID',
    'sample_key': '$SAMPLE_DIR',
    'input_path': '$REL_INPUT',
    'input_format': '${REL_INPUT##*.}',
    'volume_mm3': $VOLUME,
    'sampling': $SAMPLING,
    'phases': '$PHASES',
    'job_id': '$SLURM_JOB_ID',
    'slurm_node': '$SLURM_NODELIST',
    'start_time': '$START_TIME',
    'end_time': '$END_TIME',
    'exit_code': $EXIT_CODE,
    'status': '$FINAL_STATUS',
    'commit': '$COMMIT_HASH',
}
path = os.path.join('$JOB_OUTDIR', 'run_meta.json')
with open(path, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f'  [meta] {path}')
"
fi   # end of "if phases != stats" for run_meta.json

# --- 列出输出产物 ---
if [ -d "$JOB_OUTDIR" ]; then
    echo ""
    echo "Output files:"
    ls -la "$JOB_OUTDIR"/
fi

# --- 自动刷新样本级统计汇总（stats 或 all 阶段，且运行成功时） ---
if [ "$EXIT_CODE" -eq 0 ] && { [ "$PHASES" = "stats" ] || [ "$PHASES" = "all" ]; }; then
    echo ""
    echo "--- Refreshing sample-level aggregate statistics ---"
    python -m vascular_statistics.cli aggregate-stats \
        --output-root "$OUTPUT_ROOT" \
        --sample-key "$SAMPLE_DIR" \
        2>&1 | sed 's/^/  [aggregate] /'
fi

exit $EXIT_CODE
