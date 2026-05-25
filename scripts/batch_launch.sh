#!/bin/bash
# ============================================================================
#  批量提交脚本 —— Vascular_Statistics
# ============================================================================
#
#  用法:
#    bash scripts/batch_launch.sh --volume 0.078              # 提交所有 .tif
#    bash scripts/batch_launch.sh --volume 0.078 --dry         # 仅预览，不提交
#    bash scripts/batch_launch.sh --volume 0.078 --pattern "exp1_*.tif"
#
#  约定:
#    - 输入文件放在 DATA_ROOT（默认 /share/home/sukm/datasets/VascStats）
#    - 每个文件作为一个独立 Slurm 作业提交
#    - 提交记录存档到 logs/submit/submit_<timestamp>.log
#
# ============================================================================

set -euo pipefail

# --- 默认值 ---
DATA_ROOT="/share/home/sukm/datasets/VascStats"
PATTERN="*.tif"
DRY=false
VOLUME=""
SAMPLING="1.0"
SLURM_SCRIPT="scripts/pipeline.slurm"

# --- 解析参数 ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --volume|-v)
            VOLUME="$2"; shift 2 ;;
        --sampling|-s)
            SAMPLING="$2"; shift 2 ;;
        --pattern|-p)
            PATTERN="$2"; shift 2 ;;
        --data-root|-d)
            DATA_ROOT="$2"; shift 2 ;;
        --dry)
            DRY=true; shift ;;
        *)
            echo "Unknown option: $1"
            echo "用法: bash scripts/batch_launch.sh --volume <mm^3> [--pattern '*.tif'] [--dry]"
            exit 1 ;;
    esac
done

if [ -z "$VOLUME" ]; then
    echo "FATAL: 必须指定 --volume (组织体积 mm^3)"
    exit 1
fi

# --- 查找输入文件 ---
echo "=========================================="
echo "  Vascular_Statistics Batch Launcher"
echo "=========================================="
echo "  Data Root : $DATA_ROOT"
echo "  Pattern   : $PATTERN"
echo "  Volume    : $VOLUME mm^3"
echo "  Sampling  : $SAMPLING"
echo "  Dry Run   : $DRY"
echo "=========================================="
echo ""

FILES=()
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find "$DATA_ROOT" -maxdepth 5 -name "$PATTERN" -print0 2>/dev/null || true)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "未找到匹配文件: $DATA_ROOT/$PATTERN"
    exit 1
fi

echo "找到 ${#FILES[@]} 个文件:"
for f in "${FILES[@]}"; do
    echo "  $f"
done
echo ""

if [ "$DRY" = true ]; then
    echo "[DRY RUN] 以上文件将被提交。移除 --dry 以实际提交。"
    exit 0
fi

# --- 提交前检查：队列中是否已有 vascstats 任务 ---
EXISTING=$(squeue -u "$USER" -n vascstats -h 2>/dev/null | wc -l || echo "0")
if [ "$EXISTING" -gt 0 ]; then
    echo "WARNING: 队列中已有 $EXISTING 个 vascstats 任务。"
    read -p "是否继续提交？[y/N] " -r REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "已取消。"
        exit 0
    fi
fi

# --- 提交记录存档 ---
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SUBMIT_LOG="logs/submit/submit_${TIMESTAMP}.log"
mkdir -p "$(dirname "$SUBMIT_LOG")"

# 自存档
cp "$0" "logs/submit/batch_launch_${TIMESTAMP}.sh" 2>/dev/null || true

# --- 批量提交 ---
echo "开始提交..."
echo ""

JOB_IDS=()
for f in "${FILES[@]}"; do
    # 使用相对于 DATA_ROOT 的路径
    REL_PATH="${f#$DATA_ROOT/}"
    BASENAME="$(basename "$f" | sed 's/\.[^.]*$//')"

    JOB_ID=$(sbatch \
        --job-name="vs_${BASENAME:0:16}" \
        --output="logs/pipeline_${BASENAME}_%j.out" \
        --error="logs/pipeline_${BASENAME}_%j.err" \
        "$SLURM_SCRIPT" "$REL_PATH" "$VOLUME" "$BASENAME" "$SAMPLING" \
        2>&1 | grep -oP '\d+')

    if [ -n "$JOB_ID" ]; then
        echo "  [$JOB_ID] $BASENAME"
        JOB_IDS+=("$JOB_ID")
    else
        echo "  [FAILED] $BASENAME"
    fi
done

echo ""
echo "=========================================="
echo "  提交完成: ${#JOB_IDS[@]}/${#FILES[@]} 个作业"
echo "  日志     : $SUBMIT_LOG"
echo "  监控     : squeue -u $USER -n 'vs_*'"
echo "=========================================="

# 保存提交记录
{
    echo "Submit: $TIMESTAMP"
    echo "Volume: $VOLUME"
    echo "Files:"
    for i in "${!FILES[@]}"; do
        echo "  ${JOB_IDS[$i]:-FAIL}  ${FILES[$i]}"
    done
} > "$SUBMIT_LOG"
