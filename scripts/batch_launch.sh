#!/bin/bash
# ============================================================================
#  批量提交脚本 —— Vascular_Statistics
# ============================================================================
#
#  用法:
#    bash scripts/batch_launch.sh --volume 0.078              # 提交所有 .tif
#    bash scripts/batch_launch.sh --volume 0.078 --dry         # 仅预览，不提交
#    bash scripts/batch_launch.sh --volume 0.078 --resume       # 跳过已完成样本
#    bash scripts/batch_launch.sh --volume 0.078 --pattern "exp1_*.tif"
#
#  约定:
#    - 输入文件放在 DATA_ROOT（默认 /share/home/sukm/datasets/VascStats）
#    - 输出写入 OUTPUT_ROOT（默认 /share/home/sukm/experiments/vascstats）
#    - 每个文件作为一个独立 Slurm 作业提交
#    - 提交记录存档到 logs/submit/submit_<timestamp>.log
#
#  输出结构 (v2):
#    面向样本的输出目录结构 —— 每个样本一个目录，多次运行以时间戳子目录隔离。
#    --resume 模式读取 manifest.json 跳过已完成的样本。
#    详见 docs/output_structure_blueprint.md。
#
# ============================================================================

set -euo pipefail

# --- 项目根目录（脚本所在目录的上级） ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- 从 server_paths.json 读取默认路径（如文件存在） ---
SERVER_CONFIG="$PROJECT_DIR/configs/delta/server_paths.json"
if [ -f "$SERVER_CONFIG" ]; then
    DATA_ROOT=$(python -c "import json; print(json.load(open('$SERVER_CONFIG'))['data_root'])" 2>/dev/null || echo "/share/home/sukm/datasets/VascStats")
    OUTPUT_ROOT=$(python -c "import json; print(json.load(open('$SERVER_CONFIG'))['output_root'])" 2>/dev/null || echo "/share/home/sukm/experiments/vascstats")
else
    DATA_ROOT="/share/home/sukm/datasets/VascStats"
    OUTPUT_ROOT="/share/home/sukm/experiments/vascstats"
fi

# --- 默认值 ---
PATTERN="*.tif"
DRY=false
VOLUME=""
SAMPLING="1.0"
PHASES="all"
RESUME=false
FILES_FROM=""
SLURM_SCRIPT="scripts/pipeline.slurm"
PARTITION="compute"
ANISOTROPIC=""
PHYSICAL_RADIUS=""
SPEED="0.05"
HAS_SKELETON=false
MIN_SKELETONS=0
MAX_SKELETONS=-1
COUNT_ONLY=false

# --- 解析参数 ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --volume|-v)
            VOLUME="$2"; shift 2 ;;
        --sampling|-s)
            SAMPLING="$2"; shift 2 ;;
        --phases)
            PHASES="$2"; shift 2 ;;
        --pattern|-p)
            PATTERN="$2"; shift 2 ;;
        --data-root|-d)
            DATA_ROOT="$2"; shift 2 ;;
        --output-root|-o)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --files-from|-f)
            FILES_FROM="$2"; shift 2 ;;
        --dry)
            DRY=true; shift ;;
        --resume|-r)
            RESUME=true; shift ;;
        --min-skeletons)
            MIN_SKELETONS="$2"; shift 2 ;;
        --max-skeletons)
            MAX_SKELETONS="$2"; shift 2 ;;
        --count-skeletons)
            COUNT_ONLY=true; shift ;;
        --has-skeleton)
            HAS_SKELETON=true; shift ;;
        --partition)
            PARTITION="$2"; shift 2 ;;
        --anisotropic)
            ANISOTROPIC="--anisotropic"; shift ;;
        --physical-radius)
            PHYSICAL_RADIUS="--physical-radius"; shift ;;
        --speed)
            SPEED="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "用法: bash scripts/batch_launch.sh --volume <mm^3> [选项]"
            echo ""
            echo "必需参数:"
            echo "  --volume, -v <mm^3>     组织体积"
            echo ""
            echo "可选参数:"
            echo "  --pattern, -p <glob>    文件匹配模式（默认 *.tif）"
            echo "  --files-from, -f <file> 从文件读取样本路径列表（替换 find 扫描）"
            echo "  --data-root, -d <dir>   数据根目录"
            echo "  --output-root, -o <dir> 输出根目录"
            echo "  --sampling, -s <float>  稀疏采样率（默认 1.0）"
            echo "  --phases <phase>       执行阶段: all|skeletonize|stats（默认 all）"
            echo "  --resume, -r            跳过 manifest 中已完成全流程的样本（适用于 --phases all）"
            echo "  --min-skeletons <N>     跳过已有 >= N 个 .pajek 的样本（磁盘扫描，适用于 --phases skeletonize）"
            echo "  --max-skeletons <N>     跳过已有 >= N 个 .pajek 的样本（0 = 全部提交）"
            echo "  --count-skeletons       仅统计各样本已有 .pajek 数，不提交"
            echo "  --has-skeleton          仅提交已有 ≥1 个骨架的样本（用于 --phases stats）"
            echo "  --partition <name>     Slurm 分区（默认 compute，可选 tao/control/gpu）"
            echo "  --anisotropic          启用各向异性 spacing 物理单位口径（传给 pipeline）"
            echo "  --physical-radius      启用方案B 各向异性 EDT（半径直接出物理 μm，隐含 --anisotropic）"
            echo "  --speed <float>        收缩速度 speed_param（默认 0.05；大样本可用 0.2 加速）"
            echo "  --dry                   仅预览，不提交"
            echo ""
            echo "--files-from 文件格式:"
            echo "  每行一个路径，相对于 DATA_ROOT 或绝对路径。"
            echo "  空行和 # 开头行忽略。"
            exit 1 ;;
    esac
done

if [ "$PHASES" != "skeletonize" ] && [ -z "$VOLUME" ]; then
    echo "Note: --volume 未指定，将由管线从 sample_metadata.json 自动查找。"
    echo "      如样本元数据未提取，请先运行 extract-metadata。"
    echo ""
fi

# --- 查找输入文件 ---
echo "=========================================="
echo "  Vascular_Statistics Batch Launcher"
echo "=========================================="
echo "  Data Root  : $DATA_ROOT"
echo "  Output Root: $OUTPUT_ROOT"
echo "  Volume     : $VOLUME mm^3"
echo "  Sampling   : $SAMPLING"
echo "  Partition   : $PARTITION"
echo "  Phases       : $PHASES"
echo "  Anisotropic  : $ANISOTROPIC"
echo "  Phys Radius  : $PHYSICAL_RADIUS"
echo "  Speed        : $SPEED"
echo "  Resume       : $RESUME"
echo "  Has Skeleton : $HAS_SKELETON"
echo "  Min Skeletons: $MIN_SKELETONS"
echo "  Max Skeletons: $MAX_SKELETONS"
echo "  Count Only   : $COUNT_ONLY"
echo "  Dry Run      : $DRY"
if [ -n "$FILES_FROM" ]; then
    echo "  Files From   : $FILES_FROM"
else
    echo "  Pattern      : $PATTERN"
fi
echo "=========================================="
echo ""

FILES=()
if [ -n "$FILES_FROM" ]; then
    # 从文件读取路径列表（支持相对路径 + 绝对路径 + #注释 + 空行跳过）
    if [ ! -f "$FILES_FROM" ]; then
        echo "FATAL: --files-from 指定的文件不存在: $FILES_FROM"
        exit 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        # 跳过空行和注释行
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        line=$(echo "$line" | xargs)  # trim whitespace
        [ -z "$line" ] && continue
        # 相对路径 → 拼 DATA_ROOT 前缀
        if [[ "$line" == /* ]]; then
            FILES+=("$line")
        else
            FILES+=("$DATA_ROOT/$line")
        fi
    done < "$FILES_FROM"
else
    # 默认：find 扫描 DATA_ROOT
    while IFS= read -r -d '' f; do
        FILES+=("$f")
    done < <(find "$DATA_ROOT" -maxdepth 5 -name "$PATTERN" -print0 2>/dev/null || true)
fi

if [ ${#FILES[@]} -eq 0 ]; then
    if [ -n "$FILES_FROM" ]; then
        echo "FATAL: --files-from 文件为空或无有效路径"
    else
        echo "未找到匹配文件: $DATA_ROOT/$PATTERN"
    fi
    exit 1
fi

echo "找到 ${#FILES[@]} 个文件"

# --- 恢复模式：过滤已完成样本（仅全流程阶段有意义） ---
SKIPPED_COUNT=0
SKEL_SKIPPED=0
if [ "$RESUME" = true ] && [ "$PHASES" = "skeletonize" ]; then
    echo "Note: --phases skeletonize 下 --resume 无意义（manifest 记录的是全流程完成状态），自动跳过。"
    echo "      使用 --min-skeletons / --count-skeletons 检查骨架完成情况。"
    echo ""
    RESUME=false
fi
if [ "$RESUME" = true ]; then
    MANIFEST_PATH="$OUTPUT_ROOT/manifest.json"
    if [ -f "$MANIFEST_PATH" ]; then
        echo "Resume 模式：检查 manifest.json 过滤已完成样本..."
        # 通过 Python filter_pending() 过滤
        readarray -t FILTERED_FILES < <(
            python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/python')
from vascular_statistics.batch import filter_pending
files = [line.strip() for line in sys.stdin if line.strip()]
result = filter_pending(files, '$MANIFEST_PATH', '$DATA_ROOT')
for f in result:
    print(f)
" <<< "$(printf '%s\n' "${FILES[@]}")"
        )
        SKIPPED_COUNT=$(( ${#FILES[@]} - ${#FILTERED_FILES[@]} ))
        if [ $SKIPPED_COUNT -gt 0 ]; then
            echo "  跳过 $SKIPPED_COUNT 个已完成样本"
        fi
        FILES=("${FILTERED_FILES[@]}")
    else
        echo "Resume 模式：manifest.json 不存在，全部提交"
    fi
fi

# --- 骨架数过滤：基于磁盘实际 .pajek 文件计数 ---
if [ "$MIN_SKELETONS" -gt 0 ] || [ "$MAX_SKELETONS" -ge 0 ] || [ "$HAS_SKELETON" = true ] || [ "$COUNT_ONLY" = true ]; then
    echo "扫描已有骨架..."
    # 通过 Python filter_by_skeleton_count 过滤
    FILTER_OUTPUT=$(python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/python')
from vascular_statistics.batch import filter_by_skeleton_count

files = [line.strip() for line in sys.stdin if line.strip()]
to_submit, skipped, counts = filter_by_skeleton_count(
    files,
    '$DATA_ROOT',
    '$OUTPUT_ROOT',
    min_skeletons=$MIN_SKELETONS,
    max_skeletons=$MAX_SKELETONS,
    has_skeleton_only=('$HAS_SKELETON' == 'true'),
)

# 输出计数摘要（stderr 以避免混入文件列表）
import json
summary = {
    'total': len(files),
    'to_submit': len(to_submit),
    'skipped': len(skipped),
    'counts': counts,
}
print(json.dumps(summary), file=sys.stderr)

# 输出待提交文件列表（stdout）
for f in to_submit:
    print(f)
" <<< "$(printf '%s\n' "${FILES[@]}")" 2>&1 1>/tmp/vascstats_skeleton_filter.txt)
    # Python stderr → summary JSON; stdout → file list
    FILTER_SUMMARY=$(echo "$FILTER_OUTPUT" | tail -1)
    readarray -t FILTERED_FILES < /tmp/vascstats_skeleton_filter.txt
    rm -f /tmp/vascstats_skeleton_filter.txt

    SKEL_SKIPPED=$(echo "$FILTER_SUMMARY" | python -c "import json,sys; d=json.load(sys.stdin); print(d['skipped'])")
    echo "  总样本: $(echo "$FILTER_SUMMARY" | python -c "import json,sys; d=json.load(sys.stdin); print(d['total'])")"
    echo "  需提交: $(echo "$FILTER_SUMMARY" | python -c "import json,sys; d=json.load(sys.stdin); print(d['to_submit'])")"
    echo "  已跳过: $SKEL_SKIPPED (骨架数已达阈值)"
    echo ""

    # 显示各样本骨架计数（分组汇总）
    python -c "
import json, sys
summary = json.loads('''$FILTER_SUMMARY''')
counts = summary['counts']
# 按组输出
from collections import defaultdict
by_group = defaultdict(list)
for k, n in counts.items():
    group = k.split('/')[0]
    by_group[group].append((k, n))
for group in sorted(by_group):
    samples = by_group[group]
    n0 = sum(1 for _, n in samples if n == 0)
    n1 = sum(1 for _, n in samples if n == 1)
    n2p = sum(1 for _, n in samples if n >= 2)
    print(f'  {group}: 0骨架={n0}, 1骨架={n1}, ≥2骨架={n2p}')
"

    FILES=("${FILTERED_FILES[@]}")

    if [ "$COUNT_ONLY" = true ]; then
        echo ""
        echo "[COUNT ONLY] 仅统计，不提交。移除 --count-skeletons 以实际提交。"
        exit 0
    fi
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "无待处理文件。所有样本均已完成或已达骨架数阈值。"
    exit 0
fi

# 列出待处理文件
echo ""
TOTAL_SKIPPED=$((SKIPPED_COUNT + SKEL_SKIPPED))
echo "待处理 ${#FILES[@]} 个文件 (跳过 $TOTAL_SKIPPED: resume=$SKIPPED_COUNT, skeleton=$SKEL_SKIPPED):"
for f in "${FILES[@]}"; do
    echo "  $f"
done
echo ""

if [ "$DRY" = true ]; then
    echo "[DRY RUN] 以上 ${#FILES[@]} 个文件将被提交。移除 --dry 以实际提交。"
    exit 0
fi

# --- 提交前检查：队列中是否已有 vascstats 任务 ---
EXISTING=$(squeue -u "$USER" -n vs_pipeline -h 2>/dev/null | wc -l || echo "0")
if [ "$EXISTING" -gt 0 ]; then
    echo "WARNING: 队列中已有 $EXISTING 个 vs_pipeline 任务。"
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

# 自存档（保留本次使用的脚本副本）
cp "$0" "logs/submit/batch_launch_${TIMESTAMP}.sh" 2>/dev/null || true

# --- 确保输出根目录存在 ---
mkdir -p "$OUTPUT_ROOT"

# --- 批量提交 ---
echo "开始提交..."
echo ""

JOB_IDS=()
for f in "${FILES[@]}"; do
    # 相对于 DATA_ROOT 的路径
    REL_PATH="${f#$DATA_ROOT/}"
    BASENAME="$(basename "$f" | sed 's/\.[^.]*$//')"

    # 计算 sample_key（用于 manifest pending 预注册 + 作业名）
    SAMPLE_KEY=$(python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/python')
from vascular_statistics.batch import compute_sample_key
print(compute_sample_key('$REL_PATH'))
")

    # 提交前在 manifest 中预注册 pending 条目（如条目尚不存在）
    python -m vascular_statistics.cli update-manifest \
        --manifest "$OUTPUT_ROOT/manifest.json" \
        --sample-key "$SAMPLE_KEY" \
        --input-path "$REL_PATH" \
        --volume-mm3 "$VOLUME" \
        --status pending \
        2>/dev/null || true

    # Slurm 作业名：用 sample_key 的末段（文件名部分），截断到 16 字符限制
    SHORT_NAME="$(echo "$SAMPLE_KEY" | rev | cut -d'/' -f1 | rev | cut -c1-16)"

    # 提交作业
    # 参数: INPUT_FILE, VOLUME, OUTPUT_STEM, SAMPLING, PHASES, ANISOTROPIC, SPEED, PHYSICAL_RADIUS
    JOB_ID=$(sbatch \
        --partition="$PARTITION" \
        --job-name="vs_${SHORT_NAME}" \
        --output="logs/pipeline_${SHORT_NAME}_%j.out" \
        --error="logs/pipeline_${SHORT_NAME}_%j.err" \
        "$SLURM_SCRIPT" "$REL_PATH" "$VOLUME" "skeleton" "$SAMPLING" "$PHASES" \
        "$ANISOTROPIC" "$SPEED" "$PHYSICAL_RADIUS" \
        2>&1 | grep -oP '\d+')

    if [ -n "$JOB_ID" ]; then
        echo "  [$JOB_ID] $SAMPLE_KEY"
        JOB_IDS+=("$JOB_ID")
    else
        echo "  [FAILED] $SAMPLE_KEY"
    fi
done

echo ""
echo "=========================================="
echo "  提交完成: ${#JOB_IDS[@]}/${#FILES[@]} 个作业"
echo "  跳过    : $TOTAL_SKIPPED 个 (resume=$SKIPPED_COUNT, skeleton=$SKEL_SKIPPED)"
echo "  日志    : $SUBMIT_LOG"
echo "  进度    : python -m vascular_statistics.cli status --manifest $OUTPUT_ROOT/manifest.json"
echo "  监控    : squeue -u $USER -n 'vs_*'"
echo "=========================================="

# 保存提交记录
{
    echo "Submit: $TIMESTAMP"
    echo "Volume: $VOLUME"
    echo "Sampling: $SAMPLING"
    echo "Phases: $PHASES"
    echo "Anisotropic: $ANISOTROPIC"
    echo "PhysicalRadius: $PHYSICAL_RADIUS"
    echo "Speed: $SPEED"
    echo "Resume: $RESUME"
    echo "MinSkeletons: $MIN_SKELETONS"
    echo "MaxSkeletons: $MAX_SKELETONS"
    echo "Skipped: resume=$SKIPPED_COUNT, skeleton=$SKEL_SKIPPED"
    echo "Files (${#FILES[@]}):"
    for i in "${!FILES[@]}"; do
        echo "  ${JOB_IDS[$i]:-FAIL}  ${FILES[$i]}"
    done
} > "$SUBMIT_LOG"
