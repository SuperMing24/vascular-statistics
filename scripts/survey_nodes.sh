#!/bin/bash
# ============================================================================
#  集群全节点 CPU 检测脚本 —— Vascular_Statistics
# ============================================================================
#
#  用法:
#    bash scripts/survey_nodes.sh              # 检测所有可访问节点
#    bash scripts/survey_nodes.sh --quick       # 快速模式（仅 lscpu，不跑 numactl）
#
#  输出:
#    logs/hardware/<node>_lscpu.txt
#    logs/hardware/<node>_numactl.txt
#    logs/hardware/summary.txt                 # 汇总对比表
#
#  说明:
#    对每个节点用 srun --nodelist 登录 → 执行 lscpu + numactl → 保存。
#    若节点繁忙（1 分钟内无法分配），标记为 BUSY 并跳过。
#    需要至少 1 个 CPU + 1 GB 内存的瞬时分配。
# ============================================================================

set -euo pipefail

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

OUTDIR="logs/hardware"
mkdir -p "$OUTDIR"

# 从 sinfo 获取所有 idle/mix 状态的节点列表
echo ">>> 获取节点列表..."
NODES=$(sinfo -h -o "%n %T" 2>/dev/null | grep -v "drain\|down\|drng\|unk" | awk '{print $1}' || echo "")

if [ -z "$NODES" ]; then
    echo "FATAL: sinfo 未返回任何可用节点"
    exit 1
fi

echo "可检测节点: $(echo $NODES | tr '\n' ' ')"
echo ""

SUMMARY="$OUTDIR/summary.txt"
{
    echo "=========================================="
    echo "  Cluster CPU Survey — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
} > "$SUMMARY"

for NODE in $NODES; do
    echo "--- [$NODE] ---"

    # 用 srun 登录指定节点，1 分钟超时
    OUTPUT=$(srun \
        --nodelist="$NODE" \
        --cpus-per-task=1 \
        --mem=1G \
        --time=00:01:00 \
        --job-name="survey_${NODE}" \
        --output=/dev/null \
        --error=/dev/null \
        bash -c '
            echo "NODE=$(hostname)"
            lscpu
            if [ "'$QUICK'" != "true" ]; then
                echo "=== NUMACTL ==="
                numactl --hardware 2>/dev/null || echo "numactl not available"
            fi
        ' 2>&1) || {
        echo "  [$NODE] BUSY — 1 分钟内无法分配，跳过"
        echo "[$NODE] BUSY (1min timeout)" >> "$SUMMARY"
        continue
    }

    # 解析输出
    NODE_NAME=$(echo "$OUTPUT" | grep "^NODE=" | cut -d= -f2)
    echo "$OUTPUT" | sed '1d' > "$OUTDIR/${NODE_NAME}_lscpu.txt"

    if [ "$QUICK" != "true" ]; then
        # 分离 numactl 部分
        awk '/=== NUMACTL ===/{flag=1; next} flag' "$OUTDIR/${NODE_NAME}_lscpu.txt" > "$OUTDIR/${NODE_NAME}_numactl.txt" 2>/dev/null || true
        # 从 lscpu 文件中移除 numactl 部分
        sed -i '/=== NUMACTL ===/,$d' "$OUTDIR/${NODE_NAME}_lscpu.txt"
    fi

    # 提取关键字段写入汇总
    MODEL=$(echo "$OUTPUT" | grep "Model name:" | sed 's/.*: *//')
    CPU_COUNT=$(echo "$OUTPUT" | grep "^CPU(s):" | sed 's/.*: *//')
    SOCKETS=$(echo "$OUTPUT" | grep "Socket(s):" | sed 's/.*: *//')
    CORES_PER_SOCKET=$(echo "$OUTPUT" | grep "Core(s) per socket:" | sed 's/.*: *//')
    THREADS_PER_CORE=$(echo "$OUTPUT" | grep "Thread(s) per core:" | sed 's/.*: *//')
    L3=$(echo "$OUTPUT" | grep "L3 cache:" | sed 's/.*: *//')
    NUMA_NODES=$(echo "$OUTPUT" | grep "NUMA node(s):" | sed 's/.*: *//')

    echo "[$NODE_NAME] $MODEL | ${CPU_COUNT}核 | ${SOCKETS}路×${CORES_PER_SOCKET}核 | HT:${THREADS_PER_CORE} | L3:${L3} | NUMA:${NUMA_NODES}" >> "$SUMMARY"

    echo "  [$NODE_NAME] $MODEL"
    echo "  → $OUTDIR/${NODE_NAME}_lscpu.txt"
    echo ""
done

echo ""
echo "=========================================="
echo "  检测完成。结果目录: $OUTDIR/"
echo "  汇总: $SUMMARY"
echo "=========================================="
cat "$SUMMARY"
