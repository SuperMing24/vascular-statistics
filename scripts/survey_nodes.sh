#!/bin/bash
# ============================================================================
#  集群全节点 CPU 检测脚本 —— Vascular_Statistics
# ============================================================================
#
#  用法:
#    bash scripts/survey_nodes.sh              # 检测 compute + tao 分区所有节点
#    bash scripts/survey_nodes.sh --all         # 检测所有分区（含 control）
#    bash scripts/survey_nodes.sh --quick       # 快速模式（仅 lscpu，不跑 numactl）
#
#  输出:
#    logs/hardware/<node>_lscpu.txt
#    logs/hardware/<node>_numactl.txt
#    logs/hardware/summary.txt                 # 汇总对比表
#
#  检测策略:
#    逐节点 srun --nodelist=<X>，每个节点最多等 30 秒（timeout 防挂死）。
#    若节点完全分配无法登录，标记 BUSY 并跳过。
#    仅需 1 核 + 1 GB → 对运行中作业无影响。
# ============================================================================

# 不设 set -e —— 逐个节点处理，单节点失败不终止全脚本
set -uo pipefail

ALL_PARTITIONS=false
QUICK=false
for arg in "${@}"; do
    case "$arg" in
        --all)   ALL_PARTITIONS=true ;;
        --quick) QUICK=true ;;
    esac
done

# 输出目录：始终放在项目根目录的 logs/hardware/ 下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTDIR="$PROJECT_DIR/logs/hardware"
mkdir -p "$OUTDIR"

echo "输出目录: $OUTDIR"
echo ""

# 获取节点列表：默认仅 compute + tao 分区
echo ">>> 获取节点列表..."
if [ "$ALL_PARTITIONS" = true ]; then
    PARTITIONS="compute,tao,control"
else
    PARTITIONS="compute,tao"
fi

# sinfo 输出格式: "nodename partition state"
# 需要分区信息 —— srun 仅指定 --nodelist 在某些集群上会失败（尤其 tao 分区），
# 必须同时指定 --partition 才能正确路由。
NODE_INFO=$(sinfo -h -p "$PARTITIONS" -o "%n %P %T" 2>/dev/null \
    | grep -v "drain\|down\|drng\|unk\|reserved\|maint" \
    | sort -u || echo "")

if [ -z "$NODE_INFO" ]; then
    echo "FATAL: sinfo 未返回任何可用节点（分区: $PARTITIONS）"
    echo "提示: 尝试 --all 检测所有分区，或手动检查 sinfo 输出"
    exit 1
fi

NODE_COUNT=$(echo "$NODE_INFO" | wc -l)
echo "可检测节点 ($NODE_COUNT):"
echo "$NODE_INFO" | while read -r n p s; do echo "  $n  [$p]  $s"; done
echo ""

SUMMARY="$OUTDIR/summary.txt"
{
    echo "=========================================="
    echo "  Cluster CPU Survey — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Partitions: $PARTITIONS"
    echo "=========================================="
    echo ""
} > "$SUMMARY"

SUCCESS=0
BUSY=0

while read -r NODE NODE_PART NODE_STATE; do
    [ -z "$NODE" ] && continue
    echo "--- [$NODE] ($NODE_PART) ---"
    echo "  状态: $NODE_STATE"

    # 用 timeout 防挂死：srun 最多等 30 秒，超时则跳过。
    # srun 在节点满配时会排队；--time=00:01:00 只限制运行时长，不限制排队时长。
    # 必须同时指定 --partition 和 --nodelist：仅 --nodelist 在某些集群上
    # （尤其 tao 分区）会报 "Requested node configuration is not available"。
    # 注意：不能加 --output/--error 重定向 —— 那会把远程命令的 stdout 也吞掉。
    SRUN_STDERR=$(mktemp)
    OUTPUT=$(timeout 30 srun \
        --partition="$NODE_PART" \
        --nodelist="$NODE" \
        --cpus-per-task=1 \
        --mem=1G \
        --time=00:01:00 \
        --job-name="survey_${NODE}" \
        bash -c '
            echo "NODE=$(hostname)"
            lscpu
            if [ "'$QUICK'" != "true" ]; then
                echo "=== NUMACTL ==="
                numactl --hardware 2>/dev/null || echo "numactl not available"
            fi
        ' 2>"$SRUN_STDERR") || {
        RC=$?
        if [ $RC -eq 124 ]; then
            echo "  [$NODE] TIMEOUT — 排队超过 30 秒，跳过"
            echo "[$NODE] TIMEOUT (30s queue)" >> "$SUMMARY"
        else
            echo "  [$NODE] FAILED — srun 退出码 $RC，跳过"
            echo "[$NODE] FAILED (exit=$RC)" >> "$SUMMARY"
            # 输出 srun 的 stderr 以辅助诊断
            if [ -s "$SRUN_STDERR" ]; then
                echo "  [stderr] $(head -3 "$SRUN_STDERR" | tr '\n' ' ')"
            fi
        fi
        rm -f "$SRUN_STDERR"
        BUSY=$((BUSY + 1))
        echo ""
        continue
    }
    rm -f "$SRUN_STDERR"

    # 解析输出
    NODE_NAME=$(echo "$OUTPUT" | grep "^NODE=" | cut -d= -f2)
    if [ -z "$NODE_NAME" ]; then
        echo "  [$NODE] WARNING: srun 成功但未获取到 hostname，跳过"
        echo "[$NODE] UNKNOWN (no hostname in output)" >> "$SUMMARY"
        BUSY=$((BUSY + 1))
        echo ""
        continue
    fi

    # 写入 lscpu 文件（去掉 NODE= 行）
    echo "$OUTPUT" | grep -v "^NODE=" > "$OUTDIR/${NODE_NAME}_lscpu.txt"

    if [ "$QUICK" != "true" ]; then
        # 分离 numactl 部分到独立文件
        awk '/=== NUMACTL ===/{flag=1; next} flag' "$OUTDIR/${NODE_NAME}_lscpu.txt" \
            > "$OUTDIR/${NODE_NAME}_numactl.txt" 2>/dev/null || true
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
    SUCCESS=$((SUCCESS + 1))
done <<< "$NODE_INFO"

echo ""
echo "=========================================="
echo "  检测完成: $SUCCESS 成功, $BUSY 跳过"
echo "  结果目录: $OUTDIR/"
echo "=========================================="
echo ""
cat "$SUMMARY"
