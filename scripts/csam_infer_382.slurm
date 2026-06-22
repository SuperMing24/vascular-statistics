#!/bin/bash
# ============================================================================
#  CSAM 分割推理 —— W2R_ws1 原图 resize382 版本（方案B）
#  输入：W2R_20190903_ws1_original_resize382.tiff（原始灰度图 512→382, order=1）
#  输出：W2R_20190903_ws1_csam_seg_382fromOriginal.tiff（382 网格上 CSAM 分割）
# ============================================================================

#SBATCH --job-name=csam_inf_382
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/csam_infer_382_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/csam_infer_382_%j.err

set -euo pipefail

set +u
source ~/.bashrc
set -u
conda activate vesseg

# CSAM 模型检查点 —— 跨模型评估 Rank 1: CSAM L3-4-5 g=20 [Grid] seed2 (HM=0.8527)
EXP_DIR="/share/home/sukm/experiments/ve_phase_b_grid/fcdensenet_csam_combined_cl_dice_3d_slice3_bs1_L3-4-5_ch48_g20/seed2"
INPUT_TIFF="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_original_resize382.tiff"
OUTPUT_TIFF="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_csam_seg_382fromOriginal.tiff"

echo "=============================================="
echo "  CSAM 推理 — 方案B（先缩放原图后分割）"
echo "=============================================="
echo "  Model  : $EXP_DIR"
echo "  Input  : $INPUT_TIFF"
echo "  Output : $OUTPUT_TIFF"
echo "  Node   : $SLURM_NODELIST"
echo "=============================================="
echo ""

cd /share/home/sukm/Vascular_Extraction
echo "VascExt commit: $(git rev-parse --short HEAD)"

python src/scripts/predict.py \
    --exp_dir "$EXP_DIR" \
    --input "$INPUT_TIFF" \
    --out "$OUTPUT_TIFF" \
    --threshold 0.5

echo ""
echo "=============================================="
echo "  DONE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUTPUT_TIFF"
echo "=============================================="
