#!/bin/bash
# ============================================================================
#  CSAM 分割推理 —— W2R_ws1 原图 resize384→推理→裁回382（方案B）
#  输入：W2R_20190903_ws1_original_resize384.tiff（灰度 512→384, CSAM兼容）
#  输出：W2R_20190903_ws1_csam_seg_382fromOriginal.tiff（384→382 中心裁切）
#  注意：382 不被 32 整除导致 FC-DenseNet skip connection 维度错配，
#        故先在 384 上推理（384/32=12 兼容），再裁回 382。
# ============================================================================

#SBATCH --job-name=csam_inf_382
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/csam_infer_382_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/logs/csam_infer_382_%j.err

set -euo pipefail

set +u
source ~/.bashrc
set -u
conda activate vesseg

# CSAM 模型检查点 —— 跨模型评估 Rank 1: CSAM L3-4-5 g=20 [Grid] seed2 (HM=0.8527)
EXP_DIR="/share/home/sukm/experiments/ve_phase_b_grid/fcdensenet_csam_combined_cl_dice_3d_slice3_bs1_L3-4-5_ch48_g20/seed2"
INPUT_TIFF="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_original_resize384.tiff"
OUTPUT_384="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_csam_seg_384temp.tiff"
OUTPUT_TIFF="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg/W2R_20190903_ws1_csam_seg_382fromOriginal.tiff"

echo "=============================================="
echo "  CSAM 推理 — 方案B（384推理→382裁切）"
echo "=============================================="
echo "  Model  : $EXP_DIR"
echo "  Input  : $INPUT_TIFF (384×384×300)"
echo "  Temp   : $OUTPUT_384"
echo "  Output : $OUTPUT_TIFF (382×382×300)"
echo "  Node   : $SLURM_NODELIST"
echo "=============================================="
echo ""

cd /share/home/sukm/Vascular_Extraction
echo "VascExt commit: $(git rev-parse --short HEAD)"

# Step 1: CSAM 推理（384 网格，兼容 FC-DenseNet 32× 整除要求）
python src/scripts/predict.py \
    --exp_dir "$EXP_DIR" \
    --input "$INPUT_TIFF" \
    --out "$OUTPUT_384" \
    --threshold 0.5

echo "[$(date '+%H:%M:%S')] CSAM 推理完成，开始 384→382 中心裁切..."

# Step 2: 中心裁切 384→382（左右各裁 1px），写入最终产物
python3 -c "
import tifffile, numpy as np
seg = tifffile.imread('$OUTPUT_384')          # [300,384,384]
seg_crop = seg[:, 1:383, 1:383]               # [300,382,382] 中心裁切
tifffile.imwrite('$OUTPUT_TIFF', seg_crop)
fg = (seg_crop > 0).sum()
print(f'  裁切完成: {seg.shape} -> {seg_crop.shape}')
print(f'  前景体素: {fg:,} / {seg_crop.size:,} = {fg/seg_crop.size*100:.1f}%')
print(f'  唯一值: {sorted(np.unique(seg_crop).tolist())}')
"

# 清理临时 384 文件
rm -f "$OUTPUT_384"

echo ""
echo "=============================================="
echo "  DONE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: $OUTPUT_TIFF"
echo "=============================================="
