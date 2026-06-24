#!/bin/bash
#SBATCH --job-name=csam_inf2
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/csam_inference2_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/csam_inference2_%j.err

set -euo pipefail

source /share/home/sukm/miniconda3/etc/profile.d/conda.sh
conda activate vesseg

PROJECT_ROOT="/share/home/sukm/Vascular_Extraction"
BEST_EXP="/share/home/sukm/experiments/ve_phase_b_grid/fcdensenet_csam_combined_cl_dice_3d_slice3_bs1_L3-4-5_ch48_g20/seed2"
IMG_DIR="/share/home/sukm/datasets/2PFM_SkelGT/images"
OUT_DIR="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg"

mkdir -p "$OUT_DIR"

# Only the 4 files that failed (LZW or skipped)
REMAINING=(
    "W2N_20190911_ws1"
    "W2N_20190911_ws2"
    "W2R_20190903_ref"
    "W2R_20190903_ws1"
)

echo "============================================"
echo "CSAM 推理 — 剩余 4 例 (LZW 压缩已修复)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================"

INPUTS=()
OUTS=()
for stem in "${REMAINING[@]}"; do
    INPUTS+=("$IMG_DIR/${stem}.tiff")
    OUTS+=("$OUT_DIR/${stem}_csam_seg.tiff")
    echo "  ${stem}.tiff -> ${stem}_csam_seg.tiff"
done

cd "$PROJECT_ROOT"
python src/scripts/predict.py \
    --exp_dir "$BEST_EXP" \
    --input "${INPUTS[@]}" \
    --out "${OUTS[@]}" \
    --threshold 0.5

echo ""
echo "============================================"
echo "全部推理完成。输出文件:"
ls -lh "$OUT_DIR"/
echo "============================================"
