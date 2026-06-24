#!/bin/bash
#SBATCH --job-name=csam_infer
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/sukm/experiments/vs_2pfm_skelgt/csam_inference_%j.out
#SBATCH --error=/share/home/sukm/experiments/vs_2pfm_skelgt/csam_inference_%j.err

set -euo pipefail

PROJECT_ROOT="/share/home/sukm/Vascular_Extraction"
BEST_EXP="/share/home/sukm/experiments/ve_phase_b_grid/fcdensenet_csam_combined_cl_dice_3d_slice3_bs1_L3-4-5_ch48_g20/seed2"
IMG_DIR="/share/home/sukm/datasets/2PFM_SkelGT/images"
OUT_DIR="/share/home/sukm/experiments/vs_2pfm_skelgt/csam_seg"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

source /share/home/sukm/miniconda3/etc/profile.d/conda.sh
conda activate vesseg

mkdir -p "$OUT_DIR"

echo "============================================"
echo "CSAM 推理 — 7 例 2PFM_SkelGT"
echo "模型: seed2/fcdensenet_csam_combined_cl_dice_3d_slice3_bs1_L3-4-5_ch48_g20"
echo "输出: $OUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "时间: $TIMESTAMP"
echo "============================================"

INPUTS=()
OUTS=()
for tiff in "$IMG_DIR"/*.tiff; do
    stem=$(basename "$tiff" .tiff)
    INPUTS+=("$tiff")
    OUTS+=("$OUT_DIR/${stem}_csam_seg.tiff")
done

echo "输入文件 (${#INPUTS[@]} 个):"
for i in "${!INPUTS[@]}"; do
    echo "  $(basename "${INPUTS[$i]}") -> $(basename "${OUTS[$i]}")"
done

cd "$PROJECT_ROOT"
python src/scripts/predict.py \
    --exp_dir "$BEST_EXP" \
    --input "${INPUTS[@]}" \
    --out "${OUTS[@]}" \
    --threshold 0.5

echo ""
echo "============================================"
echo "推理完成。输出文件:"
ls -lh "$OUT_DIR"/
echo "============================================"
