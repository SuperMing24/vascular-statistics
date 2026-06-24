#!/bin/bash
#SBATCH --job-name=seg384
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/sukm/experiments/lab_data_seg_384/job_%j.out
#SBATCH --error=/share/home/sukm/experiments/lab_data_seg_384/job_%j.err

set -euo pipefail
source ~/.bashrc

# 384 批量分割：huaien 41 + xiaoqian 104 cropped_z → TIFF 掩码
VESSEG=/share/home/sukm/conda_envs/vesseg/bin/python
SCRIPT=/share/home/sukm/vascular-statistics/scripts/seg_batch_resample.py
LAB=/share/home/sukm/datasets/Lab_Data
OUT=/share/home/sukm/experiments/lab_data_seg_384

"$VESSEG" "$SCRIPT" --crop-root "$LAB/huaien/cropped_z" --resize 384 --out-root "$OUT/huaien"
"$VESSEG" "$SCRIPT" --crop-root "$LAB/xiaoqian/cropped_z" --resize 384 --out-root "$OUT/xiaoqian"

echo '=== SEG384 ALL DONE ==='
